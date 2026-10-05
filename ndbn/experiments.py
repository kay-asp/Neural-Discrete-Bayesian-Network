"""Experiment grid (MLE vs neural CPDs) and the hyperparameter sweep."""
import logging
import os
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .evaluation import kl_mle, kl_nn, mle_param_stats, timed_fit
from .generation import gen_CPTs, gen_dag_indegree, make_seed
from .mle import fit_mle, fit_mle_pseudocount
from . import models as _models
from .models import NeuralCPDs
from .tracing import Trace, history_matrix
from .training import fit_nn, split_train_val


def config_exists(df, estimator, dag, n_nodes, in_deg, alpha, sample_size):
    if df.empty or "estimator" not in df.columns:
        return False
    return (
        (df["estimator"]   == estimator) &
        (df["dag_idx"] == dag) &
        (df["n_nodes"] == n_nodes) &
        (df["in_degree"] == in_deg) &
        (df["alpha"] == alpha) &
        (df["sample_size"] == sample_size)
    ).any()

def flush(records, path):
    if not records:
        return
    df = pd.DataFrame(records)
    if os.path.exists(path):
        old = pd.read_csv(path)
        cols = list(dict.fromkeys(list(old.columns) + list(df.columns)))
        df = pd.concat([old.reindex(columns=cols), df.reindex(columns=cols)],
                       ignore_index=True)
    df.to_csv(path, index=False)
    records.clear()


def build_dag(dag, in_deg, n_nodes, alpha, card):
    """Ground-truth DAG + CPTs for one config -> (G, true_model).

    Shared by the main experiment and the sweep, so both see the same networks.
    """
    # alpha is left out of the seeds so every alpha shares the same DAG (paired)
    G = gen_dag_indegree(n_nodes, in_deg, seed=make_seed("dag", dag, in_deg, n_nodes))
    true_model = gen_CPTs(G, card, alpha, seed=make_seed("cpt", dag, in_deg, n_nodes))
    return G, true_model


def dag_data_dir(results_dir, alpha, n_nodes, in_deg, dag):
    """Where one DAG's cached train.csv / test.csv live."""
    path = os.path.join(results_dir, "data", f"alpha_{alpha}", f"BN_{n_nodes}",
                        f"indegree_{in_deg}", f"dag_{dag}")
    os.makedirs(path, exist_ok=True)
    return path


def load_or_simulate(path, true_model, n_samples, seed):
    """Read cached samples from `path`, or simulate and cache them."""
    if os.path.exists(path):
        data = pd.read_csv(path)
        if len(data) != n_samples:
            raise ValueError(f"{path} has {len(data)} rows, expected {n_samples}; "
                             "it was cached with a different sample budget")
        return data
    data = true_model.simulate(n_samples=n_samples, seed=seed, show_progress=False)
    data.to_csv(path, index=False)
    return data


def load_dag(results_dir, dag, in_deg, n_nodes, alpha, card, n_train, n_test=None):
    """Ground truth plus cached train (and test) samples for one DAG.

    -> (G, true_model, train_data, test_data or None). Simulates and caches on
    first use, so calling it once per DAG in the parent before dispatching
    parallel jobs means workers only read the CSVs (no two writers per file).
    """
    G, true_model = build_dag(dag, in_deg, n_nodes, alpha, card)
    dag_dir = dag_data_dir(results_dir, alpha, n_nodes, in_deg, dag)
    data = load_or_simulate(os.path.join(dag_dir, "train.csv"), true_model,
                            n_train, make_seed("train", dag, in_deg, n_nodes))
    test_data = None if n_test is None else load_or_simulate(
        os.path.join(dag_dir, "test.csv"), true_model, n_test,
        make_seed("test", dag, in_deg, n_nodes))
    return G, true_model, data, test_data


# ---- parallel execution -----------------------------------------------------
# Every fit is seeded from its own config (make_seed), so running jobs in
# separate processes gives the same results as the serial loop on the same
# device. Speed settings, all opt-in (defaults = serial, unchanged behaviour):
#   n_jobs        : worker processes; 1 = serial in this process, -1 = all cores
#   device        : None = auto (cuda if available), or e.g. "cpu"
#   torch_threads : torch threads per worker; None = 1 when n_jobs != 1, else
#                   torch's default. Tiny per-node nets gain nothing from more.

def _setup_worker(device, torch_threads):
    _models.set_device(device)
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)
    logging.getLogger("pgmpy").setLevel(logging.ERROR)


def _call_in_worker(fn, job, device, torch_threads):
    _setup_worker(device, torch_threads)
    return fn(job)


def _parallel_map(fn, jobs, n_jobs=1, device=None, torch_threads=None):
    """Yield fn(job) for every job, in completion order when parallel.

    fn must be a module-level function (or a partial of one) so worker
    processes can import it, which is also what makes this work on Windows.
    """
    if n_jobs == 1:
        # serial: apply the settings here, then restore them afterwards
        prev_device, prev_threads = _models.DEVICE, torch.get_num_threads()
        _setup_worker(device or prev_device, torch_threads)
        try:
            for job in jobs:
                yield fn(job)
        finally:
            _models.set_device(prev_device)
            torch.set_num_threads(prev_threads)
        return

    from joblib import Parallel, delayed
    threads = 1 if torch_threads is None else torch_threads
    yield from Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
        delayed(_call_in_worker)(fn, job, device, threads) for job in jobs)


def run_settings(n_jobs):
    """Columns recording how a row was run, so timings are read in context."""
    return {"n_jobs": n_jobs, "device": _models.DEVICE,
            "torch_threads": torch.get_num_threads(),
            "torch_version": torch.__version__}


# train_nn_cpds keyword arguments an estimator / arm / sweep config may set,
# with train_nn_cpds' defaults (used when recording a run's config)
NN_HPARAMS = {"n_epochs": 500, "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.0,
              "batch_size": 64, "patience": 20, "optimizer": "adam"}


def activation_name(est):
    return "linear" if est["activation"] is None else est["activation"].__name__


def fit_estimator(est, G, sample, train_df, val_df, card, n_epochs, seed):
    """Fit one entry of ESTIMATORS -> (model, wall_s, cpu_s, rss_delta_bytes).

    Tabular estimators fit on the whole budget `sample` (= train_df + val_df):
    they have no early stopping, so they get every row the NN sees.
    """
    if est["type"] == "mle":
        return timed_fit(fit_mle, G, sample, card)
    if est["type"] == "dirichlet":
        return timed_fit(fit_mle_pseudocount, G, sample, card)
    # optional tuned hyperparameters (lr, weight_decay, dropout, ...) pass through
    kw = {x: est[x] for x in NN_HPARAMS if x in est}
    kw.setdefault("n_epochs", n_epochs)
    return timed_fit(
        fit_nn, G, train_df, card,
        val_data=val_df,
        hidden_dims=est["hidden_dims"],
        activation=est["activation"],
        seed=seed,
        **kw,
    )


def evaluate(model, true_model, test_data, card, n_epochs):
    """KL vs ground truth plus size / early-stopping stats -> (kl, se, extra)."""
    if isinstance(model, NeuralCPDs):
        kl, se = kl_nn(true_model, model, test_data, card)
        extra = model.param_stats()
        extra["stop_epoch_mean"] = float(np.mean(list(model.stop_epoch_.values())))
        extra["frac_at_ceiling"] = float(np.mean(
            [e >= n_epochs for e in model.stop_epoch_.values()]))
    else:
        kl, se = kl_mle(true_model, model, test_data)
        extra = mle_param_stats(model)
    return kl, se, extra


def _run_dag_job(job, estimators, sample_sizes, card, val_frac, n_test, n_epochs,
                 results_dir, n_jobs):
    """Every (sample size, estimator) still to do for one DAG -> list of records.

    The body of the main experiment, as one job: a module-level function so
    run_experiments can hand it to worker processes.
    """
    n_nodes, in_deg, alpha, dag = job["n_nodes"], job["in_degree"], job["alpha"], job["dag"]
    G, true_model, data, test_data = load_dag(results_dir, dag, in_deg, n_nodes, alpha,
                                              card, max(sample_sizes), n_test)

    # actual density of this realised graph
    max_possible = n_nodes * (n_nodes - 1) / 2
    density = G.number_of_edges() / max_possible if max_possible > 0 else 0

    records = []
    for samples in sample_sizes:
        split_seed = make_seed("split", dag, in_deg, n_nodes, samples)
        nn_seed = make_seed("nn", dag, in_deg, n_nodes, samples)

        sample = data.iloc[:samples]
        train_df, val_df = split_train_val(sample, val_frac=val_frac, seed=split_seed)

        for est in estimators:
            if (est["name"], samples) not in job["todo"]:
                continue

            model, fit_time, cpu_time, rss = fit_estimator(
                est, G, sample, train_df, val_df, card, n_epochs, seed=nn_seed)
            kl, se, extra = evaluate(model, true_model, test_data, card, n_epochs)

            is_nn = est["type"] == "nn"
            n_fit = len(train_df) if is_nn else len(sample)
            records.append({
                "estimator":   est["name"],
                "hidden_dims": str(list(est.get("hidden_dims", []))) if is_nn else "",
                "activation":  activation_name(est) if is_nn else "",
                "dag_idx":     dag,
                "n_nodes":     n_nodes,
                "in_degree":   in_deg,
                "alpha":       alpha,
                "sample_size": samples,          # budget
                "n_fit":       n_fit,            # rows fitted (NN excludes val)
                "density":     density,
                "kl":          kl,
                "kl_se":       se,
                "fit_time_s":  fit_time,         # wall clock
                "fit_cpu_s":   cpu_time,         # process CPU time
                "fit_rss_delta_bytes": rss,   # unreliable for NN rows
                **extra,
                **run_settings(n_jobs),
            })
    return records


def run_experiments(node_counts, in_degrees, alphas, n_dags, sample_sizes,
                    estimators, card, val_frac, n_test, n_epochs,
                    results_dir, verbose=True, results_name="results.csv",
                    n_jobs=1, device=None, torch_threads=None):
    """Run the full grid, appending rows to <results_dir>/<results_name>.

    Resumable: configs already in that CSV are skipped, and each DAG's
    train/test samples are cached under <results_dir>/data/ (shared by every
    results_name, so a timing run reuses the same data).
    n_jobs, device, torch_threads : speed settings, see _parallel_map. One job
    is one DAG; rows are written as each job finishes. fit_time_s from a
    parallel run is inflated by contention: take timing from a serial run.
    """
    os.makedirs(results_dir, exist_ok=True)
    results_csv = os.path.join(results_dir, results_name)
    existing_df = pd.read_csv(results_csv) if os.path.exists(results_csv) else pd.DataFrame()

    jobs = []
    for n_nodes in node_counts:
        for in_deg in in_degrees:
            for alpha in alphas:
                n_before = len(jobs)
                for dag in range(1, n_dags + 1):
                    todo = {(est["name"], s) for est in estimators for s in sample_sizes
                            if not config_exists(existing_df, est["name"], dag,
                                                 n_nodes, in_deg, alpha, s)}
                    if todo:
                        jobs.append({"n_nodes": n_nodes, "in_degree": in_deg,
                                     "alpha": alpha, "dag": dag, "todo": todo})
                if verbose and len(jobs) == n_before:
                    tqdm.write(f"loaded results: n={n_nodes} k={in_deg} alpha={alpha}")
    if n_jobs != 1:
        # biggest first, so a large DAG doesn't start last and run alone
        jobs.sort(key=lambda j: (j["n_nodes"], j["in_degree"]), reverse=True)

    run_job = partial(_run_dag_job, estimators=estimators, sample_sizes=sample_sizes,
                      card=card, val_frac=val_frac, n_test=n_test, n_epochs=n_epochs,
                      results_dir=results_dir, n_jobs=n_jobs)
    plot_records = []
    progress_bar = tqdm(total=len(jobs), desc="DAGs")
    for records in _parallel_map(run_job, jobs, n_jobs, device, torch_threads):
        plot_records.extend(records)
        if records:
            r = records[0]
            progress_bar.set_postfix_str(f"n={r['n_nodes']} k={r['in_degree']} "
                                         f"α={r['alpha']} dag={r['dag_idx']}")
        flush(plot_records, results_csv)
        progress_bar.update()
    progress_bar.close()


def _run_sweep_job(job, card, val_frac, wandb_project, hist_dir, results_dir,
                   n_cached, n_test, n_jobs):
    """One sweep run (k, dag, sample size, arm) -> its sweep.csv row."""
    k, dag, samples, arm = job["k"], job["dag"], job["samples"], job["arm"]
    n_nodes, alpha, n_epochs = job["n_nodes"], job["alpha"], job["n_epochs"]
    G, true_model, data, test_data = load_dag(results_dir, dag, k, n_nodes, alpha,
                                              card, n_cached, n_test)
    split_seed = make_seed("split", dag, k, n_nodes, samples)
    nn_seed = make_seed("nn", dag, k, n_nodes, samples)
    train_df, val_df = split_train_val(data.iloc[:samples], val_frac=val_frac,
                                       seed=split_seed)

    kw = {x: v for x, v in arm.items() if x != "name"}
    kw.setdefault("n_epochs", n_epochs)

    cfg = {
        "arm": arm["name"],
        "in_degree": k, "dag_idx": dag, "n_nodes": n_nodes,
        "sample_size": samples, "n_fit": len(train_df),
        "alpha": alpha, "card": card,
        "seed": make_seed("dag", dag, k, n_nodes),   # DAG seed (structure)
        "split_seed": split_seed,      # train/val split
        "nn_seed": nn_seed,            # NN init + batch order
        **run_settings(n_jobs),
        "hidden_dims": str(list(arm["hidden_dims"])),
        "activation": activation_name(arm),
        **{x: kw.get(x, d) for x, d in NN_HPARAMS.items()},
    }

    run = None
    if wandb_project is not None:
        import wandb
        run = wandb.init(
            project=wandb_project,
            group=arm["name"],
            job_type=f"k{k}",
            name=f"{arm['name']}_k{k}_n{samples}",
            tags=[arm["name"], f"k{k}", f"n{samples}"],
            config=cfg, reinit=True,
        )

    tr = Trace(run=run, history=True)
    model, t, cpu_t, _ = timed_fit(
        fit_nn, G, train_df, card,
        val_data=val_df, seed=nn_seed, trace=tr, **kw,
    )

    stats = model.param_stats()
    stops = list(model.stop_epoch_.values())
    if model.history_:
        order = list(model.history_["train"])
        np.savez_compressed(
            os.path.join(hist_dir, f"hist_{arm['name']}_k{k}_d{dag}_n{samples}.npz"),
            train=history_matrix(model, "train"),
            val=history_matrix(model, "val"),
            in_deg=np.array([G.in_degree(v) for v in order]),
            stop=np.array([model.stop_epoch_[v] for v in order]),
        )

    # selection metric: best validation cross-entropy summed over all nodes
    val_loss = float(np.sum(list(model.best_val_.values())))
    # train loss at each node's best-val epoch (eval mode, same nodes)
    train_loss = float(np.sum(list(model.train_loss_.values())))
    summary = {"val_loss": val_loss,
               "train_loss": train_loss,
               "gen_gap": val_loss - train_loss,
               "fit_time_s": t,
               "fit_cpu_s": cpu_t,
               "stored_params": stats["stored_params"],
               "memory_bytes": stats["memory_bytes"],
               "stop_epoch_mean": float(np.mean(stops)),
               "frac_at_ceiling": float(np.mean([s >= kw["n_epochs"] for s in stops]))}
    if test_data is not None:
        summary["kl"], summary["kl_se"] = kl_nn(true_model, model, test_data, card)
    summary["wandb_run_id"] = None if run is None else run.id
    if run is not None:
        run.summary.update(summary)
        run.finish()
    return {**cfg, **summary}


def run_sweep(arms, k_values, n_dags, n_nodes, alpha, sample_sizes,
              card, val_frac, n_epochs,
              wandb_project, sweep_csv, hist_dir,
              results_dir, n_cached, n_test=None, dag_offset=0,
              n_jobs=1, device=None, torch_threads=None):
    """Train every arm on every (k, dag, sample size), logging to W&B.

    Appends one row per run to `sweep_csv` as each run finishes (so an
    interrupted sweep keeps what finished) and writes per-node loss curves to
    `hist_dir`. Uses cached train.csv (under results_dir) and the same
    train/val splits as the main experiment. n_cached must be the main
    experiment's max(SAMPLE_SIZES), so a train.csv first created here matches
    what the main experiment expects.
    n_test : the main experiment's N_TEST. If given, each run also reports
             its true KL on the cached test.csv.
    dag_offset : DAGs used are dag_offset+1 .. dag_offset+n_dags. 0 = the main
                 experiment's DAGs; a large offset (e.g. 100) gives fresh networks
                 from the same generator, for tuning without touching evaluation.
    n_jobs, device, torch_threads : speed settings, see _parallel_map. One job
                 is one run; with W&B each worker opens its own run.
    """
    os.makedirs(hist_dir, exist_ok=True)
    jobs = []
    for k in k_values:
        for dag in range(dag_offset + 1, dag_offset + n_dags + 1):
            load_dag(results_dir, dag, k, n_nodes, alpha, card, n_cached, n_test)  # cache once
            jobs += [{"k": k, "dag": dag, "samples": s, "arm": arm, "n_nodes": n_nodes,
                      "alpha": alpha, "n_epochs": n_epochs}
                     for s in sample_sizes for arm in arms]

    run_job = partial(_run_sweep_job, card=card, val_frac=val_frac,
                      wandb_project=wandb_project, hist_dir=hist_dir,
                      results_dir=results_dir, n_cached=n_cached, n_test=n_test,
                      n_jobs=n_jobs)
    sweep_rows = []
    for row in tqdm(_parallel_map(run_job, jobs, n_jobs, device, torch_threads),
                    total=len(jobs), desc="sweep runs"):
        sweep_rows.append(row)
        flush([row], sweep_csv)
    return pd.DataFrame(sweep_rows)


def parse_hidden_dims(spec):
    """W&B-config string -> hidden_dims tuple: "128_64" -> (128, 64), "linear" -> ()."""
    spec = str(spec)
    return () if spec in ("", "linear", "[]") else tuple(int(h) for h in spec.split("_"))


def _trial_cell(job, hidden_dims, activation, kw, card, val_frac, results_dir,
                n_cached, n_test):
    """One (k, dag, sample size) cell of a sweep trial -> its row (no W&B)."""
    k, dag, samples, n_nodes, alpha = (job["k"], job["dag"], job["samples"],
                                       job["n_nodes"], job["alpha"])
    G, true_model, data, test_data = load_dag(results_dir, dag, k, n_nodes, alpha,
                                              card, n_cached, n_test)
    train_df, val_df = split_train_val(
        data.iloc[:samples], val_frac=val_frac,
        seed=make_seed("split", dag, k, n_nodes, samples))
    model, t, cpu_t, _ = timed_fit(
        fit_nn, G, train_df, card, val_data=val_df,
        hidden_dims=hidden_dims, activation=activation,
        seed=make_seed("nn", dag, k, n_nodes, samples), **kw)

    kl, kl_se = kl_nn(true_model, model, test_data, card)
    stops = list(model.stop_epoch_.values())
    return {
        "in_degree": k, "dag_idx": dag,
        "n_nodes": n_nodes, "sample_size": samples,
        "n_fit": len(train_df), "alpha": alpha, "card": card,
        "hidden_dims": str(list(hidden_dims)),
        "activation": "linear" if activation is None else activation.__name__,
        **{x: kw.get(x, d) for x, d in NN_HPARAMS.items()},
        "val_loss": float(np.sum(list(model.best_val_.values()))),
        "kl": kl, "kl_se": kl_se,
        "fit_time_s": t, "fit_cpu_s": cpu_t,
        "stored_params": model.param_stats()["stored_params"],
        "stop_epoch_mean": float(np.mean(stops)),
        "frac_at_ceiling": float(np.mean([e >= kw["n_epochs"] for e in stops])),
        "device": _models.DEVICE, "torch_threads": torch.get_num_threads(),
    }


def sweep_trial(k_values, n_dags, n_nodes, alpha, sample_sizes, card, val_frac,
                results_dir, n_cached, n_test, trial_csv, dag_offset=0,
                n_jobs=1, device=None, torch_threads=None):
    """One W&B-sweep trial: a single hyperparameter config, scored on every
    (k, dag, sample size) cell. Use as wandb.agent(sweep_id, function=partial(...)).

    The config comes from wandb.config: hidden_dims as a string ("128_64",
    "linear") plus any NN_HPARAMS keys. Two objectives are logged; the sweep
    config's metric picks one:
      mean_val_loss : mean over cells of the summed per-node best validation
                      cross-entropy (uses only each DAG's train/val data)
      mean_kl       : mean true KL on each DAG's test.csv. Only use this to
                      select when dag_offset keeps the tuning DAGs disjoint
                      from the evaluation DAGs.
    dag_offset as in run_sweep; same DAG indexing, data, splits and seeds, so
    results stay paired with run_sweep at the same offset.
    n_jobs, device, torch_threads : speed settings, see _parallel_map. The
    cells are fitted in parallel; all W&B logging stays in this process.
    Appends one row per cell to `trial_csv`.
    """
    import wandb

    run = wandb.init()
    config = dict(run.config)
    hidden_dims = parse_hidden_dims(config["hidden_dims"])
    activation = nn.ReLU if hidden_dims else None
    kw = {x: config[x] for x in NN_HPARAMS if x in config}
    kw.setdefault("n_epochs", NN_HPARAMS["n_epochs"])

    jobs = []
    for k in k_values:
        for dag in range(dag_offset + 1, dag_offset + n_dags + 1):
            load_dag(results_dir, dag, k, n_nodes, alpha, card, n_cached, n_test)  # cache once
            jobs += [{"k": k, "dag": dag, "samples": s, "n_nodes": n_nodes, "alpha": alpha}
                     for s in sample_sizes]

    run_cell = partial(_trial_cell, hidden_dims=hidden_dims, activation=activation,
                       kw=kw, card=card, val_frac=val_frac, results_dir=results_dir,
                       n_cached=n_cached, n_test=n_test)
    rows = list(_parallel_map(run_cell, jobs, n_jobs, device, torch_threads))
    rows.sort(key=lambda r: (r["in_degree"], r["dag_idx"], r["sample_size"]))

    metrics = {}
    for r in rows:
        r["run_id"] = run.id
        r["n_jobs"] = n_jobs
        cell = f"k{r['in_degree']}_d{r['dag_idx']}_n{r['sample_size']}"
        metrics[f"val_loss/{cell}"] = r["val_loss"]
        metrics[f"kl/{cell}"] = r["kl"]
    metrics["mean_val_loss"] = float(np.mean([r["val_loss"] for r in rows]))
    metrics["mean_kl"] = float(np.mean([r["kl"] for r in rows]))
    metrics["frac_at_ceiling"] = float(np.mean([r["frac_at_ceiling"] for r in rows]))
    run.log(metrics)
    run.finish()
    flush(rows, trial_csv)
    return metrics
