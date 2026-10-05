"""Experiment grid (MLE vs neural CPDs) and the hyperparameter sweep."""
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .evaluation import kl_mle, kl_nn, mle_param_stats, timed_fit
from .generation import gen_CPTs, gen_dag_indegree, make_seed
from .mle import fit_mle, fit_mle_pseudocount
from .models import DEVICE, NeuralCPDs
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


# train_nn_cpds keyword arguments an estimator / arm / sweep config may set,
# with train_nn_cpds' defaults (used when recording a run's config)
NN_HPARAMS = {"n_epochs": 500, "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.0,
              "batch_size": 64, "patience": 20, "optimizer": "adam"}


def activation_name(est):
    return "linear" if est["activation"] is None else est["activation"].__name__


def fit_estimator(est, G, sample, train_df, val_df, card, n_epochs, seed):
    """Fit one entry of ESTIMATORS -> (model, fit_time_s, rss_delta_bytes).

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


def run_experiments(node_counts, in_degrees, alphas, n_dags, sample_sizes,
                    estimators, card, val_frac, n_test, n_epochs,
                    results_dir, verbose=True):
    """Run the full grid, appending rows to <results_dir>/results.csv.

    Resumable: configs already in results.csv are skipped, and each DAG's
    train/test samples are cached under <results_dir>/data/.
    """
    os.makedirs(results_dir, exist_ok=True)
    results_csv = os.path.join(results_dir, "results.csv")

    experiment_configs = [(n, k, a) for n in node_counts for k in in_degrees for a in alphas]

    plot_records = []
    existing_df = pd.read_csv(results_csv) if os.path.exists(results_csv) else pd.DataFrame()

    progress_bar = tqdm(experiment_configs, desc="configs")
    for n_nodes, in_deg, alpha in progress_bar:

        todo = [(est["name"], s, dag) for est in estimators for s in sample_sizes
                for dag in range(1, n_dags + 1)
                if not config_exists(existing_df, est["name"], dag,
                                     n_nodes, in_deg, alpha, s)]
        if not todo:
            if verbose:
                tqdm.write(f"loaded results: n={n_nodes} k={in_deg} alpha={alpha}")
            continue

        for dag in range(1, n_dags + 1):
            if not any(d == dag for _, _, d in todo):
                continue

            G, true_model = build_dag(dag, in_deg, n_nodes, alpha, card)
            dag_dir = dag_data_dir(results_dir, alpha, n_nodes, in_deg, dag)

            # actual density of this realised graph
            max_possible = n_nodes * (n_nodes - 1) / 2
            density = G.number_of_edges() / max_possible if max_possible > 0 else 0

            # Generate train & test data ONCE per DAG, outside the sample-size loop
            train_seed = make_seed("train", dag, in_deg, n_nodes)
            test_seed = make_seed("test", dag, in_deg, n_nodes)
            data = load_or_simulate(os.path.join(dag_dir, "train.csv"), true_model,
                                    max(sample_sizes), train_seed)
            test_data = load_or_simulate(os.path.join(dag_dir, "test.csv"), true_model,
                                         n_test, test_seed)

            for samples in sample_sizes:
                split_seed = make_seed("split", dag, in_deg, n_nodes, samples)
                nn_seed = make_seed("nn", dag, in_deg, n_nodes, samples)

                sample = data.iloc[:samples]
                train_df, val_df = split_train_val(sample, val_frac=val_frac, seed=split_seed)

                for est in estimators:
                    if (est["name"], samples, dag) not in todo:
                        continue
                    progress_bar.set_postfix_str(f"\nn={n_nodes} k={in_deg} α={alpha} "
                                                 f"dag={dag} N={samples} {est['name']}")

                    model, fit_time, rss = fit_estimator(
                        est, G, sample, train_df, val_df, card, n_epochs, seed=nn_seed)
                    kl, se, extra = evaluate(model, true_model, test_data, card, n_epochs)

                    is_nn = est["type"] == "nn"
                    n_fit = len(train_df) if is_nn else len(sample)
                    plot_records.append({
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
                        "fit_time_s":  fit_time,
                        "fit_rss_delta_bytes": rss,   # unreliable for NN rows
                        **extra,
                    })

            flush(plot_records, results_csv)


def run_sweep(arms, k_values, n_dags, n_nodes, alpha, sample_sizes,
              card, val_frac, n_epochs,
              wandb_project, sweep_csv, hist_dir,
              results_dir, n_cached, n_test=None, dag_offset=0):
    """Train every arm on every (k, dag, sample size), logging to W&B.

    Appends one row per run to `sweep_csv` (after each DAG, so an interrupted
    sweep keeps what finished) and writes per-node loss curves to `hist_dir`.
    Uses the main experiment's DAGs, cached train.csv (under results_dir) and
    train/val splits. n_cached must be the main experiment's max(SAMPLE_SIZES),
    so a train.csv first created here matches what the main experiment expects.
    n_test : the main experiment's N_TEST. If given, each run also reports
             its true KL on the cached test.csv.
    dag_offset : DAGs used are dag_offset+1 .. dag_offset+n_dags. 0 = the main
                 experiment's DAGs; a large offset (e.g. 100) gives fresh networks
                 from the same generator, for tuning without touching evaluation.
    """
    if wandb_project is not None:
        import wandb

    os.makedirs(hist_dir, exist_ok=True)
    sweep_rows, pending = [], []      # all rows this call / rows not yet on disk

    for k in k_values:
        for dag in range(dag_offset + 1, dag_offset + n_dags + 1):
            G, true_model = build_dag(dag, k, n_nodes, alpha, card)
            seed = make_seed("dag", dag, k, n_nodes)
            dag_dir = dag_data_dir(results_dir, alpha, n_nodes, k, dag)
            train_seed = make_seed("train", dag, k, n_nodes)
            data = load_or_simulate(os.path.join(dag_dir, "train.csv"), true_model,
                                    n_cached, train_seed)
            test_data = None if n_test is None else load_or_simulate(
                os.path.join(dag_dir, "test.csv"), true_model, n_test,
                make_seed("test", dag, k, n_nodes))

            for samples in sample_sizes:
                split_seed = make_seed("split", dag, k, n_nodes, samples)
                nn_seed = make_seed("nn", dag, k, n_nodes, samples)

                train_df, val_df = split_train_val(data.iloc[:samples], val_frac=val_frac,
                                                   seed=split_seed)

                for arm in arms:
                    kw = {x: v for x, v in arm.items() if x != "name"}
                    kw.setdefault("n_epochs", n_epochs)

                    cfg = {
                        "arm": arm["name"],
                        "in_degree": k, "dag_idx": dag, "n_nodes": n_nodes,
                        "sample_size": samples, "n_fit": len(train_df),
                        "alpha": alpha, "card": card,
                        "seed": seed,                  # DAG seed (structure)
                        "split_seed": split_seed,      # train/val split
                        "nn_seed": nn_seed,            # NN init + batch order
                        "device": DEVICE, "torch_version": torch.__version__,
                        "hidden_dims": str(list(arm["hidden_dims"])),
                        "activation": activation_name(arm),
                        **{x: kw.get(x, d) for x, d in NN_HPARAMS.items()},
                    }

                    run = None if wandb_project is None else wandb.init(
                        project=wandb_project,
                        group=arm["name"],
                        job_type=f"k{k}",
                        name=f"{arm['name']}_k{k}_n{samples}",
                        tags=[arm["name"], f"k{k}", f"n{samples}"],
                        config=cfg, reinit=True,
                    )

                    tr = Trace(run=run, history=True)
                    model, t, _ = timed_fit(
                        fit_nn, G, train_df, card,
                        val_data=val_df, seed=nn_seed, trace=tr, **kw,
                    )

                    stats = model.param_stats()
                    stops = list(model.stop_epoch_.values())
                    stop_mean = float(np.mean(stops))
                    ceiling = float(np.mean([s >= kw["n_epochs"] for s in stops]))
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
                               "stored_params": stats["stored_params"],
                               "memory_bytes": stats["memory_bytes"],
                               "stop_epoch_mean": stop_mean,
                               "frac_at_ceiling": ceiling}
                    if test_data is not None:
                        summary["kl"], summary["kl_se"] = kl_nn(true_model, model,
                                                                test_data, card)
                    summary["wandb_run_id"] = None if run is None else run.id
                    if run is not None:
                        run.summary.update(summary)
                        run.finish()

                    sweep_rows.append({**cfg, **summary})
                    pending.append(sweep_rows[-1])

            flush(pending, sweep_csv)
            print(f"k={k} dag={dag} done  ({len(sweep_rows)} rows)")
    return pd.DataFrame(sweep_rows)


def parse_hidden_dims(spec):
    """W&B-config string -> hidden_dims tuple: "128_64" -> (128, 64), "linear" -> ()."""
    spec = str(spec)
    return () if spec in ("", "linear", "[]") else tuple(int(h) for h in spec.split("_"))


def sweep_trial(k_values, n_dags, n_nodes, alpha, sample_sizes, card, val_frac,
                results_dir, n_cached, n_test, trial_csv, dag_offset=0):
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
    Appends one row per cell to `trial_csv`.
    """
    import wandb

    run = wandb.init()
    config = dict(run.config)
    hidden_dims = parse_hidden_dims(config["hidden_dims"])
    activation = nn.ReLU if hidden_dims else None
    kw = {x: config[x] for x in NN_HPARAMS if x in config}
    kw.setdefault("n_epochs", NN_HPARAMS["n_epochs"])

    rows, metrics = [], {}
    for k in k_values:
        for dag in range(dag_offset + 1, dag_offset + n_dags + 1):
            G, true_model = build_dag(dag, k, n_nodes, alpha, card)
            dag_dir = dag_data_dir(results_dir, alpha, n_nodes, k, dag)
            data = load_or_simulate(os.path.join(dag_dir, "train.csv"), true_model,
                                    n_cached, make_seed("train", dag, k, n_nodes))
            test_data = load_or_simulate(os.path.join(dag_dir, "test.csv"), true_model,
                                         n_test, make_seed("test", dag, k, n_nodes))

            for samples in sample_sizes:
                train_df, val_df = split_train_val(
                    data.iloc[:samples], val_frac=val_frac,
                    seed=make_seed("split", dag, k, n_nodes, samples))
                model, t, _ = timed_fit(
                    fit_nn, G, train_df, card, val_data=val_df,
                    hidden_dims=hidden_dims, activation=activation,
                    seed=make_seed("nn", dag, k, n_nodes, samples), **kw)

                val_loss = float(np.sum(list(model.best_val_.values())))
                kl, kl_se = kl_nn(true_model, model, test_data, card)
                stops = list(model.stop_epoch_.values())
                cell = f"k{k}_d{dag}_n{samples}"
                metrics[f"val_loss/{cell}"] = val_loss
                metrics[f"kl/{cell}"] = kl
                rows.append({
                    "run_id": run.id, "in_degree": k, "dag_idx": dag,
                    "n_nodes": n_nodes, "sample_size": samples,
                    "n_fit": len(train_df), "alpha": alpha, "card": card,
                    "hidden_dims": str(list(hidden_dims)),
                    "activation": "linear" if activation is None else activation.__name__,
                    **{x: kw.get(x, d) for x, d in NN_HPARAMS.items()},
                    "val_loss": val_loss, "kl": kl, "kl_se": kl_se,
                    "fit_time_s": t,
                    "stored_params": model.param_stats()["stored_params"],
                    "stop_epoch_mean": float(np.mean(stops)),
                    "frac_at_ceiling": float(np.mean([e >= kw["n_epochs"] for e in stops])),
                })

    metrics["mean_val_loss"] = float(np.mean([r["val_loss"] for r in rows]))
    metrics["mean_kl"] = float(np.mean([r["kl"] for r in rows]))
    metrics["frac_at_ceiling"] = float(np.mean([r["frac_at_ceiling"] for r in rows]))
    run.log(metrics)
    run.finish()
    flush(rows, trial_csv)
    return metrics
