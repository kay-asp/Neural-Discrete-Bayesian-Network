"""Experiment grid (MLE vs neural CPDs) and the hyperparameter sweep."""
import ast
import json
import logging
import os
import subprocess
from functools import lru_cache, partial

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


# Rows written before cardinality was recorded all used card=2.
LEGACY_CARD = 2


def with_card(df):
    """df with a `card` column, rows missing it (older CSVs) set to LEGACY_CARD."""
    df = df.copy()
    df["card"] = df["card"].fillna(LEGACY_CARD) if "card" in df else LEGACY_CARD
    return df


def config_exists(df, estimator, dag, n_nodes, in_deg, alpha, sample_size, card=None):
    """Is this config already in df? card=None skips the card check; pass
    with_card(df) so rows from before card was recorded count as card=2."""
    if df.empty or "estimator" not in df.columns:
        return False
    match = (
        (df["estimator"]   == estimator) &
        (df["dag_idx"] == dag) &
        (df["n_nodes"] == n_nodes) &
        (df["in_degree"] == in_deg) &
        (df["alpha"] == alpha) &
        (df["sample_size"] == sample_size)
    )
    if card is not None:
        match &= df["card"] == card
    return match.any()

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
    # alpha and card are left out of the seeds, so every alpha and card shares
    # the same DAG structure and data seeds (paired comparisons)
    G = gen_dag_indegree(n_nodes, in_deg, seed=make_seed("dag", dag, in_deg, n_nodes))
    true_model = gen_CPTs(G, card, alpha, seed=make_seed("cpt", dag, in_deg, n_nodes))
    return G, true_model


def dag_data_dir(results_dir, alpha, n_nodes, in_deg, dag, card):
    """Where one DAG's cached train.csv / test.csv live (keyed on card too, so
    different cardinalities never share cached samples)."""
    path = os.path.join(results_dir, "data", f"card_{card}", f"alpha_{alpha}",
                        f"BN_{n_nodes}", f"indegree_{in_deg}", f"dag_{dag}")
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
    dag_dir = dag_data_dir(results_dir, alpha, n_nodes, in_deg, dag, card)
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


@lru_cache(maxsize=None)
def code_version():
    """Short git commit of this code, "+dirty" if it has uncommitted changes;
    None outside a git checkout. Recorded on every row."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here,
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=here,
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("+dirty" if dirty else "")
    except Exception:
        return None


def run_settings(n_jobs):
    """Columns recording how a row was run, so timings are read in context."""
    return {"n_jobs": n_jobs, "device": _models.DEVICE,
            "torch_threads": torch.get_num_threads(),
            "torch_version": torch.__version__, "code_version": code_version()}


# train_nn_cpds keyword arguments an estimator / arm / sweep config may set,
# with train_nn_cpds' defaults (used when recording a run's config)
NN_HPARAMS = {"n_epochs": 500, "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.0,
              "batch_size": 64, "patience": 20, "optimizer": "adam"}


def activation_name(est):
    return "linear" if est["activation"] is None else est["activation"].__name__


def fit_estimator(est, G, sample, train_df, val_df, cards, n_epochs, seed):
    """Fit one entry of ESTIMATORS -> (model, wall_s, cpu_s, rss_delta_bytes).

    Tabular estimators fit on the whole budget `sample` (= train_df + val_df):
    they have no early stopping, so they get every row the NN sees.
    cards : an int (shared) or node -> cardinality.
    """
    if est["type"] == "mle":
        return timed_fit(fit_mle, G, sample, cards)
    if est["type"] == "dirichlet":
        return timed_fit(fit_mle_pseudocount, G, sample, cards,
                         pseudo_count=est.get("pseudo_count", 0.5))
    # optional tuned hyperparameters (lr, weight_decay, dropout, ...) pass through
    kw = {x: est[x] for x in NN_HPARAMS if x in est}
    kw.setdefault("n_epochs", n_epochs)
    return timed_fit(
        fit_nn, G, train_df, cards,
        val_data=val_df,
        hidden_dims=est["hidden_dims"],
        activation=est["activation"],
        seed=seed,
        **kw,
    )


def model_stats(model, n_epochs):
    """Size stats of a fitted model, plus early-stopping stats for an NN."""
    if not isinstance(model, NeuralCPDs):
        return mle_param_stats(model)
    extra = model.param_stats()
    extra["stop_epoch_mean"] = float(np.mean(list(model.stop_epoch_.values())))
    extra["best_epoch_mean"] = float(np.mean(list(model.best_epoch_.values())))
    extra["frac_at_ceiling"] = float(np.mean(
        [e >= n_epochs for e in model.stop_epoch_.values()]))
    return extra


def evaluate(model, true_model, test_data, cards, n_epochs):
    """KL vs ground truth plus size / early-stopping stats -> (kl, se, extra)."""
    if isinstance(model, NeuralCPDs):
        kl, se = kl_nn(true_model, model, test_data, cards)
    else:
        kl, se = kl_mle(true_model, model, test_data)
    return kl, se, model_stats(model, n_epochs)


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
                "card":        card,
                "sample_size": samples,          # budget
                "n_fit":       n_fit,            # rows fitted (NN excludes val)
                "density":     density,
                # every seed this row used (recomputable from the config via
                # make_seed, stored in case that ever changes)
                "dag_seed":    make_seed("dag", dag, in_deg, n_nodes),
                "cpt_seed":    make_seed("cpt", dag, in_deg, n_nodes),
                "train_seed":  make_seed("train", dag, in_deg, n_nodes),
                "test_seed":   make_seed("test", dag, in_deg, n_nodes),
                "split_seed":  split_seed,
                "nn_seed":     nn_seed,
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

    Resumable: configs already in that CSV (for this card) are skipped, and
    each DAG's train/test samples are cached under <results_dir>/data/card_*/
    (shared by every results_name, so a timing run reuses the same data).
    Rows record card, so runs with different cardinalities can share one CSV.
    n_jobs, device, torch_threads : speed settings, see _parallel_map. One job
    is one DAG; rows are written as each job finishes. fit_time_s from a
    parallel run is inflated by contention: take timing from a serial run.
    """
    os.makedirs(results_dir, exist_ok=True)
    results_csv = os.path.join(results_dir, results_name)
    existing_df = with_card(pd.read_csv(results_csv)) if os.path.exists(results_csv) \
        else pd.DataFrame()

    jobs = []
    for n_nodes in node_counts:
        for in_deg in in_degrees:
            for alpha in alphas:
                n_before = len(jobs)
                for dag in range(1, n_dags + 1):
                    todo = {(est["name"], s) for est in estimators for s in sample_sizes
                            if not config_exists(existing_df, est["name"], dag,
                                                 n_nodes, in_deg, alpha, s, card)}
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
            config=cfg, reinit="finish_previous",
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
            os.path.join(hist_dir, f"hist_{arm['name']}_c{card}_k{k}_d{dag}_n{samples}.npz"),
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
               "best_epoch_mean": float(np.mean(list(model.best_epoch_.values()))),
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


# activations a sweep config may name ("activation": "tanh"); default relu
ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU,
               "elu": nn.ELU, "leaky_relu": nn.LeakyReLU, "sigmoid": nn.Sigmoid}


def _trial_cell(job, hidden_dims, activation, kw, card, val_frac, results_dir,
                n_cached, n_test):
    """One (k, dag, sample size) cell of a sweep trial -> its row (no W&B)."""
    k, dag, samples, n_nodes, alpha = (job["k"], job["dag"], job["samples"],
                                       job["n_nodes"], job["alpha"])
    G, true_model, data, test_data = load_dag(results_dir, dag, k, n_nodes, alpha,
                                              card, n_cached, n_test)
    split_seed = make_seed("split", dag, k, n_nodes, samples)
    nn_seed = make_seed("nn", dag, k, n_nodes, samples)
    train_df, val_df = split_train_val(data.iloc[:samples], val_frac=val_frac,
                                       seed=split_seed)
    model, t, cpu_t, _ = timed_fit(
        fit_nn, G, train_df, card, val_data=val_df,
        hidden_dims=hidden_dims, activation=activation, seed=nn_seed, **kw)

    kl, kl_se = kl_nn(true_model, model, test_data, card)
    stops = list(model.stop_epoch_.values())
    return {
        "in_degree": k, "dag_idx": dag,
        "n_nodes": n_nodes, "sample_size": samples,
        "n_fit": len(train_df), "alpha": alpha, "card": card,
        "val_frac": val_frac, "n_test": n_test, "n_cached": n_cached,
        "dag_seed": make_seed("dag", dag, k, n_nodes),
        "split_seed": split_seed, "nn_seed": nn_seed,
        "hidden_dims": str(list(hidden_dims)),
        "activation": "linear" if activation is None else activation.__name__,
        **{x: kw.get(x, d) for x, d in NN_HPARAMS.items()},
        "val_loss": float(np.sum(list(model.best_val_.values()))),
        "kl": kl, "kl_se": kl_se,
        "fit_time_s": t, "fit_cpu_s": cpu_t,
        "stored_params": model.param_stats()["stored_params"],
        "stop_epoch_mean": float(np.mean(stops)),
        "best_epoch_mean": float(np.mean(list(model.best_epoch_.values()))),
        "frac_at_ceiling": float(np.mean([e >= kw["n_epochs"] for e in stops])),
        "device": _models.DEVICE, "torch_threads": torch.get_num_threads(),
        "torch_version": torch.__version__, "code_version": code_version(),
    }


def _score_config(run, run_id, run_name, sweep_id, source, settings, hidden_dims,
                  activation, kw, k_values, n_dags, n_nodes, alpha, sample_sizes, card,
                  val_frac, results_dir, n_cached, n_test, trial_csv, dag_offset,
                  n_jobs, device, torch_threads):
    """Fit one config on every (k, dag, sample size) cell, append one row per
    cell to trial_csv, log summaries to the W&B run (if any) and return them.
    Shared by sweep_trial (W&B sweep trials) and run_configs (chosen configs),
    so both write the same rows to the same tuning data."""
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

    for r in rows:
        # settings first, then the cell's own columns (they win on a clash,
        # e.g. the cell's sample_size over the trial's sample_sizes list)
        r.update({**{x: v for x, v in settings.items() if x not in r},
                  "run_id": run_id, "run_name": run_name, "sweep_id": sweep_id,
                  "source": source, "n_cells": len(rows)})

    cells = pd.DataFrame(rows)
    metrics = {
        "mean_val_loss": float(cells["val_loss"].mean()),
        "mean_kl": float(cells["kl"].mean()),
        # spread across DAGs: std of each DAG's KL averaged over sample sizes
        "std_kl": float(cells.groupby(["in_degree", "dag_idx"])["kl"].mean().std()),
        "frac_at_ceiling": float(cells["frac_at_ceiling"].mean()),
    }
    if run is not None:
        # per sample size (mean over DAGs): one logged step per N, with
        # sample_size as the x-axis, so W&B plots mean_kl against sample_size
        run.define_metric("sample_size")
        for m in ("mean_kl", "mean_val_loss"):
            run.define_metric(m, step_metric="sample_size")
        for n, g in cells.groupby("sample_size"):
            run.log({"sample_size": int(n), "mean_kl": float(g["kl"].mean()),
                     "mean_val_loss": float(g["val_loss"].mean())})
        # run summary = the objectives over every cell; set after the per-N
        # steps so it isn't left at the last sample size's value
        run.summary.update(metrics)
    flush(rows, trial_csv)
    return metrics


def sweep_trial(k_values, n_dags, n_nodes, alpha, sample_sizes, card, val_frac,
                results_dir, n_cached, n_test, trial_csv, dag_offset=0,
                n_jobs=1, device=None, torch_threads=None):
    """One W&B-sweep trial: a single hyperparameter config, scored on every
    (k, dag, sample size) cell. Use as wandb.agent(sweep_id, function=partial(...)).

    The config comes from wandb.config: hidden_dims as a string ("128_64",
    "linear"), optionally activation (a key of ACTIVATIONS, default "relu"),
    plus any NN_HPARAMS keys. Every argument of this function except file
    paths is also written to the W&B config and to each CSV row (lists as
    strings), so a setting added or changed later is recorded automatically;
    rows also carry the sweep id, seeds and code version. Two objectives are logged; the sweep
    config's metric picks one:
      mean_val_loss : mean over cells of the summed per-node best validation
                      cross-entropy (uses only each DAG's train/val data)
      mean_kl       : mean true KL on each DAG's test.csv. Only use this to
                      select when dag_offset keeps the tuning DAGs disjoint
                      from the evaluation DAGs.
    dag_offset as in run_sweep; same DAG indexing, data, splits and seeds, so
    results stay paired with run_sweep at the same offset.
    The run summary holds those two (over every cell) plus std_kl (spread
    across DAGs) and frac_at_ceiling. Per sample size, mean_kl / mean_val_loss
    (over DAGs) are also logged as steps against sample_size, for plotting
    mean_kl vs sample_size in W&B. Per-cell results go only to trial_csv
    (one row per cell). Runs are named e.g. "07_256x128_lr3e-4" (W&B trial
    number, hidden_dims, learning rate); the fixed settings (in-degree, card,
    ...) are in the sweep's name and every run's config.
    n_jobs, device, torch_threads : speed settings, see _parallel_map. The
    cells are fitted in parallel; all W&B logging stays in this process.
    """
    # every fixed setting of this trial (all arguments except paths), captured
    # before any other local exists; logged to W&B and stored on every row
    args = dict(locals())
    settings = {name: (str(list(v)) if isinstance(v, (list, tuple)) else v)
                for name, v in args.items() if name not in ("results_dir", "trial_csv")}
    settings["in_degree"] = k_values[0] if len(k_values) == 1 else str(list(k_values))

    import wandb

    run = wandb.init()
    config = dict(run.config)
    # short display name, e.g. "07_256x128_lr3e-4": W&B's trial number (its
    # auto name ends in it: "genial-sweep-7"), zero-padded so names sort in
    # trial order, then the architecture and learning rate. Other settings are
    # in the config.
    trial_no = run.name.rsplit("-", 1)[-1] if run.name else run.id
    trial_no = f"{int(trial_no):02d}" if trial_no.isdigit() else trial_no
    mantissa, exponent = f"{config.get('lr', NN_HPARAMS['lr']):.0e}".split("e")
    run.name = (f"{trial_no}_{str(config['hidden_dims']).replace('_', 'x')}"
                f"_lr{mantissa}e{int(exponent)}")
    # fixed settings, so W&B can filter / group runs by them (not sweep
    # parameters, so the search ignores them)
    run.config.update({**settings, "code_version": code_version()},
                      allow_val_change=True)
    sweep_id = getattr(run, "sweep_id", None)
    hidden_dims = parse_hidden_dims(config["hidden_dims"])
    activation = ACTIVATIONS[config.get("activation", "relu")] if hidden_dims else None
    kw = {x: config[x] for x in NN_HPARAMS if x in config}
    kw.setdefault("n_epochs", NN_HPARAMS["n_epochs"])

    metrics = _score_config(
        run, run.id, run.name, sweep_id, "sweep", settings, hidden_dims, activation, kw,
        k_values, n_dags, n_nodes, alpha, sample_sizes, card, val_frac, results_dir,
        n_cached, n_test, trial_csv, dag_offset, n_jobs, device, torch_threads)
    where = (run.entity, run.project, run.id, run.name)   # read before finish()
    run.finish()
    _rename_wandb_run(*where)
    return metrics


def _resolve_sweep(sweep_id, project):
    """(entity, project, sweep id) of an existing W&B sweep, looked up with
    W&B's API so runs can be attached to it under the right entity/project.
    sweep_id is "id", "project/id" or "entity/project/id"; a bare id is looked
    for in `project` under the default entity."""
    import wandb
    parts = str(sweep_id).split("/")
    api = wandb.Api()
    if len(parts) == 1:
        path = f"{api.default_entity}/{project}/{parts[0]}"
    elif len(parts) == 2:
        path = f"{api.default_entity}/{parts[0]}/{parts[1]}"
    else:
        path = "/".join(parts[-3:])
    try:
        sweep = api.sweep(path)
    except Exception as e:
        raise ValueError(
            f"W&B sweep {sweep_id!r} not found at {path!r} ({e}). Check the id on the sweep's page, "
            "or give it in full as 'entity/project/sweep_id'.") from None
    return sweep.entity, sweep.project, sweep.id


def run_configs(configs, k_values, n_dags, n_nodes, alpha, sample_sizes, card, val_frac,
                results_dir, n_cached, n_test, trial_csv, dag_offset=0,
                wandb_project=None, sweep_ids=None, tag="candidate", source="candidates",
                skip_existing=True, n_jobs=1, device=None, torch_threads=None):
    """Run chosen configs exactly like sweep trials and add them to the tuning data.

    configs : rows of plotting.top_config_values (e.g. top_values.loc[[0, 3]]),
              or a list of dicts as run_sweep arms take them: "name", "hidden_dims"
              (tuple), "activation" (nn class or None) and any NN_HPARAMS
              (lr, weight_decay, dropout, patience, batch_size, optimizer,
              n_epochs). Names must be unique.
    Each config is fitted on every (k in k_values, dag, sample size) cell, and
    its rows are appended to trial_csv (hparam_trials.csv) in the same format
    as sweep trials, with source=`source` (sweep trials have "sweep"), so the
    sweep summaries include them and candidate_table can pick them out.
    sweep_ids : {in_degree: W&B sweep id}, e.g. {6: "ey794fwc", 10: "p8lmbfxj"}
              ("id", "project/id" or "entity/project/id"; looked up with W&B's
              API first, so runs use the sweep's own entity and project).
              Each config then runs once per in-degree (like sweep trials), and
              each run is added to that in-degree's sweep (via WANDB_SWEEP_ID,
              as W&B's agent does) and its rows get that sweep_id. Without it,
              one run covers all of k_values.
    wandb_project : log each run to this W&B project (the sweeps' project),
              tagged `tag`; None skips W&B.
    skip_existing : skip a (config, in-degree) that already has rows from
              `source` for every cell in trial_csv, so re-running only adds
              what's missing. Returns one row of summary metrics per run.
    """
    import uuid

    if isinstance(configs, pd.DataFrame):          # rows of plotting.top_config_values
        configs = arms_from_configs(configs)
    names = [c["name"] for c in configs]
    if len(set(names)) != len(names):
        raise ValueError(f"config names must be unique: {names}")
    sweep_ids = dict(sweep_ids or {})
    groups = [[k] for k in k_values] if sweep_ids else [list(k_values)]
    # look every sweep up first, so a wrong id fails before any training
    where = {k: _resolve_sweep(sid, wandb_project) for k, sid in sweep_ids.items()
             if wandb_project is not None and k in k_values}

    # (config name, in-degree) pairs already complete in the tuning data
    have = {}
    if skip_existing and os.path.exists(trial_csv):
        old = pd.read_csv(trial_csv)
        if "source" in old:
            old = old[(old["source"] == source) & old["sample_size"].isin(sample_sizes)]
            have = (old.drop_duplicates(["run_name", "in_degree", "dag_idx", "sample_size"])
                       .groupby(["run_name", "in_degree"]).size().to_dict())

    results = []
    for cfg in tqdm(configs, desc="configs"):
        name = cfg["name"]
        hidden_dims = tuple(cfg.get("hidden_dims", ()))
        activation = cfg.get("activation", nn.ReLU) if hidden_dims else None
        kw = {x: cfg[x] for x in NN_HPARAMS if x in cfg}
        kw.setdefault("n_epochs", NN_HPARAMS["n_epochs"])
        for ks in groups:
            if all(have.get((name, k), 0) >= n_dags * len(sample_sizes) for k in ks):
                tqdm.write(f"{name} k={ks}: already in {os.path.basename(trial_csv)}, skipped")
                continue
            sweep = where.get(ks[0]) if len(ks) == 1 else None     # (entity, project, id)
            sweep_id = sweep[2] if sweep else (sweep_ids.get(ks[0]) if len(ks) == 1 else None)
            settings = {"k_values": str(ks), "n_dags": n_dags, "n_nodes": n_nodes,
                        "alpha": alpha, "sample_sizes": str(list(sample_sizes)), "card": card,
                        "val_frac": val_frac, "n_cached": n_cached, "n_test": n_test,
                        "dag_offset": dag_offset, "n_jobs": n_jobs, "device": device,
                        "torch_threads": torch_threads,
                        "in_degree": ks[0] if len(ks) == 1 else str(ks)}
            run = None
            if wandb_project is not None:
                import wandb
                env = ("WANDB_SWEEP_ID", "WANDB_ENTITY", "WANDB_PROJECT")
                prev = {v: os.environ.get(v) for v in env}
                extra = {}
                if sweep:      # attach the run to this sweep, as W&B's agent does
                    os.environ.update(dict(zip(env, (sweep[2], sweep[0], sweep[1]))))
                    # wandb reads WANDB_SWEEP_ID only when it first starts in a kernel,
                    # so also pass the sweep id in this run's own settings
                    try:
                        extra["settings"] = wandb.Settings(sweep_id=sweep[2])
                    except Exception:
                        pass
                try:
                    run = wandb.init(
                        entity=sweep[0] if sweep else None,
                        project=sweep[1] if sweep else wandb_project,
                        name=name, tags=[tag], reinit="finish_previous", **extra,
                        config={**settings, "code_version": code_version(), **kw,
                                "hidden_dims": "_".join(map(str, hidden_dims)) or "linear",
                                "activation": "linear" if activation is None
                                              else activation.__name__})
                finally:
                    for v, old in prev.items():
                        if old is None:
                            os.environ.pop(v, None)
                        else:
                            os.environ[v] = old
                if sweep and getattr(run, "sweep_id", None) != sweep[2]:
                    tqdm.write(f"warning: W&B did not attach run {name} (k={ks[0]}) to sweep "
                               f"{sweep[2]}; it is logged in {sweep[0]}/{sweep[1]} with tag {tag!r}")
            run_id = run.id if run is not None else uuid.uuid4().hex[:8]
            metrics = _score_config(
                run, run_id, name, sweep_id, source, settings, hidden_dims, activation, kw,
                ks, n_dags, n_nodes, alpha, sample_sizes, card, val_frac, results_dir,
                n_cached, n_test, trial_csv, dag_offset, n_jobs, device, torch_threads)
            if run is not None:
                run.finish()
            results.append({"name": name, "in_degree": settings["in_degree"],
                            "sweep_id": sweep_id, "run_id": run_id, **metrics})
    return pd.DataFrame(results)


def _rename_wandb_run(entity, project, run_id, name):
    """Make the W&B server show `name`. Setting run.name inside a sweep run
    doesn't always reach the server, so rename via the public API after the
    run has finished. Display only: a failure warns and never stops the sweep.
    """
    try:
        import wandb
        api_run = wandb.Api().run(f"{entity}/{project}/{run_id}")
        if api_run.name != name:
            api_run.name = name
            api_run.update()
    except Exception as e:
        print(f"W&B rename of run {run_id} to {name!r} failed ({e}); "
              "the CSV's run_name/run_id still identify it")


def arms_from_configs(configs):
    """Rows of settings (e.g. plotting.top_config_values output) -> run_sweep arms.

    Each row needs run_name and hidden_dims ("[256, 128]" or a tuple); any of
    activation ("ReLU", "linear", ...), lr, weight_decay, dropout, patience,
    batch_size, optimizer and n_epochs present (and not missing) are passed on.
    Arm names must be unique (they label rows and history files), so a
    run_name that repeats (trial numbers restart in every sweep) gets the
    first characters of its run_id appended.
    """
    by_name = {cls.__name__: cls for cls in ACTIVATIONS.values()}
    ints = ("patience", "batch_size", "n_epochs")
    repeated = configs["run_name"].duplicated(keep=False)
    arms = []
    for i, r in configs.iterrows():
        hd = r["hidden_dims"]
        hd = tuple(ast.literal_eval(hd)) if isinstance(hd, str) else tuple(hd)
        act = r.get("activation", "ReLU")
        name = str(r["run_name"])
        if repeated.loc[i] and "run_id" in r:
            name = f"{name}_{str(r['run_id'])[:4]}"
        arm = {"name": name, "hidden_dims": hd,
               "activation": None if (not hd or act == "linear") else by_name.get(act, nn.ReLU)}
        for c in ("lr", "weight_decay", "dropout", "patience", "batch_size", "optimizer", "n_epochs"):
            if c in r and pd.notna(r[c]):
                arm[c] = int(r[c]) if c in ints else r[c]
        arms.append(arm)
    return arms


def save_chosen_configs(configs, path):
    """Save the final NN config(s) for the main experiments to JSON.

    configs : rows of plotting.top_config_values (e.g. top_values.loc[[3]]) or
              a list of config dicts. They are converted with arms_from_configs,
              stored with the activation by name, and with a note of which
              top_values rows they came from, so the choice stays fixed even
              if the indices change later. Read back with load_chosen_configs.
    """
    import datetime
    rows = list(configs.index) if isinstance(configs, pd.DataFrame) else None
    arms = arms_from_configs(configs) if isinstance(configs, pd.DataFrame) else list(configs)
    saved = []
    for a in arms:
        act = a.get("activation")
        saved.append({**{k: (v.item() if hasattr(v, "item") else v) for k, v in a.items()
                         if k not in ("activation", "hidden_dims")},
                      "hidden_dims": [int(h) for h in a.get("hidden_dims", ())],
                      "activation": None if act is None else act.__name__})
    doc = {"configs": saved,
           "chosen_from": {"top_values_rows": [int(i) for i in rows] if rows else None,
                           "saved": datetime.datetime.now().isoformat(timespec="seconds")}}
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"saved {len(saved)} config(s) to {path}: {[c['name'] for c in saved]}")
    return saved


def load_chosen_configs(path):
    """The final NN config(s) saved by save_chosen_configs, as config dicts
    (activation as an nn class, hidden_dims as a tuple) ready for ESTIMATORS."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found: run the 'Chosen configs' cell at the end of "
            "2_hyperparameter_tuning.ipynb to choose the final NN config(s)")
    with open(path) as f:
        doc = json.load(f)
    by_name = {cls.__name__: cls for cls in ACTIVATIONS.values()}
    out = []
    for c in doc["configs"]:
        act = c.get("activation")
        out.append({**c, "hidden_dims": tuple(c.get("hidden_dims", ())),
                    "activation": None if act in (None, "linear") else by_name.get(act, nn.ReLU)})
    return out
