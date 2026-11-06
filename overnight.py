"""Overnight runs, one after another, from the repo root:

    .venv/bin/python overnight.py                 # steps 1-5
    .venv/bin/python overnight.py --steps 2 4     # only some steps

1. Structured ground truth (logistic CPDs), notebook 3        -> results_structured.csv, figs/structured.png
2. Refit the 800 rows of the 11 Oct run on today's cached data -> results.csv, report figures
3. Per-node KL terms, r=2, 30-node networks                   -> results_nodekl.csv, figs/rho_full_indegree.png
4. Small serial timing run (1 worker)                          -> results_timing.csv
5. Per-node KL terms, r=4 (optional; safe to stop with Ctrl-C) -> results_nodekl.csv, figure redrawn

Settings come from notebook 3's own cells, so the runs match the notebook. A step that fails is logged and the next
step still runs; a summary at the end lists which steps failed (re-run one with --steps N: results are written after
each DAG, so it resumes where it stopped). Every file a step changes is backed up first. Ctrl-C stops the script.
--smoke runs a tiny version of every step into --out (default: the scratch folder given) without touching results.
"""
import argparse
import datetime
import json
import os
import shutil
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import pandas as pd

NB3 = "3_main_experiments.ipynb"
SETTINGS_CELLS = ["c3a0b001", "146ffd57", "96908334", "e2a0e5f1", "st0a0002"]
N_JOBS_SMALL_TABLES = 10   # r=2 runs: small tables, plenty of workers
N_JOBS_LARGE_TABLES = 2    # r=4 runs: k=8-10 tables are ~1M configurations per node, so memory limits workers
LOG = None


def log(msg):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def notebook_cells():
    return {c.get("id"): "".join(c["source"]) for c in json.load(open(NB3))["cells"]}


def run_cell(cells, cid, ns):
    """Run one notebook-3 cell in ns, without IPython magics."""
    exec(compile("\n".join(l for l in cells[cid].splitlines() if not l.lstrip().startswith("%")),
                 f"{NB3}:{cid}", "exec"), ns)


def backup(path):
    if os.path.exists(path):
        dst = f"{path}.bak-{datetime.datetime.now():%Y%m%d-%H%M%S}"
        shutil.copy2(path, dst)
        log(f"  backed up {path} -> {dst}")


def results_path(ns, name, out):
    """Absolute path of a results file: in RESULTS_DIR normally, in out for a smoke run.
    Always absolute, because run_experiments joins results_dir with results_name, and
    joining onto an absolute name leaves it unchanged."""
    return os.path.abspath(os.path.join(out or ns["RESULTS_DIR"], name))


# ── steps ───────────────────────────────────────────────────────────────────

def step1_structured(cells, ns, smoke, out):
    if smoke:
        ns.update(STRUCT_IN_DEGREES=[2, 10], STRUCT_SAMPLE_SIZES=[200, 1400], SIM_C2=1)
    ns["N_JOBS"] = N_JOBS_SMALL_TABLES
    if out:   # fresh logistic data in the smoke folder, results there too
        ns.update(RESULTS_DIR=out, FIG_DIR=out, STRUCT_CSV=os.path.join(out, ns["STRUCT_RESULTS"]))
    run_cell(cells, "st0a0003", ns)
    run_cell(cells, "st0a0004", ns)


def step2_refit_unpaired(cells, ns, smoke, out):
    path = results_path(ns, "results.csv", out)
    if out:
        shutil.copy2(os.path.join(ns["RESULTS_DIR"], "results.csv"), path)
    else:
        backup(path)
    # every column as text, so the rows that stay are rewritten exactly as they were
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    old = (df["code_version"] == "3c0ff2d+dirty") & (df["card"] == "2")
    if smoke:   # one cell only: 10 nodes, k=2, DAG 1, N=1000
        old &= (df["n_nodes"] == "10") & (df["in_degree"] == "2") & (df["dag_idx"] == "1")
    n_old, n_before = int(old.sum()), len(df)
    if smoke or n_old == 800:
        df[~old].to_csv(path, index=False)
        log(f"  removed {n_old} rows fitted on the separate 1000-row draw; refitting them on the cached data")
    elif n_old == 0:   # an earlier run already removed them: refit whatever is missing
        log(f"  the 800 rows were already removed ({n_before} rows now); refitting them on the cached data")
    else:
        raise RuntimeError(f"expected 800 rows from the 11 Oct run (or none), found {n_old}: not deleting anything")
    grid = dict(node_counts=ns["NODE_COUNTS_C2"], in_degrees=ns["IN_DEGREES_C2"], n_dags=ns["SIM_C2"],
                sample_sizes=ns["SAMPLE_SIZES_C2"])
    if smoke:
        grid = dict(node_counts=[10], in_degrees=[2], n_dags=1, sample_sizes=ns["SAMPLE_SIZES_C2"])
    ns["run_experiments"](alphas=ns["ALPHAS_C2"], estimators=ns["ESTIMATORS_C2"], card=ns["CARD_C2"],
                          val_frac=ns["VAL_FRAC_C2"], n_test=ns["N_TEST_C2"], n_epochs=ns["N_EPOCHS_C2"],
                          results_dir=ns["RESULTS_DIR"], verbose=False, results_name=path,
                          n_jobs=N_JOBS_SMALL_TABLES, device=ns["DEVICE_OVERRIDE"],
                          torch_threads=ns["TORCH_THREADS"], **grid)
    # check: every r=2 estimator has the full grid, exactly once
    res = pd.read_csv(path)
    c2 = res[res["card"] == ns["CARD_C2"]]
    keys = ["estimator", "n_nodes", "in_degree", "sample_size", "dag_idx", "alpha"]
    n_grid = len(grid["node_counts"]) * len(grid["in_degrees"]) * grid["n_dags"] * len(grid["sample_sizes"])
    counts = c2[c2["estimator"].isin([e["name"] for e in ns["ESTIMATORS_C2"]])].groupby("estimator").size()
    log(f"  rows: {len(res)} in total; r=2 rows per estimator: {counts.to_dict()}")
    if c2.duplicated(keys).any():
        raise RuntimeError("duplicate r=2 rows: check results.csv against its backup")
    if not smoke and (counts != n_grid).any():
        raise RuntimeError(f"some r=2 estimators don't have all {n_grid} configurations: re-run step 2")
    if not out:   # report figures from the repaired results
        run_cell(cells, "c3a0b0f2", ns)


def nodekl_run(cells, ns, card, estimator_names, n_jobs, smoke, out):
    """Per-node KL terms for one cardinality, one estimator at a time (each pass resumable)."""
    c2 = card == 2
    ests = ns["ESTIMATORS_C2"] if c2 else ns["ESTIMATORS"]
    grid = dict(node_counts=[30], in_degrees=ns["IN_DEGREES_C2"] if c2 else ns["IN_DEGREES"],
                n_dags=ns["SIM_C2"] if c2 else ns["SIM"],
                sample_sizes=ns["SAMPLE_SIZES_C2"] if c2 else ns["SAMPLE_SIZES"])
    if smoke:
        grid = dict(node_counts=[30], in_degrees=[2], n_dags=1, sample_sizes=[200, 1400])   # max must match the 1400-row cache
    path = results_path(ns, "results_nodekl.csv", out)
    for name in estimator_names:
        log(f"  r={card}: {name}")
        ns["run_experiments"](alphas=[0.5], estimators=[e for e in ests if e["name"] == name], card=card,
                              val_frac=ns["VAL_FRAC_C2"] if c2 else ns["VAL_FRAC"],
                              n_test=ns["N_TEST_C2"] if c2 else ns["N_TEST"],
                              n_epochs=ns["N_EPOCHS_C2"] if c2 else ns["N_EPOCHS"],
                              results_dir=ns["RESULTS_DIR"], verbose=False, results_name=path,
                              n_jobs=n_jobs, device=ns["DEVICE_OVERRIDE"],
                              torch_threads=ns["TORCH_THREADS"], **grid)


def nodekl_figure(cells, ns, out):
    ns.update(TABULAR=["mle", "dirichlet"], MAIN_NN="nn_32")
    if out:
        ns.update(RESULTS_DIR=out, FIG_DIR=out)
    run_cell(cells, "c3a0b0f4", ns)


def step3_nodekl_r2(cells, ns, smoke, out):
    nodekl_run(cells, ns, 2, ["mle", "dirichlet", "nn_32"], N_JOBS_SMALL_TABLES, smoke, out)
    nodekl_figure(cells, ns, out)


def step4_timing(cells, ns, smoke, out):
    path = results_path(ns, "results_timing.csv", out)
    TIMED = {"mle", "dirichlet", "laplace", "nn_linear", "nn_32"}
    design = dict(node_counts=[20], in_degrees=[2] if smoke else [2, 6, 10], n_dags=1, alphas=[0.5],
                  sample_sizes=[1400] if smoke else [600, 1400], results_dir=ns["RESULTS_DIR"], verbose=False,
                  results_name=path, n_jobs=1, device=ns["DEVICE_OVERRIDE"], torch_threads=ns["TORCH_THREADS"])
    for card, ests, val_frac, n_test, n_epochs in [
            (ns["CARD_C2"], ns["ESTIMATORS_C2"], ns["VAL_FRAC_C2"], ns["N_TEST_C2"], ns["N_EPOCHS_C2"]),
            (ns["CARD"], ns["ESTIMATORS"], ns["VAL_FRAC"], ns["N_TEST"], ns["N_EPOCHS"])]:
        log(f"  r={card}")
        ns["run_experiments"](estimators=[e for e in ests if e["name"] in TIMED], card=card,
                              val_frac=val_frac, n_test=n_test, n_epochs=n_epochs, **design)


def step5_nodekl_r4(cells, ns, smoke, out):
    # networks first (cheap), then the slow tabular fits; each pass is kept if you stop later
    nodekl_run(cells, ns, 4, ["nn_32", "mle", "dirichlet"], N_JOBS_LARGE_TABLES, smoke, out)
    nodekl_figure(cells, ns, out)


STEPS = {1: ("structured ground truth", step1_structured),
         2: ("refit the 800 unpaired rows", step2_refit_unpaired),
         3: ("per-node KL, r=2", step3_nodekl_r2),
         4: ("serial timing", step4_timing),
         5: ("per-node KL, r=4 (optional)", step5_nodekl_r4)}


def shutdown_workers():
    """Stop any parallel workers left behind."""
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True, kill_workers=True)
    except Exception:
        pass


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, nargs="+", default=sorted(STEPS))
    ap.add_argument("--smoke", action="store_true", help="tiny version of every step, written to --out")
    ap.add_argument("--out", default=None, help="output folder for --smoke (required with it)")
    args = ap.parse_args()
    if args.smoke and not args.out:
        ap.error("--smoke needs --out")
    out = os.path.abspath(args.out) if args.out else None
    os.makedirs("logs", exist_ok=True)
    LOG = os.path.join(out or "logs", f"overnight_{datetime.datetime.now():%Y%m%d-%H%M%S}.log")
    if out:
        os.makedirs(out, exist_ok=True)

    from IPython.display import display
    cells = notebook_cells()
    log(f"overnight runs: steps {args.steps}{' (smoke)' if args.smoke else ''}; log {LOG}")
    t_all = time.time()
    status = {}
    try:
        for step in args.steps:
            name, fn = STEPS[step]
            log(f"step {step}: {name} - started")
            t = time.time()
            try:
                ns = {"display": display}
                for cid in SETTINGS_CELLS:
                    run_cell(cells, cid, ns)
                fn(cells, ns, args.smoke, out)
            except KeyboardInterrupt:
                status[step] = "stopped"
                log(f"step {step}: stopped by you after {(time.time() - t) / 60:.0f} min; finished DAGs are "
                    f"saved, so --steps {step} resumes it. Later steps not run.")
                break
            except Exception:
                status[step] = "FAILED"
                log(f"step {step}: FAILED after {(time.time() - t) / 60:.0f} min (continuing with the next step)\n"
                    f"{traceback.format_exc()}")
                shutdown_workers()   # don't let a failed parallel step's workers hold memory
            else:
                status[step] = "done"
                log(f"step {step}: done in {(time.time() - t) / 60:.0f} min")
    finally:
        shutdown_workers()
        log("summary: " + ", ".join(f"step {k} {v}" for k, v in status.items())
            + f"; total {(time.time() - t_all) / 3600:.1f} h")
    return 1 if "FAILED" in status.values() else 0


if __name__ == "__main__":
    sys.exit(main())
