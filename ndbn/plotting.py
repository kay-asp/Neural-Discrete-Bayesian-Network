"""
RQ1 plotting — single-variable-at-a-time design (no density parameter).

Every plot fixes all experimental variables except x and hue.
Change the DEFAULT_* values (e.g. `plotting.DEFAULT_ALPHA = 0.1`) to control
which slice you're viewing.
"""

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Defaults: the "held constant" values for each variable ───────────────────

DEFAULT_SAMPLE_SIZE  = 1000
DEFAULT_ALPHA        = 0.5
DEFAULT_N_NODES      = 20
DEFAULT_IN_DEGREE    = 4
DEFAULT_ESTIMATOR    ="mle"
DEFAULT_CARD         = 2
# estimators plotted (e.g. as hue="estimator"): None = every one in df, a list of names, or
# the path of a chosen_nn json (save_chosen_configs): its configs plus BASELINE_ESTIMATORS
DEFAULT_ESTIMATORS   = None
BASELINE_ESTIMATORS  = ("mle", "nn_linear")


def _estimator_names(estimators):
    """Estimator names to plot, in order: None (all), a list of names, or the
    path of a chosen_nn json -> BASELINE_ESTIMATORS + its configs' names."""
    if estimators is None:
        return None
    if isinstance(estimators, str):
        import json
        with open(estimators) as f:
            chosen = [c["name"] for c in json.load(f)["configs"]]
        return list(dict.fromkeys([*BASELINE_ESTIMATORS, *chosen]))
    return list(estimators)

# ── Palette ──────────────────────────────────────────────────────────────────

_PALETTE = ["#2563eb", "#e11d48", "#16a34a", "#d97706",
            "#7c3aed", "#0891b2", "#be185d", "#4d7c0f"]


def _colors(n):
    if n <= len(_PALETTE):
        return _PALETTE[:n]
    cmap = plt.get_cmap("viridis")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


# every series also gets its own marker and line style, so plots read in black and white
_MARKERS    = ["o", "s", "^", "D", "v", "P", "*", "X"]
_LINESTYLES = ["-", "--", "-.", ":"]


def _styles(n):
    """(marker, linestyle) for n series; no two series share both (up to 32 series).
    Each pass through the markers shifts the line styles by one."""
    return [(_MARKERS[i % len(_MARKERS)],
             _LINESTYLES[(i + i // len(_MARKERS)) % len(_LINESTYLES)]) for i in range(n)]


# ── Core filter helper ──────────────────────────────────────────────────────

def _filter(df, x, hue, alpha=None, sample_size=None,
            n_nodes=None, in_degree=None, estimator=None, card=None):
    """Filter df to fixed values for every variable that isn't x or hue.

    A variable whose column df doesn't have (e.g. alpha in real-data results)
    is not filtered on.
    """
    sub = df.copy()
    skip = {x, hue} | {c for c in ("sample_size", "alpha", "n_nodes", "in_degree", "card")
                       if c not in df.columns}

    if "sample_size" not in skip and sample_size is not None:
        sub = sub[sub["sample_size"] == sample_size]

    if "alpha" not in skip and alpha is not None:
        sub = sub[sub["alpha"] == alpha]

    if "n_nodes" not in skip and n_nodes is not None:
        sub = sub[sub["n_nodes"] == n_nodes]

    if "in_degree" not in skip and in_degree is not None:
        sub = sub[sub["in_degree"] == in_degree]

    if "estimator" not in skip and estimator is not None and "estimator" in sub.columns:
        sub = sub[sub["estimator"] == estimator]

    # rows from before card was recorded were all card=2
    if card is not None or "card" in skip:
        sub["card"] = sub["card"].fillna(2) if "card" in sub else 2
    if "card" not in skip and card is not None:
        sub = sub[sub["card"] == card]

    return sub


# ── Main plotting function ──────────────────────────────────────────────────

def plot_grouped_curves(
    df, x, hue, y="kl",
    # fixed-variable overrides (None → use DEFAULT_*)
    alpha=None, sample_size=None,
    n_nodes=None, in_degree=None, estimator=None, card=None, estimators=None,
    # plot styling
    logy=False, err=None, min_obs=3,
    xlabel=None, ylabel=None, title=None, hue_label=None,
    out_path=None, ax=None, figsize=(7, 5),
):
    """Mean y vs x, one curve per hue value, all other variables held fixed.

    estimators : which estimators to plot (None -> DEFAULT_ESTIMATORS): a list
                 of names, or the path of a chosen_nn json (its configs plus
                 BASELINE_ESTIMATORS); with hue="estimator" the curves follow
                 that order. Names not in df are reported.
    """
    if alpha is None:        alpha = DEFAULT_ALPHA
    if sample_size is None:  sample_size = DEFAULT_SAMPLE_SIZE
    if n_nodes is None:      n_nodes = DEFAULT_N_NODES
    if in_degree is None:    in_degree = DEFAULT_IN_DEGREE
    if estimator is None:    estimator = DEFAULT_ESTIMATOR
    if card is None:         card = DEFAULT_CARD
    if estimators is None:   estimators = DEFAULT_ESTIMATORS

    names = _estimator_names(estimators)
    if names is not None and "estimator" in df.columns:
        missing = [e for e in names if e not in set(df["estimator"])]
        if missing:
            print(f"WARNING: estimators not in the results: {missing}")
        df = df[df["estimator"].isin(names)]

    sub = _filter(df, x, hue,
                  alpha=alpha, sample_size=sample_size,
                  n_nodes=n_nodes, in_degree=in_degree, estimator=estimator,
                  card=card)

    if sub.empty:
        print(f"WARNING: no data after filtering for {x} vs {hue}. "
              f"Check your default values.")
        return None

    # Build subtitle showing what's held constant (only columns df has)
    skip = {x, hue} | {c for c in ("sample_size", "alpha", "n_nodes", "in_degree",
                                   "estimator", "card") if c not in df.columns}
    held = []
    if "sample_size" not in skip: held.append(f"N={sample_size}")
    if "alpha"       not in skip: held.append(f"α={alpha}")
    if "n_nodes"     not in skip: held.append(f"nodes={n_nodes}")
    if "in_degree"   not in skip: held.append(f"k={in_degree}")
    if "estimator"   not in skip: held.append(f"est={estimator}")
    if "card"        not in skip: held.append(f"card={card}")
    held_str = ",  ".join(held)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)

    hue_vals = sorted(sub[hue].dropna().unique())
    if hue == "estimator" and names is not None:     # the chosen order (baselines, then best first)
        hue_vals = [e for e in names if e in set(hue_vals)]
    colors = _colors(len(hue_vals))

    for hv, c, (m, ls) in zip(hue_vals, colors, _styles(len(hue_vals))):
        s = sub[sub[hue] == hv]
        g = s.groupby(x)[y].agg(["mean", "std", "sem", "count"]).reset_index()
        g = g[g["count"] >= min_obs]
        if g.empty:
            continue
        label = f"{hue_label or hue} = {hv}"
        if err in ("std", "sem"):
            ax.errorbar(g[x], g["mean"], yerr=g[err].fillna(0.0), marker=m, ls=ls,
                        lw=2, color=c, label=label, capsize=3, zorder=3)
        else:
            ax.plot(g[x], g["mean"], marker=m, ls=ls, lw=2, color=c,
                    label=label, zorder=3)

    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel or x, fontsize=12)
    ax.set_ylabel(ylabel or y, fontsize=12)
    if title:
        ax.set_title(f"{title}\n({held_str})", fontsize=12)
    else:
        ax.set_title(held_str, fontsize=11, color="grey")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    if own_fig:
        fig.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=150)
        return fig, ax
    return ax


# ── Memory plots (unchanged) ────────────────────────────────────────────────

def plot_cardinality_wall(cards=(2, 3, 4, 5), max_k=12,
                          wall_bytes=None, out_path=None,
                          ax=None, figsize=(7, 5)):
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)

    ks = np.arange(0, max_k + 1)
    for c, col, (m, ls) in zip(cards, _colors(len(cards)), _styles(len(cards))):
        ax.plot(ks, [8 * c ** (k + 1) for k in ks], marker=m, ls=ls, lw=2,
                color=col, label=f"card = {c}")

    if wall_bytes:
        ax.axhline(wall_bytes, color="black", ls=":", lw=1.5,
                   label=f"{wall_bytes/1024**3:.0f} GB wall")

    ax.set_yscale("log")
    ax.set_xlabel("Parent in-degree $k$", fontsize=12)
    ax.set_ylabel("CPT memory (bytes, log)", fontsize=12)
    ax.set_title("Storage cost by cardinality: reaching the memory wall",
                 fontsize=12)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    if own_fig:
        fig.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=150)
        return fig, ax
    return ax


# ── Hyperparameter sweep (hparam_trials_k*.csv) ─────────────────────────────

_HPARAM_COLS = ["hidden_dims", "activation", "lr", "weight_decay", "dropout",
                "patience", "batch_size", "optimizer", "n_epochs"]

# values for settings that older trial rows didn't record yet
TRIAL_DEFAULTS = {"card": 2, "val_frac": 0.2, "n_test": 10000, "activation": "ReLU"}


def _norm_setting(c, v):
    """One setting as canonical text for config_id: numbers to 6 significant
    figures (3e-4 == 0.00030000000000000003, 64 == 64.0), hidden_dims as
    "[256, 128]" (from a tuple, list or string), missing -> TRIAL_DEFAULTS or ""."""
    import ast
    if v is None or (isinstance(v, float) and np.isnan(v)):
        v = TRIAL_DEFAULTS.get(c)
        if v is None:
            return ""
    if c == "hidden_dims":
        hd = ast.literal_eval(v) if isinstance(v, str) else v
        return str([int(h) for h in hd])
    if isinstance(v, (bool, np.bool_)):
        return str(v)
    try:
        return f"{float(v):.6g}"
    except (TypeError, ValueError):
        return str(v)


def config_id(settings):
    """Short id (8 hex characters) of a config's exact settings (_HPARAM_COLS,
    from a dict or a trial row): identical settings always give the same id,
    at any in-degree or sample size, however often they're run."""
    import hashlib
    key = "|".join(f"{c}={_norm_setting(c, settings.get(c))}" for c in _HPARAM_COLS)
    return hashlib.sha1(key.encode()).hexdigest()[:8]


def add_config_ids(df):
    """Fill the config_id column of trial rows wherever it is missing."""
    if "config_id" not in df:
        df["config_id"] = np.nan
    df["config_id"] = df["config_id"].astype(object)    # all-NaN reads as float
    missing = df["config_id"].isna()
    if missing.any():
        df.loc[missing, "config_id"] = [config_id(r) for r in df.loc[missing].to_dict("records")]
    return df


def rewrite_csv(path):
    """One-off: add a config_id column (after run_name) to every row of a trial
    CSV such as hparam_trials.csv, keeping a copy of the original at <path>.bak."""
    import shutil
    shutil.copyfile(path, path + ".bak")
    df = add_config_ids(pd.read_csv(path))
    cols = [c for c in df.columns if c != "config_id"]
    at = cols.index("run_name") + 1 if "run_name" in cols else 0
    df = df[cols[:at] + ["config_id"] + cols[at:]]
    df.to_csv(path, index=False)
    print(f"{path}: {len(df)} rows, {df['run_id'].nunique()} runs, "
          f"{df['config_id'].nunique()} distinct configs (original kept at {path}.bak)")
    return df


def load_trials(results_dir):
    """All hyperparameter-sweep trial rows in results_dir: hparam_trials.csv
    (every sweep) plus any older hparam_trials_*.csv files, concatenated, with
    a config_id for every row (see config_id; filled in for rows without one).
    A run_id shared by several configs (W&B handed one run to several
    candidate runs) is split: those rows get run_id "<run_id>_<config_id[:4]>",
    so every trial is one config."""
    from glob import glob
    import os
    files = sorted(glob(os.path.join(results_dir, "hparam_trials*.csv")))
    if not files:
        raise FileNotFoundError(f"no hparam_trials*.csv in {results_dir}")
    t = add_config_ids(pd.concat([pd.read_csv(f) for f in files], ignore_index=True))
    shared = t.groupby("run_id")["config_id"].transform("nunique") > 1
    t.loc[shared, "run_id"] = t.loc[shared, "run_id"] + "_" + t.loc[shared, "config_id"].str[:4]
    return t


def _show(df):
    """Display a table nicely in a notebook, or print it elsewhere."""
    try:
        from IPython.display import display
        display(df)
    except ImportError:
        print(df.to_string())


def _round_sig(x, sig=1):
    """Round to `sig` significant figures (3.2e-4 -> 3e-4); non-numbers unchanged."""
    try:
        return float(f"{float(x):.{sig - 1}e}")
    except (TypeError, ValueError):
        return x


_ROUNDED = ("lr", "weight_decay")


def _snap(x, mantissas=(1, 5)):
    """Snap a positive number to the nearest m * 10^e (m in mantissas), in log
    space: with (1, 5), 0.006 -> 0.005, 0.003 -> 0.005, 0.002 -> 0.001,
    0.008 -> 0.01. Zero, negatives and non-numbers are returned unchanged."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    if not np.isfinite(x) or x <= 0:
        return x
    e0 = int(np.floor(np.log10(x)))
    candidates = [float(f"{m}e{e}") for e in (e0 - 1, e0, e0 + 1) for m in mantissas]
    return min(candidates, key=lambda c: abs(np.log10(c) - np.log10(x)))


def _for_display(df):
    """Round for printing: 4 decimals, but lr / weight_decay to 1 significant
    figure (4 decimals would turn a weight decay of 2e-5 into 0)."""
    out = df.round(4)
    for c in _ROUNDED:
        if c in df:
            out[c] = df[c].map(_round_sig)
    return out


def _value_range(top, c, min_frac=0.2):
    """The values of column c across the rows of `top` as text, each with the
    number of rows that have it, ignoring values in fewer than min_frac of the
    rows: "32 (5), 64 (3), 128 (3)". Numbers are in increasing order, other
    values (hidden_dims, ...) most common first; a value every row shares is
    shown without a count. "–" if no value is in min_frac of the rows.
    lr / weight_decay are rounded to 1 significant figure first ("3e-4")."""
    vals = top[c].map(_round_sig) if c in _ROUNDED else top[c]
    counts = vals.astype(str).value_counts()
    common = [v for v in dict.fromkeys(vals) if counts[str(v)] >= min_frac * len(vals)]
    if not common:
        return "–"
    try:
        common.sort(key=float)
        fmt = (lambda v: f"{int(f'{float(v):.0e}'.split('e')[0])}e{int(f'{float(v):.0e}'.split('e')[1])}"
               if c in _ROUNDED and float(v) != 0 else f"{float(v):g}")
    except (TypeError, ValueError):
        common.sort(key=lambda v: -counts[str(v)])
        fmt = str
    if len(common) == 1 and counts[str(common[0])] == len(vals):
        return fmt(common[0])
    return ", ".join(f"{fmt(v)} ({counts[str(v)]})" for v in common)


def _group_trials(s, cols, m="mean_kl"):
    """Average trials (rows of s, indexed by run_id) whose settings `cols` are
    identical (after summarise_trials' rounding / snapping / fixed values):
    one row per distinct config. mean_kl / mean_val_loss / extras and kl_se
    are means over every run in the group, and so is the per-DAG val loss
    (_val_by_dag, a Series indexed by dag_idx, if s has it); n_trials counts its distinct
    configs (config_ids: exact settings before tidying), so an exact re-run of
    a config (e.g. a candidate run) adds to the means but not to n_trials.
    config_id / run_name / run_id are the group's best trial (lowest `m`), run_names / config_ids list
    every trial / config in it. kl_se is the mean of the trials' kl_se, not sqrt(sum se^2)/n:
    every trial is evaluated on the same DAGs and test rows, so their errors
    are strongly correlated and averaging removes far less noise than
    independent errors would (the true SE of the average lies between
    se / sqrt(n) and se). Using the mean keeps grouped rows' within-n_se
    window comparable to single trials'."""
    s = s.reset_index()
    rows = []
    for _, g in s.groupby(cols, dropna=False, sort=False):
        best = g.loc[g[m].idxmin()]
        row = {"config_id": best["config_id"],      # the best trial's config
               "run_id": best["run_id"], "run_name": best["run_name"],
               "run_names": tuple(g.sort_values(m)["run_name"]),   # every trial merged
               "config_ids": tuple(dict.fromkeys(g.sort_values(m)["config_id"])),
               # name of each config's best trial, for labelling a config picked by id
               "id_names": dict(g.sort_values(m, ascending=False)
                                [["config_id", "run_name"]].to_numpy()),
               "mean_kl": g["mean_kl"].mean(),
               "kl_se": g["kl_se"].mean(),     # errors are correlated; see docstring
               "mean_val_loss": g["mean_val_loss"].mean(),
               "n_dags": best["n_dags"], "n_trials": g["config_id"].nunique(),
               **{c: best[c] for c in cols}}
        if "_val_by_dag" in g:
            row["_val_by_dag"] = pd.concat(list(g["_val_by_dag"]), axis=1).mean(axis=1)
        for c in ["stored_params", "best_epoch_mean", "fit_time_s"]:
            if c in g:
                row[c] = g[c].mean()
        if "sweep_id" in g:
            row["sweep_id"] = ", ".join(map(str, dict.fromkeys(g["sweep_id"].dropna()))) or np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("run_id")


def _paired_gap(vals, best):
    """Paired comparison of configs evaluated on the same DAGs (and validation
    rows) against the best one: vals is a DataFrame (configs x dag_idx) of
    per-DAG losses, best its row label. Returns (gap, se): each config's mean
    per-DAG difference from the best, and the standard error of that mean,
    sd(differences, ddof=1) / sqrt(n_dags), i.e. a paired t-test. Pairing
    removes what all configs share on a DAG (its entropy, which makes the raw
    spread of the loss across DAGs far larger than any gap between configs);
    the DAGs are independent networks, so their differences are independent.
    DAGs missing for a config are left out of its comparison."""
    diff = vals.sub(vals.loc[best], axis=1)
    n = diff.notna().sum(axis=1)
    return diff.mean(axis=1), (diff.std(axis=1, ddof=1) / np.sqrt(n)).fillna(0.0)


def summarise_trials(trials, n_se=1.0, fixed=None, snap=(), rel_tol=0.0, max_params=None,
                     metric="kl", show=True, **filters):
    """Per (in_degree, sample_size): the trials within n_se kl_se (or rel_tol) of the best.

    metric  : "kl" (default) selects on mean_kl as described below.
              "val_loss" selects on mean_val_loss (validation only, no test
              data) instead: the best is the lowest mean_val_loss, and a config
              is kept if its mean per-DAG excess over the best, val_gap, is
              within n_se of its paired standard error over the DAGs, val_se
              (see _paired_gap): every config is fitted and validated on the
              same DAGs and rows, so each DAG's own entropy cancels in the
              difference. (The raw spread of the val loss across DAGs is
              dominated by that entropy and would keep nearly every config.)
              rel_tol is not used for the val loss: a percentage of it
              (entropy + KL) means nothing. The tables get val_gap and val_se
              columns, and bold mean_kl where it is within the KL threshold
              below (n_se kl_se or rel_tol, whichever is looser), for
              reference (the KL tables bold mean_val_loss within the same
              window in nats).

    trials  : the per-cell trial CSV(s) (one row per trial x dag x sample size).
    filters : any column = value, e.g. card=2, alpha=0.5, sweep_id="abc123".
              Rows missing a filtered setting (older CSVs) use TRIAL_DEFAULTS,
              or are dropped if it has no default.
    For each (in_degree, sample_size), every trial gets mean_kl over its DAGs
    and kl_se = sqrt(sum of per-cell kl_se^2) / n_cells, the standard error of
    that mean from the test-row Monte Carlo error. Trials with fewer DAGs than
    the most complete one in that pair are dropped. The best trial is the
    lowest mean_kl; a trial is kept if mean_kl <= best mean_kl + n_se * best kl_se
    (statistically indistinguishable) or mean_kl <= best mean_kl * (1 + rel_tol)
    (practically negligible, e.g. rel_tol=0.02 for 2%), whichever is looser.
    max_params : {in-degree: limit}, e.g. {10: 100_000}: remove configs with
              `limit` or more stored_params (total parameters over a DAG's
              nodes, mean over the tuning DAGs) at that in-degree from every
              table (as in candidate_table). A config's total at k=10 is looked
              up by its hidden_dims (and card) from the trials run at k=10;
              configs whose architecture never ran there are dropped. The cap
              is applied first: the best (and so the threshold) is the best
              config under the cap, and the overall best is printed for
              reference. A table with no config under the cap is left out.
    Settings are tidied before trials are compared: lr / weight_decay are
    rounded to 1 significant figure, unless listed in
    snap    : (column, mantissas) pairs, e.g. (("lr", (1, 3)),) puts lr on
              ..., 1e-4, 3e-4, 1e-3, ... (nearest in log space, see _snap);
    fixed   : settings with a negligible effect, as {column: value}, e.g.
              {"weight_decay": 0.01}: every trial gets that value (it's also
              what the chosen configs run with).
    Trials whose settings are then identical are averaged into one row
    automatically (see _group_trials): mean_kl, kl_se and the other metrics
    are means (kl_se not shrunk by sqrt(n): the trials share DAGs and test
    rows, so their errors are correlated), n_trials counts the group and
    run_name is its best trial. The threshold is applied to these rows.
    Prints one table per pair when show=True (the other metric in bold where
    it is within the same window of its own best). Returns (tables, top_values):
    tables     : {(in_degree, sample_size): DataFrame}, sorted by mean_val_loss
                 (lowest first) with metric="val_loss", else by n_trials (most
                 first), then mean_kl, with the selection metric's columns
                 first (mean_val_loss, val_gap, val_se, or mean_kl,
                 kl_se). Printed tables are indexed by
                 config_id (the best trial's), round lr / weight_decay to 1
                 significant figure and hide run_id, run_name, sweep_id, n_dags, fixed
                 settings and settings that are constant across all trials;
                 the returned tables keep every column. Use
                 shared_settings_table(tables) for the settings each table's
                 rows have in common.
    top_values : every config in any table once (merged by settings), see
                 _merge_top_configs: what run_configs / save_chosen_configs
                 take (e.g. experiments.get_configs(top_values, config_ids)).
    """
    if metric not in ("kl", "val_loss"):
        raise ValueError(f"metric must be 'kl' or 'val_loss', not {metric!r}")
    m, other = ("mean_kl", "mean_val_loss") if metric == "kl" else ("mean_val_loss", "mean_kl")
    t = add_config_ids(trials.copy())
    for col, val in filters.items():
        if col not in t:
            t[col] = np.nan
        if col in TRIAL_DEFAULTS:
            t[col] = t[col].fillna(TRIAL_DEFAULTS[col])
        t = t[t[col] == val]
    if "run_name" not in t:
        t["run_name"] = np.nan
    t["run_name"] = t["run_name"].fillna(t["run_id"])   # pre-naming trials

    fixed, grids = dict(fixed or {}), dict(snap)
    hp = [c for c in _HPARAM_COLS if c in t]
    hp += [c for c in fixed if c not in hp]
    extra = [c for c in ["stored_params", "best_epoch_mean", "fit_time_s"] if c in t]
    # display only: identifiers, fixed settings and settings that never vary
    hidden = ["run_id", "run_names", "config_ids", "id_names", "n_dags", *fixed] + [c for c in [*hp, "sweep_id"]
                         if c in t and t[c].nunique(dropna=False) <= 1]
    # total params depend only on the architecture (and card) at a given in-degree, over
    # the same tuning DAGs: mean over those DAGs per architecture, at each capped in-degree
    arch = [c for c in ("hidden_dims", "card") if c in t]
    caps = {}
    for k, cap in (max_params or {}).items():
        at_k = t[t["in_degree"] == k] if "stored_params" in t else t.iloc[:0]
        if at_k.empty:
            raise KeyError(f"no stored_params recorded for in-degree {k}")
        caps[k] = (at_k.groupby(arch)["stored_params"].mean().to_dict(), cap)
    if show and caps:
        print("configs with " + " or ".join(f"{cap:,} or more total parameters at in-degree {k}"
                                             for k, cap in (max_params or {}).items())
              + " are removed from every table")
    if show and metric == "val_loss" and rel_tol:
        print(f"configs are kept within {n_se:g} paired SE of the best val loss (rel_tol is not "
              f"used for it: a % of entropy + KL means nothing); mean_kl is bold within "
              f"{n_se:g} kl_se or {rel_tol:.0%} of the best KL")
    tables = {}
    for (k, n), g in t.groupby(["in_degree", "sample_size"]):
        s = g.groupby("run_id").agg(
            run_name=("run_name", "first"),
            config_id=("config_id", "first"),
            mean_kl=("kl", "mean"),
            kl_se=("kl_se", lambda x: np.sqrt((x ** 2).sum()) / len(x)),
            mean_val_loss=("val_loss", "mean"),
            n_dags=("dag_idx", "nunique"),
            **{c: (c, "first") for c in hp if c in g},
            **{c: (c, "mean") for c in extra},
            **({"sweep_id": ("sweep_id", "first")} if "sweep_id" in g else {}))
        s = s[s["n_dags"] == s["n_dags"].max()]
        if s.empty:
            continue
        # per-DAG val loss of each run, for the paired comparison (metric="val_loss")
        vbd = g.pivot_table(index="run_id", columns="dag_idx", values="val_loss", aggfunc="mean")
        s["_val_by_dag"] = [vbd.loc[rid] for rid in s.index]
        # runs under the size cap; the others still set the best (and so the threshold)
        under = set(s.index)
        if caps:
            a = g.groupby("run_id")[arch].first().loc[s.index]
            keys = [r[0] if len(arch) == 1 else tuple(r) for r in a.itertuples(index=False)]
            # architecture never run at that in-degree -> no params known -> over the cap
            under = {rid for rid, key in zip(s.index, keys)
                     if all(params_at_k.get(key, np.inf) < cap for params_at_k, cap in caps.values())}
        # tidy settings, then average trials that have become identical
        for c in hp:
            if c in fixed:
                s[c] = fixed[c]
            elif c in grids:
                s[c] = s[c].map(lambda x, m=grids[c]: _snap(x, m))
            elif c in _ROUNDED:
                s[c] = s[c].map(_round_sig)
        s = _group_trials(s, hp, m)    # a grouped row's runs share hidden_dims, so its
                                       # best run tells whether it is under the cap
        s = s.sort_values(m)
        best_all = s.iloc[0]           # best on the metric over ALL configs, for reference
        s = s[s.index.isin(under)].copy()
        name = "KL" if metric == "kl" else "val loss"
        if s.empty:
            if show:
                print(f"in-degree {k}, N = {n}: no config is under the size cap "
                      f"(best {name} over all configs {best_all[m]:.4f})")
            continue
        # the thresholds are relative to the best configs under the size cap.
        # KL window in nats, from the best-KL config: n_se of its kl_se or rel_tol of its
        # KL, whichever is looser; selects for KL, and bolds val loss in the KL tables
        # (val loss = entropy, the same for all configs, + KL)
        best_kl = s.loc[s["mean_kl"].idxmin()]
        window = max(n_se * best_kl["kl_se"], rel_tol * best_kl["mean_kl"])
        limits = {c: s[c].min() + window for c in ("mean_kl", "mean_val_loss")}
        best = s.iloc[0]               # best on the metric under the cap
        if metric == "val_loss":
            # paired over the DAGs against the best config
            vals = pd.DataFrame(list(s["_val_by_dag"]), index=s.index)
            s["val_gap"], s["val_se"] = _paired_gap(vals, s.index[0])
            keep = s["val_gap"] <= n_se * s["val_se"]
        else:
            keep = s[m] <= limits[m]
        s = s.drop(columns="_val_by_dag")
        # val loss: lowest first; KL: most-merged configs first, then KL
        by, asc = ([m], [True]) if metric == "val_loss" else (["n_trials", m], [False, True])
        top = (s[keep].reset_index()
               .sort_values(by, ascending=asc, kind="stable")
               .reset_index(drop=True))
        # the selection metric's columns first
        lead = [m, "val_gap", "val_se"] if metric == "val_loss" else [m, "kl_se"]
        top = top[[c for c in lead if c in top] + [c for c in top if c not in lead]]
        head = (f"in-degree {k}, N = {n}: best {name} {best[m]:.4f}"
                + (f" ± {best['kl_se']:.4f}" if metric == "kl"
                   else f" (paired over {int(best['n_dags'])} DAGs)"))
        top.attrs["hidden"] = hidden        # for shared_settings_table
        top.attrs["n_se"] = n_se            # threshold, reported downstream
        top.attrs["rel_tol"] = rel_tol
        top.attrs["max_params"] = dict(max_params or {})
        top.attrs["metric"] = metric
        tables[(k, n)] = top
        if show:
            grouped = [f"{c} fixed at {v:g}" for c, v in fixed.items()]
            grouped += [f"{c} snapped to {'-'.join(map(str, g_))} grid" for c, g_ in snap]
            order = ("mean_val_loss (lowest first)" if metric == "val_loss"
                     else "n_trials (most first), then mean_kl (lowest first)")
            print(head + (f" under the size cap (best over all configs {best_all[m]:.4f})"
                          if caps else ""))
            print(f"  {len(top)} of {len(s)} configs {_threshold(tables)}"
                  + (f" ({', '.join(grouped)})" if grouped else ""))
            print(f"  sorted by {order}")
            # config_id first, no run_name, sweep_id or row number
            disp = _for_display(top.drop(columns=[c for c in [*hidden, "run_name", "sweep_id"]
                                                  if c in top])
                                .set_index("config_id"))
            # bold the other metric where it is within its own threshold (same window in
            # nats, from its own best over all configs), to compare the two selections
            ok = (top[other] <= limits[other]).to_numpy()
            try:
                fmt = {c: "{:.4f}" for c in disp.select_dtypes("float").columns}
                fmt.update({c: "{:g}" for c in _ROUNDED if c in disp})
                fmt.update({c: "{:.0f}" for c in ("stored_params",) if c in disp})
                _show(disp.style.format(fmt).apply(
                    lambda col, ok=ok: np.where(ok, "font-weight: bold", ""),  # bind now:
                    # the styler runs later, after `ok` has moved on to the next table
                    subset=[other]))
            except ImportError:                          # no jinja2: mark with * instead
                disp = disp.astype({other: str})
                disp.loc[ok, other] += " *"
                _show(disp)
    return tables, _merge_top_configs(tables)


def _threshold(tables):
    """'within N kl_se' (or rel_tol, params filter) for the tables' threshold
    (as set in summarise_trials)."""
    a = next(iter(tables.values())).attrs if tables else {}
    if a.get("metric") == "val_loss":
        text = f"within {a.get('n_se', 1.0):g} paired SE (over DAGs) of the best val loss"
    else:
        text = f"within {a.get('n_se', 1.0):g} kl_se"
        if a.get("rel_tol"):
            text += f" or {a['rel_tol']:.0%}"
    if a.get("max_params"):
        text += ", total params " + ", ".join(f"k{k} < {v:,}" for k, v in a["max_params"].items())
    return text


def shared_settings_table(tables, min_frac=0.2, show=True):
    """One row per summarise_trials table: the values of each setting over its
    top configs, with counts.

    Columns: in_degree, sample_size, n_top, then one per setting. A cell lists
    the values of that setting among the table's top configs with how many
    configs have each, "32 (5), 64 (3), 128 (3)", ignoring values in fewer
    than min_frac of them (default 1/5, so one-off outliers are left out; the
    counts then add up to less than n_top). A value all configs share is shown
    without a count; "–" if no value is in min_frac of the configs. Numbers
    are in increasing order, hidden_dims etc. most common first; lr /
    weight_decay are rounded to 1 significant figure. Settings that never vary
    across all trials are left out. Read down a column to compare the tables.
    """
    if not tables:
        return pd.DataFrame()
    hidden = set(next(iter(tables.values())).attrs.get("hidden", []))
    cols = [c for c in _HPARAM_COLS
            if c not in hidden and all(c in top for top in tables.values())]
    rows = [{"in_degree": k, "sample_size": n, "n_top": len(top),
             **{c: _value_range(top, c, min_frac) for c in cols}}
            for (k, n), top in tables.items()]
    out = pd.DataFrame(rows)
    if show:
        print(f"values of each setting among the top configs ({_threshold(tables)}) of each "
              f"table, (count) = number of configs with it, ignoring values in fewer than "
              f"{min_frac:.0%} of them (– = no such value; n_top = 1 means a single config)")
        _show(out)
    return out


def common_configs(tables, match_on=("hidden_dims", "dropout", "patience", "batch_size"),
                   round_cols=("lr", "weight_decay"), min_tables=2, show=True):
    """Configs in the top (the threshold set in summarise_trials) of
    more than one summarise_trials table.

    tables     : the dict returned by summarise_trials.
    match_on   : setting columns that define "the same config". Continuous
                 sampled values never repeat exactly, so the default uses the
                 discrete settings; add e.g. "lr" to also match on it.
    round_cols : columns rounded to 1 significant figure before matching, if
                 they are in match_on (3.2e-4 and 2.8e-4 both become 3e-4).
    Returns one row per config in >= min_tables tables: its settings,
    n_tables, top_in (which in-degree / sample size tables), n_trials and
    config_ids (the config_id each matched table row is indexed by, as in
    the summarise_trials tables; for experiments.get_configs).
    Mean KL is not compared across tables: its scale differs with k and N.
    """
    match_on = list(match_on)
    rows = []
    for (k, n), top in tables.items():
        for _, r in top.iterrows():
            key = {c: (_round_sig(r[c]) if c in round_cols else r[c]) for c in match_on}
            rows.append({**key, "table": f"k{k} N{n}", "run_id": r["run_id"],
                         "config_id": r["config_id"]})
    if not rows:
        return pd.DataFrame(columns=[*match_on, "n_tables", "top_in", "n_trials", "config_ids"])
    df = pd.DataFrame(rows)
    out = (df.groupby(match_on, dropna=False)
           .agg(n_tables=("table", "nunique"),
                top_in=("table", lambda x: ", ".join(dict.fromkeys(x))),
                n_trials=("run_id", "nunique"),
                config_ids=("config_id", lambda x: ", ".join(dict.fromkeys(x))))
           .reset_index())
    out = (out[out["n_tables"] >= min_tables]
           .sort_values(["n_tables", "n_trials"], ascending=False).reset_index(drop=True))
    if show:
        print(f"configs (matched on {', '.join(match_on)}) in the top "
              f"({_threshold(tables)}) of >= {min_tables} of {len(tables)} tables: {len(out)}")
        _show(out)
    return out


def _merge_top_configs(tables):
    """Every config in the summarise_trials tables once, with its settings
    (already rounded / snapped / fixed there), as summarise_trials' top_values.

    Columns: run_name / run_id of the merged row's best trial (lowest
    selection metric relative to its table's best; used to name the arm), in_tables
    (every in-degree / sample size table it is top in), n_tables, n_trials
    (distinct configs merged into the row, over all its tables: exact re-runs
    count once), then every setting, then run_names / config_ids (every
    trial / config merged into the row, so a config can be found by any of
    its trials' names or its config_id even when the row's run_name label
    changes). A row is
    what's needed to run that config again (see experiments.arms_from_configs).
    Sorted by n_tables, then n_trials (most first).
    """
    rows = {}
    for (k, n), top in tables.items():
        cols = [c for c in _HPARAM_COLS if c in top]
        val = top.attrs.get("metric") == "val_loss"
        m = "mean_val_loss" if val else "mean_kl"
        best_m = top[m].min()
        for _, r in top.iterrows():
            settings = {c: r[c] for c in cols}
            key = tuple(str(v) for v in settings.values())
            # val loss: excess in nats (a ratio of entropy + KL means nothing)
            regret = r[m] - best_m if val else r[m] / best_m - 1
            entry = rows.setdefault(key, {"run_id": r["run_id"], "run_name": r["run_name"],
                                          "_regret": regret, "tables": [],
                                          **settings, "run_names": {}, "config_ids": {},
                                          "id_names": {}})
            if regret < entry["_regret"]:      # label the row with its best trial
                entry.update(run_id=r["run_id"], run_name=r["run_name"], _regret=regret)
            entry["run_names"].update(dict.fromkeys(r.get("run_names", (r["run_name"],))))
            entry["config_ids"].update(dict.fromkeys(r.get("config_ids", ())))
            entry["id_names"] = {**r.get("id_names", {}), **entry["id_names"]}
            if f"k{k} N{n}" not in entry["tables"]:
                entry["tables"].append(f"k{k} N{n}")
    out = pd.DataFrame(list(rows.values()))
    if out.empty:
        return out
    out = out.drop(columns="_regret")
    out.insert(2, "n_trials", out["config_ids"].map(len))   # distinct configs, all tables
    out.insert(2, "n_tables", out["tables"].map(len))
    out.insert(2, "in_tables", out.pop("tables").map(", ".join))
    out["run_names"] = out["run_names"].map(tuple)
    out["config_ids"] = out["config_ids"].map(tuple)
    return (out.sort_values(["n_tables", "n_trials", "run_name"], ascending=[False, False, True])
            .reset_index(drop=True))


def candidate_table(trials, source="candidates", n_se=2.0, rel_tol=0.0, max_params=None,
                    metric="kl", card=None, show=True):
    """Compare chosen configs run on every (in-degree, sample size).

    trials : the tuning data (load_trials(RESULTS_DIR)); the rows with
             source=`source` (written by experiments.run_configs) are used.
             Runs are grouped by config_id (exact settings), not their name,
             so the same config run under different names is one row;
             duplicate cells from re-running keep the latest. The table is
             indexed by config_id (pick configs with
             experiments.get_configs(top_values, [config_id, ...])).
    max_params : {in-degree: limit}, e.g. {10: 100_000}: keep only configs with
             fewer total stored params (mean over the tuning DAGs) than the
             limit at that in-degree ("params k10" < 100000); configs with no
             params recorded there are dropped. The cap is applied first, so
             the thresholds below are from the best candidate under it.
    metric : what the configs are compared on, "kl" (default) or "val_loss".
    card   : only the runs at this cardinality (rows without card are card 2).
             Required if the candidate runs have more than one card, so
             results at different cardinalities are never averaged together.
    One column per (k, N) ("k6 N600"): the config's mean KL over the tuning
    DAGs. In the printed table, a value is bold if it is close to the best
    candidate for that (k, N) (under the size cap, if any): within n_se of
    the best's kl_se (statistically indistinguishable; kl_se = sqrt(sum of the
    DAGs' kl_se^2)/n, the DAGs have separate test sets) or within rel_tol of
    the best's KL (KL <= best * (1 + rel_tol): practically negligible, e.g.
    rel_tol=0.02 for 2%), whichever is looser. And one column per (k, N) of
    mean validation loss ("val k6 N600"). With metric="kl", a val loss is bold
    if within the same window in nats of the best val loss (val loss = entropy
    + KL with the entropy shared by all configs, so differences estimate the
    same thing). With metric="val_loss", it is bold if its mean per-DAG excess
    over the best candidate's val loss is within n_se of its paired standard
    error over the DAGs (see _paired_gap; rel_tol is not used for it), the val
    columns come first and the KL columns stay as reference. Rows are sorted
    by average rank on the metric over the (k, N) columns (rank 1 = lowest in
    a column; columns a config has no runs in are skipped), best first. Also
    listed per config: hidden_dims, lr, dropout,
    patience, batch_size, fit_time_s (mean seconds per fit) and stored_params
    per in-degree ("params k6", mean over the tuning DAGs), since the input
    layer grows with the number of parents. Returns the table.
    """
    if metric not in ("kl", "val_loss"):
        raise ValueError(f"metric must be 'kl' or 'val_loss', not {metric!r}")
    if "source" not in trials:
        return pd.DataFrame()
    r = trials[trials["source"] == source].copy()
    r["card"] = r["card"].fillna(TRIAL_DEFAULTS["card"]) if "card" in r else TRIAL_DEFAULTS["card"]
    if card is not None:
        r = r[r["card"] == card]
    elif r["card"].nunique() > 1:
        raise ValueError(f"candidate runs at cards {sorted(r['card'].unique())}: pass card=")
    if r.empty:
        return pd.DataFrame()
    if metric == "val_loss" and "val_loss" not in r:
        raise KeyError("no val_loss recorded for the candidate runs")
    r["_key"] = add_config_ids(r)["config_id"]          # identical settings = one config
    r = r.drop_duplicates(["_key", "in_degree", "sample_size", "dag_idx"], keep="last")
    cols = ["in_degree", "sample_size"]
    kl = r.pivot_table(index="_key", columns=cols, values="kl", aggfunc="mean")
    se = r.pivot_table(index="_key", columns=cols, values="kl_se",
                       aggfunc=lambda x: np.sqrt((x ** 2).sum()) / len(x))
    params = (r.pivot_table(index="_key", columns="in_degree", values="stored_params",
                            aggfunc="mean")       # depends on in-degree (input size)
              if "stored_params" in r else pd.DataFrame(index=kl.index))
    vl = (r.pivot_table(index="_key", columns=cols, values="val_loss", aggfunc="mean")
          if "val_loss" in r else None)
    best_val = vl.min() if vl is not None else None    # over all candidates, for reference
    best_kl_all = kl.min()
    if max_params:      # drop the configs over the cap first: the thresholds are from the best under it
        keep = pd.Series(True, index=kl.index)
        for k, cap in max_params.items():
            if k not in params:
                raise KeyError(f"no stored_params recorded for in-degree {k}")
            keep &= (params[k].reindex(kl.index) < cap).fillna(False)
        kl, se = kl[keep], se[keep]
        if vl is not None:
            vl = vl.reindex(kl.index)
        r = r[r["_key"].isin(kl.index)]
        kl = kl.dropna(axis=1, how="all")       # (k, N) cells no capped config ran
        if kl.empty:
            print(f"no candidates with max_params={max_params}")
            return pd.DataFrame()
    # threshold per (k, N), from the best candidate (under the size cap): within n_se
    # of its kl_se, or within rel_tol of its KL, whichever is looser
    limit, best_kl = {}, {}
    for c in kl.columns:
        best = kl[c].idxmin()
        best_kl[c] = kl.at[best, c]
        limit[c] = max(kl.at[best, c] + n_se * se.at[best, c], kl.at[best, c] * (1 + rel_tol))
    if vl is None:
        val_ok = None
    elif metric == "val_loss":
        # paired over the DAGs against the best candidate (under the size cap): within
        # n_se of the paired SE of the per-DAG differences
        val_ok = pd.DataFrame(index=vl.index)
        for c in vl.columns:
            v = r[(r["in_degree"] == c[0]) & (r["sample_size"] == c[1])].pivot_table(
                index="_key", columns="dag_idx", values="val_loss", aggfunc="mean")
            gap, gse = _paired_gap(v, v.mean(axis=1).idxmin())
            val_ok[c] = (gap <= n_se * gse).reindex(vl.index).fillna(False)
    else:
        # validation loss = entropy (the same for every config: same data) + KL, so a
        # difference in val loss estimates the same as one in KL: give it the KL window in
        # nats, from its own best (under the size cap too)
        val_ok = pd.DataFrame({c: vl[c] <= vl[c].min() + (limit[c] - best_kl[c])
                               for c in kl.columns})
    near = pd.DataFrame({c: kl[c] <= limit[c] for c in kl.columns})

    # average rank on the metric, best first
    order = (vl if metric == "val_loss" else kl).rank().mean(axis=1).sort_values(kind="stable").index

    label = lambda kn: f"k{kn[0]} N{kn[1]}"
    kl_cols = [label(c) for c in kl.columns]
    near = near.loc[order].set_axis(kl_cols, axis=1)
    kl_out = kl.loc[order].set_axis(kl_cols, axis=1)
    # mean validation loss per (k, N) (the validation-only view; scale differs with k and N)
    val_cols, near_val = [], pd.DataFrame(index=order)
    val_out = pd.DataFrame(index=order)
    if vl is not None:
        for c in kl.columns:
            val_out[f"val {label(c)}"] = vl[c].reindex(order)
            near_val[f"val {label(c)}"] = val_ok[c].reindex(order).astype(bool)
        val_cols = list(near_val.columns)
    # the metric's columns first
    out = pd.concat([val_out, kl_out] if metric == "val_loss" else [kl_out, val_out], axis=1)
    # settings and cost, per config
    by = r.groupby("_key")
    for c in ("hidden_dims", "lr", "dropout", "patience", "batch_size"):
        if c in r:
            out[c] = by[c].first()
    if "fit_time_s" in r:
        out["fit_time_s"] = by["fit_time_s"].mean()      # mean seconds per fit
    for k in params.columns:
        out[f"params k{k}"] = params[k]
    out.index.name = "config_id"
    near.index = near_val.index = out.index
    if show:
        kl_text = (f"mean KL over the tuning DAGs (k N), bold = within {n_se:g} SE or "
                   f"{rel_tol:.0%} of the best candidate")
        cap_text = (f" (only configs with total params {max_params} shown; the best is the "
                    f"best of those)" if max_params else "")
        if metric == "val_loss":
            print(f"chosen configs on every (in-degree, sample size): mean validation loss over "
                  f"the tuning DAGs (val k N), bold = within {n_se:g} paired SE (over the DAGs) of "
                  f"the best candidate for that in-degree and sample size; then, for reference, "
                  f"{kl_text}{cap_text}")
        else:
            print(f"chosen configs on every (in-degree, sample size): {kl_text} "
                  f"for that in-degree and sample size; then mean validation loss (val k N), "
                  f"bold = within the same window (in nats) of the best val loss{cap_text}")
        for c, kc in zip(kl.columns, kl_cols) if max_params else ():
            # for reference: how much the size cap costs against the best of all candidates
            if metric == "val_loss":
                if vl[c].min() > best_val[c]:
                    print(f"{kc}: the best candidate under the size cap ({vl[c].min():.4f}) is "
                          f"{vl[c].min() - best_val[c]:.4f} nats worse than the best of all "
                          f"candidates ({best_val[c]:.4f})")
            elif kl[c].min() > best_kl_all[c]:
                print(f"{kc}: the best candidate under the size cap ({kl[c].min():.4f}) is "
                      f"{kl[c].min() / best_kl_all[c] - 1:.1%} worse than the best of all "
                      f"candidates ({best_kl_all[c]:.4f})")
        try:
            fmt = {c: "{:.4f}" for c in kl_cols + val_cols}
            fmt.update({c: "{:.0f}" for c in out.columns if c.startswith("params")
                        or c in ("patience", "batch_size")})
            fmt.update({"dropout": "{:.1f}", "fit_time_s": "{:.1f}", "lr": "{:.0e}"})
            disp = out
            bold = pd.concat([near, near_val], axis=1)
            styles = pd.DataFrame(np.where(bold.to_numpy(), "font-weight: bold", ""),
                                  index=disp.index, columns=bold.columns)
            styled = disp.style.format({c: f for c, f in fmt.items() if c in disp}).apply(
                lambda _: styles, axis=None, subset=list(bold.columns))
            _show(styled)
        except ImportError:                              # no jinja2: mark with * instead
            marked = out.round(4).astype(str)
            for cs, nb in ((kl_cols, near), (val_cols, near_val)):
                marked[cs] = marked[cs].where(~nb.to_numpy(), marked[cs] + " *")
            _show(marked)
    return out




# ── Real data ───────────────────────────────────────────────────────────────

def plot_node_loglik(node_df, baseline="dirichlet_0.5", sample_size=None,
                     ax=None, figsize=(7, 5), out_path=None, title=None):
    """Per-node held-out log-likelihood gain over `baseline` vs that node's
    number of parent configurations (log x), one colour per estimator.

    node_df : real_data.node_table output. Each point is one node in one
    (discretisation, DAG), averaged over folds; positive = beats the baseline.
    Shows whether any NN advantage sits in the high in-degree nodes.
    """
    sub = node_df if sample_size is None else node_df[node_df["sample_size"] == sample_size]
    keys = ["disc_level", "dag_level", "n_train_spec", "node", "n_parent_configs"]
    mean = sub.groupby(keys + ["estimator"])["loglik"].mean().unstack("estimator")
    if baseline not in mean:
        print(f"WARNING: baseline {baseline!r} not in the results")
        return None
    gain = mean.drop(columns=[baseline]).sub(mean[baseline], axis=0).reset_index()

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    ests = [c for c in mean.columns if c != baseline]
    for est, c, (m, _) in zip(ests, _colors(len(ests)), _styles(len(ests))):
        ax.scatter(gain["n_parent_configs"], gain[est], s=18, alpha=0.6, color=c, marker=m, label=est)
    ax.axhline(0, color="grey", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("parent configurations of the node", fontsize=12)
    ax.set_ylabel(f"log-likelihood gain over {baseline}", fontsize=12)
    held = "all N" if sample_size is None else f"N={sample_size}"
    ax.set_title(title or held, fontsize=11, color=None if title else "grey")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    if own_fig:
        fig.tight_layout()
        if out_path:
            fig.savefig(out_path, dpi=150)
        return fig, ax
    return ax


# ── Report figures ──────────────────────────────────────────────────────────
# The figures in the report, at the report's column widths. Each function takes
# the results DataFrame (or file paths) and saves to out_path if given; .png is
# written at REPORT_DPI. Styling is applied inside each call only, so the
# exploratory plots above are unaffected.

REPORT_RC = {
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
    "axes.labelsize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.titlesize": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.3, "lines.markersize": 3.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
}
REPORT_DPI = 300
SINGLE_W, DOUBLE_W = 3.4, 7.0      # column and page widths (inches)

# run names -> the names used in the report
REPORT_ALIASES = {"nn_32_relu": "nn_32", "01_256x128_lr3e-4": "nn_256_128",
                  "k6_128_64_9": "nn_128_64", "dirichlet_0.5": "dirichlet"}
# per estimator: colour, marker + line style, label (same across every figure)
REPORT_COLORS = {"mle": "#222222", "dirichlet": "#7c3aed", "laplace": "#0d9488", "dirichlet_tuned": "#9333ea", "nn_linear": "#d97706",
                 "nn_32": "#2563eb", "nn_256_128": "#16a34a", "nn_128_64": "#e11d48",
                 "nn_256": "#0891b2", "nn_128": "#be185d", "nn_64_32": "#4d7c0f",
                 "nn_64": "#a16207", "nn_32_16": "#475569"}
REPORT_FMT    = {"mle": "o-", "dirichlet": "D:", "laplace": "x--", "dirichlet_tuned": "d-", "nn_linear": "s--", "nn_32": "^-.",
                 "nn_256_128": "P-", "nn_128_64": "*--",
                 "nn_256": "X-", "nn_128": "v:", "nn_64_32": "h-.", "nn_64": "<--", "nn_32_16": ">:"}
REPORT_LABELS = {"mle": "MLE", "dirichlet": "Dirichlet", "laplace": "Laplace",
                 "dirichlet_tuned": "Dirichlet (tuned)"}


def _canon(name):
    return REPORT_ALIASES.get(name, name)


def _est_style(name, i=0):
    """(colour, fmt, label) for an estimator; unknown names get the i-th fallback,
    whose marker is one no named estimator uses (so it stays distinct in black and white)."""
    c = _canon(name)
    if c in REPORT_FMT:
        fmt = REPORT_FMT[c]
    else:
        used = {f.rstrip("-.:") for f in REPORT_FMT.values()}
        free = [m for m in _MARKERS + ["h", "<", ">", "p", "d", "1", "2", "3", "4"] if m not in used] or _MARKERS
        fmt = free[i % len(free)] + _LINESTYLES[i % len(_LINESTYLES)]
    return (REPORT_COLORS.get(c, _PALETTE[i % len(_PALETTE)]), fmt, REPORT_LABELS.get(c, c))


def _shade(i, n, cmap="Blues"):
    return plt.get_cmap(cmap)(0.35 + 0.65 * i / max(n - 1, 1))


def _legend_top(fig, ax, ncol, y=1.08):
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(.5, y))


def _save(fig, out_path):
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=REPORT_DPI)
    return fig


def _with_card(df):
    """Rows from before card was recorded were all card=2."""
    return df.assign(card=df["card"].fillna(2) if "card" in df else 2)


def plot_neural_cpd_diagram(out_path=None):
    """Schematic of one node X with parents A (2 states) and B (3 states):
    (a) its CPT, one row per parent configuration, each estimated from its own
    rows (an unseen configuration stays uniform); (b) the neural CPD, which maps
    the concatenated one-hot parents through a ReLU layer to a softmax over X."""
    from matplotlib.patches import Rectangle
    ink, faint, accent = "#222222", "#9a9a9a", "#2563eb"
    with plt.rc_context(REPORT_RC):
        fig, (ta, nb) = plt.subplots(2, 1, figsize=(SINGLE_W, 3.3), gridspec_kw={"height_ratios": [1, 1.05]})
        for ax in (ta, nb):
            ax.set_xlim(0, 10); ax.set_axis_off()

        # (a) the CPT
        ta.set_ylim(0, 7.6)
        ta.set_title("(a) Conditional probability table", loc="left")
        rows = [(0, 0, ".81", ".19"), (0, 1, ".64", ".36"), (0, 2, ".50", ".50"),
                (1, 0, ".27", ".73"), (1, 1, ".12", ".88"), (1, 2, ".33", ".67")]
        cols = [0.5, 1.25, 2.5, 4.05]
        y0, dy = 6.6, 0.95
        for x, h in zip(cols, ["$A$", "$B$", "$P(X{=}0)$", "$P(X{=}1)$"]):
            ta.text(x, y0, h, ha="center", va="center")
        ta.plot([0.1, 4.75], [y0 - 0.5, y0 - 0.5], color=ink, lw=0.6)
        for i, (a, b, p0, p1) in enumerate(rows):
            y = y0 - 1 - i * dy
            unseen = (a, b) == (0, 2)
            if unseen:
                ta.add_patch(Rectangle((0.1, y - dy / 2), 4.65, dy, color="#e5e5e5", lw=0))
            for x, t in zip(cols, [a, b, p0, p1]):
                ta.text(x, y, str(t), ha="center", va="center", color=faint if unseen else ink)
        ta.text(5.3, 5.6, "One row per parent\nconfiguration: $r^k$ rows\nfor $k$ parents", va="center", fontsize=7, linespacing=1.1)
        ta.text(5.3, 2.4, "Each row is estimated\nfrom its own counts; a\nrow with no data (grey)\nstays uniform", va="center", fontsize=7, linespacing=1.1)

        # (b) the neural CPD
        nb.set_ylim(-0.7, 7.6)
        nb.set_title("(b) Neural CPD", loc="left")
        inp = [6.1, 5.1, 3.8, 2.8, 1.8]                      # A: 2 units, B: 3 units
        bits = [0, 1, 0, 0, 1]                                # A = 1, B = 2
        hid = [6.0, 4.7, 3.4, 2.1]
        out = [4.75, 3.35]
        xi, xh, xo = 2.3, 5.0, 7.6
        for yi in inp:
            for yh in hid:
                nb.plot([xi + 0.3, xh - 0.17], [yi, yh], color=faint, lw=0.4, zorder=1)
        for yh in hid:
            for yo in out:
                nb.plot([xh + 0.17, xo - 0.3], [yh, yo], color=faint, lw=0.4, zorder=1)
        for yi, bit in zip(inp, bits):
            nb.add_patch(Rectangle((xi - 0.3, yi - 0.4), 0.6, 0.8, facecolor=accent if bit else "white",
                                   edgecolor=ink, lw=0.6, zorder=2))
            nb.text(xi, yi, str(bit), ha="center", va="center", color="white" if bit else ink, zorder=3, fontsize=7)
        nb.text(xi - 0.55, 5.6, "$A{=}1$", ha="right", va="center")
        nb.text(xi - 0.55, 2.8, "$B{=}2$", ha="right", va="center")
        nb.plot([xh] * len(hid), hid, "o", ms=8, mfc="white", mec=ink, mew=0.6, zorder=2)   # markers stay round
        for yo, lab in zip(out, ["$X{=}0$", "$X{=}1$"]):
            nb.add_patch(Rectangle((xo - 0.3, yo - 0.4), 0.6, 0.8, facecolor="white", edgecolor=ink, lw=0.6, zorder=2))
            nb.text(xo + 0.5, yo, lab, va="center")
        nb.text(xi, 0.1, "One-hot parents,\nconcatenated", ha="center", va="center", fontsize=7, linespacing=1.1)
        nb.text(xh, 0.1, "Hidden layer\n(ReLU)", ha="center", va="center", fontsize=7, linespacing=1.1)
        nb.text(xo, 0.1, "Softmax:\n$P(X \\mid A, B)$", ha="center", va="center", fontsize=7, linespacing=1.1)
        return _save(fig, out_path)


def plot_mle_fails(df, estimators=("mle",), sample_size=1000, alpha=0.5,
                   ks=None, logy=None, out_path=None):
    """KL vs in-degree for each tabular estimator, then CPT memory (the same for
    all of them), one line per network size, side by side (a full-width figure).
    logy : log KL axes shared across the estimators (default with more than one
    estimator, so they compare directly); False draws each KL panel on its own
    linear axis, which shows the shape of the rise rather than the ratio."""
    if isinstance(estimators, str):
        estimators = (estimators,)
    m = df[df.estimator.isin(estimators) & (df.sample_size == sample_size) & (df.alpha == alpha)]
    nodes = sorted(m.n_nodes.unique())
    logy = len(estimators) > 1 if logy is None else logy
    n_ax = len(estimators) + 1
    with plt.rc_context(REPORT_RC):
        fig, ax = plt.subplots(1, n_ax, figsize=(DOUBLE_W, 1.9))
        for j, e in enumerate(estimators):
            if j and logy:
                ax[j].sharey(ax[0])
            me = m[m.estimator == e]
            for i, ((mk, ls), n) in enumerate(zip(_styles(len(nodes)), nodes)):
                s = me[me.n_nodes == n].groupby("in_degree")
                fmt = dict(marker=mk, ls=ls, color=_shade(i, len(nodes)))
                ax[j].plot(s.kl.mean().index, s.kl.mean(), label=f"{n} nodes", **fmt)
                if j == 0:
                    ax[-1].plot(s.memory_bytes.mean().index, s.memory_bytes.mean() / 1024, **fmt)
            ax[j].set_ylabel("KL divergence (nats)")
            if logy:
                ax[j].set_yscale("log")
            else:
                ax[j].set_ylim(bottom=0)
        ax[-1].set_ylim(bottom=0); ax[-1].set_ylabel("CPT memory (KB)")
        titles = [f"({chr(97 + j)}) KL divergence: {_est_style(e)[2]}" if len(estimators) > 1 else "(a) KL divergence"
                  for j, e in enumerate(estimators)] + [f"({chr(97 + len(estimators))}) CPT memory"]
        for a, t in zip(ax, titles):
            a.set_xticks(ks or sorted(m.in_degree.unique())); a.set_title(t, loc="left")
        for a in ax:
            a.set_xlabel("Parent in-degree $k$")
        _legend_top(fig, ax[0], len(nodes), 1.1)
        return _save(fig, out_path)


def plot_no_es(hist_dir, archs=("nn_32", "nn_64_32", "nn_128_64"), k=10, n=600, card=None, out_path=None):
    """Training and validation loss without early stopping, mean over the DAGs
    whose histories are in hist_dir (hist_{arch}_c{card}_k{k}_d*_n{n}.npz, or
    hist_{arch}_k{k}_d*_n{n}.npz from before card was in the name).
    Nothing is drawn or saved unless every architecture has histories, so a
    missing run never overwrites a saved figure with empty panels."""
    from glob import glob
    import os
    files_by_arch = {arch: sorted(glob(os.path.join(hist_dir, f"hist_{arch}_c{card or '*'}_k{k}_d*_n{n}.npz"))
                                  + glob(os.path.join(hist_dir, f"hist_{arch}_k{k}_d*_n{n}.npz")))
                     for arch in archs}
    missing = [a for a, f in files_by_arch.items() if not f]
    if missing:
        print(f"no histories for {missing} in {hist_dir}: run the no-early-stopping diagnostic first "
              "(figure not made)")
        return None
    with plt.rc_context(REPORT_RC):
        fig, axs = plt.subplots(1, len(archs), figsize=(DOUBLE_W, 1.75), sharey=True, squeeze=False)
        for ax, arch in zip(axs[0], archs):
            files = files_by_arch[arch]
            tr = np.mean([np.load(f)["train"].sum(0) for f in files], 0)
            va = np.mean([np.load(f)["val"].sum(0) for f in files], 0)
            ep = np.arange(1, len(tr) + 1)
            ax.plot(ep, tr, "--", color="#888", label="training")
            ax.plot(ep, va, "-", color="#2563eb", label="validation")
            b = va.argmin(); ax.plot(ep[b], va[b], "o", color="#2563eb", ms=4)
            ax.annotate(f"min @ {ep[b]}", (ep[b], va[b]), xytext=(4, -10), textcoords="offset points",
                        fontsize=6.5, color="#2563eb")
            ax.set_title(arch, loc="left"); ax.set_xlabel("Epoch")
        axs[0, 0].set_ylabel("Summed cross-entropy"); axs[0, 0].legend(frameon=False)
        return _save(fig, out_path)


def plot_sweep_lr(trials, sweeps, metric="val_loss", out_path=None):
    """Mean `metric` (the selection score, validation loss by default) vs learning rate per sweep trial, coloured and marked by hidden layers.

    sweeps : {in-degree: W&B sweep id}. Only runs scored on every cell of the
    sweep (all DAGs x sample sizes) are shown.
    """
    t = trials[trials.sweep_id.isin(sweeps.values())]
    if "source" in t:          # sweep trials only, not candidate runs added to the sweep
        t = t[t["source"].fillna("sweep") == "sweep"]
    # one colour + marker per hidden-layer setting, the same in every panel
    hds = sorted(t.hidden_dims.unique(), key=lambda h: (len(str(h)), str(h)))
    with plt.rc_context(REPORT_RC):
        fig, axs = plt.subplots(1, len(sweeps), figsize=(DOUBLE_W, 1.8), squeeze=False)
        for ax, (k, sid) in zip(axs[0], sweeps.items()):
            g = (t[t.sweep_id == sid].groupby("run_id")
                 .agg(lr=("lr", "first"), hd=("hidden_dims", "first"), y=(metric, "mean"), n=(metric, "size")))
            g = g[g.n == g.n.max()]
            for i, ((mk, _), hd) in enumerate(zip(_styles(len(hds)), hds)):
                q = g[g.hd == hd]
                ax.scatter(q.lr, q.y, s=9, color=_PALETTE[i % len(_PALETTE)], marker=mk, alpha=.8, lw=0,
                           label=str(hd).strip("[]()").replace(", ", "-").rstrip(","))
            ax.set_xscale("log"); ax.set_title(f"$k={k}$", loc="left"); ax.set_xlabel("Learning rate")
        axs[0, 0].set_ylabel({"val_loss": "Mean validation loss", "kl": "Mean KL"}.get(metric, metric) + " (tuning DAGs)")
        _legend_top(fig, axs[0, 0], 4, 1.1)
        return _save(fig, out_path)


_SETTING = ["n_nodes", "in_degree", "alpha", "sample_size", "dag_idx"]


def plot_estimators_by_k(df, estimators, param_nodes=20, kl_nodes=None, out_path=None):
    """(a) KL and (b) free parameters vs in-degree, one line per estimator, on
    the settings every estimator was run on. kl_nodes: the network size the KL
    panel uses (None = all sizes pooled); the parameter panel uses param_nodes.
    No error bars: the standard errors across DAGs are smaller than the markers,
    so the captions report their range instead."""
    sub = df[df.estimator.isin(estimators)]
    shared = sub.groupby(_SETTING).estimator.nunique()
    shared = shared[shared == len(estimators)].reset_index()[_SETTING]
    sub = sub.merge(shared, on=_SETTING)
    with plt.rc_context(REPORT_RC):
        fig, axs = plt.subplots(1, 2, figsize=(DOUBLE_W, 1.95))
        for i, e in enumerate(estimators):
            c, fmt, lab = _est_style(e, i)
            kl_sub = sub if kl_nodes is None else sub[sub.n_nodes == kl_nodes]
            q = kl_sub[kl_sub.estimator == e].groupby("in_degree").kl.mean()
            axs[0].plot(q.index, q, fmt, color=c, label=lab, ms=4.5)
            p = sub[(sub.estimator == e) & (sub.n_nodes == param_nodes)].groupby("in_degree").free_params.mean()
            axs[1].plot(p.index, p, fmt, color=c, ms=4.5)
        kl_title = "(a) KL divergence" + ("" if kl_nodes is None else f" ({kl_nodes} nodes)")
        for a, t, yl in zip(axs, [kl_title, f"(b) Free parameters ({param_nodes} nodes)"],
                            ["KL divergence (nats)", "Free parameters"]):
            a.set_yscale("log"); a.set_xticks(sorted(sub.in_degree.unique()))
            a.set_xlabel("Parent in-degree $k$"); a.set_title(t, loc="left"); a.set_ylabel(yl)
        _legend_top(fig, axs[0], len(estimators), 1.1)
        return _save(fig, out_path)


def plot_structured(df, estimators, out_path=None):
    """Structured ground truth: mean KL divergence vs in-degree, one panel per
    interaction share (0 = additive logistic CPDs), one line per estimator,
    averaged over sample sizes and DAGs."""
    levels = sorted(df.interaction.unique())
    titles = {0.0: "additive (no interactions)"}
    with plt.rc_context(REPORT_RC):
        fig, axs = plt.subplots(1, len(levels), figsize=(DOUBLE_W, 2.0), sharey=True, squeeze=False)
        for ax, lam, t in zip(axs[0], levels, "abcdefgh"):
            sub = df[df.interaction == lam]
            for i, e in enumerate(estimators):
                c, fmt, lab = _est_style(e, i)
                q = sub[sub.estimator == e].groupby("in_degree").kl.mean()
                ax.plot(q.index, q, fmt, color=c, label=lab)
            ax.set_yscale("log"); ax.set_xticks(sorted(sub.in_degree.unique()))
            ax.set_xlabel("Parent in-degree $k$")
            ax.set_title(f"({t}) " + titles.get(lam, f"interaction share {lam:g}"), loc="left")
        axs[0, 0].set_ylabel("KL divergence (nats)")
        _legend_top(fig, axs[0, 0], len(estimators), 1.12)
        return _save(fig, out_path)


def plot_kl_vs_n(df, estimators, in_degrees=None, out_path=None):
    """KL vs training rows, one panel per in-degree, one line per estimator
    (no error bars: see plot_estimators_by_k)."""
    in_degrees = in_degrees or sorted(df.in_degree.unique())
    with plt.rc_context(REPORT_RC):
        fig, axs = plt.subplots(1, len(in_degrees), figsize=(DOUBLE_W, 1.75), squeeze=False)
        for ax, k in zip(axs[0], in_degrees):
            for i, e in enumerate(estimators):
                c, fmt, lab = _est_style(e, i)
                q = df[(df.estimator == e) & (df.in_degree == k)].groupby("sample_size").kl.mean()
                ax.plot(q.index, q, fmt, color=c, label=lab)
            ax.set_yscale("log"); ax.set_xticks(sorted(df.sample_size.unique()))
            ax.set_title(f"$k={k}$", loc="left"); ax.set_xlabel("Training rows $N$")
        axs[0, 0].set_ylabel("KL divergence (nats)")
        _legend_top(fig, axs[0, 0], len(estimators), 1.1)
        return _save(fig, out_path)


def plot_kl_vs_n_by_size(df, estimators, sizes, in_degrees=None, out_path=None):
    """KL vs training rows, one row of panels per network size and one column per
    in-degree: whether the ordering of the estimators depends on network size."""
    in_degrees = in_degrees or sorted(df.in_degree.unique())
    with plt.rc_context(REPORT_RC):
        fig, axs = plt.subplots(len(sizes), len(in_degrees), figsize=(DOUBLE_W, 1.45 * len(sizes)), squeeze=False)
        for r, n in enumerate(sizes):
            for ax, k in zip(axs[r], in_degrees):
                sub = df[(df.n_nodes == n) & (df.in_degree == k)]
                for i, e in enumerate(estimators):
                    c, fmt, lab = _est_style(e, i)
                    q = sub[sub.estimator == e].groupby("sample_size").kl.mean()
                    ax.plot(q.index, q, fmt, color=c, label=lab)
                ax.set_yscale("log"); ax.set_xticks(sorted(df.sample_size.unique()))
                ax.tick_params(labelsize=6)
                if r == 0:
                    ax.set_title(f"$k={k}$", loc="left")
                if r == len(sizes) - 1:
                    ax.set_xlabel("Training rows $N$")
            axs[r, 0].set_ylabel(f"{n} nodes\nKL (nats)")
        _legend_top(fig, axs[0, 0], len(estimators), 1.0 + 0.08 / len(sizes) * 4)
        return _save(fig, out_path)


def rows_to_reach(df, estimators, threshold=5.0):
    """Training rows each estimator needs for its mean KL divergence to fall to
    `threshold` nats, by linear interpolation between the sample sizes run;
    "<N_min" if already reached at the smallest, ">N_max" if never reached.
    -> DataFrame, in-degree x estimator."""
    m = df.groupby(["estimator", "in_degree", "sample_size"]).kl.mean()
    out = {}
    for e in estimators:
        col = {}
        for k, s in m[e].groupby(level="in_degree"):
            s = s.droplevel("in_degree").sort_index()
            if s.iloc[0] <= threshold:
                col[k] = f"<{s.index[0]}"
                continue
            col[k] = f">{s.index[-1]}"
            for (n0, k0), (n1, k1) in zip(s.items(), list(s.items())[1:]):
                if k1 <= threshold:
                    col[k] = f"{n0 + (k0 - threshold) / (k0 - k1) * (n1 - n0):.0f}"
                    break
        out[e] = col
    return pd.DataFrame(out).rename_axis("in_degree")


def rows_to_match(res, estimator, reference="dirichlet", sample_sizes=None):
    """Real data: training rows `reference` needs for its mean test log-likelihood
    to reach `estimator`'s at each N, interpolated on log N between the sample
    sizes run; "<N_min" if reached at the smallest, ">N_max" if never reached.
    -> DataFrame, (discretisation, DAG) x the estimator's N."""
    m = res.groupby(["estimator", "disc_level", "dag_level", "sample_size"]).loglik.mean()
    out = {}
    for (dl, gl), ref in m[reference].groupby(level=["disc_level", "dag_level"]):
        ref = ref.droplevel(["disc_level", "dag_level"]).sort_index()
        est = m[estimator][dl][gl].sort_index()
        row = {}
        for n, target in est.items():
            if sample_sizes is not None and n not in sample_sizes:
                continue
            if ref.iloc[0] >= target:
                row[n] = f"<{ref.index[0]}"
                continue
            row[n] = f">{ref.index[-1]}"
            for (n0, l0), (n1, l1) in zip(ref.items(), list(ref.items())[1:]):
                if l1 >= target:
                    row[n] = f"{np.exp(np.log(n0) + (target - l0) / (l1 - l0) * np.log(n1 / n0)):.0f}"
                    break
        out[(dl, gl)] = row
    return pd.DataFrame(out).T.rename_axis(["disc_level", "dag_level"])


def plot_size_alpha(df, estimators, out_path=None):
    """Mean KL (a) by network size and (b) by the Dirichlet alpha of the true CPTs.
    With a single alpha in df, panel (b) would be one point, so only (a) is drawn."""
    by_alpha = df.alpha.nunique() > 1
    with plt.rc_context(REPORT_RC):
        fig, axs = plt.subplots(1, 2 if by_alpha else 1, figsize=(DOUBLE_W if by_alpha else SINGLE_W, 1.8),
                                squeeze=False)
        axs = axs[0]
        for i, e in enumerate(estimators):
            c, fmt, lab = _est_style(e, i)
            q = df[df.estimator == e].groupby("n_nodes").kl.mean(); axs[0].plot(q.index, q, fmt, color=c, label=lab)
            if by_alpha:
                q = df[df.estimator == e].groupby("alpha").kl.mean(); axs[1].plot(q.index, q, fmt, color=c)
        axs[0].set_xlabel("Network size (nodes)"); axs[0].set_xticks(sorted(df.n_nodes.unique()))
        if by_alpha:
            axs[1].set_xlabel(r"Dirichlet $\alpha$ of the true CPTs"); axs[1].set_xticks(sorted(df.alpha.unique()))
        titles = ["(a) By network size", "(b) By ground-truth concentration"] if by_alpha else [None]
        for a, t in zip(axs, titles):
            a.set_yscale("log"); a.set_ylabel("KL divergence (nats)")
            if t:
                a.set_title(t, loc="left")
        _legend_top(fig, axs[0], len(estimators), 1.1 if by_alpha else 1.14)
        return _save(fig, out_path)


def with_full_indegree_kl(df):
    """Rows recorded with node_kl, with kl replaced by the sum of the conditional
    KL terms of the nodes that have exactly in_degree parents. For those nodes
    rho = N / r^k is exact; the network total also mixes in the first k nodes of
    each DAG, which have fewer parents."""
    out = df[df["node_kl"].notna()].copy()
    out["kl"] = [sum(term for term, n_pa in json.loads(s).values() if n_pa == k)
                 for s, k in zip(out["node_kl"], out["in_degree"])]
    return out


def plot_ratio_vs_rho(df, baselines=("mle", "dirichlet"), estimator="nn_32", n_nodes=None, alpha=0.5,
                      logy=True, out_path=None):
    """KL(baseline) / KL(estimator) vs rows per parent configuration rho = N / r^k:
    one panel per baseline, one line per in-degree k joining its sample sizes,
    every cardinality in df on the same axes (r=2 solid with open markers, r=4
    dashed with filled ones). If rho captured in-degree, cardinality and N
    completely, all lines would fall on one curve; the gaps between lines are
    what it misses. Each point is the mean of the per-DAG ratios, pooled over the
    network sizes in n_nodes (None = every size run at all of df's cardinalities;
    rho is per node and the ratio cancels network size, so pooling is fair).
    Returns (fig, table of the plotted points)."""
    from matplotlib.lines import Line2D
    baselines = [baselines] if isinstance(baselines, str) else list(baselines)
    df = _with_card(df[df.alpha == alpha])
    if n_nodes is None:
        n_nodes = sorted(set.intersection(*(set(s.n_nodes) for _, s in df.groupby("card"))))
    n_nodes = [n_nodes] if np.isscalar(n_nodes) else list(n_nodes)
    w = (df[df.n_nodes.isin(n_nodes)]
         .pivot_table(index=["card", "in_degree", "sample_size", "n_nodes", "dag_idx"], columns="estimator",
                      values="kl")
         .reset_index())
    w["rho"] = w.sample_size / w.card ** w.in_degree
    for b in baselines:
        w[f"ratio_{b}"] = w[b] / w[estimator]
    g = (w.groupby(["card", "in_degree", "sample_size"])
         .agg(rho=("rho", "first"), n_dags=("rho", "size"),
              **{f"ratio_{b}": (f"ratio_{b}", "mean") for b in baselines})
         .reset_index().sort_values(["card", "in_degree", "sample_size"]))
    ks, cards = sorted(g.in_degree.unique()), sorted(g.card.unique())
    k_style = {k: (_PALETTE[i % len(_PALETTE)], mk) for i, ((mk, _), k) in enumerate(zip(_styles(len(ks)), ks))}
    r_style = {r: (ls, filled) for r, ls, filled in zip(cards, ["-", "--", ":", "-."], [False, True, True, True])}
    lab = _est_style(estimator)[2]
    with plt.rc_context(REPORT_RC):
        fig, axs = plt.subplots(1, len(baselines), figsize=(DOUBLE_W, 2.2), squeeze=False)
        for ax, b, t in zip(axs[0], baselines, "abcdefgh"):
            for r in cards:
                ls, filled = r_style[r]
                for k in ks:
                    s = g[(g.card == r) & (g.in_degree == k)]
                    if s.empty:
                        continue
                    c, mk = k_style[k]
                    ax.plot(s.rho, s[f"ratio_{b}"], ls=ls, marker=mk, color=c, lw=.9, ms=4,
                            mfc=c if filled else "white")
            ax.axhline(1, color="#999", lw=.6, ls=":")
            ax.set_xscale("log")   # rho spans six orders of magnitude
            if logy:
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(plt.FixedLocator([0.25, 0.5, 1, 2, 4, 8, 16]))
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
                ax.yaxis.set_minor_formatter(plt.NullFormatter())
            else:
                ax.set_ylim(bottom=0)
            ax.set_xlabel(r"Rows per parent configuration $\rho = N/r^{k}$")
            ax.set_ylabel(f"KL({_est_style(b)[2]}) / KL({lab})")
            ax.set_title(f"({t}) {_est_style(b)[2]} relative to {lab}", loc="left")
        handles = [Line2D([], [], color=k_style[k][0], marker=k_style[k][1], lw=.9, ms=4, label=f"$k={k}$")
                   for k in ks]
        handles += [Line2D([], [], color="#555", ls=r_style[r][0], marker="o", lw=.9, ms=4,
                           mfc="#555" if r_style[r][1] else "white", label=f"$r={r}$") for r in cards]
        fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(.5, 1.08))
        return _save(fig, out_path), g


def _n_label(n):
    return f"{n / 1000:.0f}k" if n >= 1000 else f"{n:.0f}"


def real_loglik_panel(ax, s, estimators, err="std", baseline="mle", title=None, ylab=False):
    """Held-out log-likelihood vs N on ax. Baseline means below the other
    estimators' range are drawn as open triangles on the bottom edge with their
    value, instead of a line leaving the axes."""
    other = s[(s.estimator != baseline) & s.estimator.isin(estimators)]
    lo, hi = other.groupby(["estimator", "sample_size"]).loglik.mean().agg(["min", "max"])
    pad = 0.08 * (hi - lo); ylo, yhi = lo - 2.2 * pad, hi + pad
    for i, e in enumerate(estimators):
        c, fmt, lab = _est_style(e, i)
        q = s[s.estimator == e].groupby("sample_size").loglik.agg(["mean", err])
        bar = dict(fmt=fmt, color=c, label=lab, capsize=1.5, elinewidth=.6)
        if e != baseline:
            ax.errorbar(q.index, q["mean"], yerr=q[err], **bar)
            continue
        inr, out = q[q["mean"] >= ylo], q[q["mean"] < ylo]
        if len(inr):
            ax.errorbar(inr.index, inr["mean"], yerr=inr[err], **bar)
        else:
            ax.plot([], [], fmt, color=c, label=lab)
        ybar = ylo + 0.35 * pad
        for xv, row in out.iterrows():
            ax.plot(xv, ybar, "v", color=c, mfc="white", ms=4, clip_on=False)
            ax.annotate(f"{row['mean']:.1f}", (xv, ybar), xytext=(4, 2), textcoords="offset points",
                        ha="left", fontsize=6, color=c)
        if len(inr) and len(out):   # dotted link from the edge marker to the first in-range point
            ax.plot([out.index[-1], inr.index[0]], [ybar, inr["mean"].iloc[0]], ":", color=c, lw=.8)
    ns = sorted(s.sample_size.unique())
    ax.set_xscale("log"); ax.set_xlim(ns[0] / 1.5, ns[-1] * 1.6)
    ax.set_xticks(ns); ax.set_xticklabels([_n_label(n) for n in ns]); ax.minorticks_off()
    ax.set_ylim(ylo, yhi); ax.set_xlabel("Training rows $N$")
    if title: ax.set_title(title, loc="left")
    if ylab: ax.set_ylabel("Held-out log-likelihood (nats/row)")
    return ax


def plot_real_mean(res, estimators, err="std", out_path=None):
    """Held-out log-likelihood vs N, averaged over every discretisation and DAG."""
    mean = res.groupby(["estimator", "sample_size", "fold"], as_index=False).loglik.mean()
    with plt.rc_context(REPORT_RC):
        fig, ax = plt.subplots(figsize=(SINGLE_W, 2.2))
        real_loglik_panel(ax, mean, estimators, err=err, ylab=True)
        _legend_top(fig, ax, 3, 1.13)
        return _save(fig, out_path)


DIAG_METRICS = [("diag_auroc", "AUROC", "higher"), ("diag_logloss", "Log-loss (nats)", "lower")]


def plot_real_diagnosis(res, estimators, prevalence=None, err="std", out_path=None):
    """Diagnostic accuracy of P(Diabetes | every other variable) vs N: one panel
    per metric, each point the mean over every discretisation and DAG (then
    mean and `err` over folds). prevalence: the positive rate; draws the score
    of always predicting it as a dotted reference line."""
    mean = res.groupby(["estimator", "sample_size", "fold"], as_index=False)[
        [m for m, _, _ in DIAG_METRICS]].mean()
    ref = None if prevalence is None else {
        "diag_auroc": 0.5,
        "diag_logloss": -(prevalence * np.log(prevalence) + (1 - prevalence) * np.log(1 - prevalence))}
    ns = sorted(mean.sample_size.unique())
    with plt.rc_context(REPORT_RC):
        # two metrics fit one column; panel titles stay short and the caption says which way is better
        fig, axs = plt.subplots(1, len(DIAG_METRICS), figsize=(SINGLE_W, 2.0))
        for ax, (m, lab, better), t in zip(axs, DIAG_METRICS, "abcdefgh"):
            for i, e in enumerate(estimators):
                c, fmt, elab = _est_style(e, i)
                q = mean[mean.estimator == e].groupby("sample_size")[m].agg(["mean", err])
                ax.errorbar(q.index, q["mean"], yerr=q[err], fmt=fmt, color=c, label=elab,
                            capsize=1.5, elinewidth=.6)
            if ref is not None:
                ax.axhline(ref[m], color="#999", lw=.7, ls=":", label="Prevalence only")
            ax.set_xscale("log"); ax.set_xlim(ns[0] / 1.5, ns[-1] * 1.6)
            ax.set_xticks(ns); ax.set_xticklabels([_n_label(n) for n in ns]); ax.minorticks_off()
            ax.set_xlabel("Training rows $N$"); ax.set_title(f"({t}) {lab}", loc="left")
        _legend_top(fig, axs[0], 2, 1.16)
        return _save(fig, out_path)


def plot_real_grid(res, estimators, disc_levels, dag_levels, err="std", out_path=None):
    """Held-out log-likelihood vs N: one panel per (discretisation, DAG confidence)."""
    with plt.rc_context(REPORT_RC):
        fig, axs = plt.subplots(len(disc_levels), len(dag_levels), squeeze=False,
                                figsize=(DOUBLE_W, 1.8 * len(disc_levels)))
        for i, dl in enumerate(disc_levels):
            for j, gl in enumerate(dag_levels):
                real_loglik_panel(axs[i, j], res[(res.disc_level == dl) & (res.dag_level == gl)], estimators,
                                  err=err, title=f"{dl}, {gl} confidence", ylab=(j == 0))
        _legend_top(fig, axs[0, 0], len(estimators), 1.04)
        return _save(fig, out_path)


def plot_real_unseen(res, disc_levels, dag_levels, estimator="mle", out_path=None):
    """Fraction of test parent configurations unseen in training vs N, one line
    per (discretisation, DAG confidence); the rate is the same for every estimator."""
    with plt.rc_context(REPORT_RC):
        fig, ax = plt.subplots(figsize=(SINGLE_W, 1.9))
        sets = [(dl, gl) for dl in disc_levels for gl in dag_levels]
        cmaps = ["Blues", "Reds", "Greens", "Purples"]
        for i, ((dl, gl), (mk, ls)) in enumerate(zip(sets, _styles(len(sets)))):
            j = dag_levels.index(gl)
            q = (res[(res.disc_level == dl) & (res.dag_level == gl) & (res.estimator == estimator)]
                 .groupby("sample_size").unseen_rate.mean())
            ax.plot(q.index, 100 * q, marker=mk, ls=ls, label=f"{dl}, {gl}",
                    color=_shade(disc_levels.index(dl), len(disc_levels), cmaps[j % len(cmaps)]))
        ax.set_xscale("log"); ax.set_xlabel("Training rows $N$")
        ax.set_ylabel("Unseen parent configurations (%)")
        # above the axes, one column per DAG confidence, so it never covers the lines
        h, l = ax.get_legend_handles_labels()
        order = [i * len(dag_levels) + j for j in range(len(dag_levels)) for i in range(len(disc_levels))]
        fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center",
                   ncol=len(dag_levels), frameon=False, bbox_to_anchor=(.5, 1.0))
        return _save(fig, out_path)


def _tabular_params(res, tabular=("mle", "dirichlet")):
    """Free parameters of the tabular CPTs per (disc_level, dag_level)."""
    t = res[res.estimator.isin(tabular)]
    return t.groupby(["disc_level", "dag_level"])["free_params"].max()


def plot_real_params(res, estimators, out_path=None):
    """Free parameters of each estimator against the size of the tabular CPTs,
    one point per (discretisation, DAG). The tabular estimators (MLE, Dirichlet)
    have exactly the CPT's count, so they are the diagonal rather than points."""
    tab = _tabular_params(res)
    with plt.rc_context(REPORT_RC):
        fig, ax = plt.subplots(figsize=(SINGLE_W, 2.4))
        for i, e in enumerate(estimators):
            p = (res[res.estimator == e].groupby(["disc_level", "dag_level"])["free_params"].median()
                 .rename("params").to_frame().join(tab.rename("tabular")).sort_values("tabular"))
            if np.allclose(p["params"], p["tabular"]):
                continue
            c, fmt, lab = _est_style(e, i)
            # markers only: the settings are separate points, not a series
            ax.plot(p["tabular"], p["params"], fmt.rstrip("-.:"), color=c, label=lab, ls="none")
        lo, hi = tab.min() / 1.5, tab.max() * 1.5
        ax.plot([lo, hi], [lo, hi], ":", color=REPORT_COLORS["mle"], lw=1, label="CPT (MLE, Dirichlet)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Free parameters of the tabular CPTs"); ax.set_ylabel("Free parameters")
        _legend_top(fig, ax, 3, 1.13)
        return _save(fig, out_path)


def plot_real_timing(timing, estimators, panels, out_path=None):
    """Serial fit time against N, one panel per (discretisation, DAG), one line
    per estimator. timing: rows of the serial timing run."""
    tab = _tabular_params(timing)
    with plt.rc_context(REPORT_RC):
        fig, axs = plt.subplots(1, len(panels), figsize=(DOUBLE_W, 1.9), sharey=True, squeeze=False)
        for ax, (dl, gl) in zip(axs[0], panels):
            sub = timing[(timing.disc_level == dl) & (timing.dag_level == gl)]
            for i, e in enumerate(estimators):
                c, fmt, lab = _est_style(e, i)
                q = sub[sub.estimator == e].groupby("sample_size").fit_time_s.mean()
                ax.plot(q.index, q, fmt, color=c, label=lab)
            ns = sorted(sub.sample_size.unique())
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xticks(ns); ax.set_xticklabels([_n_label(n) for n in ns]); ax.minorticks_off()
            ax.set_xlabel("Training rows $N$")
            ax.set_title(f"{dl}, {gl} ({tab.get((dl, gl), float('nan')):,.0f} tabular params)", loc="left")
        axs[0, 0].set_ylabel("Fit time (s, serial)")
        _legend_top(fig, axs[0, 0], len(estimators), 1.1)
        return _save(fig, out_path)
