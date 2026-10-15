"""
RQ1 plotting — single-variable-at-a-time design (no density parameter).

Every plot fixes all experimental variables except x and hue.
Change the DEFAULT_* values (e.g. `plotting.DEFAULT_ALPHA = 0.1`) to control
which slice you're viewing.
"""

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

# ── Palette ──────────────────────────────────────────────────────────────────

_PALETTE = ["#2563eb", "#e11d48", "#16a34a", "#d97706",
            "#7c3aed", "#0891b2", "#be185d", "#4d7c0f"]


def _colors(n):
    if n <= len(_PALETTE):
        return _PALETTE[:n]
    cmap = plt.get_cmap("viridis")
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


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
    n_nodes=None, in_degree=None, estimator=None, card=None,
    # plot styling
    logy=False, err=None, min_obs=3,
    xlabel=None, ylabel=None, title=None, hue_label=None,
    out_path=None, ax=None, figsize=(7, 5),
):
    """Mean y vs x, one curve per hue value, all other variables held fixed."""
    if alpha is None:        alpha = DEFAULT_ALPHA
    if sample_size is None:  sample_size = DEFAULT_SAMPLE_SIZE
    if n_nodes is None:      n_nodes = DEFAULT_N_NODES
    if in_degree is None:    in_degree = DEFAULT_IN_DEGREE
    if estimator is None:    estimator = DEFAULT_ESTIMATOR
    if card is None:         card = DEFAULT_CARD

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
    colors = _colors(len(hue_vals))

    for hv, c in zip(hue_vals, colors):
        s = sub[sub[hue] == hv]
        g = s.groupby(x)[y].agg(["mean", "std", "sem", "count"]).reset_index()
        g = g[g["count"] >= min_obs]
        if g.empty:
            continue
        label = f"{hue_label or hue} = {hv}"
        if err in ("std", "sem"):
            ax.errorbar(g[x], g["mean"], yerr=g[err].fillna(0.0), marker="o",
                        lw=2, color=c, label=label, capsize=3, zorder=3)
        else:
            ax.plot(g[x], g["mean"], marker="o", lw=2, color=c,
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
    for c, col in zip(cards, _colors(len(cards))):
        ax.plot(ks, [8 * c ** (k + 1) for k in ks], marker="o", lw=2,
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


def _group_trials(s, cols):
    """Average trials (rows of s, indexed by run_id) whose settings `cols` are
    identical (after summarise_trials' rounding / snapping / fixed values):
    one row per distinct config. mean_kl / mean_val_loss / extras and kl_se
    are means over every run in the group; n_trials counts its distinct
    configs (config_ids: exact settings before tidying), so an exact re-run of
    a config (e.g. a candidate run) adds to the means but not to n_trials.
    config_id / run_name / run_id are the group's best trial, run_names / config_ids list
    every trial / config in it. kl_se is the mean of the trials' kl_se, not sqrt(sum se^2)/n:
    every trial is evaluated on the same DAGs and test rows, so their errors
    are strongly correlated and averaging removes far less noise than
    independent errors would (the true SE of the average lies between
    se / sqrt(n) and se). Using the mean keeps grouped rows' within-n_se
    window comparable to single trials'."""
    s = s.reset_index()
    rows = []
    for _, g in s.groupby(cols, dropna=False, sort=False):
        best = g.loc[g["mean_kl"].idxmin()]
        row = {"config_id": best["config_id"],      # the best trial's config
               "run_id": best["run_id"], "run_name": best["run_name"],
               "run_names": tuple(g.sort_values("mean_kl")["run_name"]),   # every trial merged
               "config_ids": tuple(dict.fromkeys(g.sort_values("mean_kl")["config_id"])),
               # name of each config's best trial, for labelling a config picked by id
               "id_names": dict(g.sort_values("mean_kl", ascending=False)
                                [["config_id", "run_name"]].to_numpy()),
               "mean_kl": g["mean_kl"].mean(),
               "kl_se": g["kl_se"].mean(),     # errors are correlated; see docstring
               "mean_val_loss": g["mean_val_loss"].mean(),
               "n_dags": best["n_dags"], "n_trials": g["config_id"].nunique(),
               **{c: best[c] for c in cols}}
        for c in ["stored_params", "best_epoch_mean", "fit_time_s"]:
            if c in g:
                row[c] = g[c].mean()
        if "sweep_id" in g:
            row["sweep_id"] = ", ".join(map(str, dict.fromkeys(g["sweep_id"].dropna()))) or np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("run_id")


def summarise_trials(trials, n_se=1.0, fixed=None, snap=(), rel_tol=0.0, max_params=None,
                     metric="kl", show=True, **filters):
    """Per (in_degree, sample_size): the trials within n_se kl_se (or rel_tol) of the best.

    metric  : "kl" (default) selects on mean_kl as described below.
              "val_loss" selects on mean_val_loss instead, with the same window
              in nats (n_se * kl_se or rel_tol * KL of the best-KL config):
              val loss = entropy (the same for every config: same data) + KL,
              so differences in the two estimate the same thing. Use it to see
              which configs choosing on val loss would pick, next to the KL
              tables. The printed tables then bold mean_kl where it is within
              the KL threshold (and the KL tables bold mean_val_loss the same
              way).

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
              configs whose architecture never ran there are dropped. The best
              (and so the threshold) is still taken over ALL configs, so a table
              only keeps configs under the cap that are as good as the best
              found (as in the one-standard-error rule); a table where none is
              is left out, with a message saying how far the best one under
              the cap is from the best.
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
    tables     : {(in_degree, sample_size): DataFrame}, sorted by n_trials
                 (most first), then the metric. Printed tables are indexed by
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
        s = _group_trials(s, hp)       # a grouped row's runs share hidden_dims, so its
                                       # best run tells whether it is under the cap
        # window in nats, from the best-KL config over ALL configs (also those over the
        # cap): n_se of its kl_se or rel_tol of its KL, whichever is looser; used for KL and
        # for val loss alike (val loss = entropy, the same for all configs, + KL)
        best_kl = s.loc[s["mean_kl"].idxmin()]
        window = max(n_se * best_kl["kl_se"], rel_tol * best_kl["mean_kl"])
        limits = {c: s[c].min() + window for c in ("mean_kl", "mean_val_loss")}
        s = s.sort_values(m)
        best = s.iloc[0]               # best on the metric, over ALL configs
        s_under = s[s.index.isin(under)]
        top = (s_under[s_under[m] <= limits[m]].reset_index()
               .sort_values(["n_trials", m], ascending=[False, True], kind="stable")
               .reset_index(drop=True))
        name = "KL" if metric == "kl" else "val loss"
        head = (f"in-degree {k}, N = {n}: best {name} {best[m]:.4f}"
                + (f" ± {best['kl_se']:.4f}" if metric == "kl" else f" (window {window:.4f} nats)"))
        if top.empty:
            if show:
                worse = (f"{s_under[m].iloc[0] / best[m] - 1:.1%}" if metric == "kl" else
                         f"{s_under[m].iloc[0] - best[m]:.4f} nats")  # val loss: % is meaningless
                gap = ("no config is under the cap" if s_under.empty else
                       f"the best one under the cap ({s_under[m].iloc[0]:.4f}) is {worse} worse")
                print(f"{head} -> no config under the size cap is within tolerance; {gap}")
            continue
        top.attrs["hidden"] = hidden        # for shared_settings_table
        top.attrs["n_se"] = n_se            # threshold, reported downstream
        top.attrs["rel_tol"] = rel_tol
        top.attrs["max_params"] = dict(max_params or {})
        top.attrs["metric"] = metric
        tables[(k, n)] = top
        if show:
            grouped = "".join(f" ({c} fixed at {v:g})" for c, v in fixed.items())
            grouped += "".join(f" ({c} snapped to {'-'.join(map(str, g_))} grid)"
                               for c, g_ in snap)
            print(head + (" (over all configs, also those over the size cap)" if caps else "")
                  + f" -> {len(top)} of {len(s_under)} configs{grouped} {_threshold(tables)}")
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
    text = f"within {a.get('n_se', 1.0):g} kl_se"
    if a.get("rel_tol"):
        text += f" or {a['rel_tol']:.0%}"
    if a.get("metric") == "val_loss":
        text += " (as nats) of the best val loss"
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
    """Configs in the top (within n_se kl_se, as set in summarise_trials) of
    more than one summarise_trials table.

    tables     : the dict returned by summarise_trials.
    match_on   : setting columns that define "the same config". Continuous
                 sampled values never repeat exactly, so the default uses the
                 discrete settings; add e.g. "lr" to also match on it.
    round_cols : columns rounded to 1 significant figure before matching, if
                 they are in match_on (3.2e-4 and 2.8e-4 both become 3e-4).
    Returns one row per config in >= min_tables tables: its settings,
    n_tables, top_in (which in-degree / sample size tables) and n_trials.
    Mean KL is not compared across tables: its scale differs with k and N.
    """
    match_on = list(match_on)
    rows = []
    for (k, n), top in tables.items():
        for _, r in top.iterrows():
            key = {c: (_round_sig(r[c]) if c in round_cols else r[c]) for c in match_on}
            rows.append({**key, "table": f"k{k} N{n}", "run_id": r["run_id"]})
    if not rows:
        return pd.DataFrame(columns=[*match_on, "n_tables", "top_in", "n_trials"])
    df = pd.DataFrame(rows)
    out = (df.groupby(match_on, dropna=False)
           .agg(n_tables=("table", "nunique"),
                top_in=("table", lambda x: ", ".join(dict.fromkeys(x))),
                n_trials=("run_id", "nunique"))
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
    mean_kl relative to its table's best; used to name the arm), in_tables
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
        best_kl = top["mean_kl"].min()
        for _, r in top.iterrows():
            settings = {c: r[c] for c in cols}
            key = tuple(str(v) for v in settings.values())
            regret = r["mean_kl"] / best_kl - 1
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
                    show=True):
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
             params recorded there are dropped.
    One column per (k, N) ("k6 N600"): the config's mean KL over the tuning
    DAGs. In the printed table, a value is bold if it is close to the best
    candidate for that (k, N) (over all candidates, also any max_params
    removes, so the size cap never moves the bar): within n_se of
    the best's kl_se (statistically indistinguishable; kl_se = sqrt(sum of the
    DAGs' kl_se^2)/n, the DAGs have separate test sets) or within rel_tol of
    the best's KL (KL <= best * (1 + rel_tol): practically negligible, e.g.
    rel_tol=0.02 for 2%), whichever is looser. Then one column per (k, N) of
    mean validation loss ("val k6 N600"), bold if within the same window in nats
    of the best val loss: val loss = entropy + KL with the entropy shared by all
    configs (same data), so differences estimate the same thing. Compare the
    two bold patterns to see whether choosing on val loss would give the same
    choice. Rows are sorted by average KL rank over the (k, N) columns (rank 1
    = lowest KL in a column; columns a config has no runs in are skipped), best
    first. Also listed per config: hidden_dims, lr, dropout,
    patience, batch_size, fit_time_s (mean seconds per fit) and stored_params
    per in-degree ("params k6", mean over the tuning DAGs), since the input
    layer grows with the number of parents. Returns the table.
    """
    if "source" not in trials:
        return pd.DataFrame()
    r = trials[trials["source"] == source].copy()
    if r.empty:
        return pd.DataFrame()
    r["_key"] = add_config_ids(r)["config_id"]          # identical settings = one config
    r = r.drop_duplicates(["_key", "in_degree", "sample_size", "dag_idx"], keep="last")
    cols = ["in_degree", "sample_size"]
    kl = r.pivot_table(index="_key", columns=cols, values="kl", aggfunc="mean")
    se = r.pivot_table(index="_key", columns=cols, values="kl_se",
                       aggfunc=lambda x: np.sqrt((x ** 2).sum()) / len(x))
    params = (r.pivot_table(index="_key", columns="in_degree", values="stored_params",
                            aggfunc="mean")       # depends on in-degree (input size)
              if "stored_params" in r else pd.DataFrame(index=kl.index))
    # threshold per (k, N), from the best of ALL candidates (also those over the size
    # cap): within n_se of its kl_se, or within rel_tol of its KL, whichever is looser
    limit, best_kl = {}, {}
    for c in kl.columns:
        best = kl[c].idxmin()
        best_kl[c] = kl.at[best, c]
        limit[c] = max(kl.at[best, c] + n_se * se.at[best, c], kl.at[best, c] * (1 + rel_tol))
    # validation loss = entropy (the same for every config: same data) + KL, so a
    # difference in val loss estimates the same as one in KL: give it the KL window in
    # nats, from its own best (over all candidates too)
    vl = (r.pivot_table(index="_key", columns=cols, values="val_loss", aggfunc="mean")
          if "val_loss" in r else None)
    val_limit = ({c: vl[c].min() + (limit[c] - best_kl[c]) for c in kl.columns}
                 if vl is not None else {})
    if max_params:      # then drop the configs over the cap
        keep = pd.Series(True, index=kl.index)
        for k, cap in max_params.items():
            if k not in params:
                raise KeyError(f"no stored_params recorded for in-degree {k}")
            keep &= (params[k].reindex(kl.index) < cap).fillna(False)
        kl, se = kl[keep], se.reindex(kl.index)[keep]
        if kl.empty:
            print(f"no candidates with max_params={max_params}")
            return pd.DataFrame()
    near = pd.DataFrame({c: kl[c] <= limit[c] for c in kl.columns})

    order = kl.rank().mean(axis=1).sort_values(kind="stable").index   # average KL rank, best first

    label = lambda kn: f"k{kn[0]} N{kn[1]}"
    out = kl.loc[order].set_axis([label(c) for c in kl.columns], axis=1)
    kl_cols = list(out.columns)
    near = near.loc[order].set_axis(kl_cols, axis=1)
    # mean validation loss per (k, N) (the validation-only view; scale differs with k and N),
    # bold within the KL window carried over in nats (see val_limit)
    val_cols, near_val = [], pd.DataFrame(index=order)
    if vl is not None:
        for c in kl.columns:
            out[f"val {label(c)}"] = vl[c]
            near_val[f"val {label(c)}"] = vl[c].reindex(order) <= val_limit[c]
        val_cols = list(near_val.columns)
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
        print(f"chosen configs on every (in-degree, sample size): mean KL over the tuning DAGs; "
              f"bold = within {n_se:g} SE or {rel_tol:.0%} of the best candidate "
              f"for that in-degree and sample size; then mean validation loss (val k N), "
              f"bold = within the same window (in nats) of the best val loss"
              + (f" (only configs with total params {max_params} shown; the best is over "
                 f"all candidates)" if max_params else ""))
        for c in kl.columns:
            if max_params and not (kl[c] <= limit[c]).any():   # near is relabelled by now
                print(f"k{c[0]} N{c[1]}: no candidate under the size cap is within tolerance; the "
                      f"best one under it ({kl[c].min():.4f}) is {kl[c].min() / best_kl[c] - 1:.1%} "
                      f"worse than the best ({best_kl[c]:.4f})")
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
    for est, c in zip(ests, _colors(len(ests))):
        ax.scatter(gain["n_parent_configs"], gain[est], s=18, alpha=0.6, color=c, label=est)
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
