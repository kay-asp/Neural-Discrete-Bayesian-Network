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
    """Filter df to fixed values for every variable that isn't x or hue."""
    sub = df.copy()
    skip = {x, hue}

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

    # Build subtitle showing what's held constant
    skip = {x, hue}
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


def load_trials(results_dir):
    """All hyperparameter-sweep trial rows in results_dir: hparam_trials.csv
    (every sweep) plus any older hparam_trials_*.csv files, concatenated."""
    from glob import glob
    import os
    files = sorted(glob(os.path.join(results_dir, "hparam_trials*.csv")))
    if not files:
        raise FileNotFoundError(f"no hparam_trials*.csv in {results_dir}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


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


def _shared_value(top, c):
    """The value of column c shared by every row of `top` as text, or "–" if
    the rows differ. lr / weight_decay are rounded to 1 significant figure
    first and shown like run names ("≈3e-4")."""
    vals = top[c].map(_round_sig) if c in _ROUNDED else top[c]
    if vals.nunique(dropna=False) != 1:
        return "–"
    v = vals.iloc[0]
    if c in _ROUNDED:
        mantissa, exponent = f"{v:.0e}".split("e")
        return f"≈{mantissa}e{int(exponent)}"
    return str(v)


def _group_trials(s, cols):
    """Average trials (rows of s, indexed by run_id) whose settings `cols` are
    identical (after summarise_trials' rounding / snapping / fixed values):
    one row per distinct config. mean_kl / mean_val_loss / extras and kl_se
    are means, n_trials counts the group, run_name / run_id are the group's
    best trial. kl_se is the mean of the trials' kl_se, not sqrt(sum se^2)/n:
    every trial is evaluated on the same DAGs and test rows, so their errors
    are strongly correlated and averaging removes far less noise than
    independent errors would (the true SE of the average lies between
    se / sqrt(n) and se). Using the mean keeps grouped rows' within-n_se
    window comparable to single trials'."""
    s = s.reset_index()
    rows = []
    for _, g in s.groupby(cols, dropna=False, sort=False):
        best = g.loc[g["mean_kl"].idxmin()]
        row = {"run_id": best["run_id"], "run_name": best["run_name"],
               "mean_kl": g["mean_kl"].mean(),
               "kl_se": g["kl_se"].mean(),     # errors are correlated; see docstring
               "mean_val_loss": g["mean_val_loss"].mean(),
               "n_dags": best["n_dags"], "n_trials": len(g),
               **{c: best[c] for c in cols}}
        for c in ["stored_params", "best_epoch_mean", "fit_time_s"]:
            if c in g:
                row[c] = g[c].mean()
        if "sweep_id" in g:
            row["sweep_id"] = ", ".join(map(str, dict.fromkeys(g["sweep_id"].dropna()))) or np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("run_id")


def summarise_trials(trials, n_se=1.0, fixed=None, snap=(), show=True, **filters):
    """Per (in_degree, sample_size): the trials within n_se kl_se of the best.

    trials  : the per-cell trial CSV(s) (one row per trial x dag x sample size).
    filters : any column = value, e.g. card=2, alpha=0.5, sweep_id="abc123".
              Rows missing a filtered setting (older CSVs) use TRIAL_DEFAULTS,
              or are dropped if it has no default.
    For each (in_degree, sample_size), every trial gets mean_kl over its DAGs
    and kl_se = sqrt(sum of per-cell kl_se^2) / n_cells, the standard error of
    that mean from the test-row Monte Carlo error. Trials with fewer DAGs than
    the most complete one in that pair are dropped. The best trial is the
    lowest mean_kl; a trial is kept if mean_kl <= best mean_kl + n_se * best kl_se.
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
    Prints one table per pair when show=True, and returns
    {(in_degree, sample_size): DataFrame}, best first. Printed tables round
    lr / weight_decay to 1 significant figure and hide
    run_id, n_dags, fixed settings and settings that are constant across all trials; the
    returned tables keep every column. Use shared_settings_table(tables) for
    the settings each table's rows have in common.
    """
    t = trials.copy()
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
    hidden = ["run_id", "n_dags", *fixed] + [c for c in [*hp, "sweep_id"]
                                             if c in t and t[c].nunique(dropna=False) <= 1]
    tables = {}
    for (k, n), g in t.groupby(["in_degree", "sample_size"]):
        s = g.groupby("run_id").agg(
            run_name=("run_name", "first"),
            mean_kl=("kl", "mean"),
            kl_se=("kl_se", lambda x: np.sqrt((x ** 2).sum()) / len(x)),
            mean_val_loss=("val_loss", "mean"),
            n_dags=("dag_idx", "nunique"),
            **{c: (c, "first") for c in hp if c in g},
            **{c: (c, "mean") for c in extra},
            **({"sweep_id": ("sweep_id", "first")} if "sweep_id" in g else {}))
        s = s[s["n_dags"] == s["n_dags"].max()]
        # tidy settings, then average trials that have become identical
        for c in hp:
            if c in fixed:
                s[c] = fixed[c]
            elif c in grids:
                s[c] = s[c].map(lambda x, m=grids[c]: _snap(x, m))
            elif c in _ROUNDED:
                s[c] = s[c].map(_round_sig)
        s = _group_trials(s, hp)
        s = s.sort_values("mean_kl")
        best = s.iloc[0]
        top = s[s["mean_kl"] <= best["mean_kl"] + n_se * best["kl_se"]].reset_index()
        top.attrs["hidden"] = hidden        # for shared_settings_table
        top.attrs["n_se"] = n_se            # threshold, reported downstream
        top.attrs["fixed"] = fixed           # top_config_values reuses these
        top.attrs["snap"] = tuple(snap)
        tables[(k, n)] = top
        if show:
            grouped = "".join(f" ({c} fixed at {v:g})" for c, v in fixed.items())
            grouped += "".join(f" ({c} snapped to {'-'.join(map(str, m))} grid)" for c, m in snap)
            print(f"in-degree {k}, N = {n}: best {best['mean_kl']:.4f} ± {best['kl_se']:.4f}"
                  f" -> {len(top)} of {len(s)} configs{grouped} within {n_se:g} kl_se")
            _show(_for_display(top.drop(columns=[c for c in hidden if c in top])))
    return tables


def _threshold(tables):
    """'within N kl_se' for the tables' threshold (as set in summarise_trials)."""
    n_se = next(iter(tables.values())).attrs.get("n_se", 1.0) if tables else 1.0
    return f"within {n_se:g} kl_se"


def shared_settings_table(tables, show=True):
    """One row per summarise_trials table: the settings every one of its top
    configs has in common.

    Columns: in_degree, sample_size, n_top, then one per setting. A cell holds
    the value all of that table's top configs share ("–" if they differ;
    lr / weight_decay rounded to 1 significant figure, "≈3e-4"). Settings
    that never vary across all trials are left out. Read down a column to
    see whether a setting agrees across in-degrees and sample sizes.
    """
    if not tables:
        return pd.DataFrame()
    hidden = set(next(iter(tables.values())).attrs.get("hidden", []))
    cols = [c for c in _HPARAM_COLS
            if c not in hidden and all(c in top for top in tables.values())]
    rows = [{"in_degree": k, "sample_size": n, "n_top": len(top),
             **{c: _shared_value(top, c) for c in cols}}
            for (k, n), top in tables.items()]
    out = pd.DataFrame(rows)
    if show:
        print(f"settings shared by all top configs ({_threshold(tables)}) of each table "
              "(– = they differ; n_top = 1 means a single config)")
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


def top_config_values(tables, snap=None, fixed=None, show=True):
    """Every top config (from summarise_trials) once, with its settings.

    lr and weight_decay are rounded to 1 significant figure (0.000302 and
    0.00031 -> 0.0003), except columns listed in snap, which are snapped to a
    grid instead: snap=(("weight_decay", (1, 5)),) puts weight decay on
    ..., 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, ... (nearest in log space, so 0.005 and
    0.006 both become 0.005). snap=() uses plain rounding. Configs that become
    identical are merged into one row.
    fixed : settings to use for every config, e.g. {"weight_decay": 0.01} when
    summarise_trials grouped over weight decay; they're added to (or replace
    the value in) every returned row, so arms_from_configs runs them, and are
    hidden in the printed table.
    Columns: run_name / run_id of the merged row's best trial (lowest
    mean_kl relative to its table's best; used to name the arm), in_tables
    (every in-degree / sample size table any of its trials is top in),
    n_tables, n_trials (sweep trials merged into the row), then every
    setting. A row is what's needed to run that config again (see
    arms_from_configs), with the rounded / snapped values.
    """
    first = next(iter(tables.values())) if tables else pd.DataFrame()
    # default to the fixed values / snapping summarise_trials used
    grids = dict(first.attrs.get("snap", ()) if snap is None else snap)
    fixed = dict(first.attrs.get("fixed", {}) if fixed is None else fixed)
    rows = {}
    for (k, n), top in tables.items():
        cols = [c for c in _HPARAM_COLS if c in top]
        best_kl = top["mean_kl"].min()
        for _, r in top.iterrows():
            settings = {c: (_snap(r[c], grids[c]) if c in grids
                            else _round_sig(r[c]) if c in _ROUNDED else r[c]) for c in cols}
            settings.update(fixed)
            key = tuple(str(v) for v in settings.values())
            regret = r["mean_kl"] / best_kl - 1
            entry = rows.setdefault(key, {"run_id": r["run_id"], "run_name": r["run_name"],
                                          "_regret": regret, "tables": [], "trials": {},
                                          **settings})
            if regret < entry["_regret"]:      # label the row with its best trial
                entry.update(run_id=r["run_id"], run_name=r["run_name"], _regret=regret)
            if f"k{k} N{n}" not in entry["tables"]:
                entry["tables"].append(f"k{k} N{n}")
            # sweep trials behind this row (grouped rows already average several)
            entry["trials"][r["run_id"]] = int(r["n_trials"]) if "n_trials" in r else 1
    out = pd.DataFrame(list(rows.values()))
    if out.empty:
        return out
    out = out.drop(columns="_regret")
    out.insert(2, "n_trials", out.pop("trials").map(lambda d: sum(d.values())))
    out.insert(2, "n_tables", out["tables"].map(len))
    out.insert(2, "in_tables", out.pop("tables").map(", ".join))
    out = (out.sort_values(["n_tables", "n_trials", "run_name"], ascending=[False, False, True])
           .reset_index(drop=True))
    if show:
        how = [f"{c} snapped to {'-'.join(map(str, g))} grid" for c, g in grids.items()]
        how += [f"{c} rounded to 1 significant figure" for c in _ROUNDED
                if c not in grids and c not in fixed]
        how += [f"{c} fixed at {v:g} for runs" for c, v in fixed.items()]
        print(f"top configs ({_threshold(tables)}), {'; '.join(how)}, identical configs merged:")
        # optimizer / n_epochs stay in the returned table (arms_from_configs needs them)
        _show(out.drop(columns=[c for c in ["run_id", "optimizer", "n_epochs", *fixed]
                                if c in out]))
    return out


def candidate_table(trials, source="candidates", n_se=2.0, tables=None, show=True):
    """Compare chosen configs run on every (in-degree, sample size).

    trials : the tuning data (load_trials(RESULTS_DIR)); the rows with
             source=`source` (written by experiments.run_configs) are used,
             labelled by run_name. Duplicates from re-running are dropped,
             keeping the latest.
    One row per config, one column per (k, N) ("k6 N600"): the config's mean
    KL over the tuning DAGs. In the printed table, a value is bold if it is
    within n_se standard errors of the best result *overall* for that (k, N),
    i.e. among all tuning trials (sweeps and candidates): best mean KL +
    n_se * its kl_se. That best comes from `tables` (summarise_trials output,
    so it matches the top-config tables, including FIXED / SNAP grouping) if
    given, otherwise from every complete trial in `trials`, with
    kl_se = sqrt(sum of the DAGs' kl_se^2) / n (the DAGs have separate test
    sets). Rows are sorted by how many columns they are bold in, then by
    average rank. Also listed per config: dropout, patience, batch_size,
    fit_time_s (mean seconds per fit) and stored_params per in-degree
    ("params k6"), since the input layer grows with the number of parents.
    Returns the table.
    """
    if "source" not in trials:
        return pd.DataFrame()
    r = trials[trials["source"] == source]
    if r.empty:
        return pd.DataFrame()
    r = r.drop_duplicates(["run_name", "in_degree", "sample_size", "dag_idx"], keep="last")
    cols = ["in_degree", "sample_size"]
    kl = r.pivot_table(index="run_name", columns=cols, values="kl", aggfunc="mean")
    se = r.pivot_table(index="run_name", columns=cols, values="kl_se",
                       aggfunc=lambda x: np.sqrt((x ** 2).sum()) / len(x))
    # threshold per (k, N): best result overall, from all tuning trials
    limit = {}
    for c in kl.columns:
        if tables is not None and c in tables and len(tables[c]):
            b = tables[c].iloc[0]                         # summarise_trials: best first
            limit[c] = b["mean_kl"] + n_se * b["kl_se"]
            continue
        g = trials[(trials["in_degree"] == c[0]) & (trials["sample_size"] == c[1])]
        per = g.groupby("run_id").agg(
            mean_kl=("kl", "mean"), n=("dag_idx", "nunique"),
            kl_se=("kl_se", lambda x: np.sqrt((x ** 2).sum()) / len(x)))
        per = per[per["n"] == per["n"].max()]             # complete trials only
        b = per.loc[per["mean_kl"].idxmin()]
        limit[c] = b["mean_kl"] + n_se * b["kl_se"]
    near = pd.DataFrame({c: kl[c] <= limit[c] for c in kl.columns})
    order = (pd.DataFrame({"n": near.sum(axis=1), "rank": kl.rank().mean(axis=1)})
             .sort_values(["n", "rank"], ascending=[False, True]).index)
    label = lambda kn: f"k{kn[0]} N{kn[1]}"
    out = kl.loc[order].set_axis([label(c) for c in kl.columns], axis=1)
    kl_cols = list(out.columns)
    near = near.loc[order].set_axis(kl_cols, axis=1)
    # settings and cost, per config
    by = r.groupby("run_name")
    for c in ("dropout", "patience", "batch_size"):
        if c in r:
            out[c] = by[c].first()
    if "fit_time_s" in r:
        out["fit_time_s"] = by["fit_time_s"].mean()      # mean seconds per fit
    if "stored_params" in r:                              # depends on in-degree (input size)
        params = r.pivot_table(index="run_name", columns="in_degree", values="stored_params",
                               aggfunc="first")
        for k in params.columns:
            out[f"params k{k}"] = params[k]
    out.index.name = "config"
    if show:
        print(f"chosen configs on every (in-degree, sample size): mean KL over the tuning DAGs; "
              f"bold = within {n_se:g} SE of the best result overall (all tuning trials) "
              f"for that in-degree and sample size")
        try:
            fmt = {c: "{:.4f}" for c in kl_cols}
            fmt.update({c: "{:.0f}" for c in out.columns if c.startswith("params")
                        or c in ("patience", "batch_size")})
            fmt.update({"dropout": "{:.1f}", "fit_time_s": "{:.1f}"})
            styled = out.style.format({c: f for c, f in fmt.items() if c in out}).apply(
                lambda col: ["font-weight: bold" if b else "" for b in near[col.name]],
                subset=kl_cols)
            _show(styled)
        except ImportError:                              # no jinja2: mark with * instead
            marked = out.round(4).astype(str)
            marked[kl_cols] = marked[kl_cols].where(~near, marked[kl_cols] + " *")
            _show(marked)
    return out.reset_index()


