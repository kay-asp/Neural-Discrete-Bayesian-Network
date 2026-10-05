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


def summarise_trials(trials, show=True, **filters):
    """Per (in_degree, sample_size): the trials within 1 kl_se of the best.

    trials  : the per-cell trial CSV(s) (one row per trial x dag x sample size).
    filters : any column = value, e.g. card=2, alpha=0.5, sweep_id="abc123".
              Rows missing a filtered setting (older CSVs) use TRIAL_DEFAULTS,
              or are dropped if it has no default.
    For each (in_degree, sample_size), every trial gets mean_kl over its DAGs
    and kl_se = sqrt(sum of per-cell kl_se^2) / n_cells, the standard error of
    that mean from the test-row Monte Carlo error. Trials with fewer DAGs than
    the most complete one in that pair are dropped. The best trial is the
    lowest mean_kl; a trial is kept if mean_kl <= best mean_kl + best kl_se.
    Prints one table per pair when show=True, and returns
    {(in_degree, sample_size): DataFrame}, best first.
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

    hp = [c for c in _HPARAM_COLS if c in t]
    extra = [c for c in ["stored_params", "best_epoch_mean", "fit_time_s"] if c in t]
    tables = {}
    for (k, n), g in t.groupby(["in_degree", "sample_size"]):
        s = g.groupby("run_id").agg(
            run_name=("run_name", "first"),
            mean_kl=("kl", "mean"),
            kl_se=("kl_se", lambda x: np.sqrt((x ** 2).sum()) / len(x)),
            mean_val_loss=("val_loss", "mean"),
            n_dags=("dag_idx", "nunique"),
            **{c: (c, "first") for c in hp},
            **{c: (c, "mean") for c in extra},
            **({"sweep_id": ("sweep_id", "first")} if "sweep_id" in g else {}))
        s = s[s["n_dags"] == s["n_dags"].max()].sort_values("mean_kl")
        best = s.iloc[0]
        top = s[s["mean_kl"] <= best["mean_kl"] + best["kl_se"]].reset_index()
        tables[(k, n)] = top
        if show:
            print(f"in-degree {k}, N = {n}: best {best['mean_kl']:.4f} ± {best['kl_se']:.4f}"
                  f" -> {len(top)} of {len(s)} configs within 1 kl_se")
            _show(top.round(4))
    return tables


def common_configs(tables, match_on=("hidden_dims", "dropout", "patience", "batch_size"),
                   round_cols=("lr", "weight_decay"), min_tables=2, show=True):
    """Configs in the top (within 1 kl_se) of more than one summarise_trials table.

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
        print(f"configs (matched on {', '.join(match_on)}) in the top of "
              f">= {min_tables} of {len(tables)} tables: {len(out)}")
        _show(out)
    return out
