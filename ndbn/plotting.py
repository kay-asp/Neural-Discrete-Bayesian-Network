"""
RQ1 plotting — single-variable-at-a-time design (no density parameter).

Every plot fixes all experimental variables except x and hue.
Change the DEFAULT_* values (e.g. `plotting.DEFAULT_ALPHA = 0.1`) to control
which slice you're viewing.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── Defaults: the "held constant" values for each variable ───────────────────

DEFAULT_SAMPLE_SIZE  = 1000
DEFAULT_ALPHA        = 0.5
DEFAULT_N_NODES      = 20
DEFAULT_IN_DEGREE    = 4
DEFAULT_ESTIMATOR    ="mle"

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
            n_nodes=None, in_degree=None, estimator=None):
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

    return sub


# ── Main plotting function ──────────────────────────────────────────────────

def plot_grouped_curves(
    df, x, hue, y="kl",
    # fixed-variable overrides (None → use DEFAULT_*)
    alpha=None, sample_size=None,
    n_nodes=None, in_degree=None, estimator=None,
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

    sub = _filter(df, x, hue,
                  alpha=alpha, sample_size=sample_size,
                  n_nodes=n_nodes, in_degree=in_degree, estimator=estimator)

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
