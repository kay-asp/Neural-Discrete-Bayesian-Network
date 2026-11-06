"""KL divergence against the ground truth, parameter counts, and fit timing."""
import os
import time

import numpy as np
import psutil

from .models import Cards, nn_probs

EPS = 1e-12


def cpd_probs(cpd, test_data):
    """P(node | pa) for every row, using the CPD's own parent order -> (N, card)."""
    parents = list(cpd.variables[1:])
    values = cpd.get_values()                       # (card, n_configs)
    if not parents:
        return np.repeat(values.T, len(test_data), axis=0)
    cols = test_data[parents].to_numpy().astype(np.int64).T
    idx = np.ravel_multi_index(cols, list(cpd.cardinality[1:]))
    return values[:, idx].T


def kl_calculation(true_model, test_data, q_probs, return_nodes=False, eps=EPS):
    """MC estimate of D_KL(P* || Q), decomposed by node.

    q_probs : (node, test_data) -> (N, card) array of Q(X_i | pa_i) per row.
    Returns (total, se), plus, with return_nodes=True, node -> [conditional KL
    term, number of parents]: each node's term in the chain-rule decomposition
    (the terms sum to the total). eps : floor on Q, so a zero probability where
    P* is positive (unsmoothed MLE) costs log(1/eps) rather than infinity.
    """
    rows = np.zeros(len(test_data))
    nodes = {}

    for cpd in true_model.get_cpds():
        node = cpd.variable
        p = cpd_probs(cpd, test_data)
        q = q_probs(node, test_data)

        if np.isnan(q).any():
            raise ValueError(f"Q has NaN for node {node} (unseen parent config?)")

        k = (p * (np.log(np.clip(p, EPS, None)) - np.log(np.clip(q, eps, None)))).sum(axis=1)
        rows += k
        nodes[node] = [float(k.mean()), len(cpd.variables) - 1]

    n = len(rows)
    out = (float(rows.mean()), float(rows.std(ddof=1) / np.sqrt(n)))
    return out + (nodes,) if return_nodes else out

def kl_mle(true_model, mle_model, test_data, return_nodes=False, eps=EPS):
    return kl_calculation(
        true_model, test_data,
        q_probs=lambda node, td: cpd_probs(mle_model.get_cpds(node), td),
        return_nodes=return_nodes, eps=eps,
    )


def kl_nn(true_model, nets, test_data, cards, return_nodes=False):
    cards = Cards.coerce(cards, nets.nodes())
    return kl_calculation(
        true_model, test_data,
        q_probs=lambda node, td: nn_probs(nets[node], td, cards),
        return_nodes=return_nodes,
    )


def unseen_config_share(G, train_data, cards):
    """Share of all parent configurations, summed over every node with parents,
    that never occur in train_data. cards : an int (shared) or node -> cardinality."""
    cards = Cards.coerce(cards, G.nodes())
    seen = total = 0
    for v in G.nodes():
        pa = sorted(G.predecessors(v))
        if not pa:
            continue
        total += int(np.prod([cards[p] for p in pa]))
        seen += len(train_data[pa].drop_duplicates())
    return 1 - seen / total if total else 0.0


def mle_param_stats(model):

    nodes = list(model.nodes())
    stats = {}
    for nd in nodes:
        vals = model.get_cpds(nd).get_values()      # (card, n_parent_configs)
        card, n_configs = vals.shape
        stats[nd] = {
            "stored_params": int(card * n_configs),
            "free_params":   int((card - 1) * n_configs),
            "memory_bytes":  int(vals.nbytes),
        }

    out = {f"{k}": sum(s[k] for s in stats.values())
           for k in ("stored_params", "free_params", "memory_bytes")}
    out["memory_bytes_native"] = out["memory_bytes"]
    return out


def timed_fit(fit_fn, *args, **kwargs):
    """Run fit_fn(*args, **kwargs) -> (model, wall_s, cpu_s, rss_delta_bytes).

    wall_s is elapsed time; cpu_s is this process's CPU time (all its threads),
    less sensitive to other processes competing for the machine. Neither is a
    clean benchmark under parallel runs; use a serial run for timing results.
    rss_delta is only meaningful averaged over repeats and is noisy; the
    analytical/nbytes parameter curve is the precise memory evidence.
    """
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss

    t0, c0 = time.perf_counter(), time.process_time()
    model = fit_fn(*args, **kwargs)
    fit_time, cpu_time = time.perf_counter() - t0, time.process_time() - c0

    rss_delta = proc.memory_info().rss - rss_before
    return model, fit_time, cpu_time, rss_delta
