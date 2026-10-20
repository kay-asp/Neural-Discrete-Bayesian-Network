"""Real data: CDC Diabetes Health Indicators (UCI 891, BRFSS 2015) on the
expert DIABETES DAGs of Zahoor, Constantinou, Curtis & Hasanuzzaman (2025).

No ground truth here, so held-out log-likelihood replaces KL.
"""
import json
import os
from functools import partial

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm.auto import tqdm

from .evaluation import EPS, cpd_probs
from .experiments import (NN_HPARAMS, _parallel_map, activation_name, estimator_config_id,
                          fit_estimator, flush, model_stats, run_settings)
from .generation import make_seed
from .models import Cards, NeuralCPDs, nn_probs
from .training import split_train_val

TARGET = "Diabetes_binary"

# ---- data -------------------------------------------------------------------

def load_cdc(cache_path="data/cdc_diabetes.csv"):
    """The 253,680 x 22 raw table (target first), fetched once from UCI and
    cached to `cache_path`."""
    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    from ucimlrepo import fetch_ucirepo
    r = fetch_ucirepo(id=891)
    df = pd.concat([r.data.targets, r.data.features], axis=1)
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


# ---- discretisation ---------------------------------------------------------

BINARY = ["Diabetes_binary", "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
          "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
          "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"]

# native code range of each categorical column; "identity" maps lo..hi -> 0..hi-lo,
# so a column's cardinality never depends on which rows are present
NATIVE_RANGE = {**{c: (0, 1) for c in BINARY},
                "Age": (1, 13), "Income": (1, 8), "Education": (1, 6), "GenHlth": (1, 5)}

# column -> "identity", "deciles", or right-open bin edges on the raw value
# (np.digitize: x < e0 -> 0, e0 <= x < e1 -> 1, ...)
DISCRETISATIONS = {
    # Zahoor et al. (2025) Table 1. MentHlth / PhysHlth: "0-9 days (0), 10-19
    # days (1) and >=20" (their text). Age / Income / Education / GenHlth
    # groupings are the ones that reproduce Table 1's marginals
    "L1": {**{c: "identity" for c in BINARY},
           "BMI":       [25, 40],              # 0-24, 25-39, >=40
           "MentHlth":  [10, 20],
           "PhysHlth":  [10, 20],
           "Age":       [3, 5, 7, 9, 11],      # {1-2, 3-4, 5-6, 7-8, 9-10, 11-13}
           "Income":    [3, 5, 7],             # {1-2, 3-4, 5-6, 7-8}
           "Education": [3, 5],                # {1-2, 3-4, 5-6}
           "GenHlth":   [3, 5]},               # {1-2, 3-4, 5}
    # native ordinal cardinalities
    "L2": {**{c: "identity" for c in BINARY},
           "BMI":       [18.5, 25, 30],        # WHO: under, normal, over, obese
           "MentHlth":  [1, 8, 15],            # 0, 1-7, 8-14, 15-30 days
           "PhysHlth":  [1, 8, 15],
           "Age": "identity", "Income": "identity",
           "Education": "identity", "GenHlth": "identity"},
    # fine
    "L3": {**{c: "identity" for c in BINARY},
           "BMI":       "deciles",
           "MentHlth":  [1, 3, 6, 11, 21],     # 0, 1-2, 3-5, 6-10, 11-20, 21-30
           "PhysHlth":  [1, 3, 6, 11, 21],
           "Age": "identity", "Income": "identity",
           "Education": "identity", "GenHlth": "identity"},
}


def _decile_edges(x):
    # interior decile cut points; ties in integer BMI can merge some
    return list(np.unique(np.quantile(x, np.linspace(0.1, 0.9, 9))))


def discretise(df, level):
    """Raw table -> (integer codes 0..r-1 as int8, Cards).

    "deciles" edges are taken from `df` itself: pass the full table, not a
    training subset, so every fold shares the same bins.
    """
    spec = DISCRETISATIONS[level]
    missing = set(df.columns) ^ set(spec)
    if missing:
        raise ValueError(f"discretisation {level} and data columns differ: {sorted(missing)}")
    out, cards = {}, Cards()
    for col in df.columns:
        x = df[col].to_numpy()
        rule = spec[col]
        if rule == "identity":
            lo, hi = NATIVE_RANGE[col]
            if x.min() < lo or x.max() > hi:
                raise ValueError(f"{col}: values outside native range {lo}..{hi}")
            codes, cards[col] = x - lo, hi - lo + 1
        else:
            edges = _decile_edges(x) if rule == "deciles" else rule
            codes, cards[col] = np.digitize(x, edges), len(edges) + 1
        out[col] = codes.astype(np.int8)
    return pd.DataFrame(out, index=df.index), cards


# Zahoor et al. (2025) Table 1: % of rows in each L1 state
TABLE1 = {
    "Diabetes_binary": [86, 14], "HighBP": [57, 43], "HighChol": [58, 42],
    "BMI": [28, 66, 6], "HeartDiseaseorAttack": [91, 9], "CholCheck": [4, 96],
    "Stroke": [96, 4], "Smoker": [56, 44], "Fruits": [37, 63], "Veggies": [19, 81],
    "HvyAlcoholConsump": [94, 6], "AnyHealthcare": [5, 95], "NoDocbcCost": [92, 8],
    "MentHlth": [87, 5, 7], "PhysHlth": [84, 5, 10], "DiffWalk": [83, 16],
    "Sex": [56, 44], "Age": [5, 10, 14, 23, 26, 22], "Income": [8, 14, 25, 53],
    "Education": [2, 29, 70], "GenHlth": [53, 42, 5], "PhysActivity": [25, 75],
}


def check_marginals(df_int, level="L1", tol=1.0):
    """Raise if any state's share differs from the published table by more
    than `tol` percentage points (Table 1 is in whole percent, and some rows
    sum to 99). Validates the whole data path before any model is fitted.
    Only L1 has published marginals."""
    if level != "L1":
        raise ValueError(f"no published marginals for {level}; only L1 (Zahoor et al. Table 1)")
    bad = []
    if set(df_int.columns) != set(TABLE1):
        bad.append(f"columns differ from Table 1: {sorted(set(df_int.columns) ^ set(TABLE1))}")
    for col, want in TABLE1.items():
        if col not in df_int:
            continue
        got = np.bincount(df_int[col], minlength=len(want)) / len(df_int) * 100
        if len(got) != len(want):
            bad.append(f"{col}: {len(got)} states, Table 1 has {len(want)}")
        elif np.any(np.abs(got - want) > tol):
            bad.append(f"{col}: got {np.round(got, 1).tolist()}, Table 1 {want}")
    if bad:
        raise ValueError("marginals don't match Zahoor et al. Table 1:\n  " + "\n  ".join(bad))


# ---- structure --------------------------------------------------------------

# Bayesys label -> UCI column, for labels that are pure renames. Filled in from
# the CSVs; a label that is a merged / derived variable must NOT go here.
NAME_MAP = {}


def load_bayesys_dag(path, columns, name_map=None):
    """A Bayesys DAG CSV (edge list) -> nx.DiGraph over exactly `columns`.

    Accepts Bayesys' "ID, Variable 1, Dependency, Variable 2" layout (only
    "->" rows are edges) or a plain two-column (from, to) list. Labels are
    mapped through name_map; any that still aren't data columns raise, so a
    merged or derived Bayesys variable is flagged rather than silently mapped.
    Nodes with no edges are added from `columns`.
    """
    name_map = NAME_MAP if name_map is None else name_map
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    if {"Variable 1", "Dependency", "Variable 2"} <= set(raw.columns):
        dep = raw["Dependency"].astype(str).str.strip()
        if not dep.isin(["->"]).all():
            raise ValueError(f"{path}: non-directed dependencies {sorted(set(dep) - {'->'})}")
        edges = raw[["Variable 1", "Variable 2"]].to_numpy()
    elif raw.shape[1] == 2:
        edges = raw.to_numpy()
    else:
        raise ValueError(f"{path}: unrecognised edge-list columns {list(raw.columns)}")

    edges = [(name_map.get(str(u).strip(), str(u).strip()),
              name_map.get(str(v).strip(), str(v).strip())) for u, v in edges]
    unknown = sorted({n for e in edges for n in e} - set(columns))
    if unknown:
        raise ValueError(f"{path}: labels not in the data (rename -> add to name_map; "
                         f"merged/derived -> flag): {unknown}")
    G = nx.DiGraph(edges)
    G.add_nodes_from(columns)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError(f"{path}: not acyclic, cycle {nx.find_cycle(G)}")
    assert set(G.nodes()) == set(columns)
    return G


def config_counts(G, cards):
    """Per node: in-degree, parent configurations (PRODUCT of parent cards),
    tabular free parameters, and NN one-hot input width (SUM of parent cards).
    A final "TOTAL" row sums free_params and nn_input_dim."""
    cards = Cards.coerce(cards, G.nodes())
    rows = []
    for v in G.nodes():
        pa = sorted(G.predecessors(v))
        n_cfg = int(np.prod([cards[p] for p in pa])) if pa else 1
        rows.append({"node": v, "card": cards[v], "in_degree": len(pa),
                     "n_parent_configs": n_cfg,
                     "free_params": (cards[v] - 1) * n_cfg,
                     "nn_input_dim": sum(cards[p] for p in pa) or 1})
    df = pd.DataFrame(rows).sort_values("n_parent_configs", ascending=False,
                                        ignore_index=True)
    total = {"node": "TOTAL", "card": np.nan, "in_degree": df["in_degree"].max(),
             "n_parent_configs": df["n_parent_configs"].sum(),
             "free_params": df["free_params"].sum(), "nn_input_dim": df["nn_input_dim"].sum()}
    return pd.concat([df, pd.DataFrame([total])], ignore_index=True)


# ---- scoring (replaces KL) --------------------------------------------------

def probs_fn(model, cards):
    """One interface for every estimator: (node, df) -> (N, cards[node]) array
    of q(node | its parents) for each row."""
    if isinstance(model, NeuralCPDs):
        cards = Cards.coerce(cards, model.nodes())
        return lambda node, df: nn_probs(model[node], df, cards)
    if isinstance(model, DiscreteBayesianNetwork):
        return lambda node, df: cpd_probs(model.get_cpds(node), df)
    raise TypeError(f"no probs_fn for {type(model).__name__}")


def held_out_logprob(model, test_df, cards, return_floored=False):
    """Mean over test rows of sum_i log q(x_i | pa_i) -> (mean, se, per_node).

    Same (value, se) shape as evaluation.kl_calculation. per_node: node -> mean
    log q over rows. Probabilities are floored at evaluation.EPS, so a zero
    (plain MLE: a child state never seen under that parent config) costs
    log(EPS) ~ -27.6 rather than -inf; with return_floored=True a 4th value,
    node -> number of floored rows, says how much of a score that is.
    """
    q_of = probs_fn(model, cards)
    rows = np.zeros(len(test_df))
    per_node, floored = {}, {}
    for node in test_df.columns:
        q = q_of(node, test_df)
        if np.isnan(q).any():
            raise ValueError(f"q has NaN for node {node}")
        qx = q[np.arange(len(test_df)), test_df[node].to_numpy(np.int64)]
        floored[node] = int((qx < EPS).sum())
        lp = np.log(np.clip(qx, EPS, None))
        per_node[node] = float(lp.mean())
        rows += lp
    out = (float(rows.mean()), float(rows.std(ddof=1) / np.sqrt(len(rows))), per_node)
    return out + (floored,) if return_floored else out


def unseen_config_rate(G, train_df, test_df):
    """How often a test row's parent configuration never occurred in training
    -> (overall, per_node). per_node: node -> fraction of test rows unseen for
    that node (0 for roots). overall: fraction of test rows unseen for at least
    one node. Where a CPT has no data, MLE falls back to uniform."""
    any_unseen = np.zeros(len(test_df), dtype=bool)
    per_node = {}
    for v in G.nodes():
        pa = sorted(G.predecessors(v))
        if not pa:
            per_node[v] = 0.0
            continue
        seen = train_df[pa].drop_duplicates()
        m = test_df[pa].merge(seen, how="left", on=pa, indicator=True)
        unseen = (m["_merge"] == "left_only").to_numpy()
        per_node[v] = float(unseen.mean())
        any_unseen |= unseen
    return float(any_unseen.mean()), per_node


# ---- runner -----------------------------------------------------------------

KEYS = ["disc_level", "dag_level", "fold", "n_train_spec", "estimator"]


def n_spec(n):
    """N_TRAIN entry -> resume key string ("500", ..., "full")."""
    return "full" if n is None else str(int(n))


def make_folds(target, k_folds, seed):
    """(train_idx, test_idx) per fold, stratified on the target. Shared by
    every discretisation and DAG, so all comparisons are paired."""
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True,
                          random_state=make_seed("fold", seed) % 2**32)
    return list(skf.split(np.zeros(len(target)), target))


def part_sizes(n, fracs):
    """Rows of an n-row dataset in each of (train, val, test); sums to n."""
    n_tr, n_va = round(n * fracs[0]), round(n * fracs[1])
    return n_tr, n_va, n - n_tr - n_va


def make_blocks(target, n_max, n_repeats, seed, fracs=(0.6, 0.2, 0.2)):
    """Disjoint repeats, each a fixed train / val / test partition.

    All rows are shuffled once (seeded) and cut into n_repeats disjoint blocks
    of n_max rows; each block is split once, stratified on the target, into
    train / val / test parts in `fracs` proportions, each part in random order.
    A smaller N takes the first rows of each part (part_sizes(N, fracs)), so
    within a repeat the subsets are random, nested across N, and a row never
    changes role. Because the blocks are disjoint, no repeat's test rows are
    another repeat's train or val rows, so validation scores can be pooled
    across repeats without touching any test data.
    -> list of {"train", "val", "test"} index arrays, one per repeat.
    """
    target = np.asarray(target)
    if n_repeats * n_max > len(target):
        raise ValueError(f"{n_repeats} repeats x {n_max} rows > {len(target)} rows")
    perm = np.random.default_rng(make_seed("fold", seed, n_max)).permutation(len(target))
    n_tr, n_va, n_te = part_sizes(n_max, fracs)
    blocks = []
    for r in range(n_repeats):
        idx = perm[r * n_max:(r + 1) * n_max]
        rs = make_seed("split", seed, n_max, r) % 2**32
        rest, test = train_test_split(idx, test_size=n_te, stratify=target[idx], random_state=rs)
        train, val = train_test_split(rest, test_size=n_va, stratify=target[rest], random_state=rs)
        blocks.append({"train": train, "val": val, "test": test})
    used = np.concatenate([np.concatenate(list(b.values())) for b in blocks])
    assert len(np.unique(used)) == len(used), "blocks / parts overlap"
    return blocks


def _parts(job, n, data, folds, blocks, fracs, val_frac, seed):
    """(sample, train_df, val_df, test_df, seeds) for one job at size n.

    kfold: as before (nested subsample of the training fold, split into
    train/val; the whole test fold). blocks: the first rows of each part of
    the repeat's block, so every N is split train/val/test in `fracs`.
    sample = train + val: what the tabular estimators fit on.
    """
    if blocks is None:
        fold = job["fold"]
        train_idx, test_idx = folds[fold]
        # one permutation per fold: each N takes its first N rows, so the
        # subsamples are nested (the N=500 rows are inside the N=2000 rows)
        perm = np.random.default_rng(make_seed("subsample", seed, fold)).permutation(train_idx)
        sample = data.iloc[perm if n is None else perm[:n]].reset_index(drop=True)
        split_seed = make_seed("split", seed, fold, len(sample))
        nn_seed = make_seed("nn", seed, fold, len(sample))
        # NN validation set comes out of the subsample; the test fold is never touched
        train_df, val_df = split_train_val(sample, val_frac=val_frac, seed=split_seed)
        test_df = data.iloc[test_idx].reset_index(drop=True)
        return sample, train_df, val_df, test_df, split_seed, nn_seed
    b = blocks[job["fold"]]
    n_tr, n_va, n_te = part_sizes(n, fracs)
    train_df = data.iloc[b["train"][:n_tr]].reset_index(drop=True)
    val_df = data.iloc[b["val"][:n_va]].reset_index(drop=True)
    test_df = data.iloc[b["test"][:n_te]].reset_index(drop=True)
    sample = pd.concat([train_df, val_df], ignore_index=True)
    return sample, train_df, val_df, test_df, np.nan, make_seed("nn", seed, job["fold"], n)


def _ests(estimators, disc, dag_level, spec, fold=None):
    """Estimators for one (discretisation, DAG, N, fold): a list (the same
    everywhere) or a dict keyed (disc, dag, spec) or (disc, dag, spec, fold)
    (run_selected; per fold when each fold was tuned on its own)."""
    if isinstance(estimators, dict):
        return estimators.get((disc, dag_level, spec, fold),
                              estimators.get((disc, dag_level, spec), []))
    return estimators


def at_n(est, n):
    """The estimator as fitted at N = n rows. An NN with a "batch_rule"
    {"n_ref": N_ref, "max_batch": B_max, "lr": "sqrt" | "fixed"} gets
    batch(n) = its batch x n / N_ref, at least its own batch and at most B_max
    (so steps per epoch stay about as at the tuning size N_ref, where its
    batch was chosen), and with lr="sqrt" its lr times sqrt(new batch / old)
    (the square-root scaling rule for Adam); n <= N_ref leaves it unchanged.
    The rule is fixed in advance, not tuned. Anything else is returned as is."""
    rule = est.get("batch_rule")
    if not rule or est["type"] != "nn" or n is None:
        return est
    b0 = est.get("batch_size", NN_HPARAMS["batch_size"])
    b = int(min(max(round(b0 * n / rule["n_ref"]), b0), max(rule["max_batch"], b0)))
    out = {**est, "batch_size": b}
    if rule.get("lr", "sqrt") == "sqrt":
        out["lr"] = est.get("lr", NN_HPARAMS["lr"]) * np.sqrt(b / b0)
    return out


def chosen_estimators(path, names=None):
    """NN estimators from a chosen-config JSON (experiments.save_chosen_configs),
    picked by name in the given order, e.g. names=["nn_32", "nn_128_64"]; None
    takes all. Each runs under its saved name, which is also its family, and
    records its config_id."""
    from .experiments import load_chosen_configs
    configs = load_chosen_configs(path)
    if names is not None:
        by_name = {c["name"]: c for c in configs}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise KeyError(f"not in {path}: {missing}; names there: {list(by_name)}")
        configs = [by_name[n] for n in names]
    return [{"type": "nn", "family": c["name"], **c} for c in configs]


def _real_job(job, data_by_level, dags, cards_by_level, folds, estimators, n_train,
              nn_max_n, val_frac, n_epochs, seed, n_jobs, blocks=None,
              split_fracs=(0.6, 0.2, 0.2), score_test=True):
    """Every (N, estimator) still to do for one (discretisation, DAG, fold or
    repeat). score_test=False (tuning) computes no test-set quantity at all."""
    disc, dag_level, fold = job["disc_level"], job["dag_level"], job["fold"]
    data, G, cards = data_by_level[disc], dags[dag_level], cards_by_level[disc]

    records = []
    for n in n_train:
        spec = n_spec(n)
        todo = [e for e in _ests(estimators, disc, dag_level, spec, fold)
                if (spec, e["name"]) in job["todo"]]
        if not todo:
            continue
        sample, train_df, val_df, test_df, split_seed, nn_seed = _parts(
            job, n, data, folds, blocks, split_fracs, val_frac, seed)
        if score_test:
            unseen, node_unseen = unseen_config_rate(G, sample, test_df)

        for est in todo:
            est = at_n(est, n)              # batch_rule: batch (and lr) for this N
            is_nn = est["type"] == "nn"
            model, fit_time, cpu_time, rss = fit_estimator(
                est, G, sample, train_df, val_df, cards, n_epochs, seed=nn_seed)
            # validation score (selection): NNs are already fitted on train only;
            # tabular estimators are refitted on train only for it
            val_model = model if is_nn else fit_estimator(
                est, G, train_df, train_df, val_df, cards, n_epochs, seed=nn_seed)[0]
            vll, vse, _ = held_out_logprob(val_model, val_df, cards)
            rec = {
                "disc_level":   disc,
                "dag_level":    dag_level,
                "fold":         fold,                   # repeat index for split="blocks"
                "n_train_spec": spec,
                "estimator":    est["name"],
                "family":       est.get("family", est["name"]),
                "selected_config": est.get("selected_config", ""),
                # settings actually fitted (after batch_rule), and the JSON config it came from
                "config_id":    estimator_config_id(est, n_epochs),
                "chosen_config_id": est.get("config_id", ""),
                "type":         est["type"],
                "hidden_dims":  str(list(est.get("hidden_dims", []))) if is_nn else "",
                "activation":   activation_name(est) if is_nn else "",
                "lr":           est.get("lr", NN_HPARAMS["lr"]) if is_nn else np.nan,
                "batch_size":   est.get("batch_size", NN_HPARAMS["batch_size"]) if is_nn else np.nan,
                "batch_rule":   json.dumps(est["batch_rule"]) if est.get("batch_rule") else "",
                "pseudo_count": est.get("pseudo_count", 0.5) if est["type"] == "dirichlet" else np.nan,
                "split":        "kfold" if blocks is None else "blocks",
                "sample_size":  len(sample) + (0 if blocks is None else len(test_df)),  # rows in the dataset
                "n_fit":        len(train_df) if is_nn else len(sample),
                "n_val":        len(val_df),
                "n_test":       len(test_df),
                "n_edges":      G.number_of_edges(),
                "seed":         seed,
                "split_seed":   split_seed,
                "nn_seed":      nn_seed,
                "val_loglik":   vll,                                # mean per validation row
                "val_loglik_se": vse,
                "fit_time_s":   fit_time,
                "fit_cpu_s":    cpu_time,
                "fit_rss_delta_bytes": rss,
                **model_stats(model, est.get("n_epochs", n_epochs)),
                **run_settings(n_jobs),
            }
            if score_test:
                ll, se, node_ll, floored = held_out_logprob(model, test_df, cards,
                                                            return_floored=True)
                rec.update({
                    "loglik":       ll,                             # mean per test row
                    "loglik_se":    se,
                    "n_floored":    int(sum(floored.values())),     # zero-prob hits
                    "unseen_rate":  unseen,     # test rows with >=1 unseen parent config
                    "node_loglik":  json.dumps(node_ll),
                    "node_unseen":  json.dumps(node_unseen),
                    "node_floored": json.dumps(floored),
                })
            records.append(rec)
    return records


def run_real_experiments(raw_df, disc_levels, dag_paths, estimators, n_train, k_folds,
                         nn_max_n, val_frac, n_epochs, results_dir,
                         results_name="results.csv", folds_to_run=None, seed=0,
                         verbose=True, n_jobs=1, device=None, torch_threads=None,
                         split="kfold", n_max=None, n_repeats=None,
                         split_fracs=(0.6, 0.2, 0.2), score_test=True):
    """Every estimator on every (discretisation, DAG, fold/repeat, N), appending
    rows to <results_dir>/<results_name>.

    raw_df     : load_cdc() output; discretised here per level.
    dag_paths  : DAG level ("high", ...) -> Bayesys CSV.
    estimators : list of estimator dicts, or {(disc, dag, n_spec): [dicts]}. An
                 entry with "n_values" runs only at those N; one with
                 "batch_rule" scales its batch size (and lr) with N (at_n).
    n_train    : sizes; kfold: training rows (None = whole training fold);
                 blocks: rows in the whole dataset, split train/val/test.
    nn_max_n   : NN estimators are skipped above this N (and for None).
    split      : "kfold" (k_folds stratified folds; N = training rows, val_frac
                 of them for NN early stopping) or "blocks" (make_blocks:
                 n_repeats disjoint blocks of n_max rows, each N split
                 train/val/test in split_fracs; N = the whole dataset).
    folds_to_run : subset of fold / repeat indices (e.g. [0] for a smoke run).
    score_test : False for tuning: rows hold validation scores only.
    Resumable on (disc_level, dag_level, fold, n_train_spec, estimator).
    n_jobs, device, torch_threads : as experiments.run_experiments; one job
    is one (discretisation, DAG, fold/repeat).
    """
    os.makedirs(results_dir, exist_ok=True)
    results_csv = os.path.join(results_dir, results_name)
    done = set()
    if os.path.exists(results_csv):
        old = pd.read_csv(results_csv, dtype={"n_train_spec": str})
        done = set(map(tuple, old[KEYS].astype(str).to_numpy()))

    columns = list(raw_df.columns)
    data_by_level, cards_by_level = {}, {}
    for lvl in disc_levels:
        data_by_level[lvl], cards_by_level[lvl] = discretise(raw_df, lvl)
    dags = {lvl: load_bayesys_dag(p, columns) for lvl, p in dag_paths.items()}
    target = raw_df[TARGET].to_numpy()
    if split == "kfold":
        folds, blocks, n_units = make_folds(target, k_folds, seed), None, k_folds
    elif split == "blocks":
        if any(n is None or n > n_max for n in n_train):
            raise ValueError(f"split='blocks' needs every N <= n_max={n_max}")
        folds, blocks, n_units = None, make_blocks(target, n_max, n_repeats, seed, split_fracs), n_repeats
    else:
        raise ValueError(f"unknown split {split!r}")
    folds_to_run = range(n_units) if folds_to_run is None else folds_to_run

    def wanted(est, n):
        if "n_values" in est and n not in est["n_values"]:     # e.g. a sensitivity check
            return False
        return est["type"] != "nn" or (n is not None and n <= nn_max_n)

    # fold-major, so a run cut short still has the first folds of every panel
    jobs = []
    for fold in folds_to_run:
        for disc in disc_levels:
            for dag_level in dag_paths:
                todo = {(n_spec(n), e["name"]) for n in n_train
                        for e in _ests(estimators, disc, dag_level, n_spec(n), fold)
                        if wanted(e, n) and
                        (disc, dag_level, str(fold), n_spec(n), e["name"]) not in done}
                if todo:
                    jobs.append({"disc_level": disc, "dag_level": dag_level,
                                 "fold": fold, "todo": todo})
                elif verbose:
                    tqdm.write(f"loaded results: {disc} {dag_level} fold {fold}")

    run_job = partial(_real_job, data_by_level=data_by_level, dags=dags,
                      cards_by_level=cards_by_level, folds=folds, estimators=estimators,
                      n_train=n_train, nn_max_n=nn_max_n, val_frac=val_frac,
                      n_epochs=n_epochs, seed=seed, n_jobs=n_jobs, blocks=blocks,
                      split_fracs=split_fracs, score_test=score_test)
    records = []
    bar = tqdm(total=len(jobs), desc="(disc, DAG, fold)")
    for recs in _parallel_map(run_job, jobs, n_jobs, device, torch_threads):
        records.extend(recs)
        if recs:
            r = recs[0]
            bar.set_postfix_str(f"{r['disc_level']} {r['dag_level']} fold {r['fold']}")
        flush(records, results_csv)
        bar.update()
    bar.close()


def select_real_configs(tuning, metric="val_loglik", per_fold=None):
    """One candidate per (discretisation, DAG, N, family), by validation
    log-likelihood only.

    per_fold : True picks a winner separately in every fold, from that fold's
               validation scores (nested CV: its test fold plays no part), and
               run_selected then uses each fold's own winner. False picks one
               winner from the mean across folds / repeats, used everywhere.
               None (default) = per fold for split="kfold", pooled for "blocks".
               Pooling k-fold scores is refused: fold i's test rows are training
               and validation rows of the other folds, so a choice averaged
               over folds has seen every fold's test data. (split="blocks"
               repeats are disjoint, so pooling them is safe.)
    -> one row per group: selected_config, its mean and SE, the runner-up's
    gap, and how many folds / repeats the mean rests on (1 per fold)."""
    kfold = "split" in tuning and (tuning["split"] == "kfold").any()
    per_fold = kfold if per_fold is None else per_fold
    if kfold and not per_fold:
        raise ValueError("k-fold tuning must be selected per fold (per_fold=True): pooling "
                         "validation scores across folds uses every fold's test rows")
    keys = ["disc_level", "dag_level", "n_train_spec", "family"] + (["fold"] if per_fold else [])
    g = (tuning.groupby(keys + ["estimator"])[metric]
               .agg(mean="mean", se="sem", n_repeats="count").reset_index()
               .sort_values(keys + ["mean"], ascending=[True] * len(keys) + [False]))
    best = g.groupby(keys).head(1).rename(columns={"estimator": "selected_config"})
    second = g.groupby(keys)["mean"].apply(lambda m: m.iloc[1] if len(m) > 1 else np.nan)
    best = best.merge(second.rename("runner_up_mean").reset_index(), on=keys)
    best["gap_to_runner_up"] = best["mean"] - best["runner_up_mean"]
    return best.reset_index(drop=True)


def run_selected(raw_df, selection, candidates, **kwargs):
    """Refit each selected candidate and score it on the test sets, written
    under its family name with `selected_config` set. Same rows and seeds as
    the tuning run, so each fit reproduces its tuning fit exactly. A
    per-fold selection (select_real_configs per_fold=True) runs each fold's
    own winner in that fold.
    kwargs : as run_real_experiments (same split settings as the tuning)."""
    by_name = {c["name"]: c for c in candidates}
    per_fold = "fold" in selection
    chosen = {}
    for r in selection.itertuples():
        c = by_name[r.selected_config]
        key = (r.disc_level, r.dag_level, r.n_train_spec) + ((int(r.fold),) if per_fold else ())
        chosen.setdefault(key, []).append(
            {**c, "name": c["family"], "selected_config": c["name"]})
    run_real_experiments(raw_df, estimators=chosen, score_test=True, **kwargs)


def node_table(results, dag_paths, raw_df):
    """Results rows -> one row per (result, node): log-likelihood, unseen rate
    and floored count, joined to that node's config_counts under its own
    discretisation and DAG."""
    columns = list(raw_df.columns)
    dags = {lvl: load_bayesys_dag(p, columns) for lvl, p in dag_paths.items()}
    counts = []
    for disc in results["disc_level"].unique():
        _, cards = discretise(raw_df, disc)
        for dag_level in results["dag_level"].unique():
            c = config_counts(dags[dag_level], cards)
            counts.append(c[c["node"] != "TOTAL"].assign(disc_level=disc, dag_level=dag_level))
    counts = pd.concat(counts, ignore_index=True)

    keep = KEYS + ["sample_size", "type"]
    long = []
    for _, r in results.iterrows():
        ll, un = json.loads(r["node_loglik"]), json.loads(r["node_unseen"])
        fl = json.loads(r["node_floored"]) if "node_floored" in r else {}
        long += [{**{k: r[k] for k in keep}, "node": v, "loglik": ll[v],
                  "unseen_rate": un.get(v, np.nan), "n_floored": fl.get(v, np.nan)}
                 for v in ll]
    return pd.DataFrame(long).merge(counts, on=["disc_level", "dag_level", "node"], how="left")
