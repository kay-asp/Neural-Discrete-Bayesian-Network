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
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from .evaluation import EPS, cpd_probs
from .experiments import (_parallel_map, activation_name, fit_estimator, flush,
                          model_stats, run_settings)
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


def _real_job(job, data_by_level, dags, cards_by_level, folds, estimators, n_train,
              nn_max_n, val_frac, n_epochs, seed, n_jobs):
    """Every (N, estimator) still to do for one (discretisation, DAG, fold)."""
    disc, dag_level, fold = job["disc_level"], job["dag_level"], job["fold"]
    data, G, cards = data_by_level[disc], dags[dag_level], cards_by_level[disc]
    train_idx, test_idx = folds[fold]
    test_df = data.iloc[test_idx].reset_index(drop=True)
    # one permutation per fold: each N takes its first N rows, so the
    # subsamples are nested (the N=500 rows are inside the N=2000 rows)
    perm = np.random.default_rng(make_seed("subsample", seed, fold)).permutation(train_idx)

    records = []
    for n in n_train:
        spec = n_spec(n)
        todo = [e for e in estimators if (spec, e["name"]) in job["todo"]]
        if not todo:
            continue
        sample = data.iloc[perm if n is None else perm[:n]].reset_index(drop=True)
        split_seed = make_seed("split", seed, fold, len(sample))
        nn_seed = make_seed("nn", seed, fold, len(sample))
        # NN validation set comes out of the subsample; the test fold is never touched
        train_df, val_df = split_train_val(sample, val_frac=val_frac, seed=split_seed)
        unseen, node_unseen = unseen_config_rate(G, sample, test_df)

        for est in todo:
            is_nn = est["type"] == "nn"
            model, fit_time, cpu_time, rss = fit_estimator(
                est, G, sample, train_df, val_df, cards, n_epochs, seed=nn_seed)
            ll, se, node_ll, floored = held_out_logprob(model, test_df, cards,
                                                        return_floored=True)
            records.append({
                "disc_level":   disc,
                "dag_level":    dag_level,
                "fold":         fold,
                "n_train_spec": spec,
                "estimator":    est["name"],
                "type":         est["type"],
                "hidden_dims":  str(list(est.get("hidden_dims", []))) if is_nn else "",
                "activation":   activation_name(est) if is_nn else "",
                "pseudo_count": est.get("pseudo_count", 0.5) if est["type"] == "dirichlet" else np.nan,
                "sample_size":  len(sample),                        # budget
                "n_fit":        len(train_df) if is_nn else len(sample),
                "n_test":       len(test_df),
                "n_edges":      G.number_of_edges(),
                "seed":         seed,
                "split_seed":   split_seed,
                "nn_seed":      nn_seed,
                "loglik":       ll,                                 # mean per test row
                "loglik_se":    se,
                "n_floored":    int(sum(floored.values())),         # zero-prob hits
                "unseen_rate":  unseen,         # test rows with >=1 unseen parent config
                "node_loglik":  json.dumps(node_ll),
                "node_unseen":  json.dumps(node_unseen),
                "node_floored": json.dumps(floored),
                "fit_time_s":   fit_time,
                "fit_cpu_s":    cpu_time,
                "fit_rss_delta_bytes": rss,
                **model_stats(model, est.get("n_epochs", n_epochs)),
                **run_settings(n_jobs),
            })
    return records


def run_real_experiments(raw_df, disc_levels, dag_paths, estimators, n_train, k_folds,
                         nn_max_n, val_frac, n_epochs, results_dir,
                         results_name="results.csv", folds_to_run=None, seed=0,
                         verbose=True, n_jobs=1, device=None, torch_threads=None):
    """Every estimator on every (discretisation, DAG, fold, N), appending rows
    to <results_dir>/<results_name>.

    raw_df     : load_cdc() output; discretised here per level.
    dag_paths  : DAG level ("high", ...) -> Bayesys CSV.
    n_train    : training-set sizes per fold; None = the whole training fold.
    nn_max_n   : NN estimators are skipped above this N (and for None).
    folds_to_run : subset of fold indices (e.g. [0] for a smoke run); folds
                 are always made with k_folds, so fold 0 is the same fold.
    Resumable on (disc_level, dag_level, fold, n_train_spec, estimator).
    n_jobs, device, torch_threads : as experiments.run_experiments; one job
    is one (discretisation, DAG, fold).
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
    folds = make_folds(raw_df[TARGET].to_numpy(), k_folds, seed)
    folds_to_run = range(k_folds) if folds_to_run is None else folds_to_run

    def wanted(est, n):
        return est["type"] != "nn" or (n is not None and n <= nn_max_n)

    # fold-major, so a run cut short still has the first folds of every panel
    jobs = []
    for fold in folds_to_run:
        for disc in disc_levels:
            for dag_level in dag_paths:
                todo = {(n_spec(n), e["name"]) for n in n_train for e in estimators
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
                      n_epochs=n_epochs, seed=seed, n_jobs=n_jobs)
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
