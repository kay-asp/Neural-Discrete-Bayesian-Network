"""Ground-truth generation: random DAGs, Dirichlet CPTs, and seeding."""
import itertools
import random

import networkx as nx
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork


# Each random step gets its own purpose number, so e.g. the train and test data
# of one config never share a seed. Changing these numbers changes every seed.
SEED_PURPOSES = {"dag": 1, "cpt": 2, "train": 3, "test": 4, "split": 5, "nn": 6,
                 "fold": 7, "subsample": 8}   # 7-8: real-data folds / nested subsamples

def make_seed(purpose, *values):
    """Independent seed for one purpose ("dag", "train", ...) of one config.

    SeedSequence hashes the whole list, so different configs can't collide
    the way an arithmetic formula can.
    """
    return int(np.random.SeedSequence([SEED_PURPOSES[purpose], *values]).generate_state(1)[0])


def gen_dag_indegree(n, k, seed):
    rng = random.Random(seed)
    order = [f"X_{i}" for i in range(n)]
    rng.shuffle(order)
    G = nx.DiGraph()
    G.add_nodes_from(order)
    for i in range(1, n):
        n_parents = min(k, i)
        for p in rng.sample(order[:i], n_parents):
            G.add_edge(p, order[i])
    return G

def gen_dag_max_indegree(n, max_k, seed):
    rng = random.Random(seed)
    order = [f"X_{i}" for i in range(n)]
    rng.shuffle(order)
    G = nx.DiGraph()
    G.add_nodes_from(order)
    for i in range(1, n):
        n_parents = rng.randint(1, min(max_k, i))
        for p in rng.sample(order[:i], n_parents):
            G.add_edge(p, order[i])
    return G


def gen_CPTs(G, card, alpha, seed):

    rng = np.random.default_rng(seed)
    model = DiscreteBayesianNetwork(list(G.edges()))
    model.add_nodes_from(G.nodes())

    cpds = []
    for node in G.nodes():
        parents = sorted(G.predecessors(node))
        n_configs = card ** len(parents)
        vals = rng.dirichlet([alpha] * card, size=n_configs).T   # (card, n_configs)
        cpd = TabularCPD(
            variable=node, variable_card=card, values=vals,
            evidence=parents or None,
            evidence_card=[card] * len(parents) or None,
            state_names={v: list(range(card)) for v in [node] + parents},
        )
        cpds.append(cpd)

    model.add_cpds(*cpds)
    model.check_model()
    return model


def gen_logistic_CPTs(G, card, seed, scale, interaction=0.0):
    """Ground-truth CPTs that share structure across parent configurations.

    Each child state s gets a score per parent configuration,
        bias_s + sum_j W_j[pa_j, s] + sum_(j,l) V_jl[pa_j, pa_l, s],
    over the parents j and a few random parent pairs (j, l), and the CPT column is
    the softmax of the scores. Configurations that agree on a parent share that
    parent's effect, so data under one configuration is informative about
    others (unlike gen_CPTs, whose columns are independent draws).

    scale       : standard deviation of every score, whatever the in-degree
                  (see logistic_scale_matching).
    interaction : share of the score variance from the pairwise terms; 0 gives
                  a purely additive (multinomial-logistic) CPD. Nodes with fewer
                  than two parents have no pairs and are always additive.
    """
    rng = np.random.default_rng(seed)
    model = DiscreteBayesianNetwork(list(G.edges()))
    model.add_nodes_from(G.nodes())

    cpds = []
    for node in G.nodes():
        parents = sorted(G.predecessors(node))
        k = len(parents)
        # parent states of every configuration, in pgmpy's column order (last parent fastest)
        cfg = np.array(list(itertools.product(range(card), repeat=k)), dtype=np.int64).reshape(card ** k, k)
        all_pairs = list(itertools.combinations(range(k), 2))
        n_pairs = min(len(all_pairs), max(1, k // 2)) if interaction > 0 and all_pairs else 0
        pairs = [all_pairs[i] for i in rng.choice(len(all_pairs), n_pairs, replace=False)] if n_pairs else []
        share = interaction if pairs else 0.0
        sd_main = scale * np.sqrt((1 - share) / (k + 1))      # bias + one term per parent
        sd_pair = scale * np.sqrt(share / n_pairs) if pairs else 0.0

        scores = np.repeat(rng.normal(0, sd_main, size=(1, card)), len(cfg), axis=0)
        for j in range(k):
            scores += rng.normal(0, sd_main, size=(card, card))[cfg[:, j]]
        for j, l in pairs:
            scores += rng.normal(0, sd_pair, size=(card, card, card))[cfg[:, j], cfg[:, l]]

        probs = np.exp(scores - scores.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        cpds.append(TabularCPD(
            variable=node, variable_card=card, values=probs.T,
            evidence=parents or None,
            evidence_card=[card] * k or None,
            state_names={v: list(range(card)) for v in [node] + parents},
        ))

    model.add_cpds(*cpds)
    model.check_model()
    return model


def logistic_scale_matching(alpha, card, n=400_000, seed=0):
    """The gen_logistic_CPTs scale whose CPT columns have, on average, the same
    entropy as Dirichlet(alpha, ..., alpha) columns, so the two generators give
    conditional distributions of similar sharpness. Every score is N(0, scale^2)
    whatever the in-degree and interaction share, so one scale fits all nodes."""
    rng = np.random.default_rng(seed)

    def entropy(p):
        return -(p * np.log(np.clip(p, 1e-300, None))).sum(axis=1).mean()

    target = entropy(rng.dirichlet([alpha] * card, size=n))
    z = rng.normal(size=(n, card))
    lo, hi = 1e-3, 50.0
    for _ in range(60):                     # entropy falls as the scale grows
        mid = (lo + hi) / 2
        s = mid * z
        p = np.exp(s - s.max(axis=1, keepdims=True)); p /= p.sum(axis=1, keepdims=True)
        lo, hi = (mid, hi) if entropy(p) > target else (lo, mid)
    return (lo + hi) / 2
