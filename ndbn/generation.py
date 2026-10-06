"""Ground-truth generation: random DAGs, Dirichlet CPTs, and seeding."""
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
