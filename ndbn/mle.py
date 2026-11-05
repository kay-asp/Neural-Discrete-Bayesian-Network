"""Tabular (MLE / Dirichlet pseudocount) CPD estimators."""
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator, DiscreteMLE

from .models import Cards


# MLE with normal default for unseen parent configurations
def fit_mle(G, data, cards):
    """cards : an int (shared) or node -> cardinality."""
    cards = Cards.coerce(cards, G.nodes())
    model = DiscreteBayesianNetwork(list(G.edges()))
    model.add_nodes_from(G.nodes())

    estimator = DiscreteMLE()
    # force full state space so unobserved states aren't dropped
    estimator.state_names = {v: list(range(cards[v])) for v in G.nodes()}
    model.fit(data, estimator=estimator)
    return model

# MLE with a small symmetric Dirichlet pseudocount
def fit_dirichlet(G, data, cards, pseudo_count=0.5):
    cards = Cards.coerce(cards, G.nodes())
    m = DiscreteBayesianNetwork(list(G.edges()))
    m.add_nodes_from(G.nodes())
    est = DiscreteBayesianEstimator(
        state_names={v: list(range(cards[v])) for v in G.nodes()},
        prior_type="dirichlet",
        pseudo_counts=float(pseudo_count),   # 0.1–1.0 is a sensible range
    )
    m.fit(data, estimator=est)
    return m

# Dirichlet with its pseudocount chosen on validation data
PSEUDO_COUNT_GRID = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0)

def fit_dirichlet_tuned(G, train, val, data, cards, grid=PSEUDO_COUNT_GRID):
    """Each pseudocount in `grid` is fitted on `train` and scored by the mean
    log-likelihood of the `val` rows; the best is refitted on `data` (train +
    val, as every tabular estimator uses). No test rows or ground truth are
    used. The chosen value is stored as model.pseudo_count_."""
    from .evaluation import EPS, cpd_probs
    best, best_ll = None, -np.inf
    for a in grid:
        m = fit_dirichlet(G, train, cards, a)
        ll = sum(np.log(np.clip(cpd_probs(m.get_cpds(v), val)[np.arange(len(val)), val[v].to_numpy(np.int64)],
                                EPS, None)).mean() for v in G.nodes())
        if ll > best_ll:
            best, best_ll = a, ll
    model = fit_dirichlet(G, data, cards, best)
    model.pseudo_count_ = best
    return model
