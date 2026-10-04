"""Tabular (MLE / Dirichlet pseudocount) CPD estimators."""
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator, DiscreteMLE


# MLE with normal default for unseen parent configurations
def fit_mle(G, data, card):
    model = DiscreteBayesianNetwork(list(G.edges()))
    model.add_nodes_from(G.nodes())

    estimator = DiscreteMLE()
    # force full state space so unobserved states aren't dropped
    estimator.state_names = {v: list(range(card)) for v in G.nodes()}
    model.fit(data, estimator=estimator)
    return model

# MLE with a small symmetric Dirichlet pseudocount
def fit_mle_pseudocount(G, data, card):
    pseudo_count=0.5
    m = DiscreteBayesianNetwork(list(G.edges()))
    m.add_nodes_from(G.nodes())
    est = DiscreteBayesianEstimator(
        state_names={v: list(range(card)) for v in G.nodes()},
        prior_type="dirichlet",
        pseudo_counts=float(pseudo_count),   # 0.1–1.0 is a sensible range
    )
    m.fit(data, estimator=est)
    return m
