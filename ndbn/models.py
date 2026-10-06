"""Neural CPD model definitions."""
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_device(name=None):
    """Set the device used for all neural CPDs; None = auto (cuda if available).

    Read the current value as models.DEVICE (not a `from .models import DEVICE`
    copy), so a change here reaches every module.
    """
    global DEVICE
    DEVICE = name or ("cuda" if torch.cuda.is_available() else "cpu")
    return DEVICE


class Cards(dict):
    """node -> cardinality. Every public entry point coerces its `card(s)`
    argument with Cards.coerce, so a plain int (one shared cardinality, as the
    synthetic experiments use) and a per-node dict (real data) both work."""
    @classmethod
    def coerce(cls, card, nodes):
        if isinstance(card, (int, np.integer)):
            return cls({n: int(card) for n in nodes})
        return cls(card)


# Linear Softmax Baseline
class NodeNN(nn.Module):
    # one-hot parent config -> softmax distribution over child states
    def __init__(self, parent_cards, card, hidden_dims=(), activation=nn.ReLU, dropout=0.0):
        """parent_cards : each parent's cardinality (input = their one-hots,
        so in_dim is their SUM), or an int = number of parents, each with the
        child's card (the old uniform-card signature)."""
        super().__init__()
        if isinstance(parent_cards, (int, np.integer)):
            parent_cards = [card] * int(parent_cards)
        in_dim = sum(parent_cards) or 1
        dims = [in_dim] + list(hidden_dims)

        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if activation is not None:
                layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()
        self.logits_out = nn.Linear(dims[-1], card)

    def forward(self, x):
        return self.logits_out(self.feature_extractor(x))   # raw logits


def one_hot_parents(data, parents, cards):
    """Each parent one-hot at its own cardinality, concatenated in `parents`
    order -> (N, sum of parent cards); (N, 1) zeros for a root node."""
    if not parents:
        return torch.zeros(len(data), 1, dtype=torch.float32, device=DEVICE)
    cards = Cards.coerce(cards, parents)
    idx = torch.as_tensor(data[parents].to_numpy(np.int64), device=DEVICE)
    return torch.cat([torch.nn.functional.one_hot(idx[:, j], cards[p])
                      for j, p in enumerate(parents)], dim=1).float()


@dataclass
class NeuralCPDs:
    """Per-node neural CPDs over a fixed DAG.

    Indexing yields (net, parents) — exactly what nn_probs unpacks — so this
    passes straight to kl_nn. Trailing-underscore fields are fitted attributes.
    """
    models: dict                  # node -> NodeNN
    parents: dict                 # node -> list[str], canonical ordering
    cards: dict                   # node -> cardinality

    best_val_:   dict = field(default_factory=dict)
    train_loss_:  dict = field(default_factory=dict)
    train_time_: dict = field(default_factory=dict)
    stop_epoch_: dict = field(default_factory=dict)   # when patience ran out
    best_epoch_: dict = field(default_factory=dict)   # epoch of the kept weights
    history_:    dict = field(default_factory=dict)

    @property
    def card(self):
        """The shared cardinality; raises if nodes differ (use .cards)."""
        values = set(self.cards.values())
        if len(values) != 1:
            raise ValueError(f"cardinalities differ across nodes ({sorted(values)}); use .cards")
        return values.pop()

    def __getitem__(self, node):
        return self.models[node], self.parents[node]

    def nodes(self):
        return list(self.models)

    def param_stats(self):
        stored = sum(p.numel() for m in self.models.values() for p in m.parameters())
        native = sum(p.numel() * p.element_size()
                     for m in self.models.values() for p in m.parameters())
        return {
            "stored_params": int(stored),
            "free_params":   int(stored),       # every NN parameter is free
            "memory_bytes":  int(stored * 8),   # float64-equivalent, comparable to CPTs
            "memory_bytes_native": int(native),
        }


def nn_probs(net_and_parents, test_data, cards):
    """Q(node | pa) for every row, from that node's network -> (N, card).

    cards : an int (shared) or node -> cardinality (only the parents' are read).
    """
    net, parents = net_and_parents
    x = one_hot_parents(test_data, parents, cards)
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(x), dim=1).cpu().numpy()
