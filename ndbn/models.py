"""Neural CPD model definitions."""
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Linear Softmax Baseline
class NodeNN(nn.Module):
    # one-hot parent config -> softmax distribution over child states
    def __init__(self, n_parents, card, hidden_dims=(), activation=nn.ReLU, dropout=0.0):
        super().__init__()
        in_dim = n_parents * card if n_parents>0 else 1
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


def one_hot_parents(data, parents, card):
    if not parents:
        return torch.zeros(len(data), 1, dtype=torch.float32, device=DEVICE)
    idx = torch.as_tensor(data[parents].to_numpy(np.int64), device=DEVICE)
    return torch.nn.functional.one_hot(idx, card).flatten(1).float()


@dataclass
class NeuralCPDs:
    """Per-node neural CPDs over a fixed DAG.

    Indexing yields (net, parents) — exactly what nn_probs unpacks — so this
    passes straight to kl_nn. Trailing-underscore fields are fitted attributes.
    """
    models: dict                  # node -> NodeNN
    parents: dict                 # node -> list[str], canonical ordering
    card: int

    best_val_:   dict = field(default_factory=dict)
    train_loss_:  dict = field(default_factory=dict)
    train_time_: dict = field(default_factory=dict)
    stop_epoch_: dict = field(default_factory=dict)
    history_:    dict = field(default_factory=dict)

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


def nn_probs(net_and_parents, test_data, card):
    """Q(node | pa) for every row, from that node's network -> (N, card)."""
    net, parents = net_and_parents
    if parents:
        idx = torch.as_tensor(test_data[parents].to_numpy().astype(np.int64), device=DEVICE)
        x = torch.nn.functional.one_hot(idx, card).reshape(len(test_data), -1).float()
    else:
        x = torch.zeros((len(test_data), 1), dtype=torch.float32, device=DEVICE)
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(x), dim=1).cpu().numpy()
