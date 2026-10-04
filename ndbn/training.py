"""Training neural CPDs with per-node early stopping."""
import time

import numpy as np
import torch
import torch.nn as nn

from .models import DEVICE, NeuralCPDs, NodeNN, one_hot_parents
from .tracing import Trace


def split_train_val(data, val_frac=0.2, seed=0):
    """Hold out a validation subset used only for early stopping."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(data))
    n_val = max(1, round(val_frac * len(data)))
    val_idx, train_idx = np.split(perm, [n_val])
    return (data.iloc[train_idx].reset_index(drop=True),
            data.iloc[val_idx].reset_index(drop=True))


def train_nn_cpds(G, df_train, df_val, card,
                  hidden_dims, activation,
                  n_epochs=500, lr=1e-3, dropout=0.0,
                  weight_decay=1e-4, batch_size=64, patience=20,
                  seed=0, verbose=False, trace=None):
    torch.manual_seed(seed)
    nodes = list(G.nodes())
    parents = {node: sorted(G.predecessors(node)) for node in nodes}
    loss = nn.CrossEntropyLoss()

    models, optimizers = {}, {}
    Xtr, ytr, Xva, yva = {}, {}, {}, {}
    train_times = {node: 0.0 for node in nodes}

    for node in nodes:
        models[node] = NodeNN(len(parents[node]), card, hidden_dims,
                              activation, dropout).to(DEVICE)
        optimizers[node] = torch.optim.Adam(models[node].parameters(),
                                            lr=lr, weight_decay=weight_decay)
        Xtr[node] = one_hot_parents(df_train, parents[node], card)
        ytr[node] = torch.as_tensor(df_train[node].to_numpy().astype(np.int64), device=DEVICE)
        Xva[node] = one_hot_parents(df_val, parents[node], card)
        yva[node] = torch.as_tensor(df_val[node].to_numpy().astype(np.int64), device=DEVICE)
    n_train = len(df_train)

    # Early stopping state (per node, taken at the best-val epoch)
    best_train_loss = {node: float("nan") for node in nodes}
    best_val_loss   = {node: float("inf") for node in nodes}
    best_state      = {node: None for node in nodes}
    patience_ctr    = {node: 0 for node in nodes}
    stopped         = {node: False for node in nodes}
    stop_epoch      = {node: n_epochs for node in nodes}

    # Diagnostics state — inert when trace is None
    trace = trace or Trace()
    k_of = {node: len(parents[node]) for node in nodes}
    hist_train_loss = {node: [] for node in nodes} if trace.history else None
    hist_val_loss   = {node: [] for node in nodes} if trace.history else None

    for epoch in range(n_epochs):
        train_loss, val_loss = {}, {}        # this epoch, alive nodes only

        for node in nodes:
            if stopped[node]:
                continue
            model, opt = models[node], optimizers[node]

            # TRAIN (updates only)
            t0 = time.perf_counter()
            model.train()
            if batch_size is None:
                opt.zero_grad()
                loss(model(Xtr[node]), ytr[node]).backward()
                opt.step()
            else:
                perm = torch.randperm(n_train, device=DEVICE)
                for s in range(0, n_train, batch_size):
                    b = perm[s:s + batch_size]
                    opt.zero_grad()
                    loss(model(Xtr[node][b]), ytr[node][b]).backward()
                    opt.step()
            train_times[node] += time.perf_counter() - t0

            # VALIDATION
            model.eval()
            with torch.no_grad():
                val_loss[node]   = loss(model(Xva[node]), yva[node]).item()
                if trace.history or trace.logging:
                    train_loss[node] = loss(model(Xtr[node]), ytr[node]).item()


            if trace.history:
                hist_train_loss[node].append(train_loss[node])
                hist_val_loss[node].append(val_loss[node])

            # EARLY STOPPING
            if val_loss[node] < best_val_loss[node] - 1e-6:
                best_train_loss[node] = train_loss.get(node, float("nan"))
                best_val_loss[node]   = val_loss[node]
                patience_ctr[node] = 0
                best_state[node] = {k: v.detach().clone()
                                    for k, v in model.state_dict().items()}
            else:
                patience_ctr[node] += 1
                if patience_ctr[node] >= patience:
                    stopped[node] = True
                    stop_epoch[node] = epoch + 1
                    model.load_state_dict(best_state[node])
                    if verbose:
                        print(f"Node {node} early stopped at epoch {epoch+1}. "
                              f"Best val loss={best_val_loss[node]:.4f}")

        if trace.logging and val_loss:
            trace.log_epoch(epoch, nodes, k_of, val_loss, train_loss,
                            best_val_loss, models, parents, card)

        if verbose and (epoch + 1) % 50 == 0 and val_loss:
            tr = f"train {np.mean(list(train_loss.values())):.4f}, " if train_loss else ""
            print(f"Epoch {epoch+1}, {tr}val {np.mean(list(val_loss.values())):.4f}")

        if all(stopped.values()):
            if verbose:
                print(f"All nodes early-stopped at epoch {epoch+1}.")
            break

    for node in nodes:
        if not stopped[node] and best_state[node] is not None:
            models[node].load_state_dict(best_state[node])

    return NeuralCPDs(
        models=models, parents=parents, card=card,
        best_val_=best_val_loss, train_loss_=best_train_loss,
        train_time_=train_times, stop_epoch_=stop_epoch,
        history_=({"train": hist_train_loss, "val": hist_val_loss}
                  if trace.history else {}),
    )

def fit_nn(G, data, card, val_data=None, val_frac=0.2,
           hidden_dims=(32,), activation=nn.ReLU, seed=0, verbose=False, **kw):
    """Neural CPD estimator. Mirrors fit_mle(G, data, card).

    val_data : held-out split for early stopping. If None, carved from `data`
               — convenient for standalone use, but note that then `data` is a
               budget rather than the rows actually fitted.
    """
    if val_data is None:
        data, val_data = split_train_val(data, val_frac=val_frac, seed=seed)
    return train_nn_cpds(G, data, val_data, card,
                         hidden_dims=hidden_dims, activation=activation,
                         seed=seed, verbose=verbose, **kw)
