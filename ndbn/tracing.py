"""Training diagnostics: W&B logging and per-node loss histories."""
from dataclasses import dataclass
from typing import Any

import numpy as np

from .evaluation import kl_nn
from .models import NeuralCPDs


@dataclass
class Trace:
    """Diagnostics for a tuning run. Pass trace=None (the default) for a normal fit."""
    run: Any = None            # a wandb run object, or None to skip W&B entirely
    true_model: Any = None     # ground-truth BN, needed only for true-KL logging
    test_data: Any = None      # test set the true KL is evaluated on
    kl_every: int = 0          # log true KL every N epochs; 0 = never
    history: bool = False      # keep per-node train/val curves in memory

    @property
    def logging(self):
        return self.run is not None

    @property
    def tracking_kl(self):
        return (self.kl_every > 0 and self.true_model is not None
                and self.test_data is not None)

    def log_epoch(self, epoch, nodes, k_of, val_loss, train_loss, best_val_loss,
                  models, parents, card):
        """One epoch of W&B diagnostics. Safe to call unguarded."""
        if not self.logging or not val_loss:
            return

        alive = list(val_loss)
        rec = {
            "epoch":            epoch + 1,
            "n_alive":          len(alive),
            "val_loss":    float(np.mean([val_loss[n] for n in alive])),
            "train_loss":  float(np.mean([train_loss[n] for n in alive])),
            # frozen cohort: every node at its best-so-far, so the mean is not
            # dragged around by easy nodes dropping out of the alive set
            "val_loss_best": float(np.mean([best_val_loss[n] for n in nodes
                                               if np.isfinite(best_val_loss[n])])),
        }
        rec["gap_alive"] = rec["val_loss"] - rec["train_loss"]

        for b in sorted(set(k_of.values())):
            live_b = [n for n in alive if k_of[n] == b]
            if live_b:
                v = float(np.mean([val_loss[n] for n in live_b]))
                t = float(np.mean([train_loss[n] for n in live_b]))
                rec[f"val_loss_k{b}"] = v
                rec[f"train_loss_k{b}"] = t
                rec[f"gap_k{b}"] = v - t
                rec[f"n_alive_k{b}"] = len(live_b)
            finite_b = [best_val_loss[n] for n in nodes
                        if k_of[n] == b and np.isfinite(best_val_loss[n])]
            if finite_b:
                rec[f"val_loss_best_k{b}"] = float(np.mean(finite_b))

        if self.tracking_kl and (epoch + 1) % self.kl_every == 0:
            snap = NeuralCPDs(models=models, parents=parents, card=card)
            rec["true_kl"] = kl_nn(self.true_model, snap, self.test_data, card)[0]

        self.run.log(rec, step=epoch + 1)


def history_matrix(model, key="val"):
    """(n_nodes, max_epochs) array of per-node curves, NaN after each node stops."""
    h = model.history_[key]
    E = max(len(v) for v in h.values())
    M = np.full((len(h), E), np.nan)
    for i, v in enumerate(h.values()):
        M[i, :len(v)] = v
    return M
