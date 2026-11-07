# Neural CPDs for discrete Bayesian networks

Honours research project (University of the Witwatersrand). Each conditional probability table (CPT) of a
discrete Bayesian network is replaced by a small neural network, a *neural CPD*. Neural CPDs are compared with
maximum likelihood (MLE) and Dirichlet-smoothed CPTs on synthetic networks and on the CDC Diabetes Health
Indicators data with expert-elicited DAGs. The report's LaTeX source is not part of this repository; the
notebooks produce every figure, table and number in it.

## Setup

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt install graphviz          # the `dot` binary, for the DAG figures
```

Weights & Biases is only used by the hyperparameter sweeps. Set `WANDB_MODE=disabled` to run them without
an account.

## Layout

| Path | Contents |
|---|---|
| `config.py` | Settings shared by every notebook (folders, tuning-DAG offset, test size, worker count) |
| `ndbn/` | Package: `generation` (random DAGs, Dirichlet CPTs, seeds), `mle` (MLE / Dirichlet CPTs), `models` and `training` (neural CPDs), `evaluation` (KL divergence, parameter counts), `experiments` (synthetic runs and sweeps), `real_data` (CDC data, cross-validation, tuning, diagnosis), `plotting` |
| `1_setup_and_checks.ipynb` … `4_real_data.ipynb` | The experiments, run in order |
| `results-tuning/` | Sweep trials (`hparam_trials.csv`), the chosen network configs (`chosen_nn_c2.json`, `chosen_nn_c4.json`) and tuning-DAG data |
| `results-synthetic/` | `results.csv` (every synthetic fit), the cached train/test samples, and `alphas.csv` (the earlier run over Dirichlet α = 0.1, 0.5, 1.0 behind Supplementary Figure S3; its `source` column records where it came from) |
| `results-realdata/` | Real-data results: fixed configs (`results_c4.csv`), tuning (`tuning.csv`, `selection.csv`), tuned configs (`results_tuned.csv`), serial timing, diagnosis |
| `figs/` | Every report figure, written by the notebooks |
| `report/tables/` | Created when notebooks 3 and 4 run: the supplementary's LaTeX tables (`synthetic_kl_r2.tex`, `real_loglik.tex`) |
| `data/` | CDC data (downloaded on first use) and the expert DAGs |

## Reproducing the results

Run the notebooks in order. Expensive steps are either guarded by a flag (`False` reuses the saved results)
or resumable, skipping fits already in the results file. With every flag `False`, a full run regenerates
the figures from the saved CSVs.

| Notebook | Does | Writes |
|---|---|---|
| 1 | Smoke tests of the generators and estimators | nothing |
| 2 | No-early-stopping diagnostic and Bayesian sweeps on separate tuning DAGs (`RUN_SWEEPS`); candidate comparison; final configs | `results-tuning/`, `figs/no_es.png`, `figs/sweep_c{2,4}.png` |
| 3 | Main synthetic grid for r = 2 and r = 4 (resumable); report figures | `results-synthetic/results.csv`, `figs/{mle_fails,arms,kl_vs_N,size_alpha}_c{2,4}.png`, `figs/rho.png`, `figs/alpha_check.png` (from `alphas.csv`), `report/tables/synthetic_kl_r2.tex` |
| 4 | Real data: 3-fold CV over 3 discretisations × 3 expert DAGs × 4 sample sizes (resumable); per-fold tuning (`RUN_TUNING`); diagnostic accuracy (`RUN_DIAGNOSIS`); report figures | `results-realdata/`, `figs/real_*.png`, `figs/dag_moderate.png`, `report/tables/real_loglik.tex` |

The full synthetic grid is the expensive part: about 120 core-hours for every network, about 32 for `nn_32`
alone. Tabular fits at r = 4 with in-degree 8–10 build very large tables, so run them with few workers
(`n_jobs` ≤ 2 on a 16 GB machine). Every random step is seeded from its configuration
(`ndbn.generation.make_seed`), so reruns reproduce the data and splits exactly. Network fits can differ
slightly across PyTorch versions.

Each results row records `code_version`, the git commit it was produced with (`+dirty`: uncommitted changes).
The experiments ran in stages, so the files mix several versions of the same code, but this does not change the
results: all seeds match the current code, MLE rows reproduce exactly, and network rows reproduce to within
about 2%, consistent with differences between PyTorch builds.

## Data

- **CDC Diabetes Health Indicators**: UCI Machine Learning Repository, dataset 891
  (doi:10.24432/C53919), from the CDC's Behavioral Risk Factor Surveillance System. Cite it and follow the
  acknowledgment policy on the UCI page. Downloaded by `ndbn.real_data.load_cdc` and cached as
  `data/cdc_diabetes.csv`.
- **Expert DAGs** (`data/bayesys/knowledge_*_confidence.csv`): Zahoor, Constantinou, Curtis &
  Hasanuzzaman (2025), *Knowledge-Based Systems* 329:114382.
