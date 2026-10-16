# Implementation Summary

A condensed version of *Neural Network Parameter Estimation for Discrete Bayesian Networks with
Applications in Medical Diagnosis* (research proposal, K. Aspoas, May 2026), cut down to what this
repository implements. Where the implementation departs from the proposal, the implemented version
is described and the change is marked **Changed from proposal**.

Code lives in `ndbn/`; the experiments are run from four notebooks:

| Notebook | Purpose |
|---|---|
| `1_mle.ipynb` | Ground-truth generation, estimator tests, shared settings, MLE-only run (RQ1) |
| `2_hyperparameter_tuning.ipynb` | Architecture / hyperparameter sweeps and selection of the final neural CPDs |
| `3_main_experiments.ipynb` | MLE vs neural CPDs on the evaluation networks (RQ1 + RQ2), plots, serial timing run |
| `4_real_data.ipynb` | Real-data extension: CDC Diabetes Health Indicators on expert-elicited DAGs |

---

## 1. Problem

A discrete Bayesian network factorises the joint distribution along a DAG $G$:

$$P(X_1,\dots,X_n) = \prod_{i=1}^{n} P(X_i \mid \mathrm{Pa}_{X_i}).$$

Standard estimators (MLE, Bayesian/Dirichlet estimation) store each local distribution as a
conditional probability table (CPT) with one column per parent configuration. With $k$ parents of
cardinality $r$ a CPT has $r^k$ columns, so the parameter count grows exponentially with
in-degree. With a fixed amount of data, most parent configurations are then seen rarely or never,
and MLE produces zero or uninformed estimates.

The proposal replaces each CPT with a small neural network that maps the parent values to a
softmax distribution over the child's states. The DAG (the interpretable part) is unchanged; only
local parameter estimation changes. Parameters are shared across parent configurations, and the
network's input size grows with the **sum** of parent cardinalities rather than their product.

## 2. Research questions

- **RQ1:** What are the representational limitations of MLE in capturing complex dependencies in
  a discrete Bayesian network? Measured by KL divergence to a known ground truth as in-degree and
  network size increase, at a fixed sample budget.
- **RQ2:** Can the local probability model of each family be replaced by a neural network to
  capture more complex dependencies? The same ground truths, data and metric are used, compared
  directly with MLE.

**Added beyond the proposal:** a real-data evaluation (Section 7). The proposal listed this only
as future work.

## 3. Ground-truth networks and data (`ndbn/generation.py`, `ndbn/experiments.py`)

### DAG generation

**Changed from proposal.** The four density / in-degree scenarios of proposal Table 3.1 (sparse,
moderate, dense, sparse with a dense target family) are not used. A single generator,
`gen_dag_indegree(n, k)`, controls in-degree directly:

1. Shuffle the nodes into a random topological order.
2. Give the node at position $i$ exactly $\min(k, i)$ parents, sampled uniformly from the
   nodes before it.

Every node past the first $k$ therefore has in-degree exactly $k$. Graph density is not targeted;
it is recorded for each realised graph as $|E| / \binom{n}{2}$. (`gen_dag_max_indegree`, with
in-degree drawn uniformly from $1..k$, is implemented but not used in the experiments.)

### CPTs

Every ground-truth CPT column is drawn from a symmetric Dirichlet: $P^*(X_i \mid \mathrm{pa}) \sim
\mathrm{Dir}(\alpha, \dots, \alpha)$. A smaller $\alpha$ gives sharper, more peaked
distributions; $\alpha = 1$ is uniform over the simplex.

### Experimental grid

| Setting | Proposal | Implemented (main experiment, `results7-parallel/`) |
|---|---|---|
| Network size $n$ | 5–50, depending on scenario | 10, 20, 30, 40 |
| In-degree $k$ | 1–10, depending on scenario | 2, 4, 6, 8, 10 |
| Density | Controlled ranges (0.05–1.00) | Not controlled; follows from $n$ and $k$, recorded |
| Cardinality | Not specified | Binary ($r = 2$) |
| Dirichlet $\alpha$ | Not specified | 0.5 (the MLE-only run in `1_mle.ipynb` also uses 0.1) |
| Networks per configuration | "Multiple" | 5 random DAGs (evaluation DAGs 1–5) |
| Sample budget $N$ (train + val) | Fixed across $k$ | 200, 600, 1000, 1400, the same for every $k$ |

As in the proposal, the sample budget is held constant as in-degree grows, so any degradation is
due to complexity and not to differing amounts of data.

### Sampling, splits and seeds

- Training data is drawn by ancestral (forward) sampling (`pgmpy` `simulate`). Smaller budgets are
  the first $N$ rows of the largest sample, so they are nested.
- **Changed from proposal.** The proposal said no held-out test set was needed, since the true
  CPTs are known. The implementation draws a separate **test sample of 10,000 rows** from each
  ground truth and uses it only to estimate KL (Section 5). It is never used for fitting or
  selection.
- For neural CPDs, a validation subset is held out of the budget for early stopping
  (30% in the main experiment, 20% during tuning). MLE fits on the **whole** budget
  (train + validation), so it sees every row the network sees.
- Each random step (DAG, CPTs, train sample, test sample, train/val split, NN initialisation and
  batch order) has its own seed, derived by hashing the configuration (`make_seed`). Results are
  therefore reproducible, identical between serial and parallel runs, and paired across
  estimators: all estimators see the same networks and data.
- `alpha` and cardinality are not part of the seeds, so different $\alpha$ values share the same
  DAG structures (paired comparison).

## 4. Estimators

### Maximum likelihood (`ndbn/mle.py`)

$$\hat P(X_i = x_k \mid \mathrm{Pa}(X_i) = u_j) = \frac{N_{ijk}}{N_{ij}},$$

fitted with `pgmpy`'s MLE over the full state space. A parent configuration that never appears in
the data gets pgmpy's default (uniform) column. A child state never seen under an observed
configuration keeps probability 0, which is floored at $10^{-12}$ when KL / log-likelihood are
computed.

**Added:** a Dirichlet pseudocount (Bayesian) estimator, `fit_mle_pseudocount`. It is used as the
main tabular baseline on real data, where plain MLE's zero probabilities dominate the score.

### Neural CPDs (`ndbn/models.py`, `ndbn/training.py`)

One independent feed-forward network per node, $P(X_i \mid \mathrm{Pa}(X_i)) =
f_\theta(\mathrm{Pa}(X_i))$:

- **Input:** each parent one-hot encoded at its own cardinality, then concatenated (width = sum
  of parent cardinalities). A root node gets a constant zero input, so it learns its marginal
  through the output bias.
- **Hidden layers:** $h = \phi(Wx + b)$ with ReLU, followed by dropout.
- **Output:** a linear layer to $r$ logits; softmax gives a valid distribution over the child's
  states.
- **Loss:** cross-entropy (mean negative log-likelihood) of the observed child state given its
  observed parents.
- **Training:** mini-batch AdamW (Adam for the linear baseline). Each node has its own early
  stopping on validation cross-entropy, and the weights from its best validation epoch are
  restored.

**Changed from proposal: input representation.** Only one-hot inputs are implemented. The
proposed comparison with learned embeddings was not carried out.

**Changed from proposal: architectures and hyperparameters.** Proposal Table 3.2 (linear / 32 /
64-32 / 128-64-32 with fixed hyperparameters) was replaced by tuned configurations (Section 6).
The final estimators in the main experiment are:

| Estimator | Hidden layers | lr | Dropout | Weight decay | Batch | Max epochs | Patience | Optimiser |
|---|---|---|---|---|---|---|---|---|
| `mle` | — (CPT) | — | — | — | — | — | — | — |
| `nn_linear` | none (linear softmax) | 1e-3 | 0 | 1e-4 | 64 | 200 | 20 | Adam |
| `nn_64` | 64 | 6e-4 | 0.1 | 0.01 | 128 | 1000 | 50 | AdamW |
| `nn_128` | 128 | 6e-4 | 0.1 | 0.01 | 128 | 1000 | 50 | AdamW |
| `nn_256` | 256 | 6e-4 | 0.1 | 0.01 | 128 | 1000 | 50 | AdamW |
| `nn_64_32` | 64, 32 | 6e-4 | 0.1 | 0.01 | 128 | 1000 | 50 | AdamW |
| `nn_128_64` | 128, 64 | 3e-4 | 0.1 | 0.01 | 128 | 1000 | 50 | AdamW |
| `nn_256_128` | 256, 128 | 3e-4 | 0.1 | 0.01 | 128 | 1000 | 20 | AdamW |

(Source: `results7-parallel/chosen_nn_kl.json` and the `BASELINES` in `3_main_experiments.ipynb`.)
None of the final models has three hidden layers, and none uses tanh. All use ReLU, a common
dropout of 0.1 and weight decay of 0.01.

## 5. Evaluation (`ndbn/evaluation.py`)

### KL divergence

The primary metric is $D_{KL}(P^* \,\|\, \hat P)$ between the ground-truth joint and the learned
joint, as proposed. **Changed from proposal (how it is computed):** the sum over all $r^n$ joint
states is intractable at these sizes, so the implementation uses the chain-rule decomposition and
a Monte Carlo estimate over the 10,000 test rows $x^{(s)} \sim P^*$:

$$D_{KL}(P^* \| \hat P) = \mathbb{E}_{P^*}\Big[\sum_i D_{KL}\big(P^*(X_i \mid \mathrm{pa}_i)\,\|\,
\hat P(X_i \mid \mathrm{pa}_i)\big)\Big] \approx \frac{1}{S}\sum_{s}\sum_i \sum_{x_i}
P^*(x_i \mid \mathrm{pa}_i^{(s)}) \log\frac{P^*(x_i \mid \mathrm{pa}_i^{(s)})}{\hat P(x_i \mid
\mathrm{pa}_i^{(s)})}.$$

The inner sum over each child's states is exact; only the parent configurations are sampled.
Each estimate is reported with its standard error. Across the five DAGs, the mean and spread are
reported, as proposed.

### Scalability metrics (as proposed)

- **Parameter count:** stored and free parameters. For a CPT, free = $(r-1) \times$ number of
  configurations; every NN weight counts as free.
- **Memory:** bytes as float64, so that CPTs and NNs are comparable (native NN bytes are recorded
  too).
- **Training time:** wall-clock and process CPU time. Training times are taken only from a
  separate **serial** timing run (`results_timing.csv`), because parallel runs inflate them.
- Early-stopping diagnostics for the NNs: mean stop epoch, best epoch, and the fraction of nodes
  that hit the epoch ceiling.

Plots (`ndbn/plotting.py`) show KL against in-degree, network size and sample size (one curve per
in-degree or estimator), plus memory, parameters and training time against in-degree.

## 6. Hyperparameter tuning (`2_hyperparameter_tuning.ipynb`)

The proposal says only that a validation subset is used for early stopping and hyperparameter
selection, and that tuning is done on smaller configurations first. The implemented procedure:

- **Separate tuning networks.** All tuning uses DAGs 101–105, drawn from the same generator as the
  evaluation DAGs 1–5 but disjoint from them. No evaluation test set influences any choice.
- **Tuning grid:** $n = 20$, $\alpha = 0.5$, binary, $k \in \{2, 6, 10\}$, $N \in \{600, 1400\}$,
  validation fraction 0.2.
- **Initial architecture sweep** (fixed arms: linear, 32, 64-32, 128-64, plus no-weight-decay,
  high-lr and long-training variants of the 32-unit model), and a no-early-stopping run at
  $k=10$ to look at overfitting.
- **Bayesian hyperparameter sweeps** (Weights & Biases, 50 trials per in-degree, $k = 10, 6, 2$).
  Each trial is scored on every tuning DAG × sample size. Search space:
  hidden layers {32, 32-32, 64, 128, 256, 32-16, 32-16-8}; lr {6e-5, 1e-4, 3e-4, 6e-4, 1e-3};
  dropout {0, 0.1, 0.3}; weight decay {0, 0.01}; patience {10, 20, 50}; batch {32, 64, 128};
  AdamW; at most 1000 epochs. Further candidates with larger layers (e.g. 128-64, 256-128) were
  added and run like sweep trials.
- **Selection criterion:** mean validation cross-entropy (summed over nodes) across the tuning
  DAGs. True KL on the tuning DAGs is shown for reference only. Configs are compared **paired**
  per DAG: a config counts as "top" if its mean excess over the best is within 2 paired standard
  errors.
- **Simplifications found during tuning:** weight decay had a negligible effect on KL, so it is
  fixed at 0.01, and dropout is fixed at 0.1. Learning rates are snapped to a 3/6 grid.
- **Final choice:** six ReLU MLPs spanning width and depth (the table in Section 4). Each was
  re-run on every tuning in-degree / sample size before being frozen into `chosen_nn_kl.json`.

## 7. Real-data extension (`4_real_data.ipynb`, `ndbn/real_data.py`)

**Added beyond the proposal**, which named validation on real medical data as a limitation and
future work.

- **Data:** CDC Diabetes Health Indicators (UCI 891, BRFSS 2015): 253,680 rows, 22 variables,
  target `Diabetes_binary`.
- **Structure:** the expert-elicited DIABETES DAGs from Bayesys (Zahoor, Constantinou, Curtis &
  Hasanuzzaman, 2025), at the *high* and *moderate* expert-confidence levels (37 and 50 edges).
  Only the DAG is taken from experts; all parameters are learned.
- **Discretisation:** continuous and ordinal columns must be discretised (a limitation noted in
  the proposal). **L1** reproduces the published table (mostly binary; checked against the
  paper's marginals, within 1 percentage point). **L2** keeps native ordinal cardinalities (e.g.
  Age 13, Income 8, WHO BMI classes), so parent-configuration counts are much larger.
- **Metric:** there is no ground truth, so KL is replaced by **held-out log-likelihood** (mean
  over test rows of $\sum_i \log \hat P(x_i \mid \mathrm{pa}_i)$, higher is better). Also
  reported: the **unseen-configuration rate** (test rows whose parent configuration never occurs
  in training) and per-node log-likelihood against the number of parent configurations.
- **Design:** the rows are shuffled and cut into 25 disjoint blocks of 10,000. Each block is split
  60/20/20 into train / validation / test (stratified on diabetes status). Dataset sizes
  $N \in \{500, 2000, 10000\}$ are nested subsets of each block. NNs train on the training part
  and early-stop on validation; tabular estimators fit on training + validation.
- **Candidates:** MLE; Dirichlet pseudocount ∈ {0.1, 0.5, 1.0}; neural `nn_linear`, `nn_128_64`
  and `nn_256_128` with lr ∈ {3e-4, 1e-3, 3e-3} (AdamW, weight decay 0.01, batch 128, up to 1000
  epochs, patience 50).
- **Two-stage selection:** stage 1 fits every candidate and records only validation
  log-likelihood. One candidate per (discretisation, DAG, N, family) is then selected by mean
  validation score across blocks. Stage 2 refits the selected candidates and scores them once on
  the test parts.
- `results6-real/` holds an earlier version of this experiment that used 5-fold cross-validation.

## 8. Limitations (carried over, updated)

- Synthetic networks are binary, with a fixed in-degree per node and Dirichlet CPTs. They do not
  reproduce the noise, correlation structure or class imbalance of clinical data. The real-data
  extension addresses this only partly.
- Discrete variables only. Continuous clinical measurements are discretised, which loses
  information, and the results depend on the discretisation (hence L1 vs L2).
- A neural CPD is less directly readable than a CPT: the graph remains interpretable, but each
  local distribution has to be queried rather than read off a table. Neural CPDs also cost more
  training time and need hyperparameter tuning.
- The hyperparameters were tuned at $n = 20$, $r = 2$, $\alpha = 0.5$ and carried over unchanged
  to the other network sizes and to real data (where only lr was re-tuned).

## 9. Summary of changes from the proposal

| Area | Proposal | Implementation |
|---|---|---|
| Ground-truth scenarios | 4 density/in-degree scenarios (Table 3.1) | One fixed-in-degree generator; density recorded, not controlled |
| Grid | $n$ 5–50, $k$ 1–10 | $n \in \{10,20,30,40\}$, $k \in \{2,4,6,8,10\}$, binary, $\alpha = 0.5$, 5 DAGs |
| Sample sizes | Fixed across $k$ (unspecified) | 200 / 600 / 1000 / 1400 (train + val) |
| Test set | Not needed | 10,000-row test sample per ground truth, used only for Monte Carlo KL |
| KL computation | Sum over joint states | Chain-rule decomposition, exact per child, Monte Carlo over parent configurations, with SE |
| Architectures | Linear, 32, 64-32, 128-64-32 (ReLU/tanh) | Linear + six tuned ReLU MLPs: 64, 128, 256, 64-32, 128-64, 256-128 |
| Hyperparameters | Fixed per architecture (Table 3.2) | Tuned with W&B Bayesian sweeps on separate tuning DAGs; AdamW, dropout 0.1, wd 0.01, batch 128, ≤1000 epochs, patience 20–50 |
| Model selection | Validation subset | Paired validation-loss comparison across tuning DAGs (2 SE rule) |
| Input encoding | One-hot vs learned embeddings | One-hot only |
| Tabular baselines | MLE | MLE, plus Dirichlet pseudocount (BPE) estimator |
| Timing | Recorded | Recorded; reported only from a separate serial run |
| Real data | Future work | CDC Diabetes + Bayesys expert DAGs, held-out log-likelihood, two-stage selection |
