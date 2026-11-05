# Review: prioritised problems and fixes

State reviewed: commit `765c71c` ("notebook cleanup"), i.e. `report/main.tex`, `report/supplementary.tex`,
`report/references.bib`, the four notebooks, `config.py`, `ndbn/`, `results-synthetic/results.csv` (800 configs
per estimator, Dirichlet included), and `results-realdata/{results_c4,results_tuned,tuning,selection,results_timing}.csv`.
The previous, longer version of this file is in git history (`765c71c:report/REVIEW.md`). Items are moved
to **Resolved** at the end of this file once they're fixed.

**P0** = must fix before submission (wrong or unsupported claims, or the report can't be built or reproduced).
**P1** = a marker will likely object. **P2** = polish.

---

## P0: must fix

### P0-1. The synthetic conclusions are contradicted by the Dirichlet baseline
**Problem.** The abstract, the intro ("reducing divergence… by over 8-fold"), §4.2–4.3 and the conclusion
claim that neural CPDs beat tabular estimation. They only beat **unsmoothed MLE**, whose KL is about 92–95 %
the artificial `log(1e-12)` floor on zero cells. Against the Dirichlet estimator, **Dirichlet beats nn_32 in
800/800 configurations**, and beats the best tuned network in 798/800. This is expected rather than a bug. The
true CPT columns are i.i.d. `Dir(0.5)`, so the posterior-mean estimator `(N_ijk+0.5)/(N_ij+0.5r)` is
Bayes-optimal for expected KL, and no estimator can beat it on this generator. §4.1 still has the red
`\pending{Dirichlet KL…}` marker.

**Solution.**
- **§4.1:** add the Dirichlet numbers. At n=40 and N=1000, Dirichlet goes from 0.07 to 3.81 nats as k goes from 2 to
  10; MLE goes from 0.18 to 57.9. Delete `\pending`.
- **§3.3:** add one sentence on why Dirichlet(0.5) is optimal here, and change "a demanding test for any estimator that
  relies on sharing" to "sharing cannot help in expectation".
- **Reframe the synthetic study.** Neural CPDs close most of the gap between MLE and the Bayes-optimal
  estimator without knowing the prior; Dirichlet/nn_32 KL is 0.41–0.58 at r=2 and 0.50–0.80 at r=4. The
  advantage over MLE comes from regularisation and shrinkage, not from learned cross-configuration structure.
- **Make the real data the headline evidence** (P0-4). Remove "over 8-fold" and every "outperform tabular
  estimation" claim that rests on synthetic data.
- Rewrite the abstract and the conclusion to match.

### P0-2. Numbers in the text that the current results don't support
**Problem.** Several numbers come from the superseded experiment (`backup/results5-old-main-experiment`) or
are simply wrong:

| Location | Text says | Current results say |
|---|---|---|
| §4.3 | networks grow 5→40 nodes; MLE 0.05→15.7, nn_32 0.02→1.85; ratio 2.8→8.5 | n=10→50: MLE 1.35→38.0, nn_32 0.26→5.30, ratio 5.2→7.2 (Dirichlet 0.11→2.72) |
| §4.3 | α=0.1 gives 1.9×, α=1 gives 12×, nn_32 insensitive to α | no α ≠ 0.5 runs exist |
| §4.2 | nn_32 averages 1.95× the best tuned network's KL | **1.17×** (r=2), **1.075×** (r=4) |
| §3.5, §4.2 | nn_32 "trains substantially faster (202 s vs >900 s)" | nn_32 202 s; tuned nets 61–92 s at r=2. At r=4, nn_32 takes 67 s and nn_256 20 s. All are parallel wall-clock times |
| §4.1 | 40-node MLE KL "from 0.28" | 0.18 |
| §4.4.2 | Dirichlet "0.6 nats per row worse" at L3/low, N=500 | about 1.15–1.2 nats with the untuned Dirichlet (re-quote from whichever file the figures use, P0-4) |
| §4.2 | "ranking among the MLPs does not change with N or k" | best-of counts at r=2: nn_256 231, nn_128_64 97, nn_64_32 58, nn_256_128 57, nn_128 42, nn_64 15 (of 500) |
| Intro | "over 8-fold in the most data-scarce regime" | the r=2 ratio is 6.9 at ρ=0.2 and peaks around 7.5 at ρ≈0.8. Moot once P0-1 is done |

**Solution.** Replace each number with the current value, or drop the α claims. The §4.3 n=5/α paragraph
should be rewritten around Fig 4 as it now is (network size only, with Dirichlet). Drop the "trains faster"
justification for nn_32. Use "1.4–158× fewer parameters than the tuned nets (20-node networks) and within 17 % (r=2) / 8 % (r=4) of
their KL" instead.

### P0-3. The methods section doesn't describe what was actually run
**Problem → solution.**

| Report says | What ran | Fix |
|---|---|---|
| 20 % validation hold-out (§3.4.2) | **30 %** in every synthetic NN run; 20 % on real data | State both |
| Table 2: `nn_32` untuned (Adam, wd 1e-4, batch 64, patience 20) | `nn_32` is tuned. r=2 (`3e4d22b3`): AdamW, lr 6e-4, wd 0.01, dropout 0.1, batch 32, patience 50. r=4 (`48990a9c`): wd 0, dropout 0.3, batch 64 | Rewrite Table 2 from `results-tuning/chosen_nn_c{2,4}.json`. Only `nn_linear` is untuned (Adam defaults) |
| Tuned nets "wd 0.01" | r=4 configs use wd 0.0 | Per-cardinality columns in Table 2 |
| Table 2 lists 7 estimators | Results and Fig 2 also contain `nn_128`, `nn_64`, `nn_256_128`, `nn_64_32` (r=2) and `nn_32_16` (r=4) | List them all |
| Max epochs not stated | 1000 for tuned nets, 500 for `nn_linear` | State it |
| Sweep trials "scored by mean KL" | Notebook 2 selects on `mean_val_loss` | State the metric used for each final choice |
| Table 1: "900 configurations", † marks the r=4 subset, 220 tuned; Limitations: "tuned networks were run on a subset" | 800 per estimator: r=2 is 5n×5k×4N×5 DAGs = 500; r=4 is n 10–30 × k 2–10 × N 200–1400 × 5 = 300 | Update Table 1 and the counts; delete the "subset" limitation |
| §3.7: training time "recorded separately in a serial run" | True for real data only; synthetic times are parallel | Say so, or run P1-10 |
| §3.6 real-data NNs | Configs come from **r=4 synthetic tuning** (`chosen_nn_c4.json`), plus an N-dependent batch-size rule (`real_data.at_n`) and lr/pseudocount tuning per fold | Describe all three |
| Unseen rate: "fraction of test parent configurations never observed" | Code: fraction of **test rows** with at least one node whose parent configuration is unseen | Fix the definition (and the S4 caption) |
| §4.1 "unseen share rises from 0.5 % to 78 %" | Not in any saved result | Add it to the results or remove it |

### P0-4. The real-data section doesn't reflect the tuned results
**Problem.** The real-data figures and text come from `results_c4.csv`, with a fixed pseudocount of 0.5 and fixed
learning rates. The tuned run (`tuning.csv` → `selection.csv` → `results_tuned.csv`) exists but isn't
described anywhere. It is also the strongest result in the project: **nn_32 and nn_128_64 beat the tuned
Dirichlet in 108/108 fold-level comparisons** (smallest margin 0.004 nats/row).

**Solution.** Decide which file drives §4.4: `results_tuned.csv` is the defensible one. Point notebook 4's
`REAL_CSV` at it if so, and regenerate the figures. Add a short paragraph on the tuning protocol and its
justification (P1-6). Report the 108/108 result and this table of mean nn_32 − Dirichlet (nats/row, tuned):

|  | N=500 | 2 000 | 10 000 | 50 000 |
|---|---|---|---|---|
| L1 | 0.35 | 0.20 | 0.08 | 0.03 |
| L2 | 0.84 | 0.56 | 0.31 | 0.14 |
| L3 | 1.03 | 0.71 | 0.43 | 0.21 |

Also report paired fold differences ± s.e., and say plainly that the gain is small at N=50 000.

### P0-5. Results don't come from one reproducible code version
**Problem.** `results.csv` mixes rows from four code versions (`3c0ff2d`, `ddd750e`, `98f2a70`, `0df292c`, all
with uncommitted changes) and records torch 2.14.1+cu130, while this venv has 2.14.0.
- 100 r=2 MLE rows (`3c0ff2d`) don't reproduce from the cached data.
- Re-fitted nn_256 rows were 12 % and 29 % off; nn_32 matched exactly in one cell and was 1.4 % off in another.

So "all estimators see the same networks and data" (§3.3) isn't guaranteed.

**Solution.** Commit, tag (e.g. `v-report`), then re-run notebook 3 (both cardinalities, all estimators) into a
fresh `results.csv` on one machine and one torch version, and regenerate the figures. Record the tag and torch
version in the report or README. Seeding is sound: the Dirichlet rows and 580/680 MLE rows reproduce exactly.

### P0-6. The report can't be built from the repo, and `main.pdf` is stale
**Problem.**
- `report/main.pdf` was compiled from the older draft, so it still shows "under a minute", the old figures, and more.
- `supplementary.tex` includes `figs/no_es.pdf`, `figs/sweep.pdf`, `figs/dag_moderate.png` and
  `figs/real_unseen.pdf`. **None of these exist.** The PNGs are `real_unseen.png` and `sweep_c2.png`/`sweep_c4.png`;
  `no_es` and the DAG figure haven't been generated.
- `supplementary.tex` has no `\graphicspath`.
- The `cas-dc` class files aren't in the repo.

**Solution.**
- Fix the four paths and add `\graphicspath{{./}{../}}` to `supplementary.tex`.
- Generate S1 (P1-4) and S3: add a `dot.render("figs/dag_moderate", format="png")` line to notebook 4 cell
  `2df90528`.
- Commit the class files, or note where to get them.
- Recompile both PDFs and commit them.

### P0-7. No README or requirements file
**Problem.** The README is one line, and there's no `requirements.txt`. The venv is Python 3.14, torch 2.14,
pgmpy 1.1.2 (whose estimator API changes between versions), numpy 2.5, pandas 3.0, networkx 3.6, scikit-learn 1.9,
wandb 0.30, psutil, tqdm, joblib, ucimlrepo, matplotlib 3.11 and graphviz.

**Solution.**
- Add `requirements.txt` with pinned direct dependencies, and `pip freeze > requirements-lock.txt`.
- Write a README with:
  - setup instructions;
  - the run order of notebooks 1–4, what each writes, its rough runtime, and that `RUN_* = False` reuses saved
    results;
  - a figure/table → notebook cell → CSV map;
  - data sources and licences: CDC Diabetes Health Indicators (UCI 891, CC BY 4.0), and the expert DAGs from
    Zahoor et al. / Bayesys;
  - that W&B is optional (`WANDB_MODE=disabled`).

---

## P1: a marker will likely object

### P1-1. Related work is thin and misses the direct prior art
**Problem.** About a page long with 16 citations, and no prior work on neural or parametric CPDs except an
unpublished thesis. The intro also promises a subsection on "approaches that combine predictive performance
with explainability" that doesn't exist.

**Solution.** Restructure into: (i) tabular estimation and smoothing → (ii) structured/parametric local models →
(iii) neural conditionals in graphical/causal models → (iv) interpretable clinical models and BNs in medicine.
End with a gap paragraph. Add (verify each citation before using it):
- Neal 1992 (sigmoid belief networks); Bengio & Bengio 2000 (one-hot inputs, per-variable softmax); NADE
  (Larochelle & Murray 2011); MADE (Germain et al. 2015).
- Tree-CPDs and context-specific independence (Boutilier et al. 1996; Friedman & Goldszmidt 1996); decision
  graphs (Chickering et al. 1997); logistic CPDs and noisy-MAX (Koller & Friedman ch. 5).
- BDeu (Heckerman et al. 1995); equivalent-sample-size sensitivity (Silander et al. 2007; Steck 2008).
- Parameter constraints (Niculescu et al. 2006); monotonicity (Feelders & van der Gaag 2006). These back your
  Future Work.
- Neural conditionals on DAGs (GraN-DAG, NOTEARS-MLP, Deep SCMs, neural causal models).
- GA²M (Caruana et al. 2015) and Neural Additive Models. These are relevant because your additive linear
  softmax is competitive on real data.
- Classic medical BNs: ALARM, Pathfinder, Hepar II.
- Use the **15 uncited entries already in `references.bib`** (XAI in medicine, InterpretML and others) for (iv).
- Give Gaogle (2026) a one- to two-sentence summary of its model and results, since a marker can't look it up.

### P1-2. Interpretation claims that overreach
**Problem → solution.**
- Fig 5 / §4.3: "advantage falls steadily as ρ increases". The MLE/nn_32 ratio is **not monotone**: it peaks
  near ρ≈0.8 at both r=2 and r=4. Describe the shape, and the Dirichlet/nn_32 line that is now in the figure.
- §4.3: "cost… dominated by early-stopping patience at high in-degree, where fewer epochs are needed" is
  self-contradictory. Use `stop_epoch_mean` from the results, or delete it.
- Contribution 1 ("characterises where tabular estimators fail") is a known result. Call it a controlled
  confirmation.

### P1-3. Measurement choices that need stating
**Problem → solution.**
- KL is a **sum over nodes**, but figures and the iso-KL table (Table 4) pool n=10–50, so the 5-nat threshold
  means different things at different n. Use per-node KL wherever sizes are pooled: divide by `n_nodes`, no
  re-run needed. §3.7 already promises this.
- The MLE floor ε=1e-12 dominates MLE's KL. State this, or report the zero-cell share (92–95 %).
- "BPE" appears before it is defined. Define it at first use.
- L3 BMI decile edges are computed on the full table, test folds included. State this, or compute the edges per
  training fold.
- Add the ethics clearance or waiver reference if the faculty requires one.

### P1-4. The supplement is incomplete or inconsistent
**Problem → solution.**
- **S1:** the no-early-stopping figure has no data on tuning DAGs. Run it: in notebook 2, set `RUN_SWEEPS=True`
  and run only the setup cells, the no-early-stopping run and its figure cell. That takes about 10 minutes on
  12 cores.
- **S2:** the caption says r=2, so use `sweep_c2.png`. §3.5 says 256-unit and two-layer networks "dominated at
  both cardinalities", but the r=4 sweep (`sweep_c4.png`) only searched small networks. Correct the text.
- **S3:** not generated (see P0-6).
- **The abstract** lists S1–S3 and Table S1, but S4 (`real_unseen`) is included too. Add it.
- **Table S1:** I haven't verified its values against `hparam_trials.csv`. Check them.

### P1-5. "Medical diagnosis" in the title, but no diagnostic evaluation
**Problem.** Only joint log-likelihood is reported. No diagnostic query is evaluated.

**Solution.** For each fitted real-data model, compute P(Diabetes | all other variables) on the test fold. It
needs only the Markov-blanket factors: the diabetes CPD times its children's CPDs, normalised over the two
states. Report AUROC, Brier score and log-loss against N.

**Status: implemented, full run pending.**
- `real_data.diagnosis_probs` / `diagnosis_scores` compute it. It matches pgmpy variable elimination and the
  full joint to about 1e-15.
- Every test-scored real-data row now records `diag_auroc`, `diag_brier` and `diag_logloss`.
- Notebook 4 has a "Diagnosis" section: `RUN_DIAGNOSIS` refits the tuned configs into
  `results_diagnosis.csv` and checks the refits reproduce `results_tuned.csv`. It then prints the tables and
  fold-level wins, and saves `figs/real_diagnosis.png`.
- Models aren't stored, so it does need a refit, about 30–45 min with 10 workers.
- A smoke run (L1/high, fold 0) gave AUROC of about 0.79–0.80 for the networks against 0.69–0.74 for Dirichlet.
  At N=500, Dirichlet's Brier score and log-loss are *worse than predicting the prevalence*.
- Diabetes is the highest in-degree node in every DAG, so this is where the CPT bottleneck bites most. Put it in
  §4.4 and the abstract.

### P1-6. The real-data tuning protocol needs a stated justification and wider grids
**Problem.**
- Selecting once per fold, pooled over all 36 cells, is defensible but isn't explained.
- Both winners sit at the **edge** of their grids: pseudocount 1.0 of {0.1, 0.5, 1.0}, and lr 3e-3 of
  {3e-4, 1e-3, 3e-3}. Dirichlet's validation score is still improving at 1.0.

**Solution.**
- Justify pooled selection on three grounds:
  - **variance:** each cell's validation set is 100 rows at N=500;
  - **the prior argument:** a pseudocount encodes the prior, so its optimum shouldn't depend on N;
  - **an identical protocol for both families.**
  
  Don't claim tabular estimators "cannot" be tuned per cell.
- Extend the grids to pseudocount {2, 5, 10} (or a BDeu equivalent sample size) and lr {1e-2}. Tabular fits are
  cheap.
- Add an **oracle check**: for each cell, take the best pseudocount on the test fold. That is an upper bound for
  any per-cell tuning. If nn_32 still beats it, the granularity objection is closed.

### P1-7. Missing limitations
**Problem → solution.** Add:
- the real-data NN configurations were tuned on synthetic r=4 networks;
- there are only 3 folds;
- no diagnostic query was evaluated (unless P1-5 is done);
- the synthetic generator has no shared structure, which favours Dirichlet (unless P1-8 is done).

### P1-8. The synthetic study has no case where sharing can help
**Problem.** With independent Dirichlet columns, the synthetic grid can't show the effect the real data does.

**Solution (if time).** Add a ground-truth family *with* shared structure, run mle, dirichlet, nn_linear and
nn_32 on the 20-node slice, and report it. Options:
- logistic CPDs over the one-hot parents with a few pairwise interactions;
- noisy-OR / noisy-MAX;
- low-rank or tree-structured CPTs.

### P1-9. Repository hygiene
**Problem → solution.**
- `backup/` (1 746 files, 324 MB) and `.claude/settings.json` are tracked. Run
  `git rm -r --cached backup .claude` and add both to `.gitignore`.
- `results_tuned.csv` holds every row three times (identical values; see the Resolved run_selected bug).
  Deduplicate it on `["disc_level","dag_level","fold","n_train_spec","estimator"]`, or regenerate it.
- Notebook 2 still has unguarded W&B cells: `%pip install wandb`, `wandb.login()`, and the candidate cell, which
  calls `run_configs` with a W&B project. "Run all" would prompt for a login and re-check candidates. Put them
  under `RUN_SWEEPS`.
- The notebooks were executed out of order: notebook 3's counts are `[13,14,15,16,8,10,18,19,…]`, and notebook 4 is
  partly unexecuted. After P0-5, restart and run all with `RUN_* = False`, so the saved outputs match the
  results.

### P1-10. Synthetic timing claims are based on parallel runs
**Problem.** §3.7 promises serial timing. Synthetic times in §3.5 and §4.2–4.3 are parallel wall-clock times,
mixed across code versions.

**Solution.** Run notebook 3's serial timing cell on the 20-node slice for every estimator. Or drop the
synthetic timing claims, and keep the real-data serial timing, which is correct.

### P1-11. Bibliography errors
**Problem → solution.**
- `koller2007graphical` has a bogus volume and pages (43, 1359–1366). It is a chapter in Getoor & Taskar (eds.),
  *Introduction to Statistical Relational Learning*, MIT Press, pp. 13–55.
- Zahoor et al.: the code says 2025, the bib says the 2024 arXiv preprint. Cite the published version
  consistently.
- Add volume and pages to Di 2021, Hulsen 2023, Kyrimi 2021, Oniśko 2001/2013 and Polotskaya 2024, and protect
  capitals ("{B}ayesian", "{XAI}").

---

## P2: polish

| Problem | Solution |
|---|---|
| The abstract has no numbers | Add the real-data 108/108 result and the size of the gain at small N |
| No uncertainty on synthetic figures | Paired win-rates, or bootstrap CIs on the KL ratios |
| `frac_at_ceiling` in `results.csv` compares against the global `N_EPOCHS`, not each estimator's own `n_epochs`, so it reports 0.5–0.74 for nets that never hit the cap | Pass `est.get("n_epochs", n_epochs)` to `evaluate` in `_run_dag_job` |
| Sensitivity of the synthetic conclusions to the 30 % validation split and to ε | One run each on the 20-node slice: 20 % vs 30 % validation; ε=1e-6 vs 1e-12 |
| `fit_mle` comment says "normal default" (it is uniform) | Fix the comment |
| No tests | `tests/`: Dirichlet equals its closed form; `kl_calculation` equals the exact KL on a 3-node network; NN probabilities sum to 1; seeding is deterministic |
| `\pending` / `\provisional` macros still in the preamble | Remove them before submission |
| `Research_Proposal_3163666.pdf` (with a student number) is in a public repo | Your call |

---

## Appendix: evidence for P0-1 (synthetic grid, mean total KL in nats)

| r | k | MLE | Dirichlet | nn_linear | nn_32 | best tuned |
|---|---|---|---|---|---|---|
| 2 | 2 | 0.60 | **0.11** | 1.85 | 0.20 | 0.20 |
| 2 | 4 | 3.52 | **0.35** | 4.51 | 0.76 | 0.70 |
| 2 | 6 | 13.68 | **0.94** | 6.15 | 2.24 | 1.93 |
| 2 | 8 | 31.20 | **1.99** | 6.86 | 4.25 | 3.74 |
| 2 | 10 | 38.75 | **3.02** | 7.03 | 5.36 | 5.08 |
| 4 | 2 | 7.58 | **0.66** | 4.39 | 1.31 | 1.18 |
| 4 | 4 | 61.56 | **3.31** | 7.89 | 6.09 | 5.67 |
| 4 | 6 | 59.53 | **5.58** | 8.32 | 7.40 | 7.28 |
| 4 | 8 | 37.97 | **6.31** | 8.44 | 7.56 | 7.47 |
| 4 | 10 | 30.38 | **6.58** | 8.46 | 7.63 | 7.54 |

Means are over network sizes, DAGs and N. "Best tuned" is the architecture with the lowest mean at that k.

---

## Resolved

| Was | Now |
|---|---|
| Dirichlet missing from the synthetic results | Run on all 800 configs. It matches my independent re-fit to 1e-15 |
| Synthetic figures lacked Dirichlet / had a one-point α panel | Regenerated with Dirichlet; `size_alpha` is single-panel; captions updated |
| `report/main.tex` was the older draft | Replaced by the newer draft (`main_old.tex`) plus the 5 Nov edits; `\graphicspath` added |
| `references.bib` missing | Added; all 16 cited keys resolve |
| Report figures not produced by any notebook | Every `figs/*.png` comes from a notebook cell and regenerates byte-identical |
| Three settings modules after the merge; `TUNING_DIR` pointed at a missing folder | `config.py` is the only one; notebooks use `from config import *` |
| `RUN_SWEEPS` / `RUN_TUNING` guards lost | Restored, both `False` |
| Real-data Dirichlet pseudocount untuned | Tuned on validation per fold; nn_32 still wins 108/108 (see P0-4 / P1-6) |
| Tuned networks run only on a subset of the grid | Every estimator ran on the full grid (only the report text is out of date: P0-3) |
| §4.2 claimed "parameter sharing alone is not the mechanism; the hidden layer is" (was P1-2) | Reworded: the linear softmax fails because an additive model can't fit CPTs with full parent interactions (bias, flat in N), so sharing helps only with enough flexibility |
| `results7-parallel/` (was P1-9) | Held the only cached r=4, k=10 synthetic data. Verified as the data `results.csv` used (refits match; fresh simulation reproduces it), moved into `results-synthetic/data/card_4/…/indegree_10`, folder removed |
| `results6-real/` (was P1-9) | Not a rename: the earlier 5-fold real-data experiment, unused. Moved to `backup/results6-real/` |
| `run_selected` fitted every fold's winner in every fold, because `_ests` ignored the `fold` key (tripling `results_tuned.csv` and its compute) | `_ests` now filters list entries by `fold`. The existing triplicated file is still open under P1-9 |
