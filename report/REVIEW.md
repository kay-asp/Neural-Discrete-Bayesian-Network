# Critical review of the honours report and repository

Reviewed: `report/main.tex` and `report/main.pdf` (compiled 2026-11-05), `ndbn/`, the four notebooks, `config.py`,
`results-synthetic/results.csv`, `results-realdata/results_c4.csv` / `results_timing.csv`,
`results-tuning/*`. I re-computed every number I could trace to a saved result. I also re-fitted MLE and
Dirichlet estimators on the cached synthetic data (script: see "Verification run" below).

**Re-checked 2026-11-05 against the merged results (commit `b5d608b`).** This covers the full synthetic grid for
every estimator, including `dirichlet` (800 configurations: 500 at r=2, 300 at r=4), and the new real-data
`tuning.csv`, `selection.csv` and `results_tuned.csv`. `report/main.tex` and `main.pdf` are unchanged, so every
report-text issue below still stands. Items fixed by the new runs are marked ✅.

Legend: **[P0]** must fix before submission (correctness / validity) · **[P1]** should fix (clarity, consistency,
reviewer objections) · **[P2]** nice to have.

---

## 1. Headline problems (read these first)

### 1.1 [P0] The synthetic comparison is against a straw-man baseline, and the right baseline is Bayes-optimal by construction

* The ground-truth CPT columns are drawn i.i.d. from `Dir(0.5,…,0.5)` (`ndbn/generation.py: gen_CPTs`). The
  columns are independent, so data from other parent configurations carries no information about a given
  column. The posterior of column *j* is `Dir(0.5 + N_ij·)`. Its mean, `(N_ijk+0.5)/(N_ij+0.5r)`, is
  **exactly the `dirichlet` estimator with pseudocount 0.5**. The posterior mean minimises expected
  `KL(P*‖P̂)`. So in this synthetic setup **no estimator, neural or otherwise, can beat Dirichlet(0.5) in
  expectation**.
* The report compares neural CPDs only against **unsmoothed MLE**. Its KL is mostly the artificial
  `log(1e-12)` floor on zero cells (`evaluation.EPS`). Example: n=50, k=10, N=200, DAG 1 has MLE KL = 83.1 nats.
  **77.4 nats of that come from floored zeros.** On the same data, Dirichlet(0.5) gets 9.8 and nn_32 gets 13.2.
* So "reducing divergence by over 8-fold", "nn_32 beats MLE in 98.8 % of configurations", Table 4 (iso-KL)
  and Figure 5 (ρ ratio) mostly measure **how MLE handles zero cells**, not the value of neural CPDs.
* ✅ Dirichlet is now in `results.csv`. Your rows match my independent re-fit to 1e-15 on all 680 overlapping
  configurations. **It confirms the point above.** Dirichlet(0.5) beats nn_32 in **800 / 800** configurations.
  It beats the best of all tuned networks in 798 / 800: the exceptions are 2 r=2 cells, and r=4 has none.
  The report text does not use these results yet, and §4.1 still has the red `[PENDING: …]` marker.

See §5 for the Dirichlet numbers on the full grid and how to reframe this.

### 1.2 [P0] The report mixes results from two different experiments

These numbers come from the **old** main experiment (`backup/results5-old-main-experiment/results.csv`: n ∈
{5,…,40}, α ∈ {0.1, 0.5, 1}, untuned `nn_32_relu`). They cannot be reproduced from the current
`results.csv`, which has α = 0.5 only and n ∈ {10,…,50}:

| Report text (§4.3) | Source | Current results |
|---|---|---|
| "networks grow from 5 to 40 nodes, MLE 0.05→15.7 vs nn_32 0.02→1.85, ratio 2.8→8.5" | old exp. | n=10→50: MLE 1.35→38.0, nn_32 0.26→5.30, ratio 5.2→7.2 |
| "α=0.1 … 1.9×; α=1 … 12×, nn_32 insensitive to α" | old exp. | no α≠0.5 runs exist |
| "nn_32 averages 1.95× the KL of the best tuned" (§4.2) | not traceable | **1.17×** (r=2, 500 configs); 1.075× (r=4, 300) |
| "nn_32 within 3 % of best at k≥6, within 30 % at k=2" (r=4) | ✓ still holds | 2–3 % at k≥6, 19.5 % at k=2 (the full grid now includes k=10) |
| "nn_32 … mean 202 s vs >900 s per network" (§3.5, §4.2) | wrong direction | nn_32 202 s; tuned nets **61–92 s** (r=2). At r=4: nn_32 67 s, nn_256 20 s. All parallel wall-clock |
| "nn_256 and nn_128_64 alternate as best" | roughly ✓ | best-of counts r=2: nn_256 231, nn_128_64 97, nn_64_32 58, nn_256_128 57, nn_128 42, nn_64 15; r=4: nn_128_64 132, nn_256 114, nn_64_32 45, nn_32_16 9 |
| "in 40-node networks KL rises from 0.28" (§4.1) | — | 0.18 (n=40, N=1000, k=2) |
| "Dirichlet … 0.6 nats per row worse … L3/low, N=500" (§4.4.2) | — | **1.15–1.20 nats** |
| "MLE and Dirichlet fit in under a minute at every N" (§4.4.3) | contradicted | Dirichlet **147–164 s** at L3/low (serial timing) |

Figure 4(b) (`size_alpha_c2.png`) now shows **a single point** at α = 0.5, so panel (b) carries no information.

### 1.3 [P0] The methods section does not describe what was run

| Report says | Code / results say |
|---|---|
| 20 % validation hold-out (§3.4.2) | **30 %** for every synthetic NN row (`n_fit = 0.7·N`; `VAL_FRAC = 0.3` in nb 3). Real data: 20 % ✓ |
| Table 2: `nn_32` is "untuned", Adam, wd 1e-4, batch 64, patience 20 | `nn_32` **is tuned**. r=2: `3e4d22b3` = AdamW, lr 6e-4, wd 0.01, dropout 0.1, batch 32, patience 50. r=4: `48990a9c` = wd 0, dropout 0.3, batch 64 |
| Table 2: tuned nets "wd 0.01" | r=4 configs have **wd 0.0** and dropout 0.0–0.3 |
| Table 2 lists 7 estimators | results also contain `nn_128`, `nn_64`, `nn_256_128`, `nn_64_32` (r=2) and `nn_32_16` (r=4), all plotted in Fig 2 |
| Max epochs not stated | 1000 for tuned/selected nets; 500 (`NN_HPARAMS`) for `nn_linear` |
| §3.5: sweep trials "scored by mean KL" | nb 2 sets `HP_METRIC = "mean_val_loss"`, and the candidate lists are labelled "chosen by kl" vs "small by val loss". State which one picked each final config |
| Table 1 / text: "900 configurations per estimator"; † marks for the r=4 subset; Limitations: "tuned networks were run on a subset" | **Now 800 per estimator**: r=2 is n 10–50 × k 2–10 × N 200–1400 × 5 DAGs = 500; r=4 is n 10–30 × k 2–10 × N 200–1400 × 5 = 300. Every estimator ran on the full grid. Update Table 1 (r=4 now includes k=10 and N=200), the counts, and remove the "subset" limitation |
| §3.7: training time "recorded separately in a serial run" | true for real data only. No synthetic serial timing file exists, so every synthetic time quoted is a **parallel** wall-clock (`n_jobs = -1`) and the rows come from three code versions |
| §3.6 real-data NNs | They use the **r=4 synthetic configs** (`chosen_nn_c4.json`) plus an N-dependent batch-size rule (`real_data.at_n`). The `*_fixedbatch` control runs are not mentioned |
| §3.6 real-data tuning (new) | `tuning.csv` / `selection.csv` show per-fold selection on validation log-lik: pseudocount ∈ {0.1, 0.5, 1.0} and NN lr ∈ {3e-4, 1e-3, 3e-3}, one choice per fold for all 36 cells. **Both winners are at the edge of their grid** (pseudocount 1.0, lr 3e-3), and the Dirichlet validation score improves monotonically 0.1 → 0.5 → 1.0. The report says neither that this tuning happened nor which results (`results_c4` or `results_tuned`) the figures use |
| Unseen rate: "fraction of test parent configurations never observed" | `unseen_config_rate` gives the fraction of **test rows** with ≥1 node whose parent configuration is unseen. That is a different quantity |

### 1.4 [P0] The compiled PDF is out of sync with the source and with the results

* `report/main.tex` includes `figs/*.pdf`. Those files were deleted from `figs/` and the root `main.tex` was
  deleted. The PDF was built (in Prism) from the **old** PDFs (`old-figs/`). Example: PDF Fig 2 is r=2 only
  with x = in-degree, but the caption says r ∈ {2,4} and the text says the x-axis is rᵏ.
* ✅ `report/references.bib` has been added; all 16 cited keys resolve. It also holds 15 uncited entries (XAI in
  medicine, InterpretML, Rudin-adjacent work). These are a ready-made start on the missing
  "interpretable prediction" related-work subsection.
* **`main_old.tex` is the newer draft, not `report/main.tex`.** `report/main.tex` is the 30 Oct committed
  `main.tex` (`6f7f3d9`) plus two small 5 Nov edits: the `\texorpdfstring` author fix, and removal of the
  `\provisional` "smoothed tabular estimator not included" sentence. `main_old.tex` (saved 31 Oct 00:42) adds
  later edits on top of `6f7f3d9`:
  * paths to the current `figs/*_c2.png` / `real_*.png` files;
  * section-level citations, such as `\citep[Sec.~17.2]{…}`;
  * an accurate real-data timing paragraph that matches `results_timing.csv`;
  * the `real_params` / `real_timing` captions that match the actual figures.
  The `report/main.pdf` red-flag numbers in §1.2 ("under a minute", for example) come from using the older file.
  ✅ **Done:** `report/main.tex` is now `main_old.tex` with the two 5 Nov edits re-applied. I also added
  `\graphicspath{{./}{../}}` so it compiles from `report/`, and updated the captions of the three figures that
  changed (`mle_fails`, `size_alpha` now single-column, `rho`). `main_old.tex` is removed; it remains in git
  history. Not compiled here (no TeX install). The body text still has every §1–2 issue.
* ✅ **Report figures now all come from the notebooks, into `figs/`.** The merge "keep local notebooks"
  (`b5d608b`) had discarded the report-figure cells from notebooks 2–4. These are now restored:
  * nb 3: the five synthetic figures;
  * nb 4: `real_timing`, `real_params`, and `real_loglik/mean/unseen`, which now write to `figs/`;
  * nb 2: `no_es.png` and `sweep_c2/c4.png`.
  The real-data figures regenerate byte-identical to the committed ones. Remaining gaps:
  * **S1 (`no_es`)** draws nothing until the no-early-stopping run exists on tuning DAGs
    (`results-tuning/histories_no_es_tune` is missing).
  * **S2:** the old `figs/sweep.png` was identical to `sweep_c4.png`, the r=4 *small-network* sweep (32, 64,
    16-16, 32-16). That contradicts §3.5 "256-unit and two-layer networks dominated the sweeps at both
    cardinalities"; only the r=2 sweep (`sweep_c2.png`) shows those sizes.
  * **S3:** nb 4 cell `2df90528` draws the three expert DAGs with graphviz, but only displays them; it never
    saves them to `figs/`.
  * `ndbn/settings.py` had `TUNING_DIR = "results5-parallel"`, a folder that does not exist. It is fixed to
    `results-tuning` and adds `FIG_DIR = "figs"`.
* Supplementary Figures S1–S4 are cited four times, but no supplement exists. S1 (`no_es`) still uses
  **evaluation** DAGs 1–5 from `backup/results4-old-sweep-seed`, which contradicts "All tuning was performed on
  a separate set of tuning DAGs" (§3.5).

---

## 2. Report: section-by-section issues

### Abstract / Introduction
* [P1] The abstract has no numbers. Add one synthetic and one real-data figure, with the baseline named.
* [P0] The claim "neural CPDs … reduce divergence … over 8-fold in the most data-scarce regime" has two
  problems. (a) It is relative to unsmoothed MLE. (b) It is not even true at the most data-scarce point: the
  ratio is 6.6 at ρ = 0.2, and the peak is ≈7.5 at ρ ≈ 0.8 (r=2) and ≈9.0 at k=4 (r=4).
* [P1] The contribution "characterises where and how tabular estimators fail" is a known result (you cite
  Koller & Friedman for it in §4.1). Reframe it as a replication or controlled confirmation.
* [P1] The roadmap promises related work on "approaches that combine predictive performance with
  explainability". No such subsection exists.

### Related work
See §3 below. The section is about one page long, cites 16 works in total, and has no direct prior work on
neural or parametric CPDs other than an unpublished thesis.

### Methodology
* [P1] "BPE" is used in §3.1 before it is defined in §3.4.1.
* [P1] §3.3 says "Five random DAGs are generated per setting", but the r=4 grid and the tuned-network subset
  are different grids. Give one table of exactly which (n, k, N, r) cells each estimator ran on, matching
  the 500/220/180 counts.
* [P1] §3.3 claims independent columns make the ground truth "a demanding test for any estimator that relies
  on sharing". The more precise statement: with independent columns, sharing *cannot* help in expectation (see
  1.1). What the NN actually gives here is **shrinkage toward a smooth default**, not learned cross-configuration
  structure. This changes the interpretation of §4.2 ("the hidden layer is the mechanism"). Consider adding a
  synthetic family **with** shared structure, e.g. logistic / noisy-OR / low-rank ground-truth CPTs. That is
  where neural CPDs *should* beat Dirichlet (see §6).
* [P1] KL is a **sum over nodes**, and several figures and Table 4 average it over n = 10…50. The 5-nat iso-KL
  threshold therefore means different things for a 10-node and a 50-node network. Use per-node KL everywhere
  sizes are pooled (the text already promises this in §3.7).
* [P1] The MLE floor of 1e-12 is an arbitrary design choice that dominates the MLE numbers. State its effect,
  or report MLE with the zero-cell contribution separated out.
* [P2] L3 BMI decile edges are computed on the full table, test folds included (`real_data.discretise`). This
  is minor, but state it, or compute the edges per training fold.
* [P2] Ethics: give the faculty ethics-waiver/clearance reference if Wits requires one for secondary public
  data.

### Results
* [P0] §4.1 needs the Dirichlet curve, and the `[PENDING]` marker must go.
* [P1] §4.1 "share of parent configurations never seen rises from 0.5 % to 78 %". No saved result contains
  this, so add it to `results.csv` or remove it.
* [P1] §4.2 "ranking among the MLPs does not change with sample size or in-degree". Best-architecture counts at
  r=2 are nn_256 231, nn_128_64 97, nn_64_32 58, nn_256_128 57, nn_128 42, nn_64 15 (of 500), so the ranking
  does change. Soften this, or show per-cell rank agreement.
* [P1] §4.2 "Parameter sharing alone is therefore not the mechanism; the hidden layer is". The linear softmax
  is a mis-specified additive model of CPTs that have full interactions. Its failure shows additivity is wrong
  here; it does not show that sharing is irrelevant.
* [P1] §4.3 / Fig 5: the ratio is **not monotone**. It peaks at ρ ≈ 0.8 and is lower at ρ = 0.2–0.6, where
  both estimators sit near their "uniform/marginal" floor. Describe it as such.
* [P1] §4.3 "The cost is per-node and dominated by early-stopping patience at high in-degree, where fewer
  epochs are needed…". This sentence is self-contradictory. Use the saved `stop_epoch_mean` instead.
* [P1] §4.4.2 "All three neural architectures outperform the Dirichlet estimator at every sample size".
  This holds on the mean. At fold level, nn_32 and nn_128_64 win **108/108**, and nn_linear wins 105/108,
  losing at L1/high/N=50 000. Report it that way: it is a strong, clean result. Also report paired fold
  differences ± s.e.
  ✅ It also survives the new tuning. In `results_tuned.csv`, nn_32 and nn_128_64 still beat the tuned
  Dirichlet (pseudocount 1.0) in **108/108** fold-level comparisons (smallest margin 0.004 nats/row). Mean
  nn_32 − Dirichlet in nats/row:

  | | N=500 | 2 000 | 10 000 | 50 000 |
  |---|---|---|---|---|
  | L1 | 0.35 | 0.20 | 0.08 | 0.03 |
  | L2 | 0.84 | 0.56 | 0.31 | 0.14 |
  | L3 | 1.03 | 0.71 | 0.43 | 0.21 |

  Make this the headline real-data result. Say which results file each real-data figure uses: they currently
  come from `results_c4.csv`, the untuned run.
* [P1] Note that the real-data gains are large at small N and fine discretisation and small at N = 50 000 (see the table above).
* [P1] The title says "Medical Diagnosis", but no diagnostic quantity is evaluated. Add P(Diabetes | other
  variables) via the Markov blanket, with AUROC, Brier score and log-loss on the test folds (see §6).
* [P2] Fig 1 (`mle_fails_c2.png`) uses a much larger font than the other figures and a linear y-axis. Its
  panel labels are "(a) KL divergence / (b) CPT memory" in KB, while the caption says "Total CPT memory".
  Harmonise the style.
* [P2] No uncertainty is shown on any synthetic figure. 5 DAGs per cell is fine for the paired comparison,
  but add paired win-rates or bootstrap CIs on the ratios.

### Conclusion / limitations
* [P0] Rewrite once the Dirichlet baseline is in (see §5): "neural CPDs outperform tabular estimation" is not
  supported on synthetic data against a properly smoothed tabular estimator.
* [P1] Add these limitations: the Dirichlet pseudocount was fixed at 0.5 while the NNs were tuned (unequal
  tuning budget); the real-data NN configs were tuned on synthetic r=4 networks; there are only 3 folds; no
  diagnostic query was evaluated.

### References / typesetting
* [P1] Koller et al. 2007 "Graphical models in a nutshell" has bogus bibliographic data ("43, 1359–1366"). It
  is a chapter in Getoor & Taskar (eds.), *Introduction to Statistical Relational Learning*, MIT Press,
  pp. 13–55.
* [P1] Zahoor et al.: the code comments cite "(2025) Table 1" and the bib entry is the 2024 arXiv preprint.
  Cite the published version if there is one, and be consistent.
* [P2] Several bib entries lack volume and pages (Di 2021, Hulsen 2023, Kyrimi 2021, Oniśko 2001/2013,
  Polotskaya 2024). Title-case protection is missing ("bayesian", "xai").
* [P2] Remove the `\pending`/`\provisional` macros before submission.

---

## 3. Related work: coverage and relevance

The current section covers MLE/Dirichlet, CaMAP, noisy-OR, BNs in medicine (2 reviews and one application),
XAI, Rudin, and Gaogle (2026). For a paper whose contribution is "neural CPDs for discrete BNs", the
**directly comparable prior work is missing**. A marker will likely ask "how is this different from X?" for:

1. **Neural / parametric CPDs and neural autoregressive models.** A neural CPD on a fixed DAG is a
   (partially connected) neural autoregressive model:
   * Neal (1992) *Connectionist learning of belief networks* (sigmoid belief networks);
   * Bengio & Bengio (2000) *Modeling high-dimensional discrete data with multi-layer neural networks*, which
     uses exactly one-hot inputs, a softmax per variable and parameter sharing;
   * Larochelle & Murray (2011) NADE; Germain et al. (2015) MADE.
2. **Structured local CPDs** (Koller & Friedman 2009, ch. 5): tree-CPDs / context-specific independence
   (Boutilier et al. 1996; Friedman & Goldszmidt 1996), decision graphs (Chickering, Heckerman & Meek 1997),
   logistic/generalised linear CPDs, noisy-MAX. These are the classical answer to the "rᵏ problem" and are the
   natural interpretable competitors.
3. **Prior choice and smoothing for tabular CPTs.** BDeu (Heckerman, Geiger & Chickering 1995) and the
   sensitivity of the equivalent sample size (Silander, Kontkanen & Myllymäki 2007; Steck 2008). This matters
   because your synthetic baseline result hinges on it.
4. **Parameter sharing and constraints from knowledge.** Niculescu, Mitchell & Rao (2006, JMLR) *Bayesian network
   learning with parameter constraints*; monotonicity constraints (Feelders & van der Gaag 2006); Zhou, Fenton &
   Neil (2014) on expert judgement for BN parameters. Your Future Work already proposes monotonic NNs, so cite
   these.
5. **Neural conditionals inside causal/graphical models.** GraN-DAG (Lachapelle et al. 2020), NOTEARS-MLP
   (Zheng et al. 2020), Deep SCMs (Pawlowski et al. 2020), and neural causal models (Xia et al. 2021). These
   show that neural CPDs on a DAG are an established idea, and your positioning should acknowledge it. Your
   contribution is the empirical study for *discrete, sparse-data, clinical* parameter estimation.
6. **Interpretable models for clinical tabular data** as the alternative to BNs: GA²Ms (Caruana et al. 2015,
   the pneumonia case) and Neural Additive Models (Agarwal et al. 2021). These are especially relevant because
   your linear softmax (an additive model) is competitive on the real data.
7. **Classic medical BNs** to ground the motivation: ALARM (Beinlich et al. 1989), Pathfinder (Heckerman et al.
   1992), Hepar II (Oniśko et al.).
8. **The dataset.** Prior BRFSS/CDC-diabetes modelling work, beyond Zahoor et al., to place the log-likelihood
   numbers.

Relevance check on what is already there: Rudin (2019) and Hulsen (2023) support the motivation well. Zhong
(2024) is a narrow single application; one sentence from it is enough. Gaogle (2026) is unpublished, so a
marker cannot check it. Summarise its exact model and results in one or two sentences so the "extension"
claim stands on its own.

Suggested structure: (i) tabular estimation and smoothing → (ii) structured/parametric local models →
(iii) neural conditionals in graphical/causal models → (iv) interpretable clinical models and BNs in medicine.
End with a short "gap" paragraph: no systematic study of neural CPDs vs **properly smoothed** CPTs as a
function of rows per configuration, on real clinical data with an expert DAG.

---

## 4. Repository and reproducibility

| Pri | Issue | Fix |
|---|---|---|
| P0 | After the merge (`b5d608b`, "keep local notebooks…") there are **three settings modules**. `config.py` is committed but **no notebook imports it**. The notebooks import `ndbn/settings.py`, which has a different base grid (`ALPHAS = [0.1, 0.5]`, `IN_DEGREES = [2, 6, 10]`…). `ndbn/notebook_config.py` is marked deprecated. Notebook 1 is `1_setup_and_checks.ipynb` again, so the restructure in `a88a46e` was partly undone | pick one: either `config.py` or `ndbn/settings.py`. Delete the other two, and check that each notebook's settings cell reproduces the grid in `results.csv` |
| P0 | ✅ `report/` is committed and `references.bib` is added (commit it). The class files are still not in the repo | commit the class file (`cas-dc.cls`, `cas-common.sty`, `cas-model2-names.bst`) or a note on where to get them; point `\graphicspath{{../figs/}}` at the PNGs |
| P0 | No `requirements.txt` / `environment.yml`. The venv is Python 3.14.4, torch 2.14.0, pgmpy 1.1.2, numpy 2.5.3, pandas 3.0.5, networkx 3.6.1, scikit-learn 1.9.1, wandb 0.30.0, plus psutil, tqdm, joblib, ucimlrepo, matplotlib 3.11.2 | `pip freeze > requirements-lock.txt`, plus a short hand-written `requirements.txt` with pinned versions of the direct deps. pgmpy's estimator API changes between versions, so pin it exactly |
| P0 | README is one line | see the template below |
| P0 | **Part of `results.csv` does not reproduce with the current code and cache.** Mixed rows come from four code versions (`3c0ff2d`, `ddd750e`, `98f2a70`, `0df292c`, all `+dirty`). (a) 100 r=2 MLE rows (`3c0ff2d`, N=1000 slice, n≠20) differ from a re-fit on the cached data. (b) I re-fitted two cells in this checkout: nn_32 reproduced exactly in one cell (n=10, k=2) and was 1.4 % off in the other (n=20, k=4). nn_256 rows from `3c0ff2d` were 12 % and 29 % off. (c) The CSV records torch 2.14.1+cu130, but this venv has 2.14.0. So within a cell, estimators may not have seen identical data or code, and the "paired comparison" claim is not guaranteed | commit, tag `v-report`, and re-run every estimator from the tag in one environment (§6 item 7). Record the tag and the torch version in the report |
| P1 | `results7-parallel/` (cached data) and `backup/` are now tracked in git | see the `backup/` row below | commit, tag `v-report`, re-run at least the main grid from the tag (see §6), and record the tag in the report |
| P1 | Notebooks were run out of order: nb 3 execution counts are `[None,42,43,44,46,47,32,None,35,10]`, nb 4 is executed only up to cell 12, and nb 3's settings cell shows the r=4 run (`CARD = 4`, `VAL_FRAC = 0.3`), not the r=2 one | add the r=2 and r=4 settings as named presets (or a `CARD` loop). Restart-and-run-all each notebook once with `RUN_* = False` so the figures regenerate from the saved CSVs |
| P1 | `backup/` (1 746 files, 324 MB) and `results6-real/` are tracked in git | remove from the index (`git rm -r --cached backup results6-real`), add to `.gitignore`, and archive elsewhere |
| P1 | `.claude/settings.json` is tracked and holds machine-specific scratchpad paths | untrack it and add it to `.gitignore` |
| P1 | W&B is required to rerun sweeps (`entity kay-asp-wits-university`) | document `WANDB_MODE=offline` / disabled. Check that `RUN_SWEEPS = False` fully avoids W&B imports |
| P1 | The S1 no-early-stopping figure uses evaluation DAGs | re-run the no-ES cell in nb 2 on tuning DAGs 101–105 |
| P2 | Bug: `frac_at_ceiling` in `results.csv` compares stop epochs with the **global** `N_EPOCHS` (`_run_dag_job` → `evaluate(..., n_epochs)`), not the estimator's own `n_epochs` (1000). It reports 0.5–0.74 for nets that never hit their ceiling | pass `est.get("n_epochs", n_epochs)` to `evaluate`. The value is not used in the report, but anyone reading the CSV will be misled |
| P2 | `fit_mle` comment says "MLE with normal default", meaning uniform | fix the comment |
| P2 | No tests | a `tests/` with 3–4 checks: Dirichlet equals the closed form, `kl_calculation` equals the exact KL on a 3-node net, NN probabilities sum to 1, seeds are deterministic |
| P2 | The data licence and provenance are unstated | CDC Diabetes Health Indicators (UCI id 891, CC BY 4.0); `data/bayesys/*` from Zahoor et al. / Bayesys. Credit both in the README |
| P2 | `Research_Proposal_3163666.pdf` contains a student number in a public repo | your call |

**README template**

```
# Neural CPDs for discrete Bayesian networks (honours project)
Report: report/main.pdf  |  Code: ndbn/  |  Settings: ndbn/settings.py (or config.py — keep one)
## Setup      python 3.14; pip install -r requirements.txt
## Reproduce  (all notebooks read saved CSVs unless RUN_* = True)
1_setup_and_checks.ipynb       smoke tests, example DAG
2_hyperparameter_tuning.ipynb  sweeps → results-tuning/chosen_nn_c{2,4}.json   (~X h, W&B optional)
3_main_experiments.ipynb       CARD=2 and CARD=4 → results-synthetic/results.csv (~X h on N cores)
4_real_data.ipynb              → results-realdata/results_c4.csv, results_timing.csv (~X h)
## Figure → notebook/cell → CSV map   (one line per report figure/table)
## Data       CDC diabetes (UCI 891, CC BY 4.0); expert DAGs from Zahoor et al. (2024)
## Seeds      every random step seeded via ndbn.generation.make_seed
```

---

## 5. Verification run: Dirichlet on the synthetic grid

I re-fitted MLE, Dirichlet(0.5) and Dirichlet(1.0) independently on the cached train/test CSVs in
`results-synthetic/data`, covering the 680 configs on the original grid. **Your `dirichlet` rows in the merged
`results.csv` match my re-fit to 1e-15 on all 680.** The table uses your full 800-config results.

**Dirichlet(0.5) beats nn_32 in 800 / 800 configurations** and beats the best tuned network in 798 / 800.
That is what theory predicts (§1.1).

| r | k | MLE | Dirichlet | nn_linear | nn_32 | best tuned (mean) |
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

(Mean total KL in nats over network sizes, DAGs and N. "Best tuned" is the architecture with the lowest mean
at that k.) The Dirichlet/nn_32 ratio is 0.41–0.58 at r=2 and 0.50–0.80 at r=4, so the gap shrinks at high
in-degree when r=4. In my re-fit, **on average 92 % (r=2) and 95 % (r=4) of MLE's KL comes from cells
floored at 1e-12**. Even the report's own §4.1 example changes: at n=40, N=1000, Dirichlet goes from 0.07 to
3.81 nats as k goes 2→10, against MLE's 0.18 to 57.9.

**How to reframe.** On synthetic Dirichlet ground truth, a properly smoothed CPT is optimal, and neural CPDs
cannot beat it. The NN's advantage over MLE is regularisation, not learned sharing. One option is to present
the synthetic study honestly as "NNs recover most of the gap from MLE to the Bayes-optimal estimator without
knowing the prior". Then make the real data the main evidence: 108/108 wins over a validation-tuned Dirichlet.
That is a genuine positive finding, because real CPDs *do* have shared structure. Item 2 in §6, a synthetic
ground truth with shared structure, would let the synthetic study show the same thing under control.

---

## 6. Suggested runs (in priority order)

1. ✅ **Dirichlet on the full synthetic grid** is done and verified. What remains is to **rewrite §4.1–4.3, the
   abstract, the intro claims and the conclusion** around it (see §5).
2. **[P0] A synthetic family with shared structure** so the neural CPDs have something to share. Examples:
   ground-truth CPDs that are (a) logistic in the one-hot parents with a few pairwise interactions, (b) noisy-OR
   / noisy-MAX, (c) low-rank / tree-structured (context-specific). Under these, Dirichlet is no longer optimal,
   so a win for nn_32 over Dirichlet would be a real finding. Keep the same n/k/N grid (or the 20-node slice)
   and run mle, dirichlet, nn_linear and nn_32.
3. **[P0, mostly done] A tuned smoothed baseline on real data.** ✅ The pseudocount is now chosen on validation,
   and nn_32 still wins 108/108. **Remaining:** the winning pseudocount (1.0) and NN lr (3e-3) both sit at the
   *edge* of their grids, and Dirichlet's validation score is still rising at 1.0 (−16.55 → −16.51 at N=500).
   Extend the grids to pseudocount {1, 2, 5, 10} (or BDeu ESS) and lr {1e-2}. Tabular fits are cheap, so this
   costs minutes. Expect the gap to shrink a little at L2/L3 and small N, but probably not to flip.
   *Selection granularity:* one choice per fold, pooled over all 36 cells, was a deliberate choice and is
   defensible. Justify it in §3.6 on variance grounds (at N=500 each cell's validation set is 100 rows) and on
   prior grounds (a Dirichlet pseudocount encodes the prior, so its optimum should not depend on N; the NNs' N
   dependence is handled by the pre-specified batch rule). Say also that both families get the identical
   protocol. Do **not** argue that tabular estimators "cannot" be tuned per cell: they can, cheaply. To close
   the objection, add an *oracle* check: the best pseudocount per cell chosen on the test fold, an upper bound
   for any per-cell tuning. If nn_32 still beats that, per-cell selection cannot change the conclusion.
4. **[P1] Diagnostic evaluation on real data.** For each fitted model, compute P(Diabetes | all other
   variables) on the test fold. Only the Markov blanket factors are needed: the diabetes CPD × its children's
   CPDs, normalised over the 2 states, so no general inference engine is required. Report AUROC, Brier and
   log-loss vs N. This makes the "Medical Diagnosis" title true.
5. **[P1] Re-run the S1 no-early-stopping diagnostic on tuning DAGs** (nb 2, no-ES cell), so every tuning
   artefact is disjoint from evaluation.
6. **[P1] A serial timing run for the synthetic grid** (nb 3 already has the cell, writing
   `results_timing.csv`) on the 20-node slice for every estimator, so the §3.5/§4.2 time claims are measured.
7. **[P1] A clean re-run from a tagged commit** of at least the r=2 main grid for mle/nn_linear/nn_32, to remove
   the four-code-version mix. This has become **P0**, since some rows do not reproduce (see §4). Seeding itself
   works: Dirichlet and 580/680 MLE rows reproduce exactly, and so does nn_32 in one of the two cells I re-fitted.
   Re-run on the same machine and torch version as the rest, and pin that version.
8. **[P2] Sensitivity checks:** the validation fraction (20 % vs 30 %), because MLE/Dirichlet get the full N;
   the MLE floor ε ∈ {1e-12, 1e-6}, to show how much of MLE's KL is the floor; and α ∈ {0.1, 1} on the 20-node
   slice, if you want to keep the α discussion.
9. **[P2] Per-node KL** in all pooled figures and Table 4 (no re-run needed: divide by `n_nodes`).
