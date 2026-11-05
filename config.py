"""Settings shared by every notebook: `from config import *` at the top of each.

Only settings with the same value in every notebook live here. Settings that
belong to one notebook (its grid, CARD, VAL_FRAC, sample sizes, estimators, ...)
stay in that notebook's settings cell.
"""
import os as _os

# ── Folders ─────────────────────────────────────────────────────────────────
TUNING_DIR  = "results-tuning"      # sweep trials, candidates, chosen configs, tuning-DAG data
RESULTS_DIR = "results-synthetic"   # synthetic main experiment (3_main_experiments)
REAL_DIR    = "results-realdata"    # real-data experiment (4_real_data)
FIG_DIR     = "figs"                # report figures (PNG); exploratory plots stay in each results folder's plots/

# ── Files ───────────────────────────────────────────────────────────────────
RESULTS_NAME = "results.csv"                                    # one results file for every estimator
RESULTS_CSV  = _os.path.join(RESULTS_DIR, RESULTS_NAME)
TRIALS_CSV   = _os.path.join(TUNING_DIR, "hparam_trials.csv")    # tuning data: sweep trials + candidates
CHOSEN_JSON  = _os.path.join(TUNING_DIR, "chosen_nn_c4.json")    # final NN configs: written by notebook 2,
                                                                 # read by notebooks 3 and 4

# ── Synthetic DAGs ──────────────────────────────────────────────────────────
SIM             = 5       # evaluation DAGs per setting (1..SIM)
TUNE_DAG_OFFSET = 100     # tuning DAGs are TUNE_DAG_OFFSET+1, ...: never the evaluation DAGs
N_TEST          = 10000   # test rows per DAG (KL estimate only)
N_EPOCHS        = 200     # max epochs for NN CPDs without their own n_epochs

# ── Speed ───────────────────────────────────────────────────────────────────
N_JOBS          = -1      # worker processes; -1 = all cores, 1 = serial
DEVICE_OVERRIDE = "cpu"   # None = auto (cuda if available); tiny per-node nets run best on CPU
TORCH_THREADS   = 1       # torch threads per worker

VERBOSE = True

assert TUNE_DAG_OFFSET >= SIM, "tuning DAGs overlap the evaluation DAGs"
for _d in (TUNING_DIR, RESULTS_DIR, REAL_DIR, FIG_DIR):
    _os.makedirs(_d, exist_ok=True)
