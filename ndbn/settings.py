"""Shared experiment settings, imported by each notebook.

Replaces the load_notebook_config mechanism where notebooks exec'd each
other's tagged cells.  Each setting lives here once; notebooks import what
they need and override locally where their grid differs.
"""
import os

# ── Speed settings (all notebooks) ──────────────────────────────────────
N_JOBS          = -1      # worker processes; -1 = all cores
DEVICE_OVERRIDE = "cpu"   # tiny per-node nets run best on CPU
TORCH_THREADS   = 1       # torch threads per worker

# ── Base synthetic grid (card-2 tuning, MLE baseline) ───────────────────
NODE_COUNTS  = [10, 20, 30]
IN_DEGREES   = [2, 6, 10]
ALPHAS       = [0.1, 0.5]
SIM          = 5
CARD         = 2
SAMPLE_SIZES = [600, 1400]
VAL_FRAC     = 0.2
N_TEST       = 10_000
N_EPOCHS     = 200

# ── Paths ───────────────────────────────────────────────────────────────
TUNING_DIR   = "results5-parallel"
RESULTS_NAME = "results.csv"

VERBOSE = True
