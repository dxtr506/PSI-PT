#!/usr/bin/env bash
# Timing of the truncation-region computation on a rented CPU (Linux).
# Run from anywhere:  bash experiments/run_timing.sh
# Results: results/timing/*.csv (+ .json with the config and lscpu);
# figures: results/timing_workers_median.pdf, results/timing_p_median.pdf (+ .png)
set -e
cd "$(dirname "$0")/.."                      # the repository root
export PYTHONPATH="$PWD"
# one BLAS / numba thread per process, so that the number of workers is the only parallelism
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1

OUT=results/timing
N_TESTS=100         # p-values per setting
WORKERS_FIXED=16    # workers of the p sweep

# 1. number of workers (n_T = 50, p = 500); one worker (no parallelism) is timed as well
python experiments/timing_experiment.py --nts 50 --ps 500 --workers 4 8 16 \
    --n_tests $N_TESTS --out $OUT/timing_workers.csv

# 2. number of features p (n_T = 50), z-segments and per-fit only
python experiments/timing_experiment.py --nts 50 --ps 250 500 750 1000 --workers $WORKERS_FIXED \
    --n_tests $N_TESTS --no_seq --out $OUT/timing_p.csv

# figures: median time per p-value
python experiments/plot.py timing
