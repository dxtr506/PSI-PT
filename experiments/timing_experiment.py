import argparse
import csv
import json
import os
import platform
import subprocess
import time

import numpy as np

from data_generation import generate_contrasts, generate_design, generate_response
from psi_pt.inference import construct_setup, parallel_divide_and_conquer_practical
from psi_pt.utils import calculate_a_b, construct_Sigma, construct_test_statistic
from psi_pt.practical_transmission import practical_transmission_CV

# Time of the truncation region of one p-value (TPR setting):
#   seq - one process (divide_and_conquer_practical), the reference without parallelism
#   seg - z-segments, one per worker (Parallel-PPL-SI)
#   fit - per-fit parallelism (default of parallel_divide_and_conquer_practical)
# Every (n_T, p) setting draws its own n_tests tested repetitions; seq is timed once per repetition,
# seg and fit once per repetition and number of workers.


def tested_repetitions(args, nt, p):
    # the first n_tests repetitions with a tested feature, drawn as in run_experiment.py
    rng = np.random.default_rng(args.seed + nt)
    beta0 = np.zeros(p)
    candidates = np.arange(args.S)
    beta0[candidates] = args.delta
    reps = []
    while len(reps) < args.n_tests:
        X0, X_list = generate_design(rng, p, nt, args.ns, args.K)
        contrasts = generate_contrasts(rng, p, args.K, args.H)
        y0, y_list = generate_response(rng, X0, X_list, args.sigma, beta0, contrasts)
        M = np.sort(practical_transmission_CV(X0, y0, X_list, y_list, V=args.V)["feature_selection"])
        pool = np.intersect1d(M, candidates)
        if len(pool) == 0 or len(M) > nt:
            continue
        j = pool[rng.integers(len(pool))]
        S = construct_setup(X0, y0, X_list, y_list, args.V)
        Sigma = construct_Sigma(args.sigma ** 2 * np.eye(nt), [args.sigma ** 2 * np.eye(args.ns)] * args.K)
        etaj, _ = construct_test_statistic(j, S["X0"], S["Y"], M)
        a, b = calculate_a_b(etaj, S["Y"], Sigma)
        reps.append((S, a, b, M, np.sqrt(etaj @ (Sigma @ etaj))))
        print(f"n_T={nt} p={p}: repetition {len(reps)}/{args.n_tests}, |M|={len(M)}", flush=True)
    return reps


def machine_info():
    info = {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "threads": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")}}
    try:
        info["lscpu"] = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    except OSError:
        pass
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nts", type=int, nargs="+", default=[50])
    parser.add_argument("--ps", type=int, nargs="+", default=[500])
    parser.add_argument("--workers", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--n_tests", type=int, default=10, help="p-values timed per (n_T, p) setting")
    parser.add_argument("--no_seq", action="store_true", help="skip the one-process reference")
    parser.add_argument("--ns", type=int, default=200)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--V", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--S", type=int, default=10)
    parser.add_argument("--H", type=float, default=10.0)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=700000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"config": vars(args), "machine": machine_info()}, open(args.out + ".json", "w"), indent=1)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["nt", "p", "n_jobs", "rep", "model_size", "method", "seconds", "solves", "boxes", "z_pieces",
                    "same_Z"])
        for nt in args.nts:
            for p in args.ps:
                reps = tested_repetitions(args, nt, p)
                runs = ([("seq", 1, 1)] if not args.no_seq else []) + \
                       [(m, nj, nj if m == "seg" else None) for nj in args.workers for m in ("seg", "fit")]
                # warm-up: compile numba in the main process and in every pool before timing
                S, a, b, M, sd = reps[0]
                for _, nj, n_seg in runs:
                    parallel_divide_and_conquer_practical(S, a, b, M, -0.5 * sd, 0.5 * sd, nj, n_seg)
                for k, (S, a, b, M, sd) in enumerate(reps, 1):
                    ref = None
                    for i in range(len(runs)):  # rotate the order across repetitions
                        m, nj, n_seg = runs[(i + k) % len(runs)]
                        t = time.perf_counter()
                        Z, counts = parallel_divide_and_conquer_practical(S, a, b, M, -20 * sd, 20 * sd, nj, n_seg)
                        sec = time.perf_counter() - t
                        Z = np.array(Z)
                        if ref is None:
                            ref = Z
                        same = Z.shape == ref.shape and np.allclose(Z, ref, rtol=0, atol=1e-9 * sd)
                        w.writerow([nt, p, nj, k, len(M), m, f"{sec:.3f}", counts["fits"], counts["boxes"],
                                    len(Z), same])
                        print(f"n_T={nt} p={p} rep {k} {m} workers={nj}: {sec:.1f}s, {counts['fits']} solves, "
                              f"{counts['boxes']} boxes, Z same={same}", flush=True)
                    fh.flush()


if __name__ == "__main__":
    main()
