#!/usr/bin/env python3
"""
Compute confidence intervals for the reported test-split results WITHOUT
re-running any training.

What this does and does not cover
---------------------------------
Per-class recall is a binomial proportion: k correct out of n test cases of
that class. The Wilson score interval therefore gives an exact-coverage
confidence interval for EVALUATION-SET SAMPLING uncertainty — how much the
number could move if we drew a different test set of the same size.

This is NOT a confidence interval over training runs. Seed-to-seed variability
requires repeated training and remains unquantified. The two sources of
uncertainty are different and we report only the one the data supports.

The evaluation CI is nonetheless decisive for the zero-recall findings: a
recall of 0/1327 melanoma cases has a Wilson upper bound of ~0.3%, so
"FedAvg does not detect melanoma" is not a small-sample artefact.

    python eval_intervals.py [results_dir]
"""
import json
import math
import sys
from pathlib import Path

CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
Z = 1.959963985  # 95%


def wilson(k: int, n: int, z: float = Z):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    arms = [("3way_true_fedavg.json", "No decomposition"),
            ("3way_decomp_nodrift.json", "Decomposition only"),
            ("3way_full.json", "FedSA-Drift")]

    print("95% Wilson intervals on per-class recall — TEST split")
    print("(evaluation-set sampling uncertainty; NOT seed variability)\n")

    for fname, label in arms:
        p = root / fname
        if not p.exists():
            print(f"[skip] {fname}")
            continue
        d = json.load(open(p))
        s = d["splits"]["test"]
        pcs = s["per_client"]
        shared = s["mode"] == "shared"

        print(f"=== {label} ({'shared model' if shared else f'{len(pcs)} clients'}) ===")
        print(f"{'class':6s}{'n':>7}{'recall':>9}{'95% CI':>20}")
        for c in CLASSES:
            n = pcs[0]["support"][c]
            if shared:
                r = pcs[0]["per_class_recall"][c]
                k = round(r * n)
                lo, hi = wilson(k, n)
                print(f"{c:6s}{n:7d}{r:9.3f}   [{lo:.3f}, {hi:.3f}]")
            else:
                # All clients score the SAME test images, so their results are
                # correlated: pooling them as n*K independent trials would
                # understate the interval badly. Instead give each client its
                # own interval on n cases, and report the across-client mean as
                # a descriptive statistic with the per-client range.
                rs = [m["per_class_recall"][c] for m in pcs]
                bounds = [wilson(round(r * n), n) for r in rs]
                mean_r = sum(rs) / len(rs)
                lo = min(b[0] for b in bounds)
                hi = max(b[1] for b in bounds)
                print(f"{c:6s}{n:7d}{mean_r:9.3f}   "
                      f"[{lo:.3f}, {hi:.3f}] (union over clients, "
                      f"per-client n={n})")
        print()

    # Overall accuracy CI for the shared arm (one model, one test set).
    d = json.load(open(root / "3way_true_fedavg.json"))
    s = d["splits"]["test"]
    n = sum(s["per_client"][0]["support"].values())
    acc = s["mean_acc"]
    lo, hi = wilson(round(acc * n), n)
    print(f"No decomposition, overall test accuracy: {acc:.4f} "
          f"[{lo:.4f}, {hi:.4f}] on n={n}")

    d = json.load(open(root / "3way_full.json"))
    s = d["splits"]["test"]
    accs = [m["acc"] for m in s["per_client"]]
    n = sum(s["per_client"][0]["support"].values())
    print("FedSA-Drift, per-client test accuracy with 95%% CI (n=%d each):" % n)
    for i, a in enumerate(accs):
        lo, hi = wilson(round(a * n), n)
        print(f"   client {i}: {a:.4f} [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
