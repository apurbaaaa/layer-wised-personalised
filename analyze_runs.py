#!/usr/bin/env python3
"""
Aggregate federated run logs into the tables needed for the reviewer response.

Parses one or more ``logs/federated_*.log`` files, extracts each run's config
(from the ``Args`` JSON header) and its final-round metrics, then reports:

  * A per-configuration summary (global bal-acc, worst-client acc, inter-client
    std) with mean ± 95 % CI when several seeds are present.
  * The component ladder  true_fedavg → decomp_nodrift → full  at a fixed
    (alpha, K), isolating the decomposition step and the drift step.
  * Paired per-seed deltas (worst-client, std) between two arms when the same
    seeds exist for both — the statistically correct answer to "point estimates
    with no confidence intervals".

Usage
-----
    python analyze_runs.py logs/federated_*.log
    python analyze_runs.py logs/                 # scans a directory

Notes
-----
* Metrics are taken from the LAST logged round of each run.
* CIs use the Student-t interval; with n=1 the point estimate is reported and
  flagged as single-seed (no CI) — this is expected for the $10 matrix.
* true_fedavg has no personalized head, so its per-client "worst/std" on the
  shared global val set is degenerate (std≈0, worst == global). Those columns
  are therefore reported for true_fedavg but must NOT be read as a fairness
  comparison (see reviewer response, Comment 3).

* IMPORTANT — the "global_balacc" column is NOT comparable across arms.
  Group C (metadata_mlp + fusion_head) is the entire classifier and is never
  aggregated, so for the decomposed arms (full, decomp_nodrift) the global
  model keeps its RANDOM initial classifier for the whole run — verified
  empirically with diag_groupc.py (8/8 Group C tensors identical to init,
  while Group A had changed). true_fedavg has an empty Group C, so its
  classifier IS aggregated and trained. Comparing global accuracy between
  those two families conflates "decomposition removed" with "classifier
  trained vs untrained".

  Use MEAN PER-CLIENT accuracy for cross-arm comparison: every arm is then
  evaluated with a trained classifier (its own personalized head, or the
  single shared head for true_fedavg). That is the `mean_client_acc` column
  and the "comparable ladder" section below.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Final-round metric line, e.g.:
# Round  25/25 | Global Acc=0.5676 BalAcc=0.6987 F1=0.4688 AUC=0.9157 |
#   Worst Acc=0.5705 BalAcc=0.6330 | Std Acc=0.0511 BalAcc=0.0531 | 1441s
_ROUND_RE = re.compile(
    r"Round\s+(\d+)/(\d+)\s*\|\s*"
    r"Global Acc=([\d.]+)\s+BalAcc=([\d.]+)\s+F1=([\d.]+)\s+AUC=([\w.]+)\s*\|\s*"
    r"Worst Acc=([\d.]+)\s+BalAcc=([\d.]+)\s*\|\s*"
    r"Std Acc=([\d.]+)\s+BalAcc=([\d.]+)"
)

# 95 % two-sided Student-t critical values for small samples (df = n-1).
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


class Run:
    def __init__(self, path: Path, args: dict, metrics: dict):
        self.path = path
        self.args = args
        self.metrics = metrics

    @property
    def ablation(self) -> str:
        # Older runs predate --ablation. Infer: drift on → full;
        # drift off → decomp_nodrift (the paper's mislabeled "FedAvg").
        a = self.args.get("ablation")
        if a:
            return a
        return "full" if self.args.get("drift_weighting") else "decomp_nodrift"

    @property
    def key(self) -> Tuple[str, float, int]:
        return (self.ablation,
                float(self.args.get("dirichlet_alpha")),
                int(self.args.get("num_clients")))

    @property
    def seed(self) -> int:
        return int(self.args.get("seed", 42))

    @property
    def run_name(self) -> Optional[str]:
        return self.args.get("run_name")


def load_client_accs(run: Run, ckpt_root: Path) -> Optional[List[float]]:
    """
    Pull the final-round per-client accuracies from a run's checkpoint.

    Needed because the log line reports only worst/std, not the mean, and mean
    per-client accuracy is the only accuracy metric comparable ACROSS ablation
    arms (see the module docstring note on the frozen global classifier).
    """
    if not run.run_name:
        return None
    ck = ckpt_root / run.run_name / "last_federated.pt"
    if not ck.is_file():
        return None
    try:
        import torch
        blob = torch.load(ck, map_location="cpu", weights_only=False)
    except Exception:
        return None
    accs = blob.get("client_accs")
    if isinstance(accs, list) and accs:
        return [float(a) for a in accs]
    return None


def parse_log(path: Path) -> Optional[Run]:
    text = path.read_text(errors="ignore")

    # --- args JSON block (may span many lines after "Args: {") --------- #
    m = re.search(r"Args:\s*(\{.*?\n\})", text, re.DOTALL)
    if not m:
        return None
    try:
        args = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None

    # --- last round line ---------------------------------------------- #
    last = None
    for rm in _ROUND_RE.finditer(text):
        last = rm
    if last is None:
        return None

    g = last.groups()
    metrics = {
        "round": int(g[0]), "total_rounds": int(g[1]),
        "global_acc": float(g[2]), "global_bal_acc": float(g[3]),
        "f1": float(g[4]),
        "auc": float(g[5]) if g[5] not in ("nan", "NaN") else float("nan"),
        "worst_acc": float(g[6]), "worst_bal_acc": float(g[7]),
        "std_acc": float(g[8]), "std_bal_acc": float(g[9]),
    }
    return Run(path, args, metrics)


def mean_ci(vals: List[float]) -> Tuple[float, Optional[float]]:
    """Return (mean, half-width of 95 % t-CI) or (mean, None) if n < 2."""
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    se = math.sqrt(var / n)
    t = _T95.get(n - 1, 1.96)
    return mean, t * se


def fmt(mean: float, hw: Optional[float]) -> str:
    if hw is None:
        return f"{mean:.4f} (n=1, no CI)"
    return f"{mean:.4f} ± {hw:.4f}"


def main(argv: List[str]) -> None:
    paths: List[Path] = []
    ckpt_root = Path("checkpoints/federated")
    for a in argv:
        p = Path(a)
        if p.is_dir() and p.name != "federated":
            paths.extend(sorted(p.glob("federated_*.log")))
        elif p.is_dir():
            ckpt_root = p
        else:
            paths.append(p)
    if not paths:
        print("No log files given.\nUsage: python analyze_runs.py logs/federated_*.log")
        return

    runs: List[Run] = []
    for p in paths:
        r = parse_log(p)
        if r is not None:
            runs.append(r)
        else:
            print(f"[skip] could not parse {p.name}", file=sys.stderr)

    # --- group by config ---------------------------------------------- #
    by_key: Dict[Tuple, List[Run]] = defaultdict(list)
    for r in runs:
        by_key[r.key].append(r)

    # Per-client accuracies from checkpoints (for the comparable ladder).
    client_accs: Dict[Tuple, List[float]] = {}
    for key, rs in by_key.items():
        vals: List[float] = []
        for r in rs:
            ca = load_client_accs(r, ckpt_root)
            if ca:
                vals.append(sum(ca) / len(ca))
        if vals:
            client_accs[key] = vals

    print("\n=== Per-configuration summary (final round) ===")
    print("NOTE: global_balacc is NOT comparable across arms — see header.")
    print(f"{'ablation':<16}{'alpha':>6}{'K':>4}{'seeds':>7}   "
          f"{'global_balacc':<22}{'mean_client_acc':<22}"
          f"{'worst_acc':<22}{'std_acc':<20}")
    for key in sorted(by_key):
        rs = by_key[key]
        seeds = sorted(r.seed for r in rs)
        gb = mean_ci([r.metrics["global_bal_acc"] for r in rs])
        wa = mean_ci([r.metrics["worst_acc"] for r in rs])
        sa = mean_ci([r.metrics["std_acc"] for r in rs])
        mc = mean_ci(client_accs[key]) if key in client_accs else None
        abl, alpha, K = key
        mc_str = fmt(*mc) if mc else "n/a (no ckpt)"
        print(f"{abl:<16}{alpha:>6}{K:>4}{str(seeds):>7}   "
              f"{fmt(*gb):<22}{mc_str:<22}"
              f"{fmt(*wa):<22}{fmt(*sa):<20}")

    # --- COMPARABLE ladder: mean per-client accuracy ------------------- #
    print("\n=== Component ladder: MEAN PER-CLIENT accuracy (comparable) ===")
    print("(every arm evaluated with a trained classifier — use this for "
          "the decomposition ablation)")
    lad2: Dict[Tuple[float, int], Dict[str, float]] = defaultdict(dict)
    for key, vals in client_accs.items():
        abl, alpha, K = key
        lad2[(alpha, K)][abl] = sum(vals) / len(vals)
    for (alpha, K), d in sorted(lad2.items()):
        tf, dn, fu = d.get("true_fedavg"), d.get("decomp_nodrift"), d.get("full")
        parts = [f"{n}={v:.4f}" for n, v in
                 (("true_fedavg", tf), ("decomp_nodrift", dn), ("full", fu))
                 if v is not None]
        line = f"  alpha={alpha}, K={K}: " + "  ".join(parts)
        if tf is not None and dn is not None:
            line += f"  | Δdecomp={dn - tf:+.4f}"
        if dn is not None and fu is not None:
            line += f"  | Δdrift={fu - dn:+.4f}"
        print(line)

    # --- component ladder on global bal-acc --------------------------- #
    print("\n=== Component ladder: global balanced accuracy (NOT comparable "
          "across arm families) ===")
    print("(valid only WITHIN the decomposed family: decomp_nodrift vs full. "
          "true_fedavg's classifier is trained; theirs is frozen at init.)")
    ladders: Dict[Tuple[float, int], Dict[str, float]] = defaultdict(dict)
    for key, rs in by_key.items():
        abl, alpha, K = key
        ladders[(alpha, K)][abl] = mean_ci(
            [r.metrics["global_bal_acc"] for r in rs])[0]
    for (alpha, K), d in sorted(ladders.items()):
        tf, dn, fu = d.get("true_fedavg"), d.get("decomp_nodrift"), d.get("full")
        line = f"  alpha={alpha}, K={K}: "
        parts = []
        if tf is not None:
            parts.append(f"true_fedavg={tf:.4f}")
        if dn is not None:
            parts.append(f"decomp_nodrift={dn:.4f}")
        if fu is not None:
            parts.append(f"full={fu:.4f}")
        line += "  ".join(parts)
        if tf is not None and dn is not None:
            line += f"  | Δdecomp={dn - tf:+.4f}"
        if dn is not None and fu is not None:
            line += f"  | Δdrift={fu - dn:+.4f}"
        print(line)

    # --- paired deltas between two personalizing arms ----------------- #
    print("\n=== Paired deltas: full vs decomp_nodrift (fairness) ===")
    print("(per-seed differences; both arms keep the personalized head, so "
          "worst/std are meaningful. CI over shared seeds.)")
    for alpha, K in sorted({(a, k) for (_, a, k) in by_key}):
        full = {r.seed: r for r in by_key.get(("full", alpha, K), [])}
        dnod = {r.seed: r for r in by_key.get(("decomp_nodrift", alpha, K), [])}
        shared = sorted(set(full) & set(dnod))
        if not shared:
            continue
        d_worst = [full[s].metrics["worst_acc"] - dnod[s].metrics["worst_acc"]
                   for s in shared]
        d_std = [full[s].metrics["std_acc"] - dnod[s].metrics["std_acc"]
                 for s in shared]
        print(f"  alpha={alpha}, K={K} (seeds {shared}):")
        print(f"     Δ worst_acc (full − decomp) = {fmt(*mean_ci(d_worst))}")
        print(f"     Δ std_acc   (full − decomp) = {fmt(*mean_ci(d_std))}  "
              f"(negative = tighter/ fairer)")

    print()


if __name__ == "__main__":
    main(sys.argv[1:])
