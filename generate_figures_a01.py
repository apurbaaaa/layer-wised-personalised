"""
generate_figures_a01.py
=======================
Figures for the severe-heterogeneity (alpha=0.1) runs, matching the visual
style of generate_figures.py exactly.

Unlike generate_figures.py, which carries data transcribed from logs by hand,
this script PARSES the run logs directly. There is no transcription step and
therefore no opportunity for the figures to drift from the recorded results.

For each of the three alpha=0.1 arms it writes:
    <name>_a01_curves.png     convergence panels (same layout as Figures 7-10)
    <name>_a01_terminal.png   terminal transcript rendered from the real log

Usage:
    python generate_figures_a01.py [--logs DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CENTRALIZED_UPPER_BOUND = 0.8867

COLORS = {
    "bal_acc":   "#1f77b4",
    "top1_acc":  "#ff7f0e",
    "f1":        "#2ca02c",
    "auc":       "#9467bd",
    "worst":     "#d62728",
    "std_band":  "#1f77b4",
    "cent_ub":   "#d62728",
}
FONT = dict(fontsize=10)
TITLE_FONT = dict(fontsize=11, fontweight="bold")

# Terminal panel styling, chosen to match the existing screenshots.
TERM_BG = "#1c1c1c"
TERM_FG = "#d0d0d0"
TERM_ACCENT = "#5fd7a7"

ROUND_RE = re.compile(
    r"Round\s+(\d+)/(\d+)\s*\|\s*"
    r"Global Acc=([\d.]+)\s+BalAcc=([\d.]+)\s+F1=([\d.]+)\s+AUC=([\w.]+)\s*\|\s*"
    r"Worst Acc=([\d.]+)\s+BalAcc=([\d.]+)\s*\|\s*"
    r"Std Acc=([\d.]+)\s+BalAcc=([\d.]+)"
)

ARMS = [
    ("full_a0.1",           "sa_drift_severe",
     "FedSA-Drift, Severe non-IID (Drift-Aware Aggregation ON)"),
    ("decomp_nodrift_a0.1", "decomp_only_severe",
     "Decomposition Only, Severe non-IID (Uniform Aggregation)"),
    ("true_fedavg_a0.1",    "fedavg_severe",
     "FedAvg Baseline, Severe non-IID (No Decomposition)"),
]


def parse_log(path: str) -> Dict[str, List[float]]:
    d = {k: [] for k in ("rounds", "global_acc", "bal_acc", "f1", "auc",
                         "worst_acc", "worst_bal", "std_acc", "std_bal")}
    for line in open(path, errors="ignore"):
        m = ROUND_RE.search(line)
        if not m:
            continue
        g = m.groups()
        d["rounds"].append(int(g[0]))
        d["global_acc"].append(float(g[2]))
        d["bal_acc"].append(float(g[3]))
        d["f1"].append(float(g[4]))
        d["auc"].append(float(g[5]) if g[5] not in ("nan", "NaN") else np.nan)
        d["worst_acc"].append(float(g[6]))
        d["worst_bal"].append(float(g[7]))
        d["std_acc"].append(float(g[8]))
        d["std_bal"].append(float(g[9]))
    return d


def _apply_style(ax, xlabel, ylabel, title=None, legend=True):
    ax.set_xlabel(xlabel, **FONT)
    ax.set_ylabel(ylabel, **FONT)
    if title:
        ax.set_title(title, **TITLE_FONT)
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    if legend:
        ax.legend(fontsize=9, loc="lower right")


def plot_curves(data, suptitle, worst_label, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    rds = data["rounds"]
    bal = np.array(data["bal_acc"]); acc = np.array(data["global_acc"])
    f1 = np.array(data["f1"]);       auc = np.array(data["auc"])
    worst = np.array(data["worst_acc"]); std = np.array(data["std_acc"])

    ax1.plot(rds, bal, color=COLORS["bal_acc"], marker="o", markersize=4,
             linewidth=1.8, label="Balanced Accuracy")
    ax1.plot(rds, acc, color=COLORS["top1_acc"], marker="s", markersize=4,
             linewidth=1.8, label="Top-1 Accuracy")
    ax1.plot(rds, f1, color=COLORS["f1"], marker="^", markersize=4,
             linewidth=1.8, label="Macro F1-Score")
    ax1.axhline(CENTRALIZED_UPPER_BOUND, color=COLORS["cent_ub"], linestyle="--",
                linewidth=1.5,
                label=f"Centralized UB ({CENTRALIZED_UPPER_BOUND:.4f})")
    ax1.set_xlim(1, max(rds)); ax1.set_ylim(0.0, 1.0)
    _apply_style(ax1, "Communication Round", "Metric Value",
                 title="(Left) Bal. Acc, Top-1 Acc & F1-Score")

    ax2.fill_between(rds, acc - std, acc + std, color=COLORS["std_band"],
                     alpha=0.20, label="Global Acc $\\pm$ 1 Std")
    ax2.plot(rds, acc, color=COLORS["std_band"], linewidth=1.2)
    ax2.plot(rds, auc, color=COLORS["auc"], marker="D", markersize=4,
             linewidth=1.8, label="Macro ROC-AUC")
    ax2.plot(rds, worst, color=COLORS["worst"], marker="v", markersize=5,
             linewidth=1.8, label=worst_label)
    ax2.axhline(CENTRALIZED_UPPER_BOUND, color=COLORS["cent_ub"], linestyle="--",
                linewidth=1.5,
                label=f"Centralized UB ({CENTRALIZED_UPPER_BOUND:.4f})")
    ax2.set_xlim(1, max(rds)); ax2.set_ylim(0.0, 1.0)
    _apply_style(ax2, "Communication Round", "Metric Value",
                 title="(Right) ROC-AUC, Worst Client Acc & $\\pm$1 Std")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def render_terminal(lines: List[str], out_path: str, title: str):
    """Render real log lines as a dark terminal panel."""
    n = len(lines)
    fig_h = max(2.0, 0.135 * n + 0.35)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    fig.patch.set_facecolor(TERM_BG)
    ax.set_facecolor(TERM_BG)
    ax.axis("off")

    for i, ln in enumerate(lines):
        colour = TERM_ACCENT if ("**" in ln or "complete" in ln.lower()) else TERM_FG
        ax.text(0.004, 1.0 - (i + 0.85) / n, ln.rstrip(),
                family="monospace", fontsize=5.6, color=colour,
                transform=ax.transAxes, va="top", ha="left")

    plt.tight_layout(pad=0.15)
    plt.savefig(out_path, dpi=260, facecolor=TERM_BG, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}  ({n} lines, {title})")


def log_lines_for_panel(path: str) -> List[str]:
    """Keep the lines the existing figures show: config, clients, groups,
    per-round metrics, per-class recall, best markers, and the footer."""
    keep = []
    pat = re.compile(
        r"(Client \d+:|Group [ABC] |Total params|Starting federated|"
        r"Round\s+\d+/|\[Per-class\]|New best|Federated training complete|"
        r"Best global balanced|Dirichlet|Drift weighting|Ablation arm|Rounds\s+:)")
    for line in open(path, errors="ignore"):
        if pat.search(line):
            keep.append(line.rstrip("\n"))
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=os.path.expanduser(
        "~/Downloads/fedvit_runs/logs"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "FedVITpaper", "images"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import glob
    for key, stem, label in ARMS:
        matches = glob.glob(os.path.join(args.logs, f"federated_{key}_*.log"))
        if not matches:
            print(f"[skip] no log for {key}")
            continue
        path = sorted(matches)[-1]
        data = parse_log(path)
        if not data["rounds"]:
            print(f"[skip] no round lines in {path}")
            continue
        print(f"{key}: {len(data['rounds'])} rounds from {os.path.basename(path)}")

        plot_curves(
            data,
            suptitle=f"{label}  ($\\alpha$=0.1, $K$=5)",
            worst_label="Worst Client Acc",
            out_path=os.path.join(args.out, f"{stem}_curves.png"),
        )

        lines = log_lines_for_panel(path)
        half = (len(lines) + 1) // 2
        render_terminal(lines[:half],
                        os.path.join(args.out, f"{stem}_terminal_top.png"),
                        f"{key} part 1")
        render_terminal(lines[half:],
                        os.path.join(args.out, f"{stem}_terminal_bottom.png"),
                        f"{key} part 2")


if __name__ == "__main__":
    main()
