"""
generate_figures_a01.py
=======================
Figures for the severe-heterogeneity (alpha=0.1) runs, in the visual style of
generate_figures.py.

Unlike generate_figures.py, which carries data transcribed from logs by hand,
this script PARSES the run logs directly, so the figures cannot drift from the
recorded results.

Outputs (FedVITpaper/images/):
    severe_convergence.png        one row, three arms: bal-acc / worst / std
    fedavg_severe_terminal.png    condensed terminal transcript for FedAvg
    comparison_balanced_acc_all.png   balanced accuracy across all three alphas

The alpha=0.1 evidence is deliberately compact: a single convergence row plus
one terminal panel for the FedAvg arm, which is the configuration whose
per-class output carries the melanoma finding. Full per-round transcripts for
all three arms are in the accompanying repository.

Usage:
    python generate_figures_a01.py [--logs DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CENTRALIZED_UPPER_BOUND = 0.8867

COLORS = {
    "bal_acc": "#1f77b4", "top1_acc": "#ff7f0e", "f1": "#2ca02c",
    "auc": "#9467bd", "worst": "#d62728", "std_band": "#1f77b4",
    "cent_ub": "#d62728",
}
FONT = dict(fontsize=10)
TITLE_FONT = dict(fontsize=11, fontweight="bold")

TERM_BG = "#1c1c1c"
TERM_FG = "#d0d0d0"
TERM_ACCENT = "#5fd7a7"

FIG_W = 13.0
CHAR_W_RATIO = 0.601

ROUND_RE = re.compile(
    r"Round\s+(\d+)/(\d+)\s*\|\s*"
    r"Global Acc=([\d.]+)\s+BalAcc=([\d.]+)\s+F1=([\d.]+)\s+AUC=([\w.]+)\s*\|\s*"
    r"Worst Acc=([\d.]+)\s+BalAcc=([\d.]+)\s*\|\s*"
    r"Std Acc=([\d.]+)\s+BalAcc=([\d.]+)"
)

ARMS = [
    ("true_fedavg_a0.1",    "FedAvg (no decomposition)"),
    ("decomp_nodrift_a0.1", "Decomposition only"),
    ("full_a0.1",           "FedSA-Drift"),
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


def _apply_style(ax, xlabel, ylabel, title=None, legend=True, loc="lower right"):
    ax.set_xlabel(xlabel, **FONT)
    ax.set_ylabel(ylabel, **FONT)
    if title:
        ax.set_title(title, **TITLE_FONT)
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    if legend:
        ax.legend(fontsize=8, loc=loc)


def find_log(logs_dir: str, key: str) -> str | None:
    m = glob.glob(os.path.join(logs_dir, f"federated_{key}_*.log"))
    return sorted(m)[-1] if m else None


# ── Figure: one compact row, three arms ─────────────────────────────────────
def plot_severe_row(datas, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 3.6))
    fig.suptitle("Severe Non-IID Setting ($\\alpha$=0.1, $K$=5): "
                 "Convergence by Aggregation Scheme",
                 fontsize=13, fontweight="bold")

    for ax, (label, d) in zip(axes, datas):
        rds = d["rounds"]
        acc = np.array(d["global_acc"]); std = np.array(d["std_acc"])
        ax.plot(rds, d["bal_acc"], color=COLORS["bal_acc"], marker="o",
                markersize=3, linewidth=1.6, label="Balanced Accuracy")
        ax.plot(rds, d["worst_acc"], color=COLORS["worst"], marker="v",
                markersize=3.5, linewidth=1.6, label="Worst Client Acc")
        ax.fill_between(rds, acc - std, acc + std, color=COLORS["std_band"],
                        alpha=0.18, label="Global Acc $\\pm$ 1 Std")
        ax.axhline(CENTRALIZED_UPPER_BOUND, color=COLORS["cent_ub"],
                   linestyle=":", linewidth=1.3,
                   label=f"Centralized UB ({CENTRALIZED_UPPER_BOUND:.2f})")
        ax.set_xlim(1, max(rds)); ax.set_ylim(0.0, 1.0)
        _apply_style(ax, "Communication Round", "Metric Value", title=label,
                     loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Figure: balanced-accuracy comparison across all three alphas ────────────
def plot_comparison_all(sev, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, 4.0))
    fig.suptitle("Balanced Accuracy Comparison — All Heterogeneity Levels",
                 fontsize=13, fontweight="bold")

    import generate_figures as gf  # reuse the transcribed alpha=0.5/1.0 series
    rds = list(range(1, 26))

    axes[0].plot(rds, gf.FED1["bal_acc"], color="#1f77b4", marker="o",
                 markersize=3, linewidth=1.6, label="FedSA-Drift")
    axes[0].plot(rds, gf.FED2["bal_acc"], color="#ff7f0e", marker="s",
                 markersize=3, linewidth=1.6, linestyle="--", label="FedAvg Baseline")
    axes[0].set_ylim(0.30, 0.95)
    _apply_style(axes[0], "Communication Round", "Balanced Accuracy",
                 title="Near-IID ($\\alpha$=1.0, $K$=3)")

    axes[1].plot(rds, gf.FED3["bal_acc"], color="#2ca02c", marker="o",
                 markersize=3, linewidth=1.6, label="FedSA-Drift")
    axes[1].plot(rds, gf.FED4["bal_acc"], color="#d62728", marker="s",
                 markersize=3, linewidth=1.6, linestyle="--", label="FedAvg Baseline")
    axes[1].set_ylim(0.30, 0.95)
    _apply_style(axes[1], "Communication Round", "Balanced Accuracy",
                 title="Moderate ($\\alpha$=0.5, $K$=5)")

    styles = {"FedSA-Drift": ("#1f77b4", "o", "-"),
              "Decomposition only": ("#2ca02c", "^", "-."),
              "FedAvg (no decomposition)": ("#ff7f0e", "s", "--")}
    for label, d in sev:
        c, mk, ls = styles[label]
        axes[2].plot(d["rounds"], d["bal_acc"], color=c, marker=mk,
                     markersize=3, linewidth=1.6, linestyle=ls, label=label)
    axes[2].set_ylim(0.30, 0.95)
    _apply_style(axes[2], "Communication Round", "Balanced Accuracy",
                 title="Severe ($\\alpha$=0.1, $K$=5)")

    for ax in axes:
        ax.set_xlim(1, 25)
        ax.axhline(CENTRALIZED_UPPER_BOUND, color=COLORS["cent_ub"],
                   linestyle=":", linewidth=1.4,
                   label=f"Centralized UB ({CENTRALIZED_UPPER_BOUND:.4f})")
        ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ── Figure: condensed terminal transcript ───────────────────────────────────
def render_terminal(lines: List[str], out_path: str):
    n = len(lines)
    longest = max((len(l.rstrip()) for l in lines), default=80)
    fontsize = (0.98 * FIG_W * 72.0) / (CHAR_W_RATIO * longest)
    fig_h = max(1.4, 1.34 * fontsize / 72.0 * n + 0.25)

    fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
    fig.patch.set_facecolor(TERM_BG); ax.set_facecolor(TERM_BG); ax.axis("off")
    for i, ln in enumerate(lines):
        colour = TERM_ACCENT if ("**" in ln or "complete" in ln.lower()) else TERM_FG
        ax.text(0.006, 1.0 - (i + 0.85) / n, ln.rstrip(), family="monospace",
                fontsize=fontsize, color=colour, transform=ax.transAxes,
                va="top", ha="left")
    plt.subplots_adjust(left=0.004, right=0.999, top=0.995, bottom=0.005)
    plt.savefig(out_path, dpi=200, facecolor=TERM_BG)
    plt.close()
    print(f"  Saved: {out_path}  ({n} lines, {fontsize:.1f}pt, {fig_h:.1f}in)")


def condensed_lines(path: str, every: int = 7) -> List[str]:
    """Config block, then every Nth round with its per-class line, then footer.

    Showing all 25 rounds triples the figure height for no extra evidence: the
    per-class pattern is identical at every round.
    """
    keep, cur_round = [], None
    head = re.compile(r"(Dirichlet|Drift weighting|Ablation arm|Group [ABC] |"
                      r"Client \d+:|Total params|Starting federated)")
    for line in open(path, errors="ignore"):
        s = line.rstrip("\n")
        if head.search(s):
            keep.append(s); continue
        m = re.search(r"Round\s+(\d+)/", s)
        if m:
            r = int(m.group(1))
            cur_round = r if (r == 1 or r % every == 0 or r == 25) else None
            if cur_round:
                keep.append(s)
            continue
        if "[Per-class]" in s and cur_round:
            keep.append(s); continue
        if "Federated training complete" in s or "Best global balanced" in s:
            keep.append(s)
    return keep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=os.path.expanduser("~/Downloads/fedvit_runs/logs"))
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "FedVITpaper", "images"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    datas = []
    for key, label in ARMS:
        p = find_log(args.logs, key)
        if not p:
            print(f"[skip] no log for {key}"); continue
        d = parse_log(p)
        print(f"{key}: {len(d['rounds'])} rounds")
        datas.append((label, d))

    plot_severe_row(datas, os.path.join(args.out, "severe_convergence.png"))
    plot_comparison_all(datas, os.path.join(args.out,
                                            "comparison_balanced_acc_all.png"))

    build_arm_bundles(args.logs, args.out)



# ── Per-arm bundles, matching the style of the alpha=0.5/1.0 figures ────────
def plot_arm_curves(data, suptitle, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, 5))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    rds = data["rounds"]
    bal = np.array(data["bal_acc"]); acc = np.array(data["global_acc"])
    f1 = np.array(data["f1"]); auc = np.array(data["auc"])
    worst = np.array(data["worst_acc"]); std = np.array(data["std_acc"])

    ax1.plot(rds, bal, color=COLORS["bal_acc"], marker="o", markersize=4,
             linewidth=1.8, label="Balanced Accuracy")
    ax1.plot(rds, acc, color=COLORS["top1_acc"], marker="s", markersize=4,
             linewidth=1.8, label="Top-1 Accuracy")
    ax1.plot(rds, f1, color=COLORS["f1"], marker="^", markersize=4,
             linewidth=1.8, label="Macro F1-Score")
    ax1.axhline(CENTRALIZED_UPPER_BOUND, color=COLORS["cent_ub"], linestyle="--",
                linewidth=1.5, label=f"Centralized UB ({CENTRALIZED_UPPER_BOUND:.4f})")
    ax1.set_xlim(1, max(rds)); ax1.set_ylim(0.0, 1.0)
    _apply_style(ax1, "Communication Round", "Metric Value",
                 title="(Left) Bal. Acc, Top-1 Acc & F1-Score")

    ax2.fill_between(rds, acc - std, acc + std, color=COLORS["std_band"],
                     alpha=0.20, label="Global Acc $\\pm$ 1 Std")
    ax2.plot(rds, acc, color=COLORS["std_band"], linewidth=1.2)
    ax2.plot(rds, auc, color=COLORS["auc"], marker="D", markersize=4,
             linewidth=1.8, label="Macro ROC-AUC")
    ax2.plot(rds, worst, color=COLORS["worst"], marker="v", markersize=5,
             linewidth=1.8, label="Worst Client Acc")
    ax2.axhline(CENTRALIZED_UPPER_BOUND, color=COLORS["cent_ub"], linestyle="--",
                linewidth=1.5, label=f"Centralized UB ({CENTRALIZED_UPPER_BOUND:.4f})")
    ax2.set_xlim(1, max(rds)); ax2.set_ylim(0.0, 1.0)
    _apply_style(ax2, "Communication Round", "Metric Value",
                 title="(Right) ROC-AUC, Worst Client Acc & $\\pm$1 Std")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def full_lines(path):
    """Every line the alpha=0.5/1.0 terminal panels show."""
    pat = re.compile(r"(Client \d+:|Group [ABC] |Total params|Starting federated|"
                     r"Round\s+\d+/|\[Per-class\]|New best|"
                     r"Federated training complete|Best global balanced|"
                     r"Dirichlet|Drift weighting|Ablation arm|Rounds\s+:)")
    return [l.rstrip("\n") for l in open(path, errors="ignore") if pat.search(l)]


STEMS = {"full_a0.1": "sa_drift_severe",
         "decomp_nodrift_a0.1": "decomp_only_severe",
         "true_fedavg_a0.1": "fedavg_severe"}
TITLES = {"full_a0.1": "FedSA-Drift, Severe non-IID (Drift-Aware Aggregation ON)",
          "decomp_nodrift_a0.1": "Decomposition Only, Severe non-IID (Uniform Aggregation)",
          "true_fedavg_a0.1": "FedAvg Baseline, Severe non-IID (No Decomposition)"}


def build_arm_bundles(logs_dir, out_dir):
    for key, stem in STEMS.items():
        path = find_log(logs_dir, key)
        if not path:
            continue
        d = parse_log(path)
        plot_arm_curves(d, f"{TITLES[key]}  ($\\alpha$=0.1, $K$=5)",
                        os.path.join(out_dir, f"{stem}_curves.png"))
        lines = full_lines(path)
        half = (len(lines) + 1) // 2
        render_terminal(lines[:half], os.path.join(out_dir, f"{stem}_terminal_top.png"))
        render_terminal(lines[half:], os.path.join(out_dir, f"{stem}_terminal_bottom.png"))


if __name__ == "__main__":
    main()
