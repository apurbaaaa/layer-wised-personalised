"""
generate_figures.py
====================
Generates all training-curve figures used in the FedSA-Drift IEEE Access paper.
Data is parsed from actual run logs; no model inference required.

Outputs (written to FedVITpaper/images/):
  centralized_training_curves.png
  sa_drift_near_iid_curves.png
  fedavg_near_iid_curves.png
  sa_drift_non_iid_curves.png
  fedavg_non_iid_curves.png
  comparison_balanced_acc.png
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── output directory ────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(SCRIPT_DIR, "FedVITpaper", "images")
os.makedirs(IMG_DIR, exist_ok=True)

CENTRALIZED_UPPER_BOUND = 0.8867

# ════════════════════════════════════════════════════════════════════════════
# 1.  Hard-coded data extracted from logs
# ════════════════════════════════════════════════════════════════════════════

# ── Centralized (train_20260224_020732.log) ──────────────────────────────────
CENT = dict(
    epochs=list(range(1, 26)),
    train_loss=[
        1.0384, 0.6422, 0.5743, 0.5250, 0.4984,
        0.4787, 0.4631, 0.4585, 0.4447, 0.4307,
        0.4267, 0.4212, 0.4193, 0.4137, 0.4081,
        0.4020, 0.3994, 0.3996, 0.3974, 0.3961,
        0.3935, 0.3943, 0.3964, 0.3954, 0.3893,
    ],
    val_acc=[
        0.3111, 0.4729, 0.6466, 0.5095, 0.4200,
        0.6968, 0.6045, 0.6811, 0.7421, 0.7100,
        0.8155, 0.7974, 0.7882, 0.7879, 0.7818,
        0.7605, 0.8416, 0.8326, 0.8300, 0.8366,
        0.8566, 0.8576, 0.8600, 0.8558, 0.8603,
    ],
    val_bal_acc=[
        0.6130, 0.6840, 0.7793, 0.7510, 0.7430,
        0.7877, 0.7940, 0.8255, 0.8384, 0.8418,
        0.8675, 0.8706, 0.8580, 0.8556, 0.8633,
        0.8718, 0.8848, 0.8794, 0.8742, 0.8798,
        0.8867, 0.8864, 0.8865, 0.8859, 0.8866,
    ],
)

# ── Fed-1 : FedSA-Drift, α=1.0, K=3 (federated_20260223_113634 / docx) ───
# Round 16 is missing from the log; linearly interpolated.
FED1 = dict(
    rounds=list(range(1, 26)),
    global_acc=[
        0.1411, 0.3087, 0.3137, 0.4237, 0.4450,
        0.4797, 0.4921, 0.5176, 0.5039, 0.5384,
        0.5497, 0.5773, 0.5982, 0.6127, 0.6349,
        (0.6349 + 0.6453) / 2,   # round 16 interpolated
        0.6453, 0.7121, 0.6826, 0.6916,
        0.6887, 0.7195, 0.7208, 0.7421, 0.7695,
    ],
    bal_acc=[
        0.4414, 0.5681, 0.5865, 0.6335, 0.6627,
        0.6823, 0.7015, 0.7164, 0.7098, 0.7242,
        0.7328, 0.7486, 0.7619, 0.7714, 0.7768,
        (0.7768 + 0.7724) / 2,   # round 16 interpolated
        0.7724, 0.7896, 0.7779, 0.7967,
        0.7909, 0.8080, 0.7946, 0.8205, 0.8209,
    ],
    f1=[
        0.1418, 0.2789, 0.3132, 0.3805, 0.4124,
        0.4353, 0.4518, 0.4829, 0.4715, 0.5067,
        0.5214, 0.5451, 0.5714, 0.5883, 0.6125,
        (0.6125 + 0.5903) / 2,   # round 16 interpolated
        0.5903, 0.6570, 0.6472, 0.6485,
        0.6385, 0.6705, 0.6530, 0.6948, 0.7115,
    ],
    auc=[
        0.8380, 0.8668, 0.8801, 0.8967, 0.8958,
        0.9073, 0.9128, 0.9186, 0.9154, 0.9219,
        0.9238, 0.9274, 0.9302, 0.9326, 0.9347,
        (0.9347 + 0.9246) / 2,   # round 16 interpolated
        0.9246, 0.9304, 0.9316, 0.9352,
        0.9289, 0.9388, 0.9354, 0.9395, 0.9398,
    ],
    worst_acc=[
        0.1976, 0.3400, 0.4095, 0.4461, 0.4847,
        0.4421, 0.5084, 0.5322, 0.4881, 0.5603,
        0.5729, 0.6011, 0.6224, 0.6408, 0.6623,
        (0.6623 + 0.6829) / 2,   # round 16 interpolated
        0.6829, 0.7211, 0.7032, 0.7326,
        0.7742, 0.7668, 0.7850, 0.7784, 0.7850,
    ],
    std_acc=[
        0.0100, 0.0423, 0.0228, 0.0350, 0.0412,
        0.0483, 0.0312, 0.0289, 0.0347, 0.0216,
        0.0242, 0.0197, 0.0175, 0.0148, 0.0129,
        (0.0129 + 0.0102) / 2,   # round 16 interpolated
        0.0102, 0.0208, 0.0107, 0.0129,
        0.0085, 0.0129, 0.0054, 0.0031, 0.0076,
    ],
)

# ── Fed-2 : FedAvg baseline, α=1.0, K=3 (federated_20260223_112650.log) ───
FED2 = dict(
    rounds=list(range(1, 26)),
    global_acc=[
        0.1553, 0.3239, 0.3416, 0.3955, 0.5058,
        0.5221, 0.5571, 0.6092, 0.5479, 0.5703,
        0.5639, 0.6174, 0.6437, 0.6216, 0.7261,
        0.6903, 0.6595, 0.7129, 0.6955, 0.7095,
        0.6987, 0.6974, 0.7571, 0.7479, 0.6576,
    ],
    bal_acc=[
        0.4595, 0.5827, 0.6089, 0.6464, 0.6979,
        0.7167, 0.7312, 0.7431, 0.7249, 0.7413,
        0.7443, 0.7610, 0.7743, 0.7791, 0.7930,
        0.7983, 0.7926, 0.8021, 0.7996, 0.8181,
        0.8111, 0.8153, 0.8219, 0.8314, 0.8227,
    ],
    f1=[
        0.1570, 0.2877, 0.3302, 0.3713, 0.4829,
        0.4843, 0.5221, 0.5434, 0.5241, 0.5345,
        0.5574, 0.5922, 0.6329, 0.6224, 0.6756,
        0.6673, 0.6568, 0.6565, 0.6724, 0.7006,
        0.7044, 0.6931, 0.7227, 0.7438, 0.7129,
    ],
    auc=[
        0.8521, 0.8789, 0.8853, 0.9023, 0.9049,
        0.9131, 0.9195, 0.9222, 0.9142, 0.9178,
        0.9211, 0.9257, 0.9297, 0.9272, 0.9336,
        0.9379, 0.9258, 0.9310, 0.9352, 0.9336,
        0.9337, 0.9315, 0.9311, 0.9367, 0.9301,
    ],
    worst_acc=[
        0.2211, 0.3187, 0.3803, 0.4087, 0.4942,
        0.5011, 0.5437, 0.5263, 0.4487, 0.6171,
        0.4132, 0.5721, 0.6374, 0.5997, 0.6824,
        0.6082, 0.6037, 0.6166, 0.7168, 0.6942,
        0.7342, 0.7100, 0.7484, 0.7629, 0.6479,
    ],
    std_acc=[
        0.0068, 0.0641, 0.0572, 0.0346, 0.0291,
        0.0275, 0.0239, 0.0750, 0.0795, 0.0189,
        0.0813, 0.0592, 0.0392, 0.0351, 0.0342,
        0.0701, 0.0443, 0.0567, 0.0184, 0.0151,
        0.0330, 0.0240, 0.0200, 0.0083, 0.0182,
    ],
)

# ── Fed-3 : FedSA-Drift, α=0.5, K=5 (federated_20260224_123351.log) ───────
FED3 = dict(
    rounds=list(range(1, 26)),
    global_acc=[
        0.0755, 0.1789, 0.2039, 0.2945, 0.3563,
        0.3824, 0.4105, 0.4126, 0.3955, 0.4195,
        0.4063, 0.4916, 0.4482, 0.5268, 0.5268,
        0.5189, 0.4900, 0.5321, 0.5803, 0.5671,
        0.5955, 0.5655, 0.5458, 0.5876, 0.5676,
    ],
    bal_acc=[
        0.3783, 0.4121, 0.4857, 0.5223, 0.5523,
        0.5704, 0.5767, 0.5853, 0.5873, 0.6227,
        0.6146, 0.6376, 0.6267, 0.6577, 0.6446,
        0.6601, 0.6657, 0.6863, 0.6988, 0.7008,
        0.6941, 0.7066, 0.7002, 0.7005, 0.6987,
    ],
    f1=[
        0.0826, 0.1653, 0.2175, 0.2812, 0.3205,
        0.3515, 0.3574, 0.3636, 0.3606, 0.3775,
        0.3877, 0.4165, 0.4057, 0.4295, 0.4318,
        0.4387, 0.4293, 0.4536, 0.4677, 0.4711,
        0.4731, 0.4732, 0.4643, 0.4791, 0.4688,
    ],
    auc=[
        0.7812, 0.8441, 0.8603, 0.8735, 0.8782,
        0.8900, 0.8832, 0.8883, 0.8883, 0.8991,
        0.9018, 0.9004, 0.8986, 0.9091, 0.9060,
        0.9081, 0.9056, 0.9083, 0.9121, 0.9140,
        0.9160, 0.9086, 0.9095, 0.9150, 0.9157,
    ],
    worst_acc=[
        0.0634, 0.2950, 0.3579, 0.4571, 0.4145,
        0.4221, 0.4266, 0.4079, 0.4047, 0.4782,
        0.3847, 0.5376, 0.4518, 0.3442, 0.5811,
        0.5332, 0.5055, 0.5276, 0.5813, 0.5950,
        0.6392, 0.5926, 0.5968, 0.6234, 0.5705,
    ],
    std_acc=[
        0.0467, 0.0337, 0.0210, 0.0238, 0.0501,
        0.0690, 0.0625, 0.0681, 0.0776, 0.0529,
        0.0959, 0.0430, 0.0769, 0.1152, 0.0458,
        0.0509, 0.0600, 0.0556, 0.0387, 0.0361,
        0.0343, 0.0297, 0.0333, 0.0342, 0.0511,
    ],
)

# ── Fed-4 : FedAvg baseline, α=0.5, K=5 (federated_20260223_113607.log) ───
FED4 = dict(
    rounds=list(range(1, 26)),
    global_acc=[
        0.0908, 0.2163, 0.2845, 0.3087, 0.3697,
        0.4129, 0.4129, 0.4808, 0.5237, 0.5755,
        0.5813, 0.5811, 0.6263, 0.6774, 0.6697,
        0.6613, 0.6239, 0.6555, 0.6795, 0.6624,
        0.7132, 0.7018, 0.6389, 0.6908, 0.6555,
    ],
    bal_acc=[
        0.3561, 0.4285, 0.5219, 0.5519, 0.5689,
        0.5991, 0.6017, 0.6525, 0.6573, 0.6779,
        0.6735, 0.7051, 0.7014, 0.7139, 0.7126,
        0.7057, 0.7259, 0.7200, 0.7206, 0.7465,
        0.7352, 0.7600, 0.7386, 0.7622, 0.7526,
    ],
    f1=[
        0.1025, 0.2234, 0.3005, 0.3320, 0.3613,
        0.3958, 0.3920, 0.4328, 0.4608, 0.4918,
        0.5004, 0.4965, 0.5218, 0.5434, 0.5386,
        0.5341, 0.5378, 0.5575, 0.5732, 0.5587,
        0.5780, 0.5984, 0.5569, 0.5832, 0.5649,
    ],
    auc=[
        0.8064, 0.8599, 0.8659, 0.8873, 0.8860,
        0.8854, 0.8858, 0.8919, 0.8963, 0.8939,
        0.8988, 0.8996, 0.8994, 0.9060, 0.9059,
        0.9053, 0.9073, 0.9091, 0.9094, 0.9191,
        0.9147, 0.9166, 0.9135, 0.9106, 0.9171,
    ],
    worst_acc=[
        0.0668, 0.3089, 0.4113, 0.4453, 0.4650,
        0.4947, 0.5042, 0.4203, 0.4163, 0.4705,
        0.4703, 0.5379, 0.5624, 0.5245, 0.5387,
        0.5787, 0.4871, 0.5305, 0.3739, 0.4947,
        0.6547, 0.5595, 0.5679, 0.6032, 0.4800,
    ],
    std_acc=[
        0.0704, 0.0690, 0.0188, 0.0403, 0.0328,
        0.0118, 0.0307, 0.0678, 0.0737, 0.0565,
        0.0952, 0.0512, 0.0622, 0.0788, 0.0660,
        0.0556, 0.0713, 0.0718, 0.1246, 0.0939,
        0.0472, 0.0640, 0.0498, 0.0558, 0.0833,
    ],
)

# ════════════════════════════════════════════════════════════════════════════
# 2.  Plotting helpers
# ════════════════════════════════════════════════════════════════════════════

COLORS = {
    "bal_acc":   "#1f77b4",   # blue
    "top1_acc":  "#ff7f0e",   # orange
    "f1":        "#2ca02c",   # green
    "auc":       "#9467bd",   # purple
    "worst":     "#d62728",   # red
    "std_band":  "#1f77b4",   # same blue with alpha
    "cent_ub":   "#d62728",   # red dashed
    "train_loss": "#8c564b",  # brown
}

FONT = dict(fontsize=10)
TITLE_FONT = dict(fontsize=11, fontweight="bold")


def _apply_style(ax, xlabel, ylabel, title=None, legend=True):
    ax.set_xlabel(xlabel, **FONT)
    ax.set_ylabel(ylabel, **FONT)
    if title:
        ax.set_title(title, **TITLE_FONT)
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    if legend:
        ax.legend(fontsize=9, loc="lower right")


def _add_centralized_ub(ax, n_points: int, x_vals=None):
    xs = x_vals if x_vals is not None else list(range(1, n_points + 1))
    ax.axhline(CENTRALIZED_UPPER_BOUND, color=COLORS["cent_ub"],
               linestyle="--", linewidth=1.5,
               label=f"Centralized UB ({CENTRALIZED_UPPER_BOUND:.4f})")


# ════════════════════════════════════════════════════════════════════════════
# 3.  Figure 1 — Centralized training curves
# ════════════════════════════════════════════════════════════════════════════

def plot_centralized():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Centralized Training Curves (25 Epochs)", fontsize=13, fontweight="bold")

    ep = CENT["epochs"]

    # Left — accuracy
    ax1.plot(ep, CENT["val_bal_acc"], color=COLORS["bal_acc"],  marker="o", markersize=4,
             linewidth=1.8, label="Balanced Accuracy (Val)")
    ax1.plot(ep, CENT["val_acc"],     color=COLORS["top1_acc"], marker="s", markersize=4,
             linewidth=1.8, label="Top-1 Accuracy (Val)")
    ax1.axhline(CENTRALIZED_UPPER_BOUND, color=COLORS["cent_ub"], linestyle="--",
                linewidth=1.5, label=f"Peak Bal. Acc ({CENTRALIZED_UPPER_BOUND:.4f}) @ Epoch 21")
    ax1.set_xlim(1, 25); ax1.set_ylim(0.25, 0.95)
    _apply_style(ax1, "Epoch", "Accuracy",
                 title="(Left) Balanced & Top-1 Accuracy vs Epoch")

    # Right — training loss
    ax2.plot(ep, CENT["train_loss"], color=COLORS["train_loss"], marker="^", markersize=4,
             linewidth=1.8, label="Training Loss")
    ax2.set_xlim(1, 25)
    _apply_style(ax2, "Epoch", "Loss",
                 title="(Right) Training Loss vs Epoch")

    plt.tight_layout()
    out = os.path.join(IMG_DIR, "centralized_training_curves.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════════
# 4.  Generic federated figure (left: BalAcc/Acc/F1;  right: AUC/Worst/Std)
# ════════════════════════════════════════════════════════════════════════════

def plot_federated(data: dict, title_left: str, title_right: str,
                   suptitle: str, worst_label: str, out_name: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    rds = data["rounds"]
    bal   = np.array(data["bal_acc"])
    acc   = np.array(data["global_acc"])
    f1    = np.array(data["f1"])
    auc   = np.array(data["auc"])
    worst = np.array(data["worst_acc"])
    std   = np.array(data["std_acc"])

    # ── Left panel ──────────────────────────────────────────────────────────
    ax1.plot(rds, bal,  color=COLORS["bal_acc"],  marker="o", markersize=4,
             linewidth=1.8, label="Balanced Accuracy")
    ax1.plot(rds, acc,  color=COLORS["top1_acc"], marker="s", markersize=4,
             linewidth=1.8, label="Top-1 Accuracy")
    ax1.plot(rds, f1,   color=COLORS["f1"],       marker="^", markersize=4,
             linewidth=1.8, label="Macro F1-Score")
    _add_centralized_ub(ax1, len(rds))
    ax1.set_xlim(1, 25); ax1.set_ylim(0.0, 1.0)
    _apply_style(ax1, "Communication Round", "Metric Value", title=title_left)

    # ── Right panel ─────────────────────────────────────────────────────────
    # Shaded band: global accuracy ± 1 std
    ax2.fill_between(rds, acc - std, acc + std,
                     color=COLORS["std_band"], alpha=0.20,
                     label="Global Acc ± 1 Std")
    ax2.plot(rds, acc,   color=COLORS["std_band"], linewidth=1.2,
             linestyle="-", marker=None)
    ax2.plot(rds, auc,   color=COLORS["auc"],   marker="D", markersize=4,
             linewidth=1.8, label="Macro ROC-AUC")
    ax2.plot(rds, worst, color=COLORS["worst"], marker="v", markersize=5,
             linewidth=1.8, label=worst_label)
    _add_centralized_ub(ax2, len(rds))
    ax2.set_xlim(1, 25); ax2.set_ylim(0.0, 1.0)
    _apply_style(ax2, "Communication Round", "Metric Value", title=title_right)

    plt.tight_layout()
    out = os.path.join(IMG_DIR, out_name)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════════
# 5.  Figure 6 — Comparison balanced accuracy (all runs)
# ════════════════════════════════════════════════════════════════════════════

def plot_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Balanced Accuracy Comparison — All Experiments", fontsize=13, fontweight="bold")

    rds = list(range(1, 26))

    # Left — Near-IID (α=1.0, K=3)
    ax1.plot(rds, FED1["bal_acc"], color="#1f77b4", marker="o", markersize=4,
             linewidth=1.8, label="FedSA-Drift (α=1.0, K=3)")
    ax1.plot(rds, FED2["bal_acc"], color="#ff7f0e", marker="s", markersize=4,
             linewidth=1.8, linestyle="--", label="FedAvg Baseline (α=1.0, K=3)")
    ax1.axhline(CENTRALIZED_UPPER_BOUND, color=COLORS["cent_ub"], linestyle=":",
                linewidth=1.5, label=f"Centralized UB ({CENTRALIZED_UPPER_BOUND:.4f})")
    ax1.set_xlim(1, 25); ax1.set_ylim(0.35, 0.95)
    _apply_style(ax1, "Communication Round", "Balanced Accuracy",
                 title="Near-IID Setting (α=1.0, K=3)")

    # Right — Non-IID (α=0.5, K=5)
    ax2.plot(rds, FED3["bal_acc"], color="#2ca02c", marker="o", markersize=4,
             linewidth=1.8, label="FedSA-Drift (α=0.5, K=5)")
    ax2.plot(rds, FED4["bal_acc"], color="#d62728", marker="s", markersize=4,
             linewidth=1.8, linestyle="--", label="FedAvg Baseline (α=0.5, K=5)")
    ax2.axhline(CENTRALIZED_UPPER_BOUND, color=COLORS["cent_ub"], linestyle=":",
                linewidth=1.5, label=f"Centralized UB ({CENTRALIZED_UPPER_BOUND:.4f})")
    ax2.set_xlim(1, 25); ax2.set_ylim(0.30, 0.95)
    _apply_style(ax2, "Communication Round", "Balanced Accuracy",
                 title="Highly Non-IID Setting (α=0.5, K=5)")

    plt.tight_layout()
    out = os.path.join(IMG_DIR, "comparison_balanced_acc.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ════════════════════════════════════════════════════════════════════════════
# 6.  Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating figures...")

    # Centralized
    plot_centralized()

    # Fed-1: FedSA-Drift Near-IID
    plot_federated(
        FED1,
        title_left="(Left) Bal. Acc, Top-1 Acc & F1-Score",
        title_right="(Right) ROC-AUC, Worst Client Acc & ±1 Std",
        suptitle="FedSA-Drift  (α=1.0, K=3, Drift-Aware Aggregation ON)",
        worst_label="Worst Client Acc (0.7850)",
        out_name="sa_drift_near_iid_curves.png",
    )

    # Fed-2: FedAvg Near-IID
    plot_federated(
        FED2,
        title_left="(Left) Bal. Acc, Top-1 Acc & F1-Score",
        title_right="(Right) ROC-AUC, Worst Client Acc & ±1 Std",
        suptitle="FedAvg Baseline  (α=1.0, K=3, Uniform Aggregation)",
        worst_label="Worst Client Acc (0.6479)",
        out_name="fedavg_near_iid_curves.png",
    )

    # Fed-3: FedSA-Drift Non-IID
    plot_federated(
        FED3,
        title_left="(Left) Bal. Acc, Top-1 Acc & F1-Score",
        title_right="(Right) ROC-AUC, Worst Client Acc & ±1 Std",
        suptitle="FedSA-Drift  (α=0.5, K=5, Drift-Aware Aggregation ON)",
        worst_label="Worst Client Acc (0.5705)",
        out_name="sa_drift_non_iid_curves.png",
    )

    # Fed-4: FedAvg Non-IID
    plot_federated(
        FED4,
        title_left="(Left) Bal. Acc, Top-1 Acc & F1-Score",
        title_right="(Right) ROC-AUC, Worst Client Acc & ±1 Std",
        suptitle="FedAvg Baseline  (α=0.5, K=5, Uniform Aggregation)",
        worst_label="Worst Client Acc (0.4800)",
        out_name="fedavg_non_iid_curves.png",
    )

    # Comparison
    plot_comparison()

    print("Done. All figures written to", IMG_DIR)
