#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Three-way per-class comparison at alpha=0.1, every arm evaluated with a
# TRAINED classifier (each client's own Group C head; the shared head for
# true_fedavg). This is the only valid way to compare arms, because the global
# model of a decomposed arm carries an untrained classifier.
#
# Answers: does removing the decomposition cause the model to abandon whole
# classes (notably MEL) under severe heterogeneity?
#
#   bash eval_three_way.sh
#
# Expects the two prior compacted checkpoints in prior_compact/ and the new
# true_fedavg run in checkpoints/federated/.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"
mkdir -p results

TF=checkpoints/federated/true_fedavg_a0.1_K5_s42_drift1/last_federated.pt

run_one () {   # run_one <label> <checkpoint>
    local label="$1" ck="$2"
    if [ ! -s "$ck" ]; then echo "[3way] MISSING $ck - skipping $label"; return; fi
    echo "[3way] evaluating $label"
    python evaluate_federated.py --checkpoint "$ck" --split both \
        --device cuda:0 --batch_size 64 --workers 6 \
        --json "results/3way_${label}.json" > "results/3way_${label}.log" 2>&1
    echo "[3way] $label done"
}

# true_fedavg is evaluated by the main pipeline too, but re-run here so all
# three JSONs come from the same code version (with per-class recall).
run_one true_fedavg    "$TF"
run_one full           prior_compact/full_a0.1_K5_s42_drift1_last.pt
run_one decomp_nodrift prior_compact/decomp_nodrift_a0.1_K5_s42_drift1_last.pt

echo
echo "=============== THREE-WAY COMPARISON (alpha=0.1) ==============="
python - <<'PY'
import json, os
CLS = ["MEL","NV","BCC","AK","BKL","DF","VASC","SCC"]
ARMS = [("true_fedavg","no decomposition"),
        ("decomp_nodrift","decomposition only"),
        ("full","FedSA-Drift")]
for split in ("val","test"):
    print(f"\n--- {split} ---")
    print(f"{'arm':22s}{'meanAcc':>9}{'meanBal':>9}{'worst':>8}{'std':>8}   " +
          "".join(f"{c:>7}" for c in CLS))
    for key, label in ARMS:
        p = f"results/3way_{key}.json"
        if not os.path.exists(p):
            print(f"{label:22s}  (missing)"); continue
        d = json.load(open(p))
        if split not in d["splits"]:
            print(f"{label:22s}  (no {split})"); continue
        s = d["splits"][split]
        pcr = s.get("mean_per_class_recall")
        if pcr is None:                      # shared-model arm
            pcr = s["per_client"][0].get("per_class_recall", {})
        print(f"{label:22s}{s['mean_acc']:9.4f}{s['mean_bal_acc']:9.4f}"
              f"{s['worst_acc']:8.4f}{s['std_acc']:8.4f}   " +
              "".join(f"{pcr.get(c, float('nan')):7.3f}" for c in CLS))
print("\nNote: MEL recall is the clinically decisive column.")
PY
touch results/.3way_done
echo "[3way] ALL DONE"
