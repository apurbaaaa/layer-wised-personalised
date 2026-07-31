#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot: set up a fresh pod and run the missing true_fedavg @ alpha=0.1 arm,
# which completes the decomposition ablation (reviewer Comment 3).
#
#   bash run_tf_alpha01.sh
#
# Does everything: deps -> dataset -> pretrained weights -> 25-round training
# -> val+test evaluation -> compacted checkpoint ready to download.
# Safe to re-run: every step is idempotent and skips work already done.
#
# Expect ~10.5 h on one A40 (~$4). Run it inside tmux.
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO"
MODEL=swinv2_base_window12to24_192to384.ms_in22k_ft_in1k
RUN=true_fedavg_a0.1_K5_s42_drift1

log() { echo "[setup $(date -u +%H:%M:%S)] $*"; }

# --- 1. System + python deps ---------------------------------------------
# unzip is NOT in the runpod pytorch image and fetch_isic.sh needs it.
if ! command -v unzip >/dev/null; then
    log "installing unzip"
    apt-get update -qq >/dev/null 2>&1
    apt-get install -y -qq unzip >/dev/null 2>&1
fi
# Never name torch/torchvision here: pip would swap the image's CUDA build.
log "installing python deps"
pip install -q --no-cache-dir timm pandas scikit-learn tqdm matplotlib

python - <<'PY'
import torch
print(f"[setup] torch {torch.__version__} cuda={torch.version.cuda} "
      f"available={torch.cuda.is_available()} gpus={torch.cuda.device_count()}")
assert torch.cuda.is_available(), "no CUDA device visible - stop here"
PY
[ $? -ne 0 ] && { log "CUDA check FAILED"; exit 1; }

# --- 2. Dataset on local disk (faster than a network volume) --------------
if [ ! -f /root/data/.fetch_done ]; then
    log "fetching ISIC 2019 training split (~9 GB)"
    bash fetch_isic.sh /root/data || { log "dataset fetch FAILED"; exit 1; }
else
    log "dataset already present"
fi
ln -sfn /root/data data
log "images: $(ls data/ISIC_2019_Training_Input | wc -l) (expect 25331)"

# --- 3. Pre-warm pretrained weights --------------------------------------
log "caching pretrained backbone"
python -c "import timm; timm.create_model('$MODEL', pretrained=True)" >/dev/null 2>&1

# --- 4. Training ----------------------------------------------------------
CKPT="checkpoints/federated/$RUN/last_federated.pt"
if [ -s "$CKPT" ] && grep -q "Federated training complete" logs/federated_${RUN}_*.log 2>/dev/null; then
    log "training already complete, skipping"
else
    log "starting training (~10 h, 25 rounds)"
    python federated_train.py \
        --device cuda:0 --ablation true_fedavg --dirichlet_alpha 0.1 \
        --num_clients 5 --rounds 25 --local_epochs 1 \
        --batch_size 16 --lr 4e-4 --workers 8 --seed 42
fi
[ -s "$CKPT" ] || { log "no checkpoint produced - training FAILED"; exit 1; }

# --- 5. Evaluation on val + test -----------------------------------------
mkdir -p results
log "evaluating on val + test"
python evaluate_federated.py --checkpoint "$CKPT" --split both \
    --device cuda:0 --batch_size 64 --workers 6 \
    --json "results/${RUN}_eval.json" 2>&1 | tail -8

# --- 6. Compact checkpoint for download ----------------------------------
mkdir -p compact
python compact_checkpoint.py "$CKPT" "compact/${RUN}_last.pt"

# --- 7. Summary -----------------------------------------------------------
log "running analysis"
python analyze_runs.py logs/ checkpoints/federated > results/analyze_tf01.txt 2>&1
cat results/analyze_tf01.txt

touch results/.tf01_done
log "ALL DONE - download compact/ logs/ results/ then STOP THE POD"
