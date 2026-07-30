#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Reviewer-response experiment matrix (budget ~$12 of a $15 ceiling).
#
# Three independent runs, each on its own GPU of a SINGLE multi-GPU pod:
#   1. true_fedavg   @ α=0.5   → Comment 3 (decomposition, global accuracy)
#   2. full          @ α=0.1   → Comment 6 (severe non-IID)
#   3. decomp_nodrift@ α=0.1   → Comment 6 (pairs with run 2)
#
# Each run writes to its own checkpoints/federated/<run_name>/ and its own
# logs/federated_<run_name>_*.log — nothing is overwritten.
#
# Usage (from the repo root on the pod, inside tmux):
#     bash run_reviewer_matrix.sh              # parallel on cuda:0,1,2
#     MODE=seq bash run_reviewer_matrix.sh     # sequential on cuda:0
#
# Then:  python analyze_runs.py logs/
# ---------------------------------------------------------------------------
set -euo pipefail

MODE="${MODE:-par}"
ROUNDS="${ROUNDS:-25}"
# 3 concurrent runs share the pod's CPU and disk. 8 workers each over-
# subscribes a typical 3-GPU pod, so parallel mode uses fewer per run.
if [ "$MODE" = "par" ]; then
    WORKERS="${WORKERS:-4}"
else
    WORKERS="${WORKERS:-8}"
fi

COMMON="--num_clients 5 --local_epochs 1 --batch_size 16 --lr 4e-4 \
        --seed 42 --workers $WORKERS --rounds $ROUNDS"

mkdir -p logs

launch () {           # launch <gpu_id> <ablation> <alpha>
    local gpu="$1" abl="$2" alpha="$3"
    local tag="${abl}_a${alpha}"
    echo ">>> [$tag] starting on cuda:${gpu}"
    if [ "$MODE" = "par" ]; then
        nohup python federated_train.py $COMMON \
            --device "cuda:${gpu}" --ablation "$abl" --dirichlet_alpha "$alpha" \
            > "logs/nohup_${tag}.out" 2>&1 &
        echo "    pid $! → logs/nohup_${tag}.out"
    else
        python federated_train.py $COMMON \
            --device cuda:0 --ablation "$abl" --dirichlet_alpha "$alpha"
    fi
}

# Comment 3 — true FedAvg baseline. NOTE: no personalized head, so its
# per-client worst/std on the shared val set is degenerate (std≈0);
# use this run for GLOBAL balanced accuracy only.
launch 0 true_fedavg    0.5

# Comment 6 — drift on vs off at severe non-IID. Both keep the personalized
# head, so worst-client / variance are meaningful for this pair.
launch 1 full           0.1
launch 2 decomp_nodrift 0.1

if [ "$MODE" = "par" ]; then
    echo
    echo "All three launched in background. Monitor with:"
    echo "    tail -f logs/federated_*_a0.1_*.log"
    echo "    nvidia-smi"
    echo "Waiting for completion ..."
    wait
fi

echo
echo "All runs complete. Aggregate with:  python analyze_runs.py logs/"
