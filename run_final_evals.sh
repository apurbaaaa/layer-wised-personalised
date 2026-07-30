#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Wait for the reviewer matrix to finish, then evaluate every arm on BOTH the
# validation and test splits using the correct per-client classifiers.
#
# Run detached:
#   setsid nohup bash run_final_evals.sh >/root/evals.log 2>&1 </dev/null &
#
# Produces results/<run_name>_eval.json per arm and results/analyze.txt.
# ---------------------------------------------------------------------------
set -uo pipefail

REPO=/workspace/layer-wised-personalised
cd "$REPO"

CKPT=checkpoints/federated
RESULTS=results
mkdir -p "$RESULTS"

RUNS=(true_fedavg_a0.5_K5_s42_drift1
      full_a0.1_K5_s42_drift1
      decomp_nodrift_a0.1_K5_s42_drift1)

echo "[evals] waiting for all 3 runs to complete ..."
while true; do
    n=$(grep -l "Federated training complete" logs/federated_*a0*.log 2>/dev/null | wc -l)
    [ "$n" -ge 3 ] && break
    sleep 120
done
echo "[evals] training complete at $(date -u +%H:%M:%S) UTC"

# Checkpoint writes are ~2.1 GB and not atomic; let the final write settle so we
# never read a half-written file (this bit us once during the run).
sleep 60

pids=()
gpu=0
for r in "${RUNS[@]}"; do
    ck="$CKPT/$r/last_federated.pt"
    if [ ! -s "$ck" ]; then
        echo "[evals] MISSING $ck — skipping $r"
        continue
    fi
    echo "[evals] $r -> cuda:$gpu"
    nohup python evaluate_federated.py \
        --checkpoint "$ck" \
        --split both \
        --device "cuda:$gpu" \
        --batch_size 64 \
        --workers 6 \
        --json "$RESULTS/${r}_eval.json" \
        > "$RESULTS/${r}_eval.log" 2>&1 &
    pids+=($!)
    gpu=$((gpu + 1))
done

echo "[evals] waiting for ${#pids[@]} evaluations ..."
fail=0
for p in "${pids[@]}"; do
    wait "$p" || fail=$((fail + 1))
done
echo "[evals] evaluations finished ($fail failed)"

python analyze_runs.py logs/ "$CKPT" > "$RESULTS/analyze.txt" 2>&1
echo "[evals] wrote $RESULTS/analyze.txt"

touch "$RESULTS/.evals_done"
echo "[evals] ALL DONE at $(date -u +%H:%M:%S) UTC"
