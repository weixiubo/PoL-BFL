#!/bin/bash
# PoL-BFL RQ1 参数扫描 - Balanced 候选配置 (MNIST)
# 目标：在 delta=5.0 固定的前提下，探索 verification_rate ∈ {0.6,0.7,0.8,0.9,1.0}
#       是否存在既保持高安全性又明显降低开销的组合。

set -e

GPU_ID=${1:-0}
DATASET="MNIST"
MODEL="SimpleCNN"
NUM_ROUNDS=5
OUTPUT_BASE="experiments/results/param_scan_balanced_mnist"

mkdir -p "$OUTPUT_BASE"

# 固定 delta=5.0，只扫不同的 verification_rate
DELTAS=(5.0)
VRS=(0.6 0.7 0.8 0.9 1.0)

# 代表性强攻击 + 一类 free-riding
ATTACKS=(
  byzantine_random_noise
  byzantine_model_replacement
  byzantine_gradient_inversion
  free_riding_no_training
)

TOTAL=$(( ${#DELTAS[@]} * ${#VRS[@]} * ${#ATTACKS[@]} ))
CURRENT=0

LOG_FILE="$OUTPUT_BASE/scan_balanced_$(date +%Y%m%d_%H%M%S).log"

echo "PoL-BFL Balanced 参数扫描 (MNIST)" | tee -a "$LOG_FILE"
echo "GPU: $GPU_ID" | tee -a "$LOG_FILE"
echo "总实验数: $TOTAL" | tee -a "$LOG_FILE"

for delta in "${DELTAS[@]}"; do
  for vr in "${VRS[@]}"; do
    for attack in "${ATTACKS[@]}"; do
      CURRENT=$((CURRENT + 1))

      echo "" | tee -a "$LOG_FILE"
      echo "[$CURRENT/$TOTAL] Delta=$delta VR=$vr Attack=$attack" | tee -a "$LOG_FILE"
      echo "  开始时间: $(date)" | tee -a "$LOG_FILE"

      export CUDA_VISIBLE_DEVICES=$GPU_ID
      export CUBLAS_WORKSPACE_CONFIG=:4096:8

      OUTPUT_DIR="$OUTPUT_BASE/delta${delta}_vr${vr}_${attack}"
      mkdir -p "$OUTPUT_DIR"

      START_TS=$(date +%s)

      python3 experiments/scripts/runners/run_rq1_security.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --num_rounds "$NUM_ROUNDS" \
        --attack_type "$attack" \
        --baselines PoL_FL \
        --pol_delta "$delta" \
        --verification_rate "$vr" \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee -a "$LOG_FILE"

      END_TS=$(date +%s)
      DURATION=$((END_TS - START_TS))
      echo "  结束时间: $(date) (耗时: ${DURATION}秒)" | tee -a "$LOG_FILE"

      if [ -f "$OUTPUT_DIR/rq1_results.json" ]; then
        TPR=$(python - << 'EOF'
import json, sys
p = sys.argv[1]
with open(p) as f:
    data = json.load(f)
if not data:
    print('N/A')
    sys.exit(0)
print(data[0].get('detection_metrics', {}).get('TPR', 'N/A'))
EOF
"$OUTPUT_DIR/rq1_results.json")
        FPR=$(python - << 'EOF'
import json, sys
p = sys.argv[1]
with open(p) as f:
    data = json.load(f)
if not data:
    print('N/A')
    sys.exit(0)
print(data[0].get('detection_metrics', {}).get('FPR', 'N/A'))
EOF
"$OUTPUT_DIR/rq1_results.json")
        ACC=$(python - << 'EOF'
import json, sys
p = sys.argv[1]
with open(p) as f:
    data = json.load(f)
if not data:
    print('N/A')
    sys.exit(0)
print(data[0].get('final_accuracy', 'N/A'))
EOF
"$OUTPUT_DIR/rq1_results.json")
        echo "  指标: TPR=$TPR, FPR=$FPR, Acc=$ACC" | tee -a "$LOG_FILE"
      else
        echo "  [WARNING] 未找到 $OUTPUT_DIR/rq1_results.json" | tee -a "$LOG_FILE"
      fi

    done
  done
done

echo "" | tee -a "$LOG_FILE"
echo "Balanced 参数扫描完成 (MNIST)。结果目录: $OUTPUT_BASE" | tee -a "$LOG_FILE"

