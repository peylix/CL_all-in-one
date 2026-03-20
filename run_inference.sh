#!/bin/bash
# Batch inference script for continual learning evaluation.
#
# For a given test dataset, run inference using multiple checkpoints
# to evaluate performance retention (forgetting).
#
# Usage:
#   ./run_inference.sh <exp_name> <data_path> <device> <test_task> <ckpt_task1> [ckpt_task2] ...
#
# Examples:
#   # Evaluate haze test set using haze and snow checkpoints
#   ./run_inference.sh haze_rain_snow /root/autodl-tmp cuda:0 haze haze snow
#
#   # Evaluate rain test set using rain and snow checkpoints
#   ./run_inference.sh haze_rain_snow /root/autodl-tmp cuda:0 rain rain snow
#
#   # Evaluate snow test set using only snow checkpoint
#   ./run_inference.sh haze_rain_snow /root/autodl-tmp cuda:0 snow snow
#
# Output structure:
#   results/<exp_name>/<test_task>/after_<ckpt_task>/pred/
#   results/<exp_name>/<test_task>/after_<ckpt_task>/gt/

set -e

if [ $# -lt 5 ]; then
    echo "Usage: $0 <exp_name> <data_path> <device> <test_task> <ckpt_task1> [ckpt_task2] ..."
    echo ""
    echo "Example: $0 haze_rain_snow /root/autodl-tmp cuda:0 haze haze snow"
    exit 1
fi

EXP_NAME=$1
DATA_PATH=$2
DEVICE=$3
TEST_TASK=$4
shift 4
CKPT_TASKS=("$@")

# Map test task to input/gt directories and gt_name_fn
case $TEST_TASK in
    haze)
        INPUT_DIR="${DATA_PATH}/CVPR19RainTrain/test/data"
        GT_DIR="${DATA_PATH}/CVPR19RainTrain/test/gt"
        GT_NAME_FN=""
        ;;
    rain)
        INPUT_DIR="${DATA_PATH}/raindrop_data/test_a/data"
        GT_DIR="${DATA_PATH}/raindrop_data/test_a/gt"
        GT_NAME_FN="--gt_name_fn raindrop"
        ;;
    snow)
        INPUT_DIR="${DATA_PATH}/Snow100K-testing/jdway/GameSSD/overlapping/test/Snow100K-M/synthetic"
        GT_DIR="${DATA_PATH}/Snow100K-testing/jdway/GameSSD/overlapping/test/Snow100K-M/gt"
        GT_NAME_FN=""
        ;;
    *)
        echo "Unknown test task: $TEST_TASK (supported: haze, rain, snow)"
        exit 1
        ;;
esac

for CKPT_TASK in "${CKPT_TASKS[@]}"; do
    CHECKPOINT="./checkpoints/${EXP_NAME}/${CKPT_TASK}/ffa_best.pk"
    OUTPUT_DIR="./results/${EXP_NAME}/${TEST_TASK}/after_${CKPT_TASK}"

    if [ ! -f "$CHECKPOINT" ]; then
        echo "[SKIP] Checkpoint not found: $CHECKPOINT"
        continue
    fi

    echo "============================================"
    echo "Test: ${TEST_TASK} | Checkpoint: ${CKPT_TASK}"
    echo "============================================"

    uv run python inference.py \
        --checkpoint "$CHECKPOINT" \
        --input_dir "$INPUT_DIR" \
        --gt_dir "$GT_DIR" \
        $GT_NAME_FN \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE"

    echo ""
done

echo "All done!"
