#!/bin/bash
# Batch inference script for continual learning evaluation.
#
# For each test dataset, run inference using checkpoints from that task onward,
# to evaluate both current performance and forgetting.
#
# Usage:
#   ./run_inference.sh <exp_name> <data_path> <device> <task1> <task2> <task3> ...
#
# Examples:
#   # Task order: haze -> rain -> snow
#   #   haze test set evaluated with: haze, rain, snow checkpoints
#   #   rain test set evaluated with: rain, snow checkpoints
#   #   snow test set evaluated with: snow checkpoint
#   ./run_inference.sh haze_rain_snow /root/autodl-tmp cuda:0 haze rain snow
#
#   # Task order: rain -> haze -> snow
#   ./run_inference.sh rain_haze_snow /root/autodl-tmp cuda:0 rain haze snow
#
# Output structure:
#   results/<exp_name>/<test_task>/after_<ckpt_task>/pred/
#   results/<exp_name>/<test_task>/after_<ckpt_task>/gt/

set -e

if [ $# -lt 4 ]; then
    echo "Usage: $0 <exp_name> <data_path> <device> <task1> [task2] [task3] ..."
    echo ""
    echo "Example: $0 haze_rain_snow /root/autodl-tmp cuda:0 haze rain snow"
    exit 1
fi

EXP_NAME=$1
DATA_PATH=$2
DEVICE=$3
shift 3
TASK_ORDER=("$@")

# Map task name to input/gt directories and gt_name_fn
get_test_args() {
    local task=$1
    case $task in
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
            echo "Unknown task: $task (supported: haze, rain, snow)"
            exit 1
            ;;
    esac
}

NUM_TASKS=${#TASK_ORDER[@]}

for (( i=0; i<NUM_TASKS; i++ )); do
    TEST_TASK=${TASK_ORDER[$i]}
    get_test_args "$TEST_TASK"

    # Evaluate with checkpoints from this task onward
    for (( j=i; j<NUM_TASKS; j++ )); do
        CKPT_TASK=${TASK_ORDER[$j]}
        CHECKPOINT="./checkpoints/${EXP_NAME}/${CKPT_TASK}/ffa_best.pk"
        OUTPUT_DIR="./results/${EXP_NAME}/${TEST_TASK}/after_${CKPT_TASK}"

        if [ ! -f "$CHECKPOINT" ]; then
            echo "[SKIP] Checkpoint not found: $CHECKPOINT"
            continue
        fi

        echo "============================================"
        echo "Test: ${TEST_TASK} | Checkpoint: after_${CKPT_TASK}"
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
done

echo "All done!"
