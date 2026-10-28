#!/bin/bash
# 12 split 并行跑 gen_vessel_proj_mr.py，6 GPU 各 2 进程
# 用法: bash run.sh          # 启动全部
#       tail -f log/0.log    # 实时看 split 0 的输出

SCRIPT_DIR=/data1/liuyang/stent/DTE/gen_data
cd "$SCRIPT_DIR"
mkdir -p log

PY=/data1/liuyang/conda/envs/common/bin/python

# 默认保持旧输出路径；marker 重生成时显式设置这两个变量，例如：
# DTE_MARKER_PATH=/path/to/marker_vol_gt DTE_OUTPUT_NAME=vessel_with_stent_marker bash run.sh
OUTPUT_NAME="${DTE_OUTPUT_NAME:-vessel_with_stent}"
MARKER_ARGS=()
if [[ -n "${DTE_MARKER_PATH:-}" ]]; then
    MARKER_ARGS+=(--marker_path "$DTE_MARKER_PATH")
    MARKER_ARGS+=(--marker_mu_ratio "${DTE_MARKER_MU_RATIO:-2.0}")
fi

echo "Starting 12 splits..."
for SPLIT in $(seq 0 11); do
    GPU=$((SPLIT % 6))
    LOG="$SCRIPT_DIR/log/${SPLIT}.log"
    CUDA_VISIBLE_DEVICES=$GPU $PY -u gen_vessel_proj_mr.py \
        --data_name=costa_new --split=$SPLIT --train=0 --test=1 --use_split=1 \
        --output_name="$OUTPUT_NAME" "${MARKER_ARGS[@]}" \
        > "$LOG" 2>&1 &
    echo "  split=$SPLIT GPU=$GPU PID=$!  log=log/${SPLIT}.log"
done

echo ""
echo "实时监控:"
echo "  tail -f log/0.log          # 看单个 split"
echo "  tail -f log/*.log          # 看全部"
echo "  watch -n 5 'wc -l log/*.log'  # 看进度"
echo "  ps aux | grep gen_vessel | grep -v grep | wc -l  # 存活数"
wait
echo "=== All 12 splits done ==="
