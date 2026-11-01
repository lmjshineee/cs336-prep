#!/usr/bin/env bash
# 一键跑完 llama_steps 的 step01–step09，并把每个脚本的输出保存到 llama_steps/runs/。
#
# 用法（在仓库根目录，或者任意目录都行）：
#     bash llama_steps/run_all.sh
#
# step09 需要真实的 SmolLM2-135M 权重；没有权重就打印 skip，前 8 步照常跑完。
set -euo pipefail

cd "$(dirname "$0")/.."

PY=".venv/bin/python"
export MPLCONFIGDIR="$PWD/.tmp/mpl"
mkdir -p .tmp/mpl llama_steps/runs llama_steps/figs

run_one() {
    local script="$1"      # 例如 step07_train
    local log="llama_steps/runs/${script%%_*}.txt"
    echo
    echo "================================================================"
    echo ">> llama_steps/${script}.py"
    echo "================================================================"
    $PY "llama_steps/${script}.py" 2>&1 | tee "$log"
}

# 按顺序跑：每一步都建立在前一步的基础上
run_one step01_rmsnorm
run_one step02_swiglu
run_one step03_rope
run_one step04_gqa
run_one step05_block
run_one step06_model
run_one step07_train
run_one step08_parity

# step09 依赖真实权重（约 270 MB）；没有就跳过，避免第一次运行就卡在下载上。
WEIGHTS="llama_steps/weights/smollm2-135m/model.safetensors"
if [ -f "$WEIGHTS" ]; then
    run_one step09_real_llama
else
    echo
    echo ">> skip step09_real_llama.py：没有找到 $WEIGHTS"
    echo "   需要时先运行：.venv/bin/python llama_steps/download_llama_weights.py"
fi

echo
echo "全部完成。输出在 llama_steps/runs/，图片在 llama_steps/figs/"
