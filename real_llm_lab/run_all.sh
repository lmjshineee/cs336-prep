#!/usr/bin/env bash
# 一键跑完真实 LLM 实验的所有步骤，并把每个脚本的输出保存到 real_llm_lab/runs/。
#
# 用法（在仓库根目录）：
#     bash real_llm_lab/run_all.sh
#
# 第一次运行会先下载权重（约 550 MB），之后会跳过。
set -euo pipefail

cd "$(dirname "$0")/.."

PY=".venv/bin/python"
export MPLCONFIGDIR="$PWD/real_llm_lab/.mplcache"
mkdir -p real_llm_lab/runs real_llm_lab/figs

# 0. 权重（没有就下载）
if [ ! -f real_llm_lab/weights/gpt2/model.safetensors ]; then
    echo ">> 下载 GPT-2 真实权重（约 550 MB，只需一次）"
    $PY real_llm_lab/download_weights.py
fi

run_one() {
    local script="$1"      # 例如 exp1_parity
    local log="real_llm_lab/runs/${script%%_*}.txt"
    echo
    echo "================================================================"
    echo ">> real_llm_lab/${script}.py"
    echo "================================================================"
    $PY "real_llm_lab/${script}.py" 2>&1 | tee "$log"
}

# 按顺序跑：每个实验都建立在前一个的基础上
run_one gpt2_real
run_one exp1_parity
run_one exp2_heads
run_one exp3_kv_cache
run_one exp4_mask_ppl
run_one exp5_sampling

echo
echo "全部完成。输出在 real_llm_lab/runs/，图片在 real_llm_lab/figs/"
