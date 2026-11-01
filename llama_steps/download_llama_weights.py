"""下载 SmolLM2-135M 的官方权重（step09 要用）。

为什么选它：
    - 它就是**标准 Llama 架构**（RMSNorm + SwiGLU + RoPE + GQA），
      只不过很小：30 层、576 维、9 个 query 头配 3 个 kv 头；
    - 135M 参数，bf16 权重约 257 MiB，CPU 上就能跑；
    - 词表 49152，能直接生成像样的英文句子，比玩具模型有说服力；
    - Apache-2.0 许可，可以随便用来学习。
    对新手来说，这是「不用租 GPU 也能摸到真实 Llama 权重」的最省事选择。

下载内容：config.json / model.safetensors / tokenizer.json / tokenizer_config.json /
generation_config.json，总计约 260 MB，存到 `llama_steps/weights/smollm2-135m/`。

用法（在仓库根目录）：
    .venv/bin/python llama_steps/download_llama_weights.py

已经下载过会自动跳过。下载失败或想手动下载，见文件末尾的说明。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ID = "HuggingFaceTB/SmolLM2-135M"
HERE = Path(__file__).resolve().parent
TARGET = HERE / "weights" / "smollm2-135m"
KEEP = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
]


def human(n: int) -> str:
    """把字节数变成人话（MiB / GiB）。"""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def main() -> int:
    weight_file = TARGET / "model.safetensors"
    if weight_file.exists() and weight_file.stat().st_size > 200 * 1024**2:
        size = sum(f.stat().st_size for f in TARGET.glob("*") if f.is_file())
        print(f"[OK] 已经下载过：{TARGET}")
        print(f"     合计 {human(size)}，可以跳过这一步")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[FAIL] 需要 huggingface_hub：")
        print("       uv pip install --python .venv/bin/python huggingface_hub")
        return 1

    print(f"从 HuggingFace 下载 {REPO_ID}（约 260 MB，可以断点续传）...")
    print(f"目标目录：{TARGET}")
    TARGET.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=REPO_ID, local_dir=str(TARGET), allow_patterns=KEEP)

    if not weight_file.exists():
        print("[FAIL] 下载完成但没找到 model.safetensors")
        return 1

    size = sum(f.stat().st_size for f in TARGET.glob("*") if f.is_file())
    print(f"\n[OK] 下载完成：{human(size)}")
    for f in sorted(TARGET.glob("*")):
        if f.is_file():
            print(f"     {f.name:<24} {human(f.stat().st_size):>10}")
    print("\n下一步：.venv/bin/python llama_steps/step09_real_llama.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
