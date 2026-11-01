"""下载 GPT-2 small 的真实权重（约 550 MB）到 real_llm_lab/weights/gpt2/。

只需要跑一次：

    .venv/bin/python real_llm_lab/download_weights.py

为什么要下载「权重」：模型结构是你写的代码（gpt2_real.py），
但里面每一个数字（wte、wpe、144 个注意力头……）都来自 OpenAI 训练好的文件。
这个文件就是 HuggingFace 上的 openai-community/gpt2 仓库。

下载速度取决于网络，550 MB 大约 5~20 分钟。支持断点续传：
下载中断了再跑一次，会从已经下好的地方接着下。
"""

from __future__ import annotations

import hashlib
import sys
import time
import urllib.request
from pathlib import Path

BASE_URL = "https://huggingface.co/openai-community/gpt2/resolve/main"
DEST = Path(__file__).resolve().parent / "weights" / "gpt2"

# 5 个文件：3 个小的是配置/分词器，1 个 550MB 的是模型权重
FILES = [
    "config.json",        # 模型尺寸：12 层、12 头、768 维、词表 50257
    "tokenizer.json",     # BPE 分词器（50257 个子词 + 合并规则）
    "vocab.json",         # 子词 -> id 的映射
    "merges.txt",         # BPE 合并规则
    "model.safetensors",  # 真正的权重：1.24 亿个浮点数
]

# 完整性检查用的官方大小（字节）和 sha256（取自 HuggingFace 仓库的 LFS 元数据）
EXPECTED_SIZE = {
    "model.safetensors": 548_105_171,
}
EXPECTED_SHA256 = {
    "model.safetensors": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
}


def human(n: float) -> str:
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def download(name: str) -> None:
    url = f"{BASE_URL}/{name}"
    dest = DEST / name
    dest.parent.mkdir(parents=True, exist_ok=True)

    have = dest.stat().st_size if dest.exists() else 0
    request = urllib.request.Request(url)
    if have:
        # 断点续传：告诉服务器「我只要第 have 字节之后的部分」
        request.add_header("Range", f"bytes={have}-")

    with urllib.request.urlopen(request) as response:
        # 206 = 服务器同意续传；200 = 从头给，只能覆盖重下
        resumed = response.status == 206
        total = int(response.headers.get("Content-Length", 0)) + (have if resumed else 0)
        mode = "ab" if resumed else "wb"
        done = have if resumed else 0
        started = time.time()
        next_report = 0.0

        with open(dest, mode) as f:
            while True:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                pct = done / total * 100 if total else 0
                if time.time() - next_report > 2:
                    next_report = time.time()
                    speed = done / max(time.time() - started, 1e-6)
                    sys.stdout.write(
                        f"\r  {name:<20} {pct:5.1f}%  "
                        f"{human(done)} / {human(total)}  {human(speed)}/s")
                    sys.stdout.flush()

    print(f"\r  {name:<20} 完成，{human(dest.stat().st_size)}" + " " * 30)

    expected = EXPECTED_SIZE.get(name)
    if expected is not None:
        actual = dest.stat().st_size
        assert actual == expected, (
            f"{name} 大小不对：{actual} != {expected}，删掉重下")
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        assert digest == EXPECTED_SHA256[name], (
            f"{name} 的 sha256 不对：{digest}，删掉重下")


def main() -> None:
    print(f"下载到 {DEST}\n")
    for name in FILES:
        dest = DEST / name
        expected = EXPECTED_SIZE.get(name)
        if dest.exists() and expected is not None and dest.stat().st_size == expected:
            print(f"  {name:<20} 已存在，跳过")
            continue
        download(name)

    # 最后一步：真的把权重文件解析一遍，确认没下坏
    print("\n校验权重文件……")
    from safetensors import safe_open

    with safe_open(str(DEST / "model.safetensors"), framework="torch") as f:
        keys = list(f.keys())
        total = 0
        for name in keys:
            n = 1
            for dim in f.get_slice(name).get_shape():
                n *= dim
            total += n
    print(f"  张量个数 = {len(keys)}")
    print(f"  参数总量 = {total:,}")
    print("\n[OK] 真实权重已就绪，可以运行 gpt2_real.py 和 exp1_parity.py 了")


if __name__ == "__main__":
    main()
