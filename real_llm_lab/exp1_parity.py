"""exp1: 逐位校验 —— 你写的 GPT-2 和 HuggingFace 官方实现是不是同一个东西？

前 8 步你从零写出了一个 Transformer。现在把 OpenAI 训练好的 GPT-2 small
（124M 参数）权重灌进去，再看两件事：

  1. 参数量能不能对上（124,439,808）；
  2. 同样的输入，两边输出的 logits 差多少。

如果最大差在 1e-4 以内（logits 本身量级 ~60，相对误差 1e-6），说明 `gpt2_real.py` 里的结构就是 GPT-2 本体，
你学过的 step01–08 没有白学。对照组是 HuggingFace 的 `GPT2LMHeadModel`，
它是工业界的参考实现，我们只把它当黑盒用。

运行：
    .venv/bin/python real_llm_lab/exp1_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch import nn

from gpt2_real import DEFAULT_WEIGHTS, generate_cached, load_gpt2

PROMPTS = [
    "The cat sat on the mat.",
    "In 1969, humans first walked on",
    "def add(a, b):\n    return a +",
    "1, 2, 3, 4, 5, 6, 7, 8,",
]

# 一段真实英文，用来对比 loss / perplexity
PARAGRAPH = (
    "The Industrial Revolution began in Britain in the late eighteenth century. "
    "New machines made it possible to produce cloth and iron far faster than before. "
    "People moved from farms to cities to work in factories, and the size of cities "
    "grew quickly. Living conditions were often hard, but incomes and population "
    "rose steadily over the following decades."
)


def compare_tensor(name: str, mine: torch.Tensor, ref: torch.Tensor) -> float:
    """打印一个张量两边的最大差，并做断言。"""
    diff = (mine - ref).abs().max().item()
    print(f"    {name:<24} max|diff| = {diff:.2e}")
    assert diff < 1e-5, f"{name} 差异过大: {diff}"
    return diff


@torch.no_grad()
def main() -> None:
    device = torch.device("cpu")
    torch.manual_seed(0)

    # ---- 1. 加载两边模型 ----
    print("=" * 72)
    print("1. 加载模型")
    print("=" * 72)
    model, tok = load_gpt2(device="cpu")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  我们的实现 : {n_params:,} 个参数")
    print(f"  GPT-2 small 官方论文数字 : 124,439,808")

    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained(
        str(DEFAULT_WEIGHTS), local_files_only=True,
        attn_implementation="eager").to(device).eval()  # eager: 才能输出注意力矩阵
    n_hf = sum(p.numel() for p in hf.parameters())
    print(f"  HuggingFace : {n_hf:,} 个参数")
    assert n_params == n_hf == 124_439_808

    print("\n  逐张量抽查（名字对应关系见 gpt2_real.py 的 load_weights）：")
    pares = [
        ("wte.weight", model.wte.weight, hf.transformer.wte.weight),
        ("wpe.weight", model.wpe.weight, hf.transformer.wpe.weight),
        ("h.0.attn.c_attn.weight",
         model.h[0].attn.c_attn.weight, hf.transformer.h[0].attn.c_attn.weight.t()),
        ("h.0.mlp.c_fc.weight",
         model.h[0].mlp.c_fc.weight, hf.transformer.h[0].mlp.c_fc.weight.t()),
        ("h.11.ln_2.weight",
         model.h[11].ln_2.weight, hf.transformer.h[11].ln_2.weight),
    ]
    for name, mine, ref in pares:
        compare_tensor(name, mine, ref)
    print("  [OK] 权重逐个对上了（GPT-2 原版用 Conv1D，所以有几项要先转置）")

    # ---- 2. logits 对比 ----
    print()
    print("=" * 72)
    print("2. 前向 logits 对比（同样的 token id，两边各算一遍）")
    print("=" * 72)
    worst = 0.0
    for text in PROMPTS:
        ids = torch.tensor([tok.encode(text)])
        mine = model(ids)
        ref = hf(ids).logits
        diff = (mine - ref).abs().max().item()
        worst = max(worst, diff)
        print(f"  {text!r:<40} T={ids.size(1):<3} max|diff| = {diff:.2e}")
        assert mine.shape == ref.shape
        assert torch.allclose(mine, ref, atol=1e-4), "logits 对不上！"
    print(f"  [OK] {len(PROMPTS)} 条输入的 logits 全部一致（最大差 {worst:.2e}）")

    # ---- 3. loss / perplexity 对比 ----
    print()
    print("=" * 72)
    print("3. 在真实英文上算 loss / perplexity")
    print("=" * 72)
    ids = torch.tensor([tok.encode(PARAGRAPH)])
    print(f"  文本 token 数 = {ids.size(1)}")

    def loss_of(logits: torch.Tensor) -> float:
        return nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            ids[:, 1:].reshape(-1)).item()

    mine_loss = loss_of(model(ids))
    ref_loss = loss_of(hf(ids).logits)
    print(f"  我们的实现 : loss {mine_loss:.4f}  perplexity {torch.exp(torch.tensor(mine_loss)):.2f}")
    print(f"  HuggingFace: loss {ref_loss:.4f}  perplexity {torch.exp(torch.tensor(ref_loss)):.2f}")
    assert abs(mine_loss - ref_loss) < 1e-4
    print("  [OK] loss 一致（这就是 step08 里训练时用的 cross_entropy，只是模型换成了真实权重）")

    # ---- 4. 贪心生成对比 ----
    print()
    print("=" * 72)
    print("4. 贪心解码：两边生成的 token 序列是否完全一样")
    print("=" * 72)
    prompt = "1, 2, 3, 4, 5, 6, 7, 8,"
    ids = torch.tensor([tok.encode(prompt)])
    mine_ids = generate_cached(model, ids.clone(), max_new_tokens=12, greedy=True)
    ref_ids = hf.generate(ids.clone(), max_new_tokens=12, do_sample=False,
                          pad_token_id=50256)
    print(f"  prompt : {prompt!r}")
    print(f"  我们   : {tok.decode(mine_ids[0, ids.size(1):].tolist())!r}")
    print(f"  HF     : {tok.decode(ref_ids[0, ids.size(1):].tolist())!r}")
    assert torch.equal(mine_ids, ref_ids), "生成结果不一致！"
    print("  [OK] 12 个 token 逐个相同")

    # ---- 5. 注意力矩阵对比 ----
    print()
    print("=" * 72)
    print("5. 注意力矩阵对比（12 层 × 12 头）")
    print("=" * 72)
    text = "The quick brown fox jumps over the lazy dog."
    ids = torch.tensor([tok.encode(text)])
    mine_attn = model.attention_maps(ids)                       # (L,B,H,T,T)
    ref_attn = hf(ids, output_attentions=True).attentions        # tuple of (B,H,T,T)
    ref_attn = torch.stack(ref_attn)
    print(f"  我们的 attention_maps : {tuple(mine_attn.shape)}  <- (n_layer, B, n_head, T, T)")
    print(f"  HF 的 attentions      : {tuple(ref_attn.shape)}")
    d = (mine_attn - ref_attn).abs().max().item()
    print(f"  最大差 = {d:.2e}")
    assert d < 1e-5
    # 因果性自检：上三角必须全是 0（对应 step04 的掩码）
    upper = torch.triu(torch.ones_like(mine_attn[0, 0, 0]), diagonal=1)
    assert mine_attn[0, 0, 0][upper.bool()].abs().max() == 0
    print("  [OK] 注意力矩阵一致，且上三角（未来位置）严格为 0")

    # ---- 6. 分词器：真实 BPE vs 你写的字符分词器 ----
    print()
    print("=" * 72)
    print("6. 真实 BPE 分词器 vs step01 的字符分词器")
    print("=" * 72)
    for text in ["Hello, world!", "unbelievable", "我爱自然语言处理"]:
        ids = tok.encode(text)
        print(f"  {text!r:<28} 字符数 {len(text):<2} -> BPE token 数 {len(ids):<2} "
              f"-> {tok.convert_ids_to_tokens(ids)}")
    print("  [OK] 一个 BPE token 常常对应好几个字符（比如 'unbelievable' 被切成 4 块），")
    print("       所以 GPT-2 的 vocab_size=50257，而不是 step01 那几十个字符。")
    print("       注意中文：BPE 是字节级的，一个汉字 = 3 个 UTF-8 字节，")
    print("       所以 8 个汉字要 19 个 token —— 这正是 exp4 里中文 perplexity 偏高的原因之一。")

    print()
    print("=" * 72)
    print("结论")
    print("=" * 72)
    print("  你从 step01 一路写到 step08 的结构，加载官方权重后与 HuggingFace")
    print("  实现逐位一致。也就是说：**你已经写出了 GPT-2 本体**，")
    print("  剩下的实验（exp2–exp5）都是在真实权重上观察它的行为。")
    print()
    print("  顺便提醒：GPT-2 small 只有 1.24 亿参数，事实知识很不可靠")
    print("  （比如问它 \"the capital of France is\"，它更想接 \"the\" 而不是 \"Paris\"）。")
    print("  本实验验证的是「结构和权重对得上」，不是「它什么都懂」。")


if __name__ == "__main__":
    main()
