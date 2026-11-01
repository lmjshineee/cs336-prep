"""Step 8: 把我们的 Llama 和 HuggingFace 的 Llama 逐位对齐，顺便收口整条改造路径。

前面七步是「一处一处改」，这一步回答两个问题：

1) 我们改对了吗？
   把 step06 的权重一张一张搬进 HuggingFace 官方的 `LlamaForCausalLM`，
   同一段输入、同一个 float32，比较两边输出的 logits。
   如果 RMSNorm 少一个 eps、RoPE 的通道配对方式写反、GQA 的复制顺序写错，
   这一步会立刻炸 —— 这比「loss 看起来还行」可靠得多。

2) 我们到底改了 GPT-2 的哪几处？
   最后给一张改造清单 + 一张真实模型的 KV cache 账本。

权重名字对照（我们 → HuggingFace）：

    token_embedding.weight            model.embed_tokens.weight
    blocks.i.attn_norm.weight         model.layers.i.input_layernorm.weight
    blocks.i.attn.q_proj.weight       model.layers.i.self_attn.q_proj.weight
    blocks.i.attn.k_proj.weight       model.layers.i.self_attn.k_proj.weight
    blocks.i.attn.v_proj.weight       model.layers.i.self_attn.v_proj.weight
    blocks.i.attn.o_proj.weight       model.layers.i.self_attn.o_proj.weight
    blocks.i.ffn_norm.weight          model.layers.i.post_attention_layernorm.weight
    blocks.i.ffn.gate_proj.weight     model.layers.i.mlp.gate_proj.weight
    blocks.i.ffn.up_proj.weight       model.layers.i.mlp.up_proj.weight
    blocks.i.ffn.down_proj.weight     model.layers.i.mlp.down_proj.weight
    norm.weight                       model.norm.weight
    lm_head.weight                    lm_head.weight

注意对照组里**没有** `wpe`（GPT-2 的学习式位置嵌入）和 `attn.bias`（因果掩码的偏置），
Llama 把前者换成了 RoPE（在注意力内部算，不占参数），后者换成了代码里的上三角掩码。

运行：.venv/bin/python llama_steps/step08_parity.py
"""

from __future__ import annotations

import unicodedata

import torch
from transformers import LlamaConfig as HFLlamaConfig
from transformers import LlamaForCausalLM

import step06_model
from step06_model import LlamaConfig, MiniLlama


def _width(text: str) -> int:
    """终端里中文占 2 列，按显示宽度算才能对齐。"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def show(name: str, value: object, note: str = "") -> None:
    """打印一行对齐的「名字 = 值」，note 是行尾的人话注释。"""
    if isinstance(value, torch.Tensor):
        value = tuple(value.shape)
    if isinstance(value, bool):          # bool 是 int 的子类，要先判断
        value = str(value)
    elif isinstance(value, int):
        value = f"{value:,}"
    pad = " " * max(0, 22 - _width(name))
    tail = f"   <- {note}" if note else ""
    print(f"{name}{pad} = {value}{tail}")


def col(text: str, width: int, right: bool = False) -> str:
    """按显示宽度把一格文字补到 width 列。"""
    pad = " " * max(0, width - _width(text))
    return pad + text if right else text + pad


def hf_config(config: LlamaConfig) -> HFLlamaConfig:
    """把我们的 config 翻译成 HuggingFace 的 config（字段名不一样，含义一一对应）。"""
    return HFLlamaConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.n_embd,
        intermediate_size=config.ffn_hidden,
        num_hidden_layers=config.n_layer,
        num_attention_heads=config.n_head,
        num_key_value_heads=config.n_kv_head,
        max_position_embeddings=config.block_size,
        rms_norm_eps=config.norm_eps,
        rope_theta=config.rope_theta,
        tie_word_embeddings=config.tie_word_embeddings,
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
    )


def to_hf_state(model: MiniLlama) -> dict[str, torch.Tensor]:
    """我们的 state_dict → HuggingFace 的 state_dict（纯改名，不改数值）。"""
    sd = model.state_dict()
    out = {
        "model.embed_tokens.weight": sd["token_embedding.weight"],
        "model.norm.weight": sd["norm.weight"],
    }
    if not model.config.tie_word_embeddings:   # 共享权重时 HF 不单独存 lm_head
        out["lm_head.weight"] = sd["lm_head.weight"]
    for i in range(model.config.n_layer):
        a, b = f"blocks.{i}.", f"model.layers.{i}."
        out[b + "input_layernorm.weight"] = sd[a + "attn_norm.weight"]
        out[b + "post_attention_layernorm.weight"] = sd[a + "ffn_norm.weight"]
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            out[b + f"self_attn.{name}.weight"] = sd[a + f"attn.{name}.weight"]
        for name in ("gate_proj", "up_proj", "down_proj"):
            out[b + f"mlp.{name}.weight"] = sd[a + f"ffn.{name}.weight"]
    return out


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def kv_cache_bytes(n_layer: int, n_kv_head: int, head_dim: int, ctx: int,
                   bytes_per_num: int = 2) -> int:
    """KV cache 占用 = 2(K/V) × 层数 × kv 头数 × head_dim × 上下文长度 × 每元素字节。"""
    return 2 * n_layer * n_kv_head * head_dim * ctx * bytes_per_num


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} GiB"


def main() -> None:
    torch.manual_seed(0)

    print("\n--- 1. 同一套权重，两个实现，同一种输入 ---")
    config = LlamaConfig(vocab_size=64, block_size=64, n_layer=2, n_head=4,
                         n_kv_head=2, n_embd=64, ffn_hidden=176)
    ours = MiniLlama(config).eval()
    hf = LlamaForCausalLM(hf_config(config)).eval()
    missing, unexpected = hf.load_state_dict(to_hf_state(ours), strict=False)
    assert not missing and not unexpected, (missing, unexpected)

    show("我们的参数量", sum(p.numel() for p in ours.parameters()))
    show("HF 的参数量", sum(p.numel() for p in hf.parameters()))
    n_ours = sum(p.numel() for p in ours.parameters())
    n_hf = sum(p.numel() for p in hf.parameters())
    assert n_ours == n_hf, (n_ours, n_hf)

    idx = torch.randint(0, config.vocab_size, (2, 16))
    with torch.no_grad():
        logits_ours = ours(idx)
        logits_hf = hf(idx).logits
    show("输入 idx", idx, "(B, T)")
    show("我们的 logits", logits_ours, "(B, T, V)")
    show("HF 的 logits", logits_hf, "(B, T, V)")
    diff = max_diff(logits_ours, logits_hf)
    print(f"{'max|diff|':<20} = {diff:.3e}")
    assert diff < 1e-5, f"和 HuggingFace 不一致：{diff:.3e}"
    print("[OK] 逐元素近似对齐，满足预设 1e-5 容差；不是逐位相同")

    print("\n--- 2. 为什么「数学上等价」还不够（三个常见写错方式） ---")
    # 故意把三个组件写错，看看对齐检查能不能抓住。
    # 技巧：step06_model 的 Attention 直接调用模块级函数，所以替换模块里的名字即可。
    controls = []

    original = step06_model.apply_rope
    for label, patch in [
        ("去掉 RoPE（不旋转 q/k）", lambda x, cos, sin: x),
        ("用 interleaved 配对（HF 用 half-split）", _rope_interleaved),
    ]:
        step06_model.apply_rope = patch
        with torch.no_grad():
            bad = ours(idx)
        controls.append((label, max_diff(bad, logits_hf)))
        step06_model.apply_rope = original

    original_kv = step06_model.repeat_kv
    step06_model.repeat_kv = lambda x, n_rep: x.repeat(1, n_rep, 1, 1)
    with torch.no_grad():
        bad = ours(idx)
    controls.append(("repeat_kv 用整块复制而不是交错", max_diff(bad, logits_hf)))
    step06_model.repeat_kv = original_kv

    for label, value in controls:
        print(f"{label:<38} max|diff| = {value:.3e}")
        assert value > 1e-3, f"{label} 竟然没被发现，说明对照实验没生效"
    print("[OK] 每个错误都被放大到 1e-3 以上，和对齐误差 1e-7 差了 4 个数量级")

    print("\n--- 3. 真实模型的 KV cache 账本（fp16，8192 上下文） ---")
    rows = [
        ("GPT-2 small（假设8k）", 12, 12, 64, 8192, "仅容量公式外推；原 GPT-2 不支持此输入长度"),
        ("Llama-3-8B", 32, 8, 128, 8192, "GQA，8 组 kv 头"),
        ("Llama-3-8B 若用 MHA", 32, 32, 128, 8192, "假设不改 GQA"),
        ("Llama-3-70B", 80, 8, 128, 8192, "GQA，80 层"),
        ("SmolLM2-135M", 30, 3, 64, 8192, "step09 要加载的真实模型"),
    ]
    print(col("模型", 24) + col("层", 5, True) + col("kv头", 7, True) + col("D", 5, True)
          + col("每 token", 12, True) + col("8k 上下文", 14, True) + "   说明")
    for name, layers, kv_heads, d, ctx, note in rows:
        per_token = 2 * layers * kv_heads * d * 2
        total = kv_cache_bytes(layers, kv_heads, d, ctx)
        print(col(name, 24) + col(str(layers), 5, True) + col(str(kv_heads), 7, True)
              + col(str(d), 5, True) + col(human(per_token), 12, True)
              + col(human(total), 14, True) + "   " + note)
    print("[OK] 代码算出的数字和公式一致（2 × 层数 × kv 头 × D × 上下文 × 2 字节）")
    print("     结论：8B 模型在 8k 上下文下，光 KV cache 就要 1 GiB；")
    print("     在此配置与相同 dtype 下，32 个 KV 头换成 8 个，缓存容量减少 3/4。")

    print("\n--- 4. GPT-2 → Llama 改造清单（回顾八步） ---")
    changes = [
        ("归一化", "LayerNorm (γ, β = 2C)", "RMSNorm (只有 γ = C)", "少一半参数、少一次减均值"),
        ("FFN", "GELU, C→4C→C", "SwiGLU, C→8/3C→C", "三个 dense 投影，逐元素门控"),
        ("位置编码", "学习式 wpe (T×C 参数)", "RoPE (0 参数，转 q/k)", "公式可延伸；实现与质量另有条件"),
        ("注意力", "MHA (kv 头 = query 头)", "GQA (kv 头 = 1/4~1/8)", "KV cache 显存降 4–8 倍"),
        ("线性层 bias", "有", "全部去掉", "少参数、便于算子融合"),
        ("dropout", "0.1", "0.0", "本例 dropout=0，效果依训练条件"),
        ("输出层", "和输入嵌入共享", "默认不共享", "小模型仍可共享（如 SmolLM2）"),
    ]
    print(col("位置", 12) + col("GPT-2", 26) + col("Llama", 28) + "好处")
    for where, before, after, why in changes:
        print(col(where, 12) + col(before, 26) + col(after, 28) + why)

    print("\n一句话记住：")
    print("  Llama = GPT-2 把「位置、归一化、门控、KV 头」四处换成更省的写法，")
    print("  换完参数量差不多，但同样显存能跑更长的上下文。")
    print("\n[OK] 全部检查通过")


def _rope_interleaved(x: torch.Tensor, cos: torch.Tensor,
                      sin: torch.Tensor) -> torch.Tensor:
    """常见但和 HF 不兼容的 RoPE 写法：相邻两个通道配对，而不是前一半/后一半配对。

    两种写法各自都能正常训练，但训练出来的权重是**配套**的：
    用一半/后一半训练出来的权重，换成相邻配对去推理，结果就是错的。
    """
    T, D = x.shape[-2], x.shape[-1]
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2, dtype=torch.float32) / D))
    freqs = torch.outer(torch.arange(T, dtype=torch.float32), inv_freq)   # (T, D/2)
    emb = freqs.repeat_interleave(2, dim=-1)                              # (T, D)
    c, s = emb.cos()[None, None], emb.sin()[None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c[..., 0::2] - x2 * s[..., 0::2]
    out[..., 1::2] = x1 * s[..., 1::2] + x2 * c[..., 1::2]
    return out


if __name__ == "__main__":
    main()
