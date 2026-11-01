"""Step 9: 给自写的 Llama 装上真实权重（SmolLM2-135M），生成像样的英文。

前面八步里，模型都是「我们自己初始化、自己训练」的小玩具。
这一步换成 HuggingFace 官方发布的真实权重，看看你写的代码能不能直接吃下来：

    真实模型：HuggingFaceTB/SmolLM2-135M（Apache-2.0，约 257 MiB）
    架构：    采用 Llama 架构的 SmolLM2 —— 30 层、576 维、9 个 query 头配 3 个 kv 头、
             SwiGLU（1536）、RMSNorm(eps=1e-5)、RoPE(theta=100000)、共享输入输出嵌入

为什么它是「最好的试金石」：
    它不大（CPU 上 10 ms 一个 token），但**每一处 Llama 改造都用上了**：
    RMSNorm、SwiGLU、RoPE、GQA（9:3）、tied embedding、bf16 权重转 fp32。
    实现正确性由数值对齐和负例验证；生成流畅本身不能证明实现完全正确。

这一步不训练、不下载新东西（权重用 download_llama_weights.py 下过一次即可），
做四件事：
    1. 把 config.json 翻译成我们的 LlamaConfig；
    2. 把 model.safetensors 的 272 个张量逐个映射进 MiniLlama；
    3. 和 HuggingFace 的同款模型对比 logits、对比贪心生成；
    4. 测一下真实的 prefill / decode 速度和 KV cache 占用。

准备（只需一次）：
    .venv/bin/python llama_steps/download_llama_weights.py

运行：
    .venv/bin/python llama_steps/step09_real_llama.py
"""

from __future__ import annotations

import json
import time
import unicodedata
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, LlamaConfig as HFLlamaConfig
from transformers import LlamaForCausalLM

from step06_model import KVCache, LlamaConfig, MiniLlama

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights" / "smollm2-135m"
DEMO_CTX = 256          # demo 用 256 上下文（真实模型支持 8192；这里只是让 CPU 更快）


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


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GiB"


def load_config(raw: dict) -> LlamaConfig:
    """官方 config.json 的字段名 → 我们的 LlamaConfig 字段名（一一对应，只是叫法不同）。"""
    return LlamaConfig(
        vocab_size=raw["vocab_size"],
        block_size=DEMO_CTX,
        n_layer=raw["num_hidden_layers"],
        n_head=raw["num_attention_heads"],
        n_kv_head=raw["num_key_value_heads"],
        n_embd=raw["hidden_size"],
        ffn_hidden=raw["intermediate_size"],
        rope_theta=float(raw["rope_theta"]),
        norm_eps=float(raw["rms_norm_eps"]),
        dropout=0.0,
        tie_word_embeddings=bool(raw["tie_word_embeddings"]),
    )


def hf_to_ours(sd: dict[str, torch.Tensor], config: LlamaConfig) -> dict[str, torch.Tensor]:
    """HuggingFace 的 state_dict → 我们的 state_dict（纯改名，不改数值）。"""
    mapping = {
        "model.embed_tokens.weight": "token_embedding.weight",
        "model.norm.weight": "norm.weight",
    }
    if not config.tie_word_embeddings:
        mapping["lm_head.weight"] = "lm_head.weight"
    for i in range(config.n_layer):
        hf, ours = f"model.layers.{i}.", f"blocks.{i}."
        mapping[hf + "input_layernorm.weight"] = ours + "attn_norm.weight"
        mapping[hf + "post_attention_layernorm.weight"] = ours + "ffn_norm.weight"
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mapping[hf + f"self_attn.{name}.weight"] = ours + f"attn.{name}.weight"
        for name in ("gate_proj", "up_proj", "down_proj"):
            mapping[hf + f"mlp.{name}.weight"] = ours + f"ffn.{name}.weight"
    allowed_aliases = {"lm_head.weight"} if config.tie_word_embeddings else set()
    unexpected = set(sd) - set(mapping) - allowed_aliases
    if unexpected:
        raise ValueError(f"存在未映射的来源张量: {sorted(unexpected)}")
    if config.tie_word_embeddings and "lm_head.weight" in sd:
        embedding = sd.get("model.embed_tokens.weight")
        if embedding is None or not torch.equal(sd["lm_head.weight"], embedding):
            raise ValueError("共享 lm_head.weight 与输入嵌入不一致")
    return {mapping[k]: v for k, v in sd.items() if k in mapping}


def build_hf(raw: dict) -> LlamaForCausalLM:
    """同一个权重，用 HuggingFace 官方实现再搭一遍，作为对照。"""
    return LlamaForCausalLM(HFLlamaConfig(
        vocab_size=raw["vocab_size"],
        hidden_size=raw["hidden_size"],
        intermediate_size=raw["intermediate_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw["num_key_value_heads"],
        max_position_embeddings=raw["max_position_embeddings"],
        rms_norm_eps=raw["rms_norm_eps"],
        rope_theta=float(raw["rope_theta"]),
        tie_word_embeddings=bool(raw["tie_word_embeddings"]),
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
    ))


def sample_next(logits: torch.Tensor, temperature: float, top_k: int = 40) -> torch.Tensor:
    """温度 + top-k 采样，返回 (B, 1) 的下一个 token。"""
    logits = logits / temperature
    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < values[:, [-1]]] = float("-inf")
    return torch.multinomial(torch.softmax(logits, dim=-1), 1)


@torch.no_grad()
def generate(model: MiniLlama, ids: torch.Tensor, n_new: int, greedy: bool,
             temperature: float = 0.8) -> tuple[torch.Tensor, float]:
    """prefill 后生成；返回结果和每个实际新 token 的生成循环平均耗时。

    计时不含 prefill，包含选词、拼接和继续生成所需的 decode 前向。
    达到生成上限或上下文满后立即返回，不再为最后一个新 token 计算未使用的 logits。
    此时会提前停止，平均耗时按实际生成数计算；n_new=0 是无前向的空操作。
    """
    if type(n_new) is not int or n_new < 0:
        raise ValueError("n_new 必须是非负整数")
    if ids.ndim != 2 or ids.size(0) < 1 or ids.size(1) < 1:
        raise ValueError("ids 必须是非空的 (B,T) 张量")
    if ids.size(1) > model.config.block_size:
        raise ValueError("prompt 超过本实现的 RoPE 表和缓存容量")
    if n_new == 0:
        return ids.clone(), 0.0
    model.eval()
    cache = KVCache(model.config, ids.size(0), ids.device,
                    dtype=model.token_embedding.weight.dtype)
    logits = model(ids, kv_cache=cache, start_pos=0)
    out, pos, t0 = ids.clone(), ids.size(1), time.perf_counter()
    for step in range(n_new):
        if greedy:
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        else:
            nxt = sample_next(logits[:, -1, :], temperature)
        out = torch.cat([out, nxt], dim=1)
        if step + 1 == n_new or pos >= model.config.block_size:
            break
        logits = model(nxt, kv_cache=cache, start_pos=pos)
        pos += 1
    generated = out.size(1) - ids.size(1)
    return out, (time.perf_counter() - t0) / generated


def assert_same_generation(ours: torch.Tensor, reference: torch.Tensor, *,
                           prompt_length: int, expected_new_tokens: int) -> None:
    """核对完整序列及预期新 token 数，禁止仅比较共同前缀。"""
    assert ours.ndim == reference.ndim == 2, "生成结果必须为 (B,T)"
    assert ours.shape == reference.shape, f"生成长度不同: {ours.shape} vs {reference.shape}"
    assert ours.size(1) - prompt_length == expected_new_tokens, "实际生成 token 数不符合预期"
    assert torch.equal(ours, reference), "完整 token 序列不同"


def main() -> None:
    torch.manual_seed(0)
    weight_file = WEIGHTS / "model.safetensors"
    if not weight_file.exists():
        print(f"[SKIP] 没有找到真实权重：{weight_file}")
        print("       先运行：.venv/bin/python llama_steps/download_llama_weights.py")
        return

    print("\n--- 1. 官方 config.json → 我们的 LlamaConfig ---")
    raw = json.loads((WEIGHTS / "config.json").read_text())
    config = load_config(raw)
    show("hidden_size", raw["hidden_size"], "→ n_embd")
    show("intermediate_size", raw["intermediate_size"], "→ ffn_hidden（8/3·C 取整）")
    show("num_hidden_layers", raw["num_hidden_layers"], "→ n_layer")
    show("num_attention_heads", raw["num_attention_heads"], "→ n_head（不是 2 的幂也没关系）")
    show("num_key_value_heads", raw["num_key_value_heads"], "→ n_kv_head（GQA 9:3）")
    show("head_dim", config.n_embd // config.n_head, "= n_embd / n_head")
    show("rope_theta", config.rope_theta, "当前 SmolLM2 配置值，不代表其他 Llama 版本")
    show("rms_norm_eps", config.norm_eps)
    show("tie_word_embeddings", config.tie_word_embeddings, "小模型省参数，输入输出共享")
    # 上面这些字段不是「打印一下」就算对齐：逐个断言 config.json 的值确实落进了我们的 config
    assert config.n_embd == raw["hidden_size"]
    assert config.ffn_hidden == raw["intermediate_size"]
    assert config.n_layer == raw["num_hidden_layers"]
    assert config.n_head == raw["num_attention_heads"]
    assert config.n_kv_head == raw["num_key_value_heads"]
    assert config.n_embd % config.n_head == 0
    assert config.rope_theta == float(raw["rope_theta"])
    assert config.norm_eps == float(raw["rms_norm_eps"])
    assert config.tie_word_embeddings == bool(raw["tie_word_embeddings"])
    print("[OK] 每个字段都真的映射进 LlamaConfig（逐项断言），不是「看起来能对上」")

    t0 = time.perf_counter()
    sd = load_file(str(weight_file))
    print(f"\n--- 2. 把 {len(sd)} 个张量装进 MiniLlama ---")
    ours = MiniLlama(config)
    mapped = hf_to_ours(sd, config)
    if config.tie_word_embeddings:
        # 共享权重时官方文件里没有 lm_head.weight，我们自己补上同一份张量
        mapped["lm_head.weight"] = mapped["token_embedding.weight"]
    ours.load_state_dict(mapped, strict=True)
    ours.eval()
    load_s = time.perf_counter() - t0
    show("权重张量数", len(sd), "safetensors 里的张量个数")
    show("映射后的张量数", len(mapped), "多出来的是共享的 lm_head")
    show("我们的参数量", sum(p.numel() for p in ours.parameters()))
    show("加载耗时", f"{load_s:.2f} s")
    print("[OK] 没有 missing / unexpected 的键（strict=True 通过）")

    print("\n--- 3. 和 HuggingFace 的同款模型对比 ---")
    hf = build_hf(raw).eval()
    missing, unexpected = hf.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    # 共享权重时官方文件里没有单独的 lm_head.weight，这个缺失是预期的
    assert set(missing) <= {"lm_head.weight"}, missing
    tok = AutoTokenizer.from_pretrained(str(WEIGHTS), local_files_only=True)
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids
    show("prompt", repr(prompt))
    show("token ids", ids.tolist(), "(B, T)")
    with torch.no_grad():
        logits_ours = ours(ids)
        logits_hf = hf(ids).logits
    diff = (logits_ours - logits_hf).abs().max().item()
    print(f"{'max|diff|':<22} = {diff:.3e}")
    assert diff < 1e-3, f"和官方实现不一致：{diff:.3e}"
    top5 = logits_ours[0, -1].topk(5).indices.tolist()
    top5_hf = logits_hf[0, -1].topk(5).indices.tolist()
    show("我们 top-5", [tok.decode([i]) for i in top5])
    show("官方 top-5", [tok.decode([i]) for i in top5_hf])
    assert top5 == top5_hf
    print("[OK] 30 层真实权重下，logits 差在 1e-4 量级，top-5 完全一致")

    print("\n--- 4. 用 KV cache 生成，和官方 generate 对比 ---")
    ours_out, s_per_token = generate(ours, ids, 12, greedy=True)
    hf_out = hf.generate(ids, max_new_tokens=12, do_sample=False,
                         eos_token_id=None, pad_token_id=tok.eos_token_id)
    print("[我们的贪心生成]", repr(tok.decode(ours_out[0], skip_special_tokens=True)))
    print("[官方贪心生成]", repr(tok.decode(hf_out[0], skip_special_tokens=True)))
    assert_same_generation(ours_out, hf_out, prompt_length=ids.size(1), expected_new_tokens=12)
    print("[OK] 12 个 token 的贪心生成完全相同")

    sampled, _ = generate(ours, ids, 40, greedy=False, temperature=0.8)
    print("\n[温度 0.8 + top-k 40 采样]")
    print(tok.decode(sampled[0], skip_special_tokens=True))

    print("\n--- 5. 真实速度与显存 ---")
    prefill_t0 = time.perf_counter()
    with torch.no_grad():
        ours(ids)
    prefill_ms = (time.perf_counter() - prefill_t0) * 1000
    show("prefill 耗时", f"{prefill_ms:.1f} ms", f"prompt {ids.size(1)} 个 token")
    show("生成循环平均", f"{s_per_token * 1000:.1f} ms/token",
         f"≈ {1 / s_per_token:.0f} token/s；不含 prefill，含选词/拼接/循环内前向")
    cache = KVCache(config, 1, torch.device("cpu"), dtype=torch.float32)
    show("KV cache @256", human(cache.nbytes()), "fp32，demo 上下文")
    head_dim = config.n_embd // config.n_head
    kv_8k_fp16 = 2 * config.n_layer * config.n_kv_head * head_dim * 8192 * 2
    show("KV cache @8192", human(kv_8k_fp16), "fp16，真实上下文长度")
    print(f"     若 9 个 query 头各自持有 kv（MHA），同样 8k 上下文要 {human(kv_8k_fp16 * 3)}，是 GQA 的 3 倍")

    print("\n一句话记住：")
    print("  当前 SmolLM2 权重可加载进 MiniLlama；本实例通过数值和完整生成序列对照。")
    print("  这是当前输入和环境下的验证结果，流畅文本不能替代实现正确性检查。")
    print("\n[OK] 全部检查通过")


if __name__ == "__main__":
    main()
