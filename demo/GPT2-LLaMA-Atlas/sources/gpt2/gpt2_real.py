"""真实 GPT-2 权重实验的核心模块：**自己的代码 + 官方权重**。

这个文件就是 `transformer_steps/step07_model.py` 的「真实版」：
结构和 step07 完全一样——嵌入 -> N 个 Block（注意力 + FFN）-> ln_f -> lm_head，
连 KV cache 的写法都照抄过来。

唯一的区别是它按 GPT-2 原版的 4 个细节写了一点点：

  1. FFN 的激活函数用 tanh 近似的 GELU（GPT-2 config 里叫 "gelu_new"）；
     step06 用的是精确版 GELU，数值上差 1e-3 量级，对 logits 有影响。
  2. q/k/v 是**一个** Linear 一次算出来的（GPT-2 叫 `c_attn`），
     然后再切成三份——比 step03 的 qkv 组合起来更省一次矩阵乘。
  3. 权重文件里 Linear 的权重是**转置**存的：
     GPT-2 原版用 Conv1D，weight 形状是 (输入维, 输出维)；
     而 nn.Linear 的 weight 是 (输出维, 输入维)，所以加载时要 .t()。
  4. LayerNorm 的 eps = 1e-5（torch 默认也是 1e-5，这里只是显式写出来）。

所以：**你已经学会了 GPT-2 的全部结构**，剩下的只是「怎么把官方权重塞进来」。

怎么用（见 exp1_parity.py）：

    from gpt2_real import load_gpt2
    model, tok = load_gpt2()                      # 读 weights/gpt2/
    ids = tok.encode("The cat sat on the")
    logits = model(torch.tensor([ids]))           # (1, T, 50257)

    model.generate(...)                            # 生成，支持 KV cache
    model.attention_maps(...)                      # 拿到每一层的注意力矩阵
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "weights" / "gpt2"


# ---------------------------------------------------------------------------
# 配置：和 step07 的 GPTConfig 一样，只是默认值换成了 GPT-2 small 的真实尺寸
# ---------------------------------------------------------------------------
@dataclass
class GPT2Config:
    n_layer: int = 12          # 层数
    n_head: int = 12           # 头数
    n_embd: int = 768          # C
    block_size: int = 1024     # GPT-2 里叫 n_positions / n_ctx
    vocab_size: int = 50257    # BPE 词表大小（step01 的字符词表只有几十个）
    layer_norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @classmethod
    def from_json(cls, path: Path) -> "GPT2Config":
        """真实配置就写在 weights/gpt2/config.json 里，直接读。"""
        raw = json.loads(Path(path).read_text())
        return cls(
            n_layer=raw["n_layer"],
            n_head=raw["n_head"],
            n_embd=raw["n_embd"],
            block_size=raw.get("n_positions", raw.get("n_ctx", 1024)),
            vocab_size=raw["vocab_size"],
            layer_norm_eps=raw.get("layer_norm_epsilon", 1e-5),
        )


# ---------------------------------------------------------------------------
# KV cache：和 step07 的 KVCache 完全一样
# ---------------------------------------------------------------------------
class KVCache:
    """每一层预分配一份 K、V，形状 (B, n_head, block_size, head_dim)。

    真实尺寸：2 * 12 层 * 1 * 1024 * 768 * 4 字节 = 75 MiB（单个样本 fp32）。
    """

    def __init__(self, config: GPT2Config, batch_size: int,
                 device: torch.device, dtype: torch.dtype = torch.float32):
        shape = (batch_size, config.n_head, config.block_size, config.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype)
                  for _ in range(config.n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype)
                  for _ in range(config.n_layer)]

    def memory_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)


# ---------------------------------------------------------------------------
# 注意力：step03/05 的多头注意力 + step08 的 KV cache
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_embd = config.n_embd
        # GPT-2 把 q/k/v 合成一个 Linear：C -> 3C，一次矩阵乘算完
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 多头拼接后的混合矩阵：C -> C
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # exp2 用：把这个开关打开，forward 会把注意力矩阵存进 last_attn
        self.record = False
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor,
                k_cache: torch.Tensor | None = None,
                v_cache: torch.Tensor | None = None,
                start_pos: int = 0,
                causal: bool = True) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)   # 各 (B,T,C)
        q = q.view(B, T, H, D).transpose(1, 2)               # (B,H,T,D)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        if k_cache is None:
            k_all, v_all, total = k, v, T
        else:
            assert v_cache is not None
            total = start_pos + T
            k_cache[:, :, start_pos:total] = k
            v_cache[:, :, start_pos:total] = v
            k_all = k_cache[:, :, :total]
            v_all = v_cache[:, :, :total]

        scores = (q @ k_all.transpose(-2, -1)) / (D ** 0.5)  # (B,H,T,total)

        if causal:
            # query 在 start_pos..start_pos+T-1，key 在 0..total-1，
            # 只允许 key_pos <= query_pos（和 step04/step08 的掩码一样）
            q_pos = torch.arange(start_pos, total, device=x.device).unsqueeze(1)
            k_pos = torch.arange(total, device=x.device).unsqueeze(0)
            scores = scores.masked_fill(~(k_pos <= q_pos), float("-inf"))

        weights = F.softmax(scores, dim=-1)
        if self.record:
            self.last_attn = weights.detach()

        out = (weights @ v_all).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(out)


# ---------------------------------------------------------------------------
# FFN：step06 的 FeedForward，只把激活函数换成 GPT-2 用的 tanh 近似 GELU
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.GELU(approximate="tanh")     # GPT-2 的 "gelu_new"
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


# ---------------------------------------------------------------------------
# Block：step06 的 Pre-LN 结构，原样搬过来
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor,
                k_cache: torch.Tensor | None = None,
                v_cache: torch.Tensor | None = None,
                start_pos: int = 0,
                causal: bool = True) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), k_cache, v_cache, start_pos, causal)
        x = x + self.mlp(self.ln_2(x))
        return x


# ---------------------------------------------------------------------------
# 完整模型：step07 的 MiniGPT，换成 GPT-2 的参数名（方便对权重）
# ---------------------------------------------------------------------------
class GPT2(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # token 嵌入
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   # 位置嵌入
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 和 step07 一样：输出层与输入嵌入共享权重
        self.lm_head.weight = self.wte.weight

    def forward(self, idx: torch.Tensor,
                kv_cache: KVCache | None = None,
                start_pos: int = 0,
                causal: bool = True) -> torch.Tensor:
        """返回 logits，形状 (B, T, vocab_size)。

        kv_cache/start_pos 只在推理时用；训练或全量重算时用默认值即可。
        causal=False 是故意的「作弊模式」，exp4 用它演示因果掩码的作用。
        """
        B, T = idx.shape
        pos = torch.arange(start_pos, start_pos + T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)

        for i, block in enumerate(self.h):
            if kv_cache is None:
                x = block(x, causal=causal)
            else:
                x = block(x, kv_cache.k[i], kv_cache.v[i], start_pos, causal)

        return self.lm_head(self.ln_f(x))

    # -- 方便实验用的两个工具函数 ------------------------------------------
    @torch.no_grad()
    def attention_maps(self, idx: torch.Tensor) -> torch.Tensor:
        """返回所有层的注意力权重，形状 (n_layer, B, n_head, T, T)。

        (L,B,H,T,T) 里 weights[l, b, h, i, j] = 第 l 层第 h 个头，
        「位置 i 的 query 分给位置 j 的注意力」。
        """
        for block in self.h:
            block.attn.record = True
        self(idx)
        maps = torch.stack([block.attn.last_attn for block in self.h])
        for block in self.h:
            block.attn.record = False
        return maps

    @classmethod
    def from_pretrained(cls, weights_dir: Path = DEFAULT_WEIGHTS) -> "GPT2":
        weights_dir = Path(weights_dir)
        config = GPT2Config.from_json(weights_dir / "config.json")
        model = cls(config)
        load_weights(model, weights_dir / "model.safetensors")
        model.eval()
        return model


# ---------------------------------------------------------------------------
# 加载权重：唯一的新知识点——safetensors 文件 + 名字/形状转换
# ---------------------------------------------------------------------------
def _copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    assert dst.shape == src.shape, f"形状不匹配: {tuple(dst.shape)} vs {tuple(src.shape)}"
    with torch.no_grad():
        dst.copy_(src)


def load_weights(model: GPT2, path: Path) -> GPT2:
    """把官方 model.safetensors 里的张量逐个塞进 model。

    safetensors 文件本身很简单：开头 8 字节是 header 长度，
    接着是一段 JSON（记录每个张量的名字、dtype、形状、字节偏移），
    后面是原始字节。这里用 safetensors 库读，不用自己解析。

    GPT-2 官方文件里的名字长这样（不同版本可能带 "transformer." 前缀，去掉即可）：
        wte.weight             (50257, 768)      词表嵌入
        wpe.weight             (1024, 768)       位置嵌入
        h.0.ln_1.weight        (768,)            第 0 层 LN
        h.0.attn.c_attn.weight (768, 2304)       注意：要转置
        h.0.attn.c_proj.weight (768, 768)        注意：要转置
        h.0.mlp.c_fc.weight    (768, 3072)       注意：要转置
        h.0.mlp.c_proj.weight  (3072, 768)       注意：要转置
        h.0.attn.bias          (1,1,1024,1024)   因果掩码缓冲，不是学出来的参数，跳过
    """
    from safetensors.torch import load_file

    state = load_file(str(path), device="cpu")
    state = {k.removeprefix("transformer."): v for k, v in state.items()}

    with torch.no_grad():
        _copy(model.wte.weight, state["wte.weight"])
        _copy(model.wpe.weight, state["wpe.weight"])
        _copy(model.ln_f.weight, state["ln_f.weight"])
        _copy(model.ln_f.bias, state["ln_f.bias"])

        for i, block in enumerate(model.h):
            p = f"h.{i}."
            _copy(block.ln_1.weight, state[p + "ln_1.weight"])
            _copy(block.ln_1.bias, state[p + "ln_1.bias"])
            _copy(block.ln_2.weight, state[p + "ln_2.weight"])
            _copy(block.ln_2.bias, state[p + "ln_2.bias"])
            # Conv1D 的 (输入, 输出) -> nn.Linear 的 (输出, 输入)
            _copy(block.attn.c_attn.weight, state[p + "attn.c_attn.weight"].t())
            _copy(block.attn.c_attn.bias, state[p + "attn.c_attn.bias"])
            _copy(block.attn.c_proj.weight, state[p + "attn.c_proj.weight"].t())
            _copy(block.attn.c_proj.bias, state[p + "attn.c_proj.bias"])
            _copy(block.mlp.c_fc.weight, state[p + "mlp.c_fc.weight"].t())
            _copy(block.mlp.c_fc.bias, state[p + "mlp.c_fc.bias"])
            _copy(block.mlp.c_proj.weight, state[p + "mlp.c_proj.weight"].t())
            _copy(block.mlp.c_proj.bias, state[p + "mlp.c_proj.bias"])

    unused = set(state) - {
        "wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias",
        "lm_head.weight",
    } - {f"h.{i}.{n}" for i in range(model.config.n_layer) for n in (
        "ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias",
        "attn.c_attn.weight", "attn.c_attn.bias",
        "attn.c_proj.weight", "attn.c_proj.bias",
        "mlp.c_fc.weight", "mlp.c_fc.bias",
        "mlp.c_proj.weight", "mlp.c_proj.bias",
        "attn.bias",          # 因果掩码缓冲，不是参数
    )}
    assert not unused, f"权重文件里有没用上的张量: {sorted(unused)[:5]}"

    if "lm_head.weight" in state:          # GPT-2 权重共享，通常没有这一项
        _copy(model.lm_head.weight, state["lm_head.weight"])
    return model


def load_tokenizer(weights_dir: Path = DEFAULT_WEIGHTS):
    """加载 GPT-2 的 BPE 分词器（读 weights/gpt2/tokenizer.json）。

    和 step01 的字符分词器接口一样：encode(text) -> list[int]，decode(list) -> text。
    区别是它的词表有 50257 个「子词」而不是几十个字符。
    """
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(weights_dir), local_files_only=True)


def load_gpt2(weights_dir: Path = DEFAULT_WEIGHTS, device: str = "cpu"):
    """一行加载：真实权重 + 真实分词器。"""
    weights_dir = Path(weights_dir)
    if not (weights_dir / "model.safetensors").exists():
        raise FileNotFoundError(
            f"找不到 {weights_dir / 'model.safetensors'}，"
            "先运行 python real_llm_lab/download_weights.py")
    model = GPT2.from_pretrained(weights_dir).to(device)
    tok = load_tokenizer(weights_dir)
    return model, tok


# ---------------------------------------------------------------------------
# 生成：step08 的两种生成方式，逻辑一字未改
# ---------------------------------------------------------------------------
def _next_token(logits: torch.Tensor, temperature: float,
                top_k: int | None, greedy: bool) -> torch.Tensor:
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[:, [-1]]] = float("-inf")
    return torch.multinomial(F.softmax(logits, dim=-1), 1)


@torch.no_grad()
def generate_naive(model: GPT2, idx: torch.Tensor, max_new_tokens: int,
                   temperature: float = 1.0, top_k: int | None = None,
                   greedy: bool = False) -> torch.Tensor:
    """没有缓存：每一步都把整段前缀重新喂一遍（O(N^3)）。"""
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(idx[:, -model.config.block_size:])
        nxt = _next_token(logits[:, -1, :], temperature, top_k, greedy)
        idx = torch.cat((idx, nxt), dim=1)
    return idx


@torch.no_grad()
def generate_cached(model: GPT2, idx: torch.Tensor, max_new_tokens: int,
                    temperature: float = 1.0, top_k: int | None = None,
                    greedy: bool = False) -> torch.Tensor:
    """KV cache：prompt 走一次 prefill，之后每步只喂 1 个新 token（O(N^2)）。"""
    model.eval()
    config = model.config
    if idx.size(1) + max_new_tokens > config.block_size:
        raise ValueError(
            f"prompt {idx.size(1)} + new {max_new_tokens} "
            f"> block_size {config.block_size}")

    cache = KVCache(config, idx.size(0), idx.device,
                    dtype=model.wte.weight.dtype)
    logits = model(idx, kv_cache=cache, start_pos=0)   # prefill
    pos = idx.size(1)

    for step in range(max_new_tokens):
        nxt = _next_token(logits[:, -1, :], temperature, top_k, greedy)
        idx = torch.cat((idx, nxt), dim=1)
        if step + 1 == max_new_tokens or idx.size(1) >= config.block_size:
            break
        logits = model(nxt, kv_cache=cache, start_pos=pos)   # decode
        pos += 1
    return idx


if __name__ == "__main__":
    # 第一次运行前先跑 download_weights.py（约 550 MB）
    model, tok = load_gpt2()
    n_params = sum(p.numel() for p in model.parameters())

    print(f"参数量 = {n_params:,}  （GPT-2 small 官方数字 124,439,808）")
    print(f"配置   = {model.config}")

    text = "The cat sat on the mat."
    ids = tok.encode(text)
    print(f"\n{text!r} 用 BPE 分词后 = {len(ids)} 个 token: {ids}")
    print("切回来的文本 =", repr(tok.decode(ids)))
    print("每个 token 对应的原文 =", tok.convert_ids_to_tokens(ids))

    logits = model(torch.tensor([ids]))
    print(f"\nlogits.shape = {tuple(logits.shape)}  <- (B, T, V)")
    print(f"下一个 token 的 Top-5: "
          f"{[(tok.decode([i]), round(p, 3)) for p, i in zip(
              F.softmax(logits[0, -1], dim=-1).topk(5).values.tolist(),
              logits[0, -1].topk(5).indices.tolist())]}")

    assert n_params == 124_439_808, "参数对不上，加载有问题"
    print("\n[OK] 真实权重加载成功，前向形状正确")
