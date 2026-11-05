"""toy_transformer.py —— 一份「读得懂」的 pre-norm Transformer 教学实现（CS336 A1 §3.3–3.5）。

这不是 A1 的提交答案：
  * API 故意和官方 `tests/adapters.py` 不同（类名/参数更「教学」，额外提供 shape 断言与
    `__main__` 自检），避免你直接复制粘贴交作业；
  * 数值上用官方快照 `tests/_snapshots/*.npz` 对齐（`test_toy_transformer.py` 里 13 个
    parity 用例），所以「教学版」和「参考实现」逐位可比的量级在 1e-6 左右；
  * 实现只依赖 `torch`（`math`/`typing` 来自标准库），不用 einops，不用 nn.Linear 等现成层。

整体数据流（B=batch, T=序列长度, C=d_model, H=num_heads, D=C/H, V=vocab）：

    token_ids (B,T) --Embedding--> (B,T,C)
        --N × TransformerBlock--> (B,T,C)
        --RMSNorm--> (B,T,C) --Linear(C,V)--> logits (B,T,V)

每个 TransformerBlock 是 pre-norm 结构（先归一化，再做子层，最后残差相加）：

    y   = x   + Attention(RMSNorm(x))
    out = y   + FFN(RMSNorm(y))

建议阅读顺序：
  1. 函数：`softmax` → `scaled_dot_product_attention` → `silu`（这三个是纯函数，最好手算；
     `test_toy_transformer.py` 里每一项都有手算/性质对照）；
  2. 基础模块：`Linear` → `Embedding` → `RMSNorm` → `SwiGLU`（含初始化）；
  3. 位置：`RotaryPositionalEmbedding`（RoPE 的「交错配对」约定是全文最容易写错的地方）；
  4. 组装：`CausalMultiHeadSelfAttention` → `TransformerBlock` → `TransformerLM`。

数值约定（与官方参考实现一致，已用官方权重 + 快照反推验证）：
  * `Linear.weight` 存 `(out_features, in_features)`，前向是 `x @ W.T`，没有 bias；
  * 初始化：Linear 用 `trunc_normal_(std=sqrt(2/(d_in+d_out)), a=-3σ, b=3σ)`，
    Embedding 用 `trunc_normal_(std=1, a=-3, b=3)`，RMSNorm 的 gain 初始化为 1；
  * RMSNorm：先 upcast 到 fp32，`x / sqrt(mean(x²) + eps) * g`，再回到原 dtype；
  * RoPE：交错配对 `(x0,x1), (x2,x3), ...`，第 k 对角度 `pos * theta^(-2k/d_k)`，
    `R = [[cos, -sin], [sin, cos]]`，只加在 q/k 上；
  * SDPA 的 mask 是 bool，**True = 可见**；内部 `masked_fill(~mask, -inf)`；
  * MHA 的缩放是 `sqrt(d_k)`（`d_k = d_model // num_heads`），head 维当 batch；
  * `TransformerLM` 默认 untied（`lm_head` 与 `token_embeddings` 不共享权重）。

state_dict 键名（官方权重可直接 `load_state_dict(strict=True)`）：

    token_embeddings.weight
    layers.{i}.attn.q_proj.weight / k_proj / v_proj / output_proj
    layers.{i}.ln1.weight / ln2.weight
    layers.{i}.ffn.w1.weight / w2 / w3
    ln_final.weight
    lm_head.weight
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

__all__ = [
    "softmax",
    "scaled_dot_product_attention",
    "silu",
    "Linear",
    "Embedding",
    "RMSNorm",
    "SwiGLU",
    "RotaryPositionalEmbedding",
    "CausalMultiHeadSelfAttention",
    "TransformerBlock",
    "TransformerLM",
]


# ---------------------------------------------------------------------------
# 1. 纯函数：softmax / scaled dot-product attention / SiLU
# ---------------------------------------------------------------------------
def softmax(x: Tensor, dim: int) -> Tensor:
    """沿 `dim` 做数值稳定的 softmax。

    softmax(v)_i = exp(v_i) / Σ_j exp(v_j)。直接 exp 会在 v 很大时溢出成 inf，
    而 softmax 对「整体平移」不变，所以先减掉该维最大值（最大值变成 0，exp 不会溢出）：

        softmax(x) = softmax(x - max(x, dim, keepdim=True))

    Args:
        x: 任意形状的浮点张量。
        dim: 归一化的维度，可以是负数（`-1` 表示最后一维）。

    Returns:
        与 `x` 同形状的浮点张量，沿 `dim` 的每个切片和为 1。

    注意：如果某一行全是 `-inf`（整行被 mask 掉），`max = -inf`，`x - max` 会产生
    `nan`。这是「全屏蔽行」本身的病态输入，正常因果 mask 每行至少有对角线可见。
    """
    x_max = x.max(dim=dim, keepdim=True).values
    shifted = x - x_max  # 数值稳定：最大值变成 0
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """缩放点积注意力：`softmax(q @ k.T / sqrt(d_k)) @ v`。

    形状（`...` 是任意多个 batch 维，head 维也当作 batch 维处理）：

        q: (..., T_q, d_k)   k: (..., T_k, d_k)   v: (..., T_k, d_v)
        mask: (..., T_q, T_k) 的 bool 张量，True 表示该位置「可见」
        return: (..., T_q, d_v)

    mask 的语义是「信息能否流动」：`mask[..., i, j] = True` 表示第 i 个 query 可以
    看到第 j 个 key。实现上把 False 的位置在 softmax 之前填成 `-inf`，softmax 之后
    这些位置的权重恰好是 0，True 的位置权重和为 1。

    为什么除以 `sqrt(d_k)`：q·k 的方差随 d_k 线性增长，不缩放的话 softmax 会饱和
    （梯度接近 0）。handout §3.4.4 的公式 (11)。

    Args:
        q: query，最后一维是每个 head 的维度 d_k。
        k: key，最后一维是 d_k。
        v: value，最后一维是 d_v（可以和 d_k 不同）。
        mask: 可选 bool mask；True=可见。支持任意前导 batch 维广播。

    Returns:
        形状 `(..., T_q, d_v)`。
    """
    d_k = q.shape[-1]
    # (..., T_q, d_k) @ (..., d_k, T_k) -> (..., T_q, T_k)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError(f"mask 必须是 bool 张量，得到 {mask.dtype}")
        # True 保留（信息流动），False 填 -inf（softmax 后权重为 0）
        scores = scores.masked_fill(~mask, float("-inf"))
    # softmax 后与 value 加权求和： (..., T_q, T_k) @ (..., T_k, d_v)
    return softmax(scores, dim=-1) @ v


def silu(x: Tensor) -> Tensor:
    """SiLU / Swish 激活：`x * sigmoid(x)`（handout 公式 (5)）。

    与 ReLU 不同，它在 0 附近光滑；用 `torch.sigmoid` 而不是手写
    `1/(1+exp(-x))`，后者在 x 很负时会溢出。
    """
    return x * torch.sigmoid(x)


# ---------------------------------------------------------------------------
# 2. 基础模块：Linear / Embedding / RMSNorm / SwiGLU
# ---------------------------------------------------------------------------
def _truncated_normal_(param: Tensor, std: float) -> None:
    """就地把参数初始化成 `N(0, std²)` 截断在 `[-3σ, 3σ]` 的正态分布。

    截断正态的作用：既保留正态的尺度，又不让个别权重一开始就特别大（爆炸）。
    `torch.nn.init.trunc_normal_` 是 handout 指定的初始化工具。
    """
    nn.init.trunc_normal_(param, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)


class Linear(nn.Module):
    """无 bias 的线性变换 `y = x @ W.T`，`W` 形状 `(out_features, in_features)`。

    handout §3.3.2 的约定：**存 W 而不是 W.T**（和 nn.Linear 一致），这样官方
    state_dict 里的 `(d_out, d_in)` 权重可以原样 load 进来。

    不实现 bias：现代 LLM（PaLM/LLaMA 等）基本都不用 bias，少一组参数。

    初始化：`trunc_normal_(std=sqrt(2/(in+out)))` 截断在 ±3σ。这个 std 是
    「fan-in 和 fan-out 折中」的 Xavier 风格尺度，能让前向激活的方差大致保持稳定。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 参数形状是 (out, in)：前向时 x @ weight.T 才是 (…, in) -> (…, out)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        _truncated_normal_(self.weight, std)

    def forward(self, x: Tensor) -> Tensor:
        """`(..., in_features)` -> `(..., out_features)`。"""
        # 用 matmul 而不是 torch.functional.linear，保持「自己写」的可见性
        return x @ self.weight.transpose(-2, -1)

    def extra_repr(self) -> str:  # pragma: no cover - 仅方便 print(model)
        return f"in_features={self.in_features}, out_features={self.out_features}"


class Embedding(nn.Module):
    """查表式 token embedding：`(vocab, d_model)`，前向就是高级索引。

    handout §3.3.3：不要用 `nn.Embedding`，自己存一个 nn.Parameter。
    初始化 `trunc_normal_(std=1, a=-3, b=3)`（注意 std=1，比 Linear 大很多）。

    forward 的 ids 可以是任意形状 `(...)`，输出是 `(..., d_model)`：1D 的
    `(T,)`、2D 的 `(B,T)`、3D 的 `(B,T,K)` 都只是「多套一层形状」。
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        _truncated_normal_(self.weight, std=1.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        """`(...)` 的整数 id -> `(..., embedding_dim)` 的向量。"""
        return self.weight[token_ids]

    def extra_repr(self) -> str:  # pragma: no cover
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"


class RMSNorm(nn.Module):
    """RMSNorm（handout §3.4.1 公式 (4)）：

        RMS(a)      = sqrt(mean(a²) + eps)
        RMSNorm(a)  = a / RMS(a) * g

    和 LayerNorm 的差别：不减均值、也没有 bias，只按「均方根」缩放，然后每个维度
    乘一个可学习的 gain `g`。少一次均值计算，参数也只有 d_model 个。

    数值细节：平方可能溢出（尤其 bf16/fp16 输入），所以**先 upcast 到 fp32**，
    算完再 cast 回输入 dtype。eps=1e-5 防止 x 全 0 时除零。
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # gain 初始化为全 1：一开始 RMSNorm 就等价于「只做归一化」
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        """`(..., d_model)` -> 同形状。输入 dtype 保持不变。"""
        in_dtype = x.dtype
        x32 = x.to(torch.float32)  # 平方前 upcast，避免 fp16/bf16 溢出
        rms = torch.sqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x32 / rms * self.weight.to(torch.float32)
        return out.to(in_dtype)  # 回到输入 dtype，保证「dtype 透传」

    def extra_repr(self) -> str:  # pragma: no cover
        return f"d_model={self.d_model}, eps={self.eps}"


def _default_d_ff(d_model: int) -> int:
    """SwiGLU 的默认内层维度：`8/3 * d_model` 取最近的 64 的倍数，最小 64。

    为什么是 8/3：SwiGLU 有 3 个矩阵（W1/W2/W3）而 ReLU-FFN 只有 2 个，为了让
    参数量大致对齐，就把每个矩阵的宽度乘 2/3；再乘原来的 4 倍 → 8/3。

    为什么取 64 的倍数：GPU 的矩阵乘内核对齐到 64/128 更高效。

    例：d_model=512 -> 1365.33 -> 1344；d_model=1600 -> 4266.67 -> 4288。

    取整规则是 **四舍五入（half-up）**：`64 * floor(8d/(3*64) + 0.5)`，而不是 Python 的
    `round()` —— 后者是「银行家舍入」（round(2.5) == 2），会在恰好落在两个 64 倍数中间时
    向下漂，和 `accounting.py` 的解析式核算对不上。例：d_model=60 时 8/3*60 = 160，
    160/64 = 2.5，half-up 得 3（-> 192），银行家舍入得 2（-> 128）。
    """
    # 用与 accounting.d_ff_default 完全相同的 half-up 公式，保证两处结果一致
    n = math.floor(8.0 * d_model / (3.0 * 64.0) + 0.5)
    return max(64, 64 * n)


class SwiGLU(nn.Module):
    """SwiGLU 前馈网络（handout §3.4.2 公式 (7)）：

        FFN(x) = W2( SiLU(W1 x) ⊙ (W3 x) )

    三个矩阵都没有 bias：
        w1: (d_ff, d_model)   「门」的候选值，过 SiLU
        w3: (d_ff, d_model)   「门控」信号，不过激活
        w2: (d_model, d_ff)   把 d_ff 投回 d_model

    为什么需要两个上投影：GLU 的思想是让一个线性分支（w3）控制另一个分支（w1 经
    SiLU）的信息流，乘起来比单路上投影表达力更强。
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = _default_d_ff(d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """`(..., d_model)` -> `(..., d_model)`（逐位置，不动序列维）。"""
        return self.w2(silu(self.w1(x)) * self.w3(x))

    def extra_repr(self) -> str:  # pragma: no cover
        return f"d_model={self.d_model}, d_ff={self.d_ff}"


# ---------------------------------------------------------------------------
# 3. RoPE：旋转位置编码
# ---------------------------------------------------------------------------
class RotaryPositionalEmbedding(nn.Module):
    """RoPE（handout §3.4.3）：把 q/k 的相邻两维当成二维向量旋转一个角度。

    对 d_k 维向量，分成 d_k/2 对：`(x_0, x_1), (x_2, x_3), ...`（**交错配对**，
    不是「前半/后半拆开」）。第 k 对（k 从 0 开始）在位置 pos 处旋转：

        θ_{pos,k} = pos * theta^(-2k/d_k)
        [x'_2k    ]   [ cos θ   -sin θ ] [ x_2k   ]
        [x'_{2k+1}] = [ sin θ    cos θ ] [ x_{2k+1}]

    关键性质：`dot(R_i q, R_j k)` 只依赖 `(i - j)`，所以同一条序列上的相对位置
    信息被保留；这也是 RoPE 能外推到更长上下文的基础。

    实现要点：
      * 没有可学习参数；cos/sin 表在 `__init__` 里算好，用
        `register_buffer(..., persistent=False)` 存 —— 不进 state_dict，但会跟着
        `.to(device)` 移动；
      * 表形状 `(max_seq_len, d_k//2)`，dtype 是 fp32（用前再 cast 到 x.dtype）；
      * `token_positions` 形状 `(..., T)`，用高级索引按位置取表；与 x 的
        `(..., H, T, d_k)` 靠「靠右对齐」广播（MHA 里会先给 positions 插一个
        head 维，见 `CausalMultiHeadSelfAttention.forward`）。
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"RoPE 要求 d_k 为偶数（交错配对），得到 {d_k}")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 每对的角频率：theta^(-2k/d_k)，k = 0..d_k/2-1
        k = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        inv_freq = theta ** (-k / d_k)  # (d_k/2,)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = positions[:, None] * inv_freq[None, :]  # (max_seq_len, d_k/2)

        # persistent=False：cos/sin 是常数，不需要进 state_dict（否则 strict load 会报错）
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """`(..., T, d_k)` + 位置 `(..., T)` -> `(..., T, d_k)`（旋转后的 q 或 k）。"""
        if x.shape[-1] != self.d_k:
            raise ValueError(f"输入最后一维应为 d_k={self.d_k}，得到 {x.shape[-1]}")
        # 高级索引：(..., T) -> (..., T, d_k/2)；pos 超出预计算表会直接 IndexError
        idx = token_positions.to(device=self.cos_cached.device, dtype=torch.long)
        cos = self.cos_cached[idx].to(x.dtype)
        sin = self.sin_cached[idx].to(x.dtype)

        # 交错配对：偶数下标和奇数下标各取一半，旋转后再交错拼回去
        x_even = x[..., 0::2]  # (..., T, d_k/2)
        x_odd = x[..., 1::2]  # (..., T, d_k/2)
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        # stack 在最后一维得到 (..., T, d_k/2, 2)，flatten 即「交错」还原
        return torch.stack((out_even, out_odd), dim=-1).flatten(-2)

    def extra_repr(self) -> str:  # pragma: no cover
        return f"theta={self.theta}, d_k={self.d_k}, max_seq_len={self.max_seq_len}"


# ---------------------------------------------------------------------------
# 4. 因果多头自注意力
# ---------------------------------------------------------------------------
class CausalMultiHeadSelfAttention(nn.Module):
    """因果多头自注意力（handout §3.4.5 公式 (12)–(14)）。

    计算流程（B=batch, T=序列长, C=d_model, H=头数, D=C/H）：

        q = W_Q x, k = W_K x, v = W_V x            # 各一次矩阵乘，共 3 次
        q,k,v: (B,T,C) -> (B,H,T,D)                # 头维当 batch
        q = RoPE(q), k = RoPE(k)                   # 只旋转 q/k，v 不旋转
        o = softmax(q kᵀ / √D + causal_mask) v     # 每个头独立做 SDPA
        o: (B,H,T,D) -> (B,T,C)
        return W_O o                               # 输出投影

    两个容易写错的点：
      1. 缩放用的是 **每个 head 的维度 D**，不是 d_model；
      2. causal mask 用 `torch.tril`（**下三角 True = 可见**），方向搞反会让模型
         「看到未来」，loss 看起来很好但生成时完全不能用。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} 必须能被 num_heads={num_heads} 整除"
            )
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个 head 的维度，同时是 RoPE 的 d_k
        self.rope = rope  # 可以为 None（对应官方 test_multihead_self_attention）

        # 属性名必须是 q_proj/k_proj/v_proj/output_proj：官方 state_dict 的键名
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def _split_heads(self, x: Tensor) -> Tensor:
        """`(..., T, C)` -> `(..., H, T, D)`：把 head 维拆出来并挪到序列维前面。

        注意 transpose 后张量不连续，后面做 `@` 没问题；最后合并时再 transpose 回去。
        """
        *leading, seq_len, _ = x.shape
        return x.reshape(*leading, seq_len, self.num_heads, self.d_k).transpose(-3, -2)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """`(..., T, d_model)` -> `(..., T, d_model)`；默认位置是 `0..T-1`。

        `token_positions` 的形状约定（x 的 batch 维是 `...`）：

          * `(T,)`      —— 所有 batch 共享同一组位置（最常用，等价于默认）；
          * `(1, T)`    —— 显式的「单样本」位置，靠右对齐广播；
          * `(B, T)`    —— 每个样本各自的位置，内部会在 head 维前插一个 1 变成 `(B,1,T)`；
          * `(1, 1, T)` —— 已经带 head 维的位置，原样广播。

        注意 `(B, H, T)`（每个 head 不同位置）**不是这里支持的形状**：3 维输入 x=`(B,T,C)`
        的 batch 维只有一个，位置要么共享 `(T,)`、要么逐样本 `(B,T)`；`(B,H,T)` 只在 H 恰好
        等于 num_heads 时会「碰巧」广播成每 head 各用一组位置，属未测试行为，不要依赖。
        4 维及以上输入想逐 batch 给位置时，请自己补 1：`(B1, B2, 1, T)`。
        """
        seq_len = x.shape[-2]
        q = self._split_heads(self.q_proj(x))  # (..., H, T, D)
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                # 默认「从 0 开始连续」：(T,) 与 (..., H, T, D) 靠右对齐广播即可
                token_positions = torch.arange(seq_len, device=x.device)
            elif token_positions.ndim == x.ndim - 1:
                # 用户给了 (..., T)，其 leading 维与 x 的 batch 维一一对应，但 q/k 多了
                # 一个 head 维（x.ndim 比 positions.ndim 只多 1）。在 head 维处插一个 1，
                # 变成 (..., 1, T)，这样 q/k 的 (..., H, T, D) 才能正确广播。
                # 若只判断 `ndim == x.ndim`，x=(B,T,C) + positions=(B,T) 会漏掉插维，
                # 而 (1,T) 恰好靠右对齐能广播成功，把 bug 掩盖掉。
                token_positions = token_positions.unsqueeze(-2)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        # 下三角 True=可见的因果 mask，形状 (T,T)，广播到所有 batch/head
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        attn_out = scaled_dot_product_attention(q, k, v, causal_mask)  # (..., H, T, D)

        # 合并 head： (..., H, T, D) -> (..., T, H, D) -> (..., T, C)
        merged = attn_out.transpose(-3, -2).reshape(*x.shape)
        return self.output_proj(merged)

    def extra_repr(self) -> str:  # pragma: no cover
        return f"d_model={self.d_model}, num_heads={self.num_heads}, rope={self.rope is not None}"


# ---------------------------------------------------------------------------
# 5. Transformer block（pre-norm）
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """pre-norm Transformer block（handout §3.5 公式 (15) 之后的组装）：

        y   = x + attn(rms1(x))     # 第一个子层：注意力
        out = y + ffn(rms2(y))      # 第二个子层：前馈

    pre-norm 的直觉：从 embedding 到最后一层的输出有一条「干净的残差流」，中间不被
    norm 打断，梯度可以直接回传，深层模型更好训。代价是最后的输出可能尺度偏大，
    所以 LM 在最后还要再套一个 `ln_final`。

    注意属性名 `attn / ln1 / ln2 / ffn` 以及它们内部的名字都是 state_dict 的一部分，
    不能改。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.attn = CausalMultiHeadSelfAttention(
            d_model, num_heads, rope=rope, device=device, dtype=dtype
        )
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        """`(..., T, d_model)` -> 同形状。"""
        y = x + self.attn(self.ln1(x), token_positions)
        return y + self.ffn(self.ln2(y))


# ---------------------------------------------------------------------------
# 6. 完整语言模型
# ---------------------------------------------------------------------------
class TransformerLM(nn.Module):
    """完整的 decoder-only Transformer 语言模型。

        logits = lm_head(ln_final(blocks(embedding(token_ids))))

    构建参数：
        vocab_size       词表大小 V
        context_length   最大上下文长度（同时决定 RoPE cos/sin 表的长度）
        d_model          隐藏维度 C
        num_layers       block 数量 N
        num_heads        注意力头数 H（必须整除 d_model）
        d_ff             FFN 内层维度
        rope_theta       RoPE 的 Θ（默认 10000.0）
        tie_weights      是否让 lm_head 与 token_embeddings 共享权重（默认 False）

    关于 tie：GPT-2 是 untied（两份独立的 (V,C) 矩阵），官方测试 fixture 也是
    untied；只有部分模型（如早期 Transformer、部分小模型）才 tie。默认 False，
    tie 的路径靠 `test_tie_weights` 单独验证。

    所有层共享**同一个** RoPE 实例：cos/sin 是常数，没必要每个 block 各算一份。
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        tie_weights: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} 必须能被 num_heads={num_heads} 整除"
            )
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.tie_weights = tie_weights

        # 所有 block 共享一个 RoPE 实例（无参数，只有非持久化 buffer）
        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_k=d_model // num_heads,
            max_seq_len=context_length,
            device=device,
        )
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    rope=self.rope,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

        if tie_weights:
            # 直接共享同一个 Parameter 对象：两份权重永远同步，state_dict 里两个键
            # 指向同一块存储，num_parameters() 也只算一次。
            self.lm_head.weight = self.token_embeddings.weight

    def forward(self, token_ids: Tensor) -> Tensor:
        """`(B, T)` 的 token id -> `(B, T, V)` 的 logits。

        T 可以小于 context_length（例如训练时截断的输入）；但 T 超过
        context_length 会直接报错 —— 因为 RoPE 表只有那么长，也不符合模型定义。
        """
        seq_len = token_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError(
                f"输入序列长度 {seq_len} 超过 context_length={self.context_length}"
            )
        x = self.token_embeddings(token_ids)  # (B,T,C)
        for layer in self.layers:  # 每层默认用位置 0..T-1（靠右对齐广播）
            x = layer(x)
        x = self.ln_final(x)  # (B,T,C)
        return self.lm_head(x)  # (B,T,V)

    def num_parameters(self) -> int:
        """可训练参数总数（tie 时共享参数只算一次）。"""
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# 7. 自检：直接 `python model_lab/toy_transformer.py` 时跑一个小模型的形状检查
# ---------------------------------------------------------------------------
def _self_check() -> None:
    """构建一个很小的 LM，检查形状、参数量与 state_dict 键名（< 1 秒）。"""
    torch.set_num_threads(2)
    torch.manual_seed(0)
    model = TransformerLM(
        vocab_size=257,
        context_length=16,
        d_model=32,
        num_layers=2,
        num_heads=4,
        d_ff=64,
    )
    ids = torch.randint(0, 257, (2, 12))
    logits = model(ids)
    assert logits.shape == (2, 12, 257), logits.shape

    # 手算一次参数量，确认 num_parameters() 与解析式一致
    V, C, N, H, d_ff, T_ctx = 257, 32, 2, 4, 64, 16
    d = C
    expected = (
        V * d  # token_embeddings
        + N * (4 * d * d + 3 * d * d_ff + 2 * d)  # 每层 attn + ffn + 两个 RMSNorm
        + d  # ln_final
        + V * d  # lm_head（untied）
    )
    assert model.num_parameters() == expected, (model.num_parameters(), expected)

    # 键名检查：和官方参考实现的命名保持一致（不含 RoPE 的非持久 buffer）
    keys = set(model.state_dict().keys())
    assert "token_embeddings.weight" in keys
    assert "layers.0.attn.q_proj.weight" in keys
    assert "layers.0.ffn.w2.weight" in keys
    assert "ln_final.weight" in keys and "lm_head.weight" in keys
    assert all("rope" not in k for k in keys), [k for k in keys if "rope" in k]

    loss = torch.nn.functional.cross_entropy(
        logits[:, -1, :], torch.randint(0, 257, (2,))
    )
    loss.backward()
    assert all(p.grad is not None for p in model.parameters())

    print(f"shape ok: {tuple(logits.shape)}")
    print(f"num_parameters ok: {model.num_parameters():,} (解析式一致)")
    print(f"state_dict keys ok: {len(keys)} 个")
    print(f"loss: {loss.item():.4f}（反向传播成功）")
    print("OK: toy_transformer 自检通过")


if __name__ == "__main__":
    _self_check()
