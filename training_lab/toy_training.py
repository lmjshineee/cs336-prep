"""toy_training.py —— CS336 A1 §4–6 的训练工具箱（教学实现）。

本模块只依赖 ``torch`` / ``numpy`` / 标准库，**不** import ``model_lab`` 或
``bpe_lab``：训练工具应该能脱离具体模型/分词器单独测试，也避免循环依赖。

和官方 ``assignment1-basics/tests/adapters.py`` 的关系：API 故意不同（参数名更
"教学"，加了 ``rng``/``generator`` 这类显式随机源），但**数值约定严格对齐**官方
参考实现与 ``torch`` 的对应实现，因此可以直接和官方测试口径互相验证：

- ``cross_entropy`` vs ``torch.nn.functional.cross_entropy``
- ``AdamW`` vs ``torch.optim.AdamW``（1e-4 内，官方测试口径）
- ``cosine_lr`` vs 官方 ``test_optimizer.py`` 的 25 个期望值
- ``clip_gradients`` vs ``torch.nn.utils.clip_grad_norm_``
- ``get_batch`` / ``save_checkpoint`` / ``load_checkpoint`` vs 官方
  ``test_data.py`` / ``test_serialization.py`` 的行为

各函数 docstring 里记录了「为什么这么写」和「最容易写错的地方」，测试文件
``test_toy_training.py`` 覆盖 SPEC §4 的 A–H 共 35 项。
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterable, Sequence
from typing import BinaryIO

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "cross_entropy",
    "perplexity",
    "AdamW",
    "cosine_lr",
    "set_lr",
    "clip_gradients",
    "load_tokens",
    "get_batch",
    "save_checkpoint",
    "load_checkpoint",
    "softmax_temperature",
    "top_p_filter",
    "sample_token",
    "generate",
]

# 教学/测试场景全是小张量，限制线程数避免和同仓库其它脚本抢 CPU。
torch.set_num_threads(2)


# ===========================================================================
# 1. 损失：cross-entropy 与 perplexity
# ===========================================================================
def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """交叉熵：对任意前导维度取平均，返回标量。

    公式是 ``logsumexp(logits) - logits[target]``，等价于「先 softmax 再取
    -log」但数值稳定得多：

    - 不先 ``softmax`` 再 ``log``：``softmax`` 可能把很小的概率下溢成 0，
      再取 ``log(0) = -inf``；``logsumexp`` 内部做 max 平移 + log1p 级别的
      稳定计算。
    - 这里仍显式减去每行最大值再 ``logsumexp``，因为如果 logits 整体很大
      （比如 ×1000），直接的 ``logsumexp`` 也可能溢出；减去 max 后
      ``logsumexp`` 与 ``logits[target]`` 里的 max 项互相抵消，结果不变。
      注意梯度会穿过 ``max``，但数学上对任意常数平移不变，所以 autograd
      得到的仍是 ``softmax(logits) - onehot(target)``。

    易错点：官方口径是「对所有前导位置求平均」，所以先把 ``(..., V)``
    reshape 成 ``(N, V)`` 再求均值，而不是只对最后一维或只对 batch 求均值。

    Args:
        logits: ``(..., vocab_size)`` 的浮点张量。
        targets: ``(...)`` 的 int64 张量，取值在 ``[0, vocab_size)``。

    Returns:
        0 维浮点张量，表示所有前导位置上的平均 loss。
    """
    vocab_size = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)

    # 先减每行 max 再 logsumexp：logsumexp 与 target logit 里的 max 项抵消，
    # 既防溢出又不改变结果。keepdim 让广播维度对得上。
    shifted = flat_logits - flat_logits.max(dim=-1, keepdim=True).values
    logsumexp = torch.logsumexp(shifted, dim=-1)
    target_logit = shifted.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
    return (logsumexp - target_logit).mean()


def perplexity(mean_loss: float | Tensor) -> float:
    """``exp(loss)``（perplexity = 平均分支因子）。

    只接受**已经平均过**的 loss：如果这里再对 batch 求一次均值，perplexity
    就不再等于「平均每步有多少种等概率选择」，而是对每个样本先取 exp 再平均，
    两者不等。官方 per-token loss 也是平均后的标量，口径保持一致。

    Args:
        mean_loss: 标量 loss（Python float 或 0 维 Tensor）。

    Returns:
        ``exp(mean_loss)`` 的 Python float。
    """
    if isinstance(mean_loss, Tensor):
        value = float(mean_loss.detach().cpu().item())
    else:
        value = float(mean_loss)
    return math.exp(value)


# ===========================================================================
# 2. 优化器：AdamW（handout Algorithm 1）
# ===========================================================================
class AdamW(torch.optim.Optimizer):
    """AdamW：解耦权重衰减（decoupled weight decay）的 Adam。

    严格按 handout Algorithm 1 的顺序，顺序错一个地方数值就会和官方/`torch`
    对不上：

    1. ``t += 1``（**t 从 1 开始**，不是 0）；
    2. **先**做 weight decay：``p *= (1 - lr * weight_decay)``，用的是 base
       ``lr`` 而不是后面的 ``alpha_t``；
    3. 再更新一阶矩 ``m = beta1*m + (1-beta1)*g``、二阶矩
       ``v = beta2*v + (1-beta2)*g*g``；
    4. 最后一步用偏差修正后的步长
       ``alpha_t = lr * sqrt(1 - beta2**t) / (1 - beta1**t)``，
       ``p -= alpha_t * m / (sqrt(v) + eps)``。

    为什么 weight decay 必须在 moment 更新之前、且用 base lr：AdaGrad 式的
    L2 正则（把 ``wd*p`` 加进梯度）会被二阶矩归一化掉，不再等价于 weight
    decay；AdamW 的解耦写法是「优化器更新完后直接乘一个收缩因子」，因此必须
    在参数更新之前、用未做偏差修正的 lr。

    与 ``torch.optim.AdamW`` 的差异只是实现细节（``lerp_`` vs ``mul_/add_``），
    在 1e-4 内一致（官方测试口径）。
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        # 超参校验放在 super().__init__ 之前：torch 的基类不会替我们检查 lr，
        # 而 SPEC 要求非法 lr 抛 ValueError（官方 AdamW 也是这个行为）。
        if not isinstance(lr, (int, float)) or lr < 0:
            raise ValueError(f"Invalid learning rate: {lr!r}（必须是非负实数）")
        if not isinstance(eps, (int, float)) or eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps!r}（必须是非负实数）")
        if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
            raise ValueError(
                f"Invalid weight_decay value: {weight_decay!r}（必须是非负实数）"
            )
        if len(betas) != 2 or not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError(f"Invalid betas: {betas!r}（要求 0 <= beta < 1）")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """执行一步更新；有 closure 时在其中重新开梯度并返回它的 loss。"""
        loss = None
        if closure is not None:
            # 本方法被 @torch.no_grad() 包裹，closure 里多半要算 loss/backward，
            # 必须显式打开梯度，否则 closure 内部的反传会报错或静默失效。
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    # 延迟初始化：只给真正有梯度的参数分配 moment，省显存。
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                state["step"] += 1
                t = state["step"]
                m = state["exp_avg"]
                v = state["exp_avg_sq"]

                # (2) 解耦 weight decay：在 moment 更新之前、用 base lr。
                p.mul_(1 - lr * weight_decay)

                # (3) 一阶/二阶矩的指数滑动平均。注意二阶矩用 g*g（不是 g.abs()）。
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # (4) 偏差修正后的步长：t 从 1 开始，所以 t=1 时
                # 1-beta1^t = 1-beta1，1-beta2^t = 1-beta2。
                alpha_t = lr * math.sqrt(1.0 - beta2**t) / (1.0 - beta1**t)
                # p -= alpha_t * m / (sqrt(v) + eps)
                p.addcdiv_(m, v.sqrt().add_(eps), value=-alpha_t)

        return loss


# ===========================================================================
# 3. 学习率调度：linear warmup + cosine decay
# ===========================================================================
def cosine_lr(
    it: int,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """linear warmup + cosine decay 的三段式学习率。

    分段（``Tw = warmup_iters``，``Tc = cosine_cycle_iters``）：

    - ``it < Tw``：``it / Tw * max_lr``（从 0 线性升到 max）；
    - ``Tw <= it <= Tc``：
      ``min_lr + 0.5 * (1 + cos((it-Tw)/(Tc-Tw) * pi)) * (max_lr - min_lr)``；
    - ``it > Tc``：``min_lr``。

    边界是官方测试专门盯的地方：

    - ``it == Tw`` 走 cosine 分支且 ``cos(0) = 1``，正好等于 ``max_lr``
      （所以 warmup 段和 cosine 段在 Tw 处连续，不会突然跳一下）；
    - ``it == Tc`` 时 ``cos(pi) = -1``，正好等于 ``min_lr``；
    - ``warmup_iters == 0`` 时 ``it=0`` **不**走 warmup 分支（``0 < 0`` 为假），
      而是 cosine 分支的 ``progress=0``，返回 ``max_lr``，而不是除零。

    易错点：分母写成 ``cosine_cycle_iters - warmup_iters``，不要把 ``it`` 直接
    除以 ``Tc``；并且 ``Tw == Tc`` 时分母为 0，这里保护性地返回 ``min_lr``
    （否则会 ZeroDivisionError，SPEC 未定义这种退化配置）。
    """
    if it < warmup_iters:
        # Tw=0 时这个分支不会进入，避免 ZeroDivisionError。
        return (it / warmup_iters) * max_lr
    if it > cosine_cycle_iters:
        return float(min_lr)

    span = cosine_cycle_iters - warmup_iters
    if span <= 0:
        # 退化配置（Tw == Tc）：没有 cosine 段，直接当作已到终点。
        return float(min_lr)

    progress = (it - warmup_iters) / span
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr + coeff * (max_lr - min_lr))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """原地把 optimizer 所有 param_group 的 lr 都改成 ``lr``。

    为什么遍历 param_groups 而不是只改第一个：不同参数组可以配不同 lr
    （SPEC B4），训练脚本每步调度的是「同一个全局 lr」，所以这里统一覆盖。
    """
    for group in optimizer.param_groups:
        group["lr"] = lr


# ===========================================================================
# 4. 梯度裁剪
# ===========================================================================
def clip_gradients(
    parameters: Iterable[Tensor],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> float:
    """按全局 L2 norm 做梯度裁剪，返回**裁剪前**的全局 norm。

    实现细节与易错点：

    - 全局 norm 是「把所有参数梯度拼成一个长向量」的 L2 norm，不是每个
      gradient 各自算 norm 再取平均；所以要先把所有 ``g**2`` 的和累加，
      最后统一 ``sqrt``。
    - 用 Python ``float``（64 位）累加平方和，而不是在 float32 张量里累加：
      参数很多时 float32 累加会有明显舍入误差。
    - 缩放因子是 ``max / (norm + eps)``，不是 ``max / norm``：加 ``eps``
      既避免 norm 很小时的极端放大，也让「norm 恰好等于 max」时略小于 1。
      官方口径完全一致。
    - 只在 ``norm > max`` 时缩放；``norm <= max`` 时梯度一个 bit 都不动。
    - ``grad is None`` 的参数直接跳过（冻结/未参与本次反传的层）。

    Args:
        parameters: 可迭代的参数集合（``model.parameters()``）。
        max_l2_norm: 允许的最大全局 L2 norm。
        eps: 缩放因子的数值保护项。

    Returns:
        裁剪前的全局 L2 norm（Python float，方便日志记录）。
    """
    grads: list[Tensor] = []
    total_sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        grads.append(p.grad)
        # .double() 让平方和在 64 位里完成，避免 float32 累加的精度损失。
        total_sq += float(p.grad.detach().double().pow(2).sum())

    total_norm = math.sqrt(total_sq)
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(scale)
    return total_norm


# ===========================================================================
# 5. 数据：memmap 加载 + 语言建模 batch 采样
# ===========================================================================
def load_tokens(
    path: str | os.PathLike[str],
    dtype: np.dtype | str = np.uint16,
) -> np.memmap:
    """把裸二进制 token 文件映射成 ``np.memmap``（只读，不占实际内存）。

    为什么用 memmap：TinyStories 全量 token 序列是几 GB 级别，训练时只按
    ``get_batch`` 随机访问局部窗口，全部读进内存既慢又浪费；memmap 让操作系统
    按页缓存，随机访问的窗口才有真实 I/O。

    约定与易错点：文件必须是**裸** token id（没有 header），dtype 由调用方指定
    （本 lab 用 uint16，因为 vocab 2048 < 65536）；``mode="r"`` 只读，避免误写
    fixture 数据。
    """
    return np.memmap(os.fspath(path), dtype=dtype, mode="r")


def get_batch(
    dataset: np.ndarray | np.memmap,
    batch_size: int,
    context_length: int,
    device: str = "cpu",
    rng: np.random.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """从 1D token 序列里随机采 ``(x, y)`` 语言建模 batch。

    对每个样本：``start`` 随机取 ``[0, len(dataset) - context_length)``（左闭
    右开），``x = dataset[start : start+T]``，``y = dataset[start+1 : start+T+1]``。
    返回 ``(B, T)`` 的 int64 张量并放到 ``device``。

    上界是官方测试的死穴：``np.random.randint(0, len - T)`` 的上界是
    **exclusive**，所以最大的 start 是 ``len - T - 1``，这样 ``y`` 的最后一位
    索引是 ``start + T``，最多到 ``len - 1``，永远不越界。如果写成
    ``len - T + 1`` 或闭区间，就会在最后几个位置越界或引入非法起点
    （官方 ``test_get_batch`` 断言 ``max(start) == len - T - 1``）。

    随机源：``rng=None`` 时用全局 ``np.random``（训练脚本默认行为）；传入
    ``np.random.Generator`` 时完全可复现，测试用它验证确定性。
    """
    n_starts = len(dataset) - context_length
    if n_starts <= 0:
        raise ValueError(
            f"dataset 太短：len={len(dataset)}，context_length={context_length}，"
            "至少需要 len > context_length 才能切出一个窗口"
        )

    if rng is None:
        starts = np.random.randint(0, n_starts, size=batch_size)
    else:
        starts = rng.integers(0, n_starts, size=batch_size)

    # 先按 int64 组装，避免 uint16 memmap 切片直接 stack 后再转换的意外 dtype。
    x = np.empty((batch_size, context_length), dtype=np.int64)
    y = np.empty((batch_size, context_length), dtype=np.int64)
    for i, start in enumerate(starts):
        start = int(start)
        x[i] = dataset[start : start + context_length]
        y[i] = dataset[start + 1 : start + context_length + 1]

    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


# ===========================================================================
# 6. checkpoint 保存/恢复
# ===========================================================================
def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike[str] | BinaryIO,
) -> None:
    """把 model / optimizer 的 state_dict 和迭代号打包存盘。

    ``torch.save`` 同时支持路径与 file-like（``io.BytesIO``），所以 ``out``
    直接透传即可。存 ``iteration`` 是为了恢复后训练循环能接续编号
    （学习率调度、日志、checkpoint 命名都依赖它）。
    """
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    torch.save(payload, out)


def load_checkpoint(
    src: str | os.PathLike[str] | BinaryIO,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """恢复 model / optimizer 状态，返回保存时的 iteration。

    两个安全细节：

    - ``map_location="cpu"``：checkpoint 可能是在 GPU 上存的，先落到 CPU 再让
      调用方把模型搬到目标设备，避免恢复路径被设备序号绑死。
    - ``weights_only=True``：只反序列化张量/基础容器，不执行 pickle 里的任意
      Python 对象，防止加载不可信 checkpoint 时被任意代码执行。

    传入 file-like（如 ``BytesIO``）时，调用方需要先把读写位置 ``seek(0)``。
    """
    checkpoint = torch.load(src, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])


# ===========================================================================
# 7. 解码：温度 + top-p（nucleus）采样
# ===========================================================================
def softmax_temperature(logits: Tensor, temperature: float = 1.0) -> Tensor:
    """``softmax(logits / temperature, dim=-1)``。

    温度 < 1 让分布更尖（更接近 argmax），> 1 更平（更随机）；除以温度后直接
    用 ``torch.softmax``，它内部自带 max 平移，数值稳定。

    易错点：``temperature == 0`` 会除零；严格贪心应改用 ``argmax``，或传一个
    极小的正温度。这里对非正温度抛 ``ValueError``，避免静默产生 NaN。
    """
    if temperature <= 0:
        raise ValueError(f"temperature 必须为正数，收到 {temperature!r}")
    return torch.softmax(logits / temperature, dim=-1)


def top_p_filter(probs: Tensor, top_p: float) -> Tensor:
    """nucleus（top-p）过滤：保留累计概率刚达到 ``p`` 的最小 token 集合。

    做法（与 SPEC 一致）：

    1. 按概率**降序**排序；
    2. ``cumsum`` 得到累计和；
    3. 保留满足 ``cumsum - sorted_probs < p`` 的位置，即「在它之前累计的概率
       还不到 p」的 token —— 这样第一个 token 一定在内，且恰好保留到累计和
       首次 >= p 的那个 token（包含它）；
    4. 其余置 0，再按剩余概率归一化。

    易错点：不要用 ``cumsum <= p``，那样累计和恰好等于 p 时会把本该保留的
    边界 token 丢掉，且当 ``p`` 很小时可能一个都不留；``cumsum - sorted_probs``
    是「轮到这个 token 之前的累计和」，配合严格小于号才是官方口径。
    ``p >= 1`` 时保留所有正概率 token；返回张量与输入同形状。
    """
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p 必须在 (0, 1] 内，收到 {top_p!r}")
    if probs.ndim == 0:
        raise ValueError("probs 至少要有 1 维（最后一维是词表维）")

    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # 第一个 token 的 cumsum - sorted_probs == 0 < p，保证至少留一个。
    keep = (cumulative - sorted_probs) < top_p
    kept_probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))

    filtered = torch.zeros_like(probs)
    filtered.scatter_(-1, sorted_indices, kept_probs)
    return filtered / filtered.sum(dim=-1, keepdim=True)


def sample_token(
    logits: Tensor,
    temperature: float = 1.0,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> int:
    """从 ``(vocab,)`` logits 里采样一个 token id。

    先按温度 softmax，可选 top-p 过滤，再 ``torch.multinomial``。传
    ``generator`` 可复现（训练脚本采样时用它固定随机性）。
    """
    flat_logits = logits.reshape(-1)
    probs = softmax_temperature(flat_logits, temperature)
    if top_p is not None:
        probs = top_p_filter(probs, top_p)
    token = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(token.item())


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: Sequence[int] | Tensor,
    *,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_p: float | None = None,
    eos_token_id: int | None = None,
    context_length: int | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """自回归生成：返回 **prompt + 新生成** 的完整 ``list[int]``。

    每一步：

    1. 上下文取「最近 ``context_length`` 个 token」（None 表示不截断；模型有
       位置上限时必须传）；
    2. ``model(x)`` 拿最后一位 logits，按温度/top-p 采样一个 token；
    3. 追加到序列；遇到 ``eos_token_id`` 立即停止（eos 本身保留在返回值里，
       和官方/常见 tokenizer 的约定一致）；
    4. 生成满 ``max_new_tokens`` 也停止。

    易错点：EOS 判断要在 append 之后（eos 包含在返回值里）；上下文截断只影响
    喂给模型的窗口，不影响返回的完整序列；模型可能没有参数（测试里的假模型），
    所以设备推断要容错。
    """
    if isinstance(prompt_ids, Tensor):
        tokens = [int(t) for t in prompt_ids.reshape(-1).tolist()]
    else:
        tokens = [int(t) for t in prompt_ids]
    if not tokens:
        raise ValueError("prompt_ids 不能为空：模型至少需要一个 token 作为输入")
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens 不能为负，收到 {max_new_tokens}")
    if context_length is not None and context_length <= 0:
        raise ValueError(f"context_length 必须为正，收到 {context_length}")

    # 参数里的 device 就是模型所在设备；假模型可能没有参数，回退到 CPU。
    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cpu")

    for _ in range(max_new_tokens):
        window = tokens if context_length is None else tokens[-context_length:]
        x = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(x)
        next_token = sample_token(
            logits[0, -1, :],
            temperature=temperature,
            top_p=top_p,
            generator=generator,
        )
        tokens.append(next_token)
        if eos_token_id is not None and next_token == eos_token_id:
            break

    return tokens
