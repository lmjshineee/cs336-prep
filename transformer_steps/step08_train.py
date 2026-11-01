"""Step 8: 训练、生成，以及 KV cache。

训练（teacher forcing）：
    x = data[i : i+T]       输入
    y = data[i+1 : i+T+1]   目标 = 输入右移一位
    一次前向就在 T 个位置上同时算损失，效率很高。
    注意 y 的每个位置都是「用 x 的前 t 个 token 预测第 t 个」，
    这正是因果掩码保证的。

生成是两个阶段：
    prefill  把 prompt 一次性喂进去，算出并缓存所有层的 K、V
    decode   每一步只喂「上一个新生成的 token」，长度 T=1

为什么要 KV cache：
    生成第 t 个 token 时，注意力需要位置 0..t 的 K、V。
    但这些 K、V 只由 token 本身和它的位置决定，不依赖后面生成的词，
    所以算过一次就不会变。没有缓存时，每一步都要把整段前缀重新算一遍，
    生成 N 个 token 总共 O(N^3)（每步 O(t^2)，再对 t 求和）。
    有了缓存，每步只算一个新 token 的 q/k/v，
    注意力对 t 个历史位置做加权，是 O(t)，
    总共 O(N^2)——省掉的正是重复计算。

缓存大小：
    2 * n_layer * B * T_max * C * sizeof(dtype)
    两个 2 分别来自 K 和 V。注意它随 batch、层数、上下文长度线性增长，
    这是长上下文推理时显存的主要开销之一。

一个限制：
    这里用的是「绝对位置 + 学习式位置嵌入」，所以缓存的下标就是绝对位置。
    一旦序列超过 block_size，不能简单地把窗口往右滑（位置全变了），
    只能丢掉缓存重算。RoPE 这类相对位置编码可以缓解这个问题。
"""

import time

import torch
import torch.nn.functional as F

from step07_model import (CORPUS, CharTokenizer, GPTConfig, KVCache,
                          MiniGPT)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int,
              device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # 随机取 batch_size 个起点，切出等长的 x 和右移一位的 y
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


def _next_token(logits: torch.Tensor, temperature: float,
                top_k: int | None, greedy: bool) -> torch.Tensor:
    """从 (B,V) 的 logits 里选下一个 token。

    两个生成函数共用这一段，保证采样逻辑完全一致，
    这样对比结果只反映「有没有缓存」，不掺杂其他差异。
    """
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[:, [-1]]] = float("-inf")
    return torch.multinomial(F.softmax(logits, dim=-1), 1)


@torch.no_grad()
def generate_naive(model: MiniGPT, idx: torch.Tensor, max_new_tokens: int,
                   temperature: float = 0.8, top_k: int | None = 20,
                   greedy: bool = False) -> torch.Tensor:
    """没有缓存：每一步都把整段前缀重新喂一遍。"""
    model.eval()
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.config.block_size:])
        nxt = _next_token(logits[:, -1, :], temperature, top_k, greedy)
        idx = torch.cat((idx, nxt), dim=1)
    return idx


@torch.no_grad()
def generate_cached(model: MiniGPT, idx: torch.Tensor, max_new_tokens: int,
                    temperature: float = 0.8, top_k: int | None = 20,
                    greedy: bool = False) -> torch.Tensor:
    """KV cache：prompt 走一次 prefill，之后每步只喂 1 个新 token。"""
    model.eval()
    config = model.config
    if idx.size(1) + max_new_tokens > config.block_size:
        raise ValueError(
            f"prompt {idx.size(1)} + new {max_new_tokens} "
            f"> block_size {config.block_size}；缓存版不能滑动窗口")

    cache = KVCache(config, idx.size(0), idx.device,
                    dtype=model.token_embedding.weight.dtype)

    # prefill：一次算完 prompt 所有位置，并把每层的 K/V 填进缓存
    logits, _ = model(idx, kv_cache=cache, start_pos=0)
    pos = idx.size(1)          # 下一个要生成的 token 的绝对位置

    for _ in range(max_new_tokens):
        nxt = _next_token(logits[:, -1, :], temperature, top_k, greedy)
        idx = torch.cat((idx, nxt), dim=1)

        if idx.size(1) >= config.block_size:
            break

        # decode：T=1，只算新 token 的 q/k/v，k/v 追加进缓存
        logits, _ = model(nxt, kv_cache=cache, start_pos=pos)
        pos += 1

    return idx


@torch.no_grad()
def check_equivalence(model: MiniGPT, prompt: torch.Tensor,
                      n_steps: int = 8) -> None:
    """逐步对比：缓存版和全量重算版，每个位置的 logits 是否一致。

    这比「生成的句子一样」更严格——采样有随机性，
    而 logits 直接反映模型的计算结果。
    """
    model.eval()
    cache = KVCache(model.config, prompt.size(0), prompt.device,
                    dtype=model.token_embedding.weight.dtype)

    full, _ = model(prompt)
    cached, _ = model(prompt, kv_cache=cache, start_pos=0)
    diff = (full - cached).abs().max().item()
    print(f"prefill: max|diff| = {diff:.2e}")
    assert torch.allclose(full, cached, atol=1e-5)

    seq = prompt
    nxt = full[:, -1, :].argmax(dim=-1, keepdim=True)   # 贪心，去掉随机性
    for step in range(1, n_steps + 1):
        seq = torch.cat((seq, nxt), dim=1)

        full, _ = model(seq)
        cached, _ = model(nxt, kv_cache=cache, start_pos=seq.size(1) - 1)

        diff = (full[:, -1] - cached[:, -1]).abs().max().item()
        print(f"step {step:2d}:   max|diff| = {diff:.2e}")
        assert torch.allclose(full[:, -1], cached[:, -1], atol=1e-5)

        nxt = full[:, -1, :].argmax(dim=-1, keepdim=True)

    print("[OK] 缓存版与全量重算的 logits 逐步一致")


def main() -> None:
    torch.manual_seed(1337)
    device = torch.device("cpu")

    tok = CharTokenizer(CORPUS)
    data = torch.tensor(tok.encode(CORPUS), dtype=torch.long)

    config = GPTConfig(vocab_size=tok.vocab_size, block_size=48,
                       n_layer=2, n_head=4, n_embd=64)
    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  vocab={tok.vocab_size}  params={n_params:,}")
    print(f"语料长度={len(data)} tokens\n")

    model.train()
    for step in range(1, 301):
        xb, yb = get_batch(data, config.block_size, batch_size=8,
                           device=device)
        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step {step:3d} | loss {loss.item():.4f}")

    prompt = "Spring arrives, "
    idx = torch.tensor([tok.encode(prompt)])

    # ---- 1. 正确性：缓存版 == 全量重算版 ----
    print("\n--- 等价性检查 ---")
    check_equivalence(model, idx, n_steps=8)

    # ---- 2. 贪心解码，两种实现必须给出同一段文字 ----
    out_naive = generate_naive(model, idx.clone(), max_new_tokens=24,
                               greedy=True)
    out_cached = generate_cached(model, idx.clone(), max_new_tokens=24,
                                 greedy=True)
    assert torch.equal(out_naive, out_cached)
    print("\n[OK] 贪心解码下两种实现生成完全相同的序列")

    # ---- 3. 采样生成（这里用缓存版）----
    out = generate_cached(model, idx.clone(), max_new_tokens=32,
                          temperature=0.8, top_k=20)
    print("\n--- 生成结果（KV cache）---")
    print(prompt + tok.decode(out[0, len(prompt):].tolist()))

    # ---- 4. 缓存大小 ----
    B, C = 1, config.n_embd
    cache_bytes = 2 * config.n_layer * B * config.block_size * C * 4
    cache = KVCache(config, B, device)
    print(f"\n缓存大小 = 2 * n_layer * B * T_max * C * 4B "
          f"= {cache_bytes:,} bytes ({cache_bytes / 1024:.0f} KiB)")
    assert cache.memory_bytes() == cache_bytes

    # ---- 5. 速度对比 ----
    # 这个玩具模型只有 2 层、64 维，Python 调度开销占比很大，
    # 所以实测加速比远小于理论上限；模型越大、上下文越长，差距越明显。
    def timeit(fn, repeat: int = 3) -> float:
        best = float("inf")
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    t_naive = timeit(lambda: generate_naive(
        model, idx.clone(), max_new_tokens=24, greedy=True))
    t_cached = timeit(lambda: generate_cached(
        model, idx.clone(), max_new_tokens=24, greedy=True))
    print(f"\n生成 24 个 token：naive {t_naive * 1000:.1f} ms  |  "
          f"cached {t_cached * 1000:.1f} ms")
    print("（玩具模型上 Python 开销占比大，加速比远小于理论上限）")

    # ---- 6. 完整性与因果性检查 ----
    model.eval()
    with torch.no_grad():
        x = torch.randint(0, tok.vocab_size, (2, 16))
        logits, _ = model(x)
        assert logits.shape == (2, 16, tok.vocab_size)

        a = torch.randint(0, tok.vocab_size, (1, 12))
        b = a.clone()
        b[:, -1] = (b[:, -1] + 1) % tok.vocab_size
        la, _ = model(a)
        lb, _ = model(b)
        assert torch.allclose(la[:, :-1], lb[:, :-1], atol=1e-6)

    print("\n[OK] 形状检查 + 因果性检查通过")

    # 思考题：
    # 1. 把 max_new_tokens 调到 24 以上会发生什么？（超出 block_size，缓存版报错）
    # 2. 为什么 prefill 阶段仍然需要因果掩码，而 decode 阶段（T=1）几乎不需要？
    # 3. 如果 batch 里有不同长度的序列，缓存该怎么处理 padding？


if __name__ == "__main__":
    main()
