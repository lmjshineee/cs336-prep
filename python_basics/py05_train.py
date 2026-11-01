"""autograd + 优化器 + 训练循环 —— step08 的四行核心代码。

运行：
    .venv/bin/python python_basics/py05_train.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 自动求导：backward 到底做了什么
# ============================================================
print("--- 1. autograd ---")

w = torch.tensor([1.0, 2.0], requires_grad=True)
loss = (w ** 2).sum()        # loss = 1^2 + 2^2 = 5
loss.backward()
print("w.grad =", w.grad)    # dloss/dw = 2w = [2, 4]

# requires_grad=True 表示：记录对 w 的所有运算。
# backward() 沿着计算图反向走一遍，把梯度累加到 w.grad。
# 注意是「累加」——所以每一步训练前要清零，后面会看到。


# ============================================================
# 2. 一个能训练的最小模型（bigram）
# ============================================================
text = "cat sat on the mat. " * 40
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
vocab_size = len(chars)
print("\nvocab_size =", vocab_size, " tokens =", len(data))


class Bigram(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, x, y=None):
        logits = self.head(self.embed(x))               # (B,T,V)
        loss = None
        if y is not None:
            # cross_entropy 期望 (N,V) 的 logits 和 (N,) 的整数标签
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )
        return logits, loss


model = Bigram(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)


# ============================================================
# 3. 训练循环：四个固定动作
# ============================================================
print("\nstep | loss")
for step in range(201):
    x = data[:-1].unsqueeze(0)      # (1,T) 输入
    y = data[1:].unsqueeze(0)       # (1,T) 目标 = 输入右移一位

    _, loss = model(x, y)                   # 1) 前向 + 算损失

    optimizer.zero_grad(set_to_none=True)   # 2) 清掉上一步累加的梯度
    loss.backward()                         # 3) 反向传播，算出梯度
    optimizer.step()                        # 4) 按梯度更新参数

    if step % 50 == 0:
        print(f"{step:4d} | {loss.item():.4f}")

# loss 停在 0.609 左右不是没训练好，而是 bigram 模型的上限：
#   ' ' 后面可能是 c / s / o / t / m，'t' 后面可能是空格 / h / '.'，
#   只看当前一个字符无法区分。想利用更长的上下文，就要上 transformer。

# 为什么顺序是这四步？
#   梯度是累加的，不清零会把上一步的梯度也带上；
#   backward 只算梯度，不改参数；
#   step 才是真正执行 w = w - lr * grad（AdamW 会复杂一些）。


# ============================================================
# 4. 生成：拿最后一个位置的 logits
# ============================================================
model.eval()
with torch.no_grad():
    idx = torch.tensor([[stoi["c"]]])
    for _ in range(30):
        logits, _ = model(idx[:, -8:])
        # logits 是 (1,T,V)，只关心最后一个位置
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # (1,1)
        idx = torch.cat([idx, nxt], dim=1)

print("\n生成：", "".join(itos[i.item()] for i in idx[0]))

# 这段和 step08 的 generate 结构完全一样，只是模型简单得多：
#   logits[:, -1, :] 取最后一个位置
#   argmax / multinomial 选 token
#   torch.cat 接到序列末尾
