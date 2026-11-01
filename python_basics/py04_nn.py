"""nn.Module 与常用层 —— transformer 里的每个类都继承它。

运行：
    .venv/bin/python python_basics/py04_nn.py
"""

import torch
from torch import nn


# ============================================================
# 1. 自定义模型：继承 nn.Module
# ============================================================
print("--- 1. nn.Module ---")


class TinyModel(nn.Module):
    def __init__(self, vocab_size=10, n_embd=4):
        super().__init__()                     # 必须先调用
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.fc = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        x = self.embed(idx)
        return self.fc(x)


model = TinyModel()
idx = torch.tensor([[1, 2, 3]])
print("输入:", idx.shape)             # (1,3)
print("输出:", model(idx).shape)      # (1,3,10)

# 为什么写 model(idx) 而不是 model.forward(idx)？
#   nn.Module.__call__ 会先做内部记录，再调用 forward。
#   两种写法结果一样，但工程上统一用 model(idx)。

print("\n自动登记的参数：")
for name, p in model.named_parameters():
    print(f"  {name:15s} {tuple(p.shape)}")

# 只有赋值成 self.xxx 的子模块/参数才会被自动登记。
# 这就是为什么不能把层放进普通 list（后面 ModuleList 会讲）。


# ============================================================
# 2. nn.Linear：y = x @ W.T + b
# ============================================================
print("\n--- 2. nn.Linear ---")
lin = nn.Linear(4, 6)
x = torch.randn(2, 3, 4)
print("输入:", x.shape, "-> 输出:", lin(x).shape)   # (2,3,6)
print("weight:", lin.weight.shape)                  # (out, in) = (6,4)
print("bias  :", lin.bias.shape)                    # (6,)

# Linear 只作用在最后一维，前面的维度原样保留。
# 参数是 (out_features, in_features)，所以内部算的是 x @ weight.T。


# ============================================================
# 3. nn.Embedding：整数 ID -> 向量（本质是查表）
# ============================================================
print("\n--- 3. nn.Embedding ---")
emb = nn.Embedding(10, 4)
ids = torch.tensor([[1, 2], [3, 4]])          # 必须是 int64
out = emb(ids)
print("输入:", ids.shape, "-> 输出:", out.shape)    # (2,2,4)
print("表大小:", emb.weight.shape)                   # (10,4)
print("第 1 行:", emb.weight[1])
print("emb(1) 和 weight[1] 相同吗:", torch.equal(out[0, 0], emb.weight[1]))


# ============================================================
# 4. LayerNorm 与 Dropout
# ============================================================
print("\n--- 4. LayerNorm / Dropout ---")

ln = nn.LayerNorm(4)
x = torch.randn(2, 4) * 5 + 3
y = ln(x)
print("输入均值  :", x.mean(-1))
print("输出均值  :", y.mean(-1))                    # 接近 0
print("输出标准差:", y.std(-1, unbiased=False))     # 接近 1
print("可学习参数:", [tuple(p.shape) for p in ln.parameters()])  # gamma, beta

drop = nn.Dropout(0.5)
x = torch.ones(1, 10)
drop.train()
print("train 模式:", drop(x))        # 随机把一半位置置零，其余放大
drop.eval()
print("eval 模式 :", drop(x))        # 原样通过

# 所以推理前必须 model.eval()，否则每次输出都在随机变化。


# ============================================================
# 5. nn.Sequential 与 nn.ModuleList
# ============================================================
print("\n--- 5. Sequential / ModuleList ---")

ffn = nn.Sequential(
    nn.Linear(4, 16),
    nn.GELU(),
    nn.Linear(16, 4),
)
print(ffn)
print("输入/输出:", ffn(torch.randn(2, 4)).shape)
print("第 2 层  :", ffn[2])


class Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


stack = Stack()
print("ModuleList 参数个数:", len(list(stack.parameters())))   # 6 = 3 层 * (W+b)

# 如果写成 self.blocks = [nn.Linear(...), ...]（普通 list），
# 这些层不会被登记，优化器看不到它们的参数。
# 这是初学者最常见的坑之一。
# 对应：step07 的 self.blocks = nn.ModuleList([...])


# ============================================================
# 6. train() / eval() / no_grad()
# ============================================================
print("\n--- 6. 训练状态 ---")
model.train()
print("model.training =", model.training)
model.eval()
print("model.training =", model.training)

with torch.no_grad():
    out = model(idx)
print("no_grad 下 requires_grad =", out.requires_grad)

# model.eval()  只影响 Dropout、BatchNorm 等层的行为，不影响求导。
# torch.no_grad() 不记录计算图，省显存、更快，但也不能反向传播。
# 推理时两个都要用：
#   model.eval()
#   with torch.no_grad():
#       ...
# 或者像 step08 那样在函数上加 @torch.no_grad()。
