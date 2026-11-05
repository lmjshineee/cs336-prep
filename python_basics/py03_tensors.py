"""PyTorch 张量基础 —— 形状、索引、变形、广播。

运行：
    .venv/bin/python python_basics/py03_tensors.py

读 transformer 代码时，90% 的困惑都来自「这个张量现在是什么形状」。
这份文件把常用的形状操作都跑一遍。
"""

import torch

# ============================================================
# 1. 张量 = 多维数组 + 数据类型 + 所在设备
# ============================================================
print("--- 1. 创建 ---")

a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(a)
print("shape :", a.shape)  # torch.Size([2, 3])，可以当 tuple 用
print("dtype :", a.dtype)  # int64
print("device:", a.device)  # cpu

print(torch.zeros(2, 3))
print(torch.ones(2, 3))
print(torch.arange(5))  # [0,1,2,3,4]
print(torch.randint(0, 10, (2, 3)))

# torch.zeros(2, 3) 和 torch.zeros((2, 3)) 等价


# ============================================================
# 2. 索引与切片
# ============================================================
print("\n--- 2. 索引 ---")

x = torch.arange(24).reshape(2, 3, 4)  # 把它当成 (B,T,C)
print("x.shape        ", x.shape)
print("x[0].shape     ", x[0].shape)  # 取第 0 个 batch
print("x[:, 0].shape  ", x[:, 0].shape)  # 每个 batch 的第 0 个 token
print("x[:, -1].shape ", x[:, -1].shape)  # 最后一个 token
print("x[:, :-1].shape", x[:, :-1].shape)  # 去掉最后一个
print("x[0, 1, 2]     ", x[0, 1, 2])  # 一个标量

# 规则：
#   :       这一维全要
#   整数     这一维只取一个，维度会消失
#   切片     维度保留
#   -1      倒数第一个
#   含头不含尾


# ============================================================
# 3. 矩阵乘法与广播
# ============================================================
print("\n--- 3. 矩阵乘法 / 广播 ---")

A = torch.randn(2, 3)
B = torch.randn(3, 4)
print("A @ B:", (A @ B).shape)  # (2,4)，中间的 3 必须相等
print("A * A:", (A * A).shape)  # 逐元素乘，形状不变

tokens = torch.randn(2, 5, 8)  # (B,T,C)
pos = torch.randn(5, 8)  # (T,C)
print("(B,T,C)+(T,C):", (tokens + pos).shape)  # (2,5,8)

col = torch.randn(3, 1)  # 从右边对齐后是 (1,3,1)
print("(2,3,4)+(3,1):", (torch.randn(2, 3, 4) + col).shape)

# 广播规则：
#   从最右边一维开始逐个对齐；
#   某一维长度为 1 或缺失，就自动「拉伸」成对方的长度。
# step02 的 x = token_embedding(idx) + position_embedding(pos)
# 就是 (B,T,C) + (T,C) 的广播。


# ============================================================
# 4. 变形：view / reshape / transpose / contiguous
# ============================================================
print("\n--- 4. 变形 ---")

x = torch.arange(24).reshape(2, 3, 4)
print("x              ", x.shape)
print("x.view(2,12)   ", x.view(2, 12).shape)
print("x.reshape(6,4) ", x.reshape(6, 4).shape)
print("transpose(1,2) ", x.transpose(1, 2).shape)  # 交换第 1、2 维

# view 要求内存连续；transpose 只换「怎么解读」，不搬数据
y = x.transpose(1, 2)
try:
    y.view(2, 12)
except RuntimeError as e:
    print("直接 view 失败：", str(e).splitlines()[0])

print("先 contiguous 再 view:", y.contiguous().view(2, 12).shape)

# step05 的多头注意力就是这一串：
#   (B,T,C) -> view(B,T,H,D) -> transpose(1,2) -> (B,H,T,D)
# 最后再 transpose 回来，contiguous 之后 view 成 (B,T,C)。


# ============================================================
# 5. 拼接、拆分、增删维度
# ============================================================
print("\n--- 5. 拼接 / 拆分 ---")

a = torch.tensor([1, 2])
b = torch.tensor([3, 4])
print("stack:", torch.stack([a, b]).shape)  # (2,2) 新加一维
print("cat  :", torch.cat([a, b]).shape)  # (4,)  沿已有维拼接

x = torch.randn(2, 6, 12)
q, k, v = x.split(4, dim=2)  # 沿最后一维切成 3 份
print("split:", q.shape, k.shape, v.shape)  # 各 (2,6,4)

t = torch.tensor([1, 2, 3])
print("unsqueeze(0):", t.unsqueeze(0).shape)  # (1,3)
print("unsqueeze(1):", t.unsqueeze(1).shape)  # (3,1)


# ============================================================
# 6. 掩码
# ============================================================
print("\n--- 6. 掩码 ---")

mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))
print("下三角为 True：")
print(mask.int())

scores = torch.randn(4, 4)
masked = scores.masked_fill(~mask, float("-inf"))
print("上三角被填成 -inf：")
print(masked)

# ~ 是「按位取反」，对 bool 张量就是 True <-> False。
# softmax 之后 exp(-inf) = 0，这些位置就彻底没有权重了。


# ============================================================
# 7. 比较与取值
# ============================================================
print("\n--- 7. 比较 ---")

x = torch.tensor([1.0, 2.0])
y = x.clone()
print(torch.equal(x, y))  # 完全相同
print(torch.allclose(x, y + 1e-7, atol=1e-5))  # 允许误差

scalar = torch.tensor(3.14)
print(scalar.item(), type(scalar.item()))  # Python float

# 注意：.item() 只能对「一个数」调用，
# 形状是 (1,) 或 () 的标量都可以。


# ============================================================
# 8. dtype / device / 随机种子
# ============================================================
print("\n--- 8. dtype / device / seed ---")

ids = torch.tensor([1, 2, 3], dtype=torch.long)
print("ids.dtype  :", ids.dtype)  # int64，Embedding 的输入必须是它
print("randn.dtype:", torch.randn(2).dtype)  # float32

torch.manual_seed(1337)  # 固定随机数，保证结果可复现
print("seed 1337  :", torch.randn(3))
torch.manual_seed(1337)
print("再来一次   :", torch.randn(3))  # 和上面完全相同

# .to(device) 把张量搬到 CPU/GPU：
#   model = model.to(device)
#   xb = xb.to(device)
# 两边设备不一致会直接报错。
