# Python / PyTorch 基础（配合 `transformer_steps` 阅读）

这套文件只讲 `transformer_steps` 里**实际出现过**的语法和库用法，
每个概念都配了能直接运行的例子。

## 运行顺序

```bash
.venv/bin/python python_basics/py01_syntax.py
```

依次跑 01 到 05。每个文件都是独立的，不依赖前面的文件。

## 文件内容

| 文件 | 内容 | 对应 transformer 里的东西 |
|---|---|---|
| `py01_syntax.py` | 变量、字符串、list/dict/set、推导式、函数、类型注解、切片 | 分词器、配置、所有循环 |
| `py02_classes.py` | 类、`self`、继承、`super()`、dataclass、装饰器、`__call__` | `GPTConfig`、`nn.Module` |
| `py03_tensors.py` | 张量的创建、形状、索引、广播、`view/transpose`、掩码 | 所有形状操作 |
| `py04_nn.py` | `nn.Module`、Linear、Embedding、LayerNorm、Dropout、Sequential、ModuleList | Attention / Block / MiniGPT |
| `py05_train.py` | autograd、优化器、训练四步、生成循环 | step08 |

## 语法 → 代码对照表

| 语法 / API | 示例在 | transformer 里出现的位置 |
|---|---|---|
| `sorted(set(text))` | py01 | `step01` 建词表 |
| 字典推导式 | py01 | `step01` 的 `stoi` / `itos` |
| `enumerate` | py01 | `step01` |
| f-string 格式（`:,` `:.4f`） | py01 | `step07` 打印参数量 |
| 默认参数 | py01 | `GPTConfig`、各个 `forward` |
| `return a, b` 解包 | py01 | `forward` 返回 `(logits, loss)` |
| `assert` | py01 | 每个 step 文件的自检 |
| 类与 `self` | py02 | 所有模型类 |
| 继承 + `super().__init__()` | py02 | `MultiHeadAttention(nn.Module)` |
| `@dataclass` | py02 | `GPTConfig` |
| `@staticmethod` | py02 | `MiniGPT._init_weights` |
| `@torch.no_grad()` | py02 / py04 | `generate` |
| `int \| None`、`list[int]` | py02 | 所有类型注解 |
| 张量索引 `[:, -1, :]` | py03 | `step08` 取最后一个位置 |
| `view` / `transpose` / `contiguous` | py03 | `step05` 多头变形 |
| 广播 `(B,T,C)+(T,C)` | py03 | `step02` 词嵌入 + 位置嵌入 |
| `@` 矩阵乘法 | py03 | `step03` 的 `q @ k.transpose(...)` |
| `masked_fill` + `tril` | py03 | `step04` 因果掩码 |
| `nn.Module` + `forward` | py04 | 每个模型类 |
| `nn.Embedding` / `nn.Linear` | py04 | `step02` / `step03` |
| `nn.LayerNorm` / `nn.Dropout` | py04 | `step06` |
| `nn.Sequential` / `nn.ModuleList` | py04 | `step06` / `step07` |
| `model.eval()` / `model.train()` | py04 | `step08` |
| `zero_grad` / `backward` / `step` | py05 | `step08` 训练循环 |
| `cross_entropy(logits, targets)` | py05 | `step07` 算损失 |

## 读代码时的三个建议

1. **先看形状注释**。transformer 的难点不是语法，而是「这个张量现在是什么形状」。
   每个 step 文件都打印了形状，跑一遍比盯着看快得多。
2. **遇到陌生写法先猜，再跑一个最小例子**。比如不确定 `x[:, -1]` 会不会保留维度，
   就在 `py03_tensors.py` 里加一行 `print(...shape)`。
3. **改坏它**。把 `view` 换成 `reshape`、把 `transpose` 删掉、把 `eval()` 去掉，
   看报什么错。报错信息本身就是最好的老师。

## 如果只想补最必要的部分

时间有限的话，按这个优先级：

1. `py03_tensors.py` 的 1–4 节（形状、索引、广播、变形）——收益最大。
2. `py02_classes.py` 的 1–3 节（类、继承、dataclass）。
3. `py04_nn.py` 的 1、3、5 节（Module、Embedding、ModuleList）。
4. `py05_train.py` 的第 3 节（训练四步）。
