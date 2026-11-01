"""Python 基础语法 —— 全部取材于 transformer_steps 里的真实写法。

运行：
    .venv/bin/python python_basics/py01_syntax.py

每个小节开头都写了「在哪个文件里见过」。
"""

# ============================================================
# 1. 变量、类型、字符串
# ============================================================
# Python 不需要声明类型，赋值即创建变量。
# 出现在：step01 的 vocab_size、step07 的 total。

vocab_size = 30          # int
dropout = 0.0            # float
name = "MiniGPT"         # str
is_causal = True         # bool
nothing = None           # 空值，类似其他语言的 null

print("--- 1. 变量 ---")
print(type(vocab_size), type(dropout), type(name), type(nothing))

# f-string：字符串里用 {} 嵌入表达式。
# 出现在：step07 的 f"params = {total:,}"
total = 104576
print(f"params = {total:,}")        # 千分位：104,576
print(f"loss   = {3.14159:.4f}")    # 保留 4 位小数
print(f"step   = {7:3d}")           # 至少占 3 个字符宽

# repr() 显示字符串的「真实样子」（带引号、转义符）
# 出现在：step01 的 print("text ->", repr(sample))
s = "A small cat"
print(s)         # A small cat
print(repr(s))   # 'A small cat'

# 三引号可以写多行字符串；开头的 \ 表示「紧跟的换行不要算进去」
# 出现在：每个文件的 CORPUS
corpus = """\
hello
world
"""
print(repr(corpus))    # 'hello\nworld\n'


# ============================================================
# 2. 容器：list / tuple / dict / set
# ============================================================
print("\n--- 2. 容器 ---")

ids = [4, 1, 24, 19]           # list：可变、有序
point = (2, 3)                 # tuple：不可变，常用来装形状
stoi = {"A": 4, " ": 1}        # dict：键值对
chars = set("cat sat")         # set：去重、无序

print(ids[0], ids[-1])         # 第一个 / 最后一个
print(ids[1:3])                # 切片：含头不含尾
print(ids[:2], ids[2:])        # 省略表示从开头 / 到结尾
print(point, point[0])

# 字典：用 key 取值；in 判断 key 是否存在
print(stoi["A"])
print("c" in stoi)             # False
for ch, i in stoi.items():     # items() 同时拿到键和值
    print("  ", ch, i)

# set + sorted：step01 用它们建立词表
text = "cat sat"
print(set(text))               # 顺序不保证
print(sorted(set(text)))       # 变成有序 list

# 列表推导式：[表达式 for 变量 in 可迭代对象]
squares = [i * i for i in range(5)]
print(squares)

# 字典推导式：出现在 step01 的 stoi / itos
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
print(stoi)
print(itos)

# enumerate：同时给出下标和元素
for i, ch in enumerate(["c", "a", "t"]):
    print(i, ch)


# ============================================================
# 3. 函数、默认参数、类型注解
# ============================================================
print("\n--- 3. 函数 ---")


def encode(text: str) -> list[int]:
    """类型注解只是给人看的，Python 运行时不会检查。"""
    return [ord(ch) % 10 for ch in text]


print(encode("cat"))


# 默认参数：调用时可以省略
def make_config(vocab_size: int, n_layer: int = 2,
                n_embd: int = 64) -> dict:
    return {"vocab_size": vocab_size, "n_layer": n_layer, "n_embd": n_embd}


print(make_config(30))
print(make_config(30, n_layer=4))    # 关键字参数，顺序无所谓


# Python 函数只能 return 一个对象；
# 但 return a, b 会打包成 tuple，调用处可以解包。
def first_last(xs):
    return xs[0], xs[-1]


a, b = first_last([10, 20, 30])
print(a, b)


# ============================================================
# 4. 几个常用惯用法
# ============================================================
print("\n--- 4. 惯用法 ---")

# if __name__ == "__main__":
#   直接运行这个文件时，__name__ 是 "__main__"；
#   被别的文件 import 时，__name__ 是模块名，下面的代码不执行。
#   出现在：每个 step 文件的最后。

# assert：条件不成立就抛 AssertionError，用来写自检。
#   出现在：step01 的 assert tok.decode(...) == CORPUS
x = 3
assert x > 0
print("assert 通过")

# None 判断用 is，不用 ==
loss = None
if loss is not None:
    print("有 loss")
else:
    print("loss 是 None")

# 生成器表达式：不会先建一个完整的 list
#   出现在：step07 的 sum(p.numel() for p in model.parameters())
nums = [1, 2, 3, 4]
print(sum(n * n for n in nums))

# min / max / len / sum / range
print(min(nums), max(nums), len(nums), sum(nums))
print(list(range(3)))        # [0, 1, 2]


# ============================================================
# 5. 实战：复现 step01 里那行 ids
# ============================================================
print("\n--- 5. ids 是怎么来的 ---")

# step01 跑出来的词表：30 个字符，按 ASCII 码从小到大排序。
# 注意第一个字符是换行符 \n，它的 ASCII 码是 10，比空格 32 还小。
vocab = "\n ,.ANSTabcdefghiklmnoprstuvwy"
print("len(vocab) =", len(vocab))

stoi = {ch: i for i, ch in enumerate(vocab)}
sample = "A small cat"

print("逐字符查表：")
for ch in sample:
    print(f"  {ch!r:>5} -> {stoi[ch]:2d}")

print("结果：", [stoi[ch] for ch in sample])

# 所以 [4, 1, 24, 19, 8, 18, 18, 1, 10, 8, 25] 读作：
#   'A'->4  ' '->1  's'->24  'm'->19  'a'->8  'l'->18
#   'l'->18 ' '->1  'c'->10  'a'->8   't'->25
