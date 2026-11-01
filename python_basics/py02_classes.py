"""类、继承、dataclass、装饰器 —— 看懂 nn.Module 之前的准备。

运行：
    .venv/bin/python python_basics/py02_classes.py
"""

from dataclasses import dataclass


# ============================================================
# 1. 类：把「数据 + 操作数据的函数」放在一起
# ============================================================
print("--- 1. 类 ---")


class Counter:
    def __init__(self, start: int = 0) -> None:
        self.value = start

    def add(self, n: int = 1) -> None:
        self.value += n

    def get(self) -> int:
        return self.value


c = Counter(10)
c.add()
c.add(5)
print(c.value)

# self 是什么？
#   c.add() 其实等价于 Counter.add(c)
#   Python 自动把「调用者」作为第一个参数传进去，名字约定叫 self。
#
# __init__ 是构造函数：写 Counter(10) 时自动调用。
# self.value 是「实例属性」，每个对象各有一份。


# 把 step01 的分词器压缩成一个最小版本：
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


tok = CharTokenizer("cat sat")
print(tok.vocab_size)
print(tok.encode("cat"))
print(tok.decode(tok.encode("cat sat")))


# ============================================================
# 2. 继承与 super()
# ============================================================
print("\n--- 2. 继承 ---")


class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self) -> str:
        return "..."


class Cat(Animal):
    def __init__(self, name, indoor):
        super().__init__(name)    # 先让父类初始化 name
        self.indoor = indoor

    def speak(self) -> str:       # 覆盖父类的方法
        return "meow"


kitty = Cat("mimi", True)
print(kitty.name, kitty.indoor, kitty.speak())
print(isinstance(kitty, Animal))    # True

# transformer 里：
#   class MultiHeadAttention(nn.Module):
#       def __init__(self, config):
#           super().__init__()      # 不调用它，PyTorch 就无法登记参数
# 这就是每个模型类开头都有 super().__init__() 的原因。


# ============================================================
# 3. dataclass：自动生成 __init__ / __repr__ / __eq__
# ============================================================
print("\n--- 3. dataclass ---")


@dataclass
class GPTConfig:
    vocab_size: int          # 没有默认值，创建时必须传
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    dropout: float = 0.0


cfg = GPTConfig(vocab_size=30)
print(cfg)                                # 自动生成的 __repr__
print(cfg.n_embd)
print(cfg == GPTConfig(vocab_size=30))    # 自动生成的 __eq__

# 规则：没有默认值的字段必须写在有默认值的字段前面。
# 否则 Python 不知道哪个参数该用默认值。
#
# @dataclass 是一个「装饰器」，下一节解释它到底是什么。


# ============================================================
# 4. 装饰器：在函数外面套一层
# ============================================================
print("\n--- 4. 装饰器 ---")


def log_call(fn):
    def wrapper(*args, **kwargs):
        print(f"  调用 {fn.__name__}")
        return fn(*args, **kwargs)
    return wrapper


@log_call
def add(a, b):
    return a + b


print(add(2, 3))

# @log_call 等价于：
#   add = log_call(add)
#
# *args   把多余的位置参数打包成 tuple
# **kwargs 把多余的关键字参数打包成 dict


class MiniGPT:
    @staticmethod
    def init_weights(module):
        print("  初始化", module)


MiniGPT.init_weights("linear")

# @staticmethod 让这个函数不需要 self，直接挂在类上。
# 出现在：step07 的 _init_weights。
#
# @torch.no_grad() 是同一个机制：给函数套一层「不记录梯度」的开关。
# 出现在：step08 的 generate。


# ============================================================
# 5. __call__：让对象像函数一样被调用
# ============================================================
print("\n--- 5. __call__ ---")


class Adder:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        return x + self.n


add3 = Adder(3)
print(add3(10))

# nn.Module 定义了自己的 __call__：
#   它先做内部记录（hooks），再调用你写的 forward。
#   所以模型写成 model(x)，而不是 model.forward(x)。


# ============================================================
# 6. 类型注解：越写越长的那些
# ============================================================
print("\n--- 6. 类型注解 ---")


def f(a: int, b: str = "x") -> None:
    ...


def g(xs: list[int]) -> dict[str, int]:
    ...


def h(x: int | None) -> tuple[int, str]:
    ...


# int | None   表示「int 或 None」
# list[int]    表示元素是 int 的 list
# tuple[A, B]  表示固定两个元素的元组
#
# 有些文件开头写 from __future__ import annotations，
# 是为了在旧版 Python 上也能直接写 list[int] 这种注解。

print(f.__annotations__)
print("（注解只是标签，运行时不会检查类型）")
