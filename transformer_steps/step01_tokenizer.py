"""Step 1: 分词器 —— 把文本变成整数 ID。

Transformer 只认识数字，不认识字符。
这一步建立「词表」：语料里出现过的每个字符 -> 一个整数 ID。

真实模型（GPT/Llama）用 BPE/Unigram 分词器，词表约 3 万~15 万，
把常见词打包成一个 token。思想完全一样，只是切分粒度不同。
这里用字符级，因为词表小、不用外部依赖、便于观察。
"""

CORPUS = """\
Spring arrives, and the wind moves through the valley.
A small cat sits by the window and watches the clouds.
The sun goes down, the sky turns gold, and the birds return.
Night comes, the moon rises, and the stars shine above.
The cat closes its eyes and listens to the leaves.
A new day begins, and light fills the room again.
Spring arrives, and the wind moves through the valley.
A small cat sits by the window and watches the clouds.
"""


class CharTokenizer:
    def __init__(self, text: str) -> None:
        # set 去重，sorted 保证每次运行 ID 分配一致（可复现）
        chars = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # string -> int
        self.itos = {i: ch for ch, i in self.stoi.items()}  # int -> string
        self.vocab_size = len(chars)


    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


if __name__ == "__main__":
    tok = CharTokenizer(CORPUS)
    sample = "A small cat"

    print("vocab_size =", tok.vocab_size)
    print("vocab      =", "".join(sorted(tok.stoi)))
    print()
    print("text   ->", repr(sample))
    print("ids    ->", tok.encode(sample))
    print("decode ->", repr(tok.decode(tok.encode(sample))))

    assert tok.decode(tok.encode(CORPUS)) == CORPUS
    print("\n[OK] 编码 -> 解码 往返一致")

    # 思考题：为什么 vocab_size 是模型的超参数，必须和分词器一致？
    # ids    -> [4, 1, 24, 19, 8, 18, 18, 1, 10, 8, 25] 如何理解这个过程
    #   每个字符去 stoi 查表：'A'->4, ' '->1, 's'->24, 'm'->19, 'a'->8,
    #   'l'->18, 'l'->18, ' '->1, 'c'->10, 'a'->8, 't'->25。
    #   词表来自 sorted(set(CORPUS))，按 ASCII 码排序，所以换行符是 0。
