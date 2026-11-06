"""toy_bpe.py —— 一份「读得懂」的 byte-level BPE 教学实现（CS336 A1 §2.4–2.6）。

这不是 A1 的提交答案：
  * API 故意和 handout 的 `adapters.run_train_bpe` / `adapters.get_tokenizer` 不同
    （`train_bpe(corpus: str, ...)` 直接吃文本、不读文件），避免你直接复制粘贴交作业；
  * 计算换来了可读性：默认实现为了让 5MB 文本上 4096 词表的训练在几分钟内跑完，
    用了「反向索引 + 懒堆」的增量更新；同时也保留了一个每轮全量重数的 `naive` 版本
    用于对照（只在小语料上用）。

三个层次，建议按顺序读：
  1. `bytes_to_unicode` / `gpt2_pretokenize`：byte ↔ unicode 映射与 GPT-2 预分词；
  2. `train_bpe`（fast / naive 两条路径）：从语料得到 vocab 与 merges；
  3. `BPETokenizer`：给定 vocab+merges，做 encode / encode_iterable / decode / 保存加载。

关键语义（严格对齐 handout）：
  * 初始词表 = 256 个单字节 token，id 0..255；
  * special tokens 追加在 256 之后（id 256 起），按传入顺序；
  * `vocab_size` 是最终词表大小，包含 specials 与全部 merge 产物，
    所以 `len(merges) == vocab_size - 256 - len(special_tokens)`；
  * special token 是训练的硬边界：不参与 pair 统计，也不允许跨它合并；
  * pair 频次打平时取「字典序更大」的 pair（handout 用 `max(...)`）；
  * 编码时在一个 pre-token 内部按 merge rank 贪心合并，不允许跨 pre-token 合并；
  * 解码 = 字节拼接后 `decode("utf-8", errors="replace")`。
"""

from __future__ import annotations

import heapq
import json
import os
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import regex as re

# ---------------------------------------------------------------------------
# 0. GPT-2 的预分词正则
# ---------------------------------------------------------------------------
# 这是 handout 从 tiktoken PR #234 抄来的「prettier」版本。五条分支依次是：
#   1. 缩写后缀：'s 't 'd 'm 'l 'll 've 're（注意前面没有空格）
#   2. 可选一个空格 + 若干字母：把单词前的空格并进 token（GPT-2 的习惯）
#   3. 可选一个空格 + 若干数字
#   4. 可选一个空格 + 若干「既非空白也非字母数字」的符号（标点串）
#   5. 空白串：`\s+(?!\S)` 先尝试吃掉「后面没有非空白」的空白（即行尾空白），
#      失败再退回普通的 `\s+`
#
# 注意 regex 包用的是 Unicode 属性 \p{L} / \p{N}，所以中文、德文变音都按「字母」处理，
# 而 stdlib re 不支持 \p{...}，这就是为什么必须装 regex。
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_GPT2_PRE_RE = re.compile(PAT)


def gpt2_pretokenize(text: str) -> list[str]:
    """默认预分词器：返回文本被 GPT-2 正则切出来的所有片段（按出现顺序、完整覆盖）。"""
    return [m.group() for m in _GPT2_PRE_RE.finditer(text)]


def whitespace_pretokenize(text: str) -> list[str]:
    """handout 手算例子用的「按空白切」预分词器。

    handout 的 low/lower/widest/newest 例子明确假设 pre-tokenization 只是
    `text.split()`。如果改用 GPT-2 正则，前导空格会并进 token，pair 频次就变了，
    算出来的 12 条 merge 会和 handout 不一样 —— 所以 `train_bpe` 必须允许注入
    pretokenizer，这个函数就是给测试用的。
    """
    return text.split()


# ---------------------------------------------------------------------------
# 1. GPT-2 的 bytes ↔ unicode 可逆映射（用于 vocab.json / merges.txt 序列化）
# ---------------------------------------------------------------------------
# 为什么需要它：vocab 的键是「任意字节序列」，但 JSON 的键必须是合法字符串。
# 直接把 bytes.decode("utf-8") 会在遇到半个 emoji 时炸掉。GPT-2 的做法是把 256 个
# 字节值双射到一个「可打印」的 unicode 字符集：
#   * 188 个可打印字节（'!'..'~'、'¡'..'¬'、'®'..'ÿ'）原样映射；
#   * 其余 68 个（空白、控制字符等）映射到 chr(256+n)，从 'Ā' 开始往后排。
# 这样空格 ' ' 会变成 'Ġ'（U+0120），在 merges.txt 里肉眼可读。


@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """返回 {byte: 单字符 str} 的双射表（GPT-2 原版逻辑）。"""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


@lru_cache(maxsize=1)
def unicode_to_bytes() -> dict[str, int]:
    """`bytes_to_unicode()` 的逆映射：{单字符 str: byte}。"""
    return {c: b for b, c in bytes_to_unicode().items()}


def _bytes_to_gpt2_str(token: bytes) -> str:
    """把任意字节序列写成 GPT-2 的「可打印字符串」形式（序列化用）。"""
    table = bytes_to_unicode()
    return "".join(table[b] for b in token)


def _gpt2_str_to_bytes(text: str) -> bytes:
    """`_bytes_to_gpt2_str` 的逆操作（反序列化用）。"""
    table = unicode_to_bytes()
    return bytes(table[ch] for ch in text)


# ---------------------------------------------------------------------------
# 2. special token 的硬边界切分
# ---------------------------------------------------------------------------
# 训练时：special token 不参与 pair 统计 → 直接 re.split 掉它，只保留两侧普通文本。
# 编码时：special token 整体映射成一个固定 id → 同样先 split，命中就查表。
# 重叠的 special token（如 "<|endoftext|>" 与 "<|endoftext|><|endoftext|>"）
# 必须按「长度降序」放进 alternation，否则 regex 会先匹配到短的那个，
# 长 token 永远没有机会被匹配到。


def build_special_regex(special_tokens: Sequence[str]) -> "re.Pattern[str] | None":
    """把 special token 列表编译成一个带捕获组的 alternation（长 token 优先）。"""
    if not special_tokens:
        return None
    ordered = sorted(set(special_tokens), key=len, reverse=True)
    return re.compile("(" + "|".join(re.escape(tok) for tok in ordered) + ")")


def split_on_specials(
    text: str, special_regex: "re.Pattern[str] | None"
) -> Iterator[tuple[bool, str]]:
    """把 text 切成 (is_special, 片段) 流。

    `re.split` 带捕获组时，结果按「普通文本, 命中, 普通文本, 命中, ...」交替出现，
    所以奇数下标就是 special token。这个函数是训练（忽略 special 段）和编码
    （special 段直接出 id）共用的切分原语。
    """
    if special_regex is None:
        yield False, text
        return
    for idx, segment in enumerate(special_regex.split(text)):
        if idx % 2 == 1:
            yield True, segment
        elif segment:
            yield False, segment


# ---------------------------------------------------------------------------
# 3. 训练：先统计 unique pre-token 频次，再迭代 merge
# ---------------------------------------------------------------------------


def _bytes_of_pretoken(piece: str) -> tuple[bytes, ...]:
    """把一个 pre-token 字符串变成「单字节 token 的 tuple」。

    例："low" -> (b'l', b'o', b'w')。注意 Python 没有 byte 类型，单个字节也是
    `bytes` 对象，所以这里每个元素都是长度 1 的 bytes。
    """
    return tuple(bytes([b]) for b in piece.encode("utf-8"))


def count_pretokens(
    corpus: str,
    special_tokens: Sequence[str] = (),
    pretokenizer: Callable[[str], Iterable[str]] = gpt2_pretokenize,
) -> Counter[tuple[bytes, ...]]:
    """第一次（也是唯一一次）全量扫语料：得到 `Counter[tuple[bytes, ...] -> 频次]`。

    为什么要先做 pre-tokenization：
      * 加速：同一个词出现 1 万次只统计一次，pair 计数乘以频次即可；
      * 语义：不允许跨 pre-token 边界做 merge（否则 "dog!" 和 "dog." 会各成一个 token）。
    实现上 special token 先切掉（硬边界），再对剩下的普通文本用 regex finditer 切片。
    """
    special_regex = build_special_regex(special_tokens)
    counts: Counter[tuple[bytes, ...]] = Counter()
    for is_special, segment in split_on_specials(corpus, special_regex):
        if is_special:
            continue  # special token 不参与统计
        for piece in pretokenizer(segment):
            if piece:
                counts[_bytes_of_pretoken(piece)] += 1
    return counts


def _merge_pair_in_word(
    word: tuple[bytes, ...], pair: tuple[bytes, bytes]
) -> tuple[bytes, ...]:
    """在一个 pre-token 内部，把**所有**不重叠的 pair 出现合并成 `a+b`。

    例：word=(l,o,w), pair=(l,o) -> (lo, w)。
    一个 pre-token 里同一个 pair 可能出现多次（如 "aaaa" 里的 (a,a)），
    按从左到右、不重叠的规则全部合并。
    """
    a, b = pair
    out: list[bytes] = []
    i, n = 0, len(word)
    while i < n:
        if i + 1 < n and word[i] == a and word[i + 1] == b:
            out.append(a + b)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


class _GreaterPair:
    """给 heapq 用的包装类：让「字典序更大的 pair」排得更靠前（即更小）。

    Python 的 heapq 是 min-heap，而 handout 要求频次相同时取 lexicographically
    greater 的 pair。把比较反过来写，就能让堆顶正好是我们想要的 pair。
    """

    __slots__ = ("pair",)

    def __init__(self, pair: tuple[bytes, bytes]) -> None:
        self.pair = pair

    def __lt__(self, other: "_GreaterPair") -> bool:  # noqa: D105
        return self.pair > other.pair


def _pick_best_pair(
    pair_counts: Counter[tuple[bytes, bytes]],
) -> tuple[bytes, bytes] | None:
    """朴素版「选最高频 pair」：`max` 同时按 (频次, pair) 比较，天然实现 tie-break。

    这份实现是 handout 语义的直译，`naive` 训练路径和单元测试都用它。
    """
    if not pair_counts:
        return None
    return max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _pop_best_pair_from_heap(
    heap: list[tuple[int, _GreaterPair]],
    pair_counts: Counter[tuple[bytes, bytes]],
) -> tuple[bytes, bytes] | None:
    """从懒删除堆里取当前最高频 pair。

    堆里会有过期条目（pair 频次后来变了），所以 pop 出来后要用 `pair_counts`
    校验一次；不匹配就丢掉继续 pop。这是「惰性删除」的经典写法。
    """
    while heap:
        neg_count, wrapper = heapq.heappop(heap)
        pair = wrapper.pair
        if pair_counts.get(pair, 0) == -neg_count:
            return pair
    return None


def _train_merges_naive(
    word_counts: Counter[tuple[bytes, ...]], num_merges: int
) -> list[tuple[bytes, bytes]]:
    """教学版：每一轮都全量重数 pair。O(num_merges × 语料) —— 只适合小语料。"""
    word_counts = Counter(word_counts)
    merges: list[tuple[bytes, bytes]] = []
    for _ in range(num_merges):
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        for word, freq in word_counts.items():
            for a, b in zip(word, word[1:]):
                pair_counts[(a, b)] += freq
        best = _pick_best_pair(pair_counts)
        if best is None:
            break
        merges.append(best)
        # 全量重写所有 pre-token（内存换简单）
        new_counts: Counter[tuple[bytes, ...]] = Counter()
        for word, freq in word_counts.items():
            new_counts[_merge_pair_in_word(word, best)] += freq
        word_counts = new_counts
    return merges


def _train_merges_fast(
    word_counts: Counter[tuple[bytes, ...]], num_merges: int
) -> list[tuple[bytes, bytes]]:
    """增量版：反向索引（pair -> 含它的 pre-token 集合）+ 懒删除堆选最大频次。

    为什么能快：一次 merge 只会改变「包含该 pair 的那些 pre-token」内部的 pair 计数，
    其它 pre-token 完全不受影响。所以每轮只需要：
      1. 取出 pair 对应的 pre-token 集合；
      2. 对每个受影响的 pre-token：减掉旧 pair 计数 → 合并 → 加上新 pair 计数；
      3. 把变化过的 pair 新频次 push 进堆。
    总工作量大致正比于「merge 实际影响的 token 数」，而不是「每轮 × 全语料」。
    """
    # 初始化：pair 频次 + 反向索引
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    for word, freq in word_counts.items():
        for a, b in zip(word, word[1:]):
            pair = (a, b)
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)

    # 懒删除堆：每条 = (-频次, 反过来比较的 pair)
    heap: list[tuple[int, _GreaterPair]] = [
        (-count, _GreaterPair(pair)) for pair, count in pair_counts.items()
    ]
    heapq.heapify(heap)

    merges: list[tuple[bytes, bytes]] = []
    for _ in range(num_merges):
        best = _pop_best_pair_from_heap(heap, pair_counts)
        if best is None:
            break
        merges.append(best)
        merged_token = best[0] + best[1]

        # 注意：这里用 pop 把整个集合取走，之后处理 new_word 时会重新建立索引
        affected = list(pair_to_words.pop(best, ()))

        for word in affected:
            freq = word_counts.pop(word)

            # (a) 先把旧 word 贡献的 pair 计数减掉，并把它从各 pair 的反向索引里删掉
            for a, b in zip(word, word[1:]):
                pair = (a, b)
                pair_counts[pair] -= freq
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]  # 计数归零就从表里删掉，省内存
                else:
                    heapq.heappush(heap, (-pair_counts[pair], _GreaterPair(pair)))
                bucket = pair_to_words.get(pair)
                if bucket is not None:
                    bucket.discard(word)
                    if not bucket:
                        del pair_to_words[pair]

            # (b) 在 word 内部合并，得到 new_word；把它的频次并进 word_counts
            new_word = _merge_pair_in_word(word, best)
            word_counts[new_word] = word_counts.get(new_word, 0) + freq

            # (c) 把 new_word 的 pair 计数加回去，并反向索引它
            for a, b in zip(new_word, new_word[1:]):
                pair = (a, b)
                pair_counts[pair] += freq
                pair_to_words[pair].add(new_word)
                heapq.heappush(heap, (-pair_counts[pair], _GreaterPair(pair)))

    return merges


def train_bpe(
    corpus: str,
    vocab_size: int,
    special_tokens: Sequence[str] | None = None,
    *,
    pretokenizer: Callable[[str], Iterable[str]] = gpt2_pretokenize,
    naive: bool = False,
    verbose: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """在**一段文本**上训练 byte-level BPE。

    与 A1 `adapters.run_train_bpe(input_path=..., ...)` 有意不同：这里直接吃字符串，
    文件读取交给调用方（`train_bpe_from_file` 是便捷包装），这样 API 一眼能看出
    「训练语料 = 一段文本」，也方便在 notebook 里做小实验。

    Args:
        corpus: 训练语料（原始文本，可以包含 special tokens）。
        vocab_size: 最终词表大小，**包含** 256 个字节 + special tokens + merge 产物。
        special_tokens: 特殊字符串列表。训练时是硬边界，不参与 pair 统计。
        pretokenizer: 预分词函数。默认 GPT-2 正则；handout 手算例子请传
            `whitespace_pretokenize`。
        naive: True 时走「每轮全量重数」的教学实现（只用于小语料对比）。
        verbose: True 时打印 merge 进度。

    Returns:
        (vocab, merges)
        vocab: {token_id: bytes}，id 0..255 是单字节，接着是 specials，最后是 merge 产物；
        merges: [(bytes, bytes), ...]，按创建顺序排列。
    """
    specials = list(dict.fromkeys(special_tokens or []))  # 去重但保持顺序
    num_merges = vocab_size - 256 - len(specials)
    if num_merges < 0:
        raise ValueError(
            f"vocab_size={vocab_size} 小于 256+len(special_tokens)={256 + len(specials)}，"
            "没有空间做任何 merge"
        )

    t0 = time.time()
    word_counts = count_pretokens(corpus, specials, pretokenizer)
    if verbose:
        print(
            f"[train_bpe] 语料 {len(corpus):,} 字符 / {len(corpus.encode()):,} 字节，"
            f"unique pre-token {len(word_counts):,} 个，耗时 {time.time() - t0:.2f}s"
        )

    trainer = _train_merges_naive if naive else _train_merges_fast
    merges = trainer(word_counts, num_merges)
    if verbose:
        print(
            f"[train_bpe] {len(merges)} 次 merge，总耗时 {time.time() - t0:.2f}s"
            f"（{'naive' if naive else 'fast'} 实现）"
        )
    return _assemble_vocab(specials, merges), merges


# ---------------------------------------------------------------------------
# 3b. 大文件路径：按 special token 边界流式切块，不把整个文件读进内存
# ---------------------------------------------------------------------------

DEFAULT_FILE_CHUNK_BYTES = 8 << 20


def _assemble_vocab(
    specials: Sequence[str], merges: Sequence[tuple[bytes, bytes]]
) -> dict[int, bytes]:
    """按「256 单字节 → specials → merge 产物」的顺序组装词表。"""
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in specials:
        vocab[len(vocab)] = token.encode("utf-8")
    for a, b in merges:
        vocab[len(vocab)] = a + b
    return vocab


def _num_merges_or_raise(vocab_size: int, specials: Sequence[str]) -> int:
    num_merges = vocab_size - 256 - len(specials)
    if num_merges < 0:
        raise ValueError(
            f"vocab_size={vocab_size} 小于 256+len(special_tokens)={256 + len(specials)}，"
            "没有空间做任何 merge"
        )
    return num_merges


def _utf8_forward_safe(f, pos: int, size: int) -> int:
    """把从 pos 开始的位置向前挪到 UTF-8 字符边界（最多 3 字节）。"""
    f.seek(pos)
    while pos < size:
        b = f.read(1)
        if not b or (b[0] & 0xC0) != 0x80:  # 非续字节（或 EOF）就是边界
            return pos
        pos += 1
    return pos


def _find_chunk_end(
    f, start: int, size: int, target: int, needles: Sequence[bytes]
) -> int:
    """从 target 向后找「最近的安全切点」：special token 起点或换行之后。

    为什么只认这两类切点：
      * special token 是训练时的硬边界，切在它前面绝对安全；
      * 换行属于空白 run，GPT-2 正则的空白/词边界语义在换行处可独立匹配，
        切在换行之后不会改变任何 pre-token 的切分。
    不能切在「空格之后」：`"abc def"` 里的空格属于后一个 pre-token `" def"`，
    切开会把 `" def"` 变成 `" "` + `"def"`，计数和 merges 都会变。
    """
    max_scan = 8 << 20
    pos = target
    buf = b""
    f.seek(pos)  # 调用方可能刚做过别的 seek，这里必须回到 target
    while pos < size and pos < target + max_scan:
        block = f.read(min(1 << 20, size - pos))
        if not block:
            break
        data = buf + block
        base = pos - len(buf)
        best = None
        for needle in needles:
            i = data.find(needle)
            if i != -1:
                cand = base + i
                if best is None or cand < best:
                    best = cand
        nl = data.find(b"\n")
        if nl != -1:
            cand = base + nl + 1  # 切在换行之后
            if best is None or cand < best:
                best = cand
        if best is not None and best > start:
            return best
        keep = max((len(n) for n in needles), default=1) - 1
        buf = data[-keep:] if keep > 0 else b""
        pos += len(block)

    if pos >= size:
        # 已到文件末尾：剩下的文本整体作为最后一块。EOF 本身就是安全边界，
        # 不存在“后面的字符还会改变这块的预分词”的问题。
        return size

    # 没有 special/换行（超大单行）：退到 UTF-8 字符边界，并承认可能改变切分
    end = _utf8_forward_safe(f, target, size)
    return end if end > start else min(target, size)


def _special_span_containing(
    f, start: int, target: int, needles: Sequence[bytes]
) -> tuple[int, int] | None:
    """如果 target 落在某个 special token 内部，返回该 token 的 [begin, end)。

    否则返回 None。chunk 边界绝不能切在 special token 中间，否则会把
    `<|endoftext|>` 拆成 `<|endof` / `text|>`，计数和 merges 都会被污染。
    """
    max_len = max((len(n) for n in needles), default=0)
    for needle in needles:
        for off in range(max_len):
            begin = target - off
            if begin < start or begin < 0:
                break
            f.seek(begin)
            if f.read(len(needle)) == needle:
                if begin <= target < begin + len(needle):
                    return begin, begin + len(needle)
                break
    return None


def iter_file_chunks(
    input_path: str | os.PathLike[str],
    special_tokens: Sequence[str] = (),
    chunk_bytes: int = DEFAULT_FILE_CHUNK_BYTES,
) -> Iterator[str]:
    """按「特殊 token 边界 / 行边界」把文件切成文本块，内存只占一个 chunk。

    这是 A1 大文件（TinyStories ~2GB、OWT ~12GB）与 toy 版本的关键差别：
    `train_bpe_from_file` 绝不能 `Path.read_text()` 把 12GB 读进内存。
    """
    path = Path(input_path)
    size = path.stat().st_size
    if size == 0:
        return
    needles = [
        s.encode("utf-8")
        for s in sorted(set(special_tokens), key=len, reverse=True)
        if s
    ]
    with path.open("rb") as f:
        start = 0
        while start < size:
            target = min(start + chunk_bytes, size)
            if target >= size:
                end = size
            else:
                span = _special_span_containing(f, start, target, needles)
                end = span[1] if span is not None else _find_chunk_end(f, start, size, target, needles)
            f.seek(start)
            raw = f.read(end - start)
            yield raw.decode("utf-8")
            start = end


def count_pretokens_from_file(
    input_path: str | os.PathLike[str],
    special_tokens: Sequence[str] = (),
    pretokenizer: Callable[[str], Iterable[str]] = gpt2_pretokenize,
    chunk_bytes: int = DEFAULT_FILE_CHUNK_BYTES,
) -> Counter[tuple[bytes, ...]]:
    """流式版 `count_pretokens`：分块统计后把局部 Counter 相加。

    分块边界都在 special / 空白 / UTF-8 字符边界上，所以结果与整段读入一致
    （`test_file_streaming_matches_in_memory` 会逐条比对 merges）。
    """
    counts: Counter[tuple[bytes, ...]] = Counter()
    for chunk in iter_file_chunks(input_path, special_tokens, chunk_bytes):
        counts.update(count_pretokens(chunk, special_tokens, pretokenizer))
    return counts


def train_bpe_from_file(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: Sequence[str] | None = None,
    *,
    pretokenizer: Callable[[str], Iterable[str]] = gpt2_pretokenize,
    naive: bool = False,
    chunk_bytes: int = DEFAULT_FILE_CHUNK_BYTES,
    verbose: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """文件版训练入口：流式预分词，不把整个文件读进内存。

    与 `train_bpe` 的唯一区别是「怎么数 pre-token」；合并阶段完全相同，
    所以两者在同一语料上必须产出完全一致的 merges（测试会验证）。
    """
    specials = list(dict.fromkeys(special_tokens or []))
    num_merges = _num_merges_or_raise(vocab_size, specials)
    t0 = time.time()
    word_counts = count_pretokens_from_file(
        input_path, specials, pretokenizer, chunk_bytes
    )
    if verbose:
        print(
            f"[train_bpe_from_file] 流式预分词完成：unique pre-token {len(word_counts):,} 个，"
            f"chunk_bytes={chunk_bytes:,}，耗时 {time.time() - t0:.2f}s"
        )
    trainer = _train_merges_naive if naive else _train_merges_fast
    merges = trainer(word_counts, num_merges)
    if verbose:
        print(f"[train_bpe_from_file] {len(merges)} 次 merge，总耗时 {time.time() - t0:.2f}s")
    return _assemble_vocab(specials, merges), merges


# ---------------------------------------------------------------------------
# 4. Tokenizer：编码 / 流式编码 / 解码 / 保存加载
# ---------------------------------------------------------------------------

_REPLACEMENT_CHAR = "\ufffd"


class BPETokenizer:
    """byte-level BPE 分词器： vocab + merges -> encode/decode。

    与 handout 推荐的接口名字一致（encode / encode_iterable / decode / from_files），
    但构造函数多了一个 `pretokenizer` 参数，便于做对照实验。
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: Iterable[tuple[bytes, bytes]],
        special_tokens: Sequence[str] | None = None,
        pretokenizer: Callable[[str], Iterable[str]] = gpt2_pretokenize,
    ) -> None:
        # 复制一份，避免外部改 dict 影响 tokenizer
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)
        # merge rank：越早创建的 merge 越优先（rank 越小）
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }
        self.special_tokens: list[str] = list(dict.fromkeys(special_tokens or []))
        self.pretokenizer = pretokenizer

        # bytes -> id 的反查表（同一 bytes 有多个 id 时取最小 id）
        self._token_to_id: dict[bytes, int] = {}
        for token_id in sorted(self.vocab):
            self._token_to_id.setdefault(self.vocab[token_id], token_id)

        # 按 handout 要求：special token 若不在词表里，追加到词表末尾
        for token in self.special_tokens:
            raw = token.encode("utf-8")
            if raw not in self._token_to_id:
                new_id = max(self.vocab) + 1 if self.vocab else 0
                self.vocab[new_id] = raw
                self._token_to_id[raw] = new_id

        self._special_to_id = {
            token: self._token_to_id[token.encode("utf-8")] for token in self.special_tokens
        }
        self._special_regex = build_special_regex(self.special_tokens)
        # 长 special token 优先（用于流式编码时判断「尾部是不是某个 special 的前缀」）
        self._specials_by_len = sorted(self.special_tokens, key=len, reverse=True)

        # pre-token -> id tuple 的缓存。真实语料重复 pre-token 极多，缓存能显著提速；
        # 设一个上限避免流式编码时内存无界增长。
        self._pretoken_cache: dict[str, tuple[int, ...]] = {}
        self._pretoken_cache_limit = 50_000

    # -- 基本属性 ---------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def __len__(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:  # pragma: no cover - 教学用途
        return (
            f"BPETokenizer(vocab_size={len(self.vocab)}, merges={len(self.merges)}, "
            f"special_tokens={self.special_tokens!r})"
        )

    # -- 序列化 -----------------------------------------------------------
    def save(self, dirpath: str | os.PathLike[str]) -> tuple[Path, Path]:
        """保存成 GPT-2 格式的 `vocab.json` + `merges.txt`（可以用 `load` 读回）。

        vocab.json: {"<GPT-2 可打印字符串>": token_id}
        merges.txt: 每行 "tokenA tokenB"，第一行是 "#version: 0.2"（GPT-2 惯例）
        """
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        vocab_path = dirpath / "vocab.json"
        merges_path = dirpath / "merges.txt"

        vocab_json = {
            _bytes_to_gpt2_str(token): token_id
            for token_id, token in sorted(self.vocab.items())
        }
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab_json, f, ensure_ascii=False)

        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")
            for a, b in self.merges:
                f.write(f"{_bytes_to_gpt2_str(a)} {_bytes_to_gpt2_str(b)}\n")
        return vocab_path, merges_path

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike[str],
        merges_filepath: str | os.PathLike[str],
        special_tokens: Sequence[str] | None = None,
        pretokenizer: Callable[[str], Iterable[str]] = gpt2_pretokenize,
    ) -> "BPETokenizer":
        """从 GPT-2 格式的 vocab.json + merges.txt 构造 tokenizer。"""
        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {
            int(token_id): _gpt2_str_to_bytes(token_str)
            for token_str, token_id in raw_vocab.items()
        }

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#version"):
                    continue
                parts = line.split(" ")
                if len(parts) != 2:
                    raise ValueError(f"merges.txt 格式错误：{line!r}")
                merges.append((_gpt2_str_to_bytes(parts[0]), _gpt2_str_to_bytes(parts[1])))
        return cls(vocab, merges, special_tokens, pretokenizer)

    @classmethod
    def load(
        cls,
        dirpath: str | os.PathLike[str],
        special_tokens: Sequence[str] | None = None,
        pretokenizer: Callable[[str], Iterable[str]] = gpt2_pretokenize,
    ) -> "BPETokenizer":
        """`save` 的逆操作：从目录读回 vocab.json + merges.txt。"""
        dirpath = Path(dirpath)
        return cls.from_files(
            dirpath / "vocab.json",
            dirpath / "merges.txt",
            special_tokens=special_tokens,
            pretokenizer=pretokenizer,
        )

    # -- 编码 -------------------------------------------------------------
    def _encode_pretoken(self, piece: str) -> list[int]:
        """在一个 pre-token 内部做 BPE 编码（贪心按 merge rank）。

        注意 handout 特别强调的一点：不是「按 merge 列表顺序逐条重放」，而是
        **每次挑当前相邻 pair 中 merge rank 最小的那个**来合并。如果简单地按
        merges 顺序逐条扫描，遇到 (a, b, c) 这种词、merges 里先有 (b, c) 后有
        (a, b) 时就会合错（会得到 a + bc，而正确结果取决于真实训练出的 rank 顺序）。
        这里每一步都重新找 rank 最小的 pair，并把该 pair 的所有不重叠出现都合并掉。
        """
        tokens = list(_bytes_of_pretoken(piece))
        while len(tokens) > 1:
            best_rank: int | None = None
            best_idx = -1
            for i in range(len(tokens) - 1):
                rank = self.merge_ranks.get((tokens[i], tokens[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_idx < 0:
                break  # 没有可合并的 pair 了
            pair = (tokens[best_idx], tokens[best_idx + 1])
            tokens = list(_merge_pair_in_word(tuple(tokens), pair))

        ids: list[int] = []
        for token in tokens:
            token_id = self._token_to_id.get(token)
            if token_id is None:  # 理论上不会发生：vocab 覆盖所有字节 + 所有 merge 产物
                raise KeyError(f"token {token!r} 不在词表中，vocab 与 merges 不匹配")
            ids.append(token_id)
        return ids

    def _partition(self, text: str) -> list[tuple[bool, str]]:
        """把文本切成「编码单元」列表：[(is_special, 片段), ...]。

        普通片段是一个 pre-token，special 片段是完整 special token。
        这一步是编码的唯一入口：`encode` 用它，流式编码也用它，保证两者切法一致。
        """
        if not text:
            return []
        if self._special_regex is None:
            return [(False, piece) for piece in self.pretokenizer(text) if piece]
        items: list[tuple[bool, str]] = []
        for is_special, segment in split_on_specials(text, self._special_regex):
            if is_special:
                items.append((True, segment))
            else:
                items.extend((False, piece) for piece in self.pretokenizer(segment) if piece)
        return items

    def _encode_items(self, items: Iterable[tuple[bool, str]]) -> list[int]:
        """把 `_partition` 的输出逐项变成 id：special 查表，pre-token 走 BPE（带缓存）。"""
        out: list[int] = []
        cache = self._pretoken_cache
        if len(cache) > self._pretoken_cache_limit:
            cache.clear()
        for is_special, text in items:
            if is_special:
                out.append(self._special_to_id[text])
                continue
            cached = cache.get(text)
            if cached is None:
                cached = tuple(self._encode_pretoken(text))
                cache[text] = cached
            out.extend(cached)
        return out

    def encode(self, text: str) -> list[int]:
        """把一段文本编码成 token id 列表。"""
        return self._encode_items(self._partition(text))

    # -- 流式编码 ---------------------------------------------------------
    @staticmethod
    def _safe_flush_point(text: str) -> int:
        """返回一个「可以安全 flush 的前缀长度」：最后一个词连同它前面的空白 run 留在 buffer。

        规则（和 A1 `test_encode_iterable_tinystories_matches_tiktoken` 的语义一致）：
          1. 跳过 buffer 末尾的空白，找到最后一个「词」（连续非空白字符）；
          2. 再往前跳过紧挨着它的空白 run，得到 flush 点 k；
          3. flush `text[:k]`，保留 `text[k:]`（空白 run + 最后一个词 + 尾部空白）。

        为什么这样切是安全的：flush 出的前缀以非空白字符结尾，它的最后一个 pre-token
        在整段文本里也恰好结束于同一个位置（因为任何 pre-token 都不会以空白结尾），
        所以「单独预分词这个前缀」和「在整段文本里预分词到同一位置」结果相同。
        由此同时避免了两个坑：
          * 空白 run 跨块（"a\\n\\nb" 在 buffer="a\\n\\n" 时会切出 '\\n\\n'，但整段
            文本里是 '\\n','\\n'）；
          * 撇号跨块（buffer="I'l" 时会把 "'" 当成标点 token，但整段文本里 "'ll"
            是一个 token —— 注意这类不稳定可以影响到倒数第二个 pre-token）。
        """
        i = len(text) - 1
        while i >= 0 and text[i].isspace():
            i -= 1
        if i < 0:
            return 0  # 全是空白：整段都是「尾部空白 run」，留到下一块
        j = i
        while j > 0 and not text[j - 1].isspace():
            j -= 1
        if j == 0:
            return 0  # 最后一个词从头开始（前面没有空白）：没有可 flush 的前缀
        k = j
        while k > 0 and text[k - 1].isspace():
            k -= 1
        return k

    def _split_safe_prefix(
        self, buffer: str, final: bool
    ) -> tuple[list[tuple[bool, str]], str]:
        """把 buffer 切成「可以立刻安全编码的编码单元」+「必须留给下一块的后缀」。

        为什么不能「逐 chunk 独立 encode 再拼接」：chunk 边界可能落在 pre-token 中间，
        或者落在空白 run 中间，或者落在撇号缩写中间（见 `_safe_flush_point` 的说明）。

        这里先把「尾部是某个 special token 的真前缀」的部分留出来（长 special 优先），
        这样既不会把半个 special token 当普通文本编码，也不会错过更长的重叠 special；
        然后按 `_safe_flush_point` 决定 flush 到哪里。

        Args:
            buffer: 当前累计的文本。
            final: True 表示输入已经结束，此时不需要再留尾巴。

        Returns:
            (items, leftover)：items 是编码单元列表，leftover 是留给下一块的文本。
        """
        if final:
            return self._partition(buffer), ""
        if not buffer:
            return [], ""

        # (1) 先留下「尾部是某个 special token 的真前缀」的部分（长 special 优先）
        hold = ""
        stripped = buffer
        max_hold = 0
        for token in self._specials_by_len:
            upper = min(len(token) - 1, len(stripped))
            for k in range(upper, max_hold, -1):
                if stripped.endswith(token[:k]):
                    max_hold = k
                    break
        if max_hold:
            hold = stripped[-max_hold:]
            stripped = stripped[:-max_hold]

        # (2) 只 flush 到最后一个词之前的空白 run 起点
        cut = self._safe_flush_point(stripped)
        if cut <= 0:
            return [], stripped + hold
        return self._partition(stripped[:cut]), stripped[cut:] + hold

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """惰性编码一个字符串可迭代对象（例如文件句柄），逐 token yield。

        内存复杂度：O(chunk 大小 + 最长 pre-token + 最长 special token)，
        而不是 O(全文) —— 每次只把「确定不会再变」的编码单元送进 `_encode_items`。
        """
        buffer = ""
        for chunk in iterable:
            buffer += chunk
            items, buffer = self._split_safe_prefix(buffer, final=False)
            if items:
                for token_id in self._encode_items(items):
                    yield token_id
        if buffer:
            for token_id in self._encode_items(self._partition(buffer)):
                yield token_id

    # -- 解码 -------------------------------------------------------------
    def decode(self, ids: Iterable[int]) -> str:
        """把 token id 序列解码回文本。

        handout 要求：输入 id 不保证对应合法 Unicode，遇到坏字节用 U+FFFD 替换。
        这里对「id 不在词表里」也做同样的容错（输出一个 U+FFFD），
        而不是直接 KeyError，方便交互式实验时随便敲 id。
        """
        chunks: list[bytes] = []
        for token_id in ids:
            raw = self.vocab.get(int(token_id))
            if raw is None:
                chunks.append(_REPLACEMENT_CHAR.encode("utf-8"))
            else:
                chunks.append(raw)
        return b"".join(chunks).decode("utf-8", errors="replace")


# 模块级便捷函数：不想记方法名时用这两个
def save_tokenizer(tokenizer: BPETokenizer, dirpath: str | os.PathLike[str]) -> None:
    """`BPETokenizer.save` 的函数版。"""
    tokenizer.save(dirpath)


def load_tokenizer(
    dirpath: str | os.PathLike[str],
    special_tokens: Sequence[str] | None = None,
) -> BPETokenizer:
    """`BPETokenizer.load` 的函数版。"""
    return BPETokenizer.load(dirpath, special_tokens=special_tokens)


# ---------------------------------------------------------------------------
# 5. 自检：直接运行本文件会复现 handout 的 low/lower/widest/newest 例子
# ---------------------------------------------------------------------------
#   .venv/bin/python bpe_lab/toy_bpe.py
# 这只是给人看的；正式断言在 bpe_lab/test_toy_bpe.py。
if __name__ == "__main__":
    example_corpus = "low low low low low\nlower lower widest widest widest\nnewest newest newest newest newest newest"
    expected = [
        ("s", "t"),
        ("e", "st"),
        ("o", "w"),
        ("l", "ow"),
        ("w", "est"),
        ("n", "e"),
        ("ne", "west"),
        ("w", "i"),
        ("wi", "d"),
        ("wid", "est"),
        ("low", "e"),
        ("lowe", "r"),
    ]
    vocab, merges = train_bpe(
        example_corpus,
        vocab_size=256 + 1 + 12,
        special_tokens=["<|endoftext|>"],
        pretokenizer=whitespace_pretokenize,
        verbose=True,
    )
    print("merges:", [f"{a.decode()} {b.decode()}" for a, b in merges])
    assert merges == [(a.encode(), b.encode()) for a, b in expected], "与 handout 的 12 条 merge 不一致"
    print("vocab_size:", len(vocab), "（256 字节 + 1 special + 12 merges）")
    tok = BPETokenizer(vocab, merges, ["<|endoftext|>"], pretokenizer=whitespace_pretokenize)
    print("newest ->", [vocab[i].decode() for i in tok.encode("newest")])
    print("OK: 与 handout §2.4 例子一致")
