"""test_toy_bpe.py —— `toy_bpe` 的单元测试（双模式）。

两种跑法都支持：
    # 1) 直接跑（不需要 pytest；会打印 PASS/FAIL 汇总，失败时 exit 1）
    .venv/bin/python bpe_lab/test_toy_bpe.py

    # 2) pytest（如果临时环境里有）
    uv run --with pytest python -m pytest bpe_lab/test_toy_bpe.py -q

测试尽量对标 CS336 A1 handout 的语义与官方 tests/ 的检查点：
  * 手算例子 low/lower/widest/newest 的 12 条 merge 精确序列；
  * tie-break 取字典序更大的 pair；
  * §2.6.1 的 bpe_encoding 例子（"the cat ate" -> [9, 7, 1, 5, 10, 3]）；
  * 与官方 fixture `train-bpe-reference-merges.txt` 的对照（corpus.en, vocab=500）；
  * special token 是硬边界、整体保留、重叠时最长匹配；
  * UTF-8 往返（空串 / emoji / 中文 / 德文 / 长文本）；
  * 非法 id / 非法 UTF-8 解码成 U+FFFD；
  * `encode_iterable` 惰性且与整段 encode 完全一致（含跨块空白 run、撇号缩写）。
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

# 允许 `python bpe_lab/test_toy_bpe.py` 直接跑（此时 bpe_lab 不在 sys.path 上）
sys.path.insert(0, str(Path(__file__).resolve().parent))

import toy_bpe  # noqa: E402
from toy_bpe import (  # noqa: E402
    BPETokenizer,
    _pick_best_pair,
    gpt2_pretokenize,
    train_bpe,
    whitespace_pretokenize,
)

# ---------------------------------------------------------------------------
# 基础设置：fixtures 路径、pytest 兼容的 skip、结果缓存
# ---------------------------------------------------------------------------
FIXTURES = Path(
    os.environ.get(
        "BPE_LAB_FIXTURES",
        "/Users/yunni/Projects/assignment1-basics/tests/fixtures",
    )
)

try:  # pytest 可选：没有 pytest 时用自定义异常实现「跳过」
    import pytest as _pytest
except Exception:  # pragma: no cover - 取决于环境
    _pytest = None


class SkipTest(Exception):
    """直接运行时用来表示「这个测试跳过」。"""


def _skip(reason: str):
    if _pytest is not None:
        _pytest.skip(reason)
    raise SkipTest(reason)


_CACHE: dict[str, object] = {}


def _needs_fixtures(name: str) -> Path:
    path = FIXTURES / name
    if not path.exists():
        _skip(f"缺少 fixture：{path}")
    return path


def _fallback_corpus() -> str:
    """没有 fixtures 时的兜底语料（足够训练几百词表的 tokenizer）。"""
    seed = (
        "low low low low low lower lower widest widest widest "
        "newest newest newest newest newest newest "
        "the cat ate the dog ran away and the cat sat on the mat. "
        "<|endoftext|>"
        "Once upon a time there was a little girl named Lily. "
        "She had a red ball and a big dog. They played in the park. "
        "The dog ran fast and the ball rolled far away. "
        "Lily laughed and ran after it. <|endoftext|>"
    )
    return seed * 40


def _training_text(n_chars: int = 60_000) -> str:
    """训练用文本：优先用 TinyStories fixture 的一段，否则用兜底语料。"""
    path = FIXTURES / "tinystories_sample_5M.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")[:n_chars]
    corpus_path = FIXTURES / "corpus.en"
    if corpus_path.exists():
        return corpus_path.read_text(encoding="utf-8")[:n_chars]
    return _fallback_corpus()


def _small_tokenizer() -> BPETokenizer:
    """缓存一个在小型语料上训出来的 tokenizer，供多个测试复用（避免重复训练）。"""
    if "small_tok" not in _CACHE:
        text = _training_text(50_000)
        vocab, merges = train_bpe(text, 512, ["<|endoftext|>"])
        _CACHE["small_tok"] = BPETokenizer(vocab, merges, ["<|endoftext|>"])
    return _CACHE["small_tok"]  # type: ignore[return-value]


def _gpt2_tokenizer() -> BPETokenizer:
    """用 GPT-2 官方 vocab/merges fixture 构造 tokenizer（验证 from_files）。"""
    if "gpt2_tok" not in _CACHE:
        vocab_path = _needs_fixtures("gpt2_vocab.json")
        merges_path = _needs_fixtures("gpt2_merges.txt")
        _CACHE["gpt2_tok"] = BPETokenizer.from_files(
            vocab_path, merges_path, ["<|endoftext|>"]
        )
    return _CACHE["gpt2_tok"]  # type: ignore[return-value]


HANDOUT_MERGES_12 = [
    (b"s", b"t"),
    (b"e", b"st"),
    (b"o", b"w"),
    (b"l", b"ow"),
    (b"w", b"est"),
    (b"n", b"e"),
    (b"ne", b"west"),
    (b"w", b"i"),
    (b"wi", b"d"),
    (b"wid", b"est"),
    (b"low", b"e"),
    (b"lowe", b"r"),
]

HANDOUT_CORPUS = (
    "low low low low low\n"
    "lower lower widest widest widest\n"
    "newest newest newest newest newest newest"
)


# ---------------------------------------------------------------------------
# 1. handout §2.4 手算例子
# ---------------------------------------------------------------------------
def test_handout_12_merges_exact():
    """low/lower/widest/newest 例子：12 条 merge 必须与 handout 完全一致。

    注意：这个例子必须用「按空白切」的预分词器；用 GPT-2 正则会把前导空格并进
    token，pair 统计就变了。
    """
    vocab, merges = train_bpe(
        HANDOUT_CORPUS,
        vocab_size=256 + 1 + 12,
        special_tokens=["<|endoftext|>"],
        pretokenizer=whitespace_pretokenize,
    )
    assert merges == HANDOUT_MERGES_12, f"实际得到 {merges}"
    assert len(vocab) == 269, len(vocab)
    # id 0..255 是单字节，256 是 special，之后是 12 个 merge 产物
    assert all(vocab[i] == bytes([i]) for i in range(256))
    assert vocab[256] == b"<|endoftext|>"
    assert [vocab[i] for i in range(257, 269)] == [a + b for a, b in HANDOUT_MERGES_12]


def test_handout_6_merges_vocab_and_newest():
    """6 merges 时：新增 token 为 st/est/ow/low/west/ne，且 newest -> [ne, west]。"""
    vocab, merges = train_bpe(
        HANDOUT_CORPUS,
        vocab_size=256 + 1 + 6,
        special_tokens=["<|endoftext|>"],
        pretokenizer=whitespace_pretokenize,
    )
    assert merges == HANDOUT_MERGES_12[:6]
    assert len(vocab) == 263
    learned = {vocab[i] for i in range(257, 263)}
    assert learned == {b"st", b"est", b"ow", b"low", b"west", b"ne"}, learned

    tokenizer = BPETokenizer(
        vocab, merges, ["<|endoftext|>"], pretokenizer=whitespace_pretokenize
    )
    ids = tokenizer.encode("newest")
    assert ids == [tokenizer._token_to_id[b"ne"], tokenizer._token_to_id[b"west"]]
    assert tokenizer.decode(ids) == "newest"


def test_tie_break_helpers_and_training():
    """handout 的 tie-break 例子：四个 pair 同频时取 lexicographically greater (BA, A)。"""
    from collections import Counter

    pair_counts = Counter(
        {
            (b"A", b"B"): 1,
            (b"A", b"C"): 1,
            (b"B", b"ZZ"): 1,
            (b"BA", b"A"): 1,
        }
    )
    assert _pick_best_pair(pair_counts) == (b"BA", b"A")

    # 端到端：四个词各出现一次，所有 pair 频次都是 1，第一条 merge 应该是 (z, z)
    vocab, merges = train_bpe(
        "ab ac bzz baa",
        vocab_size=256 + 1 + 1,
        special_tokens=["<|endoftext|>"],
        pretokenizer=whitespace_pretokenize,
    )
    assert merges[0] == (b"z", b"z"), merges
    assert len(vocab) == 258


def test_naive_and_fast_agree():
    """naive（每轮全量重数）与 fast（增量更新）必须给出完全相同的 vocab/merges。"""
    text = _training_text(8_000)
    vocab_fast, merges_fast = train_bpe(text, 400, ["<|endoftext|>"])
    vocab_naive, merges_naive = train_bpe(
        text, 400, ["<|endoftext|>"], naive=True
    )
    assert merges_fast == merges_naive
    assert vocab_fast == vocab_naive


def test_vocab_size_too_small_raises():
    try:
        train_bpe("abc", vocab_size=256, special_tokens=["<|endoftext|>"])
    except ValueError:
        return
    raise AssertionError("vocab_size 太小却没有报错")


# ---------------------------------------------------------------------------
# 2. handout §2.6.1 的 bpe_encoding 例子
# ---------------------------------------------------------------------------
def test_handout_bpe_encoding_example():
    """给定 vocab/merges，"the cat ate" 必须编码成 [9, 7, 1, 5, 10, 3]。"""
    vocab = {
        0: b" ",
        1: b"a",
        2: b"c",
        3: b"e",
        4: b"h",
        5: b"t",
        6: b"th",
        7: b" c",
        8: b" a",
        9: b"the",
        10: b" at",
    }
    merges = [
        (b"t", b"h"),
        (b" ", b"c"),
        (b" ", b"a"),
        (b"th", b"e"),
        (b" a", b"t"),
    ]
    tokenizer = BPETokenizer(vocab, merges)
    assert tokenizer.encode("the cat ate") == [9, 7, 1, 5, 10, 3]
    assert tokenizer.decode([9, 7, 1, 5, 10, 3]) == "the cat ate"


# ---------------------------------------------------------------------------
# 3. 预分词正则
# ---------------------------------------------------------------------------
def test_gpt2_pretokenize_matches_handout_example():
    assert gpt2_pretokenize("some text that i'll pre-tokenize") == [
        "some",
        " text",
        " that",
        " i",
        "'ll",
        " pre",
        "-",
        "tokenize",
    ]


def test_pretokenizer_covers_text_exactly():
    """预分词必须完整覆盖文本（join 回去等于原文）——流式编码的正确性依赖这一点。"""
    for text in [
        "",
        "hello",
        "  leading and trailing   ",
        "a\n\nb   c\t d",
        "I'll we're he's they've I'd don't",
        "中文测试，BPE 很好玩！",
        "emoji 😀👍🏽 done",
        "'\"!?.,;:-_()[]{}",
    ]:
        pieces = gpt2_pretokenize(text)
        assert "".join(pieces) == text, (text, pieces)


# ---------------------------------------------------------------------------
# 4. UTF-8 往返（roundtrip）
# ---------------------------------------------------------------------------
def test_roundtrip_basic_strings():
    tok = _gpt2_tokenizer() if (FIXTURES / "gpt2_vocab.json").exists() else _small_tokenizer()
    texts = [
        "",
        "Hello, how are you?",
        "Hello, how are you?\n",
        "😀🎉👍🏽👨‍👩‍👧‍👦",
        "中文测试：字节级 BPE 可以无损往返。",
        "Grüße aus München — äöüß ÄÖÜ.",
        "  spaces   and\ttabs\n\nand newlines  ",
        "'.join hostile punctuation!?",
    ]
    german = FIXTURES / "german.txt"
    if german.exists():
        texts.append(german.read_text(encoding="utf-8"))
    stories = FIXTURES / "tinystories_sample.txt"
    if stories.exists():
        texts.append(stories.read_text(encoding="utf-8"))
    for text in texts:
        ids = tok.encode(text)
        assert tok.decode(ids) == text, f"roundtrip 失败: {text[:40]!r}"


def test_roundtrip_long_text():
    path = FIXTURES / "tinystories_sample_5M.txt"
    text = (
        path.read_text(encoding="utf-8")[:20_000]
        if path.exists()
        else _fallback_corpus()[:20_000]
    )
    tok = _small_tokenizer()
    assert tok.decode(tok.encode(text)) == text


def test_special_tokens_roundtrip_and_whole_token():
    """special token 编码时必须整体用一个 id，并且不参与 BPE 合并。"""
    tok = _gpt2_tokenizer()
    eos = tok._special_to_id["<|endoftext|>"]
    text = "Hello<|endoftext|>world<|endoftext|>!"
    ids = tok.encode(text)
    assert ids.count(eos) == 2, ids
    assert tok.decode(ids) == text
    # 没有 special 的文本里不应该出现 special id
    assert eos not in tok.encode("Hello, how are you?")


def test_overlapping_special_tokens_longest_match():
    """重叠 special token：<|endoftext|> 与 <|endoftext|><|endoftext|> 要按最长匹配。"""
    vocab_path = _needs_fixtures("gpt2_vocab.json")
    merges_path = _needs_fixtures("gpt2_merges.txt")
    specials = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
    tok = BPETokenizer.from_files(vocab_path, merges_path, specials)
    single_id = tok._special_to_id["<|endoftext|>"]
    double_id = tok._special_to_id["<|endoftext|><|endoftext|>"]
    assert single_id != double_id

    text = "A<|endoftext|><|endoftext|>B"
    ids = tok.encode(text)
    assert double_id in ids, ids
    assert ids.count(double_id) == 1
    assert single_id not in ids  # 双 token 不应该被拆成两个单 token
    assert tok.decode(ids) == text

    # 只传单 token 时，同样的文本必须被拆成两个单 token
    tok_single = BPETokenizer.from_files(vocab_path, merges_path, ["<|endoftext|>"])
    only_single = tok_single.encode(text)
    assert only_single.count(tok_single._special_to_id["<|endoftext|>"]) == 2


def test_special_token_is_hard_boundary_in_training():
    """训练时 special token 是硬边界：不出现在词表（除自身）也不出现在 merges 里。"""
    corpus = "aaaa<|endoftext|>aaaa bbbb<|endoftext|>bbbb abab"
    vocab, merges = train_bpe(corpus, 300, ["<|endoftext|>"])
    assert [tok for tok in vocab.values() if tok == b"<|endoftext|>"] == [b"<|endoftext|>"]
    assert vocab[256] == b"<|endoftext|>"
    for a, b in merges:
        assert b"<|" not in a and b"<|" not in b, (a, b)
        assert b"|>" not in a + b
    # merge 产物里不该出现 special token 的字符（单个字节 token 是 256 初始词表，不算）
    learned = [vocab[i] for i in range(257, len(vocab))]
    assert learned, "应该有 merge 产物"
    assert not any(b"|" in token or b"<" in token or b">" in token for token in learned)


# ---------------------------------------------------------------------------
# 5. 解码容错
# ---------------------------------------------------------------------------
def test_decode_invalid_utf8_and_unknown_ids():
    vocab = {
        0: b"\xff",  # 单独一个 0xFF 不是合法 UTF-8
        1: b"a",
        2: b"\xe4\xbd",  # 「你」的前两个字节，单独不合法
        3: b"\xa0\xe5\xa5\xbd",  # 补完「你」+「好」
    }
    tok = BPETokenizer(vocab, [])
    assert tok.decode([0]) == "\ufffd"
    assert tok.decode([2]) == "\ufffd"
    assert tok.decode([1, 0, 1]) == "a\ufffda"
    # 未知 id 也容错成一个替换字符，而不是 KeyError
    assert tok.decode([999]) == "\ufffd"
    assert tok.decode([1, 999, 1]) == "a\ufffda"
    # 教学点：单个 token 内部不合法，但拼起来可以合法
    assert tok.decode([2, 3]) == "你好"


# ---------------------------------------------------------------------------
# 6. encode_iterable：惰性 + 与整段编码一致
# ---------------------------------------------------------------------------
def test_encode_iterable_matches_encode_for_unsafe_chunk_boundaries():
    """把文本切成任意小块，encode_iterable 必须和 encode(整段) 完全一致。"""
    tok = _gpt2_tokenizer() if (FIXTURES / "gpt2_vocab.json").exists() else _small_tokenizer()
    texts = [
        "abc\n\n def",  # 空白 run 跨块的反例
        "a\n\nb   c\t d",
        "I'll wake you up, we're here, he's done, they've gone, I'd like it",
        "Hello<|endoftext|>world<|endoftext|><|endoftext|>end",
        "   leading and trailing   ",
        "中文\n\n测试  ",
        " 😀  emoji\n\n🎉 ",
    ]
    path = FIXTURES / "tinystories_sample_5M.txt"
    if path.exists():
        texts.append(path.read_text(encoding="utf-8")[:5_000])
    for text in texts:
        expected = tok.encode(text)
        for chunk_size in (1, 2, 3, 5, 7, 13, 64, 997):
            chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
            got = list(tok.encode_iterable(iter(chunks)))
            assert got == expected, (
                f"chunk_size={chunk_size} 时和整段编码不一致：{text[:30]!r}"
            )


def test_encode_iterable_is_lazy():
    """用会记录消费量的生成器验证惰性：只取几个 token 时不应该读完全文。"""
    tok = _small_tokenizer()
    text = ("The little dog ran in the park. " * 100)[:4_000]
    consumed = [0]

    def gen(chunk_size=64):
        for i in range(0, len(text), chunk_size):
            piece = text[i : i + chunk_size]
            consumed[0] += len(piece)
            yield piece

    iterator = tok.encode_iterable(gen())
    ids = [next(iterator) for _ in range(3)]
    assert len(ids) == 3
    assert consumed[0] < len(text) / 2, f"读了 {consumed[0]}/{len(text)} 字符，不够惰性"
    # 剩下的仍然要能正确接着产出
    rest = list(iterator)
    assert ids + rest == tok.encode(text)


# ---------------------------------------------------------------------------
# 7. 词表/merges 数量关系 + 压缩比
# ---------------------------------------------------------------------------
def test_vocab_merge_relation_and_compression():
    text = _training_text(40_000)
    vocab_size = 600
    specials = ["<|endoftext|>"]
    vocab, merges = train_bpe(text, vocab_size, specials)

    # 数量关系：vocab_size = 256 + 1 + len(merges)
    assert len(vocab) == vocab_size
    assert len(merges) == vocab_size - 256 - len(specials)
    assert sorted(vocab) == list(range(vocab_size))
    assert all(vocab[i] == bytes([i]) for i in range(256))
    assert vocab[256] == b"<|endoftext|>"
    # 每个 merge 的产物都在词表里
    assert {a + b for a, b in merges} == set(vocab[i] for i in range(257, vocab_size))

    tok = BPETokenizer(vocab, merges, specials)
    holdout = text[-8_000:] if len(text) > 8_000 else text
    n_bytes = len(holdout.encode("utf-8"))
    ids = tok.encode(holdout)
    compression = n_bytes / len(ids)
    assert 1.5 < compression < 8.0, compression
    assert tok.decode(ids) == holdout
    # 压缩比 = bytes/token，和 1/(tokens/byte) 一致
    assert abs(compression - 1.0 / (len(ids) / n_bytes)) < 1e-9


# ---------------------------------------------------------------------------
# 8. 序列化 / 反序列化
# ---------------------------------------------------------------------------
def test_save_and_load_roundtrip():
    text = _training_text(20_000)
    vocab, merges = train_bpe(text, 400, ["<|endoftext|>"])
    tok = BPETokenizer(vocab, merges, ["<|endoftext|>"])
    sample = text[:2_000]
    expected_ids = tok.encode(sample)

    with tempfile.TemporaryDirectory() as tmp:
        vocab_path, merges_path = tok.save(tmp)
        assert vocab_path.exists() and merges_path.exists()
        reloaded = BPETokenizer.load(tmp, special_tokens=["<|endoftext|>"])
        assert reloaded.vocab == tok.vocab
        assert reloaded.merges == tok.merges
        assert reloaded.encode(sample) == expected_ids
        assert reloaded.decode(expected_ids) == sample

        # from_files 与 load 等价
        reloaded2 = BPETokenizer.from_files(
            vocab_path, merges_path, special_tokens=["<|endoftext|>"]
        )
        assert reloaded2.encode(sample) == expected_ids


# ---------------------------------------------------------------------------
# 9. 与官方 fixture 的对照（可选，需要 assignment1-basics 的 fixtures）
# ---------------------------------------------------------------------------
def test_matches_a1_reference_merges():
    """corpus.en + vocab 500 时，merges/vocab 必须与官方 reference fixture 完全一致。"""
    corpus_path = _needs_fixtures("corpus.en")
    ref_merges_path = _needs_fixtures("train-bpe-reference-merges.txt")
    ref_vocab_path = _needs_fixtures("train-bpe-reference-vocab.json")

    corpus = corpus_path.read_text(encoding="utf-8")
    vocab, merges = train_bpe(corpus, 500, ["<|endoftext|>"])

    reference_merges = []
    for line in ref_merges_path.read_text(encoding="utf-8").splitlines():
        a, b = line.rstrip().split(" ")
        reference_merges.append((toy_bpe._gpt2_str_to_bytes(a), toy_bpe._gpt2_str_to_bytes(b)))
    assert merges == reference_merges

    import json

    raw_vocab = json.loads(ref_vocab_path.read_text(encoding="utf-8"))
    reference_vocab = {
        int(idx): toy_bpe._gpt2_str_to_bytes(token) for token, idx in raw_vocab.items()
    }
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())


def test_gpt2_tokenizer_matches_transformers():
    """自写的 from_files + encode 必须与 transformers 的 GPT-2 tokenizer 完全一致。"""
    weights = Path(
        "/Users/yunni/Projects/cs336-prep-code/real_llm_lab/weights/gpt2"
    )
    if not weights.exists():
        _skip("本地 GPT-2 权重不存在")
    try:
        from transformers import AutoTokenizer
    except Exception:
        _skip("transformers 未安装")
    ref = AutoTokenizer.from_pretrained(str(weights), local_files_only=True)
    tok = BPETokenizer.from_files(
        weights / "vocab.json", weights / "merges.txt", ["<|endoftext|>"]
    )
    texts = [
        "Hello, how are you?",
        "some text that i'll pre-tokenize",
        "the cat ate",
        "中文测试，BPE 很好玩！",
        "Grüße aus München — äöüß",
        "  spaces\n\nand   newlines\t ",
        "emoji 😀👨‍👩‍👧‍👦 done",
    ]
    for text in texts:
        assert tok.encode(text) == ref.encode(text), f"tokenize 不一致: {text!r}"


# ---------------------------------------------------------------------------
# 双模式入口
# ---------------------------------------------------------------------------
def _all_tests() -> list[tuple[str, object]]:
    return [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]


def main() -> int:
    tests = _all_tests()
    passed = failed = skipped = 0
    print(f"运行 {len(tests)} 个测试（fixtures: {FIXTURES}）\n")
    for name, fn in tests:
        try:
            fn()  # type: ignore[operator]
        except SkipTest as exc:
            skipped += 1
            print(f"SKIP {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - 测试运行器要抓住所有异常
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        else:
            passed += 1
            print(f"PASS {name}")
    print(f"\n汇总: {passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
