"""Tests for the from-scratch BPE tokenizer.

Run with:  python -m pytest 01_tokenizer -q
"""

import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from bpe import BPETokenizer, count_pairs, merge  # noqa: E402


# -- the primitives -------------------------------------------------------


def test_count_pairs_counts_adjacent_pairs():
    assert count_pairs([1, 2, 1, 2, 3]) == {(1, 2): 2, (2, 1): 1, (2, 3): 1}


def test_count_pairs_on_short_input():
    assert count_pairs([]) == {}
    assert count_pairs([7]) == {}


def test_merge_replaces_every_occurrence():
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]


def test_merge_does_not_overlap_itself():
    # "aaa" with pair (a, a) must consume the first two, not reuse the middle.
    assert merge([5, 5, 5], (5, 5), 99) == [99, 5]


def test_merge_leaves_unrelated_ids_alone():
    assert merge([1, 2, 3], (4, 5), 99) == [1, 2, 3]


# -- round-trips ----------------------------------------------------------

ROUND_TRIP_CASES = [
    "",
    "hello world",
    "The quick brown fox jumps over the lazy dog. " * 4,
    "Merhaba dünya, nasılsın?",  # non-ASCII, multi-byte UTF-8
    "def train(model, data):\n    return model.fit(data)  # code\n",
    "emoji survive byte-level BPE: 🧠🔥",
    "1234567890 " * 5,
]


@pytest.fixture(scope="module")
def trained():
    text = "\n".join(ROUND_TRIP_CASES) * 20
    tok = BPETokenizer()
    tok.train(text, vocab_size=400)
    return tok


@pytest.mark.parametrize("text", ROUND_TRIP_CASES)
def test_untrained_tokenizer_round_trips(text):
    # With zero merges every byte is its own token, so this must be exact.
    tok = BPETokenizer()
    assert tok.decode(tok.encode(text)) == text


@pytest.mark.parametrize("text", ROUND_TRIP_CASES)
def test_trained_tokenizer_round_trips(text):
    tok = BPETokenizer()
    tok.train("\n".join(ROUND_TRIP_CASES) * 20, vocab_size=400)
    assert tok.decode(tok.encode(text)) == text


def test_round_trips_text_it_never_saw(trained):
    unseen = "A byte-level tokenizer has no unknown tokens — ever. 中文もOK."
    assert trained.decode(trained.encode(unseen)) == unseen


# -- training behaviour ---------------------------------------------------


def test_training_compresses(trained):
    text = "The quick brown fox jumps over the lazy dog. " * 10
    raw_bytes = len(text.encode("utf-8"))
    assert len(trained.encode(text)) < raw_bytes


def test_vocab_size_is_respected():
    tok = BPETokenizer()
    tok.train("abcabcabc " * 500, vocab_size=300)
    assert tok.vocab_size <= 300


def test_training_below_256_is_rejected():
    with pytest.raises(ValueError):
        BPETokenizer().train("hello", vocab_size=100)


def test_training_is_deterministic():
    text = "the cat sat on the mat, the cat sat again " * 50
    a, b = BPETokenizer(), BPETokenizer()
    a.train(text, vocab_size=320)
    b.train(text, vocab_size=320)
    assert a.merges == b.merges


def _train_the_slow_obvious_way(text: str, vocab_size: int) -> dict:
    """The textbook loop: re-count everything before every merge.

    `BPETokenizer.train` deduplicates chunks and keeps an incremental pair
    counter instead. That is a large speedup and it is only worth anything if
    it learns exactly the same merges — which is what this reference is here
    to prove.
    """
    import re as _re

    from bpe import SPLIT_PATTERN

    chunks = [list(c.encode("utf-8")) for c in _re.findall(SPLIT_PATTERN, text)]
    merges: dict[tuple[int, int], int] = {}
    for step in range(vocab_size - 256):
        counts = Counter()
        for chunk in chunks:
            count_pairs(chunk, counts)
        if not counts:
            break
        pair = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
        if counts[pair] < 2:
            break
        new_id = 256 + step
        chunks = [merge(chunk, pair, new_id) for chunk in chunks]
        merges[pair] = new_id
    return merges


@pytest.mark.parametrize(
    "text",
    [
        "the cat sat on the mat, the cat sat again. " * 40,
        "def f(x):\n    return x + 1\n\ndef g(x):\n    return f(x) * 2\n" * 30,
        "aaaaaaaaaaaaaaaa " * 60,  # degenerate: overlapping identical pairs
        "Merhaba dünya. Dünya merhaba. Merhaba, merhaba! " * 40,
    ],
)
def test_fast_training_matches_the_slow_obvious_way(text):
    tok = BPETokenizer()
    tok.train(text, vocab_size=340)
    assert tok.merges == _train_the_slow_obvious_way(text, vocab_size=340)


def test_merges_never_cross_the_split_pattern():
    # " dog" and "." are separate chunks, so no token may contain both.
    tok = BPETokenizer()
    tok.train("the dog. the dog. " * 200, vocab_size=320)
    assert not any(b"g." in piece for piece in tok.vocab.values())


# -- special tokens -------------------------------------------------------


def test_special_tokens_are_atomic(trained):
    trained.register_special_tokens({"<|endoftext|>": 500})
    ids = trained.encode("hi<|endoftext|>there", allowed_special="all")
    assert 500 in ids
    assert trained.decode(ids) == "hi<|endoftext|>there"


def test_special_tokens_are_text_by_default(trained):
    trained.register_special_tokens({"<|endoftext|>": 500})
    assert 500 not in trained.encode("hi<|endoftext|>there")


# -- persistence ----------------------------------------------------------


def test_save_and_load_is_lossless(trained, tmp_path):
    path = tmp_path / "tokenizer.json"
    trained.save(path)
    reloaded = BPETokenizer.load(path)

    assert reloaded.merges == trained.merges
    assert reloaded.vocab == trained.vocab
    text = "round-tripping through disk changes nothing at all."
    assert reloaded.encode(text) == trained.encode(text)


def test_decoding_an_unknown_id_is_an_error(trained):
    with pytest.raises(ValueError):
        trained.decode([10**6])
