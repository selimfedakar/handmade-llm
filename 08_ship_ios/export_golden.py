"""Chapter 08, part one — what Swift has to agree with.

The claim this chapter makes is that the Swift tokenizer and the Swift model
produce the same token ids and the same logits as the Python ones. A claim like
that is worth exactly as much as the evidence a reader can run, so the evidence
is a set of files: this script writes what Python does, and
`HandmadeLLM/Tests/` asserts Swift does the same. No Python interpreter is
needed to check it, and no phone.

    python 08_ship_ios/export_golden.py

Four fixtures come out, and they are deliberately different in kind:

    splits.json     text -> chunks, from the real regular expression
    tokens.json     text -> ids, and back
    tiny/           a whole small model, quantized, with reference logits
    real.json       the same for the 24.9M checkpoint, when one exists

The first one is the interesting one. Chapter 01 splits text with a Python
regular expression, and Swift has no Python regular expression — the `\\w` in
`NSRegularExpression` is ICU's, which is a different set of characters than
CPython's. The inputs below are chosen to make the two disagree if they are
going to: combining marks, connector punctuation, superscripts, an emoji built
out of joiners, and the four control characters CPython calls whitespace and
Unicode does not. On English text every implementation agrees. English text is
therefore not evidence.

`docs/LESSONS.md` L12 in its original form: an equivalence test on data small
enough to read tests your algorithm, not your conventions.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "01_tokenizer"))
sys.path.insert(0, str(ROOT / "02_model"))
sys.path.insert(0, str(ROOT / "07_quantize"))

from bpe import SPLIT_PATTERN, BPETokenizer  # noqa: E402
from model import ModelConfig, Transformer  # noqa: E402
from quantize import quantize_model, save_quantized  # noqa: E402

GOLDEN = Path(__file__).resolve().parent / "HandmadeLLM/Tests/HandmadeLLMTests/Golden"

# The tiny model exists so the equivalence test runs for anyone who clones this
# repository, with no corpus, no training run, and no 15 MiB checkpoint. It is
# 106,496 weights — small enough to commit, large enough to exercise both
# attention paths, grouped-query attention, the KV-cache and a quantized tied
# embedding. Group size 64 has to divide every matrix, so `d_ff` is 192 and not
# something rounder.
TINY = ModelConfig(
    vocab_size=128,
    d_model=64,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,
    d_ff=192,
    max_seq_len=64,
)
TINY_SEED = 8
TINY_PROMPT = [3, 17, 42, 9, 100, 5, 63, 1, 88, 12]
TINY_GENERATE = 24

# Texts the splitter is asked about. Ordinary English first, then the ones
# chosen to break something.
SPLIT_CASES = [
    # Ordinary. These agree under any implementation, which is the point of
    # having the rest of the list.
    "First Citizen:\nBefore we proceed any further, hear me speak.",
    "To be, or not to be, that is the question.",
    "",
    " ",
    "   ",
    "a b",
    "a  b",
    "a   b",
    "hello   ",
    "\n\n",
    "\ttabbed\ttext\t",
    # Contractions: the branch that exists only for English, and is
    # case-sensitive.
    "don't we'll I've you're he'd it's",
    "DON'T WE'LL I'VE",
    "'s 'S 'll 'LL 'x '",
    # Digits, capped at three.
    "1 12 123 1234 1234567",
    "in 1999 and 2026",
    "3.14159",
    "007",
    # Underscores. Chapter 01's pattern has no branch that accepts one, so the
    # tokenizer cannot see them at all. Reproducing that is not optional.
    "snake_case_name",
    "_leading and trailing_",
    "a_b_c",
    "__dunder__",
    # Punctuation and symbols.
    "!!!???",
    "-- hey --",
    "a+b=c",
    "(parenthesised)",
    "€100 and $5 and £3",
    # Scripts. Alphabetic in both engines, but worth having on the record.
    "İstanbul'da güneş açtı",
    "Grüße aus Straße",
    "日本語のテキストです",
    "Здравствуй, мир",
    "مرحبا بالعالم",
    "नमस्ते दुनिया",
    # Combining marks. ICU counts `\p{M}` as a word character and CPython does
    # not, so `e` + U+0301 is one chunk to one engine and possibly two to the
    # other. This is the case the whole file is built around.
    "éclair",
    "à́̂",
    "́ alone",
    # Numeric characters that are not decimal digits: `\d` is decimal-only, and
    # `[^\W\d_]` therefore accepts them.
    "² ½ Ⅸ 一",
    "x² + y²",
    # Connector punctuation other than the underscore.
    "a‿b",
    # Whitespace CPython recognises and Unicode does not.
    "ab",
    "abcd",
    # Whitespace Unicode recognises: no-break space, ideographic space, and a
    # zero-width space that is *not* whitespace to either engine.
    "a b",
    "a　b",
    "a​b",
    # Emoji, including a family built from five scalars and four joiners. Python
    # matches code points, so this is nine things to the pattern and one
    # `Character` to Swift.
    "👨‍👩‍👧‍👦 family",
    "🇹🇷 flag",
    "hello 🌍!",
    # Mixed, because real input is mixed.
    "İyi günler! 42 çilek 🍓 _x_ é",
]


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {path.relative_to(ROOT)}  ({path.stat().st_size / 1024:.1f} KiB)")


def export_splits() -> None:
    """What `re.findall(SPLIT_PATTERN, text)` actually returns, for each case."""
    import re

    compiled = re.compile(SPLIT_PATTERN)
    write_json(
        GOLDEN / "splits.json",
        {
            "pattern": SPLIT_PATTERN,
            "cases": [{"text": text, "chunks": compiled.findall(text)} for text in SPLIT_CASES],
        },
    )


def export_tokens(tokenizer_path: Path) -> None:
    """The same texts, all the way through the tokenizer."""
    tokenizer = BPETokenizer.load(tokenizer_path)
    cases = []
    for text in SPLIT_CASES:
        ids = tokenizer.encode(text)
        cases.append({"text": text, "ids": ids, "decoded": tokenizer.decode(ids)})

    specials = tokenizer.special_tokens
    if specials:
        marker = next(iter(specials))
        text = f"before{marker}after"
        cases.append(
            {
                "text": text,
                "ids": tokenizer.encode(text, allowed_special="all"),
                "decoded": tokenizer.decode(tokenizer.encode(text, allowed_special="all")),
                "specials": "recognised",
            }
        )

    write_json(
        GOLDEN / "tokens.json",
        {
            "vocab_size": tokenizer.vocab_size,
            "special_tokens": specials,
            "cases": cases,
        },
    )


def reference_run(
    model: Transformer, prompt: list[int], steps: int, last_position_only: bool = False
) -> dict:
    """Logits over the prompt, and then a greedy continuation.

    Greedy, because that is the only decoding both languages can be expected to
    agree on token for token. Both draw from the same distribution when
    sampling, but MLX's Python bindings carry a global random state and
    mlx-swift threads an explicit key, so identical logits still produce
    different samples. Comparing the argmax compares the model; comparing a
    sample would compare two random number generators.
    """
    ids = mx.array(prompt)[None]
    logits, cache = model(ids)
    mx.eval(logits)

    generated: list[int] = []
    for _ in range(steps):
        token = mx.argmax(logits[:, -1, :], axis=-1, keepdims=True)
        mx.eval(token)
        generated.append(int(token.item()))
        logits, cache = model(token, cache)

    prompt_logits, _ = model(ids)
    if last_position_only:
        # A 4,097-wide vocabulary times every prompt position is half a megabyte
        # of JSON in the repository for a test that compares the same thing five
        # times. The last position is the one generation reads.
        prompt_logits = prompt_logits[:, -1:, :]
    mx.eval(prompt_logits)
    return {
        "prompt": prompt,
        "logits_shape": list(prompt_logits.shape),
        "logits": [float(v) for v in prompt_logits.reshape(-1).tolist()],
        "greedy": generated,
    }


def export_tiny() -> None:
    """A whole model, small enough to live in the repository."""
    mx.random.seed(TINY_SEED)
    model = Transformer(TINY)
    mx.eval(model.parameters())

    float_dir = GOLDEN / "tiny-float32"
    float_dir.mkdir(parents=True, exist_ok=True)
    from mlx.utils import tree_flatten

    mx.save_safetensors(
        str(float_dir / "weights.safetensors"), dict(tree_flatten(model.parameters()))
    )
    (float_dir / "meta.json").write_text(
        json.dumps({"step": None, "model_config": asdict(TINY)}, indent=2), encoding="utf-8"
    )
    write_json(GOLDEN / "tiny-float32.json", reference_run(model, TINY_PROMPT, TINY_GENERATE))
    print(f"  Golden/tiny-float32/weights.safetensors"
          f"  ({(float_dir / 'weights.safetensors').stat().st_size / 1024:.1f} KiB)")

    # Quantize a *fresh* copy: `quantize_model` replaces layers in place, and the
    # float32 reference above has to be the model before that happened.
    mx.random.seed(TINY_SEED)
    quantized = quantize_model(Transformer(TINY))
    mx.eval(quantized.parameters())
    save_quantized(GOLDEN / "tiny-quantized", quantized, step=None)
    write_json(
        GOLDEN / "tiny-quantized.json", reference_run(quantized, TINY_PROMPT, TINY_GENERATE)
    )
    size = (GOLDEN / "tiny-quantized" / "weights.safetensors").stat().st_size
    print(f"  Golden/tiny-quantized/weights.safetensors  ({size / 1024:.1f} KiB)")


def export_real(directory: Path, prompt: list[int], steps: int) -> None:
    """The same reference for the real checkpoint, when it is on this machine.

    The checkpoint itself is gitignored, so the test that uses this skips on a
    fresh clone and says why. The numbers are committed anyway: they are what
    the chapter's notes quote, and a quoted number with no file behind it is a
    number nobody can check.
    """
    sys.path.insert(0, str(ROOT / "07_quantize"))
    from quantize import load_quantized

    model, meta = load_quantized(directory)
    payload = reference_run(model, prompt, steps, last_position_only=True)
    payload["step"] = meta.get("step")
    payload["quantization"] = meta.get("quantization")
    payload.update(intermediates(model, prompt))
    write_json(GOLDEN / "real-quantized.json", payload)


def intermediates(model: Transformer, prompt: list[int]) -> dict:
    """Two points inside the forward pass, so a disagreement can be located.

    The tiny model in `Golden/` reproduces Python's logits to the bit. The real
    one did not — 1.4e-06 across 4,097 logits, which changes no token and was
    still worth an hour, because "close enough" is how a real port bug hides.
    Comparing only the output tells you that something differs, never where.

    So the embedding output and the final normalised hidden state come out too.
    Between them they split the network into three pieces: the table lookup, the
    eight layers, and the tied output head.

    The loop below repeats `Transformer.__call__` rather than calling it,
    because the point is to stop half way. If chapter 02's forward pass ever
    changes, this has to change with it — which is a real cost and the reason it
    lives in an export script rather than in the model.
    """
    from model import causal_mask

    ids = mx.array(prompt)[None]
    embedded = model.embed(ids)
    mx.eval(embedded)

    x = embedded
    mask = causal_mask(x.shape[1], x.shape[1], x.dtype)
    for layer in model.layers:
        x, _ = layer(x, mask, None)
    hidden = model.norm(x)
    mx.eval(hidden)

    return {
        "embedding": [float(v) for v in embedded.reshape(-1).tolist()],
        "hidden": [float(v) for v in hidden.reshape(-1).tolist()],
        "hidden_shape": list(hidden.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "data/tokenizer.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/latest/quantized")
    parser.add_argument("--prompt", type=int, nargs="+", default=[70, 105, 114, 115, 116])
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()

    print("golden fixtures ->", GOLDEN.relative_to(ROOT))
    export_splits()

    if args.tokenizer.exists():
        export_tokens(args.tokenizer)
    else:
        print(f"  skipped tokens.json: no {args.tokenizer.relative_to(ROOT)}")
        print("  run: python 01_tokenizer/train_tokenizer.py --vocab-size 4096")

    export_tiny()

    if (args.checkpoint / "meta.json").exists():
        export_real(args.checkpoint, args.prompt, args.steps)
    else:
        print(f"  skipped real-quantized.json: no {args.checkpoint}")
        print("  run: python 07_quantize/compare.py")


if __name__ == "__main__":
    main()
