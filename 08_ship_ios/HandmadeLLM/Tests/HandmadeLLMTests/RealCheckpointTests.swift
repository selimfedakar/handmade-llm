import MLX
import XCTest

@testable import HandmadeLLM

/// The same equivalence, against the real 24.9M model.
///
/// The tiny model in `ModelTests` proves the port is right. This proves it is
/// right at the size that ships — 14.9 MiB of packed codes, a 4,097-token
/// vocabulary, eight layers — which is a different claim, and chapter 07 is the
/// reason to insist on the distinction: its byte-identity test passed on
/// matrices small enough to read and failed on the real checkpoint, because
/// rounding ties only occur at scale. `docs/LESSONS.md` L12.
///
/// `runs/` is gitignored, so these skip on a fresh clone and say what to run.
final class RealCheckpointTests: XCTestCase {

    override func setUpWithError() throws {
        try super.setUpWithError()
        try TestDevice.prepare()
    }

    private var directory: URL {
        Golden.repositoryRoot.appendingPathComponent("runs/latest/quantized")
    }

    private func requireCheckpoint() throws {
        try Golden.requireCheckpoint(
            directory,
            hint: "run `python 07_quantize/compare.py`, then `python 08_ship_ios/export_golden.py`")
    }

    func test_swift_reads_the_shipped_checkpoint_and_agrees_with_python() throws {
        try requireCheckpoint()
        let golden = try Golden.json("real-quantized.json")
        let prompt = golden["prompt"] as! [Int]
        let expected = (golden["logits"] as! [Double]).map(Float.init)

        let model = try Transformer.load(directory: directory)
        XCTAssertEqual(model.quantization?.bits, 4)
        XCTAssertEqual(model.quantization?.groupSize, 64)

        let (logits, _) = try model(MLXArray(prompt.map { Int32($0) }).reshaped(1, prompt.count))
        eval(logits)
        // The fixture keeps only the last position: it is the one generation
        // reads, and a 4,097-wide vocabulary times every prompt position is half
        // a megabyte of JSON to say the same thing five times.
        let actual = logits[0..., -1, 0...].asArray(Float.self)

        XCTAssertEqual(actual.count, expected.count)
        let worst = Golden.largestDifference(actual, expected)
        print(
            String(
                format: "  real model: %d logits, worst |Δ| = %.3e at token %d",
                actual.count, worst.delta, worst.index))
        XCTAssertLessThan(worst.delta, 1e-3)

        // The argmax is what generation acts on, so it gets its own assertion.
        // Two logit vectors can differ by less than the tolerance and still
        // disagree about the winner if the top two are close.
        let swiftArgmax = actual.firstIndex(of: actual.max()!)
        let pythonArgmax = expected.firstIndex(of: expected.max()!)
        XCTAssertEqual(swiftArgmax, pythonArgmax, "the two languages pick different next tokens")
    }

    /// Where the two languages part company, if they do.
    ///
    /// The tiny model reproduces Python's logits to the bit; the real one came
    /// out 1.4e-06 away, which changes no token and is exactly the size of
    /// difference that is easiest to shrug at. Comparing outputs tells you
    /// *that* something differs and never *where*, so the fixture carries two
    /// points inside the forward pass and this splits the network into three:
    /// the table lookup, the eight layers, and the tied output head.
    ///
    /// The layer loop below mirrors `intermediates()` in `export_golden.py` —
    /// both stop half way through the same forward pass on purpose.
    func test_where_the_two_languages_stop_agreeing() throws {
        try requireCheckpoint()
        let golden = try Golden.json("real-quantized.json")
        let prompt = (golden["prompt"] as! [Int]).map { Int32($0) }
        let ids = MLXArray(prompt).reshaped(1, prompt.count)

        let model = try Transformer.load(directory: directory)

        let embedded = model.embed(ids)
        eval(embedded)
        let embeddingDelta = Golden.largestDifference(
            embedded.asArray(Float.self), (golden["embedding"] as! [Double]).map(Float.init))

        var x = embedded
        let mask = causalMask(
            nQueries: prompt.count, nKeys: prompt.count, dtype: x.dtype)
        for layer in model.layers {
            (x, _) = layer(x, mask: mask, cache: nil)
        }
        let hidden = model.norm(x)
        eval(hidden)
        let hiddenDelta = Golden.largestDifference(
            hidden.asArray(Float.self), (golden["hidden"] as! [Double]).map(Float.init))

        let (logits, _) = try model(ids)
        eval(logits)
        let logitDelta = Golden.largestDifference(
            logits[0..., -1, 0...].asArray(Float.self),
            (golden["logits"] as! [Double]).map(Float.init))

        print(
            String(
                format: """
                      python vs swift, three points in one forward pass
                        after the embedding   worst |Δ| = %.3e
                        after eight layers    worst |Δ| = %.3e
                        after the output head worst |Δ| = %.3e
                    """, embeddingDelta.delta, hiddenDelta.delta, logitDelta.delta))

        // The embedding is the one place a claim of exactness is cheap to keep:
        // it is a gather and two float operations, written out on both sides.
        XCTAssertEqual(embeddingDelta.delta, 0, "the embedding lookup diverged")
        XCTAssertLessThan(hiddenDelta.delta, 1e-4)
        XCTAssertLessThan(logitDelta.delta, 1e-4)
    }

    /// How far apart two *correct* implementations of this model are, at this
    /// size, in one language.
    ///
    /// This is the number that makes the one above readable. Swapping MLX's
    /// fused attention for the written-out softmax changes nothing about what
    /// the model computes and moves the logits by more than the whole
    /// Python-to-Swift gap. So 1.4e-06 across eight layers of float32 is not a
    /// port defect with a cause worth finding — it is the floor, and the tiny
    /// model reproduces Python exactly only because two layers of 64 numbers
    /// never reach it.
    ///
    /// Small test data understates your noise floor exactly as it understates
    /// your edge cases. `docs/LESSONS.md` L12 said the same thing about
    /// rounding ties, from the other direction.
    func test_two_correct_attentions_differ_by_more_than_the_two_languages_do() throws {
        try requireCheckpoint()
        let fused = try Transformer.load(directory: directory)
        let written = fused.withFusedAttention(false)
        let ids = MLXArray([70, 105, 114, 115, 116] as [Int32]).reshaped(1, 5)

        let (a, _) = try fused(ids)
        let (b, _) = try written(ids)
        eval(a, b)

        let worst = Golden.largestDifference(a.asArray(Float.self), b.asArray(Float.self))
        print(String(format: "  fused vs written-out on the real model: worst |Δ| = %.3e", worst.delta))
        XCTAssertEqual(
            a[0..., -1, 0...].argMax(axis: -1).item(Int.self),
            b[0..., -1, 0...].argMax(axis: -1).item(Int.self),
            "the two attention paths disagree about the next token, which is a real bug")
    }

    func test_swift_generates_the_same_text_as_python() throws {
        try requireCheckpoint()
        let golden = try Golden.json("real-quantized.json")
        let prompt = golden["prompt"] as! [Int]
        let expected = golden["greedy"] as! [Int]

        let model = try Transformer.load(directory: directory)
        var produced: [Int] = []
        var ids = MLXArray(prompt.map { Int32($0) }).reshaped(1, prompt.count)
        var cache: [KVCache]? = nil
        for _ in 0..<expected.count {
            let (logits, next) = try model(ids, cache: cache)
            cache = next
            let token = logits[0..., -1, 0...].argMax(axis: -1)
            eval(token)
            produced.append(token.item(Int.self))
            ids = token.reshaped(1, 1)
        }
        XCTAssertEqual(produced, expected)
    }

    func test_the_shipped_model_is_the_size_chapter_07_says_it_is() throws {
        try requireCheckpoint()
        let model = try Transformer.load(directory: directory)
        let mib = Double(model.weightByteCount) / 1_048_576
        print(String(format: "  weights held: %.2f MiB", mib))
        // Chapter 07 measured 14.88 MiB. Counting is counting, so this is a
        // narrow band rather than an approximate one.
        XCTAssertEqual(mib, 14.88, accuracy: 0.05)
    }

    func test_the_tokenizer_and_the_model_agree_about_the_vocabulary() throws {
        try requireCheckpoint()
        let tokenizerURL = Golden.repositoryRoot.appendingPathComponent("data/tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokenizerURL.path) else {
            throw XCTSkip("no data/tokenizer.json — it is gitignored")
        }
        let tokenizer = try BPETokenizer(contentsOf: tokenizerURL)
        let model = try Transformer.load(directory: directory)

        // The vocabulary rule, checked on the other side of the bridge: an id
        // the embedding has no row for does not raise in MLX, it returns a
        // number. `docs/LESSONS.md` L7 is what that cost the first time.
        XCTAssertEqual(
            tokenizer.vocabularySize, model.config.vocabSize,
            "the tokenizer and the checkpoint were built from different vocabularies")
    }
}
