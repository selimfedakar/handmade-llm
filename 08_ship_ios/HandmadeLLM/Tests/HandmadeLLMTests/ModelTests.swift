import MLX
import XCTest

@testable import HandmadeLLM

/// The Swift transformer against the Python one.
///
/// The model under test is 106,496 weights — two layers, 64 wide, a 128-token
/// vocabulary — and it is committed to the repository in both float32 and 4-bit
/// form, together with the logits Python got out of it. So this runs on a fresh
/// clone with no corpus, no training run, and no phone.
///
/// It is deliberately small in size and not in kind. It has grouped-query
/// attention, a tied quantized embedding, a KV-cache, and both attention
/// implementations, because those are the four places a port goes quietly wrong.
final class ModelTests: XCTestCase {

    override func setUpWithError() throws {
        try super.setUpWithError()
        try TestDevice.prepare()
    }

    // -- fixtures --------------------------------------------------------------

    private struct Reference {
        var prompt: [Int]
        var logitsShape: [Int]
        var logits: [Float]
        var greedy: [Int]
    }

    private func reference(_ name: String) throws -> Reference {
        let json = try Golden.json(name)
        return Reference(
            prompt: json["prompt"] as! [Int],
            logitsShape: json["logits_shape"] as! [Int],
            logits: (json["logits"] as! [Double]).map(Float.init),
            greedy: json["greedy"] as! [Int])
    }

    private func model(_ directory: String) throws -> Transformer {
        try Transformer.load(directory: Golden.directory.appendingPathComponent(directory))
    }

    // -- the equivalence -------------------------------------------------------

    func test_the_float32_model_produces_pythons_logits() throws {
        try assertLogitsMatch(model: try model("tiny-float32"), reference: try reference("tiny-float32.json"))
    }

    func test_the_quantized_model_produces_pythons_logits() throws {
        try assertLogitsMatch(
            model: try model("tiny-quantized"), reference: try reference("tiny-quantized.json"))
    }

    /// Same weights, same input, same numbers — and `tolerance` is **zero**.
    ///
    /// That is a real claim and it is meant to be brittle. Both languages call
    /// the same C++ kernels in the same order on the same bytes, so there is
    /// nothing left for a difference to come from, and if one appears it is
    /// information rather than noise. It already paid for itself once: the first
    /// version of this port decoded the quantized embedding through MLX's
    /// `dequantized` kernel while Python does the arithmetic itself, the two
    /// agree to 3e-08, and by the output head that had grown to 2e-06. Under a
    /// 1e-4 tolerance it would have passed and nobody would have looked.
    private func assertLogitsMatch(
        model: Transformer, reference: Reference, tolerance: Float = 0
    ) throws {
        let ids = MLXArray(reference.prompt.map { Int32($0) }).reshaped(
            1, reference.prompt.count)
        let (logits, _) = try model(ids)
        eval(logits)

        XCTAssertEqual(logits.shape, reference.logitsShape)
        let actual = logits.asArray(Float.self)
        XCTAssertEqual(actual.count, reference.logits.count)

        let worst = Golden.largestDifference(actual, reference.logits)
        print(
            String(
                format: "  logits: %d values, worst |Δ| = %.3e at index %d (python %.6f, swift %.6f)",
                actual.count, worst.delta, worst.index,
                reference.logits[worst.index], actual[worst.index]))
        XCTAssertLessThanOrEqual(worst.delta, tolerance)
    }

    /// The repository's signature move, in Swift: the version you can read next
    /// to the kernel that ships, and a test between them.
    ///
    /// They do **not** agree exactly here, and that is the finding rather than a
    /// failure. `unpackCodes` + `dequantizeGroupwise` is the arithmetic
    /// `07_quantize/quantize.py` performs; `dequantized` is a fused kernel doing
    /// the same algebra in a different order. Somewhere below 1e-07 they part
    /// company, which is why the model uses the written-out path — it is the one
    /// that reproduces Python to the bit.
    func test_the_written_out_decoder_agrees_with_mlxs_kernel() throws {
        let model = try model("tiny-quantized")
        let ids = MLXArray((0..<8).map { Int32($0 * 13 % 128) }).reshaped(1, 8)

        let written = model.embed(ids)
        let kernel = model.embed.lookupThroughKernel(ids)
        eval(written, kernel)

        let worst = Golden.largestDifference(
            written.asArray(Float.self), kernel.asArray(Float.self))
        print(String(format: "  written-out decode vs kernel: worst |Δ| = %.3e", worst.delta))
        XCTAssertLessThan(worst.delta, 1e-6)
    }

    /// `unpackCodes` on a word whose bits are written out by hand, so the layout
    /// claim is checked against something other than another implementation of
    /// itself.
    func test_unpacking_reads_the_bit_stream_low_code_first() {
        // Eight 4-bit codes packed low-first into one word: 1, 2, 3, ..., 8.
        var word: UInt32 = 0
        for (i, code) in [1, 2, 3, 4, 5, 6, 7, 8].enumerated() {
            word |= UInt32(code) << (i * 4)
        }
        let codes = unpackCodes(MLXArray([word]), bits: 4)
        eval(codes)
        XCTAssertEqual(codes.asArray(UInt8.self), [1, 2, 3, 4, 5, 6, 7, 8])
    }

    func test_greedy_generation_matches_python_token_for_token() throws {
        for (directory, fixture) in [
            ("tiny-float32", "tiny-float32.json"), ("tiny-quantized", "tiny-quantized.json"),
        ] {
            let model = try model(directory)
            let reference = try reference(fixture)
            XCTAssertEqual(
                greedy(model, prompt: reference.prompt, steps: reference.greedy.count),
                reference.greedy,
                "\(directory): the two languages disagree about what this model says")
        }
    }

    /// Greedy, because it is the only decoding the two languages can be expected
    /// to agree on. Both draw from the same distribution when sampling, but
    /// MLX's Python bindings carry a global random state and mlx-swift threads
    /// an explicit key — comparing samples would compare two random number
    /// generators, not two models.
    private func greedy(_ model: Transformer, prompt: [Int], steps: Int) -> [Int] {
        var out: [Int] = []
        var ids = MLXArray(prompt.map { Int32($0) }).reshaped(1, prompt.count)
        var cache: [KVCache]? = nil
        for _ in 0..<steps {
            guard let (logits, next) = try? model(ids, cache: cache) else { return out }
            cache = next
            let token = logits[0..., -1, 0...].argMax(axis: -1)
            eval(token)
            out.append(token.item(Int.self))
            ids = token.reshaped(1, 1)
        }
        return out
    }

    // -- the properties a port breaks quietly ----------------------------------

    /// Chapter 02's signature test, carried over. The readable implementation
    /// and the fused kernel have to agree, or the readable one is decoration.
    func test_written_out_attention_matches_the_fused_kernel() throws {
        let fused = try model("tiny-float32")
        let written = fused.withFusedAttention(false)
        let ids = MLXArray((0..<16).map { Int32($0 * 7 % 128) }).reshaped(1, 16)

        let (a, _) = try fused(ids)
        let (b, _) = try written(ids)
        eval(a, b)

        let worst = Golden.largestDifference(a.asArray(Float.self), b.asArray(Float.self))
        print(String(format: "  fused vs written out: worst |Δ| = %.3e", worst.delta))
        XCTAssertLessThan(worst.delta, 1e-4)
    }

    /// The KV-cache bug that does not crash: during generation there is one
    /// query and a growing pile of keys, so the query sits at the *end* of the
    /// key range. Get the alignment wrong and shapes stay valid while quality
    /// quietly degrades. `docs/LESSONS.md` L3.
    func test_feeding_tokens_one_at_a_time_matches_one_full_pass() throws {
        for directory in ["tiny-float32", "tiny-quantized"] {
            let model = try model(directory)
            let sequence = (0..<12).map { Int32($0 * 11 % 128) }

            let (full, _) = try model(MLXArray(sequence).reshaped(1, sequence.count))
            eval(full)

            var cache: [KVCache]? = nil
            var stepped: [Float] = []
            for token in sequence {
                let (logits, next) = try model(MLXArray([token]).reshaped(1, 1), cache: cache)
                cache = next
                eval(logits)
                stepped += logits.asArray(Float.self)
            }

            let worst = Golden.largestDifference(full.asArray(Float.self), stepped)
            print(String(format: "  %@ cached vs full: worst |Δ| = %.3e", directory, worst.delta))
            XCTAssertLessThan(worst.delta, 1e-4, directory)
        }
    }

    /// The reason the chapter exists, in one assertion.
    func test_the_quantized_model_is_much_smaller() throws {
        let float32 = try model("tiny-float32").weightByteCount
        let quantized = try model("tiny-quantized").weightByteCount
        let ratio = Double(float32) / Double(quantized)
        print(
            String(
                format: "  weights: %.1f KiB float32, %.1f KiB quantized (%.2fx)",
                Double(float32) / 1024, Double(quantized) / 1024, ratio))
        XCTAssertGreaterThan(ratio, 4.0)
    }

    func test_metadata_survives_the_round_trip() throws {
        let metadata = try Transformer.loadMetadata(
            directory: Golden.directory.appendingPathComponent("tiny-quantized"))
        XCTAssertEqual(metadata.modelConfig.vocabSize, 128)
        XCTAssertEqual(metadata.modelConfig.nLayers, 2)
        XCTAssertEqual(metadata.modelConfig.nKvHeads, 2)
        XCTAssertEqual(metadata.modelConfig.dFf, 192)
        XCTAssertTrue(metadata.modelConfig.tieEmbeddings)
        XCTAssertEqual(metadata.quantization?.bits, 4)
        XCTAssertEqual(metadata.quantization?.groupSize, 64)
        XCTAssertEqual(metadata.quantization?.embedding, true)

        // A float32 checkpoint has the same shape minus the quantization block,
        // and the loader has to read both.
        let plain = try Transformer.loadMetadata(
            directory: Golden.directory.appendingPathComponent("tiny-float32"))
        XCTAssertNil(plain.quantization)
    }

    func test_a_sequence_longer_than_the_model_allows_is_refused() throws {
        let model = try model("tiny-float32")
        let tooLong = MLXArray(
            (0..<(model.config.maxSeqLen + 1)).map { Int32($0 % 128) }
        ).reshaped(1, model.config.maxSeqLen + 1)
        XCTAssertThrowsError(try model(tooLong))
    }
}
