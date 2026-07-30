import Foundation
import MLX
import MLXRandom

/// Chapter 08 — turning the model into text arriving on a screen.
///
/// Two things here are not in the Python `generate`, and both are because this
/// one is feeding a user interface rather than a terminal:
///
///  * **Bytes are held back until they form a character.** The model emits token
///    ids; a token is a run of UTF-8 bytes; a multi-byte character can arrive
///    across two tokens. Decoding each token on its own and appending gives a
///    replacement character that then never goes away. `ByteStream` below keeps
///    the tail until it completes.
///  * **The caller can stop it.** `onToken` returns `false` and the loop ends,
///    because a phone screen has a back button and a 300-step model has a lot to
///    say.
public struct Generator {
    public let model: Transformer
    public let tokenizer: BPETokenizer

    public init(model: Transformer, tokenizer: BPETokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }

    public struct Options: Sendable {
        public var maxTokens: Int
        /// Zero means greedy — always the most likely token.
        ///
        /// Greedy is also the only setting under which Swift and Python are
        /// expected to produce the *same* text. Both draw from the same
        /// distribution when sampling, but MLX's Python bindings carry a global
        /// random state and mlx-swift threads an explicit key, so the two draw
        /// different samples from identical logits. Chapter 08's equivalence
        /// test therefore compares greedy runs, and says so.
        public var temperature: Float
        public var topK: Int?
        public var seed: UInt64
        /// Stop when the model emits this id. `nil` runs to `maxTokens`.
        public var stopToken: Int?

        public init(
            maxTokens: Int = 64, temperature: Float = 0.8, topK: Int? = nil,
            seed: UInt64 = 0, stopToken: Int? = nil
        ) {
            self.maxTokens = maxTokens
            self.temperature = temperature
            self.topK = topK
            self.seed = seed
            self.stopToken = stopToken
        }
    }

    public struct Result: Sendable {
        public var tokenCount: Int
        public var promptTokenCount: Int
        /// Seconds spent generating, **not** counting the prompt pass. That is
        /// the number a tokens/sec figure has to be built from, because the
        /// prompt is one batched forward pass and the rest are one-token steps —
        /// averaging them together measures a mixture and reports it as a rate.
        public var generationSeconds: Double
        public var promptSeconds: Double
        public var peakMemoryBytes: Int

        public var tokensPerSecond: Double {
            generationSeconds > 0 ? Double(tokenCount) / generationSeconds : 0
        }
    }

    /// Generate from a prompt, calling `onToken` with each new fragment of text.
    ///
    /// `onToken` runs on whatever thread called this. It returns `false` to
    /// stop. The text it receives may be empty — that is a token whose bytes did
    /// not finish a character yet, and the next one will carry both.
    @discardableResult
    public func generate(
        prompt: String,
        options: Options = Options(),
        onToken: (_ text: String, _ id: Int) -> Bool = { _, _ in true }
    ) throws -> Result {
        var ids = tokenizer.encode(prompt)
        if ids.isEmpty {
            // An empty prompt still has to enter the model as something. Byte 0
            // is the first token of every byte-level vocabulary and carries no
            // learned meaning, which is the least misleading thing to start
            // from.
            ids = [0]
        }
        return try generate(promptIDs: ids, options: options, onToken: onToken)
    }

    @discardableResult
    public func generate(
        promptIDs: [Int],
        options: Options = Options(),
        onToken: (_ text: String, _ id: Int) -> Bool = { _, _ in true }
    ) throws -> Result {
        precondition(!promptIDs.isEmpty, "a prompt needs at least one token")

        GPU.resetPeakMemory()
        var key = MLXRandom.key(options.seed)
        var stream = ByteStream(tokenizer: tokenizer)

        let promptStart = Date()
        var (logits, cache) = try model(MLXArray(promptIDs.map { Int32($0) }).reshaped(1, promptIDs.count))
        eval(logits)
        let promptSeconds = Date().timeIntervalSince(promptStart)

        var produced = 0
        let generationStart = Date()
        for _ in 0..<options.maxTokens {
            let (nextKey, subKey) = MLXRandom.split(key: key)
            key = nextKey

            let token = sample(
                logits[0..., -1, 0...], temperature: options.temperature,
                topK: options.topK, key: subKey)
            // MLX is lazy. Without this the loop builds a graph `maxTokens` deep
            // and nothing is measured, streamed, or stopped until the end.
            eval(token)

            let id = token.item(Int.self)
            produced += 1
            if id == options.stopToken { break }
            if !onToken(stream.append(id), id) { break }

            (logits, cache) = try model(token.reshaped(1, 1), cache: cache)
        }
        let generationSeconds = Date().timeIntervalSince(generationStart)

        return Result(
            tokenCount: produced,
            promptTokenCount: promptIDs.count,
            generationSeconds: generationSeconds,
            promptSeconds: promptSeconds,
            peakMemoryBytes: Memory.peakMemory)
    }
}

/// Pick the next token. `temperature <= 0` means always take the most likely
/// one. Mirrors `sample()` in `02_model/model.py`.
public func sample(
    _ logits: MLXArray, temperature: Float, topK: Int? = nil, key: MLXArray? = nil
) -> MLXArray {
    if temperature <= 0 {
        return logits.argMax(axis: -1)
    }

    var logits = logits.asType(.float32)
    if let topK, topK > 0, topK < logits.dim(-1) {
        // Keep the k best scores and push the rest out of reach of the softmax.
        let kth = sorted(logits, axis: -1)[.ellipsis, (logits.dim(-1) - topK)...][
            .ellipsis, 0..<1]
        logits = which(logits .< kth, MLXArray(-Float.infinity), logits)
    }
    return MLXRandom.categorical(logits / temperature, key: key)
}

/// Turns a stream of token ids into a stream of *printable* text.
///
/// The model does not emit characters, it emits token ids, and a token is a run
/// of bytes that may end half way through a character. Decoding each token
/// independently would put a U+FFFD in the output for every multi-byte character
/// that spans a token boundary — permanently, because by the time the rest
/// arrives the replacement has already been shown.
///
/// So the tail is held. Bytes accumulate, everything that forms complete
/// characters is emitted, and whatever is left over waits for the next token.
struct ByteStream {
    /// The bytes a token stands for. A closure rather than the tokenizer itself,
    /// so the holding-back logic can be tested on hand-written byte sequences
    /// instead of on whatever a trained vocabulary happens to contain.
    private let bytes: (Int) -> [UInt8]?
    private var pending: [UInt8] = []

    init(tokenizer: BPETokenizer) {
        self.bytes = { tokenizer.bytes(for: $0) }
    }

    init(bytes: @escaping (Int) -> [UInt8]?) {
        self.bytes = bytes
    }

    /// The text that became printable when this token arrived. Often "", which
    /// is not an error.
    mutating func append(_ id: Int) -> String {
        pending += bytes(id) ?? []

        let cut = releaseBoundary()
        let ready = Array(pending[0..<cut])
        pending.removeFirst(cut)
        return String(decoding: ready, as: UTF8.self)
    }

    /// How many bytes at the front of `pending` are safe to show.
    ///
    /// Walk back from the end over at most three continuation bytes (`10xxxxxx`)
    /// to the byte that started the last sequence, and ask whether that sequence
    /// finished. If it did, everything goes out. If it did not, everything up to
    /// its lead byte goes out and the rest waits.
    ///
    /// Bytes that cannot start anything — a continuation byte with no lead in
    /// front of it — are released rather than held. Waiting for the rest of
    /// something that is not a character stalls the stream permanently, and
    /// showing U+FFFD is what the Python side's `errors="replace"` does too.
    private func releaseBoundary() -> Int {
        guard !pending.isEmpty else { return 0 }

        var lead = pending.count - 1
        var trailing = 0
        while lead > 0, trailing < 3, Self.isContinuation(pending[lead]) {
            lead -= 1
            trailing += 1
        }
        if Self.isContinuation(pending[lead]) { return pending.count }
        return Self.sequenceLength(pending[lead]) <= trailing + 1 ? pending.count : lead
    }

    private static func isContinuation(_ byte: UInt8) -> Bool {
        byte & 0b1100_0000 == 0b1000_0000
    }

    /// How many bytes a UTF-8 sequence starting with `lead` occupies. 1 for an
    /// ASCII byte and for anything malformed.
    private static func sequenceLength(_ lead: UInt8) -> Int {
        if lead & 0b1000_0000 == 0 { return 1 }
        if lead & 0b1110_0000 == 0b1100_0000 { return 2 }
        if lead & 0b1111_0000 == 0b1110_0000 { return 3 }
        if lead & 0b1111_1000 == 0b1111_0000 { return 4 }
        return 1
    }
}
