import Foundation
import HandmadeLLM
import MLX

/// Everything the screen needs, and the one rule that keeps it from being a
/// concurrency exercise: **the model is only ever touched from one background
/// task.**
///
/// `MLXArray` is a handle onto a buffer the GPU also holds, so it is not
/// `Sendable` and two tasks generating at once is not a thing that works. Rather
/// than argue with the compiler about actor isolation, generation runs inside a
/// single `Task` that this class owns and cancels; every value that crosses back
/// to the UI is a `String`, an `Int` or a `Double`.
@MainActor
final class ModelStore: ObservableObject {

    enum Weights: String, CaseIterable, Identifiable {
        case quantized = "4-bit"
        case float32 = "float32"

        var id: String { rawValue }
        var resourceStem: String {
            switch self {
            case .quantized: return "model-quantized"
            case .float32: return "model-float32"
            }
        }
    }

    enum Phase: Equatable {
        case loading
        case ready
        case generating
        case benchmarking(round: Int, of: Int)
        case failed(String)
    }

    @Published private(set) var phase: Phase = .loading
    @Published private(set) var output = ""
    @Published private(set) var tokensPerSecond: Double?
    @Published private(set) var peakMemoryBytes: Int?
    @Published private(set) var weightBytes: [Weights: Int] = [:]
    @Published private(set) var benchmark: String?
    @Published var weights: Weights = .quantized {
        didSet { if weights != oldValue { reset() } }
    }
    @Published var prompt = "First Citizen:"

    /// Both models stay loaded once they have been loaded, because the paired
    /// measurement needs them resident at the same time — timing one and then
    /// the other is what chapter 07 got wrong. `docs/LESSONS.md` L13.
    private var generators: [Weights: Generator] = [:]
    private var tokenizer: BPETokenizer?
    private var work: Task<Void, Never>?

    var isBusy: Bool {
        switch phase {
        case .generating, .benchmarking, .loading: return true
        case .ready, .failed: return false
        }
    }

    var availableWeights: [Weights] {
        Weights.allCases.filter { generators[$0] != nil }
    }

    // -- loading ---------------------------------------------------------------

    func load() {
        guard case .loading = phase else { return }
        work = Task { [weak self] in
            do {
                let loaded = try Self.loadEverything()
                await MainActor.run {
                    guard let self else { return }
                    self.tokenizer = loaded.tokenizer
                    self.generators = loaded.generators
                    self.weightBytes = loaded.weightBytes
                    if loaded.generators[self.weights] == nil,
                        let first = Weights.allCases.first(where: { loaded.generators[$0] != nil })
                    {
                        self.weights = first
                    }
                    self.phase = loaded.generators.isEmpty
                        ? .failed(
                            """
                            No model in the bundle. From the repository root:
                              python 07_quantize/compare.py
                              python 08_ship_ios/bundle_model.py
                            """)
                        : .ready
                }
            } catch {
                await MainActor.run { self?.phase = .failed("\(error)") }
            }
        }
    }

    private struct Loaded {
        var tokenizer: BPETokenizer
        var generators: [Weights: Generator]
        var weightBytes: [Weights: Int]
    }

    private nonisolated static func loadEverything() throws -> Loaded {
        guard let tokenizerURL = Bundle.main.url(forResource: "tokenizer", withExtension: "json")
        else {
            throw Failure.noTokenizer
        }
        let tokenizer = try BPETokenizer(contentsOf: tokenizerURL)

        var generators: [Weights: Generator] = [:]
        var bytes: [Weights: Int] = [:]
        for weights in Weights.allCases {
            guard
                let safetensors = Bundle.main.url(
                    forResource: weights.resourceStem, withExtension: "safetensors"),
                let metadata = Bundle.main.url(
                    forResource: weights.resourceStem, withExtension: "json")
            else { continue }

            let model = try Transformer.load(weights: safetensors, metadata: metadata)
            // The vocabulary rule, on the phone: an id the embedding has no row
            // for does not raise in MLX, it returns a number that changes with
            // the batch shape. `docs/LESSONS.md` L7.
            guard model.config.vocabSize == tokenizer.vocabularySize else {
                throw Failure.vocabularyMismatch(
                    model: model.config.vocabSize, tokenizer: tokenizer.vocabularySize)
            }
            generators[weights] = Generator(model: model, tokenizer: tokenizer)
            bytes[weights] = model.weightByteCount
        }
        return Loaded(tokenizer: tokenizer, generators: generators, weightBytes: bytes)
    }

    enum Failure: Error, CustomStringConvertible {
        case noTokenizer
        case vocabularyMismatch(model: Int, tokenizer: Int)

        var description: String {
            switch self {
            case .noTokenizer:
                return "No tokenizer.json in the bundle. Run python 08_ship_ios/bundle_model.py"
            case .vocabularyMismatch(let model, let tokenizer):
                return """
                    The bundled model has \(model) rows and the bundled tokenizer has \
                    \(tokenizer) tokens. They were not built together.
                    """
            }
        }
    }

    // -- generating ------------------------------------------------------------

    func generate(maxTokens: Int = 96, temperature: Float = 0.8) {
        guard case .ready = phase, let generator = generators[weights] else { return }
        let prompt = prompt
        reset()
        phase = .generating

        work = Task { [weak self] in
            let started = Date()
            var result: Generator.Result?
            var failure: String?
            do {
                result = try generator.generate(
                    prompt: prompt,
                    options: .init(
                        maxTokens: maxTokens, temperature: temperature, topK: 40,
                        seed: UInt64(started.timeIntervalSince1970),
                        stopToken: generator.tokenizer.specialTokens.first?.id)
                ) { text, _ in
                    if Task.isCancelled { return false }
                    if !text.isEmpty {
                        Task { @MainActor in self?.output += text }
                    }
                    return true
                }
            } catch {
                failure = "\(error)"
            }

            await MainActor.run {
                guard let self else { return }
                if let failure {
                    self.phase = .failed(failure)
                } else if let result {
                    self.tokensPerSecond = result.tokensPerSecond
                    self.peakMemoryBytes = result.peakMemoryBytes
                    self.phase = .ready
                }
            }
        }
    }

    /// The measurement chapter 07 could not make: the two models, on a phone, in
    /// one alternating loop.
    ///
    /// Not one after the other. A phone throttles harder than a laptop and has
    /// less thermal mass to hide it, so measuring one model and then the other
    /// measures two different machines. `Benchmark.paired` is built so this
    /// cannot be done the wrong way from here.
    func runPairedBenchmark(rounds: Int = 7, tokens: Int = 64) {
        guard case .ready = phase,
            let quantized = generators[.quantized], let float32 = generators[.float32],
            let tokenizer
        else {
            benchmark = "Both models have to be in the bundle. Re-run bundle_model.py without --quantized-only."
            return
        }

        let promptIDs = tokenizer.encode(prompt).isEmpty ? [0] : tokenizer.encode(prompt)
        phase = .benchmarking(round: 0, of: rounds + 2)
        benchmark = nil

        work = Task { [weak self] in
            var lines: [String] = []
            do {
                // Both orders, because the honest form of the claim is that the
                // sign survives the loop being turned around.
                for bFirst in [false, true] {
                    let result = try Benchmark.paired(
                        float32, label: "float32",
                        against: quantized, label: "4-bit",
                        promptIDs: promptIDs, tokens: tokens, rounds: rounds,
                        bFirst: bFirst
                    ) { round, total in
                        Task { @MainActor in self?.phase = .benchmarking(round: round, of: total) }
                    }
                    lines.append(result.summary)
                }
            } catch {
                lines = ["\(error)"]
            }
            await MainActor.run {
                self?.benchmark = lines.joined(separator: "\n\n")
                self?.phase = .ready
            }
        }
    }

    func cancel() {
        work?.cancel()
        work = nil
        if case .failed = phase {} else { phase = .ready }
    }

    private func reset() {
        output = ""
        tokensPerSecond = nil
        peakMemoryBytes = nil
        benchmark = nil
    }
}
