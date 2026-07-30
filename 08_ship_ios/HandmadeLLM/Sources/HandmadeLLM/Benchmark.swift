import Foundation
import MLX

/// Chapter 08 — measuring two models against each other without lying to
/// yourself.
///
/// Chapter 07 asked whether the 4-bit model generates faster than the float32
/// one, answered "no measurable difference", and was wrong. The answer was wrong
/// because the *instrument* was: the two models were timed one after the other,
/// on a laptop whose thermal and allocator state moves between one measurement
/// and the next, so whichever model went second was measured somewhere else.
/// Two warm-ups and a median control the noise inside one measurement and do
/// nothing at all about drift between two. `docs/LESSONS.md` L13.
///
/// A phone is a worse offender, not a better one. It throttles harder, it has
/// less thermal mass, and it will happily hand you a 30% swing between the first
/// and the fifth run of the same code. So the same discipline is built into the
/// type here rather than left to whoever writes the benchmark screen:
///
///  * both models resident at once,
///  * one round each, alternating, inside one loop,
///  * warm-up rounds discarded,
///  * medians, never means,
///  * and `flipOrder` so the claim can be checked against the loop itself.
///
/// The honest form of the result is the **consistency of the sign** — how many
/// rounds one model won — not the size of the gap. `roundsBAhead` is reported
/// for that reason and is the number to read first.
public enum Benchmark {

    public struct PairedResult: Sendable {
        public var labelA: String
        public var labelB: String
        /// Tokens per second for each recorded round, in round order.
        public var samplesA: [Double]
        public var samplesB: [Double]
        /// Peak GPU memory for the **process**, both models resident — not for
        /// either model on its own, and there is no way to make it be that here.
        ///
        /// `Memory.peakMemory` is a process-wide counter, and the whole design of
        /// this measurement is that both sets of weights stay loaded at once so
        /// the thermal drift lands on both equally. Those two facts are not
        /// compatible: the arrangement that makes the *speed* comparison honest
        /// is the same arrangement that makes a *memory* comparison impossible.
        /// Reporting one number instead of two is the honest form. For per-model
        /// peak memory, generate with one model loaded — that is what the app's
        /// single-run readout does. `docs/LESSONS.md` L18.
        public var peakMemoryBytes: Int
        /// Whether B was generated first inside each round.
        public var bFirst: Bool

        public var medianA: Double { Benchmark.median(samplesA) }
        public var medianB: Double { Benchmark.median(samplesB) }
        public var ratio: Double { medianA > 0 ? medianB / medianA : 0 }

        /// How many rounds B beat A in. This is the claim; the ratio is colour.
        public var roundsBAhead: Int {
            zip(samplesA, samplesB).reduce(0) { $0 + ($1.1 > $1.0 ? 1 : 0) }
        }
        public var rounds: Int { min(samplesA.count, samplesB.count) }

        public var summary: String {
            let sign = roundsBAhead == rounds
                ? "\(labelB) ahead in all \(rounds)"
                : roundsBAhead == 0
                    ? "\(labelA) ahead in all \(rounds)"
                    : "\(labelB) ahead in \(roundsBAhead) of \(rounds) — no consistent sign"
            return String(
                format: "%@ %.1f vs %@ %.1f tok/s (%.3fx) · %@ · %d MiB peak, both resident · %@ first",
                labelA, medianA, labelB, medianB, ratio, sign,
                peakMemoryBytes / 1_048_576,
                bFirst ? labelB : labelA)
        }
    }

    /// Time two models in one alternating loop.
    ///
    /// Both generate greedily from the same prompt for the same number of
    /// tokens, so the only difference between them is the weights.
    public static func paired(
        _ a: Generator, label labelA: String,
        against b: Generator, label labelB: String,
        promptIDs: [Int],
        tokens: Int = 64,
        rounds: Int = 7,
        warmupRounds: Int = 2,
        bFirst: Bool = false,
        onRound: ((Int, Int) -> Void)? = nil
    ) throws -> PairedResult {
        let options = Generator.Options(maxTokens: tokens, temperature: 0)

        var samplesA: [Double] = []
        var samplesB: [Double] = []
        // One counter, not two. See `PairedResult.peakMemoryBytes`: both models
        // are resident for the whole loop, so every reading is the same
        // process-wide number wearing whichever model's name went last.
        var peak = 0

        for round in 0..<(warmupRounds + rounds) {
            // Warm-ups are inside the same loop as the measurements, not before
            // it, so the first recorded round is preceded by exactly the same
            // work as every other one.
            let recording = round >= warmupRounds
            onRound?(round + 1, warmupRounds + rounds)

            func run(_ generator: Generator) throws -> Generator.Result {
                try generator.generate(promptIDs: promptIDs, options: options)
            }

            if bFirst {
                let rb = try run(b)
                let ra = try run(a)
                if recording {
                    samplesB.append(rb.tokensPerSecond)
                    samplesA.append(ra.tokensPerSecond)
                    peak = max(peak, max(rb.peakMemoryBytes, ra.peakMemoryBytes))
                }
            } else {
                let ra = try run(a)
                let rb = try run(b)
                if recording {
                    samplesA.append(ra.tokensPerSecond)
                    samplesB.append(rb.tokensPerSecond)
                    peak = max(peak, max(ra.peakMemoryBytes, rb.peakMemoryBytes))
                }
            }
        }

        return PairedResult(
            labelA: labelA, labelB: labelB,
            samplesA: samplesA, samplesB: samplesB,
            peakMemoryBytes: peak,
            bFirst: bFirst)
    }

    /// The median. Not the mean — the first step into a new working set can be
    /// fifty times slow, and a mean lets that one sample decide the answer.
    /// `docs/LESSONS.md` L6.
    public static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        let mid = sorted.count / 2
        return sorted.count % 2 == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2
    }
}
