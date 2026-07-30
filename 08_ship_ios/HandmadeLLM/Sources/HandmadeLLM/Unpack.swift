import MLX

/// Chapter 08 — chapter 07's decoder, written out again in Swift.
///
/// MLX has `dequantized(_:scales:biases:groupSize:bits:)` and it is right here.
/// This file exists anyway, for the reason chapter 01 keeps a textbook BPE
/// trainer and chapter 02 keeps an explicit softmax: the readable version is
/// what a reader can check, the kernel is what ships, and a test asserting they
/// agree is what makes that an argument rather than a promise.
///
/// It also earns its place empirically. The first version of the Swift model
/// called the kernel for the embedding lookup while `07_quantize/quantize.py`
/// calls its own arithmetic, and the two agree to 3e-08 rather than exactly —
/// enough, after two layers and a tied output head, to move a logit by 2e-06.
/// Nothing downstream noticed: the argmax was the same and sixteen greedy tokens
/// were identical. But "the same logits" is a stronger and simpler claim than
/// "logits within 2e-06", and it costs twenty lines to be able to make it.
///
/// `docs/08-ship_ios.md` has the measurement.

/// Unpack `bits`-wide codes from uint32 words, low code in the low bits.
///
/// The layout is a **bit stream**, not codes-per-word. At 4 and 8 bits a code
/// sits inside one word; at 3, 5 and 6 it straddles two, and a decoder built on
/// "codes per word" cannot express those at all. Spreading each word into its
/// thirty-two bits, flattening, and refolding into groups of `bits` handles
/// every width with the same four lines. `docs/LESSONS.md` L11.
public func unpackCodes(_ packed: MLXArray, bits: Int) -> MLXArray {
    precondition((2...8).contains(bits), "bits must be between 2 and 8, got \(bits)")

    let leading = Array(packed.shape.dropLast())
    let totalBits = packed.dim(-1) * 32
    precondition(
        totalBits % bits == 0, "\(totalBits) bits does not divide into codes of \(bits) bits")

    let bit = MLXArray((0..<32).map { UInt32($0) })
    var stream = (packed.expandedDimensions(axis: -1) >> bit) & MLXArray(UInt32(1))
    stream = stream.reshaped(leading + [totalBits])

    let place = MLXArray((0..<bits).map { UInt32($0) })
    let spread = stream.reshaped(leading + [totalBits / bits, bits])
    return (spread << place).sum(axis: -1).asType(.uint8)
}

/// Rebuild float32 weights from codes and their per-group scale and offset.
///
/// One multiply and one add, which is the point: whatever the encoder did,
/// decoding has to be cheap enough to happen inside a matmul.
public func dequantizeGroupwise(
    codes: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int
) -> MLXArray {
    precondition(
        codes.dim(-1) % groupSize == 0,
        "last axis \(codes.dim(-1)) is not a multiple of \(groupSize)")

    let groups = codes.reshaped(-1, groupSize).asType(.float32)
    let scale = scales.reshaped(-1, 1).asType(.float32)
    let bias = biases.reshaped(-1, 1).asType(.float32)
    return (groups * scale + bias).reshaped(codes.shape)
}
