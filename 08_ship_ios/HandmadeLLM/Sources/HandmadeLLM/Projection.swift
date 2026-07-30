import MLX

/// Chapter 08 — one matrix, stored two ways.
///
/// Chapter 07 ends with a directory holding 14.9 MiB of packed 4-bit codes.
/// Chapter 03 ends with a directory holding 95 MiB of float32. Both are the same
/// model, and the phone should be able to run either — not because anyone wants
/// to ship the big one, but because the question this chapter exists to answer
/// is *how much faster is the small one on a phone*, and you cannot answer that
/// with only one of them present.
///
/// So every matrix in the model goes through this type, and the difference
/// between the two builds is which case it holds.
public enum Projection {
    /// A plain float32 matrix, `(outputs, inputs)`, applied as `x @ wᵀ`.
    case dense(weight: MLXArray)

    /// Packed codes plus one scale and one offset per group of `groupSize`
    /// weights, applied through `quantizedMM` — which unpacks each group inside
    /// the kernel, one tile at a time.
    ///
    /// That is the distinction chapter 07 drew and it is the whole point of
    /// shipping quantized: dequantizing the matrix and then multiplying gets the
    /// same numbers while rebuilding the float32 array you were trying not to
    /// hold. This case never materialises it.
    case quantized(
        weight: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int, bits: Int)

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        switch self {
        case .dense(let weight):
            return matmul(x, weight.transposed())
        case .quantized(let weight, let scales, let biases, let groupSize, let bits):
            return quantizedMM(
                x, weight, scales: scales, biases: biases,
                transpose: true, groupSize: groupSize, bits: bits)
        }
    }

    /// Bytes actually held. A packed array counts as its four bytes per word,
    /// not as the sixteen weights it stands for — the same rule
    /// `07_quantize/quantize.py` counts by, so the two chapters' size numbers
    /// mean the same thing.
    public var byteCount: Int {
        switch self {
        case .dense(let weight):
            return weight.nbytes
        case .quantized(let weight, let scales, let biases, _, _):
            return weight.nbytes + scales.nbytes + biases.nbytes
        }
    }

    /// Read `prefix.weight` and, if they are there, `prefix.scales` and
    /// `prefix.biases`. The presence of the scales is what says the checkpoint
    /// is quantized; nothing else has to be passed down the tree.
    static func load(
        _ weights: [String: MLXArray], _ prefix: String, _ quantization: Quantization?
    ) throws -> Projection {
        guard let weight = weights["\(prefix).weight"] else {
            throw CheckpointError.missingWeight("\(prefix).weight")
        }
        guard let scales = weights["\(prefix).scales"] else {
            return .dense(weight: weight)
        }
        guard let biases = weights["\(prefix).biases"] else {
            throw CheckpointError.missingWeight("\(prefix).biases")
        }
        guard let quantization else {
            throw CheckpointError.malformedMetadata(
                "\(prefix) has scales but meta.json describes no quantization")
        }
        return .quantized(
            weight: weight, scales: scales, biases: biases,
            groupSize: quantization.groupSize, bits: quantization.bits)
    }
}

/// The token table, which on this model is also the output head.
///
/// Chapter 02 ties them: one array maps ids to vectors going in and vectors to
/// scores coming out. So there is one matrix here with two call sites, and
/// quantizing it shrinks one array while speeding up two operations.
public enum TokenEmbedding {
    case dense(weight: MLXArray)
    case quantized(
        weight: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int, bits: Int)

    /// Ids in, vectors out.
    ///
    /// The quantized path dequantizes **only the rows the batch asked for** —
    /// sixteen rows out of four thousand, decoded on the way past. Dequantizing
    /// the table and then indexing it would give the same answer and defeat the
    /// entire chapter.
    ///
    /// The decoding is `unpackCodes` + `dequantizeGroupwise` rather than MLX's
    /// `dequantized` kernel, and that is deliberate: it is the arithmetic
    /// `07_quantize/quantize.py` performs, in the same order, so the logits come
    /// out **bit-identical** to Python's instead of within 2e-06 of them.
    /// `ModelTests` asserts this path agrees with the kernel as well, which is
    /// the same pair of claims chapter 01 makes about its two BPE trainers.
    public func callAsFunction(_ ids: MLXArray) -> MLXArray {
        switch self {
        case .dense(let weight):
            return weight[ids]
        case .quantized(let weight, let scales, let biases, let groupSize, let bits):
            return dequantizeGroupwise(
                codes: unpackCodes(weight[ids], bits: bits),
                scales: scales[ids], biases: biases[ids], groupSize: groupSize)
        }
    }

    /// The same lookup through MLX's fused decoder. Only the equivalence test
    /// calls it; see the note on `callAsFunction`.
    func lookupThroughKernel(_ ids: MLXArray) -> MLXArray {
        switch self {
        case .dense(let weight):
            return weight[ids]
        case .quantized(let weight, let scales, let biases, let groupSize, let bits):
            return dequantized(
                weight[ids], scales: scales[ids], biases: biases[ids],
                groupSize: groupSize, bits: bits)
        }
    }

    /// The same weights read as a projection to the vocabulary.
    public func asLinear(_ x: MLXArray) -> MLXArray {
        switch self {
        case .dense(let weight):
            return matmul(x, weight.transposed())
        case .quantized(let weight, let scales, let biases, let groupSize, let bits):
            return quantizedMM(
                x, weight, scales: scales, biases: biases,
                transpose: true, groupSize: groupSize, bits: bits)
        }
    }

    public var byteCount: Int {
        switch self {
        case .dense(let weight):
            return weight.nbytes
        case .quantized(let weight, let scales, let biases, _, _):
            return weight.nbytes + scales.nbytes + biases.nbytes
        }
    }

    static func load(
        _ weights: [String: MLXArray], _ prefix: String, _ quantization: Quantization?
    ) throws -> TokenEmbedding {
        switch try Projection.load(weights, prefix, quantization) {
        case .dense(let weight):
            return .dense(weight: weight)
        case .quantized(let weight, let scales, let biases, let groupSize, let bits):
            return .quantized(
                weight: weight, scales: scales, biases: biases,
                groupSize: groupSize, bits: bits)
        }
    }
}
