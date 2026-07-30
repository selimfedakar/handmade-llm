import Foundation
import MLX
import MLXFast
import MLXNN

/// Chapter 08 — chapter 02's transformer, in Swift.
///
/// Line for line the same network: RMSNorm, RoPE, grouped-query attention,
/// SwiGLU, pre-norm residuals, tied embeddings. The comments that explain *why*
/// each piece is there live in `02_model/model.py` and are not copied; what is
/// written down here is only what is different because this is Swift, or
/// because the weights arrive already quantized.
///
/// Attention is implemented twice, as it is in chapter 02 — once written out
/// with an explicit softmax, once through MLX's fused kernel — and
/// `Tests/ModelTests.swift` asserts the two agree. Keeping that pair alive in a
/// second language is the point: the readable version is the one you read on
/// either side of the bridge.

// -- shapes -----------------------------------------------------------------

/// The architecture, exactly as `meta.json` records it. Field names are the
/// Python dataclass's, converted from snake case.
public struct ModelConfig: Codable, Equatable, Sendable {
    public var vocabSize: Int
    public var dModel: Int
    public var nLayers: Int
    public var nHeads: Int
    public var nKvHeads: Int
    public var dFf: Int
    public var maxSeqLen: Int
    public var ropeBase: Float
    public var normEps: Float
    public var tieEmbeddings: Bool
    public var fusedAttention: Bool

    public var headDim: Int { dModel / nHeads }
}

/// How the weights were quantized, from the same file. Absent for a float32
/// checkpoint.
public struct Quantization: Codable, Equatable, Sendable {
    public var scheme: String
    public var bits: Int
    public var groupSize: Int
    /// Whether the token table was quantized too.
    public var embedding: Bool
}

/// The keys and values a layer has already seen.
///
/// Generation feeds one token at a time and the cache supplies all the history.
/// Without it, producing n tokens costs a full forward pass over the whole
/// sequence n times, which on a phone is the difference between a demo and a
/// progress bar.
public struct KVCache {
    public var keys: MLXArray
    public var values: MLXArray

    /// How many positions are already stored — which is where the next chunk
    /// starts, and therefore the rotation offset RoPE needs.
    public var length: Int { keys.dim(2) }
}

// -- pieces -----------------------------------------------------------------

/// Scale each vector to unit root-mean-square, then apply a learned gain.
struct RMSNorm {
    let weight: MLXArray
    let eps: Float

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Float32 regardless of the input dtype: a sum of squares over a long
        // vector is exactly the operation that leaves float16's range.
        let f = x.asType(.float32)
        let scale = rsqrt((f * f).mean(axis: -1, keepDims: true) + eps)
        return (f * scale).asType(x.dtype) * weight
    }
}

/// Rotary position embedding, written out — the half-split convention, matching
/// `rope()` in `02_model/model.py`.
///
/// `MLXFast.RoPE` is right here and is not used. It offers two pairing
/// conventions and the one it calls "traditional" is not the one chapter 02
/// writes; picking the wrong one produces a model that runs, generates fluent
/// nonsense, and is wrong in a way no shape check will ever catch. Writing the
/// four lines out means the convention is visible instead of selected by a
/// boolean.
///
/// `x` is `(batch, heads, seq, headDim)`. `offset` is where this chunk starts in
/// the full sequence, which is what makes a cached step line up with a full
/// forward pass.
func rope(_ x: MLXArray, offset: Int, base: Float) -> MLXArray {
    let seqLen = x.dim(2)
    let headDim = x.dim(3)
    let half = headDim / 2

    let positions = MLXArray((0..<seqLen).map { Float(offset + $0) }).reshaped(seqLen, 1)
    // freq_i = base ** (-i / half): a geometric spread of rotation speeds.
    let idx = MLXArray((0..<half).map { Float($0) })
    let invFreq = exp(-log(base) * idx / Float(half)).reshaped(1, half)

    let angles = positions * invFreq  // (seq, half)
    let cosines = cos(angles)
    let sines = sin(angles)

    // The two halves of the vector are the two coordinates of each pair.
    let x1 = x[.ellipsis, 0..<half]
    let x2 = x[.ellipsis, half...]
    let rotated = concatenated(
        [x1 * cosines - x2 * sines, x2 * cosines + x1 * sines], axis: -1)
    return rotated.asType(x.dtype)
}

/// An additive mask: 0 where a query may look, -infinity where it may not.
///
/// During generation `nQueries` is 1 while `nKeys` keeps growing, so the query
/// sits at the **end** of the key range. Get that alignment wrong and nothing
/// raises — shapes stay valid, quality quietly degrades. `docs/LESSONS.md` L3.
func causalMask(nQueries: Int, nKeys: Int, dtype: DType) -> MLXArray {
    let qPos = MLXArray((nKeys - nQueries..<nKeys).map { Int32($0) }).reshaped(nQueries, 1)
    let kPos = MLXArray((0..<nKeys).map { Int32($0) }).reshaped(1, nKeys)
    return which(kPos .<= qPos, MLXArray(Float(0)), MLXArray(-Float.infinity)).asType(dtype)
}

/// Grouped-query attention with rotary positions and a cache.
struct Attention {
    let config: ModelConfig
    let qProj: Projection
    let kProj: Projection
    let vProj: Projection
    let oProj: Projection

    var scale: Float { pow(Float(config.headDim), -0.5) }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cache: KVCache?
    ) -> (MLXArray, KVCache) {
        let batch = x.dim(0)
        let seqLen = x.dim(1)

        func splitHeads(_ t: MLXArray, _ heads: Int) -> MLXArray {
            t.reshaped(batch, seqLen, heads, config.headDim).transposed(0, 2, 1, 3)
        }

        var queries = splitHeads(qProj(x), config.nHeads)
        var keys = splitHeads(kProj(x), config.nKvHeads)
        var values = splitHeads(vProj(x), config.nKvHeads)

        let offset = cache?.length ?? 0
        queries = rope(queries, offset: offset, base: config.ropeBase)
        keys = rope(keys, offset: offset, base: config.ropeBase)

        if let cache {
            keys = concatenated([cache.keys, keys], axis: 2)
            values = concatenated([cache.values, values], axis: 2)
        }
        let newCache = KVCache(keys: keys, values: values)

        let out: MLXArray
        if config.fusedAttention {
            out = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        } else {
            out = writtenOut(queries: queries, keys: keys, values: values, mask: mask)
        }

        let merged = out.transposed(0, 2, 1, 3).reshaped(batch, seqLen, -1)
        return (oProj(merged), newCache)
    }

    /// The same operation, spelled out. Slower, and the one to read.
    func writtenOut(
        queries: MLXArray, keys: MLXArray, values: MLXArray, mask: MLXArray?
    ) -> MLXArray {
        // Each key/value head serves several query heads, so repeat the KV heads
        // to line up. The fused kernel does this without materialising anything.
        var keys = keys
        var values = values
        let repeats = config.nHeads / config.nKvHeads
        if repeats > 1 {
            keys = repeated(keys, count: repeats, axis: 1)
            values = repeated(values, count: repeats, axis: 1)
        }

        var scores = matmul(queries * scale, keys.transposed(0, 1, 3, 2))
        if let mask { scores = scores + mask }
        // Softmax in float32: exponentials of attention logits leave float16's
        // range far more easily than people expect.
        let weights = softmax(scores.asType(.float32), axis: -1).asType(values.dtype)
        return matmul(weights, values)
    }
}

/// A gated feed-forward block.
struct SwiGLU {
    let gateProj: Projection
    let upProj: Projection
    let downProj: Projection

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

/// One transformer layer: attention, then feed-forward, both pre-normed.
struct Block {
    let attnNorm: RMSNorm
    let attn: Attention
    let ffnNorm: RMSNorm
    let ffn: SwiGLU

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cache: KVCache?
    ) -> (MLXArray, KVCache) {
        let (attnOut, newCache) = attn(attnNorm(x), mask: mask, cache: cache)
        var x = x + attnOut
        x = x + ffn(ffnNorm(x))
        return (x, newCache)
    }
}

// -- the model ---------------------------------------------------------------

/// Token ids in, logits over the vocabulary out.
public final class Transformer {
    public let config: ModelConfig
    public let quantization: Quantization?

    let embed: TokenEmbedding
    let layers: [Block]
    let norm: RMSNorm
    /// `nil` when the embedding is tied, which it is for every model this
    /// repository trains.
    let lmHead: Projection?

    init(
        config: ModelConfig, quantization: Quantization?, embed: TokenEmbedding,
        layers: [Block], norm: RMSNorm, lmHead: Projection?
    ) {
        self.config = config
        self.quantization = quantization
        self.embed = embed
        self.layers = layers
        self.norm = norm
        self.lmHead = lmHead
    }

    public enum ForwardError: Error, CustomStringConvertible {
        case sequenceTooLong(length: Int, limit: Int)

        public var description: String {
            switch self {
            case .sequenceTooLong(let length, let limit):
                return "sequence of \(length) exceeds maxSeqLen \(limit)"
            }
        }
    }

    /// One forward pass. `cache` is `nil` for the prompt and the previous step's
    /// cache for everything after it.
    public func callAsFunction(
        _ ids: MLXArray, cache: [KVCache]? = nil
    ) throws -> (logits: MLXArray, cache: [KVCache]) {
        let ids = ids.ndim == 1 ? ids.reshaped(1, ids.dim(0)) : ids
        let seqLen = ids.dim(1)

        let past = cache?.first?.length ?? 0
        guard past + seqLen <= config.maxSeqLen else {
            throw ForwardError.sequenceTooLong(length: past + seqLen, limit: config.maxSeqLen)
        }

        var x = embed(ids)

        // A single query attends to everything before it, so token-by-token
        // generation needs no mask at all.
        let mask = seqLen > 1
            ? causalMask(nQueries: seqLen, nKeys: past + seqLen, dtype: x.dtype)
            : nil

        var newCache: [KVCache] = []
        newCache.reserveCapacity(layers.count)
        for (i, layer) in layers.enumerated() {
            let (next, layerCache) = layer(x, mask: mask, cache: cache?[i])
            x = next
            newCache.append(layerCache)
        }

        x = norm(x)
        let logits = lmHead.map { $0(x) } ?? embed.asLinear(x)
        return (logits, newCache)
    }

    /// Bytes of weights held, counted the way `07_quantize/quantize.py` counts
    /// them so the two chapters' numbers are comparable.
    public var weightByteCount: Int {
        var total = embed.byteCount + norm.weight.nbytes + (lmHead?.byteCount ?? 0)
        for layer in layers {
            total += layer.attnNorm.weight.nbytes + layer.ffnNorm.weight.nbytes
            total += layer.attn.qProj.byteCount + layer.attn.kProj.byteCount
            total += layer.attn.vProj.byteCount + layer.attn.oProj.byteCount
            total += layer.ffn.gateProj.byteCount + layer.ffn.upProj.byteCount
            total += layer.ffn.downProj.byteCount
        }
        return total
    }

    /// A copy of this model with the other attention implementation selected.
    /// Used only by the equivalence test — the weights are shared, not copied.
    func withFusedAttention(_ fused: Bool) -> Transformer {
        var config = self.config
        config.fusedAttention = fused
        let layers = self.layers.map { layer in
            Block(
                attnNorm: layer.attnNorm,
                attn: Attention(
                    config: config, qProj: layer.attn.qProj, kProj: layer.attn.kProj,
                    vProj: layer.attn.vProj, oProj: layer.attn.oProj),
                ffnNorm: layer.ffnNorm, ffn: layer.ffn)
        }
        return Transformer(
            config: config, quantization: quantization, embed: embed,
            layers: layers, norm: norm, lmHead: lmHead)
    }
}
