import Foundation
import MLX

/// Chapter 08 — reading what chapter 07 wrote.
///
/// `07_quantize/compare.py` writes a directory with two files in it:
///
/// ```
/// runs/latest/quantized/
///   weights.safetensors    14.9 MiB of packed codes, scales and offsets
///   meta.json              the architecture, the bit width, the group size
/// ```
///
/// Nothing else is needed to rebuild the model, and that was a decision made on
/// the Python side specifically so that this file could exist: a checkpoint that
/// needs a human to remember how it was made is not a checkpoint. The same
/// loader reads chapter 03's float32 directory, which has the identical
/// `meta.json` shape minus the `quantization` block.
///
/// The weight names are flat, exactly as `tree_flatten` produced them —
/// `layers.3.attn.q_proj.weight`, `embed.scales`, `norm.weight`. So the loader
/// walks the architecture and asks for the names it expects, rather than
/// rebuilding a module tree in Swift to pour a dictionary into. If a name is
/// missing, it says which one.

public enum CheckpointError: Error, CustomStringConvertible {
    case missingWeight(String)
    case malformedMetadata(String)
    case unsupportedScheme(String)

    public var description: String {
        switch self {
        case .missingWeight(let name):
            return "the checkpoint has no weight named \"\(name)\""
        case .malformedMetadata(let what):
            return "meta.json: \(what)"
        case .unsupportedScheme(let scheme):
            return """
                meta.json describes quantization scheme "\(scheme)"; this loader implements \
                "affine", which is what 07_quantize writes
                """
        }
    }
}

/// Everything `meta.json` carries.
public struct CheckpointMetadata: Decodable, Sendable {
    public var step: Int?
    public var modelConfig: ModelConfig
    public var quantization: Quantization?
}

extension Transformer {

    /// Load a model from a directory written by `07_quantize` or `03_train`.
    ///
    /// The dtype of what comes back is whatever is on disk. A quantized
    /// checkpoint arrives as `uint32` codes with float32 scales, and stays that
    /// way — it is never widened, because widening it is precisely the thing
    /// chapter 07 exists to avoid.
    public static func load(directory: URL) throws -> Transformer {
        try load(
            weights: directory.appendingPathComponent("weights.safetensors"),
            metadata: directory.appendingPathComponent("meta.json"))
    }

    /// The same, from two files that need not sit in a directory together.
    ///
    /// An app bundle is flat: `08_ship_ios/bundle_model.py` copies the pair in
    /// as `model-quantized.safetensors` and `model-quantized.json`, so the app
    /// asks for them by name rather than for a folder that iOS would have to be
    /// persuaded to keep as a folder.
    public static func load(weights: URL, metadata: URL) throws -> Transformer {
        try Transformer(
            metadata: try loadMetadata(url: metadata),
            weights: try loadArrays(url: weights))
    }

    public static func loadMetadata(directory: URL) throws -> CheckpointMetadata {
        try loadMetadata(url: directory.appendingPathComponent("meta.json"))
    }

    public static func loadMetadata(url: URL) throws -> CheckpointMetadata {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(CheckpointMetadata.self, from: try Data(contentsOf: url))
    }

    public convenience init(metadata: CheckpointMetadata, weights: [String: MLXArray]) throws {
        let config = metadata.modelConfig
        let quantization = metadata.quantization

        if let quantization, quantization.scheme != "affine" {
            throw CheckpointError.unsupportedScheme(quantization.scheme)
        }
        if let quantization, config.dModel % quantization.groupSize != 0 {
            // The same rule chapter 07 enforces on the way out, checked again on
            // the way in: a group size has to divide every matrix in the model,
            // and `d_ff` is where it fails first.
            throw CheckpointError.malformedMetadata(
                "d_model \(config.dModel) is not a multiple of group size \(quantization.groupSize)")
        }

        func norm(_ prefix: String) throws -> RMSNorm {
            guard let weight = weights["\(prefix).weight"] else {
                throw CheckpointError.missingWeight("\(prefix).weight")
            }
            return RMSNorm(weight: weight, eps: config.normEps)
        }

        var layers: [Block] = []
        layers.reserveCapacity(config.nLayers)
        for i in 0..<config.nLayers {
            let attn = "layers.\(i).attn"
            let ffn = "layers.\(i).ffn"
            layers.append(
                Block(
                    attnNorm: try norm("layers.\(i).attn_norm"),
                    attn: Attention(
                        config: config,
                        qProj: try Projection.load(weights, "\(attn).q_proj", quantization),
                        kProj: try Projection.load(weights, "\(attn).k_proj", quantization),
                        vProj: try Projection.load(weights, "\(attn).v_proj", quantization),
                        oProj: try Projection.load(weights, "\(attn).o_proj", quantization)),
                    ffnNorm: try norm("layers.\(i).ffn_norm"),
                    ffn: SwiGLU(
                        gateProj: try Projection.load(weights, "\(ffn).gate_proj", quantization),
                        upProj: try Projection.load(weights, "\(ffn).up_proj", quantization),
                        downProj: try Projection.load(weights, "\(ffn).down_proj", quantization))))
        }

        // The embedding may be quantized while the rest of the model is, or left
        // in float32 by `--keep-embedding-float32`. `Projection.load` decides
        // from the file rather than from the flag, so a checkpoint whose
        // `meta.json` and weights disagree loads as its weights, not as its
        // claim.
        let embed = try TokenEmbedding.load(weights, "embed", quantization)
        let lmHead: Projection? =
            config.tieEmbeddings ? nil : try Projection.load(weights, "lm_head", quantization)

        self.init(
            config: config, quantization: quantization, embed: embed,
            layers: layers, norm: try norm("norm"), lmHead: lmHead)
    }
}
