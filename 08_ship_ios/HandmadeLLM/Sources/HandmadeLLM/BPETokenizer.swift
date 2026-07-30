import Foundation

/// Chapter 08 — chapter 01's byte-level BPE tokenizer, in Swift.
///
/// This is a port, not a reimplementation. Every decision here was already made
/// in `01_tokenizer/bpe.py`; the job of this file is to make the same decisions
/// again in a different language and then prove it, which is what
/// `Tests/TokenizerTests.swift` does against token sequences Python produced.
///
/// Only the encoder and decoder are here. Training a tokenizer is a
/// once-per-corpus job for a laptop with a Python interpreter on it, and
/// nothing on the phone will ever do it.
public struct BPETokenizer: Sendable {

    /// The split pattern this port reproduces. `TextSplitter` implements these
    /// semantics in Swift rather than reading the string, so loading a
    /// tokenizer trained with a *different* pattern has to fail loudly instead
    /// of tokenizing text almost correctly.
    public static let supportedPattern =
        #"'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+(?!\S)|\s+"#

    /// Merge rank for a pair of ids. The value is the id the merge produces,
    /// and because ids are handed out in merge order, the id *is* the rank.
    private let merges: [Int: Int]

    /// Every token's bytes, including the 256 single-byte tokens the vocabulary
    /// starts from and any registered specials.
    private let vocabulary: [Int: [UInt8]]

    /// Registered special tokens, in the order they were registered.
    public let specialTokens: [(text: String, id: Int)]

    public var vocabularySize: Int { vocabulary.count }

    public enum LoadError: Error, CustomStringConvertible {
        case unsupportedVersion(Int)
        case unsupportedPattern(String)
        case malformed(String)

        public var description: String {
            switch self {
            case .unsupportedVersion(let v):
                return "tokenizer.json version \(v); this reader understands version 1"
            case .unsupportedPattern(let p):
                return """
                    this tokenizer was trained with a split pattern Swift does not implement.
                    Expected:
                      \(BPETokenizer.supportedPattern)
                    Found:
                      \(p)
                    TextSplitter reproduces one specific pattern's semantics by hand. Loading a \
                    tokenizer trained with a different one would encode text almost correctly, \
                    which is worse than refusing.
                    """
            case .malformed(let what):
                return "tokenizer.json is malformed: \(what)"
            }
        }
    }

    // -- loading --------------------------------------------------------------

    /// Read a tokenizer written by `01_tokenizer/bpe.py`'s `save`.
    ///
    /// The merges are replayed in file order so the vocabulary rebuilds exactly
    /// as it was built during training — the same reason the Python loader
    /// replays them rather than storing the byte strings.
    public init(contentsOf url: URL) throws {
        try self.init(data: try Data(contentsOf: url))
    }

    public init(data: Data) throws {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw LoadError.malformed("top level is not an object")
        }
        let version = root["version"] as? Int ?? 0
        guard version == 1 else { throw LoadError.unsupportedVersion(version) }

        guard let pattern = root["pattern"] as? String else {
            throw LoadError.malformed("no \"pattern\"")
        }
        guard pattern == Self.supportedPattern else {
            throw LoadError.unsupportedPattern(pattern)
        }
        guard let rawMerges = root["merges"] as? [[Int]] else {
            throw LoadError.malformed("\"merges\" is not a list of [first, second, id]")
        }

        var vocabulary: [Int: [UInt8]] = [:]
        vocabulary.reserveCapacity(256 + rawMerges.count)
        for byte in 0..<256 { vocabulary[byte] = [UInt8(byte)] }

        var merges: [Int: Int] = [:]
        merges.reserveCapacity(rawMerges.count)
        for triple in rawMerges {
            guard triple.count == 3 else {
                throw LoadError.malformed("a merge entry is not [first, second, id]")
            }
            let (first, second, id) = (triple[0], triple[1], triple[2])
            guard let left = vocabulary[first], let right = vocabulary[second] else {
                throw LoadError.malformed("merge \(id) refers to ids that do not exist yet")
            }
            merges[Self.key(first, second)] = id
            vocabulary[id] = left + right
        }

        // JSON objects have no order, and the order specials are tried in
        // decides which one wins when one is a prefix of another. Sorting by id
        // reproduces registration order for any tokenizer this repository
        // writes, where ids are handed out in the order specials are declared.
        var specials: [(text: String, id: Int)] = []
        if let rawSpecials = root["special_tokens"] as? [String: Int] {
            specials = rawSpecials.map { (text: $0.key, id: $0.value) }.sorted { $0.id < $1.id }
            for special in specials {
                vocabulary[special.id] = Array(special.text.utf8)
            }
        }

        self.merges = merges
        self.vocabulary = vocabulary
        self.specialTokens = specials
    }

    // -- encoding -------------------------------------------------------------

    /// How special tokens in the input text are treated.
    public enum SpecialHandling {
        /// `<|endoftext|>` typed by a user is thirteen ordinary characters.
        case asOrdinaryText
        /// `<|endoftext|>` typed by a user is the special token.
        case recognised
    }

    /// Encode text to token ids.
    public func encode(_ text: String, specials: SpecialHandling = .asOrdinaryText) -> [Int] {
        guard case .recognised = specials, !specialTokens.isEmpty else {
            return encodeOrdinary(text)
        }

        var ids: [Int] = []
        var rest = Substring(text)
        while !rest.isEmpty {
            guard let hit = firstSpecial(in: rest) else { break }
            if hit.range.lowerBound > rest.startIndex {
                ids += encodeOrdinary(String(rest[rest.startIndex..<hit.range.lowerBound]))
            }
            ids.append(hit.id)
            rest = rest[hit.range.upperBound...]
        }
        if !rest.isEmpty { ids += encodeOrdinary(String(rest)) }
        return ids
    }

    /// Encode text with every special token treated as ordinary characters.
    public func encodeOrdinary(_ text: String) -> [Int] {
        var ids: [Int] = []
        for chunk in TextSplitter.split(text) {
            ids += encodeChunk(Array(chunk.utf8))
        }
        return ids
    }

    /// One chunk, merged until no learned merge applies.
    ///
    /// The rule that matters is *which* merge goes next: always the
    /// earliest-learned one still present, never the leftmost or the longest.
    /// Merge order is what makes encoding deterministic and consistent with
    /// training, and getting it wrong produces a tokenizer that works — it
    /// round-trips, it never fails — while handing the model a different
    /// sequence than the one it was trained on.
    private func encodeChunk(_ bytes: [UInt8]) -> [Int] {
        var ids = bytes.map(Int.init)

        while ids.count >= 2 {
            // Python's `min` keeps the first of several equal minima, so ties
            // resolve to the leftmost pair. It never matters — a rank appears
            // once — but "never matters" is how conventions drift apart.
            var bestRank = Int.max
            var bestPair: (Int, Int)? = nil
            for i in 0..<(ids.count - 1) {
                if let rank = merges[Self.key(ids[i], ids[i + 1])], rank < bestRank {
                    bestRank = rank
                    bestPair = (ids[i], ids[i + 1])
                }
            }
            guard let pair = bestPair else { break }
            ids = Self.applyMerge(ids, pair, bestRank)
        }
        return ids
    }

    /// Replace every occurrence of `pair`, scanning left to right so that a run
    /// like `a a a` merges the first two and leaves the third.
    static func applyMerge(_ ids: [Int], _ pair: (Int, Int), _ newID: Int) -> [Int] {
        var out: [Int] = []
        out.reserveCapacity(ids.count)
        var i = 0
        while i < ids.count {
            if i < ids.count - 1, ids[i] == pair.0, ids[i + 1] == pair.1 {
                out.append(newID)
                i += 2
            } else {
                out.append(ids[i])
                i += 1
            }
        }
        return out
    }

    private func firstSpecial(in text: Substring) -> (id: Int, range: Range<String.Index>)? {
        var best: (id: Int, range: Range<String.Index>)? = nil
        for special in specialTokens {
            guard let range = text.range(of: special.text) else { continue }
            if best == nil || range.lowerBound < best!.range.lowerBound {
                best = (special.id, range)
            }
        }
        return best
    }

    // -- decoding -------------------------------------------------------------

    /// Turn ids back into text.
    ///
    /// Invalid UTF-8 becomes U+FFFD rather than an error, matching the Python
    /// side's `errors="replace"`. That is not laxness: during generation the
    /// model emits one token at a time and a multi-byte character arrives in
    /// pieces, so a partial sequence is the normal state of a decoder, not a
    /// failure of one.
    public func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        for id in ids {
            guard let piece = vocabulary[id] else { continue }
            bytes += piece
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// The raw bytes a single token stands for, or `nil` if it is not in the
    /// vocabulary. Streaming decoders need this to hold a half-finished
    /// character back until the rest of it arrives.
    public func bytes(for id: Int) -> [UInt8]? { vocabulary[id] }

    // -- internals ------------------------------------------------------------

    /// Two ids packed into one key. Vocabularies here are far below 2^31, so
    /// this is a dictionary key that costs no allocation and no hashing of a
    /// tuple wrapper.
    @inline(__always)
    private static func key(_ first: Int, _ second: Int) -> Int {
        (first << 32) | second
    }
}
