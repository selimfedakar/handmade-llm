import XCTest

@testable import HandmadeLLM

/// The Swift tokenizer against the Python one, on the same texts.
///
/// Splitting is only the first half. The second half is the merge loop, and its
/// one subtlety is *which* merge goes next: always the earliest-learned one
/// still present, never the leftmost and never the longest. A port that picks
/// the leftmost still round-trips, still never fails, and hands the model a
/// sequence it was not trained on.
final class TokenizerTests: XCTestCase {

    /// The trained tokenizer lives in `data/`, which is gitignored — a corpus
    /// and a trained tokenizer do not belong in a repository people clone to
    /// learn from. So this test skips on a fresh clone and says how to make it
    /// run.
    private func loadTokenizer() throws -> BPETokenizer {
        let url = Golden.repositoryRoot.appendingPathComponent("data/tokenizer.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw XCTSkip(
                """
                no data/tokenizer.json — it is gitignored. To run this test:
                  python data/download.py
                  python 01_tokenizer/train_tokenizer.py --vocab-size 4096
                  python 08_ship_ios/export_golden.py
                """)
        }
        return try BPETokenizer(contentsOf: url)
    }

    func test_swift_encodes_to_the_same_ids_as_python() throws {
        let tokenizer = try loadTokenizer()
        let golden = try Golden.json("tokens.json")

        XCTAssertEqual(
            tokenizer.vocabularySize, golden["vocab_size"] as? Int,
            "the tokenizer on disk is not the one the fixture was made from")

        let cases = golden["cases"] as! [[String: Any]]
        var mismatches: [String] = []
        for testCase in cases {
            let text = testCase["text"] as! String
            let expected = testCase["ids"] as! [Int]
            let recognise = testCase["specials"] != nil
            let actual = tokenizer.encode(
                text, specials: recognise ? .recognised : .asOrdinaryText)
            if actual != expected {
                mismatches.append("\(text.debugDescription)\n  python \(expected)\n  swift  \(actual)")
            }
        }
        XCTAssertTrue(
            mismatches.isEmpty,
            "\(mismatches.count) of \(cases.count) cases differ:\n"
                + mismatches.joined(separator: "\n"))
    }

    func test_swift_decodes_to_the_same_text_as_python() throws {
        let tokenizer = try loadTokenizer()
        let golden = try Golden.json("tokens.json")

        for testCase in golden["cases"] as! [[String: Any]] {
            let ids = testCase["ids"] as! [Int]
            let expected = testCase["decoded"] as! String
            XCTAssertEqual(
                tokenizer.decode(ids), expected,
                "decoding \(String(describing: testCase["text"]).debugDescription)")
        }
    }

    func test_every_byte_survives_a_round_trip() throws {
        let tokenizer = try loadTokenizer()
        // Byte-level BPE never emits an unknown token and decoding is exact.
        // That is the property that makes it worth the trouble, so it is worth
        // asserting on something other than English.
        for text in ["", "a", "İstanbul", "日本語", "🌍🍓", "\u{0}\u{1}\u{7F}", "x²½"] {
            XCTAssertEqual(tokenizer.decode(tokenizer.encode(text)), text, text.debugDescription)
        }
    }

    /// Merge application is shared between the two languages and is small
    /// enough to check directly. Overlapping runs are the case where a
    /// left-to-right scan and a naive replace-all disagree.
    func test_merging_a_run_consumes_pairs_left_to_right() {
        XCTAssertEqual(BPETokenizer.applyMerge([1, 1, 1], (1, 1), 9), [9, 1])
        XCTAssertEqual(BPETokenizer.applyMerge([1, 1, 1, 1], (1, 1), 9), [9, 9])
        XCTAssertEqual(BPETokenizer.applyMerge([2, 1, 1, 2], (1, 1), 9), [2, 9, 2])
        XCTAssertEqual(BPETokenizer.applyMerge([1, 2], (1, 1), 9), [1, 2])
    }
}
