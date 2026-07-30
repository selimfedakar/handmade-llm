import XCTest

@testable import HandmadeLLM

/// The Swift splitter against the Python regular expression, on every case in
/// `Golden/splits.json`.
///
/// This is the test the chapter is built around. Chapter 01 chops text with a
/// pattern; chapter 08 chops it with hand-written predicates; if those two ever
/// disagree, the model on the phone is fed a different token sequence than the
/// one it was trained on and nothing crashes. The cases below were chosen to
/// make them disagree — combining marks, connector punctuation, superscripts,
/// zero-width joiners, and the control characters CPython counts as whitespace.
final class SplitterTests: XCTestCase {

    func test_swift_splits_text_exactly_as_the_python_pattern_does() throws {
        let golden = try Golden.json("splits.json")
        let pattern = golden["pattern"] as? String
        XCTAssertEqual(
            pattern, BPETokenizer.supportedPattern,
            "the fixture was generated from a different pattern than this port implements")

        let cases = golden["cases"] as! [[String: Any]]
        XCTAssertGreaterThan(cases.count, 40, "the fixture is smaller than it should be")

        var mismatches: [String] = []
        for testCase in cases {
            let text = testCase["text"] as! String
            let expected = testCase["chunks"] as! [String]
            let actual = TextSplitter.split(text)
            if actual != expected {
                mismatches.append(
                    """
                    input     \(escaped(text))
                      python  \(expected.map(escaped).joined(separator: " "))
                      swift   \(actual.map(escaped).joined(separator: " "))
                    """)
            }
        }
        XCTAssertTrue(
            mismatches.isEmpty,
            "\(mismatches.count) of \(cases.count) cases differ:\n"
                + mismatches.joined(separator: "\n"))
    }

    /// The cases above are data. These are the three properties that data is
    /// checking, written out so that a reader who never opens the JSON still
    /// learns what is surprising about this splitter.

    func test_an_underscore_is_dropped_entirely() {
        // No branch of the pattern accepts one: it is a word character, so both
        // `[^\W\d_]` and `[^\s\w]` exclude it, and `findall` returns matches
        // rather than a partition. Chapter 01's tokenizer cannot see it.
        XCTAssertEqual(TextSplitter.split("snake_case_name"), ["snake", "case", "name"])
        XCTAssertEqual(TextSplitter.split("__dunder__"), ["dunder"])
    }

    func test_a_combining_mark_is_not_a_word_character_here() {
        // ICU — which is what `NSRegularExpression` runs on — puts `\p{M}` in
        // `\w`. CPython does not. So a lone combining acute is punctuation to
        // the Python pattern and would have been swallowed by a regex port.
        XCTAssertEqual(TextSplitter.split("\u{0301} alone"), ["\u{0301}", " alone"])
        XCTAssertFalse(TextSplitter.isWord("\u{0301}"))
    }

    func test_a_run_of_spaces_leaves_one_for_the_next_word() {
        // `\s+(?!\S)` gives a character back so that the following word keeps
        // its leading space, which is the whole reason the tokenizer learns
        // " the" as one token.
        XCTAssertEqual(TextSplitter.split("a   b"), ["a", "  ", " b"])
        XCTAssertEqual(TextSplitter.split("a b"), ["a", " b"])
        XCTAssertEqual(TextSplitter.split("hello   "), ["hello", "   "])
    }

    func test_digits_are_capped_at_three() {
        XCTAssertEqual(TextSplitter.split("1234567"), ["123", "456", "7"])
    }

    func test_contractions_are_case_sensitive() {
        XCTAssertEqual(TextSplitter.split("don't"), ["don", "'t"])
        XCTAssertEqual(TextSplitter.split("DON'T"), ["DON", "'", "T"])
    }

    private func escaped(_ text: String) -> String {
        var out = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\n": out += "\\n"
            case "\t": out += "\\t"
            case "\"": out += "\\\""
            default:
                if scalar.value < 0x20 || (0x200B...0x200D).contains(scalar.value)
                    || scalar.value == 0xA0
                {
                    out += String(format: "\\u{%04X}", scalar.value)
                } else {
                    out.unicodeScalars.append(scalar)
                }
            }
        }
        return out + "\""
    }
}
