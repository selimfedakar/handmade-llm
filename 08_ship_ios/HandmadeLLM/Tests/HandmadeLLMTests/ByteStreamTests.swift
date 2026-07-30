import XCTest

@testable import HandmadeLLM

/// The streaming decoder, which exists because of a bug that only shows up on
/// screen.
///
/// A token is a run of UTF-8 bytes, and a byte-level BPE tokenizer is under no
/// obligation to keep a character inside one token. Decode each token on its own
/// and append the result, and every multi-byte character that spans a token
/// boundary becomes a permanent U+FFFD — permanent because by the time the rest
/// of the character arrives, the replacement is already on the screen.
///
/// `swift test` is where this gets caught. On a phone it looks like the model
/// has a stutter.
final class ByteStreamTests: XCTestCase {

    /// A vocabulary where token `n` is byte `n`, so a test can spell out exactly
    /// which bytes arrive in which order.
    private func byteStream() -> ByteStream {
        ByteStream { id in id < 256 ? [UInt8(id)] : nil }
    }

    func test_ascii_comes_through_one_character_at_a_time() {
        var stream = byteStream()
        XCTAssertEqual(stream.append(0x68), "h")
        XCTAssertEqual(stream.append(0x69), "i")
    }

    func test_a_two_byte_character_split_across_tokens_is_held_and_then_released() {
        var stream = byteStream()
        // "é" is C3 A9.
        XCTAssertEqual(stream.append(0xC3), "", "half a character must not be shown")
        XCTAssertEqual(stream.append(0xA9), "é")
    }

    func test_a_four_byte_emoji_arriving_in_four_pieces() {
        var stream = byteStream()
        // "🌍" is F0 9F 8C 8D.
        XCTAssertEqual(stream.append(0xF0), "")
        XCTAssertEqual(stream.append(0x9F), "")
        XCTAssertEqual(stream.append(0x8C), "")
        XCTAssertEqual(stream.append(0x8D), "🌍")
    }

    func test_a_whole_character_inside_one_token_is_released_immediately() {
        var stream = ByteStream { _ in Array("günaydın".utf8) }
        XCTAssertEqual(stream.append(0), "günaydın")
    }

    func test_malformed_bytes_are_not_held_forever() {
        // A lone continuation byte is not the start of anything. Waiting for the
        // rest of it would stall the stream permanently, so it is released and
        // rendered as U+FFFD — which is what the Python side's
        // `errors="replace"` does too.
        var stream = byteStream()
        XCTAssertEqual(stream.append(0xA9), "\u{FFFD}")
        XCTAssertEqual(stream.append(0x68), "h")
    }

    func test_an_unknown_token_contributes_nothing_and_breaks_nothing() {
        var stream = byteStream()
        XCTAssertEqual(stream.append(9_999), "")
        XCTAssertEqual(stream.append(0x68), "h")
    }
}
