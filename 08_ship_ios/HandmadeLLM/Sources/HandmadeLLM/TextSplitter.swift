/// Chapter 08 — the tokenizer's first step, rewritten in Swift.
///
/// Chapter 01 chops text into chunks before any merge happens, so that merges
/// never glue a word to the punctuation after it. In Python that is one regular
/// expression:
///
/// ```python
/// SPLIT_PATTERN = (
///     r"'(?:[sdmt]|ll|ve|re)"   # contractions, kept whole
///     r"| ?[^\W\d_]+"           # letters, with an optional leading space
///     r"| ?\d{1,3}"             # digits in groups of at most three
///     r"| ?[^\s\w]+"            # punctuation and symbols
///     r"|\s+(?!\S)"             # trailing whitespace at end of input
///     r"|\s+"                   # any remaining whitespace
/// )
/// ```
///
/// The obvious thing to do here was hand that same string to
/// `NSRegularExpression` and move on. I did not, and the reason is the whole
/// point of this file.
///
/// **`\w` does not mean the same thing in the two engines.** Python's `re`
/// defines a word character as "alphanumeric or underscore" using CPython's own
/// Unicode tables; ICU — which is what `NSRegularExpression` runs on — defines
/// it as `[\p{Alphabetic}\p{M}\p{Nd}\p{Pc}‌‍]`. Those agree on every
/// character in TinyShakespeare and disagree on combining marks, on connector
/// punctuation, and on a handful of scripts. A tokenizer that splits text one
/// way in training and another way at inference does not crash. It produces
/// slightly different token sequences for some inputs and a slightly worse
/// model, silently, forever.
///
/// That is chapter 07's lesson wearing different clothes: an equivalence test on
/// data small enough to read tests your algorithm, not your conventions
/// (`docs/LESSONS.md` L12). ASCII English is data small enough to read.
///
/// So the character classes are written out here as explicit predicates, each
/// naming the CPython rule it reproduces, and `Tests/SplitterTests.swift` checks
/// them against splits that Python actually produced — including the inputs
/// chosen specifically to make the two engines disagree.
public enum TextSplitter {

    // -- the character classes ---------------------------------------------
    //
    // CPython's `re` module, in Unicode mode, resolves its classes through
    // `Py_UNICODE_IS*` in `Objects/unicodectype.c`. Each predicate below names
    // the one it stands for. Where Swift's own property is not the same set,
    // that difference is spelled out rather than assumed away.

    /// `\w` — CPython: `Py_UNICODE_ISALNUM(ch) || ch == '_'`, where `ISALNUM`
    /// is alphabetic, or decimal, or digit, or numeric.
    static func isWord(_ s: Unicode.Scalar) -> Bool {
        isAlphabetic(s) || s.properties.numericType != nil || s == "_"
    }

    /// `Py_UNICODE_ISALPHA` — general category `L*`, and nothing else.
    ///
    /// Not `Unicode.Scalar.Properties.isAlphabetic`. That is the Unicode
    /// *Alphabetic* property, which also contains `Nl`, `Other_Alphabetic`, and
    /// most combining marks. CPython's `ALPHA_MASK` is set for the letter
    /// categories only, so this is the set to match.
    static func isAlphabetic(_ s: Unicode.Scalar) -> Bool {
        switch s.properties.generalCategory {
        case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter,
            .modifierLetter, .otherLetter:
            return true
        default:
            return false
        }
    }

    /// `\d` — CPython: `Py_UNICODE_ISDECIMAL`, a character with a *decimal*
    /// value. `²` has a digit value but not a decimal one, so it is not `\d`,
    /// and Swift's `numericType == .decimal` draws the line in the same place.
    static func isDecimal(_ s: Unicode.Scalar) -> Bool {
        s.properties.numericType == .decimal
    }

    /// `[^\W\d_]` — a word character that is neither a decimal digit nor an
    /// underscore. In practice: letters, plus the numeric characters that are
    /// not decimal digits. Called "the letter class" below.
    static func isLetterClass(_ s: Unicode.Scalar) -> Bool {
        isWord(s) && !isDecimal(s) && s != "_"
    }

    /// `\s` — CPython: `Py_UNICODE_ISSPACE`. That is Unicode's `White_Space`
    /// property **plus** the four ASCII separators `U+001C`–`U+001F`, which
    /// Unicode does not consider whitespace and CPython does. Four code points
    /// nobody types on purpose; they are here because the claim this file makes
    /// is exactness, and exactness has no small exceptions.
    static func isSpace(_ s: Unicode.Scalar) -> Bool {
        s.properties.isWhitespace || (0x1C...0x1F).contains(s.value)
    }

    /// `[^\s\w]` — punctuation, symbols, and everything else that is neither.
    static func isSymbol(_ s: Unicode.Scalar) -> Bool {
        !isSpace(s) && !isWord(s)
    }

    // -- the scan ------------------------------------------------------------

    /// Split `text` the way `re.findall(SPLIT_PATTERN, text)` does.
    ///
    /// The scanner walks scalars — not `Character`s. Python's `re` matches code
    /// points, so a family emoji built from five scalars and two joiners is five
    /// separate things to the pattern and one `Character` to Swift. Iterating
    /// graphemes here would be a different tokenizer.
    ///
    /// Alternatives are tried in the order they appear in the pattern, which is
    /// what an alternation does, and the first one that matches wins. A scalar
    /// that matches nothing is **dropped** — `re.findall` returns matches, not a
    /// partition, and it simply moves on. The underscore is the character this
    /// happens to: it is a word character, so `[^\W\d_]` and `[^\s\w]` both
    /// exclude it, and no other branch wants it. Chapter 01's tokenizer cannot
    /// see an underscore. Surprising, true, and reproduced here on purpose.
    public static func split(_ text: String) -> [String] {
        let scalars = Array(text.unicodeScalars)
        var chunks: [String] = []
        var i = 0

        while i < scalars.count {
            if let end = matchAlternative(scalars, i) {
                var view = String.UnicodeScalarView()
                view.append(contentsOf: scalars[i..<end])
                chunks.append(String(view))
                i = end
            } else {
                i += 1  // matched nothing; `findall` skips it
            }
        }
        return chunks
    }

    /// The index one past the end of the first alternative that matches at `i`,
    /// or `nil` if none does.
    private static func matchAlternative(_ s: [Unicode.Scalar], _ i: Int) -> Int? {
        matchContraction(s, i)
            ?? matchRun(s, i, isLetterClass)
            ?? matchDigits(s, i)
            ?? matchRun(s, i, isSymbol)
            ?? matchTrailingSpace(s, i)
            ?? matchSpace(s, i)
    }

    /// `'(?:[sdmt]|ll|ve|re)` — the contractions English writes as one word and
    /// a tokenizer should not split into three tokens. Case-sensitive, because
    /// the pattern is: `'S` goes down the punctuation branch instead.
    private static func matchContraction(_ s: [Unicode.Scalar], _ i: Int) -> Int? {
        guard s[i] == "'", i + 1 < s.count else { return nil }
        if "sdmt".unicodeScalars.contains(s[i + 1]) { return i + 2 }
        guard i + 2 < s.count else { return nil }
        let two = String(String.UnicodeScalarView(s[(i + 1)...(i + 2)]))
        return ["ll", "ve", "re"].contains(two) ? i + 3 : nil
    }

    /// ` ?<class>+` — an optional single leading space, then one or more of a
    /// class. No backtracking is needed over the optional space: every class
    /// this is used with excludes the space character, so if the body fails
    /// after consuming the space it would fail without it too.
    private static func matchRun(
        _ s: [Unicode.Scalar], _ i: Int, _ inClass: (Unicode.Scalar) -> Bool
    ) -> Int? {
        var j = i
        if s[j] == " " { j += 1 }
        let bodyStart = j
        while j < s.count, inClass(s[j]) { j += 1 }
        return j > bodyStart ? j : nil
    }

    /// ` ?\d{1,3}` — digits in groups of at most three, so that a year and a
    /// house number tokenize the same way as the first three digits of a
    /// thousand-digit number. Greedy up to the cap.
    private static func matchDigits(_ s: [Unicode.Scalar], _ i: Int) -> Int? {
        var j = i
        if s[j] == " " { j += 1 }
        let bodyStart = j
        while j < s.count, j - bodyStart < 3, isDecimal(s[j]) { j += 1 }
        return j > bodyStart ? j : nil
    }

    /// `\s+(?!\S)` — whitespace not followed by a non-space.
    ///
    /// Read it slowly, because it does not do what it looks like it does. The
    /// greedy `\s+` eats the whole run; the lookahead then demands that what
    /// follows is not a non-space character. At the end of the input that holds
    /// and the whole run matches. In the middle of the input it fails, the
    /// engine gives one character back, and *now* the next character is a space
    /// — so the lookahead passes and the match is the run **minus its last
    /// character**, leaving that one space to be picked up as the leading space
    /// of the following word. A run of exactly one space in the middle of text
    /// cannot give a character back and falls through to the plain `\s+` branch.
    private static func matchTrailingSpace(_ s: [Unicode.Scalar], _ i: Int) -> Int? {
        var j = i
        while j < s.count, isSpace(s[j]) { j += 1 }
        guard j > i else { return nil }
        if j == s.count { return j }  // the run ends the input
        return j - 1 > i ? j - 1 : nil  // give one back, if there is one to give
    }

    /// `\s+` — whatever whitespace the branch above declined.
    private static func matchSpace(_ s: [Unicode.Scalar], _ i: Int) -> Int? {
        var j = i
        while j < s.count, isSpace(s[j]) { j += 1 }
        return j > i ? j : nil
    }
}
