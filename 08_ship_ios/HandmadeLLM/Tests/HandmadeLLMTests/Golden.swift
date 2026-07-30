import Foundation
import XCTest

/// The fixtures `08_ship_ios/export_golden.py` wrote, and where to find them.
///
/// They are committed. That is the whole design: the claim chapter 08 makes is
/// that Swift agrees with Python, and a claim you can only check by having
/// Python installed and a corpus downloaded and a model trained is a claim
/// almost nobody checks. `swift test` on a fresh clone runs every one of these.
enum Golden {

    /// The `Golden` directory inside the test bundle.
    static var directory: URL {
        Bundle.module.resourceURL!.appendingPathComponent("Golden")
    }

    /// The repository root, found from this file rather than from the working
    /// directory — `swift test` and Xcode disagree about the latter.
    static var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // HandmadeLLMTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // HandmadeLLM
            .deletingLastPathComponent()  // 08_ship_ios
            .deletingLastPathComponent()  // handmade-llm
    }

    static func json(_ name: String) throws -> [String: Any] {
        let url = directory.appendingPathComponent(name)
        let data = try Data(contentsOf: url)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Failure.notAnObject(name)
        }
        return object
    }

    enum Failure: Error, CustomStringConvertible {
        case notAnObject(String)
        var description: String {
            switch self {
            case .notAnObject(let name): return "\(name) is not a JSON object"
            }
        }
    }

    /// Skip the rest of a test, saying why, when something that is not in the
    /// repository is missing. Used for the real 24.9M checkpoint, which is
    /// gitignored and correctly so.
    static func requireCheckpoint(
        _ directory: URL, hint: String, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        guard FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("meta.json").path)
        else {
            throw XCTSkip(
                "no checkpoint at \(directory.path) — \(hint)", file: file, line: line)
        }
    }

    /// The largest absolute difference between two float arrays, and where it
    /// is. Reported rather than asserted away: "they agree" is a claim, and the
    /// number behind it belongs in the test output.
    static func largestDifference(_ a: [Float], _ b: [Float]) -> (delta: Float, index: Int) {
        var worst: Float = 0
        var at = 0
        for i in 0..<min(a.count, b.count) {
            let d = abs(a[i] - b[i])
            if d > worst {
                worst = d
                at = i
            }
        }
        return (worst, at)
    }
}
