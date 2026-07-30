import Foundation
import MLX
import XCTest

/// Why the first MLX call in this package can die four layers down, and what to
/// do about it.
///
/// **What happened.** Every Swift file compiled, the package linked, the test
/// bundle started, and then the first MLX operation aborted inside C++:
///
/// ```
/// MLX error: Failed to load the default metallib.
/// library not found library not found library not found library not found
///   at .../mlx-c/mlx/c/stream.cpp:106
/// ```
///
/// **Why.** MLX's GPU kernels are 49 `.metal` files inside mlx-swift, and
/// SwiftPM compiles them into `mlx-swift_Cmlx.bundle/default.metallib` — if it
/// can run the `metal` compiler. Since Xcode 26 that compiler is not part of
/// Xcode. It is a separate component:
///
/// ```
/// xcodebuild -downloadComponent MetalToolchain
/// ```
///
/// Without it, `swift build` succeeds and says nothing. A missing *compiler*
/// surfacing as a missing *file* at run time is a long way to travel from the
/// cause to the symptom.
///
/// **And installing it is not enough.** `swift test` cannot build Metal shaders
/// at all — mlx-swift's README says so in one line that is easy to read past —
/// so this package's tests run through `xcodebuild test`, which can. Both
/// reasons are in the error message below, because the second one is the one a
/// reader will actually hit and it looks exactly like the first.
///
/// **And there is no CPU fallback**, which was the first thing tried. MLX's
/// scheduler builds its per-device streams when it is first touched, GPU
/// included, so `Device.setDefault(device: .cpu)` fails inside the same
/// `default_stream` call it was meant to avoid. The line number in the error is
/// `mlx_default_cpu_stream_new`, which reads like the CPU path is broken and is
/// really the scheduler initialising both.
///
/// So the check below runs before MLX does, and fails with the command to run
/// rather than letting the process abort. It does **not** skip: a chapter whose
/// central test quietly reports "skipped" on a machine that cannot run it is
/// exactly the kind of green checkmark this repository is written against.
enum TestDevice {

    struct MetalLibraryMissing: Error, CustomStringConvertible {
        var description: String {
            """
            MLX cannot start: no default.metallib was built for mlx-swift.

            There are two reasons this happens, and the common one is the second:

            1. Xcode 26 does not ship the Metal compiler. Install it once:

                   xcodebuild -downloadComponent MetalToolchain

            2. `swift test` cannot build Metal shaders even when it is
               installed — mlx-swift's own README says so. Use:

                   xcodebuild test -scheme HandmadeLLM \\
                     -destination 'platform=OS X,arch=arm64' \\
                     -skipPackagePluginValidation -skipMacroValidation

            There is no CPU-only fallback to fall back to: MLX builds its GPU \
            stream whichever device you ask for, so setting the default device \
            to .cpu fails inside the same call.
            """
        }
    }

    /// Call at the top of any test that touches MLX.
    static func prepare() throws {
        guard metalLibraryURL() != nil else { throw MetalLibraryMissing() }
        _ = announceOnce
    }

    private static let announceOnce: Void = {
        print("  MLX backend: GPU (default.metallib found at \(metalLibraryURL()?.path ?? "?"))")
    }()

    /// Look for the compiled Metal library beside whatever is running the tests.
    ///
    /// Xcode copies `mlx-swift_Cmlx.bundle` into the `.xctest` bundle's
    /// resources; a plain `swift build` would put it alongside. Both are
    /// checked, and each candidate is opened as a `Bundle` rather than probed as
    /// a path — a macOS bundle keeps its resources under `Contents/Resources`
    /// and an iOS one does not, and `Bundle(url:)` already knows which is which.
    /// Reaching into the directory by hand is how the first version of this
    /// looked straight at a metallib that was there and reported it missing.
    private static func metalLibraryURL() -> URL? {
        let testBundle = Bundle(for: BundleAnchor.self).bundleURL
        let candidates = [
            testBundle.appendingPathComponent("Contents/Resources"),
            testBundle,
            testBundle.deletingLastPathComponent(),
        ]
        for directory in candidates {
            let url = directory.appendingPathComponent("mlx-swift_Cmlx.bundle")
            if let bundle = Bundle(url: url),
                let library = bundle.url(forResource: "default", withExtension: "metallib")
            {
                return library
            }
        }
        return nil
    }

    /// Only here so `Bundle(for:)` has a class in this bundle to look up.
    private final class BundleAnchor {}
}
