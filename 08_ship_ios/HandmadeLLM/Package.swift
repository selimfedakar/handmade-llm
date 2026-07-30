// swift-tools-version: 6.0
//
// Chapter 08 — the Swift side of this repository.
//
// Everything the phone needs lives in a library, not in the app: the tokenizer,
// the transformer, the loader for chapter 07's file. The app target is a screen
// on top of it. That split is not tidiness — it is what lets the chapter's
// central claim be checked with `swift test` from a terminal, on a Mac, with no
// device and no Xcode:
//
//     cd 08_ship_ios/HandmadeLLM && swift test
//
// A claim that can only be verified by tapping a button on somebody's phone is
// not a claim a reader can check.
//
// The mlx-swift version is pinned exactly, and that is deliberate. Chapter 07
// asserts byte-identical agreement with MLX's quantize kernel in Python; this
// chapter asserts the Swift side reads those bytes back to identical logits.
// Both claims are about one specific build of one C++ library, so both sides
// name it. `.upToNextMinor` would let the kernel change under a claim that is
// about the kernel.

import PackageDescription

let package = Package(
    name: "HandmadeLLM",
    platforms: [
        // mlx-swift's own floor. Nothing here needs anything newer.
        .macOS(.v14),
        .iOS(.v17),
    ],
    products: [
        .library(name: "HandmadeLLM", targets: ["HandmadeLLM"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.31.6")
    ],
    targets: [
        .target(
            name: "HandmadeLLM",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            // Swift 5 language mode on purpose. MLXArray is a reference to a
            // buffer the GPU also holds, so it is not `Sendable`, and Swift 6's
            // strict concurrency checking turns every generation loop into an
            // argument about actor isolation. This chapter is about a language
            // model on a phone; the concurrency story here is "one background
            // task owns the model", enforced by construction in `Generator`.
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
        .testTarget(
            name: "HandmadeLLMTests",
            dependencies: ["HandmadeLLM"],
            // The golden vectors Python wrote. Committed, small, and the only
            // way the equivalence test means anything without a Python
            // interpreter present.
            resources: [.copy("Golden")],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
