import SwiftUI

/// Chapter 08 — the app.
///
/// There is no network code in this target and no permission it asks for. The
/// model was trained on a laptop, quantized to 14.9 MiB, and copied into the
/// bundle; everything after that happens on the phone. That is the sentence the
/// whole repository has been building toward, and it is worth noticing that it
/// is a sentence about what is *absent*.
@main
struct HandmadeLLMApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
