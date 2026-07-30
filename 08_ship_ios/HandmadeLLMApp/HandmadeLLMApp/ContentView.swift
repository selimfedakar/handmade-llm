import HandmadeLLM
import SwiftUI

/// One screen: a prompt, the tokens as they arrive, and the two numbers that
/// decide whether any of this was worth doing.
///
/// The weights picker is not a settings toggle. Chapter 07 measured the 4-bit
/// model about 6% faster than float32 on a laptop and said, in as many words,
/// that whether the gap widens on a phone — where the memory bandwidth is
/// scarcer and the model is the same size — was a question for chapter 08. The
/// picker and the **Compare** button are that question, in a form you can hold.
struct ContentView: View {
    @StateObject private var store = ModelStore()
    @FocusState private var promptFocused: Bool

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                output
                Divider()
                readout
                composer
            }
            .navigationTitle("handmade-llm")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { weightsPicker }
        }
        .task { store.load() }
    }

    // -- the transcript --------------------------------------------------------

    private var output: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    if case .failed(let message) = store.phase {
                        Text(message)
                            .font(.callout.monospaced())
                            .foregroundStyle(.red)
                    } else if store.output.isEmpty {
                        Text(placeholder)
                            .font(.callout)
                            .foregroundStyle(.secondary)
                    } else {
                        Text(store.prompt + store.output)
                            .font(.system(.body, design: .serif))
                            .textSelection(.enabled)
                    }

                    if let benchmark = store.benchmark {
                        Text(benchmark)
                            .font(.caption.monospaced())
                            .padding(12)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(.quaternary, in: RoundedRectangle(cornerRadius: 10))
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
            }
            .onChange(of: store.output) { _, _ in
                withAnimation { proxy.scrollTo("bottom", anchor: .bottom) }
            }
        }
    }

    private var placeholder: String {
        switch store.phase {
        case .loading:
            return "Loading the model…"
        case .benchmarking(let round, let total):
            return "Timing both models, round \(round) of \(total).\nBoth are resident; they alternate inside one loop."
        default:
            return """
                Three hundred training steps on TinyShakespeare is four passes over the \
                corpus and far too few to expect English. What you should see is a model \
                that has learned the shape of the text — line breaks, capitalised speaker \
                names, the rhythm of a colon — and not much more. That is the honest result.
                """
        }
    }

    // -- the numbers -----------------------------------------------------------

    private var readout: some View {
        HStack(spacing: 16) {
            metric(
                "tok/s",
                store.tokensPerSecond.map { String(format: "%.1f", $0) })
            metric(
                "peak",
                store.peakMemoryBytes.map { "\($0 / 1_048_576) MiB" })
            metric(
                "weights",
                store.weightBytes[store.weights].map {
                    String(format: "%.1f MiB", Double($0) / 1_048_576)
                })
            Spacer()
            if store.generatorCount > 1 {
                Button("Compare") { store.runPairedBenchmark() }
                    .font(.caption.weight(.semibold))
                    .disabled(store.isBusy)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
        .background(.bar)
    }

    private func metric(_ label: String, _ value: String?) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(value ?? "—")
                .font(.system(.footnote, design: .monospaced).weight(.semibold))
                .contentTransition(.numericText())
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    // -- input -----------------------------------------------------------------

    private var composer: some View {
        HStack(spacing: 10) {
            TextField("prompt", text: $store.prompt, axis: .vertical)
                .lineLimit(1...3)
                .textFieldStyle(.roundedBorder)
                .focused($promptFocused)
                .disabled(store.isBusy)
                .submitLabel(.go)

            if store.isBusy {
                Button("Stop", role: .destructive) { store.cancel() }
                    .buttonStyle(.bordered)
            } else {
                Button("Run") {
                    promptFocused = false
                    store.generate()
                }
                .buttonStyle(.borderedProminent)
                .disabled(store.prompt.isEmpty)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 10)
        .background(.bar)
    }

    private var weightsPicker: some ToolbarContent {
        ToolbarItem(placement: .topBarTrailing) {
            Picker("Weights", selection: $store.weights) {
                ForEach(store.availableWeights) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.menu)
            .disabled(store.isBusy || store.availableWeights.count < 2)
        }
    }
}

extension ModelStore {
    /// How many sets of weights made it into the bundle. The comparison needs
    /// two; a release build would ship one.
    var generatorCount: Int { availableWeights.count }
}

#Preview {
    ContentView()
}
