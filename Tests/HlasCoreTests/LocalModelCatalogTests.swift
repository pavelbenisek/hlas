import Foundation
import HlasCore
import Testing

@Suite("Local model catalog")
struct LocalModelCatalogTests {
    @Test("Default stays the shipped quantized turbo model")
    func defaultModel() {
        #expect(LocalWhisperModel.default.fileName == "ggml-large-v3-turbo-q5_0.bin")
        #expect(LocalWhisperModel.default.byteCount == 574_041_195)
        #expect(LocalWhisperModel.default.sizeLabel == "547 MB")
    }

    @Test("Unknown saved values fall back to the default")
    func fallback() {
        #expect(LocalWhisperModel(savedValue: nil) == .default)
        #expect(LocalWhisperModel(savedValue: "nonsense") == .default)
        #expect(LocalWhisperModel(savedValue: "large-v3") == .largeV3)
    }

    @Test("Every model has a distinct file and a well-formed checksum")
    func metadata() {
        let all = LocalWhisperModel.allCases
        #expect(Set(all.map(\.fileName)).count == all.count)
        for model in all {
            #expect(model.sha256.count == 64)
            #expect(model.sha256.allSatisfy { $0.isHexDigit })
            #expect(model.byteCount > 0)
            #expect(model.remoteURL.lastPathComponent == model.fileName)
        }
    }

    @Test("Sizes read as MB below 1 GB and GB above")
    func sizeLabels() {
        #expect(LocalWhisperModel.smallQ5.sizeLabel == "181 MB")
        #expect(LocalWhisperModel.largeV3.sizeLabel == "2.9 GB")
    }
}
