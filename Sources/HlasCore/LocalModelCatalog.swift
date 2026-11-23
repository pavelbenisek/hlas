import Foundation

/// whisper.cpp ggml models the Local engine can download and run.
/// Sizes and SHA-256 come from the Hugging Face LFS metadata of
/// ggerganov/whisper.cpp, so every download is verified before install.
public enum LocalWhisperModel: String, CaseIterable, Codable, Sendable {
    case largeV3TurboQ5 = "large-v3-turbo-q5_0"
    case largeV3TurboQ8 = "large-v3-turbo-q8_0"
    case largeV3Turbo = "large-v3-turbo"
    case largeV3Q5 = "large-v3-q5_0"
    case largeV3 = "large-v3"
    case mediumQ5 = "medium-q5_0"
    case smallQ5 = "small-q5_1"

    /// The model Hlas has always shipped with; also the fallback for unknown saved values.
    public static let `default`: LocalWhisperModel = .largeV3TurboQ5

    public var fileName: String { "ggml-\(rawValue).bin" }

    public var remoteURL: URL {
        URL(string: "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/\(fileName)")!
    }

    public var byteCount: Int64 {
        switch self {
        case .largeV3TurboQ5: return 574_041_195
        case .largeV3TurboQ8: return 874_188_075
        case .largeV3Turbo: return 1_624_555_275
        case .largeV3Q5: return 1_081_140_203
        case .largeV3: return 3_095_033_483
        case .mediumQ5: return 539_212_467
        case .smallQ5: return 190_085_487
        }
    }

    public var sha256: String {
        switch self {
        case .largeV3TurboQ5: return "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2"
        case .largeV3TurboQ8: return "317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1"
        case .largeV3Turbo: return "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69"
        case .largeV3Q5: return "d75795ecff3f83b5faa89d1900604ad8c780abd5739fae406de19f23ecd98ad1"
        case .largeV3: return "64d182b440b98d5203c4f9bd541544d84c605196c4f7b845dfa11fb23594d1e2"
        case .mediumQ5: return "19fea4b380c3a618ec4723c3eef2eb785ffba0d0538cf43f8f235e7b3b34220f"
        case .smallQ5: return "ae85e4a935d7a567bd102fe55afc16bb595bdb618e11b2fc7591bc08120411bb"
        }
    }

    public var title: String {
        switch self {
        case .largeV3TurboQ5: return "large-v3-turbo (quantized)"
        case .largeV3TurboQ8: return "large-v3-turbo (q8)"
        case .largeV3Turbo: return "large-v3-turbo (full)"
        case .largeV3Q5: return "large-v3 (quantized)"
        case .largeV3: return "large-v3 (full)"
        case .mediumQ5: return "medium (quantized)"
        case .smallQ5: return "small (quantized)"
        }
    }

    public var detail: String {
        switch self {
        case .largeV3TurboQ5: return "Recommended. Fast, very good Czech."
        case .largeV3TurboQ8: return "Turbo with less quantization. Slightly more accurate, a bit slower."
        case .largeV3Turbo: return "Unquantized turbo. Most accurate turbo, more RAM."
        case .largeV3Q5: return "Full decoder. Best accuracy, about 2-3x slower than turbo."
        case .largeV3: return "Unquantized large-v3. Highest accuracy, slowest, ~3 GB RAM."
        case .mediumQ5: return "Smaller and older. Noticeably weaker Czech."
        case .smallQ5: return "Tiny and fast. Rough Czech, for slow machines."
        }
    }

    /// "547 MB", "1.5 GB" (binary units, matching what Finder-adjacent UIs have shown so far).
    public var sizeLabel: String {
        let mb = Double(byteCount) / 1_048_576
        return mb >= 1024 ? String(format: "%.1f GB", mb / 1024) : "\(Int(mb.rounded())) MB"
    }

    /// Unknown or missing saved values fall back to the default model.
    public init(savedValue: String?) {
        self = savedValue.flatMap(LocalWhisperModel.init(rawValue:)) ?? .default
    }
}
