import Foundation
import CWhisper
import HlasCore

/// whisper.cpp engine. The model loads on first use and is freed after
/// `modelKeepAliveSeconds` of inactivity, so idle RAM stays near zero.
final class LocalWhisper: Transcriber, @unchecked Sendable {
    private var ctx: OpaquePointer?
    private var unloadGeneration = 0
    private var memoryPressure: DispatchSourceMemoryPressure?
    private let queue = DispatchQueue(label: "hlas.whisper", qos: .userInitiated)

    static let modelDir: URL = {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Hlas/models", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    /// The model picked in Settings (UserDefaults is thread-safe, so the
    /// whisper queue can read it without hopping to the main actor).
    static var selectedModel: LocalWhisperModel {
        LocalWhisperModel(savedValue: UserDefaults.standard.string(forKey: "localModel"))
    }

    static func url(for model: LocalWhisperModel) -> URL { modelDir.appendingPathComponent(model.fileName) }

    static func isDownloaded(_ model: LocalWhisperModel) -> Bool {
        Int64((try? url(for: model).resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) == model.byteCount
    }

    /// The model dictation actually uses: the selected one once it is on disk,
    /// otherwise any downloaded model, so switching never breaks dictation
    /// while the new model is still downloading.
    static var activeModel: LocalWhisperModel? {
        let selected = selectedModel
        if isDownloaded(selected) { return selected }
        return LocalWhisperModel.allCases.first(where: isDownloaded)
    }

    /// HLAS_MODEL_PATH points tests at a model outside Application Support.
    private static var overridePath: String? { ProcessInfo.processInfo.environment["HLAS_MODEL_PATH"] }

    static var modelURL: URL? {
        if let overridePath { return URL(fileURLWithPath: overridePath) }
        return activeModel.map(url(for:))
    }
    static var modelExists: Bool {
        if let overridePath { return FileManager.default.fileExists(atPath: overridePath) }
        return activeModel != nil
    }

    /// Downloaded models other than the selected one (empty until the selected one is on disk).
    static var unusedDownloadedModels: [LocalWhisperModel] {
        let selected = selectedModel
        guard isDownloaded(selected) else { return [] }
        return LocalWhisperModel.allCases.filter { $0 != selected && isDownloaded($0) }
    }

    static func deleteUnusedModels() {
        for model in unusedDownloadedModels {
            try? FileManager.default.removeItem(at: url(for: model))
            Log.write("local model deleted: \(model.fileName)")
        }
    }

    /// Path the current `ctx` was loaded from; a different path means the model changed.
    private var loadedPath: String?

    private func loadModelIfNeeded() {
        guard let path = Self.modelURL?.path else { return }
        if ctx != nil, loadedPath != path { freeModel() }
        guard ctx == nil else { return }
        var params = whisper_context_default_params()
        params.use_gpu = true; params.flash_attn = true
        ctx = whisper_init_from_file_with_params(path, params)
        loadedPath = ctx == nil ? nil : path
        Log.write("local model \(ctx == nil ? "failed to load" : "loaded"): \((path as NSString).lastPathComponent)")
    }

    func prepare(keepAlive: Double) {
        guard Self.modelExists else { return }
        queue.async { [self] in
            unloadGeneration += 1
            monitorMemoryPressure()
            let began = ContinuousClock.now
            let cold = ctx == nil || loadedPath != Self.modelURL?.path
            loadModelIfNeeded()
            if cold { Log.write("local prefetch: elapsed=\(began.duration(to: .now))") }
            scheduleUnload(after: keepAlive)
        }
    }

    func transcribe(samples: [Float], language: String?, prompt: String?, keepAlive: Double) async throws -> String {
        try await transcribe(samples: samples, language: language, prompt: prompt, priming: true, keepAlive: keepAlive)
    }

    func transcribeWithoutPrompt(samples: [Float], language: String?, keepAlive: Double) async throws -> String {
        try await transcribe(samples: samples, language: language, prompt: nil, priming: false, keepAlive: keepAlive)
    }

    private func transcribe(samples: [Float], language: String?, prompt: String?, priming: Bool,
                            keepAlive: Double) async throws -> String {
        guard Self.modelExists else {
            throw NSError(domain: "Hlas", code: 404, userInfo: [
                NSLocalizedDescriptionKey: "Model not downloaded - open Settings"
            ])
        }
        let token = CancellationToken()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                queue.async { [self] in
                    unloadGeneration += 1
                    do {
                        try token.check()
                        monitorMemoryPressure()
                        let began = ContinuousClock.now
                        let cold = ctx == nil || loadedPath != Self.modelURL?.path
                        let text = try run(samples: samples, language: language, prompt: prompt, priming: priming, token: token)
                        try token.check()
                        Log.write("local inference: cold=\(cold) elapsed=\(began.duration(to: .now))")
                        scheduleUnload(after: keepAlive)
                        continuation.resume(returning: text)
                    } catch {
                        scheduleUnload(after: keepAlive)
                        continuation.resume(throwing: error)
                    }
                }
            }
        } onCancel: { token.cancel() }
    }

    private func run(samples: [Float], language: String?, prompt: String?, priming usePriming: Bool,
                     token: CancellationToken) throws -> String {
        loadModelIfNeeded()
        guard let ctx else {
            throw NSError(domain: "Hlas", code: 500, userInfo: [
                NSLocalizedDescriptionKey: "Failed to load model"
            ])
        }

        // Beam search: measurably better Czech than greedy, near-zero cost on
        // Metal (benchmarked ~1.1–1.5s for 11s audio on M1 Pro either way).
        var params = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH)
        params.beam_search.beam_size = 5
        params.print_progress = false
        params.print_realtime = false
        params.print_special = false
        params.print_timestamps = false
        params.no_timestamps = true
        params.suppress_blank = true
        params.temperature = 0
        params.n_threads = Int32(max(2, min(6, ProcessInfo.processInfo.activeProcessorCount - 2)))

        let langHolder = strdup(language ?? "auto")
        defer { free(langHolder) }
        params.language = UnsafePointer(langHolder)

        // Priming prompt (VoiceInk technique): a well-formed sentence in the
        // target language steers diacritics + punctuation; custom vocabulary
        // is appended so names/jargon decode correctly.
        let priming: String
        switch language {
        case "cs": priming = "Toto je přesný přepis mluveného projevu se správnou diakritikou, interpunkcí a velkými písmeny."
        case "sk": priming = "Toto je presný prepis hovoreného prejavu so správnou diakritikou a interpunkciou."
        case "en": priming = "This is an accurate transcript with proper punctuation and capitalization."
        default: priming = ""
        }
        let fullPrompt = [usePriming ? priming : "", prompt ?? ""].filter { !$0.isEmpty }.joined(separator: " ")
        let promptHolder = fullPrompt.isEmpty ? nil : strdup(fullPrompt)
        defer { promptHolder.map { free($0) } }
        if let promptHolder { params.initial_prompt = UnsafePointer(promptHolder) }

        try token.check()
        params.abort_callback = { context in
            guard let context else { return false }
            return Unmanaged<CancellationToken>.fromOpaque(context).takeUnretainedValue().isCancelled
        }
        params.abort_callback_user_data = Unmanaged.passUnretained(token).toOpaque()
        let status = samples.withUnsafeBufferPointer { buf in
            whisper_full(ctx, params, buf.baseAddress, Int32(buf.count))
        }
        try token.check()
        guard status == 0 else {
            throw NSError(domain: "Hlas", code: Int(status), userInfo: [
                NSLocalizedDescriptionKey: "Transcription failed (\(status))"
            ])
        }

        var text = ""
        for i in 0..<whisper_full_n_segments(ctx) {
            if let seg = whisper_full_get_segment_text(ctx, i) {
                text += String(cString: seg)
            }
        }
        return text
    }

    func releaseAfter(_ seconds: Double) { queue.async { [self] in scheduleUnload(after: seconds) } }

    private func monitorMemoryPressure() {
        guard memoryPressure == nil else { return }
        let source = DispatchSource.makeMemoryPressureSource(eventMask: [.warning, .critical], queue: queue)
        source.setEventHandler { [weak self] in self?.freeModel() }
        source.resume()
        memoryPressure = source
    }

    deinit {
        memoryPressure?.cancel()
        if let ctx { whisper_free(ctx) }
    }

    private func scheduleUnload(after seconds: Double) {
        unloadGeneration += 1
        let generation = unloadGeneration
        guard seconds >= 0 else { return }
        queue.asyncAfter(deadline: .now() + seconds) { [weak self] in
            guard let self, self.unloadGeneration == generation else { return }
            self.freeModel()
        }
    }

    private func freeModel() {
        unloadGeneration += 1
        loadedPath = nil
        if let ctx { whisper_free(ctx); self.ctx = nil; Log.write("local model unloaded") }
    }

    func unload() { queue.async { [self] in freeModel() } }
    /// For quitting: ggml's Metal device aborts in its exit-time destructor
    /// while a model is still loaded, so free it before the process exits.
    func unloadAndWait() { queue.sync { freeModel() } }
}

/// Single shared instance so repeated dictations reuse the loaded model.
enum LocalTranscriberHolder {
    static let shared = LocalWhisper()
}

/// Downloads a whisper.cpp model from Hugging Face with progress, verifies
/// its size and SHA-256, then installs it next to the other models.
@MainActor
final class ModelDownloader: NSObject, ObservableObject, URLSessionDownloadDelegate {
    @Published var progress: Double = 0
    @Published var downloading = false
    @Published var error: String?
    /// The model being downloaded, or the one that last failed.
    @Published private(set) var model: LocalWhisperModel = LocalWhisper.selectedModel

    nonisolated private static func stagingURL(for model: LocalWhisperModel) -> URL {
        LocalWhisper.modelDir.appendingPathComponent("\(model.fileName).download")
    }

    private var session: URLSession!
    private var task: URLSessionDownloadTask?

    /// Downloads the model picked in Settings.
    func start() { start(LocalWhisper.selectedModel) }

    func start(_ model: LocalWhisperModel) {
        guard !downloading else { return }
        self.model = model
        downloading = true
        error = nil
        progress = 0
        try? FileManager.default.removeItem(at: Self.stagingURL(for: model))
        let delegateQueue = OperationQueue()
        delegateQueue.qualityOfService = .userInitiated
        delegateQueue.maxConcurrentOperationCount = 1
        session = URLSession(configuration: .default, delegate: self, delegateQueue: delegateQueue)
        let task = session.downloadTask(with: model.remoteURL)
        task.taskDescription = model.rawValue
        self.task = task
        task.resume()
    }

    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                                didWriteData: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        let p = Double(totalBytesWritten) / Double(max(1, totalBytesExpectedToWrite))
        Task { @MainActor in self.progress = p }
    }

    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                                didFinishDownloadingTo location: URL) {
        do {
            guard let response = downloadTask.response as? HTTPURLResponse,
                  (200..<300).contains(response.statusCode) else {
                throw ModelDownloadError.invalidResponse
            }
            let model = LocalWhisperModel(savedValue: downloadTask.taskDescription)
            let fileManager = FileManager.default
            let staging = Self.stagingURL(for: model)
            try? fileManager.removeItem(at: staging)
            try fileManager.moveItem(at: location, to: staging)
            try ModelFileValidator.validate(
                staging,
                expectedByteCount: model.byteCount,
                expectedSHA256: model.sha256
            )
            try Self.install(staging, at: LocalWhisper.url(for: model))
            Task { @MainActor in
                self.downloading = false
                self.progress = 1
                self.task = nil
            }
        } catch {
            try? FileManager.default.removeItem(at: Self.stagingURL(for: LocalWhisperModel(savedValue: downloadTask.taskDescription)))
            Task { @MainActor in
                self.downloading = false
                self.error = error.localizedDescription
                self.task = nil
            }
        }
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error {
            let cancelled = (error as NSError).domain == NSURLErrorDomain
                && (error as NSError).code == NSURLErrorCancelled
            Task { @MainActor in
                self.downloading = false
                self.task = nil
                self.error = cancelled ? nil : "Could not download the model. Check your connection and try again."
            }
        }
    }

    nonisolated private static func install(_ staging: URL, at destination: URL) throws {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: destination.path) {
            _ = try fileManager.replaceItemAt(destination, withItemAt: staging)
        } else {
            try fileManager.moveItem(at: staging, to: destination)
        }
    }
}

private enum ModelDownloadError: LocalizedError {
    case invalidResponse

    var errorDescription: String? {
        switch self {
        case .invalidResponse:
            return "The model server did not return a valid download. Try again later."
        }
    }
}
