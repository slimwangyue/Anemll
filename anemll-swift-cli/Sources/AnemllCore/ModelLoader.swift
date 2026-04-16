import Foundation
@preconcurrency import CoreML

/// Model collection for inference.
public struct LoadedModels: @unchecked Sendable {
    public let embedModel: MLModel?
    public let lmheadModel: MLModel?
    public let ffnChunks: [FFNChunk]
    public let isMonolithic: Bool

    // Convenience initializer for non-monolithic models
    public init(embedModel: MLModel, lmheadModel: MLModel, ffnChunks: [FFNChunk]) {
        self.embedModel = embedModel
        self.lmheadModel = lmheadModel
        self.ffnChunks = ffnChunks
        self.isMonolithic = false
    }

    // Initializer for monolithic models
    public init(monolithicChunk: FFNChunk) {
        self.embedModel = nil
        self.lmheadModel = nil
        self.ffnChunks = [monolithicChunk]
        self.isMonolithic = true
    }
}

/// Protocol for receiving model loading progress updates.
public protocol ModelLoadingProgressDelegate: AnyObject, Sendable {
    /// Called when loading progress changes.
    /// - Parameters:
    ///   - percentage: The overall loading progress from 0.0 to 1.0.
    ///   - stage: Description of the current loading stage.
    ///   - detail: Optional detailed information about the current loading step.
    func loadingProgress(percentage: Double, stage: String, detail: String?)
    
    /// Called when the loading operation has been cancelled.
    func loadingCancelled()
    
    /// Called when all models have been successfully loaded.
    func loadingCompleted(models: LoadedModels)
    
    /// Called when an error occurs during model loading.
    /// - Parameter error: The error that occurred.
    func loadingFailed(error: Error)
}

/// Loads and configures CoreML models with appropriate settings for LLM inference.
public actor ModelLoader {
    /// Configuration for model loading.
    public struct Configuration: Sendable {
        public let computeUnits: MLComputeUnits
        public let allowLowPrecision: Bool
        public let memoryLimit: UInt64?
        public let functionName: String?
        
        public init(
            computeUnits: MLComputeUnits = .cpuAndNeuralEngine,
            //computeUnits: MLComputeUnits = .cpuOnly,
            allowLowPrecision: Bool = false,
            memoryLimit: UInt64? = nil,
            functionName: String? = nil
        ) {
            self.computeUnits = computeUnits
            self.allowLowPrecision = allowLowPrecision
            self.memoryLimit = memoryLimit
            self.functionName = functionName
        }
    }
    
    /// Progress weights for different loading stages
    private struct ProgressWeights {
        static let embedModel = 0.1
        static let lmheadModel = 0.1
        static let ffnChunks = 0.8  // This is distributed evenly across all chunks
    }
    
    /// The delegate that receives progress updates.
    private weak var progressDelegate: (any ModelLoadingProgressDelegate)?
    
    /// Task that can be cancelled to interrupt the loading process.
    private var loadingTask: Task<LoadedModels, Error>?
    
    /// Initializes a new ModelLoader with an optional progress delegate.
    /// - Parameter progressDelegate: Delegate that will receive progress updates.
    public init(progressDelegate: (any ModelLoadingProgressDelegate)? = nil) {
        self.progressDelegate = progressDelegate
    }
    
    /// Cancels any ongoing model loading.
    public func cancelLoading() {
        loadingTask?.cancel()
        Task { 
            let delegate = self.progressDelegate
            await MainActor.run {
                delegate?.loadingCancelled()
            }
        }
    }
    
    /// Compile a .mlpackage on-device and cache the result as .mlmodelc next to the source.
    /// If the compiled model already exists and is newer than the source, skip recompilation.
    private static func compilePackageIfNeeded(at packageURL: URL) throws -> URL {
        let fm = FileManager.default
        // Place compiled model next to the .mlpackage with .mlmodelc extension
        let compiledURL = packageURL.deletingPathExtension().appendingPathExtension("mlmodelc")

        // Check if cached compiled model is still valid
        if fm.fileExists(atPath: compiledURL.path) {
            let srcDate = (try? fm.attributesOfItem(atPath: packageURL.path)[.modificationDate] as? Date) ?? .distantPast
            let dstDate = (try? fm.attributesOfItem(atPath: compiledURL.path)[.modificationDate] as? Date) ?? .distantPast
            if dstDate >= srcDate {
                print("  ♻️ Using cached compiled model: \(compiledURL.lastPathComponent)")
                return compiledURL
            }
            // Source is newer — remove stale cache
            try? fm.removeItem(at: compiledURL)
        }

        print("  🔨 Compiling \(packageURL.lastPathComponent) on-device...")
        let tempCompiledURL = try MLModel.compileModel(at: packageURL)
        // Move from temp location to persistent cache location
        if fm.fileExists(atPath: compiledURL.path) {
            try fm.removeItem(at: compiledURL)
        }
        try fm.moveItem(at: tempCompiledURL, to: compiledURL)
        print("  ✅ Compiled → \(compiledURL.lastPathComponent)")
        return compiledURL
    }

    /// Resolve the file path for a given chunk index.
    /// Supports two naming conventions:
    ///   - 0-indexed: ffn_LUT4_chunk0.mlmodelc  (chunkN pattern)
    ///   - 1-indexed: ffn_LUT4_chunk_01of09.mlmodelc  (_chunk_NNofMM pattern)
    /// Falls back to constructing a path from the base name if no pattern matches.
    private static func resolveChunkPath(basePath: String, chunkIndex0: Int, numChunks: Int) -> String {
        // Try 0-indexed naming: replace trailing digit(s) before .mlmodelc/.mlpackage
        // e.g. ffn_LUT4_chunk0.mlmodelc → ffn_LUT4_chunk3.mlmodelc
        if let range = basePath.range(of: #"chunk\d+\.(mlmodelc|mlpackage)"#, options: .regularExpression) {
            let candidate = basePath.replacingCharacters(
                in: range,
                with: "chunk\(chunkIndex0).\(basePath.hasSuffix(".mlpackage") ? "mlpackage" : "mlmodelc")"
            )
            if modelFileExists(atPath: candidate) { return candidate }
        }
        
        // Try 1-indexed naming: _chunk_NNofMM
        let i1 = chunkIndex0 + 1
        if basePath.contains("_chunk_") {
            let candidate = basePath.replacingOccurrences(
                of: "_chunk_\\d+of",
                with: "_chunk_\(String(format: "%02d", i1))of",
                options: .regularExpression
            )
            if modelFileExists(atPath: candidate) { return candidate }
        }
        
        // Fallback: construct from base name with 1-indexed _chunk_NNofMM
        let directory = (basePath as NSString).deletingLastPathComponent
        let filename = (basePath as NSString).lastPathComponent
        var baseName = filename
        if baseName.hasSuffix(".mlmodelc") {
            baseName = String(baseName.dropLast(9))
        } else if baseName.hasSuffix(".mlpackage") {
            baseName = String(baseName.dropLast(10))
        }
        // Strip any existing chunk suffix for a clean base
        if let r = baseName.range(of: #"_chunk.*$"#, options: .regularExpression) {
            baseName = String(baseName[..<r.lowerBound])
        }
        return "\(directory)/\(baseName)_chunk_\(String(format: "%02d", i1))of\(String(format: "%02d", numChunks)).mlmodelc"
    }

    /// Check if a model file exists at the given path, also checking for a .mlpackage variant.
    private static func modelFileExists(atPath path: String) -> Bool {
        let fm = FileManager.default
        if fm.fileExists(atPath: path) {
            return true
        }
        // Check for .mlpackage variant
        if path.hasSuffix(".mlmodelc") {
            let packagePath = String(path.dropLast(9)) + ".mlpackage"
            return fm.fileExists(atPath: packagePath)
        }
        return false
    }

    private static func loadMLModel(at url: URL, configuration: MLModelConfiguration) throws -> MLModel {
        var loadURL = url
        if url.pathExtension == "mlpackage" {
            loadURL = try compilePackageIfNeeded(at: url)
        } else if url.pathExtension == "mlmodelc" && !FileManager.default.fileExists(atPath: url.path) {
            // .mlmodelc not found — try .mlpackage variant and compile on-device
            let packageURL = url.deletingPathExtension().appendingPathExtension("mlpackage")
            if FileManager.default.fileExists(atPath: packageURL.path) {
                print("  📦 .mlmodelc not found, using .mlpackage: \(packageURL.lastPathComponent)")
                loadURL = try compilePackageIfNeeded(at: packageURL)
            }
        }
        return try MLModel(contentsOf: loadURL, configuration: configuration)
    }

    /// Helper class to avoid data races with currentProgress
    private actor ProgressTracker {
        private var currentProgress = 0.0
        private let delegate: (any ModelLoadingProgressDelegate)?
        
        init(delegate: (any ModelLoadingProgressDelegate)?) {
            self.delegate = delegate
        }
        
        func updateProgress(increment: Double, stage: String, detail: String? = nil) async throws {
            if Task.isCancelled {
                throw ModelError.loadingCancelled
            }
            
            currentProgress += increment
            let percentage = min(currentProgress, 1.0)
            
            if let delegate = delegate {
                await MainActor.run {
                    delegate.loadingProgress(
                        percentage: percentage, 
                        stage: stage,
                        detail: detail
                    )
                }
            }
        }
        
        func getCurrentProgress() -> Double {
            return currentProgress
        }
    }
    
    /// Loads a CoreML model with the specified configuration.
    /// - Parameters:
    ///   - config: YAML configuration containing model paths and settings.
    ///   - configuration: Additional CoreML-specific configuration.
    /// - Returns: A LoadedModels instance containing the embeddings, LM head, and FFN chunks.
    @discardableResult
    public func loadModel(
        from config: YAMLConfig,
        configuration: Configuration = Configuration()
    ) async throws -> LoadedModels {
        // Create a task that can be cancelled
        // We need to capture the config and configuration in a Sendable way
        let configCopy = config
        let configurationCopy = configuration
        let progressTracker = ProgressTracker(delegate: progressDelegate)
        
        loadingTask = Task<LoadedModels, Error> {
            print("\nLoading Models:")

            // Configure compute units
            let modelConfig = MLModelConfiguration()
            modelConfig.computeUnits = configurationCopy.computeUnits

            // Check if this is a monolithic model
            if configCopy.isMonolithic {
                return try await self.loadMonolithicModel(
                    config: configCopy,
                    modelConfig: modelConfig,
                    progressTracker: progressTracker
                )
            }

            // Load embeddings model
            try await progressTracker.updateProgress(
                increment: 0.0,
                stage: "Loading Embeddings Model",
                detail: nil
            )

            print("\nLoading Embeddings Model:")
            let embedURL = URL(fileURLWithPath: configCopy.embedPath)
            print("Path: \(embedURL.path)")
            let embedModel = try ModelLoader.loadMLModel(at: embedURL, configuration: modelConfig)
            print("✓ Embeddings model loaded")
            
            try await progressTracker.updateProgress(
                increment: ProgressWeights.embedModel,
                stage: "Embeddings Model Loaded",
                detail: configCopy.embedPath
            )
            
            // Load LM head model
            print("\nLoading LM Head Model:")
            let lmheadURL = URL(fileURLWithPath: configCopy.lmheadPath)
            print("Path: \(lmheadURL.path)")
            let lmheadModel = try ModelLoader.loadMLModel(at: lmheadURL, configuration: modelConfig)
            print("✓ LM Head model loaded")
            
            try await progressTracker.updateProgress(
                increment: ProgressWeights.lmheadModel,
                stage: "LM Head Model Loaded",
                detail: configCopy.lmheadPath
            )
            
            // Load all FFN chunks
            print("\nLoading FFN Chunks:")
            var ffnChunks: [FFNChunk] = []
            
            // Calculate per-chunk progress increment
            let chunkProgressIncrement = ProgressWeights.ffnChunks / Double(configCopy.numChunks * 2)
            
            // Validate model files exist before attempting to load
            let fileManager = FileManager.default
            let modelDir = (configCopy.ffnPath as NSString).deletingLastPathComponent
            print("Model directory: \(modelDir)")
            
            // Verify embeddings model
            if !ModelLoader.modelFileExists(atPath: configCopy.embedPath) {
                print("❌ ERROR: Embeddings model not found at path: \(configCopy.embedPath)")
                throw ModelError.failedToLoadModel
            }
            
            // Verify LM head model
            if !ModelLoader.modelFileExists(atPath: configCopy.lmheadPath) {
                print("❌ ERROR: LM head model not found at path: \(configCopy.lmheadPath)")
                throw ModelError.failedToLoadModel
            }
            
            // For multi-chunk models, verify at least one chunk exists
            if configCopy.numChunks > 1 {
                var foundAnyChunk = false
                var availableChunks: [Int] = []
                
                // Check all possible chunks (0-indexed internally)
                for i in 0..<configCopy.numChunks {
                    let chunkPath = ModelLoader.resolveChunkPath(
                        basePath: configCopy.ffnPath,
                        chunkIndex0: i,
                        numChunks: configCopy.numChunks
                    )
                    
                    if ModelLoader.modelFileExists(atPath: chunkPath) {
                        foundAnyChunk = true
                        availableChunks.append(i)
                    }
                }
                
                if !foundAnyChunk {
                    print("❌ ERROR: No FFN chunks found for model")
                    if let files = try? fileManager.contentsOfDirectory(atPath: modelDir) {
                        print("Available files in \(modelDir):")
                        for file in files {
                            print("  - \(file)")
                        }
                    }
                    throw ModelError.failedToLoadModel
                }
                
                print("✅ Found \(availableChunks.count) available chunks: \(availableChunks)")
            } else {
                // Single chunk model - verify the FFN file exists
                if !ModelLoader.modelFileExists(atPath: configCopy.ffnPath) {
                    print("❌ ERROR: FFN model not found at path: \(configCopy.ffnPath)")
                    if let files = try? fileManager.contentsOfDirectory(atPath: modelDir) {
                        print("Available files in \(modelDir):")
                        for file in files {
                            print("  - \(file)")
                        }
                    }
                    throw ModelError.failedToLoadModel
                }
            }
            
            // Load chunks sequentially to avoid memory pressure
            for i in 0..<configCopy.numChunks {
                if Task.isCancelled {
                    throw ModelError.loadingCancelled
                }
                
                // Resolve path for this chunk (handles both 0-indexed and 1-indexed naming)
                let chunkPath = ModelLoader.resolveChunkPath(
                    basePath: configCopy.ffnPath,
                    chunkIndex0: i,
                    numChunks: configCopy.numChunks
                )
                
                // Skip this chunk if it doesn't exist
                if !ModelLoader.modelFileExists(atPath: chunkPath) {
                    print("⚠️ Chunk \(i) not found at: \(chunkPath) - skipping")
                    continue
                }
                
                print("Loading chunk \(i): \(chunkPath)")
                let ffnURL = URL(fileURLWithPath: chunkPath)
                
                // Load inference model for this chunk
                try await progressTracker.updateProgress(
                    increment: 0.0,
                    stage: "Loading FFN Chunk",
                    detail: "Inference \(i)/\(configCopy.numChunks)"
                )
                
                var inferModel: MLModel
                var prefillModel: MLModel
                var inferRotateModel: MLModel? = nil
                var prefillRotateModel: MLModel? = nil

                // Try multi-function model first (functionName = "infer"),
                // then fall back to single-function separate files.
                print("Loading inference chunk \(i): \(chunkPath)")
                modelConfig.functionName = "infer"
                do {
                    inferModel = try ModelLoader.loadMLModel(at: ffnURL, configuration: modelConfig)
                    print("✅ Inference chunk \(i) loaded (multi-function)")
                } catch {
                    // Fallback: load as single-function model without functionName
                    print("ℹ️ Multi-function load failed for infer chunk \(i), trying single-function fallback...")
                    modelConfig.functionName = nil
                    do {
                        inferModel = try ModelLoader.loadMLModel(at: ffnURL, configuration: modelConfig)
                        print("✅ Inference chunk \(i) loaded (single-function)")
                    } catch {
                        // Final fallback: try CPU+GPU
                        print("⚠️ ANE load failed for infer chunk \(i), trying CPU+GPU...")
                        modelConfig.computeUnits = .cpuAndGPU
                        do {
                            inferModel = try ModelLoader.loadMLModel(at: ffnURL, configuration: modelConfig)
                            print("✅ Inference chunk \(i) loaded (CPU+GPU)")
                        } catch let fallbackError {
                            print("❌ Error loading inference chunk \(i): \(fallbackError)")
                            throw ModelError.inferenceError("Failed to load inference chunk \(i): \(String(reflecting: fallbackError))")
                        }
                        modelConfig.computeUnits = configurationCopy.computeUnits
                    }
                }

                try await progressTracker.updateProgress(
                    increment: chunkProgressIncrement,
                    stage: "FFN Chunk Loaded",
                    detail: "Inference \(i)/\(configCopy.numChunks)"
                )

                try await progressTracker.updateProgress(
                    increment: 0.0,
                    stage: "Loading FFN Chunk",
                    detail: "Prefill \(i)/\(configCopy.numChunks)"
                )

                // Try multi-function prefill first, then fall back to separate prefill file.
                print("Loading prefill chunk \(i): \(chunkPath)")
                modelConfig.functionName = "prefill"
                do {
                    prefillModel = try ModelLoader.loadMLModel(at: ffnURL, configuration: modelConfig)
                    print("✅ Prefill chunk \(i) loaded (multi-function)")
                } catch {
                    // Fallback: try loading a separate prefill model file
                    // Replace "ffn_" with "prefill_" in the chunk path
                    let prefillPath = chunkPath.replacingOccurrences(of: "/ffn_", with: "/prefill_")
                    let prefillURL = URL(fileURLWithPath: prefillPath)
                    print("ℹ️ Multi-function load failed for prefill chunk \(i), trying separate file: \(prefillPath)")
                    modelConfig.functionName = nil
                    do {
                        prefillModel = try ModelLoader.loadMLModel(at: prefillURL, configuration: modelConfig)
                        print("✅ Prefill chunk \(i) loaded (separate file)")
                    } catch {
                        // Final fallback: try CPU+GPU for the separate prefill file
                        print("⚠️ ANE load failed for prefill chunk \(i), trying CPU+GPU...")
                        modelConfig.computeUnits = .cpuAndGPU
                        do {
                            prefillModel = try ModelLoader.loadMLModel(at: prefillURL, configuration: modelConfig)
                            print("✅ Prefill chunk \(i) loaded (separate file, CPU+GPU)")
                        } catch let fallbackError {
                            print("❌ Error loading prefill chunk \(i): \(fallbackError)")
                            throw ModelError.inferenceError("Failed to load prefill chunk \(i): \(String(reflecting: fallbackError))")
                        }
                        modelConfig.computeUnits = configurationCopy.computeUnits
                    }
                }

                try await progressTracker.updateProgress(
                    increment: chunkProgressIncrement,
                    stage: "FFN Chunk Loaded",
                    detail: "Prefill \(i)/\(configCopy.numChunks)"
                )

                // Try to load rotation functions (4-function model for Gemma3 with sliding window)
                if configCopy.slidingWindow != nil {
                    print("Sliding window configured, attempting to load rotation functions...")
                    modelConfig.functionName = "infer_rotate"
                    do {
                        inferRotateModel = try ModelLoader.loadMLModel(at: ffnURL, configuration: modelConfig)
                        print("✅ Inference rotate chunk \(i) loaded")
                    } catch {
                        print("ℹ️ Inference rotate function not available (2-function model)")
                    }

                    modelConfig.functionName = "prefill_rotate"
                    do {
                        prefillRotateModel = try ModelLoader.loadMLModel(at: ffnURL, configuration: modelConfig)
                        print("✅ Prefill rotate chunk \(i) loaded")
                    } catch {
                        print("ℹ️ Prefill rotate function not available (2-function model)")
                    }

                    if inferRotateModel != nil && prefillRotateModel != nil {
                        print("✅ Chunk \(i) loaded as 4-function model (with rotation support)")
                    }
                }

                ffnChunks.append(FFNChunk(
                    inferModel: inferModel,
                    prefillModel: prefillModel,
                    inferRotateModel: inferRotateModel,
                    prefillRotateModel: prefillRotateModel
                ))
            }
            
            // Verify that we loaded all expected chunks
            if ffnChunks.count != configCopy.numChunks {
                print("❌ ERROR: Not all FFN chunks were loaded. Expected \(configCopy.numChunks), got \(ffnChunks.count)")
                throw ModelError.inferenceError("Failed to load all FFN chunks. Expected \(configCopy.numChunks), got \(ffnChunks.count)")
            } else {
                print("✅ Successfully loaded all \(ffnChunks.count) FFN chunks")
            }
            
            // Final update to ensure we reach 100%
            let currentProgress = await progressTracker.getCurrentProgress()
            try await progressTracker.updateProgress(
                increment: max(0, 1.0 - currentProgress),
                stage: "Loading Complete",
                detail: nil
            )
            
            let loadedModels = LoadedModels(
                embedModel: embedModel,
                lmheadModel: lmheadModel,
                ffnChunks: ffnChunks
            )
            
            let delegate = self.progressDelegate
            if let delegate = delegate {
                await MainActor.run {
                    delegate.loadingCompleted(models: loadedModels)
                }
            }
            
            return loadedModels
        }
        
        do {
            // Copy the task reference to avoid actor-isolated property access in closure
            let task = loadingTask!
            return try await withTaskCancellationHandler {
                try await task.value
            } onCancel: { [task] in
                task.cancel()
            }
        } catch {
            let delegate = self.progressDelegate
            if let delegate = delegate {
                await MainActor.run {
                    delegate.loadingFailed(error: error)
                }
            }
            throw error
        }
    }
    
    /// Backward compatibility for static loading without progress reporting
    public static func loadModel(
        from config: YAMLConfig,
        configuration: Configuration = Configuration()
    ) async throws -> LoadedModels {
        // Since YAMLConfig is now Sendable, we can pass it directly to the actor method
        let loader = ModelLoader()
        return try await loader.loadModel(from: config, configuration: configuration)
    }

    /// Load a monolithic model (single file with infer/prefill functions)
    private func loadMonolithicModel(
        config: YAMLConfig,
        modelConfig: MLModelConfiguration,
        progressTracker: ProgressTracker
    ) async throws -> LoadedModels {
        guard let monolithicPath = config.monolithicModelPath else {
            throw ModelError.invalidModelFormat("Monolithic model path not specified")
        }

        print("\n=== Loading Monolithic Model ===")
        print("Path: \(monolithicPath)")

        let fileManager = FileManager.default
        if !fileManager.fileExists(atPath: monolithicPath) {
            print("❌ ERROR: Monolithic model not found at path: \(monolithicPath)")
            throw ModelError.failedToLoadModel
        }

        let monolithicURL = URL(fileURLWithPath: monolithicPath)

        // Load inference model
        try await progressTracker.updateProgress(
            increment: 0.0,
            stage: "Loading Monolithic Model",
            detail: "Inference function"
        )

        let inferModel: MLModel
        let prefillModel: MLModel
        var inferRotateModel: MLModel? = nil
        var prefillRotateModel: MLModel? = nil

        // Multi-function model: load infer and prefill functions separately.
        print("Loading monolithic infer function...")
        modelConfig.functionName = "infer"
        do {
            inferModel = try ModelLoader.loadMLModel(at: monolithicURL, configuration: modelConfig)
            print("✅ Monolithic infer function loaded")
        } catch {
            print("❌ Error loading monolithic infer function: \(error)")
            throw ModelError.inferenceError("Failed to load monolithic infer function: \(String(reflecting: error))")
        }

        try await progressTracker.updateProgress(
            increment: 0.4,
            stage: "Monolithic Infer Loaded",
            detail: nil
        )

        try await progressTracker.updateProgress(
            increment: 0.0,
            stage: "Loading Monolithic Model",
            detail: "Prefill function"
        )

        print("Loading monolithic prefill function...")
        modelConfig.functionName = "prefill"
        do {
            prefillModel = try ModelLoader.loadMLModel(at: monolithicURL, configuration: modelConfig)
            print("✅ Monolithic prefill function loaded")
        } catch {
            print("❌ Error loading monolithic prefill function: \(error)")
            throw ModelError.inferenceError("Failed to load monolithic prefill function: \(String(reflecting: error))")
        }

        try await progressTracker.updateProgress(
            increment: 0.5,
            stage: "Monolithic Prefill Loaded",
            detail: nil
        )

        // Optional rotation functions for sliding-window models.
        if let slidingWindow = config.slidingWindow, config.contextLength > slidingWindow {
            print("Loading optional monolithic rotate functions...")

            modelConfig.functionName = "infer_rotate"
            do {
                inferRotateModel = try ModelLoader.loadMLModel(at: monolithicURL, configuration: modelConfig)
                print("✅ Monolithic infer_rotate function loaded")
            } catch {
                print("⚠️ Monolithic infer_rotate function unavailable: \(error)")
            }

            modelConfig.functionName = "prefill_rotate"
            do {
                prefillRotateModel = try ModelLoader.loadMLModel(at: monolithicURL, configuration: modelConfig)
                print("✅ Monolithic prefill_rotate function loaded")
            } catch {
                print("⚠️ Monolithic prefill_rotate function unavailable: \(error)")
            }
        }

        // Create FFNChunk with monolithic models
        let monolithicChunk = FFNChunk(
            inferModel: inferModel,
            prefillModel: prefillModel,
            inferRotateModel: inferRotateModel,
            prefillRotateModel: prefillRotateModel
        )

        // Final progress update
        try await progressTracker.updateProgress(
            increment: 0.1,
            stage: "Loading Complete",
            detail: nil
        )

        let loadedModels = LoadedModels(monolithicChunk: monolithicChunk)

        let delegate = self.progressDelegate
        if let delegate = delegate {
            await MainActor.run {
                delegate.loadingCompleted(models: loadedModels)
            }
        }

        print("✅ Monolithic model loaded successfully")
        return loadedModels
    }
}

public enum ModelError: Error, Sendable, LocalizedError {
    case failedToLoadModel
    case invalidModelFormat(String)
    case inferenceError(String)
    case loadingCancelled
    
    public var errorDescription: String? {
        switch self {
        case .failedToLoadModel:
            return "Failed to load model"
        case .invalidModelFormat(let message):
            return message
        case .inferenceError(let message):
            return message
        case .loadingCancelled:
            return "Model loading cancelled"
        }
    }
}
