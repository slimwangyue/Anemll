#!/usr/bin/env swift
// test_mlpackage_ane.swift
// Test that .mlpackage files can be compiled on-device and loaded on ANE.
//
// Usage: swift test_mlpackage_ane.swift <path_to.mlpackage> [functionName]
// Example: swift test_mlpackage_ane.swift ../qwen3_5_v4_lut4_p2/embed_single.mlpackage
// Example: swift test_mlpackage_ane.swift ../qwen3_5_v4_lut4_p2/ffn_LUT4_chunk0.mlpackage infer

import Foundation
import CoreML

func testMLPackageANE(packagePath: String, functionName: String?) {
    let packageURL = URL(fileURLWithPath: packagePath)
    let fm = FileManager.default

    guard fm.fileExists(atPath: packageURL.path) else {
        print("❌ File not found: \(packagePath)")
        exit(1)
    }

    print("=== .mlpackage ANE Load Test ===")
    print("Source: \(packageURL.lastPathComponent)")
    if let fn = functionName {
        print("Function: \(fn)")
    }

    // Step 1: Compile .mlpackage on-device
    print("\n--- Step 1: Compile on-device ---")
    let compileStart = Date()
    let compiledURL: URL
    do {
        let tempURL = try MLModel.compileModel(at: packageURL)
        let elapsed = Date().timeIntervalSince(compileStart)
        print("✅ Compiled in \(String(format: "%.1f", elapsed))s → \(tempURL.path)")

        // Move to a persistent location next to source
        let persistentURL = packageURL.deletingPathExtension().appendingPathExtension("test_compiled.mlmodelc")
        if fm.fileExists(atPath: persistentURL.path) {
            try fm.removeItem(at: persistentURL)
        }
        try fm.moveItem(at: tempURL, to: persistentURL)
        compiledURL = persistentURL
        print("   Saved: \(compiledURL.lastPathComponent)")
    } catch {
        print("❌ Compilation failed: \(error)")
        exit(1)
    }

    // Step 2: Load with cpuAndNeuralEngine
    print("\n--- Step 2: Load on ANE (cpuAndNeuralEngine) ---")
    let config = MLModelConfiguration()
    config.computeUnits = .cpuAndNeuralEngine
    if let fn = functionName {
        config.functionName = fn
    }

    let loadStart = Date()
    let model: MLModel
    do {
        model = try MLModel(contentsOf: compiledURL, configuration: config)
        let elapsed = Date().timeIntervalSince(loadStart)
        print("✅ Loaded on ANE in \(String(format: "%.1f", elapsed))s")
    } catch {
        print("❌ ANE load failed: \(error)")
        print("   Trying CPU+GPU fallback...")
        config.computeUnits = .cpuAndGPU
        do {
            let _ = try MLModel(contentsOf: compiledURL, configuration: config)
            print("⚠️  Loaded on CPU+GPU (NOT ANE)")
        } catch {
            print("❌ CPU+GPU also failed: \(error)")
        }
        exit(1)
    }

    // Step 3: Print model details
    print("\n--- Step 3: Model Details ---")
    let desc = model.modelDescription
    print("Inputs:")
    for (name, feature) in desc.inputDescriptionsByName {
        let shape: String
        if let constraint = feature.multiArrayConstraint {
            shape = "\(constraint.shape)"
        } else {
            shape = feature.type.rawValue.description
        }
        print("  \(name): \(shape)")
    }
    print("Outputs:")
    for (name, feature) in desc.outputDescriptionsByName {
        let shape: String
        if let constraint = feature.multiArrayConstraint {
            shape = "\(constraint.shape)"
        } else {
            shape = feature.type.rawValue.description
        }
        print("  \(name): \(shape)")
    }

    // Step 4: Try ALL compute unit options to see what works
    print("\n--- Step 4: Compute Unit Compatibility ---")
    let units: [(String, MLComputeUnits)] = [
        ("cpuOnly",           .cpuOnly),
        ("cpuAndGPU",         .cpuAndGPU),
        ("cpuAndNeuralEngine", .cpuAndNeuralEngine),
        ("all",               .all),
    ]
    for (label, unit) in units {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = unit
        if let fn = functionName {
            cfg.functionName = fn
        }
        do {
            let _ = try MLModel(contentsOf: compiledURL, configuration: cfg)
            print("  ✅ \(label)")
        } catch {
            print("  ❌ \(label): \(error.localizedDescription)")
        }
    }

    // Cleanup test compiled model
    try? fm.removeItem(at: compiledURL)

    print("\n=== Test Complete ===")
}

// Parse arguments
let args = CommandLine.arguments
guard args.count >= 2 else {
    print("Usage: swift \(args[0]) <path_to.mlpackage> [functionName]")
    print("Example: swift test_mlpackage_ane.swift ../qwen3_5_v4_lut4_p2/embed_single.mlpackage")
    exit(1)
}

let packagePath = args[1]
let functionName: String? = args.count >= 3 ? args[2] : nil

testMLPackageANE(packagePath: packagePath, functionName: functionName)
