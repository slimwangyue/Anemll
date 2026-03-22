# ANEMLL Roadmap 

## Completed Milestones ✅
- [x] Initial Release - Basic conversion/test
- [x] Automated single script workflow
- [x] Documentation and guides
- [x] Swift Sample Code
- [x] iOS/SwiftUI Sample Application
- [x] Qwen Architecture Support
- [X] Model Quality testing  (/eval)
- [X] Gemma3 Architecture Support
- [x] Monolithic model support ( export_coreml)
- [x] Profiling/Benchmarking scripts
- [x] Qwen3.5-4B Milestone 1.3 — Dynamic KV Cache on ANE via tensor-value slice (`current_pos[0]`), 10.5 tok/s, LUT4 quantized
- [x] Qwen3.5-4B Milestone 1.4 — LUT6+Argmax LM Head: 60% size reduction (485 MB vs 1212 MB fp16), fused argmax on ANE, 9.9 tok/s end-to-end
- [x] Qwen3.5-4B Milestone 2.0 — Batch Prefill: 96 tok/s (7.4x speedup), true 256-token batched prefill with `valid_len` gating, dual-instance model loading, crossover at 32 tokens, multi-round + overflow validated

## In Progress Milestones 🚧
- [ ] Native ANE Quantization
- [ ] Variable Context Size
- [ ] Gemma 3N
- [ ] Improve Quantization Speed


## Upcoming Milestones 🚧
- [ ] KV-Cache quantization
- [ ] Add Int4 Linear Quantization
- [ ] Conversion/Quantization Pipeline Re-factoring
- [ ] Add LoRA Support
- [ ] Basic ANE Server Implementation w/Third-party API
- [ ] Add AFM-like Swift API to call models
- [ ] MoE models support



