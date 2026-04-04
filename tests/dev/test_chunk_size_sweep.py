#!/usr/bin/env python3
"""Test different chunk_sizes for op count and accuracy.
Goal: Find chunk_size that minimizes total MIL ops while maintaining accuracy."""
import sys, os, time
sys.path.insert(0, '/Users/yw68/Anemll')
os.environ['QWEN35_NUM_CHUNKS'] = '4'
os.environ['QWEN35_HF_MODEL'] = '/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B'
os.environ['QWEN35_STATIC_PREFILL_CTX'] = '2048'

import torch, numpy as np
from anemll.models.qwen3_5_model import Qwen35Config, Qwen35ForCausalLM

model_dir = '/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B'

print("Loading model...", flush=True)
cfg = Qwen35Config.from_json(os.path.join(model_dir, "config.json"))
cfg.context_length = 2048
cfg.state_length = 2048
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(model_dir), "Failed to load weights"
model.eval()
for p in model.parameters():
    p.requires_grad = False
print("Model loaded.", flush=True)

# Test chunk_size accuracy against reference
torch.manual_seed(42)
hidden = torch.randn(1, 64, cfg.hidden_size, dtype=torch.float16, device='cpu')

# Reference: chunk_size=64 (original)
ref_out = model.model.process_layers_prefill_export_local_state(
    hidden.clone(),
    position_ids=torch.arange(64),
    causal_mask=torch.zeros(1,1,64,2048, dtype=torch.float16),
    current_pos=torch.tensor([0]),
    start_layer=0, end_layer=4,
    linear_conv_state=None, linear_recurrent_state=None,
    valid_len=torch.tensor([64]),
    chunk_size=64,
)

print(f"\nChunk size sweep (4 layers, seq=64, accuracy vs cs=64):")
print(f"{'cs':>4} | {'cos_sim':>8} | {'max_diff':>10}")
print("-" * 30)

for cs in [1, 2, 4, 8, 16, 32, 64]:
    out = model.model.process_layers_prefill_export_local_state(
        hidden.clone(),
        position_ids=torch.arange(64),
        causal_mask=torch.zeros(1,1,64,2048, dtype=torch.float16),
        current_pos=torch.tensor([0]),
        start_layer=0, end_layer=4,
        linear_conv_state=None, linear_recurrent_state=None,
        valid_len=torch.tensor([64]),
        chunk_size=cs,
    )
    
    r = ref_out[0].flatten().float()
    o = out[0].flatten().float()
    cos = torch.nn.functional.cosine_similarity(r.unsqueeze(0), o.unsqueeze(0)).item()
    maxd = (r - o).abs().max().item()
    print(f"{cs:4d} | {cos:.6f} | {maxd:.6f}")

print("\nDone.", flush=True)
