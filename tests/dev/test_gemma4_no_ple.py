#!/usr/bin/env python3
"""Test Gemma4 with PLE disabled to isolate the bug."""
import os, torch, sys
os.environ['ANEMLL_ALLOW_MISSING_WEIGHTS'] = '1'
sys.path.insert(0, '/Users/yw68/Anemll')

import anemll.models.gemma4_model as g4
g4.MODEL_DTYPE = torch.float32
from anemll.models.gemma4_model import Gemma4Config, Gemma4ForCausalLM

model_path = '/Users/yw68/local_llm/models/google__gemma-4-E4B-it'

cfg = Gemma4Config.from_json(f'{model_path}/config.json')
cfg.context_length = 512; cfg.state_length = 512

# Disable PLE
cfg.hidden_size_per_layer_input = 0

print("=== FP32 model WITHOUT PLE ===")
model = Gemma4ForCausalLM(cfg)
model.load_pretrained_weights(model_path)
model.float()
model.eval()
for p in model.parameters():
    p.requires_grad = False

# Test single token
input_ids = torch.tensor([[2]], dtype=torch.long)
pos = torch.tensor([0], dtype=torch.long)
cm = torch.full((1, 1, 1, 512), -1e9, dtype=torch.float32)
cm[0, 0, 0, 0] = 0.0
cp = torch.tensor([0], dtype=torch.long)
um = torch.zeros((1, 1, 512, 1), dtype=torch.float32)
um[0, 0, 0, 0] = 1.0

with torch.no_grad():
    output = model(input_ids, um, pos, cm, cp)
    if isinstance(output, tuple):
        logits = torch.cat(list(output), dim=-1)
    else:
        logits = output
    print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    topk = logits.topk(5, dim=-1)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(f'{model_path}/tokenizer.json')
    for i in range(5):
        tid = topk.indices[0,0,i].item()
        decoded = tokenizer.decode([tid])
        print(f"  {tid:>6d} ({decoded:>15s}): {topk.values[0,0,i].item():.4f}")

# Test "Hello, I am" prompt
print("\n=== Prompt: 'Hello, I am' ===")
prompt_tokens = [2, 9259, 236764, 564, 1006]
for idx, tok_id in enumerate(prompt_tokens):
    tok = torch.tensor([[tok_id]], dtype=torch.long)
    pos = torch.tensor([idx], dtype=torch.long)
    cp = torch.tensor([idx], dtype=torch.long)
    um = torch.zeros((1, 1, 512, 1), dtype=torch.float32)
    um[0, 0, idx, 0] = 1.0
    cm = torch.full((1, 1, 1, 512), -1e9, dtype=torch.float32)
    cm[0, 0, 0, :idx+1] = 0.0
    output = model(tok, um, pos, cm, cp)
    if isinstance(output, tuple):
        logits = torch.cat(list(output), dim=-1)
    else:
        logits = output

topk = logits.topk(10, dim=-1)
print(f"After prompt, top 10:")
for i in range(10):
    tid = topk.indices[0,0,i].item()
    decoded = tokenizer.decode([tid])
    print(f"  {tid:>6d} ({decoded:>15s}): {topk.values[0,0,i].item():.4f}")
