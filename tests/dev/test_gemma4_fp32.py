#!/usr/bin/env python3
"""Test Gemma4 in full float32 to check if degeneration is FP16-related or a logic bug."""
import os, torch, sys
os.environ['ANEMLL_ALLOW_MISSING_WEIGHTS'] = '1'
sys.path.insert(0, '/Users/yw68/Anemll')

# Override MODEL_DTYPE to float32 before importing
import anemll.models.gemma4_model as g4
g4.MODEL_DTYPE = torch.float32
from anemll.models.gemma4_model import Gemma4Config, Gemma4ForCausalLM
from tokenizers import Tokenizer

model_path = '/Users/yw68/local_llm/models/google__gemma-4-E4B-it'

cfg = Gemma4Config.from_json(f'{model_path}/config.json')
cfg.context_length = 512; cfg.state_length = 512

print("=== FP32 model ===")
model = Gemma4ForCausalLM(cfg)
model.load_pretrained_weights(model_path)
model.float()  # ensure everything is float32
model.eval()
for p in model.parameters():
    p.requires_grad = False

tokenizer = Tokenizer.from_file(f'{model_path}/tokenizer.json')

prompt = "Hello, I am"
encoded = tokenizer.encode(prompt)
input_tokens = [2] + encoded.ids
print(f"Input tokens: {input_tokens}")
print(f"Prompt: {prompt}")

NEGINF = -1e9
generated = list(input_tokens)

with torch.no_grad():
    # Process prompt tokens
    for idx, tok_id in enumerate(generated):
        tok = torch.tensor([[tok_id]], dtype=torch.long)
        pos = torch.tensor([idx], dtype=torch.long)
        cp = torch.tensor([idx], dtype=torch.long)
        um = torch.zeros((1, 1, 512, 1), dtype=torch.float32)
        um[0, 0, idx, 0] = 1.0
        cm = torch.full((1, 1, 1, 512), NEGINF, dtype=torch.float32)
        cm[0, 0, 0, :idx+1] = 0.0
        output = model(tok, um, pos, cm, cp)
        if isinstance(output, tuple):
            logits = torch.cat(list(output), dim=-1)
        else:
            logits = output

    # Print logits for last prompt token
    topk = logits.topk(10, dim=-1)
    print(f"\nAfter prompt - top 10:")
    for i in range(10):
        tid = topk.indices[0,0,i].item()
        decoded = tokenizer.decode([tid])
        print(f"  {tid:>6d} ({decoded:>10s}): {topk.values[0,0,i].item():.4f}")

    # Generate
    for g in range(20):
        next_token = logits.argmax(dim=-1).item()
        generated.append(next_token)
        decoded_tok = tokenizer.decode([next_token])
        
        idx = len(generated) - 1
        tok = torch.tensor([[next_token]], dtype=torch.long)
        pos = torch.tensor([idx], dtype=torch.long)
        cp = torch.tensor([idx], dtype=torch.long)
        um = torch.zeros((1, 1, 512, 1), dtype=torch.float32)
        um[0, 0, idx, 0] = 1.0
        cm = torch.full((1, 1, 1, 512), NEGINF, dtype=torch.float32)
        cm[0, 0, 0, :idx+1] = 0.0
        output = model(tok, um, pos, cm, cp)
        if isinstance(output, tuple):
            logits = torch.cat(list(output), dim=-1)
        else:
            logits = output
        
        top1 = logits.argmax(dim=-1).item()
        top1_decoded = tokenizer.decode([top1])
        print(f"Step {g:2d}: generated '{decoded_tok}' (id={next_token}), next top1='{top1_decoded}' (id={top1}), logit={logits[0,0,top1].item():.3f}")

    text = tokenizer.decode(generated)
    print(f"\nFull output: {text}")
