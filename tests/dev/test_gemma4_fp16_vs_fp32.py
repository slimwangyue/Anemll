#!/usr/bin/env python3
"""Compare Gemma4 FP16 vs FP32 generation to check precision loss."""
import os, torch, sys
os.environ['ANEMLL_ALLOW_MISSING_WEIGHTS'] = '1'
sys.path.insert(0, '/Users/yw68/Anemll')

# First run with FP16 (default)
import anemll.models.gemma4_model as g4
from anemll.models.gemma4_model import Gemma4Config, Gemma4ForCausalLM

model_path = '/Users/yw68/local_llm/models/google__gemma-4-E4B-it'

cfg = Gemma4Config.from_json(f'{model_path}/config.json')
cfg.context_length = 512; cfg.state_length = 512

print("=== FP16 model ===")
model = Gemma4ForCausalLM(cfg)
model.load_pretrained_weights(model_path)
model.eval()
for p in model.parameters():
    p.requires_grad = False

NEGINF = torch.finfo(g4.MODEL_DTYPE).min
input_tokens = [2, 9259, 236764, 564, 1006]  # "<bos>Hello, I am"

def generate_tokens(model, input_tokens, num_gen=10, dtype=g4.MODEL_DTYPE):
    NEGINF_VAL = torch.finfo(dtype).min if dtype != torch.float32 else -1e9
    tokens = list(input_tokens)
    with torch.no_grad():
        for idx, tok_id in enumerate(tokens):
            tok = torch.tensor([[tok_id]], dtype=torch.long)
            pos = torch.tensor([idx], dtype=torch.long)
            cp = torch.tensor([idx], dtype=torch.long)
            um = torch.zeros((1, 1, 512, 1), dtype=dtype)
            um[0, 0, idx, 0] = 1.0
            cm = torch.full((1, 1, 1, 512), NEGINF_VAL, dtype=dtype)
            cm[0, 0, 0, :idx+1] = 0.0
            output = model(tok, um, pos, cm, cp)
            if isinstance(output, tuple):
                logits = torch.cat(list(output), dim=-1)
            else:
                logits = output
        
        # Print logits info for last prompt token
        topk = logits.topk(10, dim=-1)
        print(f"After prompt, top-10 tokens:")
        for i in range(10):
            print(f"  {topk.indices[0,0,i].item():>6d}: {topk.values[0,0,i].item():.4f}")
        
        # Generate
        for g in range(num_gen):
            next_token = logits.argmax(dim=-1).item()
            tokens.append(next_token)
            idx = len(tokens) - 1
            tok = torch.tensor([[next_token]], dtype=torch.long)
            pos = torch.tensor([idx], dtype=torch.long)
            cp = torch.tensor([idx], dtype=torch.long)
            um = torch.zeros((1, 1, 512, 1), dtype=dtype)
            um[0, 0, idx, 0] = 1.0
            cm = torch.full((1, 1, 1, 512), NEGINF_VAL, dtype=dtype)
            cm[0, 0, 0, :idx+1] = 0.0
            output = model(tok, um, pos, cm, cp)
            if isinstance(output, tuple):
                logits = torch.cat(list(output), dim=-1)
            else:
                logits = output
    return tokens

fp16_tokens = generate_tokens(model, input_tokens, num_gen=15)
print(f"FP16 generated IDs: {fp16_tokens}")
