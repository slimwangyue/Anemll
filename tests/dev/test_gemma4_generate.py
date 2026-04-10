#!/usr/bin/env python3
"""Multi-token generation test for Gemma4 model."""
import os, torch, sys
os.environ['ANEMLL_ALLOW_MISSING_WEIGHTS'] = '1'
sys.path.insert(0, '/Users/yw68/Anemll')
from anemll.models.gemma4_model import *
from tokenizers import Tokenizer

model_path = '/Users/yw68/local_llm/models/google__gemma-4-E4B-it'

cfg = Gemma4Config.from_json(f'{model_path}/config.json')
cfg.context_length = 512
cfg.state_length = 512

model = Gemma4ForCausalLM(cfg)
model.load_pretrained_weights(model_path)
model.eval()
for p in model.parameters():
    p.requires_grad = False

tokenizer = Tokenizer.from_file(f'{model_path}/tokenizer.json')
EOS_TOKEN_ID = 1  # Gemma EOS

prompt = "Hello, I am"
encoded = tokenizer.encode(prompt)
input_ids = [2] + encoded.ids  # BOS=2
print(f"Input tokens: {input_ids}")
print(f"Prompt: {prompt}")

max_new_tokens = 30
generated_ids = list(input_ids)

with torch.no_grad():
    m = model.model
    cm = torch.zeros((1, 1, 1, 512), dtype=MODEL_DTYPE)
    um = torch.zeros((1, 1, 512, 1), dtype=MODEL_DTYPE)
    NEGINF = torch.finfo(MODEL_DTYPE).min

    # Process each input token
    for idx, tok_id in enumerate(generated_ids):
        tok = torch.tensor([[tok_id]], dtype=torch.long)
        pos = torch.tensor([idx], dtype=torch.long)
        cp = torch.tensor([idx], dtype=torch.long)
        um_step = torch.zeros((1, 1, 512, 1), dtype=MODEL_DTYPE)
        um_step[0, 0, idx, 0] = 1.0

        # Set causal mask: attend to positions 0..idx, mask positions idx+1..511
        cm_step = torch.full((1, 1, 1, 512), NEGINF, dtype=MODEL_DTYPE)
        cm_step[0, 0, 0, :idx+1] = 0.0

        output = model(tok, um_step, pos, cm_step, cp)
        if isinstance(output, tuple):
            logits = torch.cat(list(output), dim=-1)
        else:
            logits = output

    # Generate new tokens
    for gen_step in range(max_new_tokens):
        next_token = logits.argmax(dim=-1).item()
        generated_ids.append(next_token)

        if next_token == EOS_TOKEN_ID:
            break

        idx = len(generated_ids) - 1
        tok = torch.tensor([[next_token]], dtype=torch.long)
        pos = torch.tensor([idx], dtype=torch.long)
        cp = torch.tensor([idx], dtype=torch.long)
        um_step = torch.zeros((1, 1, 512, 1), dtype=MODEL_DTYPE)
        um_step[0, 0, idx, 0] = 1.0

        cm_step = torch.full((1, 1, 1, 512), NEGINF, dtype=MODEL_DTYPE)
        cm_step[0, 0, 0, :idx+1] = 0.0

        output = model(tok, um_step, pos, cm_step, cp)
        if isinstance(output, tuple):
            logits = torch.cat(list(output), dim=-1)
        else:
            logits = output

    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"\nGenerated ({len(generated_ids)} tokens):")
    print(text)
    print(f"Token IDs: {generated_ids[:20]}...")
