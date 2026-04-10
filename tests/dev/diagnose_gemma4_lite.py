#!/usr/bin/env python3
"""Lightweight diagnostic for Gemma4 weight loading correctness."""
import os, sys, torch, json
os.environ['ANEMLL_ALLOW_MISSING_WEIGHTS'] = '1'
sys.path.insert(0, '/Users/yw68/Anemll')

import anemll.models.gemma4_model as g4
g4.MODEL_DTYPE = torch.float32
from anemll.models.gemma4_model import Gemma4Config, Gemma4ForCausalLM

model_path = '/Users/yw68/local_llm/models/google__gemma-4-E4B-it'
cfg = Gemma4Config.from_json(f'{model_path}/config.json')
cfg.context_length = 512; cfg.state_length = 512

model = Gemma4ForCausalLM(cfg)
model.load_pretrained_weights(model_path)
model.float()
model.eval()

# Load only specific weights we need
from safetensors.torch import load_file
import glob

def get_raw_weight(name):
    """Load a specific weight by name from safetensors."""
    for sf_file in sorted(glob.glob(f'{model_path}/*.safetensors')):
        st = load_file(sf_file, device='cpu')
        full_name = f'model.language_model.{name}'
        if full_name in st:
            return st[full_name].float()
    return None

# === 1. Embedding check ===
print("=== Embedding ===")
raw_emb_w = get_raw_weight('embed_tokens.weight')
our_emb_w = model.model.embed_tokens.weight.data.float()
diff = (raw_emb_w - our_emb_w).abs().max().item()
print(f"  embed_tokens weight shape: {raw_emb_w.shape}, max_diff: {diff:.8f}")
del raw_emb_w

# === 2. Layer 0 weights ===
print("\n=== Layer 0 weights ===")
for name in ['input_layernorm.weight', 'self_attn.q_proj.weight', 'self_attn.k_proj.weight',
             'self_attn.v_proj.weight', 'self_attn.o_proj.weight',
             'self_attn.q_norm.weight', 'self_attn.k_norm.weight',
             'post_attention_layernorm.weight']:
    raw = get_raw_weight(f'layers.0.{name}')
    if raw is None:
        print(f"  {name}: NOT FOUND in safetensors!")
        continue
    
    # Navigate our model
    parts = name.split('.')
    obj = model.model.layers[0]
    for p in parts[:-1]:
        obj = getattr(obj, p)
    our_w = getattr(obj, parts[-1])
    
    if hasattr(our_w, 'data'):
        our_w = our_w.data.float()
    
    # For Conv2d, squeeze out the spatial dims
    if our_w.dim() == 4:
        our_w = our_w.squeeze(-1).squeeze(-1)
    
    diff = (raw - our_w).abs().max().item()
    match_str = "OK" if diff < 1e-5 else "MISMATCH!"
    print(f"  {name}: raw_shape={raw.shape}, our_shape={our_w.shape}, max_diff={diff:.8f} {match_str}")
    del raw

# === 3. Layer scalars ===
print("\n=== Layer scalars ===")
non_one = []
for i in range(42):
    raw = get_raw_weight(f'layers.{i}.layer_scalar')
    our = model.model.layers[i].layer_scalar.data.float()
    if raw is not None:
        diff = (raw - our).abs().max().item()
        if abs(raw.item() - 1.0) > 0.001 or diff > 1e-5:
            non_one.append((i, raw.item(), our.item(), diff))
    else:
        non_one.append((i, None, our.item(), -1))

if non_one:
    for i, raw_v, our_v, diff in non_one:
        print(f"  Layer {i}: raw={raw_v}, our={our_v}, diff={diff:.8f}")
else:
    print("  All layer scalars are 1.0 and match")

# === 4. PLE weights ===
print("\n=== PLE weights ===")
raw = get_raw_weight('embed_tokens_per_layer.weight')
if raw is not None:
    our = model.model.embed_tokens_per_layer.weight.data.float()
    diff = (raw - our).abs().max().item()
    print(f"  embed_tokens_per_layer: shape={raw.shape}, max_diff={diff:.8f}")
    del raw

raw = get_raw_weight('per_layer_model_projection.weight')
if raw is not None:
    our = model.model.per_layer_model_projection.weight.data.float()
    if our.dim() == 4:
        our = our.squeeze(-1).squeeze(-1)
    diff = (raw - our).abs().max().item()
    print(f"  per_layer_model_projection: raw={raw.shape}, our={our.shape}, max_diff={diff:.8f}")
    del raw

# === 5. LM head (tied to embed_tokens) ===
print("\n=== LM Head ===")
# With tied weights, lm_head = embed_tokens
lm_w = model.model.embed_tokens.weight.data.float()  # [262144, 2560]
vocab_split = cfg.vocab_size // 16
lm1_w = model.lm_head16_1.weight.data.float().squeeze(-1).squeeze(-1)  # [split, 2560]
diff = (lm1_w - lm_w[:vocab_split]).abs().max().item()
print(f"  lm_head16_1 vs embed_tokens[:{vocab_split}]: max_diff={diff:.8f}")

# === 6. Quick forward test ===
print("\n=== Forward test (BOS token) ===")
with torch.no_grad():
    NEGINF = -1e9
    tok = torch.tensor([[2]], dtype=torch.long)
    pos = torch.tensor([0], dtype=torch.long)
    cp = torch.tensor([0], dtype=torch.long)
    um = torch.zeros((1, 1, 512, 1), dtype=torch.float32)
    um[0, 0, 0, 0] = 1.0
    cm = torch.full((1, 1, 1, 512), NEGINF, dtype=torch.float32)
    cm[0, 0, 0, 0] = 0.0
    
    logits = model(tok, um, pos, cm, cp)
    if isinstance(logits, tuple):
        logits = torch.cat(list(logits), dim=-1)
    
    print(f"  Logits shape: {logits.shape}")
    print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(f'{model_path}/tokenizer.json')
    topk = logits.topk(5, dim=-1)
    print(f"  Top 5:")
    for i in range(5):
        tid = topk.indices[0,0,i].item()
        decoded = tokenizer.decode([tid])
        print(f"    {tid:>6d} ({decoded:>12s}): {topk.values[0,0,i].item():.4f}")
