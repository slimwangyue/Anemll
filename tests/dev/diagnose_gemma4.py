#!/usr/bin/env python3
"""Layer-by-layer diagnostic: compare our Gemma4 against raw HF weight computation."""
import os, sys, torch, json, glob
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

# Load raw weights for comparison
from safetensors.torch import load_file
raw_weights = {}
for sf_file in sorted(glob.glob(f'{model_path}/*.safetensors')):
    st = load_file(sf_file)
    for k, v in st.items():
        if k.startswith('model.language_model.'):
            raw_weights[k.replace('model.language_model.', '')] = v.float()

token_ids = [2, 9259, 236764, 564, 1006]  # <bos> Hello, I am
print(f"Token IDs: {token_ids}")

# === 1. Embedding check ===
emb_weight = raw_weights['embed_tokens.weight']
scale = cfg.hidden_size ** 0.5
print(f"\n=== Embedding ===")
for tid in token_ids[:3]:
    raw_emb = emb_weight[tid] * scale
    our_emb = model.model.embed_tokens(torch.tensor([[tid]]))[0, 0]
    our_emb_scaled = our_emb * model.model.embedding_scale
    diff = (raw_emb - our_emb_scaled).abs().max().item()
    print(f"  Token {tid}: raw_norm={raw_emb.norm():.4f}, our_norm={our_emb_scaled.norm():.4f}, max_diff={diff:.6f}")

# === 2. First layer input_layernorm check ===
print(f"\n=== Layer 0 input_layernorm ===")
tok_t = torch.tensor([token_ids], dtype=torch.long)
with torch.no_grad():
    emb = model.model.embed_tokens(tok_t) * model.model.embedding_scale

    # Our layernorm
    our_ln = model.model.layers[0].input_layernorm(emb)
    
    # Raw layernorm
    ln_weight = raw_weights['layers.0.input_layernorm.weight']
    x = emb[0]
    rms = (x.pow(2).mean(-1, keepdim=True) + 1e-6).pow(-0.5)
    raw_ln = x * rms * ln_weight
    
    diff = (our_ln[0] - raw_ln).abs().max().item()
    print(f"  our_ln norm={our_ln[0,0].norm():.4f}, raw_ln norm={raw_ln[0].norm():.4f}, max_diff={diff:.6f}")

# === 3. Check q/k/v projections (layer 0, first token) ===
print(f"\n=== Layer 0 Q/K/V projections ===")
with torch.no_grad():
    # Use the normalized states as input
    input_to_attn = our_ln
    
    # Our model's projection (Conv2d)
    attn = model.model.layers[0].self_attn
    x_conv = input_to_attn.permute(0, 2, 1).unsqueeze(2).to(torch.float32)  # [B, H, 1, S]
    q_out = attn.q_proj(x_conv)  # [B, num_heads*head_dim, 1, S]
    k_out = attn.k_proj(x_conv)
    v_out = attn.v_proj(x_conv)
    q_out = q_out.squeeze(2).permute(0, 2, 1)  # [B, S, num_heads*head_dim]
    k_out = k_out.squeeze(2).permute(0, 2, 1)
    v_out = v_out.squeeze(2).permute(0, 2, 1)
    
    # Raw projection (Linear)
    q_weight = raw_weights['layers.0.self_attn.q_proj.weight']  # [out, in]
    k_weight = raw_weights['layers.0.self_attn.k_proj.weight']
    v_weight = raw_weights['layers.0.self_attn.v_proj.weight']
    
    x_flat = input_to_attn[0]  # [S, H]
    raw_q = x_flat @ q_weight.T
    raw_k = x_flat @ k_weight.T
    raw_v = x_flat @ v_weight.T
    
    q_diff = (q_out[0] - raw_q).abs().max().item()
    k_diff = (k_out[0] - raw_k).abs().max().item()
    v_diff = (v_out[0] - raw_v).abs().max().item()
    
    print(f"  Q: our_norm={q_out[0,0].norm():.4f}, raw_norm={raw_q[0].norm():.4f}, max_diff={q_diff:.6f}")
    print(f"  K: our_norm={k_out[0,0].norm():.4f}, raw_norm={raw_k[0].norm():.4f}, max_diff={k_diff:.6f}")
    print(f"  V: our_norm={v_out[0,0].norm():.4f}, raw_norm={raw_v[0].norm():.4f}, max_diff={v_diff:.6f}")

# === 4. Check PLE weights ===
print(f"\n=== PLE weights check ===")
if 'embed_tokens_per_layer.weight' in raw_weights:
    ple_w = raw_weights['embed_tokens_per_layer.weight']
    our_ple_w = model.model.embed_tokens_per_layer.weight.data.float()
    diff = (ple_w - our_ple_w).abs().max().item()
    print(f"  embed_tokens_per_layer: shape={ple_w.shape}, max_diff={diff:.6f}")
else:
    print("  WARNING: embed_tokens_per_layer.weight not found in raw weights!")

if 'per_layer_model_projection.weight' in raw_weights:
    proj_w = raw_weights['per_layer_model_projection.weight']
    # Our model uses Conv2d, so weight shape is [out, in, 1, 1]
    our_proj_w = model.model.per_layer_model_projection.weight.data.float().squeeze(-1).squeeze(-1)
    diff = (proj_w - our_proj_w).abs().max().item()
    print(f"  per_layer_model_projection: raw_shape={proj_w.shape}, our_shape={our_proj_w.shape}, max_diff={diff:.6f}")

# === 5. Check LM head ===
print(f"\n=== LM Head ===")
# For tied embeddings, lm_head.weight = embed_tokens.weight
lm_w = raw_weights['embed_tokens.weight']
# Our model splits into 16
our_lm_0 = model.lm_head16_1.weight.data.float().squeeze(-1).squeeze(-1)
# first chunk should match first vocab_split rows of embed_tokens
vocab_split = cfg.vocab_size // 16
lm_diff = (our_lm_0[:vocab_split] - lm_w[:vocab_split]).abs().max().item()
print(f"  lm_head16_1: max_diff vs embed_tokens[:vocab_split]={lm_diff:.6f}")

# === 6. Run single token through full model and check hidden states ===
print(f"\n=== Full forward (single token, BOS) ===")
with torch.no_grad():
    NEGINF = -1e9
    tok = torch.tensor([[2]], dtype=torch.long)  # BOS
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
    print(f"  Logits mean: {logits.mean():.4f}, std: {logits.std():.4f}")
    
    topk = logits.topk(5, dim=-1)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(f'{model_path}/tokenizer.json')
    print(f"  Top 5 after BOS:")
    for i in range(5):
        tid = topk.indices[0,0,i].item()
        decoded = tokenizer.decode([tid])
        print(f"    {tid:>6d} ({decoded:>12s}): {topk.values[0,0,i].item():.4f}")

print("\n=== Layer scalars ===")
for i in range(42):
    s = model.model.layers[i].layer_scalar.item()
    if abs(s - 1.0) > 0.001:
        print(f"  Layer {i}: {s:.6f}")
    elif i < 3 or i == 41:
        print(f"  Layer {i}: {s:.6f}")
