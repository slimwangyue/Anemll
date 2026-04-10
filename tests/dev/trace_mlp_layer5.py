#!/usr/bin/env python3
"""Trace MLP internals at layer 5 to find NaN source."""
import os, torch, math
os.environ['ANEMLL_ALLOW_MISSING_WEIGHTS'] = '1'
import sys; sys.path.insert(0, '/Users/yw68/Anemll')
from anemll.models.gemma4_model import *

cfg = Gemma4Config.from_json('/Users/yw68/local_llm/models/google__gemma-4-E4B-it/config.json')
cfg.context_length = 512
cfg.state_length = 512

model = Gemma4ForCausalLM(cfg)
model.load_pretrained_weights('/Users/yw68/local_llm/models/google__gemma-4-E4B-it')
model.eval()
for p in model.parameters():
    p.requires_grad = False

input_ids = torch.tensor([[2]], dtype=torch.long)
pos = torch.tensor([0], dtype=torch.long)
cm = torch.zeros((1, 1, 1, 512), dtype=MODEL_DTYPE)
cp = torch.tensor([0], dtype=torch.long)
um = torch.zeros((1, 1, 512, 1), dtype=MODEL_DTYPE)
um[0, 0, 0, 0] = 1.0

with torch.no_grad():
    m = model.model
    h = m.embed_tokens(input_ids) * (cfg.hidden_size ** 0.5)
    h = h.to(MODEL_DTYPE)
    ple = m.compute_per_layer_embeddings(input_ids)
    for i in range(5):
        h = m.process_layer_regular(i, h, pos, cm, cp, per_layer_emb=ple, update_mask=um)

    # Layer 5 manual trace - use same flow as process_layer_regular
    i = 5
    layer = m.layers[i]
    ple_slice = m.get_per_layer_emb_slice(ple, i)
    if ple_slice is not None:
        h = layer.apply_ple(h, ple_slice)

    rotary_emb = m.get_rotary_embeddings_s(cp, i)
    rotary_dim = m._get_rotary_dim(i)
    norm = layer.input_layernorm(h)

    is_shared = cfg.is_kv_shared_layer(i)
    print(f"Layer {i}: is_shared={is_shared}, type={cfg.layer_types[i]}")
    
    q, k, v = layer.self_attn.get_new_kv_cache(norm, cp, rotary_emb, rotary_dim)
    
    if not is_shared:
        if cfg.layer_types[i] == "full_attention":
            m._store_kv_global(i, k, v, cp, um)
        else:
            m._store_kv_local(i, k, v, cp, um)
    
    kc, vc = m._get_kv_cache_for_layer(i, cp)
    attn_out = layer.self_attn.forward_regular(
        norm, q, kv_cache_layer=(kc, vc), causal_mask=cm, current_pos=cp, layer_idx=i
    )
    
    print(f'Attention output: max={attn_out.abs().max():.4f}, nan={attn_out.isnan().any()}')
    attn_out_norm = layer.post_attention_layernorm(attn_out)
    print(f'Post-attn norm: max={attn_out_norm.abs().max():.4f}, nan={attn_out_norm.isnan().any()}')
    print(f'layer_scalar: {layer.layer_scalar.item():.6f}')
    h = h + attn_out_norm * layer.layer_scalar
    print(f'Post-attn residual: max={h.abs().max():.4f}, nan={h.isnan().any()}')

    residual = h
    h_normed = layer.pre_feedforward_layernorm(h)
    print(f'MLP input (normed): max={h_normed.abs().max():.4f}, nan={h_normed.isnan().any()}')

    mlp = layer.mlp
    x = h_normed.permute(0, 2, 1).unsqueeze(2).to(MODEL_DTYPE)
    print(f'MLP x for conv2d: max={x.abs().max():.4f}')

    a = mlp.gate_proj(x)
    print(f'gate_proj: max={a.abs().max():.4f}, nan={a.isnan().any()}, inf={a.isinf().any()}')
    b = mlp.up_proj(x)
    print(f'up_proj:   max={b.abs().max():.4f}, nan={b.isnan().any()}, inf={b.isinf().any()}')

    act = mlp.act_fn(a)
    print(f'gelu(gate): max={act.abs().max():.4f}, nan={act.isnan().any()}, inf={act.isinf().any()}')

    d = act * b
    print(f'gate*up:   max={d.abs().max():.4f}, nan={d.isnan().any()}, inf={d.isinf().any()}')

    e = mlp.down_proj(d)
    print(f'down_proj: max={e.abs().max():.4f}, nan={e.isnan().any()}, inf={e.isinf().any()}')

    # Also try with float32 computation
    print("\n--- FP32 MLP trace ---")
    x32 = h_normed.float().permute(0, 2, 1).unsqueeze(2)
    gw = mlp.gate_proj.weight.float()
    uw = mlp.up_proj.weight.float()
    dw = mlp.down_proj.weight.float()

    a32 = torch.nn.functional.conv2d(x32, gw)
    print(f'gate_proj (f32): max={a32.abs().max():.4f}')
    b32 = torch.nn.functional.conv2d(x32, uw)
    print(f'up_proj (f32):   max={b32.abs().max():.4f}')

    def gelu_tanh(x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))

    act32 = gelu_tanh(a32)
    print(f'gelu(gate) (f32): max={act32.abs().max():.4f}')
    d32 = act32 * b32
    print(f'gate*up (f32):   max={d32.abs().max():.4f}')
    e32 = torch.nn.functional.conv2d(d32, dw)
    print(f'down_proj (f32): max={e32.abs().max():.4f}')
