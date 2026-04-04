#!/usr/bin/env python3
"""Diagnose the exp(g) amplification and test mitigations.

Key finding: at seq=256 with random g, error explodes to max_abs=22.
Root cause: per-chunk error amplified exponentially by exp(g_cum) across chunks.

Test: (1) pure-decay g (all negative), (2) small g, (3) chunk_size variations.
"""
import sys, os, torch, math
import torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from anemll.models.qwen3_5_model import Qwen35LinearAttention


def compare(name, a, b, indent=2):
    a_f, b_f = a.float().flatten(), b.float().flatten()
    max_abs = (a_f - b_f).abs().max().item()
    mean_abs = (a_f - b_f).abs().mean().item()
    cos = F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()
    print(f"{' '*indent}{name:55s} max={max_abs:.6e} mean={mean_abs:.6e} cos={cos:.8f}")
    return max_abs


def run(seq_len, chunk_size, n_heads=16, k_dim=128, v_dim=128, g_scale=0.5, g_bias=0.0, seed=42):
    torch.manual_seed(seed)
    q = torch.randn(1, seq_len, n_heads, k_dim, dtype=torch.float16)
    k = torch.randn(1, seq_len, n_heads, k_dim, dtype=torch.float16)
    v = torch.randn(1, seq_len, n_heads, v_dim, dtype=torch.float16)
    g = (torch.randn(1, seq_len, n_heads, dtype=torch.float16) * g_scale + g_bias)
    beta = torch.sigmoid(torch.randn(1, seq_len, n_heads, dtype=torch.float16))
    init = torch.randn(1, n_heads, k_dim, v_dim, dtype=torch.float32) * 0.1

    out_rec, st_rec = Qwen35LinearAttention._recurrent_gated_delta_rule(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        recurrent_state=init.clone(), output_final_state=True
    )
    out_chu, st_chu = Qwen35LinearAttention._chunk_gated_delta_rule(
        q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(),
        chunk_size=chunk_size, initial_state=init.clone(), output_final_state=True
    )
    m = compare(f"output  (g_scale={g_scale}, g_bias={g_bias:+.1f}, cs={chunk_size})", out_chu, out_rec)
    s = compare(f"state   (g_scale={g_scale}, g_bias={g_bias:+.1f}, cs={chunk_size})", st_chu.float(), st_rec.float())
    return m, s


if __name__ == "__main__":
    print("="*90)
    print("TEST 1: Effect of g magnitude on error (seq=128, 128d, 16 heads)")
    print("="*90)
    for g_scale in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
        print(f"\n  g_scale={g_scale}:")
        run(128, 16, g_scale=g_scale)

    print(f"\n{'='*90}")
    print("TEST 2: Pure decay (g always negative) vs mixed (seq=256)")
    print(f"{'='*90}")
    print("\n  Random g (mixed sign):")
    run(256, 16, g_scale=0.5, g_bias=0.0)
    print("\n  Pure decay (g_bias=-1.0):")
    run(256, 16, g_scale=0.5, g_bias=-1.0)
    print("\n  Strong decay (g_bias=-2.0):")
    run(256, 16, g_scale=0.5, g_bias=-2.0)
    print("\n  Very small g:")
    run(256, 16, g_scale=0.05, g_bias=0.0)

    print(f"\n{'='*90}")
    print("TEST 3: Effect of chunk_size (seq=128, realistic dims)")
    print(f"{'='*90}")
    for cs in [4, 8, 16, 32, 64]:
        if cs > 128:
            continue
        print(f"\n  chunk_size={cs} ({128//cs} chunks):")
        run(128, cs, g_scale=0.5)

    print(f"\n{'='*90}")
    print("TEST 4: Seq length scaling with controlled g (g_scale=0.1)")
    print(f"{'='*90}")
    for sl in [32, 64, 128, 256, 512]:
        print(f"\n  seq_len={sl} ({sl//16} chunks):")
        run(sl, 16, g_scale=0.1)

    print(f"\n{'='*90}")
    print("TEST 5: Seq length scaling with typical g (g_scale=0.5)")
    print(f"{'='*90}")
    for sl in [32, 64, 128, 256]:
        print(f"\n  seq_len={sl} ({sl//16} chunks):")
        run(sl, 16, g_scale=0.5)
