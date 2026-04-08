#!/usr/bin/env python3
"""Diagnostic: sub-stage error analysis for _chunk_gated_delta_rule.

Instruments the chunk delta rule to compare fp32 vs fp16 at every
internal sub-stage, then optionally exports to CoreML and compares
ANE vs CPU.

Goal: find the FIRST intermediate tensor that becomes materially wrong
when run in fp16 (which is what ANE uses due to compute_precision=FLOAT16).

Usage:
    python3 scripts_qwen3_5/diag_chunk_substage.py
"""
import sys, os, gc, time, shutil, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from anemll.models.qwen3_5_model import Qwen35LinearAttention, _l2norm

# ── Config matching Qwen3.5-4B linear attention ──
BATCH      = 1
NUM_HEADS  = 32
K_DIM      = 128
V_DIM      = 128
CHUNK_SIZE = 16
SEQ_LEN    = 64   # 4 chunks — fast but enough to show accumulation
N_CHUNKS   = SEQ_LEN // CHUNK_SIZE
SEED       = 42

OUT_DIR    = "/tmp/diag_chunk_substage"


# ── Metrics ──
def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.flatten().double()
    b_f = b.flatten().double()
    if a_f.norm() < 1e-10 and b_f.norm() < 1e-10:
        return 1.0
    return F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()


def report(label: str, ref: torch.Tensor, test: torch.Tensor, indent: int = 2):
    diff = (ref.double() - test.double()).abs()
    cos  = cosine_sim(ref, test)
    max_abs  = diff.max().item()
    mean_abs = diff.mean().item()
    ref_scale = ref.double().abs().max().item()
    rel_max  = max_abs / max(ref_scale, 1e-10)
    prefix = " " * indent
    print(f"{prefix}{label:45s}  cos={cos:.10f}  max_abs={max_abs:.6e}  "
          f"mean_abs={mean_abs:.6e}  rel_max={rel_max:.4e}  scale={ref_scale:.3e}")
    return {"cos": cos, "max_abs": max_abs, "mean_abs": mean_abs, "rel_max": rel_max}


def report_per_head(label: str, ref: torch.Tensor, test: torch.Tensor,
                    max_heads: int = 5):
    """Report worst heads for a (B, H, ...) tensor."""
    assert ref.shape[1] == NUM_HEADS
    diffs = []
    for h in range(NUM_HEADS):
        d = (ref[:, h].double() - test[:, h].double()).abs().max().item()
        diffs.append((h, d))
    diffs.sort(key=lambda x: -x[1])
    worst = diffs[:max_heads]
    heads_str = ", ".join(f"h{h}={d:.4e}" for h, d in worst)
    print(f"    {label} worst heads: {heads_str}")


# ── Instrumented _chunk_gated_delta_rule ──
def chunk_gdr_instrumented(
    query, key, value, g, beta,
    chunk_size=CHUNK_SIZE,
    initial_state=None,
    math_dtype=torch.float32,
):
    """Same as _chunk_gated_delta_rule but returns dict of intermediates."""
    intermediates = {}

    # ── Stage 1: Input preparation ──
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(math_dtype) for x in (query, key, value, beta, g)
    ]
    intermediates["s1_query_raw"]  = query.clone()
    intermediates["s1_key_raw"]    = key.clone()

    query = _l2norm(query, dim=-1)
    key   = _l2norm(key, dim=-1)
    intermediates["s1_query_l2normed"] = query.clone()
    intermediates["s1_key_l2normed"]   = key.clone()

    batch_size = BATCH
    num_heads  = NUM_HEADS
    seq_len    = SEQ_LEN
    k_dim      = K_DIM
    v_dim      = V_DIM
    pad_size   = (chunk_size - seq_len % chunk_size) % chunk_size
    query      = F.pad(query, (0, 0, 0, pad_size))
    key        = F.pad(key,   (0, 0, 0, pad_size))
    value      = F.pad(value, (0, 0, 0, pad_size))
    beta       = F.pad(beta,  (0, pad_size))
    g          = F.pad(g,     (0, pad_size))
    total_sequence_length = seq_len + pad_size
    scale = 1 / (k_dim ** 0.5)
    query = query * scale
    intermediates["s1_query_scaled"] = query.clone()

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    intermediates["s1_v_beta"] = v_beta.clone()
    intermediates["s1_k_beta"] = k_beta.clone()

    n_chunks = total_sequence_length // chunk_size
    query, key, value, k_beta, v_beta = [
        x.reshape(batch_size, num_heads, n_chunks, chunk_size, k_dim if idx != 2 and idx != 4 else v_dim)
        for idx, x in enumerate((query, key, value, k_beta, v_beta))
    ]
    g = g.reshape(batch_size, num_heads, n_chunks, chunk_size)
    intermediates["s1_g_chunks"] = g.clone()

    # ── Stage 2: Cumulative g + decay mask ──
    tril_ones    = torch.tril(torch.ones(chunk_size, chunk_size, device=query.device, dtype=math_dtype))
    strict_lower = torch.tril(torch.ones(chunk_size, chunk_size, device=query.device, dtype=math_dtype), diagonal=-1)

    g = (tril_ones @ g.unsqueeze(-1)).squeeze(-1)
    intermediates["s2_g_cumulative"] = g.clone()

    decay_raw  = (g.unsqueeze(-1) - g.unsqueeze(-2)) * tril_ones
    intermediates["s2_decay_raw"] = decay_raw.clone()

    decay_mask = decay_raw.exp() * tril_ones
    intermediates["s2_decay_mask"] = decay_mask.clone()

    # ── Stage 3: Intra-chunk attention (Woodbury) ──
    attn_base  = -((k_beta @ key.transpose(-1, -2)) * decay_mask) * strict_lower
    intermediates["s3_attn_base"] = attn_base.clone()

    attn_rows = [attn_base[..., 0:1, :]]
    for i in range(1, chunk_size):
        row = attn_base[..., i, :i].clone()
        sub = torch.cat([prev_row[..., :i] for prev_row in attn_rows[:i]], dim=-2)
        updated_row = row + (row.unsqueeze(-1) * sub).sum(-2)
        tail = attn_base[..., i:i+1, i:]
        full_row = torch.cat([updated_row.unsqueeze(-2), tail], dim=-1)
        attn_rows.append(full_row)

    attn = torch.cat(attn_rows, dim=-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    intermediates["s3_attn_woodbury"] = attn.clone()

    value = attn @ v_beta
    intermediates["s3_value_after_attn"] = value.clone()

    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    intermediates["s3_k_cumdecay"] = k_cumdecay.clone()

    # ── Stage 4: Inter-chunk recurrence ──
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_dim, v_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    strict_lower_diag1 = torch.tril(torch.ones(chunk_size, chunk_size, device=query.device, dtype=math_dtype))
    core_attn_chunks = []

    for ci in range(n_chunks):
        q_i, k_i, v_i = query[:, :, ci], key[:, :, ci], value[:, :, ci]
        attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, ci]) * strict_lower_diag1
        v_prime = k_cumdecay[:, :, ci] @ last_recurrent_state
        v_new   = v_i - v_prime
        attn_inter = (q_i * g[:, :, ci, :, None].exp()) @ last_recurrent_state
        output_i = attn_inter + attn_i @ v_new
        core_attn_chunks.append(output_i.unsqueeze(2))

        last_recurrent_state = (
            last_recurrent_state * g[:, :, ci, -1, None, None].exp()
            + (k_i * (g[:, :, ci, -1, None] - g[:, :, ci]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

        intermediates[f"s4_chunk{ci}_attn"]       = attn_i.clone()
        intermediates[f"s4_chunk{ci}_v_prime"]     = v_prime.clone()
        intermediates[f"s4_chunk{ci}_v_new"]       = v_new.clone()
        intermediates[f"s4_chunk{ci}_attn_inter"]  = attn_inter.clone()
        intermediates[f"s4_chunk{ci}_output"]      = output_i.clone()
        intermediates[f"s4_chunk{ci}_next_state"]  = last_recurrent_state.clone()

    # ── Stage 5: Final outputs ──
    core_attn_out = torch.cat(core_attn_chunks, dim=2)
    core_attn_out = core_attn_out.reshape(batch_size, num_heads, total_sequence_length, v_dim)
    core_attn_out = core_attn_out[:, :, :seq_len]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    intermediates["s5_final_output"] = core_attn_out.clone()
    intermediates["s5_final_state"]  = last_recurrent_state.clone()

    return core_attn_out, last_recurrent_state, intermediates


# ── CoreML export wrapper (for ANE comparison) ──
class ChunkGDRModule(nn.Module):
    """Wraps _chunk_gated_delta_rule for CoreML tracing."""
    def forward(self, query, key, value, g, beta, initial_state):
        out, state = Qwen35LinearAttention._chunk_gated_delta_rule(
            query, key, value, g, beta,
            chunk_size=CHUNK_SIZE,
            initial_state=initial_state,
            output_final_state=True,
            expected_batch_size=BATCH,
            expected_num_heads=NUM_HEADS,
            expected_seq_len=SEQ_LEN,
            expected_k_dim=K_DIM,
            expected_v_dim=V_DIM,
            math_dtype=torch.float32,
        )
        return out, state


# ── Sub-stage CoreML wrappers ──
class Stage2CumGDecay(nn.Module):
    """Just cumulative g + decay_mask computation."""
    def __init__(self):
        super().__init__()
        tril = torch.tril(torch.ones(CHUNK_SIZE, CHUNK_SIZE))
        self.register_buffer("tril_ones", tril)

    def forward(self, g_chunks):
        # g_chunks: (B, H, n_chunks, chunk_size)
        g_cum = (self.tril_ones @ g_chunks.unsqueeze(-1)).squeeze(-1)
        decay_raw = (g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)) * self.tril_ones
        decay_mask = decay_raw.exp() * self.tril_ones
        return g_cum, decay_mask


class Stage4SingleChunk(nn.Module):
    """Single inter-chunk update step."""
    def __init__(self):
        super().__init__()
        tril = torch.tril(torch.ones(CHUNK_SIZE, CHUNK_SIZE))
        self.register_buffer("strict_lower_diag1", tril)

    def forward(self, q_i, k_i, v_i, k_cumdecay_i, decay_mask_i, g_i, state):
        # q_i, k_i: (B, H, CS, K_DIM), v_i: (B, H, CS, V_DIM)
        # k_cumdecay_i: (B, H, CS, K_DIM), decay_mask_i: (B, H, CS, CS)
        # g_i: (B, H, CS), state: (B, H, K_DIM, V_DIM)
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask_i) * self.strict_lower_diag1
        v_prime = k_cumdecay_i @ state
        v_new = v_i - v_prime
        attn_inter = (q_i * g_i[:, :, :, None].exp()) @ state
        output = attn_inter + attn @ v_new
        next_state = (
            state * g_i[:, :, -1, None, None].exp()
            + (k_i * (g_i[:, :, -1, None] - g_i).exp()[..., None]).transpose(-1, -2) @ v_new
        )
        return output, next_state


def generate_inputs(dtype=torch.float32, realistic_g=True, seed=SEED):
    """Generate realistic inputs for _chunk_gated_delta_rule."""
    torch.manual_seed(seed)
    # Input shapes: (B, S, H, D) for query/key/value, (B, S, H) for g/beta
    query = torch.randn(BATCH, SEQ_LEN, NUM_HEADS, K_DIM, dtype=dtype) * 0.1
    key   = torch.randn(BATCH, SEQ_LEN, NUM_HEADS, K_DIM, dtype=dtype) * 0.1
    value = torch.randn(BATCH, SEQ_LEN, NUM_HEADS, V_DIM, dtype=dtype) * 0.1
    if realistic_g:
        # Realistic g: negative values from -A_log * softplus(a + dt_bias)
        # Typical range: [-3, -0.01]
        g = -torch.rand(BATCH, SEQ_LEN, NUM_HEADS, dtype=dtype).abs() * 2.0 - 0.1
    else:
        g = torch.randn(BATCH, SEQ_LEN, NUM_HEADS, dtype=dtype) * 0.5
    beta = torch.sigmoid(torch.randn(BATCH, SEQ_LEN, NUM_HEADS, dtype=dtype))
    initial_state = torch.zeros(BATCH, NUM_HEADS, K_DIM, V_DIM, dtype=dtype)
    return query, key, value, g, beta, initial_state


def phase1_cpu_substage_comparison():
    """Compare fp32 vs fp16 at each sub-stage (CPU only)."""
    print("\n" + "=" * 100)
    print("PHASE 1: CPU sub-stage comparison (fp32 reference vs fp16)")
    print("=" * 100)
    print(f"Config: B={BATCH} H={NUM_HEADS} K={K_DIM} V={V_DIM} "
          f"S={SEQ_LEN} CS={CHUNK_SIZE} n_chunks={N_CHUNKS}")

    # Generate inputs in fp32
    q32, k32, v32, g32, b32, s32 = generate_inputs(torch.float32)
    # fp16 copies
    q16 = q32.half()
    k16 = k32.half()
    v16 = v32.half()
    g16 = g32.half()
    b16 = b32.half()
    s16 = s32.half()

    print("\n[1] Running fp32 instrumented...")
    out32, state32, interm32 = chunk_gdr_instrumented(
        q32.clone(), k32.clone(), v32.clone(), g32.clone(), b32.clone(),
        initial_state=s32.clone(), math_dtype=torch.float32,
    )

    print("[2] Running fp16 instrumented...")
    out16, state16, interm16 = chunk_gdr_instrumented(
        q16.clone(), k16.clone(), v16.clone(), g16.clone(), b16.clone(),
        initial_state=s16.clone(), math_dtype=torch.float16,
    )

    # Also run recurrent for reference
    print("[3] Running recurrent fp32 (reference)...")
    out_rec, state_rec = Qwen35LinearAttention._recurrent_gated_delta_rule(
        q32.clone(), k32.clone(), v32.clone(), g32.clone(), b32.clone(),
        recurrent_state=s32.clone(),
        output_final_state=True,
        expected_batch_size=BATCH,
        expected_num_heads=NUM_HEADS,
        expected_seq_len=SEQ_LEN,
        expected_k_dim=K_DIM,
        expected_v_dim=V_DIM,
        math_dtype=torch.float32,
    )

    # ── Stage-by-stage comparison ──
    print("\n" + "-" * 100)
    print("STAGE-BY-STAGE: fp32 chunk vs fp16 chunk")
    print("-" * 100)

    stages = sorted(set(k.rsplit("_", 1)[0] if k[0] == 's' else k
                        for k in interm32.keys()))
    # Actually, just iterate all keys in order
    keys = sorted(interm32.keys())

    results = {}
    for key in keys:
        if key in interm16:
            r = report(key, interm32[key], interm16[key])
            results[key] = r
            # Per-head analysis for tensors with head dimension
            if interm32[key].dim() >= 2 and interm32[key].shape[1] == NUM_HEADS:
                report_per_head(key, interm32[key], interm16[key])

    # ── Summary: first bad stage ──
    print("\n" + "-" * 100)
    print("SUMMARY: ordered by max_abs error")
    print("-" * 100)
    sorted_results = sorted(results.items(), key=lambda x: -x[1]["max_abs"])
    for i, (key, r) in enumerate(sorted_results[:15]):
        flag = " *** WORST" if i == 0 else ""
        print(f"  {i+1:2d}. {key:45s}  max_abs={r['max_abs']:.6e}  "
              f"cos={r['cos']:.10f}  rel_max={r['rel_max']:.4e}{flag}")

    # ── Cross-algorithm comparison ──
    print("\n" + "-" * 100)
    print("CROSS-ALGORITHM: chunk fp32 vs recurrent fp32")
    print("-" * 100)
    report("output (chunk_fp32 vs rec_fp32)", out32, out_rec)
    report("state  (chunk_fp32 vs rec_fp32)", state32, state_rec)

    print("\n" + "-" * 100)
    print("CROSS-ALGORITHM: chunk fp16 vs recurrent fp32")
    print("-" * 100)
    report("output (chunk_fp16 vs rec_fp32)", out16, out_rec)
    report("state  (chunk_fp16 vs rec_fp32)", state16, state_rec)

    print("\n" + "-" * 100)
    print("CROSS-ALGORITHM: chunk fp32 vs chunk fp16")
    print("-" * 100)
    report("output (chunk_fp32 vs chunk_fp16)", out32, out16)
    report("state  (chunk_fp32 vs chunk_fp16)", state32, state16)

    # ── Decay value range analysis ──
    print("\n" + "-" * 100)
    print("DECAY VALUE RANGE ANALYSIS")
    print("-" * 100)
    dr32 = interm32["s2_decay_raw"]
    print(f"  decay_raw fp32: min={dr32.min().item():.4f}  max={dr32.max().item():.4f}")
    print(f"  decay_raw fp32 range per chunk:")
    for ci in range(N_CHUNKS):
        chunk_dr = dr32[:, :, ci]
        print(f"    chunk {ci}: min={chunk_dr.min().item():.4f}  max={chunk_dr.max().item():.4f}")

    dm32 = interm32["s2_decay_mask"]
    print(f"  decay_mask fp32: min={dm32.min().item():.6e}  max={dm32.max().item():.6e}")
    dm16 = interm16["s2_decay_mask"]
    print(f"  decay_mask fp16: min={dm16.min().item():.6e}  max={dm16.max().item():.6e}")

    # g values that would cause exp overflow/underflow in fp16
    g_cum32 = interm32["s2_g_cumulative"]
    g_vals = g_cum32.flatten()
    print(f"\n  g_cumulative fp32: min={g_vals.min().item():.4f}  max={g_vals.max().item():.4f}")
    # fp16 exp safe range: roughly [-17, 11]
    n_overflow = (g_vals > 11.0).sum().item()
    n_underflow = (g_vals < -17.0).sum().item()
    print(f"  g values causing fp16 exp overflow (>11): {n_overflow}/{g_vals.numel()}")
    print(f"  g values causing fp16 exp underflow (<-17): {n_underflow}/{g_vals.numel()}")

    # ── Chunk-by-chunk error accumulation ──
    print("\n" + "-" * 100)
    print("CHUNK-BY-CHUNK ERROR ACCUMULATION")
    print("-" * 100)
    for ci in range(N_CHUNKS):
        state_key = f"s4_chunk{ci}_next_state"
        out_key   = f"s4_chunk{ci}_output"
        if state_key in interm32 and state_key in interm16:
            sr = report(f"chunk {ci} state", interm32[state_key], interm16[state_key])
        if out_key in interm32 and out_key in interm16:
            or_ = report(f"chunk {ci} output", interm32[out_key], interm16[out_key])

    return interm32, interm16, out32, out16, state32, state16


def phase2_ane_comparison(interm32, out32_cpu, state32_cpu):
    """Export full _chunk_gated_delta_rule to CoreML, run on ANE, compare."""
    print("\n" + "=" * 100)
    print("PHASE 2: ANE comparison (CoreML export + ANE predict)")
    print("=" * 100)

    import coremltools as ct

    q32, k32, v32, g32, b32, s32 = generate_inputs(torch.float32)

    # Trace the module
    print("\n[1] Tracing ChunkGDRModule...")
    model = ChunkGDRModule()
    model.eval()
    with torch.no_grad():
        example_inputs = (q32, k32, v32, g32, b32, s32)
        traced = torch.jit.trace(model, example_inputs)

    # Convert to CoreML
    print("[2] Converting to CoreML (compute_precision=FLOAT16)...")
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    try:
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="query",  shape=q32.shape),
                ct.TensorType(name="key",    shape=k32.shape),
                ct.TensorType(name="value",  shape=v32.shape),
                ct.TensorType(name="g",      shape=g32.shape),
                ct.TensorType(name="beta",   shape=b32.shape),
                ct.TensorType(name="initial_state", shape=s32.shape),
            ],
            outputs=[
                ct.TensorType(name="output"),
                ct.TensorType(name="final_state"),
            ],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
        )
    except Exception as e:
        print(f"  ** CoreML CONVERT FAILED: {e}")
        traceback.print_exc()
        return

    model_path = os.path.join(OUT_DIR, "chunk_gdr.mlpackage")
    mlmodel.save(model_path)
    print(f"    Converted in {time.time()-t0:.1f}s")
    del mlmodel; gc.collect()

    # Load on ANE
    print("[3] Loading on ANE (CPU_AND_NE)...")
    t0 = time.time()
    try:
        ane_model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        print(f"    Loaded in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  ** ANE LOAD FAILED: {e}")
        return

    # Also load on CPU_ONLY for comparison
    print("[4] Loading on CPU_ONLY (for comparison)...")
    cpu_model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_ONLY)

    # Prepare inputs (fp16 as numpy)
    inputs_np = {
        "query":         q32.numpy().astype(np.float16),
        "key":           k32.numpy().astype(np.float16),
        "value":         v32.numpy().astype(np.float16),
        "g":             g32.numpy().astype(np.float16),
        "beta":          b32.numpy().astype(np.float16),
        "initial_state": s32.numpy().astype(np.float16),
    }

    # Run on ANE
    print("[5] Running predict on ANE...")
    t0 = time.time()
    ane_out = ane_model.predict(inputs_np)
    print(f"    ANE predict: {time.time()-t0:.3f}s")

    # Run on CPU (CoreML)
    print("[6] Running predict on CPU (CoreML)...")
    cpu_out = cpu_model.predict(inputs_np)

    # Extract results
    ane_output = torch.from_numpy(np.array(ane_out["output"]))
    ane_state  = torch.from_numpy(np.array(ane_out["final_state"]))
    cpu_output = torch.from_numpy(np.array(cpu_out["output"]))
    cpu_state  = torch.from_numpy(np.array(cpu_out["final_state"]))

    # Compare
    print("\n" + "-" * 100)
    print("ANE vs CPU fp32 PyTorch (reference)")
    print("-" * 100)
    report("output (ANE vs CPU_fp32_pytorch)", out32_cpu, ane_output)
    report("state  (ANE vs CPU_fp32_pytorch)", state32_cpu, ane_state)

    print("\n" + "-" * 100)
    print("CPU CoreML (fp16) vs CPU fp32 PyTorch")
    print("-" * 100)
    report("output (CoreML_CPU vs CPU_fp32_pytorch)", out32_cpu, cpu_output)
    report("state  (CoreML_CPU vs CPU_fp32_pytorch)", state32_cpu, cpu_state)

    print("\n" + "-" * 100)
    print("ANE vs CPU CoreML (fp16) — pure ANE lowering error")
    print("-" * 100)
    report("output (ANE vs CoreML_CPU)", cpu_output, ane_output)
    report("state  (ANE vs CoreML_CPU)", cpu_state, ane_state)

    del ane_model, cpu_model; gc.collect()


def phase3_substage_ane(interm32):
    """Export individual sub-stages to CoreML, compare ANE vs CPU."""
    print("\n" + "=" * 100)
    print("PHASE 3: Sub-stage ANE isolation")
    print("=" * 100)

    import coremltools as ct

    q32, k32, v32, g32, b32, s32 = generate_inputs(torch.float32)

    # Run fp32 instrumented to get intermediate inputs for sub-stages
    _, _, interm = chunk_gdr_instrumented(
        q32.clone(), k32.clone(), v32.clone(), g32.clone(), b32.clone(),
        initial_state=s32.clone(), math_dtype=torch.float32,
    )

    # ── Sub-stage A: Cumulative G + Decay Mask ──
    print("\n--- Sub-stage A: Cumulative G + Decay Mask ---")
    g_chunks_fp32 = interm["s1_g_chunks"]  # (B, H, n_chunks, CS)

    model_a = Stage2CumGDecay()
    model_a.eval()
    with torch.no_grad():
        traced_a = torch.jit.trace(model_a, (g_chunks_fp32,))

    try:
        ml_a = ct.convert(
            traced_a,
            inputs=[ct.TensorType(name="g_chunks", shape=g_chunks_fp32.shape)],
            outputs=[ct.TensorType(name="g_cum"), ct.TensorType(name="decay_mask")],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
        )
        path_a = os.path.join(OUT_DIR, "stage_a.mlpackage")
        ml_a.save(path_a)
        del ml_a; gc.collect()

        ane_a = ct.models.MLModel(path_a, compute_units=ct.ComputeUnit.CPU_AND_NE)
        cpu_a = ct.models.MLModel(path_a, compute_units=ct.ComputeUnit.CPU_ONLY)

        inp_a = {"g_chunks": g_chunks_fp32.numpy().astype(np.float16)}
        out_ane_a = ane_a.predict(inp_a)
        out_cpu_a = cpu_a.predict(inp_a)

        ref_gcum = interm["s2_g_cumulative"]
        ref_decay = interm["s2_decay_mask"]

        ane_gcum = torch.from_numpy(np.array(out_ane_a["g_cum"]))
        ane_decay = torch.from_numpy(np.array(out_ane_a["decay_mask"]))
        cpu_gcum = torch.from_numpy(np.array(out_cpu_a["g_cum"]))
        cpu_decay = torch.from_numpy(np.array(out_cpu_a["decay_mask"]))

        report("g_cum      (ANE vs fp32_ref)", ref_gcum, ane_gcum)
        report("g_cum      (CoreML_CPU vs fp32_ref)", ref_gcum, cpu_gcum)
        report("g_cum      (ANE vs CoreML_CPU)", cpu_gcum, ane_gcum)
        report("decay_mask (ANE vs fp32_ref)", ref_decay, ane_decay)
        report("decay_mask (CoreML_CPU vs fp32_ref)", ref_decay, cpu_decay)
        report("decay_mask (ANE vs CoreML_CPU)", cpu_decay, ane_decay)

        del ane_a, cpu_a; gc.collect()
    except Exception as e:
        print(f"  ** Sub-stage A failed: {e}")
        traceback.print_exc()

    # ── Sub-stage B: Single inter-chunk step ──
    print("\n--- Sub-stage B: Single Inter-Chunk Step (chunk 0) ---")
    # Get inputs for chunk 0 from fp32 reference
    # Need: q_i, k_i, v_i, k_cumdecay_i, decay_mask_i, g_i, state
    # These are post-reshape, post-Woodbury tensors

    # Re-run fp32 to get properly shaped intermediates
    query_chunks = interm["s1_query_scaled"].reshape(BATCH, NUM_HEADS, N_CHUNKS, CHUNK_SIZE, K_DIM)
    # Actually we need the query AFTER reshape. Let me re-extract from the full computation.
    # The intermediates store pre-chunk shapes. Let me get post-Woodbury values.

    # For simplicity, compute the needed inputs on CPU
    q32_t, k32_t, v32_t, b32_t, g32_t = [
        x.transpose(1, 2).contiguous() for x in (q32, k32, v32, b32, g32)
    ]
    q32_n = _l2norm(q32_t, dim=-1) * (1.0 / K_DIM**0.5)
    k32_n = _l2norm(k32_t, dim=-1)
    v32_t_padded = v32_t  # no padding needed for seq_len=64

    v_beta_32 = v32_t_padded * b32_t.unsqueeze(-1)
    k_beta_32 = k32_n * b32_t.unsqueeze(-1)

    q_chunks = q32_n.reshape(BATCH, NUM_HEADS, N_CHUNKS, CHUNK_SIZE, K_DIM)
    k_chunks = k32_n.reshape(BATCH, NUM_HEADS, N_CHUNKS, CHUNK_SIZE, K_DIM)
    v_chunks = interm["s3_value_after_attn"]  # post-Woodbury value
    k_cumdecay = interm["s3_k_cumdecay"]
    decay_mask = interm["s2_decay_mask"]
    g_cum = interm["s2_g_cumulative"]

    q_i = q_chunks[:, :, 0]           # (B, H, CS, K)
    k_i = k_chunks[:, :, 0]           # (B, H, CS, K)
    v_i = v_chunks[:, :, 0]           # (B, H, CS, V)
    kcd_i = k_cumdecay[:, :, 0]       # (B, H, CS, K)
    dm_i = decay_mask[:, :, 0]        # (B, H, CS, CS)
    g_i = g_cum[:, :, 0]              # (B, H, CS)
    state_0 = s32.clone()             # (B, H, K, V) zero initial state

    model_b = Stage4SingleChunk()
    model_b.eval()
    with torch.no_grad():
        traced_b = torch.jit.trace(model_b, (q_i, k_i, v_i, kcd_i, dm_i, g_i, state_0))

    try:
        ml_b = ct.convert(
            traced_b,
            inputs=[
                ct.TensorType(name="q_i",           shape=q_i.shape),
                ct.TensorType(name="k_i",           shape=k_i.shape),
                ct.TensorType(name="v_i",           shape=v_i.shape),
                ct.TensorType(name="k_cumdecay_i",  shape=kcd_i.shape),
                ct.TensorType(name="decay_mask_i",   shape=dm_i.shape),
                ct.TensorType(name="g_i",            shape=g_i.shape),
                ct.TensorType(name="state",          shape=state_0.shape),
            ],
            outputs=[
                ct.TensorType(name="output_chunk"),
                ct.TensorType(name="next_state"),
            ],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
        )
        path_b = os.path.join(OUT_DIR, "stage_b.mlpackage")
        ml_b.save(path_b)
        del ml_b; gc.collect()

        ane_b = ct.models.MLModel(path_b, compute_units=ct.ComputeUnit.CPU_AND_NE)
        cpu_b = ct.models.MLModel(path_b, compute_units=ct.ComputeUnit.CPU_ONLY)

        inp_b = {
            "q_i":          q_i.numpy().astype(np.float16),
            "k_i":          k_i.numpy().astype(np.float16),
            "v_i":          v_i.numpy().astype(np.float16),
            "k_cumdecay_i": kcd_i.numpy().astype(np.float16),
            "decay_mask_i":  dm_i.numpy().astype(np.float16),
            "g_i":           g_i.numpy().astype(np.float16),
            "state":         state_0.numpy().astype(np.float16),
        }
        out_ane_b = ane_b.predict(inp_b)
        out_cpu_b = cpu_b.predict(inp_b)

        ref_out0 = interm["s4_chunk0_output"]
        ref_st0  = interm["s4_chunk0_next_state"]

        ane_out0 = torch.from_numpy(np.array(out_ane_b["output_chunk"]))
        ane_st0  = torch.from_numpy(np.array(out_ane_b["next_state"]))
        cpu_out0 = torch.from_numpy(np.array(out_cpu_b["output_chunk"]))
        cpu_st0  = torch.from_numpy(np.array(out_cpu_b["next_state"]))

        report("chunk0 output (ANE vs fp32_ref)", ref_out0, ane_out0)
        report("chunk0 output (CoreML_CPU vs fp32_ref)", ref_out0, cpu_out0)
        report("chunk0 output (ANE vs CoreML_CPU)", cpu_out0, ane_out0)
        report("chunk0 state  (ANE vs fp32_ref)", ref_st0, ane_st0)
        report("chunk0 state  (CoreML_CPU vs fp32_ref)", ref_st0, cpu_st0)
        report("chunk0 state  (ANE vs CoreML_CPU)", cpu_st0, ane_st0)

        # Per-head state error
        report_per_head("chunk0 state ANE vs fp32", ref_st0, ane_st0)

        del ane_b, cpu_b; gc.collect()
    except Exception as e:
        print(f"  ** Sub-stage B failed: {e}")
        traceback.print_exc()


def phase4_detailed_exp_analysis(interm32, interm16):
    """Deep-dive into exp() and accumulation-related sub-stages."""
    print("\n" + "=" * 100)
    print("PHASE 4: Detailed exp() / accumulation error analysis")
    print("=" * 100)

    # 1. g_cumulative analysis
    g_cum32 = interm32["s2_g_cumulative"]  # (B, H, n_chunks, CS)
    g_cum16 = interm16["s2_g_cumulative"]

    print("\n--- g_cumulative per-position error ---")
    for pos in range(CHUNK_SIZE):
        ref = g_cum32[:, :, :, pos]
        tst = g_cum16[:, :, :, pos]
        d = (ref.double() - tst.double()).abs()
        print(f"  pos {pos:2d}: max_abs={d.max().item():.6e}  "
              f"mean_abs={d.mean().item():.6e}")

    # 2. decay_raw analysis: differences g_i - g_j across chunk positions
    dr32 = interm32["s2_decay_raw"]  # (B, H, n_chunks, CS, CS)
    dr16 = interm16["s2_decay_raw"]
    print("\n--- decay_raw error analysis ---")
    report("decay_raw overall", dr32, dr16)

    # 3. decay_mask (exp of decay_raw)
    dm32 = interm32["s2_decay_mask"]
    dm16 = interm16["s2_decay_mask"]
    print("\n--- decay_mask error analysis ---")
    report("decay_mask overall", dm32, dm16)

    # Where decay_mask error is worst: find positions
    diff_dm = (dm32.double() - dm16.double()).abs()
    B, H, NC, CS1, CS2 = diff_dm.shape
    max_idx = diff_dm.reshape(-1).argmax().item()
    b_i = max_idx // (H * NC * CS1 * CS2)
    rem = max_idx % (H * NC * CS1 * CS2)
    h_i = rem // (NC * CS1 * CS2)
    rem = rem % (NC * CS1 * CS2)
    c_i = rem // (CS1 * CS2)
    rem = rem % (CS1 * CS2)
    r_i = rem // CS2
    col_i = rem % CS2
    print(f"  Worst decay_mask error at: batch={b_i} head={h_i} chunk={c_i} "
          f"row={r_i} col={col_i}")
    print(f"    fp32 value: {dm32[b_i, h_i, c_i, r_i, col_i].item():.10e}")
    print(f"    fp16 value: {dm16[b_i, h_i, c_i, r_i, col_i].item():.10e}")
    print(f"    decay_raw fp32: {dr32[b_i, h_i, c_i, r_i, col_i].item():.10f}")

    # 4. Woodbury attn error growth
    ab32 = interm32["s3_attn_base"]
    aw32 = interm32["s3_attn_woodbury"]
    ab16 = interm16["s3_attn_base"]
    aw16 = interm16["s3_attn_woodbury"]
    print("\n--- Woodbury amplification ---")
    report("attn_base", ab32, ab16)
    report("attn_woodbury (after iteration)", aw32, aw16)
    # Ratio of error amplification
    base_err = (ab32.double() - ab16.double()).abs().max().item()
    wood_err = (aw32.double() - aw16.double()).abs().max().item()
    if base_err > 0:
        print(f"  Woodbury amplification ratio: {wood_err/base_err:.2f}x")

    # Per-row error in Woodbury
    print("\n--- Woodbury per-row error (all chunks) ---")
    for row in range(CHUNK_SIZE):
        ref_row = aw32[:, :, :, row, :]
        tst_row = aw16[:, :, :, row, :]
        d = (ref_row.double() - tst_row.double()).abs()
        print(f"  row {row:2d}: max_abs={d.max().item():.6e}  "
              f"mean_abs={d.mean().item():.6e}")

    # 5. Inter-chunk state accumulation
    print("\n--- Inter-chunk state error growth ---")
    for ci in range(N_CHUNKS):
        st32 = interm32[f"s4_chunk{ci}_next_state"]
        st16 = interm16[f"s4_chunk{ci}_next_state"]
        d = (st32.double() - st16.double()).abs()
        print(f"  chunk {ci} state: max_abs={d.max().item():.6e}  "
              f"mean_abs={d.mean().item():.6e}  "
              f"cos={cosine_sim(st32, st16):.10f}")

    # 6. v_prime error (k_cumdecay @ state — critical matmul)
    print("\n--- v_prime (k_cumdecay @ state) error ---")
    for ci in range(N_CHUNKS):
        vp32 = interm32[f"s4_chunk{ci}_v_prime"]
        vp16 = interm16[f"s4_chunk{ci}_v_prime"]
        if vp32.abs().max().item() > 1e-10:  # skip if near zero
            report(f"chunk {ci} v_prime", vp32, vp16)

    # 7. exp(g) values at inter-chunk boundaries
    print("\n--- exp(g) at chunk boundaries (decay factor) ---")
    g_cum32 = interm32["s2_g_cumulative"]
    for ci in range(N_CHUNKS):
        g_last = g_cum32[:, :, ci, -1]  # last position in chunk
        exp_g = g_last.exp()
        print(f"  chunk {ci} exp(g_last): min={exp_g.min().item():.6e}  "
              f"max={exp_g.max().item():.6e}  "
              f"mean={exp_g.mean().item():.6e}")


def main():
    print("=" * 100)
    print("DIAGNOSTIC: _chunk_gated_delta_rule sub-stage error analysis")
    print(f"Config: B={BATCH} H={NUM_HEADS} K={K_DIM} V={V_DIM} "
          f"S={SEQ_LEN} CS={CHUNK_SIZE} n_chunks={N_CHUNKS}")
    print("=" * 100)

    # Phase 1: CPU-only sub-stage comparison (fast)
    interm32, interm16, out32, out16, state32, state16 = phase1_cpu_substage_comparison()

    # Phase 4: Detailed analysis (CPU only, fast)
    phase4_detailed_exp_analysis(interm32, interm16)

    # Phase 2: ANE full-pipeline comparison (requires CoreML, slower)
    try:
        phase2_ane_comparison(interm32, out32, state32)
    except ImportError:
        print("\n[SKIP] Phase 2: coremltools not available")

    # Phase 3: ANE sub-stage isolation (requires CoreML, slowest)
    try:
        phase3_substage_ane(interm32)
    except ImportError:
        print("\n[SKIP] Phase 3: coremltools not available")

    # Cleanup
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR, ignore_errors=True)

    print("\n" + "=" * 100)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
