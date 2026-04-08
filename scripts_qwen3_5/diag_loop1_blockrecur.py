#!/usr/bin/env python3
"""
Diagnostic: Block-recursive Loop 1 vs row-by-row baseline.

Tests the ISOLATED _chunk_gated_delta_rule function as a CoreML model.
Measures:
  1. CPU/PyTorch correctness (new vs baseline)
  2. MIL op count after conversion
  3. ANE loadability (error -14 or success)
  4. ANE vs CPU numerical agreement

Uses reduced dimensions for fast turnaround. The MIL op count for
Loop 1 scales linearly with n_chunks and num_heads so relative
comparison (baseline vs block-recursive) is valid at any size.
"""

import os, sys, time, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import coremltools as ct
from collections import Counter

# ─── Test dims (reduced for fast trace+convert, ~2 min total) ───
# Real model: 32 heads, 128 k/v dim, seq=512 → 16 chunks
# Test:       4 heads,  64 k/v dim, seq=128 → 4 chunks
NUM_HEADS = 4
K_DIM     = 64
V_DIM     = 64
CHUNK_SIZE = 32
SEQ_LEN   = 128
BATCH     = 1

# ─── Helpers ───
def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


# ─── BASELINE: row-by-row Loop 1 (original) ───
def chunk_gdr_baseline(query, key, value, g, beta, chunk_size=CHUNK_SIZE):
    """Original row-by-row implementation."""
    math_dtype = torch.float32
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(math_dtype) for x in (query, key, value, beta, g)
    ]
    query = _l2norm(query); key = _l2norm(key)
    batch_size, num_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0,0,0,pad_size)); key = F.pad(key, (0,0,0,pad_size))
    value = F.pad(value, (0,0,0,pad_size)); beta = F.pad(beta, (0,pad_size)); g = F.pad(g, (0,pad_size))
    total_seq = seq_len + pad_size
    scale = 1/(k_dim**0.5); query = query * scale
    v_beta = value * beta.unsqueeze(-1); k_beta = key * beta.unsqueeze(-1)
    n_chunks = total_seq // chunk_size
    query, key, value, k_beta, v_beta = [
        x.reshape(batch_size, num_heads, n_chunks, chunk_size, k_dim if idx not in (2,4) else v_dim)
        for idx, x in enumerate((query, key, value, k_beta, v_beta))
    ]
    g = g.reshape(batch_size, num_heads, n_chunks, chunk_size)
    tril_ones = torch.tril(torch.ones(chunk_size, chunk_size, dtype=math_dtype))
    strict_lower = torch.tril(torch.ones(chunk_size, chunk_size, dtype=math_dtype), diagonal=-1)
    g = (tril_ones @ g.unsqueeze(-1)).squeeze(-1)
    decay_raw = (g.unsqueeze(-1) - g.unsqueeze(-2)) * tril_ones
    decay_mask = decay_raw.exp() * tril_ones
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask) * strict_lower

    # === ORIGINAL ROW-BY-ROW LOOP 1 ===
    attn_rows = [attn[..., 0:1, :]]
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = torch.cat([prev_row[..., :i] for prev_row in attn_rows[:i]], dim=-2)
        updated_row = row + (row.unsqueeze(-1) * sub).sum(-2)
        tail = attn[..., i : i + 1, i:]
        full_row = torch.cat([updated_row.unsqueeze(-2), tail], dim=-1)
        attn_rows.append(full_row)
    attn = torch.cat(attn_rows, dim=-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = torch.zeros(batch_size, num_heads, k_dim, v_dim).to(value)
    strict_lower_diag1 = torch.tril(torch.ones(chunk_size, chunk_size, dtype=math_dtype))
    core_attn_chunks = []
    for i in range(n_chunks):
        q_i, k_i, v_i = query[:,:,i], key[:,:,i], value[:,:,i]
        attn2 = (q_i @ k_i.transpose(-1,-2)*decay_mask[:,:,i])*strict_lower_diag1
        v_prime = k_cumdecay[:,:,i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:,:,i,:,None].exp()) @ last_recurrent_state
        core_attn_chunks.append((attn_inter + attn2 @ v_new).unsqueeze(2))
        last_recurrent_state = (
            last_recurrent_state * g[:,:,i,-1,None,None].exp()
            + (k_i * (g[:,:,i,-1,None] - g[:,:,i]).exp()[...,None]).transpose(-1,-2) @ v_new
        )
    core_attn_out = torch.cat(core_attn_chunks, dim=2)
    core_attn_out = core_attn_out.reshape(batch_size, num_heads, total_seq, v_dim)[:,:,:seq_len]
    return core_attn_out.transpose(1,2).contiguous().to(initial_dtype), last_recurrent_state


# ─── BLOCK-RECURSIVE: BC=16 Loop 1 (current code) ───
def chunk_gdr_blockrecur(query, key, value, g, beta, chunk_size=CHUNK_SIZE, bc=16):
    """Block-recursive implementation with sub-block size bc."""
    math_dtype = torch.float32
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(math_dtype) for x in (query, key, value, beta, g)
    ]
    query = _l2norm(query); key = _l2norm(key)
    batch_size, num_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0,0,0,pad_size)); key = F.pad(key, (0,0,0,pad_size))
    value = F.pad(value, (0,0,0,pad_size)); beta = F.pad(beta, (0,pad_size)); g = F.pad(g, (0,pad_size))
    total_seq = seq_len + pad_size
    scale = 1/(k_dim**0.5); query = query * scale
    v_beta = value * beta.unsqueeze(-1); k_beta = key * beta.unsqueeze(-1)
    n_chunks = total_seq // chunk_size
    query, key, value, k_beta, v_beta = [
        x.reshape(batch_size, num_heads, n_chunks, chunk_size, k_dim if idx not in (2,4) else v_dim)
        for idx, x in enumerate((query, key, value, k_beta, v_beta))
    ]
    g = g.reshape(batch_size, num_heads, n_chunks, chunk_size)
    tril_ones = torch.tril(torch.ones(chunk_size, chunk_size, dtype=math_dtype))
    strict_lower = torch.tril(torch.ones(chunk_size, chunk_size, dtype=math_dtype), diagonal=-1)
    g = (tril_ones @ g.unsqueeze(-1)).squeeze(-1)
    decay_raw = (g.unsqueeze(-1) - g.unsqueeze(-2)) * tril_ones
    decay_mask = decay_raw.exp() * tril_ones
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask) * strict_lower

    # === BLOCK-RECURSIVE LOOP 1 ===
    n_blks = chunk_size // bc
    _blks = {}
    for _r in range(n_blks):
        for _c in range(_r + 1):
            _blks[(_r, _c)] = attn[..., _r*bc:(_r+1)*bc, _c*bc:(_c+1)*bc]
    _inv_d = {}
    for _b in range(n_blks):
        _A = _blks[(_b, _b)]
        _rows = [_A[..., 0:1, :]]
        for _i in range(1, bc):
            _row = _A[..., _i, :_i].clone()
            _sub = torch.cat([_pr[..., :_i] for _pr in _rows[:_i]], dim=-2)
            _urow = _row + (_row.unsqueeze(-1) * _sub).sum(-2)
            _tail = _A[..., _i:_i+1, _i:]
            _rows.append(torch.cat([_urow.unsqueeze(-2), _tail], dim=-1))
        _inv_d[_b] = torch.cat(_rows, dim=-2) + torch.eye(bc, dtype=attn.dtype, device=attn.device)
    _inv_f = {}
    for _b in range(n_blks):
        _inv_f[(_b, _b)] = _inv_d[_b]
    for _c in range(n_blks):
        for _r in range(_c + 1, n_blks):
            _acc = torch.zeros_like(_blks[(_r, _c)])
            for _m in range(_c, _r):
                _acc = _acc + _blks[(_r, _m)] @ _inv_f[(_m, _c)]
            _inv_f[(_r, _c)] = _inv_d[_r] @ _acc
    _result_rows = []
    for _r in range(n_blks):
        _rblks = []
        for _c in range(n_blks):
            if _c <= _r:
                _rblks.append(_inv_f[(_r, _c)])
            else:
                _rblks.append(torch.zeros_like(_blks[(_r, _r)]))
        _result_rows.append(torch.cat(_rblks, dim=-1))
    attn = torch.cat(_result_rows, dim=-2)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = torch.zeros(batch_size, num_heads, k_dim, v_dim).to(value)
    strict_lower_diag1 = torch.tril(torch.ones(chunk_size, chunk_size, dtype=math_dtype))
    core_attn_chunks = []
    for i in range(n_chunks):
        q_i, k_i, v_i = query[:,:,i], key[:,:,i], value[:,:,i]
        attn2 = (q_i @ k_i.transpose(-1,-2)*decay_mask[:,:,i])*strict_lower_diag1
        v_prime = k_cumdecay[:,:,i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:,:,i,:,None].exp()) @ last_recurrent_state
        core_attn_chunks.append((attn_inter + attn2 @ v_new).unsqueeze(2))
        last_recurrent_state = (
            last_recurrent_state * g[:,:,i,-1,None,None].exp()
            + (k_i * (g[:,:,i,-1,None] - g[:,:,i]).exp()[...,None]).transpose(-1,-2) @ v_new
        )
    core_attn_out = torch.cat(core_attn_chunks, dim=2)
    core_attn_out = core_attn_out.reshape(batch_size, num_heads, total_seq, v_dim)[:,:,:seq_len]
    return core_attn_out.transpose(1,2).contiguous().to(initial_dtype), last_recurrent_state


# ─── CoreML wrapper ───
class ChunkGDRModule(nn.Module):
    """Wraps a chunk_gdr function for CoreML tracing."""
    def __init__(self, func, chunk_size=CHUNK_SIZE):
        super().__init__()
        self.func = func
        self.chunk_size = chunk_size

    def forward(self, query, key, value, g, beta, initial_state):
        out, state = self.func(query, key, value, g, beta, self.chunk_size)
        return out, state


def make_random_inputs(seq_len=SEQ_LEN):
    """Create random inputs matching Qwen3.5-4B prefill dimensions."""
    torch.manual_seed(42)
    q = torch.randn(BATCH, seq_len, NUM_HEADS, K_DIM, dtype=torch.float16)
    k = torch.randn(BATCH, seq_len, NUM_HEADS, K_DIM, dtype=torch.float16)  # num_k_heads=16 but expanded to 32 before call
    v = torch.randn(BATCH, seq_len, NUM_HEADS, V_DIM, dtype=torch.float16)
    g = torch.randn(BATCH, seq_len, NUM_HEADS, dtype=torch.float16)
    beta = torch.sigmoid(torch.randn(BATCH, seq_len, NUM_HEADS, dtype=torch.float16))
    state = torch.zeros(BATCH, NUM_HEADS, K_DIM, V_DIM, dtype=torch.float16)
    return q, k, v, g, beta, state


def count_mil_ops(mlpackage_path):
    """Count MIL ops by type from an mlpackage."""
    try:
        spec = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_ONLY).get_spec()
        model_desc = spec.mlProgram
        ops_count = Counter()
        total = 0
        for func in model_desc.functions.values():
            for block in func.block_specializations.values():
                for op in block.operations:
                    ops_count[op.type] += 1
                    total += 1
        return total, dict(ops_count)
    except Exception as e:
        print(f"  [WARN] MIL op counting via proto failed: {e}")
        # Fallback: count from weight dir
        try:
            import glob
            weight_dir = os.path.join(mlpackage_path, "Data", "com.apple.CoreML", "weights")
            mil_dir = os.path.join(mlpackage_path, "Data", "com.apple.CoreML")
            # Just report -1
            return -1, {}
        except:
            return -1, {}


def trace_and_convert(module, inputs, name, output_dir):
    """Trace module and convert to CoreML mlpackage."""
    import logging
    # Suppress coremltools verbose output
    logging.getLogger("coremltools").setLevel(logging.WARNING)

    q, k, v, g, beta, state = inputs
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Trace
    print("  Tracing...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        traced = torch.jit.trace(module, (q, k, v, g, beta, state))
    print(f"  Traced in {time.time()-t0:.1f}s")

    # Convert
    mlpackage_path = os.path.join(output_dir, f"{name}.mlpackage")
    if os.path.exists(mlpackage_path):
        shutil.rmtree(mlpackage_path)
    print("  Converting to CoreML...", flush=True)
    t0 = time.time()

    # Redirect stdout to suppress "y" progress spam
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="query",   shape=q.shape),
                ct.TensorType(name="key",     shape=k.shape),
                ct.TensorType(name="value",   shape=v.shape),
                ct.TensorType(name="g",       shape=g.shape),
                ct.TensorType(name="beta",    shape=beta.shape),
                ct.TensorType(name="state",   shape=state.shape),
            ],
            outputs=[
                ct.TensorType(name="core_attn_out"),
                ct.TensorType(name="rec_state_out"),
            ],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
    mlmodel.save(mlpackage_path)
    elapsed = time.time() - t0
    print(f"  Converted in {elapsed:.1f}s -> {mlpackage_path}")
    return mlpackage_path


def test_ane_loadability(mlpackage_path, name):
    """Test if model loads on ANE without error -14."""
    print(f"\n  Testing ANE loadability for {name}...")
    try:
        t0 = time.time()
        model_ane = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        print(f"  ANE load: SUCCESS ({time.time()-t0:.1f}s)")
        return model_ane
    except Exception as e:
        print(f"  ANE load: FAILED - {e}")
        return None


def compare_ane_vs_cpu(mlpackage_path, name):
    """Run prediction on both ANE and CPU, compare outputs."""
    print(f"\n  Comparing ANE vs CPU for {name}...")

    # Load CPU
    try:
        model_cpu = ct.models.MLModel(mlpackage_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    except Exception as e:
        print(f"  CPU load failed: {e}")
        return None

    # Load ANE
    model_ane = test_ane_loadability(mlpackage_path, name)
    if model_ane is None:
        return None

    # Make inputs
    q, k, v, g, beta, state = make_random_inputs()
    # CoreML renames "state" to "state_workaround" to avoid keyword collision
    feed = {
        "query": q.numpy(),
        "key": k.numpy(),
        "value": v.numpy(),
        "g": g.numpy(),
        "beta": beta.numpy(),
        "state_workaround": state.numpy(),
    }

    # Predict
    print("  Running CPU prediction...", flush=True)
    out_cpu = model_cpu.predict(feed)
    print("  Running ANE prediction...", flush=True)
    out_ane = model_ane.predict(feed)

    # Compare
    results = {}
    for key_name in ["core_attn_out", "rec_state_out"]:
        cpu_val = out_cpu[key_name]
        ane_val = out_ane[key_name]
        diff = np.abs(cpu_val - ane_val)
        # Cosine similarity
        cpu_flat = cpu_val.flatten()
        ane_flat = ane_val.flatten()
        cos_sim = np.dot(cpu_flat, ane_flat) / (np.linalg.norm(cpu_flat) * np.linalg.norm(ane_flat) + 1e-12)
        results[key_name] = {
            "cos_sim": float(cos_sim),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
        }
        print(f"  {key_name}:")
        print(f"    cos_sim={cos_sim:.6f}  max_abs={diff.max():.6f}  mean_abs={diff.mean():.6f}")

    return results


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "qwen3_5_diag_loop1")
    os.makedirs(output_dir, exist_ok=True)

    inputs = make_random_inputs()
    q, k, v, g, beta, state = inputs

    # ─── Step 1: PyTorch correctness (baseline vs block-recursive) ───
    print("\n" + "="*60)
    print("  Step 1: PyTorch correctness check")
    print("="*60)
    with torch.no_grad():
        out_base, st_base = chunk_gdr_baseline(q, k, v, g, beta)
        out_br, st_br = chunk_gdr_blockrecur(q, k, v, g, beta)

    out_base_f = out_base.float().numpy().flatten()
    out_br_f = out_br.float().numpy().flatten()
    cos = np.dot(out_base_f, out_br_f) / (np.linalg.norm(out_base_f) * np.linalg.norm(out_br_f) + 1e-12)
    max_diff = np.abs(out_base_f - out_br_f).max()
    mean_diff = np.abs(out_base_f - out_br_f).mean()
    print(f"  core_attn_out:  cos={cos:.8f}  max_abs={max_diff:.8f}  mean_abs={mean_diff:.8f}")

    st_base_f = st_base.float().numpy().flatten()
    st_br_f = st_br.float().numpy().flatten()
    cos_st = np.dot(st_base_f, st_br_f) / (np.linalg.norm(st_base_f) * np.linalg.norm(st_br_f) + 1e-12)
    max_diff_st = np.abs(st_base_f - st_br_f).max()
    print(f"  rec_state_out:  cos={cos_st:.8f}  max_abs={max_diff_st:.8f}")

    if cos < 0.999:
        print("  [FAIL] Block-recursive does NOT match baseline. Aborting.")
        return
    print("  [PASS] Block-recursive matches baseline")

    # ─── Step 2: Trace + convert both ───
    print("\n" + "="*60)
    print("  Step 2: CoreML conversion")
    print("="*60)

    mod_base = ChunkGDRModule(chunk_gdr_baseline)
    mod_br = ChunkGDRModule(chunk_gdr_blockrecur)

    path_base = trace_and_convert(mod_base, inputs, "baseline_rowbyrow", output_dir)
    path_br   = trace_and_convert(mod_br, inputs, "blockrecur_bc16", output_dir)

    # ─── Step 3: MIL op count ───
    print("\n" + "="*60)
    print("  Step 3: MIL op count comparison")
    print("="*60)

    total_base, ops_base = count_mil_ops(path_base)
    total_br, ops_br = count_mil_ops(path_br)
    print(f"  Baseline (row-by-row):  {total_base:,} total MIL ops")
    print(f"  Block-recursive BC=16:  {total_br:,} total MIL ops")
    if total_base > 0 and total_br > 0:
        pct = (1 - total_br / total_base) * 100
        print(f"  Reduction: {pct:.1f}%")

    # Show top op types for both
    for label, ops in [("Baseline", ops_base), ("BlockRecur", ops_br)]:
        if ops:
            top10 = sorted(ops.items(), key=lambda x: -x[1])[:10]
            print(f"\n  Top ops ({label}):")
            for op_type, cnt in top10:
                print(f"    {op_type:30s} {cnt:5d}")

    # ─── Step 4: ANE loadability ───
    print("\n" + "="*60)
    print("  Step 4: ANE loadability")
    print("="*60)

    ane_base = test_ane_loadability(path_base, "baseline")
    ane_br   = test_ane_loadability(path_br, "blockrecur_bc16")

    # ─── Step 5: ANE vs CPU numerical agreement ───
    print("\n" + "="*60)
    print("  Step 5: ANE vs CPU numerical agreement")
    print("="*60)

    if ane_base is not None:
        compare_ane_vs_cpu(path_base, "baseline")
    else:
        print("  Baseline: Skipped (ANE load failed)")

    if ane_br is not None:
        compare_ane_vs_cpu(path_br, "blockrecur_bc16")
    else:
        print("  BlockRecur: Skipped (ANE load failed)")

    # ─── Summary ───
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    print(f"  PyTorch parity:        cos={cos:.8f} {'PASS' if cos >= 0.999 else 'FAIL'}")
    print(f"  MIL ops baseline:      {total_base:,}")
    print(f"  MIL ops blockrecur:    {total_br:,}")
    if total_base > 0 and total_br > 0:
        print(f"  MIL op reduction:      {(1 - total_br/total_base)*100:.1f}%")
    print(f"  ANE load baseline:     {'YES' if ane_base else 'NO'}")
    print(f"  ANE load blockrecur:   {'YES' if ane_br else 'NO'}")


if __name__ == "__main__":
    main()
