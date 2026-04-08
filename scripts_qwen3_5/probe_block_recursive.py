#!/usr/bin/env python3
"""Probe: block-recursive vs row-by-row Loop 1 in _chunk_gated_delta_rule.

Tests:
1. PyTorch CPU correctness (both must agree)
2. CoreML export + MIL op count
3. ANE loadability
4. ANE vs CPU numerical agreement

Block-recursive approach: split chunk_size=32 into 4 sub-blocks of BC=8.
Solve each diagonal block independently, then merge via matmul — matching
the FLA library's block merge strategy (Steps 3+4 in chunk_fwd.py).
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
CHUNK_SIZE = 32
BC         = 16      # sub-block size (32/2 = 16)
N_BLOCKS   = CHUNK_SIZE // BC   # 2
SEQ_LEN    = 64      # 2 chunks, enough for A/B comparison
SEED       = 42
OUT_DIR    = "/tmp/probe_block_recursive"

# ──────────────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────────────
def cosine_sim(a, b):
    a_f = a.flatten().double()
    b_f = b.flatten().double()
    if a_f.norm() < 1e-10 and b_f.norm() < 1e-10:
        return 1.0
    return F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()


def report(label, ref, test, indent=2):
    diff = (ref.double() - test.double()).abs()
    cos  = cosine_sim(ref, test)
    max_abs  = diff.max().item()
    mean_abs = diff.mean().item()
    ref_scale = ref.double().abs().max().item()
    rel_max  = max_abs / max(ref_scale, 1e-10)
    prefix = " " * indent
    tag = "OK" if cos > 0.999 else ("WARN" if cos > 0.99 else "FAIL")
    print(f"{prefix}[{tag}] {label:50s}  cos={cos:.10f}  max_abs={max_abs:.6e}  "
          f"mean_abs={mean_abs:.6e}  rel_max={rel_max:.4e}")
    return {"cos": cos, "max_abs": max_abs, "mean_abs": mean_abs, "rel_max": rel_max}


# ──────────────────────────────────────────────────────
#  CURRENT Loop 1: row-by-row forward substitution
# ──────────────────────────────────────────────────────
def loop1_current(attn, chunk_size):
    """Original row-by-row Neumann series solve of (I+A)^{-1}."""
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
    return attn


# ──────────────────────────────────────────────────────
#  NEW Loop 1: block-recursive forward substitution
# ──────────────────────────────────────────────────────
def _solve_block_diag(A_block, bc):
    """Forward-substitution on a single [bc, bc] strictly-lower-triangular block.
    
    Computes (I + A_block)^{-1} where A_block is strictly lower triangular.
    Uses the same row-by-row Neumann iteration but only bc rows (8 vs 32).
    """
    # A_block: (..., bc, bc) — strictly lower triangular
    rows = [A_block[..., 0:1, :]]
    for i in range(1, bc):
        row = A_block[..., i, :i].clone()
        sub = torch.cat([prev_row[..., :i] for prev_row in rows[:i]], dim=-2)
        updated_row = row + (row.unsqueeze(-1) * sub).sum(-2)
        tail = A_block[..., i:i+1, i:]
        full_row = torch.cat([updated_row.unsqueeze(-2), tail], dim=-1)
        rows.append(full_row)
    result = torch.cat(rows, dim=-2)
    result = result + torch.eye(bc, dtype=result.dtype, device=result.device)
    return result


def loop1_block_recursive(attn, chunk_size, bc=BC):
    """Block-recursive solve: split into N_BLOCKS sub-blocks of size bc.
    
    Strategy (matching FLA chunk_fwd.py):
    Step 1: Extract all diagonal and off-diagonal [bc, bc] sub-blocks from attn.
    Step 2: Forward-substitute each diagonal block (small loop of bc iterations).
    Step 3: Block-merge to get the full inverse via matmul chains.
    Step 4: Assemble full result.
    
    For chunk_size=32, bc=8: 4 blocks, 7 iterations each (vs 31 iterations total).
    """
    n_blocks = chunk_size // bc
    
    # Step 1: Extract all [bc, bc] sub-blocks
    # attn is (..., chunk_size, chunk_size) — strictly lower triangular (negated)
    # We need the diagonal blocks and lower-off-diagonal blocks.
    blocks = {}
    for r in range(n_blocks):
        for c in range(r + 1):  # only lower triangle blocks
            blocks[(r, c)] = attn[..., r*bc:(r+1)*bc, c*bc:(c+1)*bc]
    
    # Step 2: Solve each diagonal block independently
    # (I + A_diag)^{-1} for each diagonal sub-block
    inv_diag = {}
    for b in range(n_blocks):
        inv_diag[b] = _solve_block_diag(blocks[(b, b)], bc)
    
    # Step 3: Block-merge to get full (I+A)^{-1}
    # For a 4x4 block lower-triangular matrix, the inverse has the form:
    #   Ai[r][c] = -Ai[r][r] @ (sum over m from c to r-1 of A[r][m] @ Ai[m][c]) for r > c
    #   Ai[r][r] = inv_diag[r]
    inv_full = {}
    for b in range(n_blocks):
        inv_full[(b, b)] = inv_diag[b]
    
    # Fill off-diagonal blocks column by column, top to bottom
    # (I - A)X = I  =>  X_rc = (I - A_rr)^{-1} @ sum_{m=c}^{r-1} A_rm @ X_mc
    # No negation: the minus signs are already embedded in A's off-diagonal blocks.
    for c in range(n_blocks):
        for r in range(c + 1, n_blocks):
            acc = torch.zeros_like(blocks[(r, c)])
            for m in range(c, r):
                acc = acc + blocks[(r, m)] @ inv_full[(m, c)]
            inv_full[(r, c)] = inv_diag[r] @ acc
    
    # Step 4: Assemble the full [chunk_size, chunk_size] result
    rows = []
    for r in range(n_blocks):
        row_blocks = []
        for c in range(n_blocks):
            if c <= r:
                row_blocks.append(inv_full[(r, c)])
            else:
                # Upper triangle: zeros (but attn had zeros there too, from strict_lower mask)
                row_blocks.append(torch.zeros_like(blocks[(r, r)]))
        rows.append(torch.cat(row_blocks, dim=-1))
    result = torch.cat(rows, dim=-2)
    return result


# ──────────────────────────────────────────────────────
#  Full _chunk_gated_delta_rule with pluggable Loop 1
# ──────────────────────────────────────────────────────
def chunk_gdr_with_loop1(query, key, value, g, beta, loop1_fn, chunk_size=CHUNK_SIZE,
                         initial_state=None, math_dtype=torch.float32):
    """Exact copy of _chunk_gated_delta_rule but with loop1_fn pluggable."""
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(math_dtype) for x in (query, key, value, beta, g)
    ]
    query = _l2norm(query, dim=-1)
    key = _l2norm(key, dim=-1)
    batch_size, num_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = seq_len + pad_size
    scale = 1 / (k_dim ** 0.5)
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    n_chunks = total_sequence_length // chunk_size
    query, key, value, k_beta, v_beta = [
        x.reshape(batch_size, num_heads, n_chunks, chunk_size, k_dim if idx != 2 and idx != 4 else v_dim)
        for idx, x in enumerate((query, key, value, k_beta, v_beta))
    ]
    g = g.reshape(batch_size, num_heads, n_chunks, chunk_size)
    tril_ones = torch.tril(torch.ones(chunk_size, chunk_size, device=query.device, dtype=math_dtype))
    strict_lower = torch.tril(torch.ones(chunk_size, chunk_size, device=query.device, dtype=math_dtype), diagonal=-1)
    g = (tril_ones @ g.unsqueeze(-1)).squeeze(-1)
    decay_raw = (g.unsqueeze(-1) - g.unsqueeze(-2)) * tril_ones
    decay_mask = decay_raw.exp() * tril_ones
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask) * strict_lower

    # ==== LOOP 1: pluggable ====
    attn = loop1_fn(attn, chunk_size)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_dim, v_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    strict_lower_diag1 = torch.tril(torch.ones(chunk_size, chunk_size, device=query.device, dtype=math_dtype))
    core_attn_chunks = []

    # ==== LOOP 2: unchanged ====
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]) * strict_lower_diag1
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_chunks.append((attn_inter + attn @ v_new).unsqueeze(2))
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = torch.cat(core_attn_chunks, dim=2)
    core_attn_out = core_attn_out.reshape(batch_size, num_heads, total_sequence_length, v_dim)
    core_attn_out = core_attn_out[:, :, :seq_len]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


# ──────────────────────────────────────────────────────
#  CoreML export wrappers
# ──────────────────────────────────────────────────────
class ChunkGDR_Current(nn.Module):
    """Wraps full pipeline with CURRENT loop1."""
    def forward(self, query, key, value, g, beta, initial_state):
        out, state = chunk_gdr_with_loop1(
            query, key, value, g, beta,
            loop1_fn=loop1_current,
            chunk_size=CHUNK_SIZE,
            initial_state=initial_state,
            math_dtype=torch.float32,
        )
        return out, state


class ChunkGDR_BlockRecursive(nn.Module):
    """Wraps full pipeline with BLOCK-RECURSIVE loop1."""
    def forward(self, query, key, value, g, beta, initial_state):
        out, state = chunk_gdr_with_loop1(
            query, key, value, g, beta,
            loop1_fn=loop1_block_recursive,
            chunk_size=CHUNK_SIZE,
            initial_state=initial_state,
            math_dtype=torch.float32,
        )
        return out, state


# ──────────────────────────────────────────────────────
#  MIL op counting
# ──────────────────────────────────────────────────────
def count_mil_ops(mlpackage_path):
    """Count MIL operations in a CoreML model."""
    try:
        import coremltools as ct
        spec = ct.utils.load_spec(mlpackage_path)
        # Walk the MIL program
        prog = spec.mlProgram
        total = 0
        op_types = {}
        for fn in prog.functions.values():
            for block in fn.block_specializations.values():
                for op in block.operations:
                    total += 1
                    ot = op.type
                    op_types[ot] = op_types.get(ot, 0) + 1
        return total, op_types
    except Exception as e:
        print(f"    [WARN] MIL op count failed: {e}")
        return -1, {}


# ──────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────
def main():
    import coremltools as ct

    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Generate random inputs ──
    print("="*70)
    print(f" Block-Recursive Loop 1 Probe")
    print(f" chunk_size={CHUNK_SIZE}, bc={BC}, n_blocks={N_BLOCKS}")
    print(f" seq_len={SEQ_LEN}, heads={NUM_HEADS}, k_dim={K_DIM}, v_dim={V_DIM}")
    print("="*70)

    # Input in BHSD format (batch, seq, heads, dim) — matching model's convention
    q = torch.randn(BATCH, SEQ_LEN, NUM_HEADS, K_DIM, dtype=torch.float32)
    k = torch.randn(BATCH, SEQ_LEN, NUM_HEADS, K_DIM, dtype=torch.float32)
    v = torch.randn(BATCH, SEQ_LEN, NUM_HEADS, V_DIM, dtype=torch.float32)
    g = torch.randn(BATCH, SEQ_LEN, NUM_HEADS, dtype=torch.float32) * 0.1
    beta = torch.sigmoid(torch.randn(BATCH, SEQ_LEN, NUM_HEADS, dtype=torch.float32))
    s0 = torch.zeros(BATCH, NUM_HEADS, K_DIM, V_DIM, dtype=torch.float32)

    # ─────────────────────────────────────
    #  TEST 1: PyTorch CPU correctness
    # ─────────────────────────────────────
    print("\n[TEST 1] PyTorch CPU correctness (fp32)")
    print("-"*50)

    out_cur, state_cur = chunk_gdr_with_loop1(q, k, v, g, beta, loop1_current,
                                               chunk_size=CHUNK_SIZE, initial_state=s0)
    out_blk, state_blk = chunk_gdr_with_loop1(q, k, v, g, beta, loop1_block_recursive,
                                               chunk_size=CHUNK_SIZE, initial_state=s0)

    r1_out   = report("output (current vs block-recursive)", out_cur, out_blk)
    r1_state = report("state  (current vs block-recursive)", state_cur, state_blk)

    if r1_out["max_abs"] < 1e-5 and r1_state["max_abs"] < 1e-5:
        print("  ✓ Block-recursive matches current implementation (fp32 exact)")
    else:
        print("  ✗ MISMATCH — check block-recursive implementation")
        # Still continue with export to see if issue is numerical vs algorithmic

    # Also test Loop 1 only (isolated)
    print("\n  [Isolated Loop 1 test]")
    # Create a dummy strictly-lower-triangular attn tensor
    dummy_attn = torch.randn(BATCH, NUM_HEADS, 2, CHUNK_SIZE, CHUNK_SIZE) * 0.01
    strict_lower = torch.tril(torch.ones(CHUNK_SIZE, CHUNK_SIZE), diagonal=-1)
    dummy_attn = dummy_attn * strict_lower
    
    r_cur = loop1_current(dummy_attn.clone(), CHUNK_SIZE)
    r_blk = loop1_block_recursive(dummy_attn.clone(), CHUNK_SIZE)
    report("Loop 1 isolated", r_cur, r_blk)

    # ─────────────────────────────────────
    #  TEST 2: CoreML export + MIL ops
    # ─────────────────────────────────────
    print("\n[TEST 2] CoreML export + MIL op count")
    print("-"*50)

    # Prepare numpy inputs for CoreML
    q_np = q.numpy().astype(np.float16)
    k_np = k.numpy().astype(np.float16)
    v_np = v.numpy().astype(np.float16)
    g_np = g.numpy().astype(np.float16)
    b_np = beta.numpy().astype(np.float16)
    s_np = s0.numpy().astype(np.float16)

    # Trace input: use fp32 for tracing
    trace_inputs = (q, k, v, g, beta, s0)

    for name, module_cls in [("current", ChunkGDR_Current),
                              ("block_recursive", ChunkGDR_BlockRecursive)]:
        print(f"\n  --- {name} ---")
        path = os.path.join(OUT_DIR, f"{name}.mlpackage")

        # Trace
        print(f"  Tracing...")
        t0 = time.time()
        module = module_cls().eval()
        with torch.no_grad():
            traced = torch.jit.trace(module, trace_inputs, check_trace=False)
        t_trace = time.time() - t0
        print(f"  Traced in {t_trace:.1f}s")

        # Convert
        print(f"  Converting to CoreML...")
        t0 = time.time()
        try:
            mlmodel = ct.convert(
                traced,
                inputs=[
                    ct.TensorType(name="query",         shape=q.shape),
                    ct.TensorType(name="key",           shape=k.shape),
                    ct.TensorType(name="value",         shape=v.shape),
                    ct.TensorType(name="g",             shape=g.shape),
                    ct.TensorType(name="beta",          shape=beta.shape),
                    ct.TensorType(name="initial_state", shape=s0.shape),
                ],
                outputs=[
                    ct.TensorType(name="output"),
                    ct.TensorType(name="final_state"),
                ],
                compute_precision=ct.precision.FLOAT16,
                compute_units=ct.ComputeUnit.CPU_AND_NE,
                minimum_deployment_target=ct.target.iOS18,
                convert_to="mlprogram",
            )
            t_convert = time.time() - t0
            print(f"  Converted in {t_convert:.1f}s")
            mlmodel.save(path)
            print(f"  Saved: {path}")

            # MIL op count
            total_ops, op_breakdown = count_mil_ops(path)
            print(f"  MIL ops: {total_ops}")
            if op_breakdown:
                top_ops = sorted(op_breakdown.items(), key=lambda x: -x[1])[:10]
                for op_name, cnt in top_ops:
                    print(f"    {op_name}: {cnt}")

        except Exception as e:
            print(f"  ✗ CoreML conversion FAILED: {e}")
            traceback.print_exc()
            continue

    # ─────────────────────────────────────
    #  TEST 3: ANE loadability
    # ─────────────────────────────────────
    print("\n[TEST 3] ANE loadability")
    print("-"*50)

    results = {}
    for name in ["current", "block_recursive"]:
        path = os.path.join(OUT_DIR, f"{name}.mlpackage")
        if not os.path.exists(path):
            print(f"  {name}: SKIPPED (no model)")
            continue

        print(f"\n  --- {name} ---")
        # Try loading on ANE
        print(f"  Loading on ANE (CPU_AND_NE)...")
        t0 = time.time()
        try:
            ane_model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            t_load = time.time() - t0
            print(f"  ✓ ANE load OK ({t_load:.1f}s)")

            # Run prediction on ANE
            inputs_dict = {
                "query": q_np, "key": k_np, "value": v_np,
                "g": g_np, "beta": b_np, "initial_state": s_np,
            }
            ane_out = ane_model.predict(inputs_dict)
            print(f"  ✓ ANE prediction OK")
            results[f"{name}_ane"] = ane_out
            del ane_model
            gc.collect()

        except Exception as e:
            print(f"  ✗ ANE load FAILED: {e}")
            results[f"{name}_ane"] = None

        # Load on CPU for reference
        print(f"  Loading on CPU_ONLY...")
        try:
            cpu_model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
            cpu_out = cpu_model.predict({
                "query": q_np, "key": k_np, "value": v_np,
                "g": g_np, "beta": b_np, "initial_state": s_np,
            })
            print(f"  ✓ CPU prediction OK")
            results[f"{name}_cpu"] = cpu_out
            del cpu_model
            gc.collect()
        except Exception as e:
            print(f"  ✗ CPU prediction FAILED: {e}")
            results[f"{name}_cpu"] = None

    # ─────────────────────────────────────
    #  TEST 4: ANE vs CPU agreement
    # ─────────────────────────────────────
    print("\n[TEST 4] ANE vs CPU numerical agreement")
    print("-"*50)

    for name in ["current", "block_recursive"]:
        ane_key = f"{name}_ane"
        cpu_key = f"{name}_cpu"
        if results.get(ane_key) is None or results.get(cpu_key) is None:
            print(f"  {name}: SKIPPED (missing results)")
            continue

        print(f"\n  --- {name} ---")
        ane_out_t = torch.from_numpy(np.array(results[ane_key]["output"]))
        cpu_out_t = torch.from_numpy(np.array(results[cpu_key]["output"]))
        ane_st_t  = torch.from_numpy(np.array(results[ane_key]["final_state"]))
        cpu_st_t  = torch.from_numpy(np.array(results[cpu_key]["final_state"]))

        report(f"{name} output  ANE vs CPU", cpu_out_t, ane_out_t, indent=4)
        report(f"{name} state   ANE vs CPU", cpu_st_t, ane_st_t,  indent=4)

    # ─────────────────────────────────────
    #  SUMMARY
    # ─────────────────────────────────────
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    print(f"  PyTorch fp32 parity (out):   cos={r1_out['cos']:.10f}  max_abs={r1_out['max_abs']:.2e}")
    print(f"  PyTorch fp32 parity (state): cos={r1_state['cos']:.10f}  max_abs={r1_state['max_abs']:.2e}")
    
    for name in ["current", "block_recursive"]:
        path = os.path.join(OUT_DIR, f"{name}.mlpackage")
        if os.path.exists(path):
            total_ops, _ = count_mil_ops(path)
            ane_ok = results.get(f"{name}_ane") is not None
            print(f"  {name:20s}: MIL ops={total_ops:>6d}  ANE={'OK' if ane_ok else 'FAIL'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
