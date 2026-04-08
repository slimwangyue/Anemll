#!/usr/bin/env python3
"""Chunk-size sweep: find the sweet spot for _chunk_gated_delta_rule on ANE.

For a fixed sequence length (512), tests chunk_size = 16, 32, 64, 128, 256, 512.
This corresponds to n_chunks = 32, 16, 8, 4, 2, 1.

For each configuration, measures:
1. ANE vs CoreML CPU numerical error (output + state)
2. ANE loadability / compile success
3. Runtime latency (ANE predict, CoreML CPU predict)
4. Graph size / compile notes
"""
import sys, os, gc, time, shutil, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from anemll.models.qwen3_5_model import _l2norm

# ── Config ──
BATCH = 1
H = 32        # num heads
K = 128       # key dim
V = 128       # value dim
SEQ = 512     # fixed prefill length
SEED = 42
OUT_DIR = "/tmp/diag_chunk_sweep"
N_LATENCY_RUNS = 5  # repeats for timing

CHUNK_SIZES = [16, 32, 64, 128, 256, 512]


def cosine_sim(a, b):
    a_f = a.flatten().double()
    b_f = b.flatten().double()
    if a_f.norm() < 1e-10 and b_f.norm() < 1e-10:
        return 1.0
    return F.cosine_similarity(a_f.unsqueeze(0), b_f.unsqueeze(0)).item()


def metrics(ref, test):
    """Return dict of cos, max_abs, mean_abs."""
    diff = (ref.double() - test.double()).abs()
    return {
        "cos": cosine_sim(ref, test),
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
    }


class ChunkGDR(nn.Module):
    """Wraps _chunk_gated_delta_rule with a specific chunk_size."""

    def __init__(self, chunk_size):
        super().__init__()
        self.chunk_size = chunk_size
        cs = min(chunk_size, SEQ)
        # Register masks as buffers for tracing
        self.register_buffer("tril_ones", torch.tril(torch.ones(cs, cs)))
        self.register_buffer("strict_lower", torch.tril(torch.ones(cs, cs), diagonal=-1))
        self.register_buffer("strict_lower_diag1", torch.tril(torch.ones(cs, cs)))
        self.register_buffer("eye_cs", torch.eye(cs))

    def forward(self, query, key, value, g, beta, initial_state):
        # Inline _chunk_gated_delta_rule with configurable chunk_size
        chunk_size = self.chunk_size
        initial_dtype = query.dtype
        math_dtype = torch.float32

        query, key, value, beta, g = [
            x.transpose(1, 2).contiguous().to(math_dtype) for x in (query, key, value, beta, g)
        ]
        query = _l2norm(query, dim=-1)
        key = _l2norm(key, dim=-1)

        batch_size = BATCH
        num_heads = H
        seq_len = SEQ
        k_dim = K
        v_dim = V
        chunk_size = min(chunk_size, seq_len)
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
        if pad_size > 0:
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
        query = query.reshape(batch_size, num_heads, n_chunks, chunk_size, k_dim)
        key = key.reshape(batch_size, num_heads, n_chunks, chunk_size, k_dim)
        value = value.reshape(batch_size, num_heads, n_chunks, chunk_size, v_dim)
        k_beta = k_beta.reshape(batch_size, num_heads, n_chunks, chunk_size, k_dim)
        v_beta = v_beta.reshape(batch_size, num_heads, n_chunks, chunk_size, v_dim)
        g = g.reshape(batch_size, num_heads, n_chunks, chunk_size)

        tril_ones = self.tril_ones
        strict_lower = self.strict_lower

        g = (tril_ones @ g.unsqueeze(-1)).squeeze(-1)
        decay_raw = (g.unsqueeze(-1) - g.unsqueeze(-2)) * tril_ones
        decay_mask = decay_raw.exp() * tril_ones

        attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask) * strict_lower
        attn_rows = [attn[..., 0:1, :]]
        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = torch.cat([prev_row[..., :i] for prev_row in attn_rows[:i]], dim=-2)
            updated_row = row + (row.unsqueeze(-1) * sub).sum(-2)
            tail = attn[..., i : i + 1, i:]
            full_row = torch.cat([updated_row.unsqueeze(-2), tail], dim=-1)
            attn_rows.append(full_row)
        attn = torch.cat(attn_rows, dim=-2)
        attn = attn + self.eye_cs
        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

        last_recurrent_state = initial_state.to(value)
        strict_lower_diag1 = self.strict_lower_diag1
        core_attn_chunks = []

        for i in range(n_chunks):
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


def test_chunk_size(cs, q, k, v, g, beta, s0, ref_output, ref_state):
    """Test one chunk_size. Returns dict of results or error info."""
    n_chunks = SEQ // cs
    label = f"cs{cs}_nc{n_chunks}"
    result = {
        "chunk_size": cs,
        "n_chunks": n_chunks,
        "loads_ane": False,
        "runs_ane": False,
        "output_cos": None,
        "output_max_abs": None,
        "output_mean_abs": None,
        "state_cos": None,
        "state_max_abs": None,
        "state_mean_abs": None,
        "output_vs_ref_cos": None,
        "output_vs_ref_max_abs": None,
        "state_vs_ref_cos": None,
        "state_vs_ref_max_abs": None,
        "cpu_output_vs_ref_cos": None,
        "ane_latency_ms": None,
        "cpu_latency_ms": None,
        "export_time_s": None,
        "error": None,
        "note": "",
    }

    print(f"\n{'='*90}")
    print(f"CHUNK_SIZE={cs}, N_CHUNKS={n_chunks}")
    print(f"  Woodbury row-update loop: {cs-1} iterations")
    print(f"  Inter-chunk loop: {n_chunks} iterations")
    print(f"  Per-chunk matrices: ({cs}x{K}), ({cs}x{V}), ({cs}x{cs})")
    print(f"{'='*90}")

    # ── Build model ──
    model = ChunkGDR(cs)
    model.eval()

    # ── PyTorch fp32 reference with this chunk_size ──
    with torch.no_grad():
        pt_out, pt_state = model(
            q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), s0.clone()
        )
    pt_vs_ref = metrics(ref_output, pt_out)
    print(f"  PyTorch fp32 this-cs vs cs=16 ref:  cos={pt_vs_ref['cos']:.10f}  max_abs={pt_vs_ref['max_abs']:.6e}")

    # ── Export to CoreML ──
    inputs = {
        "query": q.to(torch.float16),
        "key": k.to(torch.float16),
        "value": v.to(torch.float16),
        "g": g.to(torch.float16),
        "beta": beta.to(torch.float16),
        "initial_state": s0.to(torch.float32),
    }
    ct_inputs = [ct.TensorType(name=n, shape=t.shape) for n, t in inputs.items()]
    ct_outputs = [ct.TensorType(name="output"), ct.TensorType(name="state")]

    try:
        print(f"  Exporting CoreML model...")
        t0 = time.time()
        with torch.no_grad():
            traced = torch.jit.trace(model, [t for t in inputs.values()])
        ml = ct.convert(
            traced,
            inputs=ct_inputs,
            outputs=ct_outputs,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
        )
        export_time = time.time() - t0
        result["export_time_s"] = round(export_time, 1)
        print(f"  Export OK in {export_time:.1f}s")

        path = os.path.join(OUT_DIR, f"{label}.mlpackage")
        ml.save(path)
        del ml, traced
        gc.collect()
    except Exception as e:
        result["error"] = f"export: {e}"
        print(f"  EXPORT FAILED: {e}")
        traceback.print_exc()
        return result

    # ── Load on ANE ──
    try:
        print(f"  Loading on ANE (CPU_AND_NE)...")
        t0 = time.time()
        ane_model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        load_time = time.time() - t0
        result["loads_ane"] = True
        print(f"  ANE load OK in {load_time:.1f}s")
    except Exception as e:
        result["error"] = f"ane_load: {e}"
        print(f"  ANE LOAD FAILED: {e}")
        # Still try CPU
        try:
            cpu_model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
            np_inp = {n: t.numpy().astype(np.float16) if t.dtype==torch.float16 else t.numpy() for n, t in inputs.items()}
            cpu_pred = cpu_model.predict(np_inp)
            # Find output key by shape
            cpu_keys = sorted(cpu_pred.keys())
            cpu_out_key = cpu_keys[0]
            for kk in cpu_keys:
                arr = np.array(cpu_pred[kk])
                if arr.ndim >= 3 and arr.shape[1] >= SEQ // 2:
                    cpu_out_key = kk
                    break
            cpu_out_t = torch.from_numpy(np.array(cpu_pred[cpu_out_key]))
            cpu_vs_ref = metrics(ref_output, cpu_out_t)
            result["cpu_output_vs_ref_cos"] = cpu_vs_ref["cos"]
            print(f"  CoreML CPU output vs ref:  cos={cpu_vs_ref['cos']:.10f}")
            del cpu_model
        except:
            pass
        return result

    # ── Load on CPU ──
    try:
        cpu_model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_ONLY)
    except Exception as e:
        result["error"] = f"cpu_load: {e}"
        del ane_model
        return result

    # ── Run predictions ──
    np_inp = {n: t.numpy().astype(np.float16) if t.dtype==torch.float16 else t.numpy() for n, t in inputs.items()}

    try:
        # Warm-up
        ane_pred = ane_model.predict(np_inp)
        cpu_pred = cpu_model.predict(np_inp)
        result["runs_ane"] = True

        # Discover output keys (CoreML may rename tuple outputs)
        pred_keys = sorted(ane_pred.keys())
        print(f"  CoreML output keys: {pred_keys}")
        # Find output and state keys by shape matching
        # output shape: (1, SEQ, H*V) or similar with seq dim
        # state shape: (1, H, K, V) 
        out_key, state_key = None, None
        for kk in pred_keys:
            arr = np.array(ane_pred[kk])
            if arr.ndim == 4 and arr.shape[-1] == V and arr.shape[-2] == K:
                state_key = kk
            elif arr.ndim >= 3 and arr.shape[1] >= SEQ // 2:
                out_key = kk
        # Fallback: if exactly 2 keys, assign by name or order
        if out_key is None or state_key is None:
            if len(pred_keys) == 2:
                for kk in pred_keys:
                    if "state" in kk.lower() or "recurrent" in kk.lower():
                        state_key = kk
                    else:
                        out_key = kk
                if out_key is None:
                    out_key = pred_keys[0]
                    state_key = pred_keys[1]
        if out_key is None or state_key is None:
            # Last resort: use sorted order — first=output, second=state
            out_key = pred_keys[0]
            state_key = pred_keys[1] if len(pred_keys) > 1 else pred_keys[0]
        print(f"  Using output_key={out_key}, state_key={state_key}")

        # ANE latency
        times_ane = []
        for _ in range(N_LATENCY_RUNS):
            t0 = time.time()
            ane_pred = ane_model.predict(np_inp)
            times_ane.append((time.time() - t0) * 1000)
        result["ane_latency_ms"] = round(np.median(times_ane), 2)

        # CPU latency
        times_cpu = []
        for _ in range(N_LATENCY_RUNS):
            t0 = time.time()
            cpu_pred = cpu_model.predict(np_inp)
            times_cpu.append((time.time() - t0) * 1000)
        result["cpu_latency_ms"] = round(np.median(times_cpu), 2)

        # Extract outputs
        ane_out_t = torch.from_numpy(np.array(ane_pred[out_key]))
        ane_state_t = torch.from_numpy(np.array(ane_pred[state_key]))
        cpu_out_t = torch.from_numpy(np.array(cpu_pred[out_key]))
        cpu_state_t = torch.from_numpy(np.array(cpu_pred[state_key]))

        # ANE vs CoreML CPU
        out_m = metrics(cpu_out_t, ane_out_t)
        state_m = metrics(cpu_state_t, ane_state_t)
        result["output_cos"] = out_m["cos"]
        result["output_max_abs"] = out_m["max_abs"]
        result["output_mean_abs"] = out_m["mean_abs"]
        result["state_cos"] = state_m["cos"]
        result["state_max_abs"] = state_m["max_abs"]
        result["state_mean_abs"] = state_m["mean_abs"]

        # ANE vs fp32 ref
        ane_vs_ref = metrics(ref_output, ane_out_t)
        result["output_vs_ref_cos"] = ane_vs_ref["cos"]
        result["output_vs_ref_max_abs"] = ane_vs_ref["max_abs"]

        # CoreML CPU vs fp32 ref
        cpu_vs_ref = metrics(ref_output, cpu_out_t)
        result["cpu_output_vs_ref_cos"] = cpu_vs_ref["cos"]

        # State vs fp32 ref
        state_vs_ref = metrics(ref_state, ane_state_t)
        result["state_vs_ref_cos"] = state_vs_ref["cos"]
        result["state_vs_ref_max_abs"] = state_vs_ref["max_abs"]

        print(f"  --- ANE vs CoreML CPU ---")
        print(f"    output:  cos={out_m['cos']:.10f}  max_abs={out_m['max_abs']:.6e}  mean_abs={out_m['mean_abs']:.6e}")
        print(f"    state:   cos={state_m['cos']:.10f}  max_abs={state_m['max_abs']:.6e}  mean_abs={state_m['mean_abs']:.6e}")
        print(f"  --- ANE vs fp32 ref ---")
        print(f"    output:  cos={ane_vs_ref['cos']:.10f}  max_abs={ane_vs_ref['max_abs']:.6e}")
        print(f"    state:   cos={state_vs_ref['cos']:.10f}  max_abs={state_vs_ref['max_abs']:.6e}")
        print(f"  --- CoreML CPU vs fp32 ref ---")
        print(f"    output:  cos={cpu_vs_ref['cos']:.10f}")
        print(f"  --- Latency ---")
        print(f"    ANE:     {result['ane_latency_ms']:.2f} ms (median of {N_LATENCY_RUNS})")
        print(f"    CPU:     {result['cpu_latency_ms']:.2f} ms (median of {N_LATENCY_RUNS})")

    except Exception as e:
        result["error"] = f"predict: {e}"
        print(f"  PREDICT FAILED: {e}")
        traceback.print_exc()

    del ane_model, cpu_model
    gc.collect()
    return result


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 100)
    print(f"CHUNK-SIZE SWEEP for _chunk_gated_delta_rule on ANE")
    print(f"SEQ={SEQ}, BATCH={BATCH}, H={H}, K={K}, V={V}")
    print(f"Chunk sizes to test: {CHUNK_SIZES}")
    print("=" * 100)

    # ── Generate inputs ──
    torch.manual_seed(SEED)
    q = torch.randn(BATCH, SEQ, H, K) * 0.1
    k = torch.randn(BATCH, SEQ, H, K) * 0.1
    v = torch.randn(BATCH, SEQ, H, V) * 0.1
    g = -torch.rand(BATCH, SEQ, H).abs() * 2.0 - 0.1
    beta = torch.sigmoid(torch.randn(BATCH, SEQ, H))
    s0 = torch.zeros(BATCH, H, K, V)

    # ── fp32 reference (chunk_size=16, canonical) ──
    print("\nComputing fp32 reference (chunk_size=16)...")
    ref_model = ChunkGDR(16)
    ref_model.eval()
    with torch.no_grad():
        ref_output, ref_state = ref_model(
            q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), s0.clone()
        )
    print(f"  ref output shape: {ref_output.shape}")
    print(f"  ref state shape:  {ref_state.shape}")

    # Verify all chunk sizes produce identical fp32 output
    print("\nVerifying fp32 parity across chunk sizes...")
    for cs in CHUNK_SIZES:
        m = ChunkGDR(cs)
        m.eval()
        with torch.no_grad():
            o, st = m(q.clone(), k.clone(), v.clone(), g.clone(), beta.clone(), s0.clone())
        d = metrics(ref_output, o)
        print(f"  cs={cs:>4d}  n_chunks={SEQ//cs:>3d}  output cos={d['cos']:.10f}  max_abs={d['max_abs']:.6e}")

    # ── Sweep each chunk size ──
    results = []
    for cs in CHUNK_SIZES:
        r = test_chunk_size(cs, q, k, v, g, beta, s0, ref_output, ref_state)
        results.append(r)
        gc.collect()

    # ── Summary table ──
    print("\n\n" + "=" * 130)
    print("SUMMARY TABLE")
    print("=" * 130)
    hdr = f"{'CS':>4s} {'NC':>3s} {'ANE?':>5s} {'Run?':>5s} {'Out cos(ANEvCPU)':>18s} {'Out max_abs':>12s} {'State cos':>12s} {'State max_abs':>12s} {'ANE ms':>8s} {'CPU ms':>8s} {'Export s':>9s} {'Note':>20s}"
    print(hdr)
    print("-" * 130)
    for r in results:
        cs = r["chunk_size"]
        nc = r["n_chunks"]
        loads = "YES" if r["loads_ane"] else "NO"
        runs = "YES" if r["runs_ane"] else "NO"
        oc = f"{r['output_cos']:.8f}" if r["output_cos"] is not None else "N/A"
        om = f"{r['output_max_abs']:.4e}" if r["output_max_abs"] is not None else "N/A"
        sc = f"{r['state_cos']:.8f}" if r["state_cos"] is not None else "N/A"
        sm = f"{r['state_max_abs']:.4e}" if r["state_max_abs"] is not None else "N/A"
        al = f"{r['ane_latency_ms']:.1f}" if r["ane_latency_ms"] is not None else "N/A"
        cl = f"{r['cpu_latency_ms']:.1f}" if r["cpu_latency_ms"] is not None else "N/A"
        et = f"{r['export_time_s']:.1f}" if r["export_time_s"] is not None else "N/A"
        note = r["error"] or r["note"] or ""
        if len(note) > 20:
            note = note[:17] + "..."
        print(f"{cs:>4d} {nc:>3d} {loads:>5s} {runs:>5s} {oc:>18s} {om:>12s} {sc:>12s} {sm:>12s} {al:>8s} {cl:>8s} {et:>9s} {note:>20s}")

    # ── ANE vs ref summary ──
    print("\n\nANE OUTPUT vs FP32 REFERENCE:")
    print(f"{'CS':>4s} {'NC':>3s} {'ANE vs ref cos':>18s} {'ANE vs ref max_abs':>20s} {'CoreML CPU vs ref cos':>22s}")
    print("-" * 80)
    for r in results:
        cs = r["chunk_size"]
        nc = r["n_chunks"]
        arc = f"{r['output_vs_ref_cos']:.8f}" if r["output_vs_ref_cos"] is not None else "N/A"
        arm = f"{r['output_vs_ref_max_abs']:.4e}" if r["output_vs_ref_max_abs"] is not None else "N/A"
        crc = f"{r['cpu_output_vs_ref_cos']:.8f}" if r["cpu_output_vs_ref_cos"] is not None else "N/A"
        print(f"{cs:>4d} {nc:>3d} {arc:>18s} {arm:>20s} {crc:>22s}")

    # ── State vs ref summary ──
    print("\n\nANE STATE vs FP32 REFERENCE:")
    print(f"{'CS':>4s} {'NC':>3s} {'ANE state vs ref cos':>22s} {'ANE state vs ref max_abs':>25s}")
    print("-" * 60)
    for r in results:
        cs = r["chunk_size"]
        nc = r["n_chunks"]
        src = f"{r['state_vs_ref_cos']:.8f}" if r["state_vs_ref_cos"] is not None else "N/A"
        srm = f"{r['state_vs_ref_max_abs']:.4e}" if r["state_vs_ref_max_abs"] is not None else "N/A"
        print(f"{cs:>4d} {nc:>3d} {src:>22s} {srm:>25s}")

    # Cleanup
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    print(f"\nCleaned up {OUT_DIR}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
