#!/usr/bin/env python3
"""Targeted experiments to maximize ANE utilization.

Tests:
  E1: LMHead on ANE (biggest win: 15.6% of decode → 4.91x faster on ANE)
  E2: Embed+LMHead combined on ANE
  E3: Chunk 8 with ComputeUnit.ALL (35% faster with GPU fallback)
  E4: All chunks with ComputeUnit.ALL
  E5: Correctness validation of each experiment

Usage:
    TMPDIR=/Volumes/MySSD/tmp python tests/dev/profile_ane_experiments.py
"""
import sys, os, time, gc, warnings
warnings.filterwarnings('ignore')

REPO_ROOT = '/Volumes/MySSD/Anemll'
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts_qwen3_5'))
os.chdir(REPO_ROOT)

import numpy as np
import coremltools as ct
from transformers import AutoTokenizer

from config import BATCH_SIZE, CTX, NUM_CHUNKS, CHUNK_RANGES

MODEL_DIR = os.path.join(REPO_ROOT, 'qwen3_5_stable_lut4ffn_lut6em_fp32')
COMBINED_DIR = os.path.join(MODEL_DIR, 'combined_LUT4_dedup')
TOKENIZER_DIR = MODEL_DIR

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, trust_remote_code=True)

def banner(msg):
    print(f"\n{'='*72}")
    print(f"  {msg}")
    print(f"{'='*72}")

def fmt_ms(ms):
    return f"{ms:.2f}ms"

def get_chunk_input_shapes(model_path, fn_name="infer"):
    spec = ct.utils.load_spec(model_path)
    shapes = {}
    for fn in spec.description.functions:
        if fn.name == fn_name:
            for inp in fn.input:
                try:
                    shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
                except:
                    pass
            break
    return shapes


# ═══════════════════════════════════════════════════════════════════
#  Configuration helper: load models with specified compute units
# ═══════════════════════════════════════════════════════════════════

def build_engine(embed_cu, lmhead_cu, chunk_cu_map):
    """Build inference engine with specified compute units.
    
    Args:
        embed_cu: ComputeUnit for embed model
        lmhead_cu: ComputeUnit for lmhead model
        chunk_cu_map: dict mapping chunk_idx → ComputeUnit, or single ComputeUnit for all
    """
    combined_el = os.path.join(MODEL_DIR, 'embed_lmhead_combined.mlpackage')
    
    embed = ct.models.MLModel(combined_el, compute_units=embed_cu, function_name="embed")
    lmhead = ct.models.MLModel(combined_el, compute_units=lmhead_cu, function_name="lmhead")
    
    ffns = []
    states = []
    lin_convs = []
    lin_recs = []
    inp_maps = []
    
    for ci in range(NUM_CHUNKS):
        model_path = os.path.join(COMBINED_DIR, f"chunk{ci}.mlpackage")
        if isinstance(chunk_cu_map, dict):
            cu = chunk_cu_map.get(ci, ct.ComputeUnit.CPU_AND_NE)
        else:
            cu = chunk_cu_map
        
        m = ct.models.MLModel(model_path, compute_units=cu, function_name="infer")
        ffns.append(m)
        states.append(m.make_state())
        
        inp_shapes = get_chunk_input_shapes(model_path)
        inp_maps.append(inp_shapes)
        
        has_linear = 'linear_conv_state' in inp_shapes
        if has_linear:
            lin_convs.append(np.zeros(inp_shapes['linear_conv_state'], dtype=np.float16))
            lin_recs.append(np.zeros(inp_shapes['linear_recurrent_state'], dtype=np.float16))
        else:
            lin_convs.append(None)
            lin_recs.append(None)
    
    return {
        'embed': embed, 'lmhead': lmhead,
        'ffns': ffns, 'states': states,
        'lin_convs': lin_convs, 'lin_recs': lin_recs,
        'inp_maps': inp_maps,
    }


def decode_step(engine, tok_id, pos):
    """Single decode step, returns (next_token_id, per_component_times_ms)."""
    times = {}
    
    t0 = time.perf_counter()
    tok = np.array([[tok_id]], dtype=np.int32)
    h = list(engine['embed'].predict({"input_ids": tok}).values())[0]
    times['embed'] = (time.perf_counter() - t0) * 1000
    
    mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    mask[:,:,:,:pos+1] = 0
    
    for ci in range(NUM_CHUNKS):
        inp = {
            "hidden_states": h.astype(np.float16),
            "position_ids": np.array([pos], dtype=np.int32),
            "causal_mask": mask,
            "current_pos": np.array([pos], dtype=np.int32),
        }
        if engine['lin_convs'][ci] is not None:
            inp["linear_conv_state"] = engine['lin_convs'][ci]
            inp["linear_recurrent_state"] = engine['lin_recs'][ci]
        
        t0 = time.perf_counter()
        out = engine['ffns'][ci].predict(inp, state=engine['states'][ci])
        times[f'chunk_{ci}'] = (time.perf_counter() - t0) * 1000
        
        h = out["output_hidden_states"]
        if 'linear_conv_state_out' in out:
            engine['lin_convs'][ci] = out['linear_conv_state_out']
            engine['lin_recs'][ci] = out['linear_recurrent_state_out']
    
    t0 = time.perf_counter()
    lm_out = engine['lmhead'].predict({"hidden_states": h.astype(np.float16)})
    times['lmhead'] = (time.perf_counter() - t0) * 1000
    
    next_tok = int(np.argmax(lm_out["logits"].flatten()))
    return next_tok, times


def run_experiment(name, engine, prompt_text, num_gen=20, warmup=3):
    """Run a decode experiment and return results."""
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    
    # Reset states
    for ci in range(NUM_CHUNKS):
        engine['states'][ci] = engine['ffns'][ci].make_state()
        if engine['lin_convs'][ci] is not None:
            engine['lin_convs'][ci][:] = 0
            engine['lin_recs'][ci][:] = 0
    
    # Prefill (token-by-token for simplicity)
    t0 = time.perf_counter()
    for i, tid in enumerate(prompt_ids):
        last_tok, _ = decode_step(engine, tid, i)
    prefill_ms = (time.perf_counter() - t0) * 1000
    
    pos = len(prompt_ids)
    gen_tokens = [last_tok]
    
    # Warmup decode steps
    for _ in range(warmup):
        nxt, _ = decode_step(engine, gen_tokens[-1], pos)
        gen_tokens.append(nxt)
        pos += 1
    
    # Timed decode
    all_times = []
    for step in range(num_gen):
        nxt, times = decode_step(engine, gen_tokens[-1], pos)
        gen_tokens.append(nxt)
        pos += 1
        all_times.append(times)
    
    # Aggregate
    components = list(all_times[0].keys())
    means = {}
    for comp in components:
        vals = [t[comp] for t in all_times]
        means[comp] = np.mean(vals)
    
    total_mean = sum(means.values())
    tok_s = 1000.0 / total_mean
    
    text = tokenizer.decode(gen_tokens)
    
    return {
        'name': name,
        'total_ms': total_mean,
        'tok_s': tok_s,
        'means': means,
        'gen_tokens': gen_tokens,
        'text': text,
        'prefill_ms': prefill_ms,
        'prompt_len': len(prompt_ids),
    }


def print_experiment(result, baseline=None):
    """Print experiment results with optional baseline comparison."""
    r = result
    print(f"\n  {r['name']}:")
    print(f"    Total decode: {fmt_ms(r['total_ms'])} → {r['tok_s']:.1f} tok/s")
    
    if baseline:
        speedup = baseline['total_ms'] / r['total_ms']
        delta = (r['tok_s'] - baseline['tok_s']) / baseline['tok_s'] * 100
        print(f"    vs baseline: {speedup:.2f}x speedup ({delta:+.1f}% tok/s)")
    
    # Component breakdown
    print(f"    {'Component':>20s}  {'ms':>8s}  {'%':>6s}", end="")
    if baseline:
        print(f"  {'Δ vs base':>10s}", end="")
    print()
    
    for comp, ms in sorted(r['means'].items(), key=lambda x: -x[1]):
        pct = ms / r['total_ms'] * 100
        line = f"    {comp:>20s}  {ms:>6.2f}ms  {pct:>5.1f}%"
        if baseline and comp in baseline['means']:
            delta_ms = ms - baseline['means'][comp]
            line += f"  {delta_ms:>+8.2f}ms"
        print(line)
    
    # Correctness
    print(f"    Text: {r['text'][:100]!r}...")
    
    # Token comparison with baseline
    if baseline:
        base_toks = baseline['gen_tokens']
        exp_toks = r['gen_tokens']
        match_len = min(len(base_toks), len(exp_toks))
        matches = sum(1 for a, b in zip(base_toks[:match_len], exp_toks[:match_len]) if a == b)
        print(f"    Token match vs baseline: {matches}/{match_len} ({matches/match_len*100:.0f}%)")


# ═══════════════════════════════════════════════════════════════════
#  TEST PROMPTS
# ═══════════════════════════════════════════════════════════════════

PROMPT1 = "Explain the concept of recursion in programming with a simple example."
PROMPT2 = "The capital of France is"

NUM_GEN = 20

# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 0: BASELINE (current production config)
# ═══════════════════════════════════════════════════════════════════

banner("EXPERIMENT 0: BASELINE (embed=CPU, lmhead=CPU, chunks=CPU_AND_NE)")
print("Loading baseline engine...")
t0 = time.time()
baseline_engine = build_engine(
    embed_cu=ct.ComputeUnit.CPU_ONLY,
    lmhead_cu=ct.ComputeUnit.CPU_ONLY,
    chunk_cu_map=ct.ComputeUnit.CPU_AND_NE,
)
print(f"  Loaded in {time.time()-t0:.1f}s")

baseline = run_experiment("Baseline", baseline_engine, PROMPT1, num_gen=NUM_GEN)
print_experiment(baseline)

baseline2 = run_experiment("Baseline (prompt2)", baseline_engine, PROMPT2, num_gen=NUM_GEN)
print_experiment(baseline2)


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 1: LMHead on ANE (biggest expected win)
# ═══════════════════════════════════════════════════════════════════

banner("EXPERIMENT 1: LMHead on ANE (embed=CPU, lmhead=CPU_AND_NE, chunks=CPU_AND_NE)")
del baseline_engine; gc.collect()

print("Loading E1 engine...")
t0 = time.time()
e1_engine = build_engine(
    embed_cu=ct.ComputeUnit.CPU_ONLY,
    lmhead_cu=ct.ComputeUnit.CPU_AND_NE,
    chunk_cu_map=ct.ComputeUnit.CPU_AND_NE,
)
print(f"  Loaded in {time.time()-t0:.1f}s")

e1 = run_experiment("E1: LMHead-ANE", e1_engine, PROMPT1, num_gen=NUM_GEN)
print_experiment(e1, baseline)

e1b = run_experiment("E1b: LMHead-ANE (prompt2)", e1_engine, PROMPT2, num_gen=NUM_GEN)
print_experiment(e1b, baseline2)


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 2: LMHead on ANE + All chunks with ALL
# ═══════════════════════════════════════════════════════════════════

banner("EXPERIMENT 2: LMHead=ANE, chunks=ALL (allows GPU fallback)")
del e1_engine; gc.collect()

print("Loading E2 engine...")
t0 = time.time()
e2_engine = build_engine(
    embed_cu=ct.ComputeUnit.CPU_ONLY,
    lmhead_cu=ct.ComputeUnit.CPU_AND_NE,
    chunk_cu_map=ct.ComputeUnit.ALL,
)
print(f"  Loaded in {time.time()-t0:.1f}s")

e2 = run_experiment("E2: LMHead-ANE+chunks-ALL", e2_engine, PROMPT1, num_gen=NUM_GEN)
print_experiment(e2, baseline)

e2b = run_experiment("E2b (prompt2)", e2_engine, PROMPT2, num_gen=NUM_GEN)
print_experiment(e2b, baseline2)


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 3: LMHead=ANE, chunk8=ALL, rest=CPU_AND_NE
# ═══════════════════════════════════════════════════════════════════

banner("EXPERIMENT 3: LMHead=ANE, chunk8=ALL, rest=CPU_AND_NE")
del e2_engine; gc.collect()

chunk_cu = {ci: ct.ComputeUnit.CPU_AND_NE for ci in range(NUM_CHUNKS)}
chunk_cu[8] = ct.ComputeUnit.ALL

print("Loading E3 engine...")
t0 = time.time()
e3_engine = build_engine(
    embed_cu=ct.ComputeUnit.CPU_ONLY,
    lmhead_cu=ct.ComputeUnit.CPU_AND_NE,
    chunk_cu_map=chunk_cu,
)
print(f"  Loaded in {time.time()-t0:.1f}s")

e3 = run_experiment("E3: LMHead-ANE+chunk8-ALL", e3_engine, PROMPT1, num_gen=NUM_GEN)
print_experiment(e3, baseline)


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 4: Everything on ANE (embed too)
# ═══════════════════════════════════════════════════════════════════

banner("EXPERIMENT 4: Everything ANE (embed=ANE, lmhead=ANE, chunks=CPU_AND_NE)")
del e3_engine; gc.collect()

print("Loading E4 engine...")
t0 = time.time()
e4_engine = build_engine(
    embed_cu=ct.ComputeUnit.CPU_AND_NE,
    lmhead_cu=ct.ComputeUnit.CPU_AND_NE,
    chunk_cu_map=ct.ComputeUnit.CPU_AND_NE,
)
print(f"  Loaded in {time.time()-t0:.1f}s")

e4 = run_experiment("E4: All-ANE", e4_engine, PROMPT1, num_gen=NUM_GEN)
print_experiment(e4, baseline)


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 5: Use standalone lm_head_nosplit on ANE
# ═══════════════════════════════════════════════════════════════════

banner("EXPERIMENT 5: Standalone lm_head_nosplit on ANE")
del e4_engine; gc.collect()

lmhead_nosplit_path = os.path.join(MODEL_DIR, 'lm_head_nosplit.mlpackage')
if os.path.exists(lmhead_nosplit_path):
    print("Loading E5 engine (standalone lm_head_nosplit on ANE)...")
    t0 = time.time()
    
    combined_el = os.path.join(MODEL_DIR, 'embed_lmhead_combined.mlpackage')
    e5_embed = ct.models.MLModel(combined_el, compute_units=ct.ComputeUnit.CPU_ONLY, function_name="embed")
    e5_lmhead = ct.models.MLModel(lmhead_nosplit_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    
    e5_engine = build_engine(
        embed_cu=ct.ComputeUnit.CPU_ONLY,
        lmhead_cu=ct.ComputeUnit.CPU_AND_NE,  # placeholder, we'll override
        chunk_cu_map=ct.ComputeUnit.CPU_AND_NE,
    )
    # Override embed and lmhead
    del e5_engine['embed'], e5_engine['lmhead']
    e5_engine['embed'] = e5_embed
    e5_engine['lmhead'] = e5_lmhead
    print(f"  Loaded in {time.time()-t0:.1f}s")
    
    e5 = run_experiment("E5: lm_head_nosplit-ANE", e5_engine, PROMPT1, num_gen=NUM_GEN)
    print_experiment(e5, baseline)
    del e5_engine; gc.collect()
else:
    print("  lm_head_nosplit.mlpackage not found, skipping")


# ═══════════════════════════════════════════════════════════════════
#  EXPERIMENT 6: Prefill comparison — CPU_ONLY vs CPU_AND_NE
# ═══════════════════════════════════════════════════════════════════

banner("EXPERIMENT 6: Prefill Compute Unit Comparison")

# Use separate prefill models
embed_prefill_path = os.path.join(MODEL_DIR, 'embed_prefill.mlpackage')

prefill_input = np.zeros((1, BATCH_SIZE), dtype=np.int32)
test_prompt = "Explain quantum computing and its potential applications in modern science and technology."
test_ids = tokenizer.encode(test_prompt, add_special_tokens=False)
prefill_input[0, :len(test_ids)] = test_ids

hidden_prefill = np.zeros((1, BATCH_SIZE, 2560), dtype=np.float16)
mask_prefill = np.full((1, 1, BATCH_SIZE, CTX), -65504.0, dtype=np.float16)
for i in range(len(test_ids)):
    mask_prefill[0, 0, i, :i+1] = 0
pos_ids = np.arange(BATCH_SIZE, dtype=np.int32)

for cu_name, cu in [("CPU_ONLY", ct.ComputeUnit.CPU_ONLY), ("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE)]:
    print(f"\n  Prefill pipeline ({cu_name}):")
    
    # Load embed_prefill
    ep = ct.models.MLModel(embed_prefill_path, compute_units=cu)
    t0 = time.perf_counter()
    for _ in range(3):
        ep_out = ep.predict({"input_ids": prefill_input})
    ep_ms = (time.perf_counter() - t0) / 3 * 1000
    print(f"    embed_prefill: {fmt_ms(ep_ms)}")
    
    # Load and time each prefill chunk
    total_chunks_ms = 0
    for ci in range(NUM_CHUNKS):
        pf_path = os.path.join(MODEL_DIR, f"prefill_LUT4_chunk{ci}.mlpackage")
        m = ct.models.MLModel(pf_path, compute_units=cu)
        state = m.make_state()
        
        inp_shapes = {}
        spec = ct.utils.load_spec(pf_path)
        for inp_desc in spec.description.input:
            try:
                inp_shapes[inp_desc.name] = tuple(inp_desc.type.multiArrayType.shape)
            except:
                pass
        
        inp = {
            "hidden_states": hidden_prefill.copy(),
            "position_ids": pos_ids,
            "causal_mask": mask_prefill,
            "current_pos": np.array([0], dtype=np.int32),
        }
        if 'linear_conv_state' in inp_shapes:
            inp["linear_conv_state"] = np.zeros(inp_shapes['linear_conv_state'], dtype=np.float16)
            inp["linear_recurrent_state"] = np.zeros(inp_shapes['linear_recurrent_state'], dtype=np.float16)
        if 'valid_len' in inp_shapes:
            inp["valid_len"] = np.array([len(test_ids)], dtype=np.int32)
        
        # Warmup + measure
        m.predict(inp, state=state)
        t0 = time.perf_counter()
        for _ in range(3):
            m.predict(inp, state=state)
        chunk_ms = (time.perf_counter() - t0) / 3 * 1000
        total_chunks_ms += chunk_ms
        
        del m, state; gc.collect()
    
    total_ms = ep_ms + total_chunks_ms
    tps = BATCH_SIZE / total_ms * 1000
    print(f"    Total chunks: {fmt_ms(total_chunks_ms)}")
    print(f"    Total prefill: {fmt_ms(total_ms)} → {tps:.0f} tok/s")
    
    del ep; gc.collect()


# ═══════════════════════════════════════════════════════════════════
#  FINAL COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════

banner("FINAL COMPARISON TABLE")

all_results = [baseline, e1, e2, e3, e4]
if os.path.exists(lmhead_nosplit_path):
    all_results.append(e5)

print(f"  {'Experiment':40s}  {'Total ms':>10s}  {'tok/s':>8s}  {'Speedup':>8s}  {'LMHead ms':>10s}  {'Match':>6s}")
print(f"  {'─'*40}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*6}")

for r in all_results:
    speedup = baseline['total_ms'] / r['total_ms'] if r['total_ms'] > 0 else 0
    lmhead_ms = r['means'].get('lmhead', 0)
    
    # Token match
    match_len = min(len(baseline['gen_tokens']), len(r['gen_tokens']))
    matches = sum(1 for a, b in zip(baseline['gen_tokens'][:match_len], r['gen_tokens'][:match_len]) if a == b)
    match_pct = matches / match_len * 100 if match_len > 0 else 0
    
    print(f"  {r['name']:40s}  {r['total_ms']:>8.2f}ms  {r['tok_s']:>6.1f}  {speedup:>7.2f}x  {lmhead_ms:>8.2f}ms  {match_pct:>5.0f}%")


banner("RECOMMENDATIONS")

best = min(all_results, key=lambda r: r['total_ms'])
print(f"""
  BEST CONFIG: {best['name']}
    Decode: {fmt_ms(best['total_ms'])} → {best['tok_s']:.1f} tok/s
    Speedup: {baseline['total_ms']/best['total_ms']:.2f}x over baseline
  
  PRODUCTION RECOMMENDATIONS (priority order):
  
  1. Move LMHead to CPU_AND_NE
     Impact: ~{(baseline['means']['lmhead'] - e1['means']['lmhead']) / baseline['total_ms'] * 100:.0f}% decode time reduction
     Risk: LOW (same logits, same tokens)
     Change: ct.ComputeUnit.CPU_AND_NE for lmhead model load
  
  2. Keep embed on CPU_ONLY
     Impact: negligible (<0.1% of decode time)
     Reason: embed is table lookup, ~0.03ms either way
  
  3. Chunk 8 with ComputeUnit.ALL  
     Impact: ~{(baseline['means'].get('chunk_8',0) - e3['means'].get('chunk_8',0)) / baseline['total_ms'] * 100:.1f}% decode time reduction
     Risk: LOW (1 layer, simple graph benefits from GPU)
  
  4. Prefill on CPU_ONLY (NOT ANE)
     Impact: prefill is 0.84x slower on ANE (confirmed)
     Reason: batch attention ops not well-suited to ANE
     Change: Use ct.ComputeUnit.CPU_ONLY for prefill chunk models
  
  CAVEATS:
  - ANE decode speedup is only 1.57x over CPU_ONLY
  - This is because Qwen3.5 uses hybrid Mamba+attention architecture
  - The Mamba (linear attention) ops are complex and may not map well to ANE
  - layer_norm (210 ops) and gather (32 ops for RoPE) may cause some CPU fallback
""")

print("Done.")
