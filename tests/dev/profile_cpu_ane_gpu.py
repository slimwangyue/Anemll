#!/usr/bin/env python3
"""Profile CPU vs ANE vs GPU for Qwen3.5-4B P2 deduped models.

Compares latency and throughput across all three compute units:
  - CPU_ONLY
  - CPU_AND_NE (ANE)
  - CPU_AND_GPU (GPU)

Profiles:
  1. Per-component latency (embed, ffn chunks, lm_head) — single-token decode
  2. Full pipeline decode (10 tokens) — end-to-end
  3. Prefill throughput (batch=512)

Usage:
    TMPDIR=/Volumes/MySSD/tmp python tests/dev/profile_cpu_ane_gpu.py
    TMPDIR=/Volumes/MySSD/tmp python tests/dev/profile_cpu_ane_gpu.py --model-dir qwen3_5_v4_lut4_p2
    TMPDIR=/Volumes/MySSD/tmp python tests/dev/profile_cpu_ane_gpu.py --decode-tokens 20
"""
import sys, os, gc, time, argparse, resource, warnings
warnings.filterwarnings('ignore')

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts_qwen3_5'))
os.chdir(REPO_ROOT)

import numpy as np
import coremltools as ct

from config import CTX, NUM_CHUNKS, BATCH_SIZE

# ── Compute units to test ────────────────────────────────────────────
CU_MAP = {
    'CPU_ONLY':    ct.ComputeUnit.CPU_ONLY,
    'CPU_AND_NE':  ct.ComputeUnit.CPU_AND_NE,
    'CPU_AND_GPU': ct.ComputeUnit.CPU_AND_GPU,
}

HIDDEN = 2560
N_WARMUP = 3
N_RUNS = 15


def find_model(base_dir, name):
    # Prefer .mlpackage (always has Manifest); fall back to .mlmodelc
    for ext in ('.mlpackage', '.mlmodelc'):
        p = os.path.join(base_dir, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model found for {name} in {base_dir}")


def load_model(path, cu, function_name=None):
    kwargs = {'compute_units': cu}
    if function_name:
        kwargs['function_name'] = function_name
    return ct.models.MLModel(path, **kwargs)


def measure(fn, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """Measure wall and CPU time for fn(), return (wall_ms, cpu_ms)."""
    for _ in range(n_warmup):
        fn()
    walls, cpus = [], []
    for _ in range(n_runs):
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.perf_counter()
        fn()
        wall = (time.perf_counter() - t0) * 1000
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        cpu = ((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)) * 1000
        walls.append(wall)
        cpus.append(cpu)
    return float(np.median(walls)), float(np.median(cpus))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='qwen3_5_v4_lut4_p2')
    parser.add_argument('--decode-tokens', type=int, default=10,
                        help='Number of decode tokens for full-pipeline test')
    parser.add_argument('--skip-prefill', action='store_true',
                        help='Skip prefill benchmark')
    args = parser.parse_args()

    model_dir = args.model_dir
    combined_dir = os.path.join(model_dir, 'combined_LUT4_dedup')
    use_combined = os.path.isdir(combined_dir)

    print('=' * 90)
    print('  CPU vs ANE vs GPU Profiling — Qwen3.5-4B P2')
    print(f'  Models: {model_dir}')
    print(f'  Combined dedup: {use_combined}')
    print(f'  CTX={CTX}, BATCH={BATCH_SIZE}, chunks={NUM_CHUNKS}')
    print(f'  Decode tokens: {args.decode_tokens}, Warmup: {N_WARMUP}, Runs: {N_RUNS}')
    print('=' * 90)

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1: Per-component single-token latency
    # ═══════════════════════════════════════════════════════════════════
    print('\n' + '=' * 90)
    print('  SECTION 1: Per-component single-token decode latency')
    print('=' * 90)

    # Prepare inputs
    np.random.seed(42)
    embed_input = {'input_ids': np.array([[1]], dtype=np.int32)}
    mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    mask[:, :, :, :10] = 0
    pos = np.array([9], dtype=np.int32)

    results = {}

    for cu_name, cu in CU_MAP.items():
        print(f'\n--- {cu_name} ---')
        component_times = {}

        # Embed
        try:
            embed_path = find_model(model_dir, 'embed_single')
            m_embed = load_model(embed_path, cu)
            w, c = measure(lambda: m_embed.predict(embed_input))
            component_times['embed'] = (w, c)
            print(f'  embed_single:    wall={w:7.2f}ms  cpu={c:7.2f}ms')
            hidden_out = m_embed.predict(embed_input)
            hidden = list(hidden_out.values())[0].astype(np.float16)
            del m_embed; gc.collect()
        except Exception as e:
            print(f'  embed_single:    FAILED — {e}')
            hidden = np.random.randn(1, 1, HIDDEN).astype(np.float16) * 0.01

        # FFN chunks (first 3 + last, to save time)
        test_chunks = [0, 1, 2, NUM_CHUNKS - 1] if NUM_CHUNKS > 4 else list(range(NUM_CHUNKS))
        chunk_total_w, chunk_total_c = 0, 0

        for ci in test_chunks:
            try:
                if use_combined:
                    chunk_path = find_model(combined_dir, f'chunk{ci}')
                    m_ffn = load_model(chunk_path, cu, function_name='infer')
                else:
                    chunk_path = find_model(model_dir, f'ffn_LUT4_chunk{ci}')
                    m_ffn = load_model(chunk_path, cu)

                state = m_ffn.make_state()
                spec = m_ffn.get_spec()
                # Detect state shapes — check function-specific inputs first
                inp_map = {}
                fn_inputs = None
                if hasattr(spec.description, 'functions'):
                    for fn in spec.description.functions:
                        if fn.name == 'infer':
                            fn_inputs = fn.input
                            break
                if fn_inputs is None:
                    fn_inputs = spec.description.input
                for inp in fn_inputs:
                    try:
                        inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
                    except Exception:
                        pass

                ffn_input = {
                    'hidden_states': hidden,
                    'position_ids': pos,
                    'causal_mask': mask,
                    'current_pos': pos,
                }
                if 'linear_conv_state' in inp_map:
                    ffn_input['linear_conv_state'] = np.zeros(inp_map['linear_conv_state'], dtype=np.float16)
                    ffn_input['linear_recurrent_state'] = np.zeros(inp_map['linear_recurrent_state'], dtype=np.float16)

                w, c = measure(lambda: m_ffn.predict(ffn_input, state=state))
                component_times[f'ffn_chunk{ci}'] = (w, c)
                chunk_total_w += w
                chunk_total_c += c
                label = f'combined' if use_combined else f'separate'
                print(f'  ffn chunk{ci} ({label}): wall={w:7.2f}ms  cpu={c:7.2f}ms')

                # Pass hidden through for next chunk
                out = m_ffn.predict(ffn_input, state=state)
                hidden = out['output_hidden_states'].astype(np.float16)
                del m_ffn, state; gc.collect()
            except Exception as e:
                print(f'  ffn chunk{ci}:    FAILED — {e}')

        # Estimate total FFN time (extrapolate from sampled chunks)
        if len(test_chunks) < NUM_CHUNKS and len(test_chunks) > 0:
            avg_w = chunk_total_w / len(test_chunks)
            avg_c = chunk_total_c / len(test_chunks)
            est_total_w = avg_w * NUM_CHUNKS
            est_total_c = avg_c * NUM_CHUNKS
            print(f'  ffn ALL {NUM_CHUNKS} est:  wall={est_total_w:7.2f}ms  cpu={est_total_c:7.2f}ms  (extrapolated from {len(test_chunks)} chunks)')
            component_times['ffn_total_est'] = (est_total_w, est_total_c)
        else:
            component_times['ffn_total_est'] = (chunk_total_w, chunk_total_c)

        # LM Head
        try:
            lm_path = find_model(model_dir, 'lm_head_nosplit')
            m_lm = load_model(lm_path, cu)
            lm_input = {'hidden_states': hidden}
            w, c = measure(lambda: m_lm.predict(lm_input))
            component_times['lm_head'] = (w, c)
            print(f'  lm_head_nosplit: wall={w:7.2f}ms  cpu={c:7.2f}ms')
            del m_lm; gc.collect()
        except Exception as e:
            print(f'  lm_head:         FAILED — {e}')

        # Total estimate
        total_w = sum(v[0] for v in component_times.values() if 'chunk' not in v or 'est' in v)
        # Use estimated FFN total
        if 'embed' in component_times and 'ffn_total_est' in component_times and 'lm_head' in component_times:
            pipe_w = component_times['embed'][0] + component_times['ffn_total_est'][0] + component_times['lm_head'][0]
            pipe_c = component_times['embed'][1] + component_times['ffn_total_est'][1] + component_times['lm_head'][1]
            tps = 1000.0 / pipe_w if pipe_w > 0 else 0
            print(f'  ─────────────────')
            print(f'  PIPELINE TOTAL:  wall={pipe_w:7.2f}ms  cpu={pipe_c:7.2f}ms  → {tps:.1f} t/s')

        results[cu_name] = component_times

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 2: Full pipeline decode (actual token generation)
    # ═══════════════════════════════════════════════════════════════════
    print('\n' + '=' * 90)
    print(f'  SECTION 2: Full pipeline decode ({args.decode_tokens} tokens)')
    print('=' * 90)

    for cu_name, cu in CU_MAP.items():
        print(f'\n--- {cu_name} ---')
        try:
            # Load all models
            m_embed = load_model(find_model(model_dir, 'embed_single'), cu)
            m_lm = load_model(find_model(model_dir, 'lm_head_nosplit'), cu)

            ffns = []
            states = []
            inp_maps = []
            for ci in range(NUM_CHUNKS):
                if use_combined:
                    p = find_model(combined_dir, f'chunk{ci}')
                    m = load_model(p, cu, function_name='infer')
                else:
                    p = find_model(model_dir, f'ffn_LUT4_chunk{ci}')
                    m = load_model(p, cu)
                ffns.append(m)
                states.append(m.make_state())
                spec = m.get_spec()
                imap = {}
                fn_inputs = None
                if hasattr(spec.description, 'functions'):
                    for fn in spec.description.functions:
                        if fn.name == 'infer':
                            fn_inputs = fn.input
                            break
                if fn_inputs is None:
                    fn_inputs = spec.description.input
                for inp in fn_inputs:
                    try:
                        imap[inp.name] = tuple(inp.type.multiArrayType.shape)
                    except Exception:
                        pass
                inp_maps.append(imap)

            # Detect lm_head output name
            lm_spec = m_lm.get_spec()
            lm_out_names = [o.name for o in lm_spec.description.output]
            logits_key = 'logits1' if 'logits1' in lm_out_names else 'logits' if 'logits' in lm_out_names else lm_out_names[0]

            # Pre-allocate
            lin_convs = [np.zeros(inp_maps[ci].get('linear_conv_state', (4, 1024, 32)), dtype=np.float16)
                         for ci in range(NUM_CHUNKS)]
            lin_recs = [np.zeros(inp_maps[ci].get('linear_recurrent_state', (4, 32, 128, 128)), dtype=np.float16)
                        for ci in range(NUM_CHUNKS)]

            def decode_step(tok_id, pos_val):
                tok = np.array([[tok_id]], dtype=np.int32)
                hidden = list(m_embed.predict({'input_ids': tok}).values())[0].astype(np.float16)
                m = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
                m[:, :, :, :pos_val + 1] = 0
                p = np.array([pos_val], dtype=np.int32)
                for ci in range(NUM_CHUNKS):
                    inp = {
                        'hidden_states': hidden,
                        'position_ids': p,
                        'causal_mask': m,
                        'current_pos': p,
                    }
                    if 'linear_conv_state' in inp_maps[ci]:
                        inp['linear_conv_state'] = lin_convs[ci]
                        inp['linear_recurrent_state'] = lin_recs[ci]
                    out = ffns[ci].predict(inp, state=states[ci])
                    hidden = out['output_hidden_states'].astype(np.float16)
                    if 'linear_conv_state_out' in out:
                        lin_convs[ci] = out['linear_conv_state_out']
                        lin_recs[ci] = out['linear_recurrent_state_out']
                lm_out = m_lm.predict({'hidden_states': hidden})
                return int(np.argmax(lm_out[logits_key].flatten()))

            # Warmup (2 tokens)
            for i in range(2):
                decode_step(1, i)

            # Timed decode
            gen_tokens = args.decode_tokens
            step_times = []
            tok = 1
            for i in range(gen_tokens):
                r0 = resource.getrusage(resource.RUSAGE_SELF)
                t0 = time.perf_counter()
                tok = decode_step(tok, 2 + i)
                wall = (time.perf_counter() - t0) * 1000
                r1 = resource.getrusage(resource.RUSAGE_SELF)
                cpu = ((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)) * 1000
                step_times.append((wall, cpu))

            med_wall = float(np.median([t[0] for t in step_times]))
            med_cpu = float(np.median([t[1] for t in step_times]))
            tps = 1000.0 / med_wall if med_wall > 0 else 0
            p5_wall = float(np.percentile([t[0] for t in step_times], 5))
            p95_wall = float(np.percentile([t[0] for t in step_times], 95))
            print(f'  Median step:  wall={med_wall:7.2f}ms  cpu={med_cpu:7.2f}ms  → {tps:.1f} t/s')
            print(f'  P5/P95 wall:  {p5_wall:.2f} / {p95_wall:.2f} ms')

            # Cleanup
            del m_embed, m_lm
            for m in ffns:
                del m
            ffns.clear()
            gc.collect()

        except Exception as e:
            print(f'  FAILED: {e}')
            import traceback; traceback.print_exc()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3: Prefill throughput (batch=BATCH_SIZE)
    # ═══════════════════════════════════════════════════════════════════
    if not args.skip_prefill:
        print('\n' + '=' * 90)
        print(f'  SECTION 3: Prefill throughput (batch={BATCH_SIZE})')
        print('=' * 90)

        for cu_name, cu in CU_MAP.items():
            print(f'\n--- {cu_name} ---')
            try:
                # Embeddings prefill
                embed_pf_path = find_model(model_dir, 'embed_prefill')
                m_embed_pf = load_model(embed_pf_path, cu)
                pf_ids = np.ones((1, BATCH_SIZE), dtype=np.int32)
                w, c = measure(lambda: m_embed_pf.predict({'input_ids': pf_ids}), n_warmup=2, n_runs=5)
                print(f'  embed_prefill:   wall={w:7.1f}ms  cpu={c:7.1f}ms  ({BATCH_SIZE/w*1000:.0f} tok/s)')

                hidden_pf = list(m_embed_pf.predict({'input_ids': pf_ids}).values())[0].astype(np.float16)
                del m_embed_pf; gc.collect()

                # Prefill through chunk 0 only (representative)
                if use_combined:
                    p = find_model(combined_dir, 'chunk0')
                    m_pf = load_model(p, cu, function_name='prefill')
                else:
                    p = find_model(model_dir, 'prefill_LUT4_chunk0')
                    m_pf = load_model(p, cu)

                state_pf = m_pf.make_state()
                spec = m_pf.get_spec()
                imap = {}
                fn_inputs = None
                if hasattr(spec.description, 'functions'):
                    for fn in spec.description.functions:
                        if fn.name == 'prefill':
                            fn_inputs = fn.input
                            break
                if fn_inputs is None:
                    fn_inputs = spec.description.input
                for inp in fn_inputs:
                    try:
                        imap[inp.name] = tuple(inp.type.multiArrayType.shape)
                    except Exception:
                        pass

                mask_pf = np.zeros((1, 1, BATCH_SIZE, CTX), dtype=np.float16)
                # lower-triangular causal mask
                for i in range(BATCH_SIZE):
                    mask_pf[:, :, i, i + 1:] = -65504.0
                pos_pf = np.arange(BATCH_SIZE, dtype=np.int32)

                pf_inp = {
                    'hidden_states': hidden_pf,
                    'position_ids': pos_pf,
                    'causal_mask': mask_pf,
                    'current_pos': np.array([BATCH_SIZE - 1], dtype=np.int32),
                }
                if 'linear_conv_state' in imap:
                    pf_inp['linear_conv_state'] = np.zeros(imap['linear_conv_state'], dtype=np.float16)
                    pf_inp['linear_recurrent_state'] = np.zeros(imap['linear_recurrent_state'], dtype=np.float16)

                w, c = measure(lambda: m_pf.predict(pf_inp, state=state_pf), n_warmup=2, n_runs=5)
                print(f'  prefill chunk0:  wall={w:7.1f}ms  cpu={c:7.1f}ms  ({BATCH_SIZE/w*1000:.0f} tok/s)')

                del m_pf; gc.collect()

            except Exception as e:
                print(f'  FAILED: {e}')
                import traceback; traceback.print_exc()

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════
    print('\n' + '=' * 90)
    print('  SUMMARY — Per-component median wall time (ms)')
    print('=' * 90)

    # Header
    components = ['embed', 'ffn_total_est', 'lm_head']
    labels = ['Embed', 'FFN (9 chunks)', 'LM Head']
    print(f'{"Component":20s}', end='')
    for cu_name in CU_MAP:
        print(f'  {cu_name:>14s}', end='')
    print(f'  {"Best":>8s}')
    print('-' * 78)

    for comp, label in zip(components, labels):
        print(f'{label:20s}', end='')
        vals = {}
        for cu_name in CU_MAP:
            if cu_name in results and comp in results[cu_name]:
                w = results[cu_name][comp][0]
                vals[cu_name] = w
                print(f'  {w:11.2f} ms', end='')
            else:
                print(f'  {"N/A":>14s}', end='')
        if vals:
            best = min(vals, key=vals.get)
            print(f'  {best:>8s}')
        else:
            print()

    # Pipeline total
    print('-' * 78)
    print(f'{"PIPELINE TOTAL":20s}', end='')
    pipe_vals = {}
    for cu_name in CU_MAP:
        if cu_name in results:
            r = results[cu_name]
            if 'embed' in r and 'ffn_total_est' in r and 'lm_head' in r:
                total = r['embed'][0] + r['ffn_total_est'][0] + r['lm_head'][0]
                pipe_vals[cu_name] = total
                tps = 1000.0 / total if total > 0 else 0
                print(f'  {total:7.1f}ms {tps:4.1f}t/s', end='')
            else:
                print(f'  {"N/A":>14s}', end='')
        else:
            print(f'  {"N/A":>14s}', end='')
    if pipe_vals:
        best = min(pipe_vals, key=pipe_vals.get)
        worst = max(pipe_vals, key=pipe_vals.get)
        speedup = pipe_vals[worst] / pipe_vals[best] if pipe_vals[best] > 0 else 0
        print(f'  {best:>8s} ({speedup:.1f}x)')
    else:
        print()

    print('\n' + '=' * 90)
    print('  SPEEDUP vs CPU_ONLY')
    print('=' * 90)
    if 'CPU_ONLY' in results:
        cpu_r = results['CPU_ONLY']
        for cu_name in ['CPU_AND_NE', 'CPU_AND_GPU']:
            if cu_name in results:
                r = results[cu_name]
                print(f'\n  {cu_name}:')
                for comp, label in zip(components, labels):
                    if comp in cpu_r and comp in r:
                        cpu_w = cpu_r[comp][0]
                        cu_w = r[comp][0]
                        speedup = cpu_w / cu_w if cu_w > 0 else 0
                        print(f'    {label:20s}: {speedup:5.2f}x  ({cpu_w:.1f}ms → {cu_w:.1f}ms)')

    print('\nDone.')


if __name__ == '__main__':
    main()
