
import os, sys, time, gc, shutil, numpy as np, torch
sys.path.insert(0, '/Users/yw68/Anemll')
sys.path.insert(0, '/Users/yw68/Anemll/scripts_qwen3_5')
from config import BATCH_SIZE, CTX, NUM_CHUNKS, LUT_BITS, PER_CHANNEL, DEFAULT_HF_MODEL
from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE, ane_conv_state_shape
from anemll.models import qwen3_5_model as qwen35_mod
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
import coremltools as ct

CHUNK_IDX = 0
OUT_DIR = '/tmp/qwen35_ane_safe'

def cosine_sim(a, b):
    af, bf = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-12))

def main():
    print('Loading model...')
    cfg = Qwen35Config.from_json(os.path.join(DEFAULT_HF_MODEL, 'config.json'))
    cfg.context_length = CTX; cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(DEFAULT_HF_MODEL)
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    base, rem = divmod(cfg.num_hidden_layers, NUM_CHUNKS)
    start = CHUNK_IDX * base + min(CHUNK_IDX, rem)
    end = start + base + (1 if CHUNK_IDX < rem else 0)
    local = end - start
    cd = cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2 + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
    ck = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    ad1, ad2 = ane_conv_state_shape(cd, ck)
    torch.manual_seed(42)
    inputs = dict(
        hidden_states=(torch.randn(1, 1, cfg.hidden_size, dtype=torch.float16)*0.1).numpy(),
        position_ids=np.array([5], dtype=np.int32),
        causal_mask=np.zeros((1,1,1,CTX), dtype=np.float16),
        current_pos=np.array([5], dtype=np.int32),
        linear_conv_state=np.zeros((local, ad1, ad2), dtype=np.float16),
        linear_recurrent_state=np.zeros((local, cfg.text_config.linear_num_value_heads, cfg.text_config.linear_key_head_dim, cfg.text_config.linear_value_head_dim), dtype=np.float16),
    )
    print(f'Chunk {CHUNK_IDX}: layers {start}..{end-1}')
    qwen35_mod.ANE_SAFE_NUMERICS = False
    model.model.kv_cache_0.zero_()
    with torch.no_grad():
        ref = model.model.process_layers_regular_single_token_export_local_state(
            hidden_states=torch.from_numpy(inputs['hidden_states']),
            position_ids=torch.from_numpy(inputs['position_ids']),
            causal_mask=torch.from_numpy(inputs['causal_mask']),
            current_pos=torch.from_numpy(inputs['current_pos']), kv_cache_0=None,
            k_cache=torch.zeros(local, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim, dtype=MODEL_DTYPE),
            v_cache=torch.zeros(local, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim, dtype=MODEL_DTYPE),
            linear_conv_state=torch.zeros((local, ad1, ad2), dtype=MODEL_DTYPE),
            linear_recurrent_state=torch.zeros((local,cfg.text_config.linear_num_value_heads,cfg.text_config.linear_key_head_dim,cfg.text_config.linear_value_head_dim), dtype=MODEL_DTYPE),
            start_layer=start, end_layer=end, apply_final_norm=False,
        ).numpy()
    print(f'Ref: range=[{ref.min():.4f}, {ref.max():.4f}]')

    configs = [('A: LUT4 baseline', 4, False), ('B: LUT4 ane-safe', 4, True), ('C: FP16 baseline', None, False), ('D: FP16 ane-safe', None, True)]
    results = {}
    for label, lut, safe in configs:
        print(f'\n[{label}]')
        tag = label.replace(' ','_').replace(':','')
        odir = os.path.join(OUT_DIR, tag)
        os.makedirs(odir, exist_ok=True)
        pkg = os.path.join(odir, f'c{CHUNK_IDX}.mlpackage')
        try:
            t0 = time.time()
            conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE, num_chunks=NUM_CHUNKS, lut_bits=lut, per_channel=PER_CHANNEL, ane_safe_numerics=safe)
            ml = conv.convert_part_2(model, chunk_idx=CHUNK_IDX, total_chunks=NUM_CHUNKS)
            ml.save(pkg); del ml, conv; gc.collect()
            print(f'  Exported ({time.time()-t0:.1f}s)')
        except Exception as e:
            import traceback; traceback.print_exc()
            shutil.rmtree(odir, ignore_errors=True); results[label]=None; continue
        try:
            mlm = ct.models.MLModel(pkg, compute_units=ct.ComputeUnit.CPU_AND_NE)
            out = np.array(mlm.predict(inputs, mlm.make_state())['output_hidden_states'])
            cos = cosine_sim(ref, out)
            mad = float(np.abs(ref.astype(np.float32)-out.astype(np.float32)).max())
            print(f'  cosine={cos:.6f}, mad={mad:.6f}')
            for _ in range(5): mlm.predict(inputs, mlm.make_state())
            times = []
            for _ in range(30):
                st=mlm.make_state(); t0=time.perf_counter(); mlm.predict(inputs,st); times.append((time.perf_counter()-t0)*1000)
            times.sort(); tr=max(1,len(times)//10); trimmed=times[tr:-tr]
            lat = sum(trimmed)/len(trimmed)
            print(f'  latency={lat:.2f}ms')
            results[label] = dict(cos=cos, mad=mad, lat=lat)
        except Exception as e:
            import traceback; traceback.print_exc(); results[label]=None
        finally:
            try: del mlm
            except: pass
            gc.collect(); shutil.rmtree(odir, ignore_errors=True); print('  [cleaned]')

    print('\n' + '='*80)
    print(f'  ANE-SAFE NUMERICS - chunk {CHUNK_IDX}, layers {start}..{end-1}')
    print('='*80)
    for lb, r in results.items():
        if r: print(f'  {lb:<24s} cos={r["cos"]:.6f} mad={r["mad"]:.6f} lat={r["lat"]:.1f}ms')
        else: print(f'  {lb:<24s} FAILED')
    a,b = results.get('A: LUT4 baseline'), results.get('B: LUT4 ane-safe')
    c,d = results.get('C: FP16 baseline'), results.get('D: FP16 ane-safe')
    if a and b: print(f'  LUT4 improvement: {a["cos"]:.6f} -> {b["cos"]:.6f} (delta={b["cos"]-a["cos"]:+.6f})')
    if c and d: print(f'  FP16 improvement: {c["cos"]:.6f} -> {d["cos"]:.6f} (delta={d["cos"]-c["cos"]:+.6f})')

if __name__ == '__main__': main()
