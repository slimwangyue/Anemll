#!/usr/bin/env python3
"""Quick test: export one chunk with per-layer KV states."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from anemll.models.qwen3_5_model import Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter
import coremltools as ct

HF = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
OUT = "/tmp/qwen35_per_layer_test"
CTX = 1024
os.makedirs(OUT, exist_ok=True)

print("Loading model...")
cfg = Qwen35Config.from_json(os.path.join(HF, "config.json"))
cfg.context_length = CTX
cfg.state_length = CTX
model = Qwen35ForCausalLM(cfg)
assert model.load_pretrained_weights(HF), "Failed to load weights"
model.eval()
for p in model.parameters():
    p.requires_grad = False
print(f"Loaded {model.config.num_hidden_layers} layers")

converter = Qwen35Converter(
    model=model,
    context_length=CTX,
    batch_size=256,
    lut_bits=None,  # skip quantization for speed
)

print("Exporting chunk 0 (decode)...")
t0 = time.time()
ml = converter.convert_part_2(model, chunk_idx=0, total_chunks=4)
print(f"  Export OK in {time.time()-t0:.0f}s")

path = os.path.join(OUT, "ffn_chunk0.mlpackage")
ml.save(path)
print(f"  Saved to {path}")

# Check states
spec = ct.utils.load_spec(path)
print(f"  States: {len(spec.description.state)}")
for st in spec.description.state:
    tp = st.type.stateType.arrayType
    shape = list(tp.shape)
    print(f"    {st.name}: shape={shape}")

# Try loading on ANE
print("\nTesting ANE load...")
try:
    loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    state = loaded.make_state()
    print("  ANE load: OK")
    print("  make_state: OK")
except Exception as e:
    print(f"  ANE load: FAILED - {e}")
    sys.exit(1)

# Try predict
import numpy as np
spec2 = loaded.get_spec()
inp_map = {}
for inp in spec2.description.input:
    try:
        inp_map[inp.name] = list(inp.type.multiArrayType.shape)
    except Exception:
        pass
print(f"  Inputs: {list(inp_map.keys())}")

hidden = np.random.randn(1, 1, 2560).astype(np.float16)
mask = np.full((1, 1, 1, 1024), -65504.0, dtype=np.float16)
mask[:, :, :, :1] = 0
kv_write_end = np.zeros((1,), dtype=np.int32)

feed = {
    "hidden_states": hidden,
    "position_ids": np.array([0], dtype=np.int32),
    "causal_mask": mask,
    "current_pos": np.array([0], dtype=np.int32),
    "kv_write_end": kv_write_end,
}
# Add linear state inputs if present
for name, shape in inp_map.items():
    if name not in feed:
        feed[name] = np.zeros(shape, dtype=np.float16)

try:
    out = loaded.predict(feed, state=state)
    print("  ANE predict pos=0: OK")
except Exception as e:
    print(f"  ANE predict pos=0: FAILED - {str(e)[:150]}")

# Test at pos=5
kv_write_end2 = np.zeros((6,), dtype=np.int32)
feed["kv_write_end"] = kv_write_end2
try:
    out = loaded.predict(feed, state=state)
    print("  ANE predict pos=5: OK")
except Exception as e:
    print(f"  ANE predict pos=5: FAILED - {str(e)[:150]}")

# Test at pos=100
kv_write_end3 = np.zeros((101,), dtype=np.int32)
feed["kv_write_end"] = kv_write_end3
try:
    out = loaded.predict(feed, state=state)
    print("  ANE predict pos=100: OK")
except Exception as e:
    print(f"  ANE predict pos=100: FAILED - {str(e)[:150]}")

print("\nDone!")
