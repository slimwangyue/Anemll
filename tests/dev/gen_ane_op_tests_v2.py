"""Generate targeted CoreML test models for ANE-blocking op candidates.

Round 2: less/one_hot/gather all passed. Now test:
  - constexpr_lut_to_dense (LUT6 quantization)
  - coreml_update_state + read_state (stateful KV cache)
  - slice_update
  - Combinations and scale
"""

import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
from coremltools.optimize.coreml import (
    OpPalettizerConfig,
    OptimizationConfig,
    palettize_weights,
)
import os, torch, torch.nn as nn

OUT_DIR = "/Users/yw68/Anemll/tests/dev/ane_op_test_models"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ = 512
HIDDEN = 64
HEAD_DIM = 16
NUM_HEADS = 4
CTX = 2048

lut_config = OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=6))


def save_traced(ct_model, name):
    path = os.path.join(OUT_DIR, f"{name}.mlpackage")
    ct_model.save(path)
    spec = ct_model.get_spec()
    prog = ct.models.utils._milproto_to_pymil.load(spec, spec.specificationVersion, ct_model.weights_dir)
    op_types = set()
    for fn in prog.functions:
        for op in prog.functions[fn].operations:
            op_types.add(op.op_type)
    print(f"  {name}  ops: {sorted(op_types)}")


def save_mil(prog, name):
    m = ct.convert(prog, compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.iOS18)
    save_traced(m, name)


# ── 1. LUT6 only ──
print("1. test_lut6 (constexpr_lut_to_dense)")
class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(HIDDEN, HIDDEN, bias=False)
    def forward(self, x):
        return self.linear(x)

m1 = ct.convert(
    torch.jit.trace(SimpleLinear().eval().half(), torch.randn(1, SEQ, HIDDEN, dtype=torch.float16)),
    inputs=[ct.TensorType(shape=(1, SEQ, HIDDEN), dtype=np.float16)],
    compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.iOS18,
)
save_traced(palettize_weights(m1, lut_config), "test_lut6")


# ── 2. Stateful model (coreml_update_state + read_state) ──
print("2. test_state (coreml_update_state + read_state)")
class StatefulModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("kv_cache", torch.zeros(1, NUM_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
        self.linear = nn.Linear(HIDDEN, HIDDEN, bias=False)
    def forward(self, x):
        B, S, H = x.shape
        new_kv = x.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        self.kv_cache[:, :, :S, :] = new_kv
        return self.linear(x)

m2 = ct.convert(
    torch.jit.trace(StatefulModel().eval().half(), torch.randn(1, SEQ, HIDDEN, dtype=torch.float16)),
    inputs=[ct.TensorType(shape=(1, SEQ, HIDDEN), dtype=np.float16)],
    states=[ct.StateType(wrapped_type=ct.TensorType(shape=(1, NUM_HEADS, CTX, HEAD_DIM), dtype=np.float16), name="kv_cache")],
    compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.iOS18,
)
save_traced(m2, "test_state")


# ── 3. Stateful + LUT6 ──
print("3. test_state_lut6")
save_traced(palettize_weights(m2, lut_config), "test_state_lut6")


# ── 4. Multi-state (9 layers × 2 caches = 18 states) + LUT6 ──
print("4. test_multi_state_lut6 (9 layers, 18 states)")
class MultiStateModel(nn.Module):
    def __init__(self, n=9):
        super().__init__()
        for i in range(n):
            self.register_buffer(f"k_cache_{i}", torch.zeros(1, NUM_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
            self.register_buffer(f"v_cache_{i}", torch.zeros(1, NUM_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
        self.layers = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN, bias=False) for _ in range(n)])
        self.n = n
    def forward(self, x):
        B, S, H = x.shape
        for i in range(self.n):
            kv = x.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            getattr(self, f"k_cache_{i}")[:, :, :S, :] = kv
            getattr(self, f"v_cache_{i}")[:, :, :S, :] = kv
            x = x + self.layers[i](x)
        return x

ms_model = MultiStateModel(9).eval().half()
ms_states = []
for i in range(9):
    ms_states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, NUM_HEADS, CTX, HEAD_DIM), dtype=np.float16), name=f"k_cache_{i}"))
    ms_states.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, NUM_HEADS, CTX, HEAD_DIM), dtype=np.float16), name=f"v_cache_{i}"))

m4 = ct.convert(
    torch.jit.trace(ms_model, torch.randn(1, SEQ, HIDDEN, dtype=torch.float16)),
    inputs=[ct.TensorType(shape=(1, SEQ, HIDDEN), dtype=np.float16)],
    states=ms_states,
    compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.iOS18,
)
save_traced(palettize_weights(m4, lut_config), "test_multi_state_lut6")


# ── 5. slice_update (MIL program) ──
print("5. test_slice_update")
@mb.program(input_specs=[mb.TensorSpec(shape=(1, SEQ, HIDDEN), dtype=types.fp16)], opset_version=ct.target.iOS18)
def test_slice_update(x):
    buf = mb.const(val=np.zeros((1, CTX, HIDDEN), dtype=np.float16))
    updated = mb.slice_update(x=buf, update=x,
                              begin=np.array([0, 0, 0], dtype=np.int32),
                              end=np.array([1, SEQ, HIDDEN], dtype=np.int32))
    out = mb.slice_by_index(x=updated,
                            begin=np.array([0, 0, 0], dtype=np.int32),
                            end=np.array([1, SEQ, HIDDEN], dtype=np.int32))
    return mb.add(x=x, y=out)
save_mil(test_slice_update, "test_slice_update")


# ── 6. Mini transformer with state + LUT6 ──
print("6. test_mini_transformer_state_lut6")
class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("k_cache", torch.zeros(1, NUM_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
        self.register_buffer("v_cache", torch.zeros(1, NUM_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
        self.ln = nn.LayerNorm(HIDDEN)
        self.qkv = nn.Linear(HIDDEN, 3 * HIDDEN, bias=False)
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.ffn1 = nn.Linear(HIDDEN, HIDDEN * 4, bias=False)
        self.ffn2 = nn.Linear(HIDDEN * 4, HIDDEN, bias=False)
        self.ln2 = nn.LayerNorm(HIDDEN)
    def forward(self, x):
        B, S, _ = x.shape
        h = self.ln(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        self.k_cache[:, :, :S, :] = k
        self.v_cache[:, :, :S, :] = v
        attn = torch.matmul(q, k.transpose(-2, -1)) / (HEAD_DIM ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, HIDDEN)
        out = self.proj(out)
        x = x + out
        h2 = self.ln2(x)
        x = x + self.ffn2(torch.silu(self.ffn1(h2)))
        return x

m6 = ct.convert(
    torch.jit.trace(MiniTransformer().eval().half(), torch.randn(1, SEQ, HIDDEN, dtype=torch.float16)),
    inputs=[ct.TensorType(shape=(1, SEQ, HIDDEN), dtype=np.float16)],
    states=[
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, NUM_HEADS, CTX, HEAD_DIM), dtype=np.float16), name="k_cache"),
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, NUM_HEADS, CTX, HEAD_DIM), dtype=np.float16), name="v_cache"),
    ],
    compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.iOS18,
)
save_traced(palettize_weights(m6, lut_config), "test_mini_transformer_state_lut6")


# ── 7. 9-layer transformer with state + LUT6 ──
print("7. test_9layer_transformer_state_lut6")
class NineLayerTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(9):
            self.register_buffer(f"k_cache_{i}", torch.zeros(1, NUM_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
            self.register_buffer(f"v_cache_{i}", torch.zeros(1, NUM_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
        self.lns = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(9)])
        self.qkvs = nn.ModuleList([nn.Linear(HIDDEN, 3 * HIDDEN, bias=False) for _ in range(9)])
        self.projs = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN, bias=False) for _ in range(9)])
        self.ffn1s = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN * 4, bias=False) for _ in range(9)])
        self.ffn2s = nn.ModuleList([nn.Linear(HIDDEN * 4, HIDDEN, bias=False) for _ in range(9)])
        self.ln2s = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(9)])
    def forward(self, x):
        B, S, _ = x.shape
        for i in range(9):
            h = self.lns[i](x)
            qkv = self.qkvs[i](h)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            k = k.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            v = v.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
            getattr(self, f"k_cache_{i}")[:, :, :S, :] = k
            getattr(self, f"v_cache_{i}")[:, :, :S, :] = v
            attn = torch.matmul(q, k.transpose(-2, -1)) / (HEAD_DIM ** 0.5)
            attn = torch.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, S, HIDDEN)
            out = self.projs[i](out)
            x = x + out
            h2 = self.ln2s[i](x)
            x = x + self.ffn2s[i](torch.silu(self.ffn1s[i](h2)))
        return x

nlstates = []
for i in range(9):
    nlstates.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, NUM_HEADS, CTX, HEAD_DIM), dtype=np.float16), name=f"k_cache_{i}"))
    nlstates.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, NUM_HEADS, CTX, HEAD_DIM), dtype=np.float16), name=f"v_cache_{i}"))

m7 = ct.convert(
    torch.jit.trace(NineLayerTransformer().eval().half(), torch.randn(1, SEQ, HIDDEN, dtype=torch.float16)),
    inputs=[ct.TensorType(shape=(1, SEQ, HIDDEN), dtype=np.float16)],
    states=nlstates,
    compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.iOS18,
)
save_traced(palettize_weights(m7, lut_config), "test_9layer_transformer_state_lut6")


# ── 8. Real-size hidden dim (2560) with LUT6 + state - small model ──
print("8. test_realsize_lut6 (hidden=2560, 1 layer)")
REAL_HIDDEN = 2560
REAL_HEADS = 32
REAL_HD = 80
class RealSizeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("k_cache", torch.zeros(1, REAL_HEADS, CTX, REAL_HD, dtype=torch.float16))
        self.register_buffer("v_cache", torch.zeros(1, REAL_HEADS, CTX, REAL_HD, dtype=torch.float16))
        self.ln = nn.LayerNorm(REAL_HIDDEN)
        self.qkv = nn.Linear(REAL_HIDDEN, 3 * REAL_HIDDEN, bias=False)
        self.proj = nn.Linear(REAL_HIDDEN, REAL_HIDDEN, bias=False)
        self.ffn1 = nn.Linear(REAL_HIDDEN, REAL_HIDDEN * 4, bias=False)
        self.ffn2 = nn.Linear(REAL_HIDDEN * 4, REAL_HIDDEN, bias=False)
        self.ln2 = nn.LayerNorm(REAL_HIDDEN)
    def forward(self, x):
        B, S, _ = x.shape
        h = self.ln(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, REAL_HEADS, REAL_HD).transpose(1, 2)
        k = k.view(B, S, REAL_HEADS, REAL_HD).transpose(1, 2)
        v = v.view(B, S, REAL_HEADS, REAL_HD).transpose(1, 2)
        self.k_cache[:, :, :S, :] = k
        self.v_cache[:, :, :S, :] = v
        attn = torch.matmul(q, k.transpose(-2, -1)) / (REAL_HD ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, REAL_HIDDEN)
        out = self.proj(out)
        x = x + out
        h2 = self.ln2(x)
        x = x + self.ffn2(torch.silu(self.ffn1(h2)))
        return x

m8 = ct.convert(
    torch.jit.trace(RealSizeModel().eval().half(), torch.randn(1, SEQ, REAL_HIDDEN, dtype=torch.float16)),
    inputs=[ct.TensorType(shape=(1, SEQ, REAL_HIDDEN), dtype=np.float16)],
    states=[
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, REAL_HEADS, CTX, REAL_HD), dtype=np.float16), name="k_cache"),
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, REAL_HEADS, CTX, REAL_HD), dtype=np.float16), name="v_cache"),
    ],
    compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.iOS18,
)
save_traced(palettize_weights(m8, lut_config), "test_realsize_lut6")


print("\n=== All models generated ===")
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith('.mlpackage'):
        size_mb = sum(
            os.path.getsize(os.path.join(dp, fn))
            for dp, _, fns in os.walk(os.path.join(OUT_DIR, f))
            for fn in fns
        ) / 1024 / 1024
        print(f"  {f}  ({size_mb:.1f} MB)")
