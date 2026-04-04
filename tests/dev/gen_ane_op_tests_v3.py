"""Generate real-size test models to find the ANE compilation threshold.

Previous tests show: all ops (less, one_hot, gather, LUT6, state, slice_update)
load on ANE individually and with 9 layers + 18 states at tiny (hidden=64) scale.

Now test with real dimensions: hidden=2560, 32 heads, 80 head_dim.
"""

import numpy as np
import coremltools as ct
from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, palettize_weights
import os, torch, torch.nn as nn, torch.nn.functional as F

OUT_DIR = "/Users/yw68/Anemll/tests/dev/ane_op_test_models"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ = 512
CTX = 2048

# Real Qwen3.5-4B dimensions
HIDDEN = 2560
NUM_HEADS = 32
HEAD_DIM = 80
FFN_DIM = 6912  # intermediate_size for Qwen3.5-4B

lut6_config = OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=6))


def save_model(ct_model, name, apply_lut6=True):
    if apply_lut6:
        ct_model = palettize_weights(ct_model, lut6_config)
    path = os.path.join(OUT_DIR, f"{name}.mlpackage")
    ct_model.save(path)
    size_mb = sum(
        os.path.getsize(os.path.join(dp, fn))
        for dp, _, fns in os.walk(path) for fn in fns
    ) / 1024 / 1024
    print(f"  Saved {name} ({size_mb:.1f} MB)")


# ── 1. Real-size 1 layer, no state ──
print("1. test_real_1layer_no_state")
class OneLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(HIDDEN)
        self.qkv = nn.Linear(HIDDEN, 3 * HIDDEN, bias=False)
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.ffn1 = nn.Linear(HIDDEN, FFN_DIM, bias=False)
        self.ffn2 = nn.Linear(FFN_DIM, HIDDEN, bias=False)
        self.ln2 = nn.LayerNorm(HIDDEN)
    def forward(self, x):
        B, S, _ = x.shape
        h = self.ln(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (HEAD_DIM ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, HIDDEN)
        out = self.proj(out)
        x = x + out
        h2 = self.ln2(x)
        x = x + self.ffn2(F.silu(self.ffn1(h2)))
        return x

m1 = ct.convert(
    torch.jit.trace(OneLayer().eval(), torch.randn(1, SEQ, HIDDEN, dtype=torch.float32)),
    inputs=[ct.TensorType(shape=(1, SEQ, HIDDEN), dtype=np.float16)],
    compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.iOS18,
)
save_model(m1, "test_real_1layer_no_state")


# ── 2. Real-size 1 layer with KV state ──
print("2. test_real_1layer_state")
class OneLayerStateful(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("k_cache", torch.zeros(1, NUM_HEADS, CTX, HEAD_DIM, dtype=torch.float32))
        self.register_buffer("v_cache", torch.zeros(1, NUM_HEADS, CTX, HEAD_DIM, dtype=torch.float32))
        self.ln = nn.LayerNorm(HIDDEN)
        self.qkv = nn.Linear(HIDDEN, 3 * HIDDEN, bias=False)
        self.proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.ffn1 = nn.Linear(HIDDEN, FFN_DIM, bias=False)
        self.ffn2 = nn.Linear(FFN_DIM, HIDDEN, bias=False)
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
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, HIDDEN)
        out = self.proj(out)
        x = x + out
        h2 = self.ln2(x)
        x = x + self.ffn2(F.silu(self.ffn1(h2)))
        return x

m2 = ct.convert(
    torch.jit.trace(OneLayerStateful().eval(), torch.randn(1, SEQ, HIDDEN, dtype=torch.float32)),
    inputs=[ct.TensorType(shape=(1, SEQ, HIDDEN), dtype=np.float16)],
    states=[
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, NUM_HEADS, CTX, HEAD_DIM), dtype=np.float16), name="k_cache"),
        ct.StateType(wrapped_type=ct.TensorType(shape=(1, NUM_HEADS, CTX, HEAD_DIM), dtype=np.float16), name="v_cache"),
    ],
    compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.iOS18,
)
save_model(m2, "test_real_1layer_state")


# ── 3. Real-size 9 layers with KV state ──
print("3. test_real_9layer_state")
class NineLayerStateful(nn.Module):
    def __init__(self):
        super().__init__()
        for i in range(9):
            self.register_buffer(f"k_cache_{i}", torch.zeros(1, NUM_HEADS, CTX, HEAD_DIM, dtype=torch.float32))
            self.register_buffer(f"v_cache_{i}", torch.zeros(1, NUM_HEADS, CTX, HEAD_DIM, dtype=torch.float32))
        self.lns = nn.ModuleList([nn.LayerNorm(HIDDEN) for _ in range(9)])
        self.qkvs = nn.ModuleList([nn.Linear(HIDDEN, 3 * HIDDEN, bias=False) for _ in range(9)])
        self.projs = nn.ModuleList([nn.Linear(HIDDEN, HIDDEN, bias=False) for _ in range(9)])
        self.ffn1s = nn.ModuleList([nn.Linear(HIDDEN, FFN_DIM, bias=False) for _ in range(9)])
        self.ffn2s = nn.ModuleList([nn.Linear(FFN_DIM, HIDDEN, bias=False) for _ in range(9)])
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
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(B, S, HIDDEN)
            out = self.projs[i](out)
            x = x + out
            h2 = self.ln2s[i](x)
            x = x + self.ffn2s[i](F.silu(self.ffn1s[i](h2)))
        return x

states_9 = []
for i in range(9):
    states_9.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, NUM_HEADS, CTX, HEAD_DIM), dtype=np.float16), name=f"k_cache_{i}"))
    states_9.append(ct.StateType(wrapped_type=ct.TensorType(shape=(1, NUM_HEADS, CTX, HEAD_DIM), dtype=np.float16), name=f"v_cache_{i}"))

m3 = ct.convert(
    torch.jit.trace(NineLayerStateful().eval(), torch.randn(1, SEQ, HIDDEN, dtype=torch.float32)),
    inputs=[ct.TensorType(shape=(1, SEQ, HIDDEN), dtype=np.float16)],
    states=states_9,
    compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.iOS18,
)
save_model(m3, "test_real_9layer_state")


print("\n=== Real-size models generated ===")
for f in sorted(os.listdir(OUT_DIR)):
    if f.startswith("test_real") and f.endswith('.mlpackage'):
        size_mb = sum(
            os.path.getsize(os.path.join(dp, fn))
            for dp, _, fns in os.walk(os.path.join(OUT_DIR, f)) for fn in fns
        ) / 1024 / 1024
        print(f"  {f}  ({size_mb:.1f} MB)")
