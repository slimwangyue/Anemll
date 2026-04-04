"""Generate test models for Qwen3.5-specific linear attention patterns.

Previous tests showed: real-size 9-layer transformers with LUT6 + state load on ANE.
The blocker must be something specific to the Qwen3.5 linear attention.

Test candidates:
  - Depthwise conv2d with thousands of groups (8192 groups)
  - softplus activation
  - repeat_interleave (tile ops)
  - Combined linear attention patterns
  - Chunk loop with tril matmul
"""

import numpy as np
import coremltools as ct
from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, palettize_weights
import os, torch, torch.nn as nn, torch.nn.functional as F

OUT_DIR = "/Users/yw68/Anemll/tests/dev/ane_op_test_models"
os.makedirs(OUT_DIR, exist_ok=True)

SEQ = 512
CTX = 2048
HIDDEN = 2560
NUM_HEADS = 32
HEAD_DIM = 80

# Linear attention dims from Qwen3.5-4B
KEY_DIM = 128   # per-head key dim for linear attention
VALUE_DIM = 128 # per-head value dim for linear attention
NUM_KV_HEADS = 4  # number of KV heads for linear attention
CONV_DIM = NUM_KV_HEADS * (2 * KEY_DIM + VALUE_DIM)  # 4 * 384 = 1536
CONV_KERNEL = 4

lut6_config = OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=6))


def save_model(ct_model, name, apply_lut6=False):
    if apply_lut6:
        ct_model = palettize_weights(ct_model, lut6_config)
    path = os.path.join(OUT_DIR, f"{name}.mlpackage")
    ct_model.save(path)
    # Count ops
    spec = ct_model.get_spec()
    prog = ct.models.utils._milproto_to_pymil.load(spec, spec.specificationVersion, ct_model.weights_dir)
    op_counts = {}
    for fn in prog.functions:
        for op in prog.functions[fn].operations:
            op_counts[op.op_type] = op_counts.get(op.op_type, 0) + 1
    total = sum(op_counts.values())
    size_mb = sum(
        os.path.getsize(os.path.join(dp, fn))
        for dp, _, fns in os.walk(path) for fn in fns
    ) / 1024 / 1024
    print(f"  {name} ({size_mb:.1f} MB, {total} ops): {sorted(op_counts.keys())}")


# ── 1. Depthwise conv2d with many groups (the linear attention conv_stage) ──
print("1. test_dw_conv (depthwise conv2d, groups=1536)")
class DWConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Mimics Qwen35LinearConvStage: depthwise conv over concatenated QKV
        self.conv = nn.Conv2d(
            in_channels=CONV_DIM,
            out_channels=CONV_DIM,
            kernel_size=(1, CONV_KERNEL),
            groups=CONV_DIM,  # fully depthwise
            bias=True,
        )
    def forward(self, x):
        # x: (1, SEQ, CONV_DIM) -> reshape for conv
        B, S, C = x.shape
        # [B, C, 1, S]
        x4d = x.permute(0, 2, 1).unsqueeze(2)
        # Pad left for causal conv
        x_padded = F.pad(x4d, (CONV_KERNEL - 1, 0))
        out = self.conv(x_padded)
        return F.silu(out.squeeze(2).permute(0, 2, 1))

m1 = ct.convert(
    torch.jit.trace(DWConvModel().eval(), torch.randn(1, SEQ, CONV_DIM)),
    inputs=[ct.TensorType(shape=(1, SEQ, CONV_DIM), dtype=np.float16)],
    compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.iOS18,
)
save_model(m1, "test_dw_conv")


# ── 2. softplus activation ──
print("2. test_softplus")
class SoftplusModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.A_log = nn.Parameter(torch.randn(NUM_KV_HEADS, KEY_DIM))
        self.dt_bias = nn.Parameter(torch.randn(NUM_KV_HEADS, KEY_DIM))
        self.proj = nn.Linear(HIDDEN, NUM_KV_HEADS * KEY_DIM, bias=False)
    def forward(self, x):
        a = self.proj(x)  # (1, S, num_kv_heads * key_dim)
        a = a.view(1, SEQ, NUM_KV_HEADS, KEY_DIM)
        g = -self.A_log.exp() * F.softplus(a + self.dt_bias)
        return g.sum(dim=-1)  # (1, S, NUM_KV_HEADS)

m2 = ct.convert(
    torch.jit.trace(SoftplusModel().eval(), torch.randn(1, SEQ, HIDDEN)),
    inputs=[ct.TensorType(shape=(1, SEQ, HIDDEN), dtype=np.float16)],
    compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.iOS18,
)
save_model(m2, "test_softplus")


# ── 3. repeat_interleave (tile pattern) ──
print("3. test_repeat_interleave")
class RepeatModel(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # x: (1, SEQ, NUM_KV_HEADS, KEY_DIM) -> repeat to match NUM_HEADS
        rep = NUM_HEADS // NUM_KV_HEADS  # 32 // 4 = 8
        return x.repeat_interleave(rep, dim=2)

m3 = ct.convert(
    torch.jit.trace(RepeatModel().eval(), torch.randn(1, SEQ, NUM_KV_HEADS, KEY_DIM)),
    inputs=[ct.TensorType(shape=(1, SEQ, NUM_KV_HEADS, KEY_DIM), dtype=np.float16)],
    compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.iOS18,
)
save_model(m3, "test_repeat_interleave")


# ── 4. tril matmul (cumsum replacement) ──
print("4. test_tril_matmul")
class TrilMatmulModel(nn.Module):
    def __init__(self):
        super().__init__()
        CHUNK = 64
        self.register_buffer("tril_ones", torch.tril(torch.ones(CHUNK, CHUNK)))
    def forward(self, g):
        # g: (1, NUM_KV_HEADS, 8, 64) - chunk of decay values (simplified)
        return (self.tril_ones @ g.unsqueeze(-1)).squeeze(-1)

m4 = ct.convert(
    torch.jit.trace(TrilMatmulModel().eval(), torch.randn(1, NUM_KV_HEADS, 8, 64)),
    inputs=[ct.TensorType(shape=(1, NUM_KV_HEADS, 8, 64), dtype=np.float16)],
    compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.iOS18,
)
save_model(m4, "test_tril_matmul")


# ── 5. Combined linear attention layer (conv + softplus + no state) ──
print("5. test_linear_attn_combined")
class LinearAttnCombined(nn.Module):
    def __init__(self):
        super().__init__()
        total_kv = NUM_KV_HEADS * (2 * KEY_DIM + VALUE_DIM)
        self.qkv_proj = nn.Linear(HIDDEN, total_kv, bias=False)
        self.conv = nn.Conv2d(total_kv, total_kv, kernel_size=(1, CONV_KERNEL), groups=total_kv, bias=True)
        self.A_log = nn.Parameter(torch.randn(NUM_KV_HEADS, KEY_DIM))
        self.dt_bias = nn.Parameter(torch.randn(NUM_KV_HEADS, KEY_DIM))
        self.a_proj = nn.Linear(HIDDEN, NUM_KV_HEADS * KEY_DIM, bias=False)
        self.b_proj = nn.Linear(HIDDEN, NUM_KV_HEADS * KEY_DIM, bias=False)
        self.out_proj = nn.Linear(NUM_KV_HEADS * VALUE_DIM, HIDDEN, bias=False)
    def forward(self, x):
        B, S, _ = x.shape
        qkv = self.qkv_proj(x)
        qkv_4d = qkv.permute(0, 2, 1).unsqueeze(2)
        qkv_padded = F.pad(qkv_4d, (CONV_KERNEL - 1, 0))
        conv_out = F.silu(self.conv(qkv_padded).squeeze(2).permute(0, 2, 1))
        key, key2, value = conv_out.split([NUM_KV_HEADS * KEY_DIM, NUM_KV_HEADS * KEY_DIM, NUM_KV_HEADS * VALUE_DIM], dim=-1)
        a = self.a_proj(x).view(B, S, NUM_KV_HEADS, KEY_DIM)
        g = -self.A_log.exp() * F.softplus(a + self.dt_bias)
        # Compute beta
        beta = self.b_proj(x).view(B, S, NUM_KV_HEADS, KEY_DIM).sigmoid()
        # Simple output (skip full chunked recurrence for testing)
        value = value.view(B, S, NUM_KV_HEADS, VALUE_DIM)
        out = value.reshape(B, S, NUM_KV_HEADS * VALUE_DIM)
        return self.out_proj(out)

m5 = ct.convert(
    torch.jit.trace(LinearAttnCombined().eval(), torch.randn(1, SEQ, HIDDEN)),
    inputs=[ct.TensorType(shape=(1, SEQ, HIDDEN), dtype=np.float16)],
    compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.iOS18,
)
save_model(m5, "test_linear_attn_combined")


# ── 6. Full mixed-attention layer: 1 full_attn + 1 linear_attn (no states) ──
print("6. test_mixed_attn_1layer")
class MixedAttnLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Full attention layer
        self.ln1 = nn.LayerNorm(HIDDEN)
        self.full_qkv = nn.Linear(HIDDEN, 3 * HIDDEN, bias=False)
        self.full_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.ffn1_1 = nn.Linear(HIDDEN, 6912, bias=False)
        self.ffn2_1 = nn.Linear(6912, HIDDEN, bias=False)
        self.ln1_2 = nn.LayerNorm(HIDDEN)
        # Linear attention layer
        total_kv = NUM_KV_HEADS * (2 * KEY_DIM + VALUE_DIM)
        self.ln2 = nn.LayerNorm(HIDDEN)
        self.lin_qkv = nn.Linear(HIDDEN, total_kv, bias=False)
        self.lin_conv = nn.Conv2d(total_kv, total_kv, kernel_size=(1, CONV_KERNEL), groups=total_kv, bias=True)
        self.A_log = nn.Parameter(torch.randn(NUM_KV_HEADS, KEY_DIM))
        self.dt_bias = nn.Parameter(torch.randn(NUM_KV_HEADS, KEY_DIM))
        self.a_proj = nn.Linear(HIDDEN, NUM_KV_HEADS * KEY_DIM, bias=False)
        self.b_proj = nn.Linear(HIDDEN, NUM_KV_HEADS * KEY_DIM, bias=False)
        self.lin_out = nn.Linear(NUM_KV_HEADS * VALUE_DIM, HIDDEN, bias=False)
        self.ffn1_2 = nn.Linear(HIDDEN, 6912, bias=False)
        self.ffn2_2 = nn.Linear(6912, HIDDEN, bias=False)
        self.ln2_2 = nn.LayerNorm(HIDDEN)

    def forward(self, x):
        B, S, _ = x.shape
        # === Full attention layer ===
        h = self.ln1(x)
        qkv = self.full_qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(B, S, NUM_HEADS, HEAD_DIM).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (HEAD_DIM ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, HIDDEN)
        x = x + self.full_proj(out)
        h2 = self.ln1_2(x)
        x = x + self.ffn2_1(F.silu(self.ffn1_1(h2)))
        # === Linear attention layer ===
        h = self.ln2(x)
        qkv = self.lin_qkv(h)
        qkv_4d = qkv.permute(0, 2, 1).unsqueeze(2)
        qkv_padded = F.pad(qkv_4d, (CONV_KERNEL - 1, 0))
        conv_out = F.silu(self.lin_conv(qkv_padded).squeeze(2).permute(0, 2, 1))
        key, key2, value = conv_out.split([NUM_KV_HEADS * KEY_DIM, NUM_KV_HEADS * KEY_DIM, NUM_KV_HEADS * VALUE_DIM], dim=-1)
        a = self.a_proj(h).view(B, S, NUM_KV_HEADS, KEY_DIM)
        g = -self.A_log.exp() * F.softplus(a + self.dt_bias)
        beta = self.b_proj(h).view(B, S, NUM_KV_HEADS, KEY_DIM).sigmoid()
        value = value.view(B, S, NUM_KV_HEADS, VALUE_DIM)
        out = value.reshape(B, S, NUM_KV_HEADS * VALUE_DIM)
        x = x + self.lin_out(out)
        h2 = self.ln2_2(x)
        x = x + self.ffn2_2(F.silu(self.ffn1_2(h2)))
        return x

m6 = ct.convert(
    torch.jit.trace(MixedAttnLayer().eval(), torch.randn(1, SEQ, HIDDEN)),
    inputs=[ct.TensorType(shape=(1, SEQ, HIDDEN), dtype=np.float16)],
    compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.iOS18,
)
save_model(m6, "test_mixed_attn_1layer", apply_lut6=True)


print("\n=== Done ===")
for f in sorted(os.listdir(OUT_DIR)):
    if f.startswith("test_") and "real" not in f and "baseline" not in f and "less" not in f and "one_hot" not in f and "gather" not in f and f.endswith('.mlpackage'):
        size_mb = sum(
            os.path.getsize(os.path.join(dp, fn))
            for dp, _, fns in os.walk(os.path.join(OUT_DIR, f)) for fn in fns
        ) / 1024 / 1024
        print(f"  {f}  ({size_mb:.1f} MB)")
