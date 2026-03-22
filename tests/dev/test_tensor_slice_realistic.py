#!/usr/bin/env python3
"""Stress test: tensor-value slice on a realistic-size model with ANE.

This adds Conv2d layers (like real transformer) to see if the ANE
execution plan builder still succeeds with ~3000+ ops.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
import tempfile, os

CTX = 1024
HEAD_DIM = 128
NUM_KV_HEADS = 4  # GQA heads
NUM_Q_HEADS = 20  # query heads
HIDDEN = 2560
NUM_LAYERS = 8  # layers per chunk
# Layers 3 and 7 are full-attention (with KV cache)


class FakeRMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float16))
        self.eps = 1e-6

    def forward(self, x):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(torch.float16)


class FakeFFN(nn.Module):
    def __init__(self, dim, ff_dim):
        super().__init__()
        self.gate = nn.Conv2d(dim, ff_dim, 1, bias=False)
        self.up = nn.Conv2d(dim, ff_dim, 1, bias=False)
        self.down = nn.Conv2d(ff_dim, dim, 1, bias=False)
        for m in [self.gate, self.up, self.down]:
            m.weight.data = m.weight.data.half()

    def forward(self, x):
        # x: (1, 1, dim) -> reshape for Conv2d
        b, s, d = x.shape
        x4d = x.permute(0, 2, 1).unsqueeze(-1)  # (1, dim, 1, 1)
        g = F.silu(self.gate(x4d))
        u = self.up(x4d)
        out = self.down(g * u)
        return out.squeeze(-1).permute(0, 2, 1)  # back to (1, 1, dim)


class RealisticChunk(nn.Module):
    """8-layer chunk with 2 full-attention layers using tensor-value KV slice."""

    def __init__(self):
        super().__init__()
        # KV cache — 4D state, one block per chunk
        self.register_buffer(
            "k_cache",
            torch.zeros(NUM_LAYERS, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16),
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(NUM_LAYERS, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16),
        )

        # 8 layer norms + FFNs
        self.norms = nn.ModuleList([FakeRMSNorm(HIDDEN) for _ in range(NUM_LAYERS)])
        self.ffns = nn.ModuleList([FakeFFN(HIDDEN, HIDDEN * 2) for _ in range(NUM_LAYERS)])

        # QKV projections for full-attention layers (3, 7)
        self.q_proj_3 = nn.Conv2d(HIDDEN, NUM_Q_HEADS * HEAD_DIM, 1, bias=False)
        self.k_proj_3 = nn.Conv2d(HIDDEN, NUM_KV_HEADS * HEAD_DIM, 1, bias=False)
        self.v_proj_3 = nn.Conv2d(HIDDEN, NUM_KV_HEADS * HEAD_DIM, 1, bias=False)
        self.o_proj_3 = nn.Conv2d(NUM_Q_HEADS * HEAD_DIM, HIDDEN, 1, bias=False)

        self.q_proj_7 = nn.Conv2d(HIDDEN, NUM_Q_HEADS * HEAD_DIM, 1, bias=False)
        self.k_proj_7 = nn.Conv2d(HIDDEN, NUM_KV_HEADS * HEAD_DIM, 1, bias=False)
        self.v_proj_7 = nn.Conv2d(HIDDEN, NUM_KV_HEADS * HEAD_DIM, 1, bias=False)
        self.o_proj_7 = nn.Conv2d(NUM_Q_HEADS * HEAD_DIM, HIDDEN, 1, bias=False)

        # Half precision
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data = m.weight.data.half()

    def _attn_layer(self, hidden, current_pos, causal_mask, layer_idx, q_proj, k_proj, v_proj, o_proj):
        """Full attention with KV cache write using tensor-value slice."""
        b, s, d = hidden.shape
        x4d = hidden.permute(0, 2, 1).unsqueeze(-1)  # (1, dim, 1, 1)

        q = q_proj(x4d).squeeze(-1).permute(0, 2, 1)  # (1, 1, q_heads*head_dim)
        k = k_proj(x4d).squeeze(-1).permute(0, 2, 1)  # (1, 1, kv_heads*head_dim)
        v = v_proj(x4d).squeeze(-1).permute(0, 2, 1)

        # Reshape
        q = q.view(1, 1, NUM_Q_HEADS, HEAD_DIM).permute(0, 2, 1, 3)   # (1, q_heads, 1, head_dim)
        k = k.view(1, 1, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)  # (1, kv_heads, 1, head_dim)
        v = v.view(1, 1, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)

        # === THE KEY: tensor-value slice, no .item(), no RangeDim ===
        pos = current_pos[0]  # aten::select — stays as tensor
        self.k_cache[layer_idx, :, pos:pos + 1, :] = k.squeeze(0)
        self.v_cache[layer_idx, :, pos:pos + 1, :] = v.squeeze(0)

        # Read full cache for attention
        k_full = self.k_cache[layer_idx:layer_idx + 1].squeeze(0)  # (kv_heads, CTX, head_dim)
        v_full = self.v_cache[layer_idx:layer_idx + 1].squeeze(0)

        # GQA expand
        rep = NUM_Q_HEADS // NUM_KV_HEADS
        k_exp = k_full.unsqueeze(0).repeat(1, rep, 1, 1)  # (1, q_heads, CTX, head_dim)
        v_exp = v_full.unsqueeze(0).repeat(1, rep, 1, 1)

        # Attention
        scale = HEAD_DIM ** -0.5
        attn = torch.matmul(q, k_exp.transpose(-2, -1)) * scale  # (1, q_heads, 1, CTX)
        attn = attn + causal_mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v_exp)  # (1, q_heads, 1, head_dim)

        out = out.permute(0, 2, 1, 3).reshape(1, 1, NUM_Q_HEADS * HEAD_DIM)
        out4d = out.permute(0, 2, 1).unsqueeze(-1)
        out = o_proj(out4d).squeeze(-1).permute(0, 2, 1)
        return out

    def forward(self, hidden_states, current_pos, causal_mask):
        """
        hidden_states: (1, 1, HIDDEN)
        current_pos: (1,) — tensor, NOT scalar
        causal_mask: (1, 1, 1, CTX)
        """
        for i in range(NUM_LAYERS):
            normed = self.norms[i](hidden_states)

            if i == 3:
                attn_out = self._attn_layer(
                    normed, current_pos, causal_mask, 3,
                    self.q_proj_3, self.k_proj_3, self.v_proj_3, self.o_proj_3,
                )
                hidden_states = hidden_states + attn_out
            elif i == 7:
                attn_out = self._attn_layer(
                    normed, current_pos, causal_mask, 7,
                    self.q_proj_7, self.k_proj_7, self.v_proj_7, self.o_proj_7,
                )
                hidden_states = hidden_states + attn_out
            else:
                # Linear attention placeholder — just FFN
                pass

            hidden_states = hidden_states + self.ffns[i](normed)

        return hidden_states


def main():
    print("=" * 60)
    print("REALISTIC CHUNK: 8 layers, tensor-value KV slice, ANE test")
    print("=" * 60)

    model = RealisticChunk().eval()
    hidden = torch.randn(1, 1, HIDDEN, dtype=torch.float16)
    pos = torch.tensor([0], dtype=torch.int32)
    mask = torch.zeros(1, 1, 1, CTX, dtype=torch.float16)

    print("Tracing...")
    traced = torch.jit.trace(model, (hidden, pos, mask), check_trace=False)
    print("  OK")

    print("Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=(1, 1, HIDDEN), dtype=np.float16),
            ct.TensorType(name="current_pos", shape=(1,), dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=(1, 1, 1, CTX), dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="output", dtype=np.float16),
        ],
        states=[
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(NUM_LAYERS, NUM_KV_HEADS, CTX, HEAD_DIM), dtype=np.float16
                ),
                name="k_cache",
            ),
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(NUM_LAYERS, NUM_KV_HEADS, CTX, HEAD_DIM), dtype=np.float16
                ),
                name="v_cache",
            ),
        ],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        convert_to="mlprogram",
    )
    print("  OK")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "realistic_chunk.mlpackage")
        mlmodel.save(path)

        print("Loading on ANE...")
        try:
            loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            state = loaded.make_state()
            print("  ANE load: OK")
        except Exception as e:
            print(f"  ANE load: FAILED — {e}")
            return

        h = np.random.randn(1, 1, HIDDEN).astype(np.float16)
        mask_np = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)

        for test_pos in [0, 1, 5, 50, 100, 500, 1023]:
            mask_np[:, :, :, :test_pos + 1] = 0
            try:
                out = loaded.predict(
                    {
                        "hidden_states": h,
                        "current_pos": np.array([test_pos], dtype=np.int32),
                        "causal_mask": mask_np,
                    },
                    state=state,
                )
                print(f"  ANE predict pos={test_pos}: OK — shape={out['output'].shape}")
            except Exception as e:
                print(f"  ANE predict pos={test_pos}: FAILED — {str(e)[:120]}")
                return

    print("\n  ALL PASSED on ANE!")


if __name__ == "__main__":
    main()
