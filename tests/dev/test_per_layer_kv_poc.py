#!/usr/bin/env python3
"""PoC: Compare 4D (multi-layer) vs 3D (per-layer) KV state with dynamic slice on ANE.

Hypothesis: ANE error -14 comes from combining layer-dim indexing (dim 0) with
dynamic RangeDim slicing (dim 2) on a 4D StateType tensor. If we split to
per-layer 3D states, the simpler dynamic slice might be ANE-legal.
"""
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
import tempfile, os, time

CTX = 1024
HEAD_DIM = 128
NUM_KV_HEADS = 8
NUM_LAYERS = 8  # layers per chunk — same as real model

# ============================================================
# Model A: ONE 4D state (current approach — fails on ANE)
# ============================================================
class MultiLayerKV_4D(nn.Module):
    """KV cache as single 4D state: (num_layers, kv_heads, CTX, head_dim)."""
    def __init__(self):
        super().__init__()
        self.register_buffer("k_cache", torch.zeros(NUM_LAYERS, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
        self.register_buffer("v_cache", torch.zeros(NUM_LAYERS, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
        # Simple linear to simulate computation
        self.proj = nn.Conv2d(HEAD_DIM, HEAD_DIM, kernel_size=1, bias=False)
        self.proj.weight.data = self.proj.weight.data.half()

    def forward(self, hidden, kv_write_end, layer_idx_tensor):
        """
        hidden: (1, NUM_KV_HEADS, 1, HEAD_DIM)
        kv_write_end: shape = (pos+1,) — RangeDim carrier
        layer_idx_tensor: (1,) — which layer to write to
        """
        end = kv_write_end.shape[0]
        begin = end - 1
        li = int(layer_idx_tensor.item())
        # Write to cache — compound slice: dim0 (static) + dim2 (dynamic)
        kv = hidden.squeeze(0)  # (kv_heads, 1, head_dim)
        self.k_cache[li, :, begin:end, :] = kv
        self.v_cache[li, :, begin:end, :] = kv
        # Read from cache
        k = self.k_cache[li:li+1].squeeze(0)  # (kv_heads, CTX, head_dim)
        v = self.v_cache[li:li+1].squeeze(0)
        return k, v


# ============================================================
# Model B: SEPARATE 3D states per layer (proposed approach)
# ============================================================
class MultiLayerKV_PerLayer(nn.Module):
    """KV cache as separate 3D states per full-attention layer."""
    def __init__(self, num_full_attn_layers=2):
        super().__init__()
        self.num_full_attn = num_full_attn_layers
        for i in range(num_full_attn_layers):
            self.register_buffer(f"k_cache_{i}", torch.zeros(NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
            self.register_buffer(f"v_cache_{i}", torch.zeros(NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16))
        self.proj = nn.Conv2d(HEAD_DIM, HEAD_DIM, kernel_size=1, bias=False)
        self.proj.weight.data = self.proj.weight.data.half()

    def forward(self, hidden, kv_write_end):
        """
        Process both full-attention layers sequentially.
        hidden: (1, NUM_KV_HEADS, 1, HEAD_DIM)
        kv_write_end: shape = (pos+1,) — RangeDim carrier
        """
        end = kv_write_end.shape[0]
        begin = end - 1
        kv = hidden.squeeze(0)  # (kv_heads, 1, head_dim)

        # Layer 0 — direct 3D slice, NO layer dim indexing
        self.k_cache_0[:, begin:end, :] = kv
        self.v_cache_0[:, begin:end, :] = kv
        k0 = self.k_cache_0.clone()
        v0 = self.v_cache_0.clone()

        # Layer 1 — same pattern
        self.k_cache_1[:, begin:end, :] = kv
        self.v_cache_1[:, begin:end, :] = kv
        k1 = self.k_cache_1.clone()
        v1 = self.v_cache_1.clone()

        return k0, v0, k1, v1


def test_4d():
    """Test current approach: 4D state with compound slice."""
    print("=" * 60)
    print("TEST A: 4D state (num_layers, heads, CTX, dim) — CURRENT")
    print("=" * 60)

    model = MultiLayerKV_4D().eval()
    hidden = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16)
    kv_write_end = torch.zeros(1, dtype=torch.int32)
    layer_idx = torch.tensor([0], dtype=torch.int32)

    traced = torch.jit.trace(model, (hidden, kv_write_end, layer_idx), check_trace=False)

    pos_dim = ct.RangeDim(lower_bound=1, upper_bound=CTX, default=1)
    try:
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="hidden", shape=hidden.shape, dtype=np.float16),
                ct.TensorType(name="kv_write_end", shape=(pos_dim,), dtype=np.int32),
                ct.TensorType(name="layer_idx_tensor", shape=(1,), dtype=np.int32),
            ],
            outputs=[
                ct.TensorType(name="k_out", dtype=np.float16),
                ct.TensorType(name="v_out", dtype=np.float16),
            ],
            states=[
                ct.StateType(wrapped_type=ct.TensorType(
                    shape=(NUM_LAYERS, NUM_KV_HEADS, CTX, HEAD_DIM), dtype=np.float16), name="k_cache"),
                ct.StateType(wrapped_type=ct.TensorType(
                    shape=(NUM_LAYERS, NUM_KV_HEADS, CTX, HEAD_DIM), dtype=np.float16), name="v_cache"),
            ],
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            convert_to="mlprogram",
        )
        print("  Convert: OK")
    except Exception as e:
        print(f"  Convert: FAILED — {e}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_4d.mlpackage")
        mlmodel.save(path)
        try:
            loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            state = loaded.make_state()
            print("  ANE load: OK")
        except Exception as e:
            print(f"  ANE load: FAILED — {e}")
            return

        # Test predict at pos=0
        h = np.random.randn(1, NUM_KV_HEADS, 1, HEAD_DIM).astype(np.float16)
        try:
            out = loaded.predict({
                "hidden": h,
                "kv_write_end": np.zeros((1,), dtype=np.int32),
                "layer_idx_tensor": np.array([0], dtype=np.int32),
            }, state=state)
            print("  ANE predict pos=0: OK")
        except Exception as e:
            print(f"  ANE predict pos=0: FAILED — {str(e)[:100]}")
            return

        # Test predict at pos=5
        try:
            out = loaded.predict({
                "hidden": h,
                "kv_write_end": np.zeros((6,), dtype=np.int32),
                "layer_idx_tensor": np.array([3], dtype=np.int32),
            }, state=state)
            print("  ANE predict pos=5: OK")
        except Exception as e:
            print(f"  ANE predict pos=5: FAILED — {str(e)[:100]}")


def test_per_layer():
    """Test proposed approach: separate 3D states per full-attention layer."""
    print("\n" + "=" * 60)
    print("TEST B: Per-layer 3D states (heads, CTX, dim) — PROPOSED")
    print("=" * 60)

    model = MultiLayerKV_PerLayer(num_full_attn_layers=2).eval()
    hidden = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16)
    kv_write_end = torch.zeros(1, dtype=torch.int32)

    traced = torch.jit.trace(model, (hidden, kv_write_end), check_trace=False)

    pos_dim = ct.RangeDim(lower_bound=1, upper_bound=CTX, default=1)
    states = []
    for i in range(2):
        states.append(ct.StateType(wrapped_type=ct.TensorType(
            shape=(NUM_KV_HEADS, CTX, HEAD_DIM), dtype=np.float16), name=f"k_cache_{i}"))
        states.append(ct.StateType(wrapped_type=ct.TensorType(
            shape=(NUM_KV_HEADS, CTX, HEAD_DIM), dtype=np.float16), name=f"v_cache_{i}"))

    try:
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="hidden", shape=hidden.shape, dtype=np.float16),
                ct.TensorType(name="kv_write_end", shape=(pos_dim,), dtype=np.float16),
            ],
            outputs=[
                ct.TensorType(name="k0", dtype=np.float16),
                ct.TensorType(name="v0", dtype=np.float16),
                ct.TensorType(name="k1", dtype=np.float16),
                ct.TensorType(name="v1", dtype=np.float16),
            ],
            states=states,
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            convert_to="mlprogram",
        )
        print("  Convert: OK")
    except Exception as e:
        print(f"  Convert: FAILED — {e}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_per_layer.mlpackage")
        mlmodel.save(path)
        try:
            loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            state = loaded.make_state()
            print("  ANE load: OK")
        except Exception as e:
            print(f"  ANE load: FAILED — {e}")
            return

        h = np.random.randn(1, NUM_KV_HEADS, 1, HEAD_DIM).astype(np.float16)

        for pos in [0, 1, 5, 100, 500]:
            try:
                out = loaded.predict({
                    "hidden": h,
                    "kv_write_end": np.zeros((pos + 1,), dtype=np.float16),
                }, state=state)
                print(f"  ANE predict pos={pos}: OK")
            except Exception as e:
                print(f"  ANE predict pos={pos}: FAILED — {str(e)[:100]}")
                break


if __name__ == "__main__":
    test_4d()
    test_per_layer()
    print("\nDone.")
