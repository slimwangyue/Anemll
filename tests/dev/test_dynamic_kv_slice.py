#!/usr/bin/env python3
"""Proof-of-concept: dynamic KV cache indexing via shape-derived bounds.

The HuggingFace swift-transformers Mistral export uses a clever trick:
  - causal_mask has shape (1, 1, q_len, end_step) with RangeDim on end_step
  - end = causal_mask.shape[-1]  → symbolic shape query (NOT .item() !)
  - begin = end - q_len          → arithmetic on symbolic shapes
  - cache[:, begin:end, :] = new_kv  → dynamic slice with shape-derived bounds

Key difference from our approach:
  - We use int(current_pos.item()) → aten::Int → freezes at trace time
  - They use tensor.shape[dim] → aten::size → coremltools makes it dynamic via RangeDim

This test verifies whether shape-derived slice_by_index works on ANE.
"""
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
import time

CTX = 1024
BATCH = 256
HEAD_DIM = 128
NUM_HEADS = 4  # small for testing


class DynamicSliceKVCache(nn.Module):
    """Minimal model: writes KV at shape-derived position, reads back."""

    def __init__(self):
        super().__init__()
        # KV cache as buffer (will become CoreML State)
        self.register_buffer(
            "k_cache", torch.zeros(NUM_HEADS, CTX, HEAD_DIM, dtype=torch.float16)
        )
        self.register_buffer(
            "v_cache", torch.zeros(NUM_HEADS, CTX, HEAD_DIM, dtype=torch.float16)
        )

    def forward(self, key_states, value_states, causal_mask):
        """
        key_states:   (NUM_HEADS, BATCH, HEAD_DIM)
        value_states:  (NUM_HEADS, BATCH, HEAD_DIM)
        causal_mask:   (1, 1, BATCH, end_step) — end_step varies via RangeDim!
        """
        # Derive write position from mask SHAPE — not from a value tensor
        end = causal_mask.shape[-1]  # aten::size → stays symbolic with RangeDim
        begin = end - key_states.shape[1]  # aten::size → symbolic batch dim

        # HF-style: in-place slice update on state buffer (NOT clone→reassign)
        self.k_cache[:, begin:end, :] = key_states
        self.v_cache[:, begin:end, :] = value_states

        # Read valid portion for attention (also dynamic shape)
        k_valid = self.k_cache[:, :end, :].clone()
        v_valid = self.v_cache[:, :end, :].clone()

        return k_valid, v_valid


class StaticSliceKVCache(nn.Module):
    """Control: uses int(current_pos.item()) — freezes at trace time."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "k_cache", torch.zeros(NUM_HEADS, CTX, HEAD_DIM, dtype=torch.float16)
        )
        self.register_buffer(
            "v_cache", torch.zeros(NUM_HEADS, CTX, HEAD_DIM, dtype=torch.float16)
        )

    def forward(self, key_states, value_states, current_pos):
        start = int(current_pos.item())  # FREEZES at trace time
        seq_len = key_states.shape[1]

        k = self.k_cache.clone()
        v = self.v_cache.clone()
        k[:, start:start + seq_len, :] = key_states
        v[:, start:start + seq_len, :] = value_states
        self.k_cache = k
        self.v_cache = v

        return self.k_cache, self.v_cache


def test_dynamic():
    """Test shape-derived dynamic slicing with RangeDim."""
    print("=" * 70)
    print("TEST 1: Dynamic slice via shape-derived bounds (HF-style)")
    print("=" * 70)

    model = DynamicSliceKVCache().eval()

    # Trace with: end_step=256 (block 0), batch=256
    key = torch.randn(NUM_HEADS, BATCH, HEAD_DIM, dtype=torch.float16)
    val = torch.randn(NUM_HEADS, BATCH, HEAD_DIM, dtype=torch.float16)
    mask = torch.zeros(1, 1, BATCH, BATCH, dtype=torch.float16)  # end_step=256 at trace

    traced = torch.jit.trace(model, (key, val, mask), check_trace=False)

    # Convert with RangeDim on mask's last dimension
    end_step_dim = ct.RangeDim(lower_bound=BATCH, upper_bound=CTX, default=BATCH)

    try:
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="key_states", shape=key.shape, dtype=np.float16),
                ct.TensorType(name="value_states", shape=val.shape, dtype=np.float16),
                ct.TensorType(name="causal_mask", shape=(1, 1, BATCH, end_step_dim), dtype=np.float16),
            ],
            outputs=[
                ct.TensorType(name="k_valid", dtype=np.float16),
                ct.TensorType(name="v_valid", dtype=np.float16),
            ],
            states=[
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=(NUM_HEADS, CTX, HEAD_DIM), dtype=np.float16),
                    name="k_cache",
                ),
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=(NUM_HEADS, CTX, HEAD_DIM), dtype=np.float16),
                    name="v_cache",
                ),
            ],
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            convert_to="mlprogram",
        )
        print("  ✓ CoreML conversion SUCCESS")
    except Exception as e:
        print(f"  ✗ CoreML conversion FAILED: {e}")
        return False

    # Save and try to load on ANE
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "dynamic_slice.mlpackage")
        mlmodel.save(path)
        print(f"  Saved to {path}")

        try:
            loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            state = loaded.make_state()
            print("  ✓ Model loaded on ANE, state created")
        except Exception as e:
            print(f"  ✗ ANE load FAILED: {e}")
            return False

        # Test block 0: mask shape (1,1,256,256) → write at [0:256]
        k_in = np.random.randn(NUM_HEADS, BATCH, HEAD_DIM).astype(np.float16)
        v_in = np.random.randn(NUM_HEADS, BATCH, HEAD_DIM).astype(np.float16)
        mask_b0 = np.zeros((1, 1, BATCH, BATCH), dtype=np.float16)  # end=256

        try:
            out = loaded.predict(
                {"key_states": k_in, "value_states": v_in, "causal_mask": mask_b0},
                state=state,
            )
            k_out = out["k_valid"]
            print(f"  ✓ Block 0 predict OK — k_valid shape: {k_out.shape}")
            # Verify: k_valid should be shape (NUM_HEADS, 256, HEAD_DIM) and match k_in
            if k_out.shape[1] == BATCH:
                match = np.allclose(k_out, k_in, atol=1e-2)
                print(f"    k_valid matches input: {match}")
            else:
                print(f"    WARNING: unexpected shape {k_out.shape}")
        except Exception as e:
            print(f"  ✗ Block 0 predict FAILED: {e}")
            return False

        # Test block 1: mask shape (1,1,256,512) → write at [256:512]
        mask_b1 = np.zeros((1, 1, BATCH, BATCH * 2), dtype=np.float16)  # end=512
        k_in2 = np.random.randn(NUM_HEADS, BATCH, HEAD_DIM).astype(np.float16)
        v_in2 = np.random.randn(NUM_HEADS, BATCH, HEAD_DIM).astype(np.float16)

        try:
            out2 = loaded.predict(
                {"key_states": k_in2, "value_states": v_in2, "causal_mask": mask_b1},
                state=state,
            )
            k_out2 = out2["k_valid"]
            print(f"  ✓ Block 1 predict OK — k_valid shape: {k_out2.shape}")
            if k_out2.shape[1] == BATCH * 2:
                # First 256 should be from block 0, next 256 from block 1
                first_half_match = np.allclose(k_out2[:, :BATCH, :], k_in, atol=1e-2)
                second_half_match = np.allclose(k_out2[:, BATCH:, :], k_in2, atol=1e-2)
                print(f"    Block 0 KV preserved: {first_half_match}")
                print(f"    Block 1 KV written:   {second_half_match}")
                if first_half_match and second_half_match:
                    print("    ✓ DYNAMIC SLICING WORKS ON ANE!")
                    return True
                else:
                    print("    ✗ Data mismatch — slicing may not be position-correct")
                    return False
            else:
                print(f"    WARNING: unexpected shape {k_out2.shape}")
        except Exception as e:
            print(f"  ✗ Block 1 predict FAILED: {e}")
            return False

    return False


def test_static():
    """Control test: static slice via .item() — what we currently do."""
    print("\n" + "=" * 70)
    print("TEST 2: Static slice via int(current_pos.item()) — our current approach")
    print("=" * 70)

    model = StaticSliceKVCache().eval()

    key = torch.randn(NUM_HEADS, BATCH, HEAD_DIM, dtype=torch.float16)
    val = torch.randn(NUM_HEADS, BATCH, HEAD_DIM, dtype=torch.float16)

    # Trace with current_pos=0 → slice [0:256] frozen
    pos = torch.zeros(1, dtype=torch.int32)
    traced = torch.jit.trace(model, (key, val, pos), check_trace=False)

    try:
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="key_states", shape=key.shape, dtype=np.float16),
                ct.TensorType(name="value_states", shape=val.shape, dtype=np.float16),
                ct.TensorType(name="current_pos", shape=pos.shape, dtype=np.int32),
            ],
            outputs=[
                ct.TensorType(name="k_cache", dtype=np.float16),
                ct.TensorType(name="v_cache", dtype=np.float16),
            ],
            states=[
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=(NUM_HEADS, CTX, HEAD_DIM), dtype=np.float16),
                    name="k_cache",
                ),
                ct.StateType(
                    wrapped_type=ct.TensorType(shape=(NUM_HEADS, CTX, HEAD_DIM), dtype=np.float16),
                    name="v_cache",
                ),
            ],
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            convert_to="mlprogram",
        )
        print("  ✓ Conversion OK (but slice bounds frozen at trace-time values)")
        print("  NOTE: current_pos=256 would STILL write at [0:256] because int() froze it")
    except Exception as e:
        print(f"  ✗ Conversion failed: {e}")

    return True


if __name__ == "__main__":
    print("Dynamic vs Static KV Cache Slice Indexing — ANE PoC")
    print(f"Config: CTX={CTX}, BATCH={BATCH}, HEADS={NUM_HEADS}, HEAD_DIM={HEAD_DIM}")
    print()

    dynamic_ok = test_dynamic()
    test_static()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if dynamic_ok:
        print("  ✓ DYNAMIC shape-derived slicing WORKS on ANE!")
        print("  → We can use ONE prefill model for ALL block positions")
        print("  → Replace 16 models with 1 model + variable-shape mask")
    else:
        print("  ✗ Dynamic slicing failed — keep multi-model approach")
