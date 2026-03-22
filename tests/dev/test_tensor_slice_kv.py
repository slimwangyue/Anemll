#!/usr/bin/env python3
"""PoC: Can a (1,) tensor value be used for dynamic slicing on ANE?

Three approaches for KV cache write position:
  A) int(current_pos.item()) → aten::Int → FREEZES at trace time
  B) kv_write_end.shape[0]  → aten::size → symbolic via RangeDim (works but error -14 on full model)
  C) current_pos[0]         → aten::select → tensor value, stays dynamic?

This test checks approach C: pass current_pos as (1,) tensor, use pos = current_pos[0]
for slicing without calling .item().
"""
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct
import tempfile, os

CTX = 1024
HEAD_DIM = 128
NUM_KV_HEADS = 8
NUM_LAYERS = 8  # same as real model chunk


class KVCache_TensorSlice(nn.Module):
    """KV cache using tensor value (not .item()) for slice bounds."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "k_cache",
            torch.zeros(NUM_LAYERS, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16),
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(NUM_LAYERS, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16),
        )

    def forward(self, key_states, value_states, current_pos):
        """
        key_states:   (1, NUM_KV_HEADS, 1, HEAD_DIM)
        value_states:  (1, NUM_KV_HEADS, 1, HEAD_DIM)
        current_pos:   (1,) tensor — NOT scalar, NOT .item()
        """
        # Use tensor value directly — no .item()!
        pos = current_pos[0]  # aten::select → still a tensor
        # Slice using tensor-valued bounds
        self.k_cache[0, :, pos:pos + 1, :] = key_states.squeeze(0)
        self.v_cache[0, :, pos:pos + 1, :] = value_states.squeeze(0)

        # Read back
        k = self.k_cache[0:1].squeeze(0)
        v = self.v_cache[0:1].squeeze(0)
        return k, v


class KVCache_TensorSlice_MultiLayer(nn.Module):
    """Same but writes to multiple layers (closer to real model)."""

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "k_cache",
            torch.zeros(NUM_LAYERS, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16),
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(NUM_LAYERS, NUM_KV_HEADS, CTX, HEAD_DIM, dtype=torch.float16),
        )

    def forward(self, key_states, value_states, current_pos):
        pos = current_pos[0]  # tensor value

        # Write to layer 3 (like full_attn at local_idx=3)
        self.k_cache[3, :, pos:pos + 1, :] = key_states.squeeze(0)
        self.v_cache[3, :, pos:pos + 1, :] = value_states.squeeze(0)

        # Write to layer 7 (like full_attn at local_idx=7)
        self.k_cache[7, :, pos:pos + 1, :] = key_states.squeeze(0)
        self.v_cache[7, :, pos:pos + 1, :] = value_states.squeeze(0)

        k3 = self.k_cache[3:4].squeeze(0)
        v3 = self.v_cache[3:4].squeeze(0)
        k7 = self.k_cache[7:8].squeeze(0)
        v7 = self.v_cache[7:8].squeeze(0)
        return k3, v3, k7, v7


def test_model(model_class, name, num_outputs=2):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")

    model = model_class().eval()
    k = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16)
    v = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=torch.float16)
    pos = torch.tensor([0], dtype=torch.int32)

    # Trace
    try:
        traced = torch.jit.trace(model, (k, v, pos), check_trace=False)
        print("  Trace: OK")
    except Exception as e:
        print(f"  Trace: FAILED — {e}")
        return False

    # Convert to CoreML
    outputs = []
    if num_outputs == 2:
        outputs = [
            ct.TensorType(name="k_out", dtype=np.float16),
            ct.TensorType(name="v_out", dtype=np.float16),
        ]
    else:
        for i in range(num_outputs):
            outputs.append(ct.TensorType(name=f"out_{i}", dtype=np.float16))

    try:
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="key_states", shape=k.shape, dtype=np.float16),
                ct.TensorType(name="value_states", shape=v.shape, dtype=np.float16),
                ct.TensorType(name="current_pos", shape=(1,), dtype=np.int32),
            ],
            outputs=outputs,
            states=[
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(NUM_LAYERS, NUM_KV_HEADS, CTX, HEAD_DIM),
                        dtype=np.float16,
                    ),
                    name="k_cache",
                ),
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(NUM_LAYERS, NUM_KV_HEADS, CTX, HEAD_DIM),
                        dtype=np.float16,
                    ),
                    name="v_cache",
                ),
            ],
            minimum_deployment_target=ct.target.iOS18,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            convert_to="mlprogram",
        )
        print("  Convert: OK")
    except Exception as e:
        print(f"  Convert: FAILED — {e}")
        return False

    # Save, load on ANE, test predict
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.mlpackage")
        mlmodel.save(path)

        try:
            loaded = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
            state = loaded.make_state()
            print("  ANE load + make_state: OK")
        except Exception as e:
            print(f"  ANE load: FAILED — {e}")
            return False

        k_np = np.random.randn(1, NUM_KV_HEADS, 1, HEAD_DIM).astype(np.float16)
        v_np = np.random.randn(1, NUM_KV_HEADS, 1, HEAD_DIM).astype(np.float16)

        for test_pos in [0, 1, 5, 50, 100, 500, 1023]:
            pos_np = np.array([test_pos], dtype=np.int32)
            try:
                out = loaded.predict(
                    {"key_states": k_np, "value_states": v_np, "current_pos": pos_np},
                    state=state,
                )
                print(f"  ANE predict pos={test_pos}: OK")
            except Exception as e:
                print(f"  ANE predict pos={test_pos}: FAILED — {str(e)[:100]}")
                return False

    print("  ALL PASSED!")
    return True


if __name__ == "__main__":
    r1 = test_model(KVCache_TensorSlice, "Single-layer tensor slice", num_outputs=2)
    r2 = test_model(KVCache_TensorSlice_MultiLayer, "Multi-layer tensor slice", num_outputs=4)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Single-layer: {'PASS' if r1 else 'FAIL'}")
    print(f"  Multi-layer:  {'PASS' if r2 else 'FAIL'}")
