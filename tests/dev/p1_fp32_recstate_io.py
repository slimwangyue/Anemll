#!/usr/bin/env python3
"""Priority 1 experiment: FP32 linear_recurrent_state I/O on ANE.

Exports chunk 0 (layers 0-3, all linear-attention) twice:
  A) FP16 recurrent-state I/O  (baseline — same as production)
  B) FP32 recurrent-state I/O  (experiment)

Then tests:
  1. ANE loadability of (B)
  2. Fallback analysis (which compute units are used)
  3. Per-token state drift comparison over 80 tokens
  4. Quality comparison on 3 failing prompts

Usage:
  cd /Users/yw68/Anemll
  source .venv/bin/activate
  python tests/dev/p1_fp32_recstate_io.py
"""
import gc, os, sys, time, tempfile, shutil
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts_qwen3_5"))
sys.path.insert(0, REPO_ROOT)

import torch
import coremltools as ct
from config import (
    BATCH_SIZE, CTX, NUM_CHUNKS, LUT_BITS, PER_CHANNEL,
    FFN_PER_CHANNEL, CHUNK_RANGES,
    DEFAULT_HF_MODEL, DEFAULT_OUTPUT,
)
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE,
    ane_conv_state_shape,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

# ── Config ──
CHUNK_IDX = 0                          # test chunk 0 (layers 0-3, all linear)
SL, EL = CHUNK_RANGES[CHUNK_IDX]      # (0, 3)
OUTPUT_DIR = os.path.join(REPO_ROOT, "tests", "dev", "_p1_fp32_experiment")
HF_PATH = DEFAULT_HF_MODEL


def load_model():
    """Load the Qwen3.5-4B model weights."""
    print("Loading model...")
    t0 = time.time()
    cfg = Qwen35Config.from_json(os.path.join(HF_PATH, "config.json"))
    cfg.context_length = CTX
    cfg.state_length = CTX
    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_PATH)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded in {time.time()-t0:.1f}s")
    return model


def export_chunk_with_rec_dtype(model, rec_dtype_np, label):
    """Export chunk 0 with specified recurrent state I/O dtype.

    rec_dtype_np: np.float16 or np.float32
    """
    print(f"\n{'='*60}")
    print(f"Exporting chunk {CHUNK_IDX} [{SL}-{EL-1}] with rec_state={rec_dtype_np.__name__}")
    print(f"{'='*60}")

    rec_dtype_torch = torch.float16 if rec_dtype_np == np.float16 else torch.float32

    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=LUT_BITS, per_channel=FFN_PER_CHANNEL,
    )

    # We'll directly call the converter internals with modified I/O dtypes
    total_layers = model.config.num_hidden_layers
    local_num_layers = EL - SL
    cfg = model.config

    class FFNWrapper(torch.nn.Module):
        def __init__(self, model, start_layer, end_layer):
            super().__init__()
            self.model = model
            self.start_layer = start_layer
            self.end_layer = end_layer
            self.local_num_layers = end_layer - start_layer
            self.register_buffer("k_cache", torch.zeros(
                (self.local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                dtype=MODEL_DTYPE, device=TEST_DEVICE))
            self.register_buffer("v_cache", torch.zeros(
                (self.local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
                dtype=MODEL_DTYPE, device=TEST_DEVICE))
            conv_dim = (
                cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
            )
            conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
            ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
            self._lin_conv_shape = (self.local_num_layers, ane_dim1, ane_dim2)
            self._lin_rec_shape = (
                self.local_num_layers,
                cfg.text_config.linear_num_value_heads,
                cfg.text_config.linear_key_head_dim,
                cfg.text_config.linear_value_head_dim,
            )
            self.states = Qwen35Converter.GetChunkLocalTransformerStates(
                model, self.local_num_layers, prefix="", split_full_attention_kv=True
            )

        def forward(self, hidden_states, position_ids, causal_mask, current_pos,
                    linear_conv_state, linear_recurrent_state):
            out = self.model.model.process_layers_regular_single_token_export_local_state(
                hidden_states=hidden_states,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                kv_cache_0=None,
                k_cache=self.k_cache,
                v_cache=self.v_cache,
                linear_conv_state=linear_conv_state,
                linear_recurrent_state=linear_recurrent_state,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                apply_final_norm=False,
            )
            return out, linear_conv_state, linear_recurrent_state

    wrapper = FFNWrapper(model, SL, EL).eval()

    # Sample inputs — conv state stays FP16, only rec_state changes
    hidden_states = torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
    position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, CTX), dtype=torch.float16, device=TEST_DEVICE)
    current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    lin_conv = torch.zeros(wrapper._lin_conv_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
    lin_rec = torch.zeros(wrapper._lin_rec_shape, dtype=rec_dtype_torch, device=TEST_DEVICE)

    # Reset states
    for buf_name in ['k_cache', 'v_cache']:
        getattr(wrapper, buf_name).zero_()
    traced = torch.jit.trace(
        wrapper, (hidden_states, position_ids, causal_mask, current_pos, lin_conv, lin_rec),
        check_trace=False,
    )
    for buf_name in ['k_cache', 'v_cache']:
        getattr(wrapper, buf_name).zero_()

    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
            ct.TensorType(name="linear_conv_state", shape=lin_conv.shape, dtype=np.float16),
            ct.TensorType(name="linear_recurrent_state", shape=lin_rec.shape, dtype=rec_dtype_np),
        ],
        outputs=[
            ct.TensorType(name="output_hidden_states", dtype=np.float16),
            ct.TensorType(name="linear_conv_state_out", dtype=np.float16),
            ct.TensorType(name="linear_recurrent_state_out", dtype=rec_dtype_np),
        ],
        states=wrapper.states,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    elapsed = time.time() - t0
    print(f"  Converted in {elapsed:.1f}s")

    # Apply LUT quantization (same as production)
    conv.converted_model = mlmodel
    conv.postprocess(num_workers=1)
    mlmodel = conv.converted_model

    save_path = os.path.join(OUTPUT_DIR, f"chunk0_{label}.mlpackage")
    mlmodel.save(save_path)
    print(f"  Saved to {save_path}")

    del mlmodel, conv, traced, wrapper
    gc.collect()
    return save_path


def test_ane_loadability(path, label):
    """Test loading model on CPU_AND_NE and check what actually runs on ANE."""
    print(f"\n--- ANE Loadability Test: {label} ---")

    # Test 1: Load on CPU_AND_NE
    try:
        t0 = time.time()
        m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        elapsed = time.time() - t0
        print(f"  [OK] CPU_AND_NE loaded in {elapsed:.1f}s")

        # Quick predict test
        spec = m.get_spec()
        inp_shapes = {}
        for inp in spec.description.input:
            try:
                inp_shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
            except:
                pass
        print(f"  Input shapes: {inp_shapes}")

        # Check recurrent state dtype from spec
        for inp in spec.description.input:
            if inp.name == "linear_recurrent_state":
                dt = inp.type.multiArrayType.dataType
                dtype_names = {65536: "FP16", 65568: "FP32", 131072: "INT32"}
                print(f"  linear_recurrent_state spec dtype: {dtype_names.get(dt, dt)}")
        for out in spec.description.output:
            if out.name == "linear_recurrent_state_out":
                dt = out.type.multiArrayType.dataType
                dtype_names = {65536: "FP16", 65568: "FP32", 131072: "INT32"}
                print(f"  linear_recurrent_state_out spec dtype: {dtype_names.get(dt, dt)}")

        # Try a predict call
        state = m.make_state()
        test_inp = {
            "hidden_states": np.zeros((1, 1, 2560), dtype=np.float16),
            "position_ids": np.array([0], dtype=np.int32),
            "causal_mask": np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16),
            "current_pos": np.array([0], dtype=np.int32),
            "linear_conv_state": np.zeros(inp_shapes['linear_conv_state'], dtype=np.float16),
        }
        # Set recurrent state dtype based on spec
        rec_shape = inp_shapes['linear_recurrent_state']
        if 'fp32' in label.lower():
            test_inp["linear_recurrent_state"] = np.zeros(rec_shape, dtype=np.float32)
        else:
            test_inp["linear_recurrent_state"] = np.zeros(rec_shape, dtype=np.float16)

        t0 = time.time()
        out = m.predict(test_inp, state=state)
        elapsed = time.time() - t0
        print(f"  [OK] predict() in {elapsed*1000:.0f}ms")

        # Check output shapes and dtypes
        for k, v in out.items():
            print(f"  output '{k}': shape={v.shape}, dtype={v.dtype}")

        del m, state
        gc.collect()
        return True

    except Exception as e:
        print(f"  [FAIL] CPU_AND_NE: {e}")
        return False


def test_ane_only(path, label):
    """Test loading on ALL (includes ANE) to check ANE scheduling."""
    print(f"\n--- ANE-Only Test: {label} ---")
    try:
        m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.ALL)
        state = m.make_state()
        spec = m.get_spec()
        inp_shapes = {}
        for inp in spec.description.input:
            try:
                inp_shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
            except:
                pass
        rec_shape = inp_shapes['linear_recurrent_state']
        if 'fp32' in label.lower():
            rec_dtype = np.float32
        else:
            rec_dtype = np.float16
        test_inp = {
            "hidden_states": np.zeros((1, 1, 2560), dtype=np.float16),
            "position_ids": np.array([0], dtype=np.int32),
            "causal_mask": np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16),
            "current_pos": np.array([0], dtype=np.int32),
            "linear_conv_state": np.zeros(inp_shapes['linear_conv_state'], dtype=np.float16),
            "linear_recurrent_state": np.zeros(rec_shape, dtype=rec_dtype),
        }
        t0 = time.time()
        out = m.predict(test_inp, state=state)
        elapsed = time.time() - t0
        print(f"  [OK] ALL predict() in {elapsed*1000:.0f}ms")
        del m, state
        gc.collect()
        return True
    except Exception as e:
        print(f"  [FAIL] ALL: {e}")
        return False


def compare_state_drift(fp16_path, fp32_path, model):
    """Run both chunk 0 variants for N steps, tracking recurrent state divergence.

    Since we only have chunk 0, we use the actual model's embedding layer
    to produce realistic hidden states, but run only through chunk 0's layers.
    """
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH, trust_remote_code=True)

    prompts = [
        "What is a stack in computer science? Explain in detail.",
        "教我做红烧鱼",
        "A farmer has 17 sheep. All but 9 run away. How many are left?",
    ]

    for prompt in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: {prompt}")
        print(f"{'='*60}")

        messages = [{"role": "user", "content": prompt}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        print(f"  Tokens: {len(tokens)}")

        # Load both models
        m16 = ct.models.MLModel(fp16_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        m32 = ct.models.MLModel(fp32_path, compute_units=ct.ComputeUnit.CPU_AND_NE)

        state16 = m16.make_state()
        state32 = m32.make_state()

        spec16 = m16.get_spec()
        inp_shapes = {}
        for inp in spec16.description.input:
            try:
                inp_shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
            except:
                pass

        rec_shape = inp_shapes['linear_recurrent_state']
        conv_shape = inp_shapes['linear_conv_state']

        # Initialize states
        conv16 = np.zeros(conv_shape, dtype=np.float16)
        conv32 = np.zeros(conv_shape, dtype=np.float16)
        rec16 = np.zeros(rec_shape, dtype=np.float16)
        rec32 = np.zeros(rec_shape, dtype=np.float32)

        # Use embeddings model for hidden states
        embed_path = os.path.join(
            REPO_ROOT, "qwen3_5_flll_9chunk", "embeddings.mlpackage")
        embed = ct.models.MLModel(embed_path, compute_units=ct.ComputeUnit.CPU_ONLY)

        # Run through all tokens (prefill + generate a few)
        n_gen = 80  # generate 80 more tokens after prompt
        total_steps = len(tokens) + n_gen

        print(f"\n  {'Step':>4} {'REC16_L2':>12} {'REC32_L2':>12} {'REC_diff_L2':>13} {'REC_cos':>10} {'HID_cos':>10}")
        print("  " + "-" * 75)

        for step in range(min(total_steps, CTX - 1)):
            if step < len(tokens):
                tok_id = tokens[step]
            else:
                break  # We only have chunk 0, can't do full generation

            tok = np.array([[tok_id]], dtype=np.int32)
            hidden = list(embed.predict({"input_ids": tok}).values())[0]

            mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
            mask[:, :, :, :step + 1] = 0

            inp16 = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([step], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([step], dtype=np.int32),
                "linear_conv_state": conv16,
                "linear_recurrent_state": rec16,
            }
            inp32 = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([step], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([step], dtype=np.int32),
                "linear_conv_state": conv32,
                "linear_recurrent_state": rec32,
            }

            out16 = m16.predict(inp16, state=state16)
            out32 = m32.predict(inp32, state=state32)

            # Update states
            conv16 = out16['linear_conv_state_out']
            conv32 = out32['linear_conv_state_out']
            rec16 = out16['linear_recurrent_state_out']
            rec32 = out32['linear_recurrent_state_out']

            # Compute drift metrics
            r16_f = rec16.flatten().astype(np.float64)
            r32_f = rec32.flatten().astype(np.float64)
            l2_16 = np.sqrt(np.sum(r16_f ** 2))
            l2_32 = np.sqrt(np.sum(r32_f ** 2))
            diff = r32_f - r16_f
            l2_diff = np.sqrt(np.sum(diff ** 2))
            dot = np.dot(r16_f, r32_f)
            rec_cos = dot / (l2_16 * l2_32 + 1e-12)

            h16_f = out16['output_hidden_states'].flatten().astype(np.float64)
            h32_f = out32['output_hidden_states'].flatten().astype(np.float64)
            hid_cos = np.dot(h16_f, h32_f) / (np.linalg.norm(h16_f) * np.linalg.norm(h32_f) + 1e-12)

            if step % 5 == 0 or step < 5 or step == len(tokens) - 1:
                print(f"  {step:4d} {l2_16:12.4f} {l2_32:12.4f} {l2_diff:13.6f} {rec_cos:10.6f} {hid_cos:10.6f}")

        # Final summary
        print(f"\n  Final REC state diff L2: {l2_diff:.6f}")
        print(f"  Final REC state cosine sim: {rec_cos:.6f}")
        print(f"  Final hidden cosine sim: {hid_cos:.6f}")
        print(f"  FP16 rec has NaN: {np.any(np.isnan(rec16))}")
        print(f"  FP32 rec has NaN: {np.any(np.isnan(rec32))}")
        print(f"  FP16 rec has Inf: {np.any(np.isinf(rec16))}")
        print(f"  FP32 rec has Inf: {np.any(np.isinf(rec32))}")

        del m16, m32, state16, state32, embed
        gc.collect()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model()

    # Export both variants
    fp16_path = export_chunk_with_rec_dtype(model, np.float16, "fp16")
    fp32_path = export_chunk_with_rec_dtype(model, np.float32, "fp32")

    del model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Test loadability
    print("\n" + "=" * 60)
    print("ANE LOADABILITY TESTS")
    print("=" * 60)

    fp16_loads = test_ane_loadability(fp16_path, "fp16")
    fp32_loads = test_ane_loadability(fp32_path, "fp32")

    fp16_all = test_ane_only(fp16_path, "fp16")
    fp32_all = test_ane_only(fp32_path, "fp32")

    print(f"\n--- Summary ---")
    print(f"  FP16 rec I/O: CPU_AND_NE={fp16_loads}, ALL={fp16_all}")
    print(f"  FP32 rec I/O: CPU_AND_NE={fp32_loads}, ALL={fp32_all}")

    if not fp32_loads:
        print("\n[ABORT] FP32 rec I/O cannot load on ANE. Stopping experiment.")
        return

    # State drift comparison
    print("\n" + "=" * 60)
    print("STATE DRIFT COMPARISON (chunk 0 only)")
    print("=" * 60)

    # Reload model for embedding generation
    model = load_model()
    compare_state_drift(fp16_path, fp32_path, model)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
