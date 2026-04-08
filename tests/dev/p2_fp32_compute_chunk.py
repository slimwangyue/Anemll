#!/usr/bin/env python3
"""P2: Test FP32 compute_precision on chunk 0 — ANE-first approach.

The P1 experiment proved:
  - PyTorch FP16 ≈ PyTorch FP32 (cos 0.999999) — FP16 precision is NOT the issue
  - CoreML diverges from both (cos 0.992 → 0.981) — CoreML/ANE MIL lowering differs
  - FP32 I/O has zero effect (states are identical to FP16 I/O)

This test exports chunk 0 with compute_precision=FLOAT32 to force all ops
(including recurrence) to FP32, then tests:
  1. ANE loadability (will ANE still accept FP32 compute?)
  2. Whether FP32 compute eliminates the divergence from PyTorch
  3. If ANE rejects it, what the fallback is

Usage:
  cd /Users/yw68/Anemll
  source .venv/bin/activate
  python tests/dev/p2_fp32_compute_chunk.py
"""
import gc, os, sys, time
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts_qwen3_5"))
sys.path.insert(0, REPO_ROOT)

import coremltools as ct
from config import (
    BATCH_SIZE, CTX, NUM_CHUNKS, LUT_BITS,
    PER_CHANNEL, FFN_PER_CHANNEL, CHUNK_RANGES,
    DEFAULT_HF_MODEL,
)
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, MODEL_DTYPE, TEST_DEVICE,
    ane_conv_state_shape,
)
from anemll.ane_converter.qwen3_5_converter import Qwen35Converter

HF_PATH = DEFAULT_HF_MODEL
CHUNK_IDX = 0
SL, EL = CHUNK_RANGES[CHUNK_IDX]
OUTPUT_DIR = os.path.join(REPO_ROOT, "tests", "dev", "_p2_fp32_compute")
EMBED_PATH = os.path.join(REPO_ROOT, "qwen3_5_flll_9chunk", "embeddings.mlpackage")

FP16_CHUNK = os.path.join(REPO_ROOT, "tests", "dev", "_p2_fp32_compute", "chunk0_fp16compute_lut6.mlpackage")


def export_fp16_compute_chunk(model):
    """Export chunk 0 with compute_precision=FLOAT16 (baseline)."""
    print(f"\n{'='*60}")
    print(f"Exporting chunk {CHUNK_IDX} [{SL}-{EL-1}] with compute_precision=FLOAT16 (baseline)")
    print(f"{'='*60}")
    cfg = model.config
    local_num_layers = EL - SL

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
            self.register_buffer("v_cache", torch.zeros_like(self.k_cache))
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
                hidden_states=hidden_states, position_ids=position_ids,
                causal_mask=causal_mask, current_pos=current_pos,
                kv_cache_0=None, k_cache=self.k_cache, v_cache=self.v_cache,
                linear_conv_state=linear_conv_state, linear_recurrent_state=linear_recurrent_state,
                start_layer=self.start_layer, end_layer=self.end_layer, apply_final_norm=False)
            return out, linear_conv_state, linear_recurrent_state

    wrapper = FFNWrapper(model, SL, EL).eval()
    hidden_states = torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
    position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, CTX), dtype=torch.float16, device=TEST_DEVICE)
    current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    lin_conv = torch.zeros(wrapper._lin_conv_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
    lin_rec = torch.zeros(wrapper._lin_rec_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
    wrapper.k_cache.zero_(); wrapper.v_cache.zero_()
    traced = torch.jit.trace(wrapper, (hidden_states, position_ids, causal_mask, current_pos, lin_conv, lin_rec), check_trace=False)
    wrapper.k_cache.zero_(); wrapper.v_cache.zero_()
    mlmodel = ct.convert(traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
            ct.TensorType(name="linear_conv_state", shape=lin_conv.shape, dtype=np.float16),
            ct.TensorType(name="linear_recurrent_state", shape=lin_rec.shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="output_hidden_states", dtype=np.float16),
            ct.TensorType(name="linear_conv_state_out", dtype=np.float16),
            ct.TensorType(name="linear_recurrent_state_out", dtype=np.float16),
        ],
        states=wrapper.states, compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE, minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram")
    conv = Qwen35Converter(model, context_length=CTX, batch_size=BATCH_SIZE,
                           num_chunks=NUM_CHUNKS, lut_bits=LUT_BITS, per_channel=FFN_PER_CHANNEL)
    conv.converted_model = mlmodel
    conv.postprocess(num_workers=1)
    mlmodel = conv.converted_model
    save_path = os.path.join(OUTPUT_DIR, "chunk0_fp16compute_lut6.mlpackage")
    mlmodel.save(save_path)
    print(f"  Saved to {save_path}")
    del mlmodel, conv, traced, wrapper; gc.collect()
    return save_path


def export_fp32_compute_chunk(model):
    """Export chunk 0 with compute_precision=FLOAT32."""
    print(f"\n{'='*60}")
    print(f"Exporting chunk {CHUNK_IDX} [{SL}-{EL-1}] with compute_precision=FLOAT32")
    print(f"{'='*60}")

    cfg = model.config
    local_num_layers = EL - SL

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
            self.register_buffer("v_cache", torch.zeros_like(self.k_cache))
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

    hidden_states = torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
    position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    causal_mask = torch.zeros((1, 1, 1, CTX), dtype=torch.float16, device=TEST_DEVICE)
    current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
    lin_conv = torch.zeros(wrapper._lin_conv_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
    lin_rec = torch.zeros(wrapper._lin_rec_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)

    wrapper.k_cache.zero_()
    wrapper.v_cache.zero_()
    traced = torch.jit.trace(
        wrapper, (hidden_states, position_ids, causal_mask, current_pos, lin_conv, lin_rec),
        check_trace=False,
    )
    wrapper.k_cache.zero_()
    wrapper.v_cache.zero_()

    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
            ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
            ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
            ct.TensorType(name="linear_conv_state", shape=lin_conv.shape, dtype=np.float16),
            ct.TensorType(name="linear_recurrent_state", shape=lin_rec.shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="output_hidden_states", dtype=np.float16),
            ct.TensorType(name="linear_conv_state_out", dtype=np.float16),
            ct.TensorType(name="linear_recurrent_state_out", dtype=np.float16),
        ],
        states=wrapper.states,
        # KEY CHANGE: FP32 compute precision
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    elapsed = time.time() - t0
    print(f"  Converted with FP32 compute in {elapsed:.1f}s")

    # Apply LUT6 quantization immediately (saves disk space, orthogonal to compute precision)
    conv = Qwen35Converter(
        model, context_length=CTX, batch_size=BATCH_SIZE,
        num_chunks=NUM_CHUNKS, lut_bits=LUT_BITS, per_channel=FFN_PER_CHANNEL,
    )
    conv.converted_model = mlmodel
    conv.postprocess(num_workers=1)
    mlmodel_lut = conv.converted_model
    save_path_lut = os.path.join(OUTPUT_DIR, "chunk0_fp32compute_lut6.mlpackage")
    mlmodel_lut.save(save_path_lut)
    print(f"  Saved (LUT6 + FP32 compute) to {save_path_lut}")

    del mlmodel, mlmodel_lut, conv, traced, wrapper
    gc.collect()
    return save_path_lut


def test_load_and_predict(path, label, compute_unit=ct.ComputeUnit.CPU_AND_NE):
    """Load model and test predict."""
    cu_name = str(compute_unit).split('.')[-1]
    print(f"\n--- Load Test: {label} ({cu_name}) ---")
    try:
        t0 = time.time()
        m = ct.models.MLModel(path, compute_units=compute_unit)
        load_time = time.time() - t0
        print(f"  [OK] Loaded in {load_time:.1f}s")

        state = m.make_state()
        spec = m.get_spec()
        inp_shapes = {}
        for inp in spec.description.input:
            try:
                inp_shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
            except:
                pass

        test_inp = {
            "hidden_states": np.zeros((1, 1, 2560), dtype=np.float16),
            "position_ids": np.array([0], dtype=np.int32),
            "causal_mask": np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16),
            "current_pos": np.array([0], dtype=np.int32),
            "linear_conv_state": np.zeros(inp_shapes['linear_conv_state'], dtype=np.float16),
            "linear_recurrent_state": np.zeros(inp_shapes['linear_recurrent_state'], dtype=np.float16),
        }
        t0 = time.time()
        out = m.predict(test_inp, state=state)
        pred_time = time.time() - t0
        print(f"  [OK] predict() in {pred_time*1000:.0f}ms")
        for k, v in out.items():
            print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
        del m, state
        gc.collect()
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def compare_3way(fp16_path, fp32_nolut_path, fp32_lut_path, model):
    """Compare FP16-compute vs FP32-compute vs FP32+LUT on prompt tokens."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_PATH, trust_remote_code=True)

    prompt = "What is a stack in computer science? Explain in detail."
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    print(f"\nPrompt: {prompt}")
    print(f"Tokens: {len(tokens)}")

    # Load models
    print("Loading 3 CoreML variants + PyTorch reference...")
    cm16 = ct.models.MLModel(fp16_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    cm32 = ct.models.MLModel(fp32_nolut_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    cm32_lut = ct.models.MLModel(fp32_lut_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    embed = ct.models.MLModel(EMBED_PATH, compute_units=ct.ComputeUnit.CPU_ONLY)

    s16 = cm16.make_state()
    s32 = cm32.make_state()
    s32_lut = cm32_lut.make_state()

    spec = cm16.get_spec()
    inp_shapes = {}
    for inp in spec.description.input:
        try:
            inp_shapes[inp.name] = tuple(inp.type.multiArrayType.shape)
        except:
            pass
    conv_shape = inp_shapes['linear_conv_state']
    rec_shape = inp_shapes['linear_recurrent_state']

    # Init states
    conv16 = np.zeros(conv_shape, dtype=np.float16)
    rec16 = np.zeros(rec_shape, dtype=np.float16)
    conv32 = np.zeros(conv_shape, dtype=np.float16)
    rec32 = np.zeros(rec_shape, dtype=np.float16)
    conv32_lut = np.zeros(conv_shape, dtype=np.float16)
    rec32_lut = np.zeros(rec_shape, dtype=np.float16)

    # PyTorch FP32 reference
    cfg = model.config
    local_num_layers = EL - SL
    pt_k = torch.zeros(
        (local_num_layers, cfg.num_key_value_heads, cfg.state_length, cfg.head_dim),
        dtype=torch.float16)
    pt_v = torch.zeros_like(pt_k)
    conv_dim = (
        cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
        + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
    )
    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
    pt_conv = torch.zeros((local_num_layers, ane_dim1, ane_dim2), dtype=torch.float16)
    pt_rec = torch.zeros(rec_shape, dtype=torch.float32)

    def cosine(a, b):
        d = np.dot(a, b)
        n = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
        return d / n

    print(f"\n{'Step':>4} {'CM16-PT32':>12} {'CM32-PT32':>12} {'CM32L-PT32':>13} {'CM16-CM32':>12}")
    print("-" * 65)

    for step in range(len(tokens)):
        tok_id = tokens[step]
        tok_np = np.array([[tok_id]], dtype=np.int32)
        hidden = list(embed.predict({"input_ids": tok_np}).values())[0]

        mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
        mask[:, :, :, :step + 1] = 0

        base_inp = {
            "hidden_states": hidden.astype(np.float16),
            "position_ids": np.array([step], dtype=np.int32),
            "causal_mask": mask,
            "current_pos": np.array([step], dtype=np.int32),
        }

        # CoreML FP16 compute
        inp16 = {**base_inp, "linear_conv_state": conv16, "linear_recurrent_state": rec16}
        out16 = cm16.predict(inp16, state=s16)
        conv16, rec16 = out16['linear_conv_state_out'], out16['linear_recurrent_state_out']

        # CoreML FP32 compute (no LUT)
        inp32 = {**base_inp, "linear_conv_state": conv32, "linear_recurrent_state": rec32}
        out32 = cm32.predict(inp32, state=s32)
        conv32, rec32 = out32['linear_conv_state_out'], out32['linear_recurrent_state_out']

        # CoreML FP32 compute + LUT6
        inp32l = {**base_inp, "linear_conv_state": conv32_lut, "linear_recurrent_state": rec32_lut}
        out32l = cm32_lut.predict(inp32l, state=s32_lut)
        conv32_lut, rec32_lut = out32l['linear_conv_state_out'], out32l['linear_recurrent_state_out']

        # PyTorch FP32
        hidden_pt = torch.from_numpy(hidden.astype(np.float16))
        with torch.no_grad():
            pt_out = model.model.process_layers_regular_single_token_export_local_state(
                hidden_states=hidden_pt.half(),
                position_ids=torch.tensor([step], dtype=torch.int32),
                causal_mask=torch.from_numpy(mask),
                current_pos=torch.tensor([step], dtype=torch.int32),
                kv_cache_0=None,
                k_cache=pt_k, v_cache=pt_v,
                linear_conv_state=pt_conv,
                linear_recurrent_state=pt_rec,
                start_layer=SL, end_layer=EL,
                apply_final_norm=False,
            )

        # Compare recurrent states
        pt_f = pt_rec.detach().numpy().flatten().astype(np.float64)
        r16_f = rec16.flatten().astype(np.float64)
        r32_f = rec32.flatten().astype(np.float64)
        r32l_f = rec32_lut.flatten().astype(np.float64)

        cos_16_pt = cosine(r16_f, pt_f)
        cos_32_pt = cosine(r32_f, pt_f)
        cos_32l_pt = cosine(r32l_f, pt_f)
        cos_16_32 = cosine(r16_f, r32_f)

        if step % 3 == 0 or step < 5 or step == len(tokens) - 1:
            print(f"{step:4d} {cos_16_pt:12.8f} {cos_32_pt:12.8f} {cos_32l_pt:13.8f} {cos_16_32:12.8f}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL (after {len(tokens)} tokens):")
    print(f"  CoreML FP16-compute vs PyTorch-FP32: {cosine(r16_f, pt_f):.8f}")
    print(f"  CoreML FP32-compute vs PyTorch-FP32: {cosine(r32_f, pt_f):.8f}")
    print(f"  CoreML FP32+LUT6   vs PyTorch-FP32:  {cosine(r32l_f, pt_f):.8f}")
    print(f"  CoreML FP16 vs FP32-compute:          {cosine(r16_f, r32_f):.8f}")

    # Hidden state comparison
    h16 = out16['output_hidden_states'].flatten().astype(np.float64)
    h32 = out32['output_hidden_states'].flatten().astype(np.float64)
    h32l = out32l['output_hidden_states'].flatten().astype(np.float64)
    h_pt = pt_out.detach().numpy().flatten().astype(np.float64)
    print(f"\n  Hidden cosine (last step):")
    print(f"    CM-FP16 vs PT-FP32: {cosine(h16, h_pt):.8f}")
    print(f"    CM-FP32 vs PT-FP32: {cosine(h32, h_pt):.8f}")
    print(f"    CM-FP32+LUT vs PT:  {cosine(h32l, h_pt):.8f}")

    del cm16, cm32, cm32_lut, embed, s16, s32, s32_lut
    gc.collect()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
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

    # Export FP16 baseline and FP32-compute chunk
    fp16_path = export_fp16_compute_chunk(model)
    fp32_lut_path = export_fp32_compute_chunk(model)

    del model
    gc.collect()

    # Loadability tests
    print(f"\n{'='*60}")
    print("ANE LOADABILITY TESTS")
    print(f"{'='*60}")

    # Existing FP16 chunk
    test_load_and_predict(fp16_path, "FP16-compute (baseline)")
    # FP32 + LUT6
    ok = test_load_and_predict(fp32_lut_path, "FP32-compute + LUT6")

    # Also test CPU_ONLY to check if ANE rejects it
    test_load_and_predict(fp32_lut_path, "FP32-compute (CPU_ONLY)", ct.ComputeUnit.CPU_ONLY)

    # 3-way comparison
    print(f"\n{'='*60}")
    print("COMPARISON: FP16-compute vs FP32-compute+LUT6")
    print(f"{'='*60}")

    model = Qwen35ForCausalLM(cfg)
    assert model.load_pretrained_weights(HF_PATH)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    compare_3way(fp16_path, fp32_lut_path, fp32_lut_path, model)

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
