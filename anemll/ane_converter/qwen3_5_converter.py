"""Converter for Qwen 3.5 text models with chunked transformer export."""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Dict, List, Optional

import coremltools as ct
import coremltools.optimize as cto
import numpy as np
import torch
try:
    from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
except Exception:  # pragma: no cover - sklearn optional
    SklearnConvergenceWarning = None

from .base_converter import BaseConverter
from .environment import require_coreml
from .metadata import AddMetadata, ModelPart
from ..models.qwen3_5_model import (
    CONTEXT_LENGTH,
    MODEL_DTYPE,
    TEST_DEVICE,
    Qwen35Config,
    Qwen35ForCausalLM,
    ane_conv_state_shape,
)

if SklearnConvergenceWarning is not None:
    warnings.filterwarnings("ignore", category=SklearnConvergenceWarning)
warnings.filterwarnings("ignore", message="Number of distinct clusters .* smaller than n_clusters")


class Qwen35Converter(BaseConverter):
    """Handle conversion of Qwen3.5 models to Core ML."""

    model_cls = Qwen35ForCausalLM

    def __init__(
        self,
        model: Qwen35ForCausalLM,
        context_length: int = CONTEXT_LENGTH,
        batch_size: int = 64,
        lut_bits: int | None = 4,
        per_channel: int = 8,
        num_chunks: int = 1,
        argmax_in_model: bool = False,
        lut_embeddings_bits=None,
        lut_embeddings_per_channel=8,
        lut_lmhead_bits=None,
        lut_lmhead_per_channel=8,
    ) -> None:
        super().__init__(model)
        self.context_length = context_length
        self.batch_size = batch_size
        self.lut_bits = lut_bits
        self.per_channel = per_channel
        self.num_chunks = num_chunks
        self.argmax_in_model = argmax_in_model
        self.lut_embeddings_bits = lut_embeddings_bits
        self.lut_embeddings_per_channel = lut_embeddings_per_channel
        self.lut_lmhead_bits = lut_lmhead_bits
        self.lut_lmhead_per_channel = lut_lmhead_per_channel
        self.converted_model = None

    @staticmethod
    def _effective_lut_bits_for_part(part: str, lut_bits: int | None) -> int | None:
        """Return the LUT setting that is actually safe for a given export part.

        Chunked transformer stages currently show input-dependent CoreML runtime
        failures after broad LUT palettization. Keep embeddings / lm_head LUT-capable,
        but export transformer chunk models in fp16 weights until we have a proven
        safe selective palettization plan for Qwen3.5.
        """
        if part in {"2", "2_prefill"}:
            return None
        return lut_bits

    @staticmethod
    def _chunk_postprocess_workers(total_chunks: int) -> int | None:
        """Return the palettization worker count for chunk exports.

        By default, historical behavior kept multi-chunk exports on a single
        worker to minimize memory spikes. Exploratory export workflows can
        override this with `QWEN35_CHUNK_POSTPROCESS_WORKERS` to trade memory
        for substantially faster LUT postprocess time.
        """
        env_value = os.environ.get("QWEN35_CHUNK_POSTPROCESS_WORKERS")
        if env_value is not None:
            try:
                parsed = int(env_value)
                if parsed > 0:
                    return parsed
            except ValueError:
                pass
        return None if total_chunks > 1 else 8

    @staticmethod
    def GetTransformerStates(
        model: Qwen35ForCausalLM,
        part=None,
        prefix: str = "model.model.",
    ):
        del part
        cfg = model.config
        states = [
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(
                        2 * cfg.num_hidden_layers,
                        cfg.num_key_value_heads,
                        cfg.state_length,
                        cfg.head_dim,
                    ),
                    dtype=np.float16,
                ),
                name=f"{prefix}kv_cache_0",
            )
        ]

        if cfg.has_linear_attention():
            conv_dim = (
                cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
            )
            conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
            ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
            states.append(
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(cfg.num_hidden_layers, ane_dim1, ane_dim2),
                        dtype=np.float16,
                    ),
                    name=f"{prefix}linear_conv_state",
                )
            )
            states.append(
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(
                            cfg.num_hidden_layers,
                            cfg.text_config.linear_num_value_heads,
                            cfg.text_config.linear_key_head_dim,
                            cfg.text_config.linear_value_head_dim,
                        ),
                        dtype=np.float16,
                    ),
                    name=f"{prefix}linear_recurrent_state",
                )
            )
        return states

    @staticmethod
    def GetChunkLocalTransformerStates(
        model: Qwen35ForCausalLM,
        num_layers: int,
        prefix: str = "",
        split_full_attention_kv: bool = False,
    ):
        cfg = model.config
        if split_full_attention_kv:
            states = [
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(
                            num_layers,
                            cfg.num_key_value_heads,
                            cfg.state_length,
                            cfg.head_dim,
                        ),
                        dtype=np.float16,
                    ),
                    name=f"{prefix}k_cache",
                ),
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(
                            num_layers,
                            cfg.num_key_value_heads,
                            cfg.state_length,
                            cfg.head_dim,
                        ),
                        dtype=np.float16,
                    ),
                    name=f"{prefix}v_cache",
                ),
            ]
        else:
            states = [
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(
                            2 * num_layers,
                            cfg.num_key_value_heads,
                            cfg.state_length,
                            cfg.head_dim,
                        ),
                        dtype=np.float16,
                    ),
                    name=f"{prefix}kv_cache_0",
                )
            ]

        # NOTE: linear_conv_state and linear_recurrent_state are intentionally
        # NOT included as CoreML states.  They are passed as regular I/O tensors
        # to avoid the rounding corruption that ct.StateType introduces in the
        # read/write cycle of recurrent states (see stateless parity tests).
        return states

    @staticmethod
    def GetChunkLocalPerLayerStates(
        model: Qwen35ForCausalLM,
        num_full_attn_layers: int,
        prefix: str = "",
    ):
        """Per-layer 3D KV cache states for ANE-friendly dynamic slicing.

        Instead of one 4D tensor (num_layers, kv_heads, CTX, head_dim) that
        requires compound indexing (layer dim + position dim), this creates
        separate 3D (kv_heads, CTX, head_dim) states for each full-attention
        layer.  The simpler 3D slice_update is more reliably ANE-legal.
        """
        cfg = model.config
        states = []
        for i in range(num_full_attn_layers):
            states.append(
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(
                            cfg.num_key_value_heads,
                            cfg.state_length,
                            cfg.head_dim,
                        ),
                        dtype=np.float16,
                    ),
                    name=f"{prefix}k_cache_{i}",
                )
            )
            states.append(
                ct.StateType(
                    wrapped_type=ct.TensorType(
                        shape=(
                            cfg.num_key_value_heads,
                            cfg.state_length,
                            cfg.head_dim,
                        ),
                        dtype=np.float16,
                    ),
                    name=f"{prefix}v_cache_{i}",
                )
            )
        return states

    @staticmethod
    def _make_palettizer_config(nbits, per_channel, num_workers):
        if per_channel <= 0:
            return cto.coreml.OpPalettizerConfig(
                mode="kmeans",
                nbits=nbits,
                granularity="per_tensor",
                num_kmeans_workers=num_workers if num_workers is not None else 1,
            )
        return cto.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=nbits,
            granularity="per_grouped_channel",
            group_size=per_channel,
            num_kmeans_workers=num_workers if num_workers is not None else 1,
        )

    def postprocess(self, num_workers=None):
        if self.converted_model is None or self.lut_bits is None:
            return
        try:
            with warnings.catch_warnings():
                if SklearnConvergenceWarning is not None:
                    warnings.simplefilter("ignore", SklearnConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                global_cfg = self._make_palettizer_config(self.lut_bits, self.per_channel, num_workers)
                config = cto.coreml.OptimizationConfig(global_config=global_cfg)
                try:
                    self.converted_model = cto.coreml.palettize_weights(self.converted_model, config)
                except Exception as exc:
                    print(f"Warning: palettize_weights raised: {exc}")
                    print("Retrying without worker parallelism...")
                    fallback_cfg = self._make_palettizer_config(self.lut_bits, self.per_channel, None)
                    fallback = cto.coreml.OptimizationConfig(global_config=fallback_cfg)
                    self.converted_model = cto.coreml.palettize_weights(self.converted_model, fallback)
        except Exception as exc:
            print(f"LUT quantization failed: {exc}")
            print("Continuing without quantization...")

    @staticmethod
    def _reset_state_buffers(module: torch.nn.Module | torch.jit.ScriptModule) -> None:
        with torch.no_grad():
            for name, buffer in module.named_buffers():
                if (
                    "kv_cache_" in name
                    or name in {"k_cache", "v_cache"}
                ):
                    buffer.zero_()

    def convert(self, part: str = "full") -> ct.models.MLModel | List[ct.models.MLModel]:
        require_coreml()
        self.preprocess()

        if part in ("full", "all", "123"):
            mlmodel = self.convert_to_coreml(self.model)
        elif part == "monolithic":
            mlmodel = self.convert_monolithic(
                self.model,
                is_prefill=False,
                argmax_in_model=self.argmax_in_model,
            )
        elif part == "monolithic_prefill":
            mlmodel = self.convert_monolithic(
                self.model,
                is_prefill=True,
                argmax_in_model=False,
            )
        elif part in ("embeddings", "1"):
            mlmodel = self.convert_part_1(self.model)
        elif part in ("prefill", "2_prefill"):
            if self.num_chunks > 1:
                mlmodel = [
                    self.convert_part_2_prefill(self.model, i, self.num_chunks)
                    for i in range(self.num_chunks)
                ]
            else:
                mlmodel = self.convert_part_2_prefill(self.model)
        elif part == "2":
            if self.num_chunks > 1:
                mlmodel = [
                    self.convert_part_2(self.model, i, self.num_chunks)
                    for i in range(self.num_chunks)
                ]
            else:
                mlmodel = self.convert_part_2(self.model)
        elif part == "3":
            mlmodel = self.convert_part_3(self.model, argmax_in_model=self.argmax_in_model)
        else:
            raise ValueError(f"Unsupported part: {part}")
        return mlmodel

    def convert_to_coreml(self, model: Qwen35ForCausalLM) -> ct.models.MLModel:
        require_coreml()

        class Wrapper(torch.nn.Module):
            def __init__(self, model: Qwen35ForCausalLM) -> None:
                super().__init__()
                self.model = model

            def forward(self, input_ids, position_ids, causal_mask, current_pos):
                return self.model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    causal_mask=causal_mask,
                    current_pos=current_pos,
                    IN_PREFILL=False,
                )

        wrapper = Wrapper(model).eval()
        sample_input_ids = torch.zeros((1, 1), dtype=torch.int32, device=TEST_DEVICE)
        sample_position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
        sample_causal_mask = torch.zeros(
            (1, 1, 1, self.context_length), dtype=torch.float16, device=TEST_DEVICE
        )
        sample_current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

        self._reset_state_buffers(wrapper)
        traced = torch.jit.trace(
            wrapper,
            (sample_input_ids, sample_position_ids, sample_causal_mask, sample_current_pos),
        )
        self._reset_state_buffers(wrapper)
        self._reset_state_buffers(traced)

        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="input_ids", shape=sample_input_ids.shape, dtype=np.int32),
                ct.TensorType(name="position_ids", shape=sample_position_ids.shape, dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=sample_causal_mask.shape, dtype=np.float16),
                ct.TensorType(name="current_pos", shape=sample_current_pos.shape, dtype=np.int32),
            ],
            outputs=[ct.TensorType(name="logits", dtype=np.float16)],
            states=self.GetTransformerStates(model, prefix="model.model."),
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        if self.lut_bits:
            self.converted_model = mlmodel
            self.postprocess(num_workers=1)
            mlmodel = self.converted_model
        return mlmodel

    def convert_part_1(self, model: Qwen35ForCausalLM) -> ct.models.MLModel:
        require_coreml()

        class EmbeddingsWrapper(torch.nn.Module):
            def __init__(self, model: Qwen35ForCausalLM) -> None:
                super().__init__()
                self.embed_tokens = model.model.embed_tokens

            def forward(self, input_ids):
                return self.embed_tokens(input_ids).to(MODEL_DTYPE)

        wrapper = EmbeddingsWrapper(model).eval()
        sample_input = torch.zeros((1, 1), dtype=torch.int32, device=TEST_DEVICE)
        traced = torch.jit.trace(wrapper, sample_input)
        input_shape = ct.EnumeratedShapes(shapes=[[1, 1], [1, self.batch_size]], default=[1, 1])
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input_ids", shape=input_shape, dtype=np.int32)],
            outputs=[ct.TensorType(name="hidden_states", dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        if self.lut_bits:
            self.converted_model = mlmodel
            self.postprocess(num_workers=1)
            mlmodel = self.converted_model
        return mlmodel

    def convert_part_2(
        self, model: Qwen35ForCausalLM, chunk_idx: int = 0, total_chunks: int = 1
    ) -> ct.models.MLModel:
        require_coreml()
        total_layers = model.config.num_hidden_layers
        if total_chunks > 1:
            base, rem = divmod(total_layers, total_chunks)
            start_layer = chunk_idx * base + min(chunk_idx, rem)
            end_layer = start_layer + base + (1 if chunk_idx < rem else 0)
        else:
            start_layer = 0
            end_layer = None
        local_num_layers = (end_layer - start_layer) if end_layer is not None else total_layers

        class FFNWrapper(torch.nn.Module):
            def __init__(self, model: Qwen35ForCausalLM, start_layer: int, end_layer: int | None) -> None:
                super().__init__()
                self.model = model
                self.start_layer = start_layer
                self.end_layer = end_layer
                self.local_num_layers = (end_layer - start_layer) if end_layer is not None else len(model.model.layers)
                cfg = model.config
                self.register_buffer(
                    "k_cache",
                    torch.zeros(
                        (
                            self.local_num_layers,
                            cfg.num_key_value_heads,
                            cfg.state_length,
                            cfg.head_dim,
                        ),
                        dtype=MODEL_DTYPE,
                        device=TEST_DEVICE,
                    ),
                )
                self.register_buffer(
                    "v_cache",
                    torch.zeros(
                        (
                            self.local_num_layers,
                            cfg.num_key_value_heads,
                            cfg.state_length,
                            cfg.head_dim,
                        ),
                        dtype=MODEL_DTYPE,
                        device=TEST_DEVICE,
                    ),
                )
                if cfg.has_linear_attention():
                    conv_dim = (
                        cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                        + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
                    )
                    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
                    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
                    # Store shapes for forward() — states are I/O, NOT register_buffer
                    self._lin_conv_shape = (self.local_num_layers, ane_dim1, ane_dim2)
                    self._lin_rec_shape = (
                        self.local_num_layers,
                        cfg.text_config.linear_num_value_heads,
                        cfg.text_config.linear_key_head_dim,
                        cfg.text_config.linear_value_head_dim,
                    )
                    self._has_linear = True
                else:
                    self._has_linear = False
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
                if self.end_layer is None or self.end_layer == len(self.model.model.layers):
                    out = self.model.model.norm(out)
                return out, linear_conv_state, linear_recurrent_state

        wrapper = FFNWrapper(model, start_layer, end_layer).eval()
        cfg = model.config
        hidden_states = torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float16, device=TEST_DEVICE)
        position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
        # causal_mask: full CTX width for attention masking.
        causal_mask = torch.zeros((1, 1, 1, self.context_length), dtype=torch.float16, device=TEST_DEVICE)
        current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

        # Linear attention states as regular I/O tensors (not CoreML state)
        if wrapper._has_linear:
            lin_conv = torch.zeros(wrapper._lin_conv_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
            lin_rec = torch.zeros(wrapper._lin_rec_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
        else:
            # Dummy zero-size placeholders (model has no linear attention)
            lin_conv = torch.zeros((local_num_layers, 1, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)
            lin_rec = torch.zeros((local_num_layers, 1, 1, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)

        self._reset_state_buffers(wrapper)
        traced = torch.jit.trace(
            wrapper, (hidden_states, position_ids, causal_mask, current_pos, lin_conv, lin_rec),
            check_trace=False,
        )
        self._reset_state_buffers(wrapper)
        self._reset_state_buffers(traced)

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
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        if self.lut_bits:
            self.converted_model = mlmodel
            num_workers = self._chunk_postprocess_workers(total_chunks)
            self.postprocess(num_workers=num_workers)
            mlmodel = self.converted_model
        return mlmodel

    def convert_part_2_prefill(
        self, model: Qwen35ForCausalLM, chunk_idx: int = 0, total_chunks: int = 1,
        block_start: int = 0,
    ) -> ct.models.MLModel:
        require_coreml()
        total_layers = model.config.num_hidden_layers
        if total_chunks > 1:
            base, rem = divmod(total_layers, total_chunks)
            start_layer = chunk_idx * base + min(chunk_idx, rem)
            end_layer = start_layer + base + (1 if chunk_idx < rem else 0)
        else:
            start_layer = 0
            end_layer = None
        local_num_layers = (end_layer - start_layer) if end_layer is not None else total_layers

        class PrefillWrapper(torch.nn.Module):
            def __init__(
                self,
                model: Qwen35ForCausalLM,
                start_layer: int,
                end_layer: int | None,
                export_seq_len: int,
            ) -> None:
                super().__init__()
                self.model = model
                self.start_layer = start_layer
                self.end_layer = end_layer
                self.export_seq_len = export_seq_len
                self.local_num_layers = (end_layer - start_layer) if end_layer is not None else len(model.model.layers)
                self._is_last_chunk = (
                    (end_layer is None) or end_layer == len(model.model.layers)
                )
                self._hidden_size = model.config.hidden_size
                cfg = model.config
                self.register_buffer(
                    "k_cache",
                    torch.zeros(
                        (
                            self.local_num_layers,
                            cfg.num_key_value_heads,
                            cfg.state_length,
                            cfg.head_dim,
                        ),
                        dtype=MODEL_DTYPE,
                        device=TEST_DEVICE,
                    ),
                )
                self.register_buffer(
                    "v_cache",
                    torch.zeros(
                        (
                            self.local_num_layers,
                            cfg.num_key_value_heads,
                            cfg.state_length,
                            cfg.head_dim,
                        ),
                        dtype=MODEL_DTYPE,
                        device=TEST_DEVICE,
                    ),
                )
                if cfg.has_linear_attention():
                    conv_dim = (
                        cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
                        + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim
                    )
                    conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
                    ane_dim1, ane_dim2 = ane_conv_state_shape(conv_dim, conv_kernel)
                    # Store shapes for forward() — states are I/O, NOT register_buffer
                    self._lin_conv_shape = (self.local_num_layers, ane_dim1, ane_dim2)
                    self._lin_rec_shape = (
                        self.local_num_layers,
                        cfg.text_config.linear_num_value_heads,
                        cfg.text_config.linear_key_head_dim,
                        cfg.text_config.linear_value_head_dim,
                    )
                    self._has_linear = True
                else:
                    self._has_linear = False
                for layer_idx in range(self.start_layer, self.end_layer if self.end_layer is not None else len(self.model.model.layers)):
                    layer = self.model.model.layers[layer_idx]
                    if getattr(layer, "layer_type", None) == "linear_attention":
                        layer.self_attn.export_expected_batch_size = 1
                        layer.self_attn.export_expected_seq_len = export_seq_len
                self.states = Qwen35Converter.GetChunkLocalTransformerStates(
                    model, self.local_num_layers, prefix="", split_full_attention_kv=True
                )

            def forward(self, hidden_states, position_ids, causal_mask, current_pos,
                        linear_conv_state, linear_recurrent_state, valid_len):
                out = self.model.model.process_layers_prefill_export_local_state(
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
                    expected_batch_size=1,
                    expected_seq_len=self.export_seq_len,
                    valid_len=valid_len,
                )
                if self._is_last_chunk:
                    # Apply final RMSNorm (matches FFN/infer wrapper behavior).
                    out = self.model.model.norm(out)
                    # Extract the last valid token's hidden state.
                    # Use one-hot bmm instead of torch.gather to avoid
                    # aten::Int / int64 cast issues with coremltools.
                    # positions: (seq_len,) int32 — static at trace time
                    # selector: (1, 1, seq_len) float16 — one-hot at valid_len-1
                    seq_len = self.export_seq_len
                    positions = torch.arange(
                        seq_len, device=out.device, dtype=torch.int32)
                    target = valid_len - 1  # (1,) int32
                    selector = (positions == target).to(out.dtype)  # (seq_len,)
                    selector = selector.reshape(1, 1, seq_len)
                    out = torch.bmm(selector, out)  # (1, 1, hidden)
                    return out, linear_conv_state, linear_recurrent_state
                return out, linear_conv_state, linear_recurrent_state

        wrapper = PrefillWrapper(model, start_layer, end_layer, self.batch_size).eval()
        cfg = model.config
        hidden_states = torch.zeros(
            (1, self.batch_size, cfg.hidden_size), dtype=torch.float16, device=TEST_DEVICE
        )
        position_ids = torch.zeros((self.batch_size,), dtype=torch.int32, device=TEST_DEVICE)
        # causal_mask: full CTX width for attention masking.
        causal_mask = torch.zeros(
            (1, 1, self.batch_size, self.context_length), dtype=torch.float16, device=TEST_DEVICE
        )
        current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

        # Linear attention states as regular I/O tensors (not CoreML state)
        if wrapper._has_linear:
            lin_conv = torch.zeros(wrapper._lin_conv_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
            lin_rec = torch.zeros(wrapper._lin_rec_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
        else:
            lin_conv = torch.zeros((local_num_layers, 1, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)
            lin_rec = torch.zeros((local_num_layers, 1, 1, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)

        valid_len = torch.tensor([self.batch_size], dtype=torch.int32, device=TEST_DEVICE)

        self._reset_state_buffers(wrapper)
        traced = torch.jit.trace(
            wrapper, (hidden_states, position_ids, causal_mask, current_pos, lin_conv, lin_rec, valid_len),
            check_trace=False,
        )
        self._reset_state_buffers(wrapper)
        self._reset_state_buffers(traced)

        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="hidden_states", shape=hidden_states.shape, dtype=np.float16),
                ct.TensorType(name="position_ids", shape=position_ids.shape, dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
                ct.TensorType(name="current_pos", shape=current_pos.shape, dtype=np.int32),
                ct.TensorType(name="linear_conv_state", shape=lin_conv.shape, dtype=np.float16),
                ct.TensorType(name="linear_recurrent_state", shape=lin_rec.shape, dtype=np.float16),
                ct.TensorType(name="valid_len", shape=valid_len.shape, dtype=np.int32),
            ],
            outputs=[
                ct.TensorType(name="output_hidden_states", dtype=np.float16),
                ct.TensorType(name="linear_conv_state_out", dtype=np.float16),
                ct.TensorType(name="linear_recurrent_state_out", dtype=np.float16),
            ],
            states=wrapper.states,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        if self.lut_bits:
            self.converted_model = mlmodel
            num_workers = self._chunk_postprocess_workers(total_chunks)
            self.postprocess(num_workers=num_workers)
            mlmodel = self.converted_model
        return mlmodel

    def convert_part_2_prefill_exact(
        self,
        model: Qwen35ForCausalLM,
        chunk_idx: int = 0,
        total_chunks: int = 1,
        block_start: int = 0,
        exact_seq_len: int | None = None,
    ) -> ct.models.MLModel:
        """Convert a static-shape prefill chunk with no valid_len input.

        This exporter is intended for exact bucket experiments where the
        prefill sequence length is fixed at trace time and every token position
        in the bucket is considered valid.
        """
        require_coreml()
        total_layers = model.config.num_hidden_layers
        if total_chunks > 1:
            base, rem = divmod(total_layers, total_chunks)
            start_layer = chunk_idx * base + min(chunk_idx, rem)
            end_layer = start_layer + base + (1 if chunk_idx < rem else 0)
        else:
            start_layer = 0
            end_layer = None
        local_num_layers = (end_layer - start_layer) if end_layer is not None else total_layers
        export_seq_len = int(exact_seq_len if exact_seq_len is not None else self.batch_size)

        class ExactPrefillWrapper(torch.nn.Module):
            def __init__(
                self,
                model: Qwen35ForCausalLM,
                start_layer: int,
                end_layer: int | None,
                export_seq_len: int,
            ) -> None:
                super().__init__()
                self.model = model
                self.start_layer = start_layer
                self.end_layer = end_layer
                self.export_seq_len = export_seq_len
                self.local_num_layers = (
                    (end_layer - start_layer)
                    if end_layer is not None
                    else len(model.model.layers)
                )
                cfg = model.config
                self.register_buffer(
                    "k_cache",
                    torch.zeros(
                        (
                            self.local_num_layers,
                            cfg.num_key_value_heads,
                            cfg.state_length,
                            cfg.head_dim,
                        ),
                        dtype=MODEL_DTYPE,
                        device=TEST_DEVICE,
                    ),
                )
                self.register_buffer(
                    "v_cache",
                    torch.zeros(
                        (
                            self.local_num_layers,
                            cfg.num_key_value_heads,
                            cfg.state_length,
                            cfg.head_dim,
                        ),
                        dtype=MODEL_DTYPE,
                        device=TEST_DEVICE,
                    ),
                )
                if cfg.has_linear_attention():
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
                    self._has_linear = True
                else:
                    self._has_linear = False
                for layer_idx in range(
                    self.start_layer,
                    self.end_layer if self.end_layer is not None else len(self.model.model.layers),
                ):
                    layer = self.model.model.layers[layer_idx]
                    if getattr(layer, "layer_type", None) == "linear_attention":
                        layer.self_attn.export_expected_batch_size = 1
                        layer.self_attn.export_expected_seq_len = export_seq_len
                self.states = Qwen35Converter.GetChunkLocalTransformerStates(
                    model,
                    self.local_num_layers,
                    prefix="",
                    split_full_attention_kv=True,
                )

            def forward(
                self,
                hidden_states,
                position_ids,
                causal_mask,
                current_pos,
                linear_conv_state,
                linear_recurrent_state,
            ):
                out = self.model.model.process_layers_prefill_export_local_state(
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
                    expected_batch_size=1,
                    expected_seq_len=self.export_seq_len,
                    valid_len=None,
                )
                return out[:, -1:, :], linear_conv_state, linear_recurrent_state

        wrapper = ExactPrefillWrapper(model, start_layer, end_layer, export_seq_len).eval()
        cfg = model.config
        hidden_states = torch.zeros(
            (1, export_seq_len, cfg.hidden_size), dtype=torch.float16, device=TEST_DEVICE
        )
        position_ids = torch.arange(export_seq_len, dtype=torch.int32, device=TEST_DEVICE)
        causal_mask = torch.zeros(
            (1, 1, export_seq_len, self.context_length), dtype=torch.float16, device=TEST_DEVICE
        )
        current_pos = torch.full((1,), int(block_start), dtype=torch.int32, device=TEST_DEVICE)

        if wrapper._has_linear:
            lin_conv = torch.zeros(wrapper._lin_conv_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
            lin_rec = torch.zeros(wrapper._lin_rec_shape, dtype=MODEL_DTYPE, device=TEST_DEVICE)
        else:
            lin_conv = torch.zeros((local_num_layers, 1, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)
            lin_rec = torch.zeros((local_num_layers, 1, 1, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE)

        self._reset_state_buffers(wrapper)
        traced = torch.jit.trace(
            wrapper,
            (hidden_states, position_ids, causal_mask, current_pos, lin_conv, lin_rec),
            check_trace=False,
        )
        self._reset_state_buffers(wrapper)
        self._reset_state_buffers(traced)

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
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        if self.lut_bits:
            self.converted_model = mlmodel
            num_workers = self._chunk_postprocess_workers(total_chunks)
            self.postprocess(num_workers=num_workers)
            mlmodel = self.converted_model
        return mlmodel

    def convert_part_3(
        self, model: Qwen35ForCausalLM, argmax_in_model: bool = False
    ) -> ct.models.MLModel:
        require_coreml()

        class LMHeadWrapper(torch.nn.Module):
            def __init__(self, model: Qwen35ForCausalLM, argmax_mode: bool = False) -> None:
                super().__init__()
                self.heads = [getattr(model, f"lm_head16_{i+1}") for i in range(model.lm_head_split)]
                self.argmax_mode = argmax_mode

            def forward(self, hidden_states):
                h = hidden_states.permute(0, 2, 1).unsqueeze(2)
                logits_parts = [head(h).squeeze(2).permute(0, 2, 1) for head in self.heads]
                if self.argmax_mode:
                    logits = torch.cat(logits_parts, dim=-1)
                    argmax_idx = torch.argmax(logits, dim=-1).to(torch.int32)
                    argmax_val = torch.gather(logits, -1, argmax_idx.unsqueeze(-1)).squeeze(-1)
                    return argmax_idx, argmax_val
                return tuple(logits_parts)

        wrapper = LMHeadWrapper(model, argmax_mode=argmax_in_model).eval()
        sample_input = torch.zeros((1, 1, model.config.hidden_size), dtype=MODEL_DTYPE, device=TEST_DEVICE)
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, sample_input)

        outputs = (
            [
                ct.TensorType(name="argmax_idx", dtype=np.int32),
                ct.TensorType(name="argmax_val", dtype=np.float16),
            ]
            if argmax_in_model
            else [ct.TensorType(name=f"logits{i+1}", dtype=np.float16)
                  for i in range(model.lm_head_split)]
        )
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="hidden_states", shape=sample_input.shape, dtype=np.float16)],
            outputs=outputs,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        if self.lut_bits:
            self.converted_model = mlmodel
            self.postprocess(num_workers=1)
            mlmodel = self.converted_model
        return mlmodel

    def convert_monolithic(
        self,
        model: Qwen35ForCausalLM,
        is_prefill: bool = False,
        argmax_in_model: bool = False,
    ) -> ct.models.MLModel:
        require_coreml()

        class MonolithicWrapper(torch.nn.Module):
            def __init__(self, model: Qwen35ForCausalLM, is_prefill: bool, argmax_in_model: bool) -> None:
                super().__init__()
                self.model = model
                self.is_prefill = is_prefill
                self.argmax_in_model = argmax_in_model

            def forward(self, input_ids, position_ids, causal_mask, current_pos):
                hidden_states = self.model.model.embed_tokens(input_ids).to(MODEL_DTYPE)
                hidden_states = self.model.model.process_layers(
                    hidden_states=hidden_states,
                    position_ids=position_ids,
                    causal_mask=causal_mask,
                    current_pos=current_pos,
                    start_layer=0,
                    end_layer=None,
                    IN_PREFILL=self.is_prefill,
                    apply_final_norm=True,
                )
                h = hidden_states.permute(0, 2, 1).unsqueeze(2)
                logits_parts = [getattr(self.model, f"lm_head16_{i+1}")(h).squeeze(2).permute(0, 2, 1)
                                for i in range(self.model.lm_head_split)]
                if self.argmax_in_model and not self.is_prefill:
                    logits = torch.cat(logits_parts, dim=-1)
                    argmax_idx = torch.argmax(logits, dim=-1).to(torch.int32)
                    argmax_val = torch.gather(logits, -1, argmax_idx.unsqueeze(-1)).squeeze(-1)
                    return argmax_idx, argmax_val
                return tuple(logits_parts)

        wrapper = MonolithicWrapper(model, is_prefill, argmax_in_model).eval()
        if is_prefill:
            sample_input_ids = torch.zeros((1, self.batch_size), dtype=torch.int32, device=TEST_DEVICE)
            sample_position_ids = torch.zeros((self.batch_size,), dtype=torch.int32, device=TEST_DEVICE)
            sample_causal_mask = torch.zeros(
                (1, 1, self.batch_size, self.context_length), dtype=torch.float16, device=TEST_DEVICE
            )
        else:
            sample_input_ids = torch.zeros((1, 1), dtype=torch.int32, device=TEST_DEVICE)
            sample_position_ids = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)
            sample_causal_mask = torch.zeros(
                (1, 1, 1, self.context_length), dtype=torch.float16, device=TEST_DEVICE
            )
        sample_current_pos = torch.zeros((1,), dtype=torch.int32, device=TEST_DEVICE)

        self._reset_state_buffers(wrapper)
        with torch.no_grad():
            traced = torch.jit.trace(
                wrapper, (sample_input_ids, sample_position_ids, sample_causal_mask, sample_current_pos)
            )
        self._reset_state_buffers(wrapper)
        self._reset_state_buffers(traced)

        outputs = (
            [
                ct.TensorType(name="argmax_idx", dtype=np.int32),
                ct.TensorType(name="argmax_val", dtype=np.float16),
            ]
            if (argmax_in_model and not is_prefill)
            else [ct.TensorType(name="logits", dtype=np.float16)]
        )
        mlmodel = ct.convert(
            traced,
            inputs=[
                ct.TensorType(name="input_ids", shape=sample_input_ids.shape, dtype=np.int32),
                ct.TensorType(name="position_ids", shape=sample_position_ids.shape, dtype=np.int32),
                ct.TensorType(name="causal_mask", shape=sample_causal_mask.shape, dtype=np.float16),
                ct.TensorType(name="current_pos", shape=sample_current_pos.shape, dtype=np.int32),
            ],
            outputs=outputs,
            states=self.GetTransformerStates(model, prefix="model.model."),
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        if self.lut_bits:
            self.converted_model = mlmodel
            self.postprocess(num_workers=1)
            mlmodel = self.converted_model
        return mlmodel


def parse_lut_arg(lut_value):
    if lut_value is None:
        return None, 8
    if isinstance(lut_value, int):
        return lut_value, 8
    lut_str = str(lut_value).strip().lower()
    if lut_str in ("none", "no", "false", ""):
        return None, 8
    if "," in lut_str:
        bits_str, per_channel_str = lut_str.split(",", 1)
        lut_bits = int(bits_str)
        per_channel = 0 if per_channel_str.strip().lower() in ("tensor", "t", "0") else int(per_channel_str)
        return lut_bits, per_channel
    return int(lut_str), 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Qwen3.5 model to CoreML format")
    parser.add_argument("--model", type=str, help="Path to model directory")
    parser.add_argument("--prefix", type=str, default="qwen35", help="Prefix for output filenames")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for prefill")
    parser.add_argument("--context-length", type=int, default=CONTEXT_LENGTH, help="Maximum context length")
    parser.add_argument("--lut", type=str, default=None, help='Use LUT quantization with "bits" or "bits,per_channel"')
    parser.add_argument("--chunk", type=int, default=None, help="Split FFN/prefill into N chunks")
    parser.add_argument(
        "--dynamic-prefill-slice",
        action="store_true",
        help="Use dynamic slicing for prefill KV writes (default ON for meta generation; no-op here).",
    )
    parser.add_argument(
        "--static-prefill-slice",
        action="store_true",
        help="Disable dynamic slicing for prefill KV writes (no-op here).",
    )
    parser.add_argument(
        "--part",
        type=str,
        choices=["1", "2", "2_prefill", "3", "all", "full", "prefill", "embeddings", "monolithic", "monolithic_prefill"],
        default="all",
        help="Model part to convert",
    )
    parser.add_argument("--output", type=str, default=".", help="Output directory")
    parser.add_argument("--argmax", action="store_true", help="Compute argmax inside LM head for inference models")
    return parser.parse_args()


def test_conversion(
    model: Optional[Qwen35ForCausalLM] = None,
    model_path: Optional[str] = None,
    prefix: str = "qwen35",
    context_length: int = CONTEXT_LENGTH,
    lut_bits: Optional[int] = None,
    batch_size: int = 64,
    output_dir: str = ".",
    part: str = "full",
    num_chunks: int = 1,
    per_channel: int = 8,
    argmax_in_model: bool = False,
):
    if model is None:
        if model_path is None:
            raise ValueError("model_path must be provided if model is None")
        config = Qwen35Config.from_json(os.path.join(model_path, "config.json"))
        config.context_length = context_length
        config.state_length = max(config.state_length, context_length)
        model = Qwen35ForCausalLM(config)
        ok = model.load_pretrained_weights(model_path)
        if not ok:
            raise RuntimeError(f"Failed to load Qwen3.5 weights from {model_path}")
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    effective_lut_bits = Qwen35Converter._effective_lut_bits_for_part(part, lut_bits)

    converter = Qwen35Converter(
        model=model,
        context_length=context_length,
        batch_size=batch_size,
        lut_bits=effective_lut_bits,
        num_chunks=num_chunks,
        per_channel=per_channel,
        argmax_in_model=argmax_in_model,
    )
    mlmodel = converter.convert(part=part)

    os.makedirs(output_dir, exist_ok=True)
    models = mlmodel if isinstance(mlmodel, list) else [mlmodel]
    vocab_size = int(getattr(model.config, "vocab_size", 0)) if model is not None else None
    lm_head_chunk_sizes = None
    if model is not None:
        lm_head_chunk_sizes = [
            int(getattr(model, f"lm_head16_{i+1}").out_channels)
            for i in range(model.lm_head_split)
        ]

    for i, m in enumerate(models):
        AddMetadata(
            m,
            {
                "context_length": context_length,
                "batch_size": batch_size if part in ["2_prefill", "prefill", "monolithic_prefill"] else None,
                "lut_bits": effective_lut_bits,
                "num_chunks": num_chunks if part in ["2", "2_prefill"] else None,
                "chunk_no": i + 1 if part in ["2", "2_prefill"] else None,
                "split_part": ModelPart.FULL.value if part in ["full", "all", "123"] else part,
                "argmax_in_model": argmax_in_model if part in ["3", "monolithic"] else None,
                "vocab_size": vocab_size if part in ["3", "monolithic"] else None,
                "lm_head_chunk_sizes": lm_head_chunk_sizes if part in ["3", "monolithic"] else None,
            },
        )
        fname = f"{prefix}"
        if part in ["1", "embeddings"]:
            fname += "_embeddings"
        elif part == "3":
            fname += "_lm_head"
        elif part == "monolithic":
            fname += "_monolithic"
        elif part == "monolithic_prefill":
            fname += "_monolithic_prefill"
        elif part in ["2", "2_prefill"]:
            base = "FFN" if part == "2" else "prefill"
            fname += f"_{base}"
            if effective_lut_bits is not None:
                fname += f"_lut{effective_lut_bits}"
            fname += f"_chunk_{i+1:02d}of{num_chunks:02d}"
        if part not in ["2", "2_prefill"] and effective_lut_bits is not None:
            fname += f"_lut{effective_lut_bits}"
        out_path = os.path.join(output_dir, f"{fname}.mlpackage")
        print(f"Saving model to: {out_path}")
        m.save(out_path)
    return mlmodel


def main() -> None:
    args = parse_args()
    lut_bits, per_channel = parse_lut_arg(args.lut)
    part_map = {"full": "all", "embeddings": "1", "prefill": "2_prefill"}
    part = part_map.get(args.part, args.part)
    test_conversion(
        model_path=args.model,
        prefix=args.prefix,
        context_length=args.context_length,
        lut_bits=lut_bits,
        batch_size=args.batch_size,
        output_dir=args.output,
        part=part,
        num_chunks=args.chunk or 1,
        per_channel=per_channel,
        argmax_in_model=args.argmax,
    )


if __name__ == "__main__":
    main()
