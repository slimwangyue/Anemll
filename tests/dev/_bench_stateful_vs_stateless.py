#!/usr/bin/env python3
"""Benchmark: Stateful (ct.StateType) vs Stateless (input/output tensors) for
linear attention recurrent state.

Linear attention state is fixed-size:
  - recurrent_state: (num_layers_in_chunk, 4, 128, 128) per chunk
  - conv_state: (num_layers_in_chunk, ane_dim1, ane_dim2) per chunk

Unlike full-attention KV cache which grows with context, these are constant.
So we can pass them as regular inputs and outputs instead of CoreML state.

Benefits of stateless:
  - Can use fp32 dtype for inputs/outputs (CoreML state is fp16-only)
  - No MIL optimizer defeating compensation patterns
  - Simpler debugging and state inspection
  - More portable (no iOS 18 state requirement)

Cost: additional I/O bandwidth per inference step.

This test measures the latency overhead of passing state as I/O vs CoreML state,
at realistic dimensions for Qwen3.5-4B chunks.
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os, gc, time, tempfile
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct

TMPDIR = tempfile.mkdtemp(prefix="stateful_vs_stateless_")

# Qwen3.5-4B dims
NUM_V_HEADS = 4
HEAD_K_DIM = 128
HEAD_V_DIM = 128
HIDDEN_DIM = 2560
CTX = 256

# Chunk configs to test (layers_per_chunk)
CHUNK_CONFIGS = [6, 8, 12, 24]  # 24 linear layers / 4 chunks = 6, or 3 chunks = 8, etc.

# conv state per layer
CONV_DIM = 1024  # ~(num_k_heads * key_dim * 2 + num_v_heads * value_dim)
CONV_KERNEL = 4
# ANE-safe reshape: pad to multiples
ANE_DIM1 = 1024
ANE_DIM2 = 32  # from ane_conv_state_shape

NUM_WARMUP = 5
NUM_ITERS = 50


def _l2norm(x):
    return x / (x.norm(dim=-1, keepdim=True).clamp(min=1e-4))


class SimpleRecurrence(nn.Module):
    """Minimal recurrence block for benchmarking."""
    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
    
    def _one_layer(self, q, k, v, g, beta, state):
        q = _l2norm(q) * (1.0 / (HEAD_K_DIM ** 0.5))
        k = _l2norm(k)
        g_t = g.exp().unsqueeze(-1).unsqueeze(-1)
        state = state * g_t
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        out = (state * q.unsqueeze(-1)).sum(dim=-2)
        return out, state

    def _pad_to_hidden(self, layer_out):
        """Pad layer output (1, NUM_V_HEADS*HEAD_V_DIM) to (1, HIDDEN_DIM).
        Uses static padding size to avoid aten::Int from shape access."""
        flat = layer_out.reshape(1, NUM_V_HEADS * HEAD_V_DIM)
        # Static pad: HIDDEN_DIM - NUM_V_HEADS*HEAD_V_DIM = 2560 - 512 = 2048
        pad = torch.zeros(1, HIDDEN_DIM - NUM_V_HEADS * HEAD_V_DIM,
                         dtype=flat.dtype, device=flat.device)
        return torch.cat([flat, pad], dim=1)


class StatefulModel(SimpleRecurrence):
    """Recurrent state stored as CoreML StateType."""
    def __init__(self, num_layers):
        super().__init__(num_layers)
        self.register_buffer('recurrent_state', torch.zeros(
            num_layers, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16))
        self.register_buffer('conv_state', torch.zeros(
            num_layers, ANE_DIM1, ANE_DIM2, dtype=torch.float16))
    
    def forward(self, hidden_states):
        out = hidden_states
        for i in range(self.num_layers):
            q = out[:, :NUM_V_HEADS * HEAD_K_DIM].view(1, NUM_V_HEADS, HEAD_K_DIM)
            k = out[:, :NUM_V_HEADS * HEAD_K_DIM].view(1, NUM_V_HEADS, HEAD_K_DIM)
            v = out[:, :NUM_V_HEADS * HEAD_V_DIM].view(1, NUM_V_HEADS, HEAD_V_DIM)
            g = out[:, :NUM_V_HEADS].view(1, NUM_V_HEADS)
            beta = out[:, :NUM_V_HEADS].view(1, NUM_V_HEADS).sigmoid()
            
            layer_state = self.recurrent_state[i:i+1]
            layer_out, new_state = self._one_layer(q, k, v, g, beta, layer_state)
            self.recurrent_state[i:i+1] = new_state
            
            cs = self.conv_state[i:i+1]
            self.conv_state[i:i+1] = cs
            
            out = out + self._pad_to_hidden(layer_out)
        
        return out


class StatelessModel(SimpleRecurrence):
    """Recurrent state passed as input/output tensors (no StateType)."""
    def __init__(self, num_layers):
        super().__init__(num_layers)
    
    def forward(self, hidden_states, recurrent_state_in, conv_state_in):
        out = hidden_states
        recurrent_state = recurrent_state_in
        conv_state = conv_state_in
        
        for i in range(self.num_layers):
            q = out[:, :NUM_V_HEADS * HEAD_K_DIM].view(1, NUM_V_HEADS, HEAD_K_DIM)
            k = out[:, :NUM_V_HEADS * HEAD_K_DIM].view(1, NUM_V_HEADS, HEAD_K_DIM)
            v = out[:, :NUM_V_HEADS * HEAD_V_DIM].view(1, NUM_V_HEADS, HEAD_V_DIM)
            g = out[:, :NUM_V_HEADS].view(1, NUM_V_HEADS)
            beta = out[:, :NUM_V_HEADS].view(1, NUM_V_HEADS).sigmoid()
            
            layer_state = recurrent_state[i:i+1]
            layer_out, new_state = self._one_layer(q, k, v, g, beta, layer_state)
            recurrent_state = torch.cat([
                recurrent_state[:i], new_state, recurrent_state[i+1:]
            ], dim=0) if self.num_layers > 1 else new_state
            
            # Touch conv_state (identity)
            cs = conv_state[i:i+1]
            conv_state = torch.cat([
                conv_state[:i], cs, conv_state[i+1:]
            ], dim=0) if self.num_layers > 1 else cs
            
            out = out + self._pad_to_hidden(layer_out)
        
        return out, recurrent_state, conv_state


class StatelessSliceModel(SimpleRecurrence):
    """Stateless but avoids torch.cat — uses slice assignment on clones."""
    def __init__(self, num_layers):
        super().__init__(num_layers)
    
    def forward(self, hidden_states, recurrent_state_in, conv_state_in):
        out = hidden_states
        recurrent_state = recurrent_state_in.clone()
        conv_state = conv_state_in.clone()
        
        for i in range(self.num_layers):
            q = out[:, :NUM_V_HEADS * HEAD_K_DIM].view(1, NUM_V_HEADS, HEAD_K_DIM)
            k = out[:, :NUM_V_HEADS * HEAD_K_DIM].view(1, NUM_V_HEADS, HEAD_K_DIM)
            v = out[:, :NUM_V_HEADS * HEAD_V_DIM].view(1, NUM_V_HEADS, HEAD_V_DIM)
            g = out[:, :NUM_V_HEADS].view(1, NUM_V_HEADS)
            beta = out[:, :NUM_V_HEADS].view(1, NUM_V_HEADS).sigmoid()
            
            layer_state = recurrent_state[i:i+1]
            layer_out, new_state = self._one_layer(q, k, v, g, beta, layer_state)
            recurrent_state[i:i+1] = new_state
            conv_state[i:i+1] = conv_state[i:i+1]  # identity
            
            out = out + self._pad_to_hidden(layer_out)
        
        return out, recurrent_state, conv_state


def export_stateful(model, name, num_layers):
    model.eval()
    h = torch.zeros(1, HIDDEN_DIM, dtype=torch.float16)
    with torch.no_grad():
        traced = torch.jit.trace(model, h, check_trace=False)
    
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(num_layers, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), dtype=np.float16),
            name="recurrent_state"),
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(num_layers, ANE_DIM1, ANE_DIM2), dtype=np.float16),
            name="conv_state"),
    ]
    
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="hidden_states", shape=(1, HIDDEN_DIM), dtype=np.float16)],
        outputs=[ct.TensorType(name="output", dtype=np.float16)],
        states=states,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    path = os.path.join(TMPDIR, f"{name}.mlpackage")
    mlmodel.save(path)
    del mlmodel; gc.collect()
    return path


def export_stateless(model, name, num_layers, dtype=np.float16):
    model.eval()
    h = torch.zeros(1, HIDDEN_DIM, dtype=torch.float16)
    rec = torch.zeros(num_layers, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM, dtype=torch.float16)
    conv = torch.zeros(num_layers, ANE_DIM1, ANE_DIM2, dtype=torch.float16)
    
    with torch.no_grad():
        traced = torch.jit.trace(model, (h, rec, conv), check_trace=False)
    
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="hidden_states", shape=(1, HIDDEN_DIM), dtype=np.float16),
            ct.TensorType(name="recurrent_state_in",
                         shape=(num_layers, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), dtype=dtype),
            ct.TensorType(name="conv_state_in",
                         shape=(num_layers, ANE_DIM1, ANE_DIM2), dtype=dtype),
        ],
        outputs=[
            ct.TensorType(name="output", dtype=np.float16),
            ct.TensorType(name="recurrent_state_out", dtype=dtype),
            ct.TensorType(name="conv_state_out", dtype=dtype),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
    )
    path = os.path.join(TMPDIR, f"{name}.mlpackage")
    mlmodel.save(path)
    del mlmodel; gc.collect()
    return path


def benchmark_stateful(path, num_layers, compute_units):
    cml = ct.models.MLModel(path, compute_units=compute_units)
    state_obj = cml.make_state()
    h = np.zeros((1, HIDDEN_DIM), dtype=np.float16)
    
    # Warmup
    for _ in range(NUM_WARMUP):
        cml.predict({"hidden_states": h}, state=state_obj)
    
    # Measure
    times = []
    for _ in range(NUM_ITERS):
        t0 = time.perf_counter()
        cml.predict({"hidden_states": h}, state=state_obj)
        times.append(time.perf_counter() - t0)
    
    del cml, state_obj; gc.collect()
    return np.array(times) * 1000  # ms


def benchmark_stateless(path, num_layers, compute_units, dtype=np.float16):
    cml = ct.models.MLModel(path, compute_units=compute_units)
    h = np.zeros((1, HIDDEN_DIM), dtype=np.float16)
    rec = np.zeros((num_layers, NUM_V_HEADS, HEAD_K_DIM, HEAD_V_DIM), dtype=dtype)
    conv = np.zeros((num_layers, ANE_DIM1, ANE_DIM2), dtype=dtype)
    
    # Warmup
    for _ in range(NUM_WARMUP):
        out = cml.predict({
            "hidden_states": h,
            "recurrent_state_in": rec,
            "conv_state_in": conv,
        })
        rec = out["recurrent_state_out"]
        conv = out["conv_state_out"]
    
    # Measure
    times = []
    for _ in range(NUM_ITERS):
        t0 = time.perf_counter()
        out = cml.predict({
            "hidden_states": h,
            "recurrent_state_in": rec,
            "conv_state_in": conv,
        })
        rec = out["recurrent_state_out"]
        conv = out["conv_state_out"]
        times.append(time.perf_counter() - t0)
    
    del cml; gc.collect()
    return np.array(times) * 1000  # ms


def size_in_mb(num_layers, dtype_bytes=2):
    """Calculate state I/O size in MB."""
    rec_size = num_layers * NUM_V_HEADS * HEAD_K_DIM * HEAD_V_DIM * dtype_bytes
    conv_size = num_layers * ANE_DIM1 * ANE_DIM2 * dtype_bytes
    return (rec_size + conv_size) / (1024 * 1024)


if __name__ == "__main__":
    print(f"Temp: {TMPDIR}")
    print(f"Stateful vs Stateless Benchmark")
    print(f"recurrent_state per layer: ({NUM_V_HEADS}, {HEAD_K_DIM}, {HEAD_V_DIM}) = {NUM_V_HEADS*HEAD_K_DIM*HEAD_V_DIM} elements")
    print(f"conv_state per layer: ({ANE_DIM1}, {ANE_DIM2}) = {ANE_DIM1*ANE_DIM2} elements")
    print(f"Warmup={NUM_WARMUP}, Iters={NUM_ITERS}")
    print()
    
    # Size estimates
    print(f"{'Layers':>6} {'State MB (fp16)':>15} {'State MB (fp32)':>15}")
    print("-" * 40)
    for nl in CHUNK_CONFIGS:
        print(f"{nl:6d} {size_in_mb(nl, 2):15.2f} {size_in_mb(nl, 4):15.2f}")
    print()
    
    # Test with a moderate chunk size
    test_layers = 8  # typical chunk
    
    print(f"=== Testing with {test_layers} layers ===")
    print(f"State size: {size_in_mb(test_layers):.2f} MB (fp16), {size_in_mb(test_layers, 4):.2f} MB (fp32)")
    print()
    
    # Export models
    print("Exporting stateful model...")
    sf_model = StatefulModel(test_layers)
    sf_path = export_stateful(sf_model, "stateful", test_layers)
    del sf_model
    
    print("Exporting stateless model (slice-assign)...")
    sl_model = StatelessSliceModel(test_layers)
    sl_path = export_stateless(sl_model, "stateless_fp16", test_layers, np.float16)
    del sl_model
    
    # Try fp32 stateless
    print("Exporting stateless fp32 model...")
    sl32_model = StatelessSliceModel(test_layers)
    try:
        sl32_path = export_stateless(sl32_model, "stateless_fp32", test_layers, np.float32)
        sl32_ok = True
    except Exception as e:
        print(f"  fp32 export failed: {e}")
        sl32_path = None
        sl32_ok = False
    del sl32_model
    
    # Benchmark on CPU_AND_NE
    for cu_name, cu in [("CPU_AND_NE", ct.ComputeUnit.CPU_AND_NE), ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY)]:
        print(f"\n--- Compute: {cu_name} ---")
        
        print(f"  Benchmarking stateful...")
        sf_times = benchmark_stateful(sf_path, test_layers, cu)
        
        print(f"  Benchmarking stateless fp16...")
        sl_times = benchmark_stateless(sl_path, test_layers, cu, np.float16)
        
        sl32_times = None
        if sl32_ok:
            print(f"  Benchmarking stateless fp32...")
            sl32_times = benchmark_stateless(sl32_path, test_layers, cu, np.float32)
        
        print(f"\n  {'Method':<25} {'Median ms':>10} {'Mean ms':>10} {'P95 ms':>10} {'Overhead':>10}")
        print(f"  {'-'*70}")
        
        sf_med = np.median(sf_times)
        print(f"  {'stateful':<25} {sf_med:10.3f} {np.mean(sf_times):10.3f} {np.percentile(sf_times, 95):10.3f} {'baseline':>10}")
        
        sl_med = np.median(sl_times)
        overhead_pct = (sl_med - sf_med) / sf_med * 100
        print(f"  {'stateless fp16':<25} {sl_med:10.3f} {np.mean(sl_times):10.3f} {np.percentile(sl_times, 95):10.3f} {overhead_pct:+9.1f}%")
        
        if sl32_times is not None:
            sl32_med = np.median(sl32_times)
            overhead_pct32 = (sl32_med - sf_med) / sf_med * 100
            print(f"  {'stateless fp32':<25} {sl32_med:10.3f} {np.mean(sl32_times):10.3f} {np.percentile(sl32_times, 95):10.3f} {overhead_pct32:+9.1f}%")
    
    print(f"\nCleanup: rm -rf {TMPDIR}")
