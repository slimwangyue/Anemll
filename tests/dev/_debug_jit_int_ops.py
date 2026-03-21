#!/usr/bin/env python3
"""Debug: trace FFNWrapper and search for aten::Int ops in JIT graph."""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")
import os
import torch
from anemll.models.qwen3_5_model import (
    Qwen35ForCausalLM, Qwen35Config, ane_conv_state_shape, MODEL_DTYPE,
)

MODEL_PATH = "/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B"
cfg = Qwen35Config.from_json(os.path.join(MODEL_PATH, "config.json"))
cfg.context_length = 256
cfg.state_length = 256
model = Qwen35ForCausalLM(cfg)
model.load_pretrained_weights(MODEL_PATH)
model.eval()
for p in model.parameters():
    p.requires_grad = False

conv_dim = (cfg.text_config.linear_num_key_heads * cfg.text_config.linear_key_head_dim * 2
            + cfg.text_config.linear_num_value_heads * cfg.text_config.linear_value_head_dim)
conv_kernel = max(1, int(cfg.text_config.linear_conv_kernel_dim))
d1, d2 = ane_conv_state_shape(conv_dim, conv_kernel)

class FFNWrapper(torch.nn.Module):
    def __init__(self, mdl, start, end):
        super().__init__()
        self.model = mdl
        self.start_layer = start
        self.end_layer = end
        c = mdl.config
        n = end - start
        self.k_cache = torch.zeros(n, c.num_key_value_heads, c.state_length, c.head_dim, dtype=MODEL_DTYPE)
        self.v_cache = torch.zeros(n, c.num_key_value_heads, c.state_length, c.head_dim, dtype=MODEL_DTYPE)
        self.linear_conv_state = torch.zeros(n, d1, d2, dtype=MODEL_DTYPE)
        self.linear_recurrent_state = torch.zeros(
            n, c.text_config.linear_num_value_heads,
            c.text_config.linear_key_head_dim, c.text_config.linear_value_head_dim, dtype=MODEL_DTYPE)

    def forward(self, hidden_states, position_ids, causal_mask, current_pos):
        return self.model.model.process_layers_regular_single_token_export_local_state(
            hidden_states=hidden_states, position_ids=position_ids,
            causal_mask=causal_mask, current_pos=current_pos,
            kv_cache_0=None, k_cache=self.k_cache, v_cache=self.v_cache,
            linear_conv_state=self.linear_conv_state,
            linear_recurrent_state=self.linear_recurrent_state,
            start_layer=0, end_layer=8, apply_final_norm=False,
        )

wrapper = FFNWrapper(model, 0, 8).eval()
h = torch.zeros((1, 1, cfg.hidden_size), dtype=torch.float16)
p = torch.zeros((1,), dtype=torch.int32)
m = torch.zeros((1, 1, 1, 256), dtype=torch.float16)
c = torch.zeros((1,), dtype=torch.int32)

print("Tracing...")
with torch.no_grad():
    traced = torch.jit.trace(wrapper, (h, p, m, c))

graph_str = str(traced.graph)
lines = graph_str.split('\n')
print(f"Total graph lines: {len(lines)}")

# Find all aten::Int lines
int_lines = []
for i, line in enumerate(lines):
    stripped = line.strip()
    if 'aten::Int' in stripped:
        int_lines.append((i, stripped))

print(f"Found {len(int_lines)} aten::Int ops:")
for idx, (lineno, line) in enumerate(int_lines):
    # Show context: 2 lines before
    for ctx in range(max(0, lineno-2), lineno):
        print(f"  ctx  L{ctx}: {lines[ctx].strip()[:200]}")
    print(f"  >>> L{lineno}: {line[:200]}")
    print()
