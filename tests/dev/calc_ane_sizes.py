"""Calculate state tensor and model sizes for ANE compatibility analysis."""

num_hidden_layers = 32
num_key_value_heads = 8
state_length = 1024
head_dim = 256
hidden_size = 2560

linear_num_key_heads = 2
linear_key_head_dim = 256
linear_num_value_heads = 2
linear_value_head_dim = 256
linear_conv_kernel_dim = 4

conv_dim = (linear_num_key_heads * linear_key_head_dim * 2
            + linear_num_value_heads * linear_value_head_dim)

for num_chunks in [4, 6, 8]:
    base, rem = divmod(num_hidden_layers, num_chunks)
    print(f"\n=== {num_chunks} chunks ===")
    for ci in range(num_chunks):
        start = ci * base + min(ci, rem)
        end = start + base + (1 if ci < rem else 0)
        local_layers = end - start

        full_attn = sum(1 for l in range(start, end) if (l + 1) % 4 == 0)
        linear = local_layers - full_attn

        # KV cache (k + v)
        kv_bytes = 2 * local_layers * num_key_value_heads * state_length * head_dim * 2
        kv_mb = kv_bytes / (1024 * 1024)

        # Conv state
        conv_bytes = local_layers * conv_dim * linear_conv_kernel_dim * 2
        conv_mb = conv_bytes / (1024 * 1024)

        # Recurrent state
        rec_bytes = local_layers * linear_num_value_heads * linear_key_head_dim * linear_value_head_dim * 2
        rec_mb = rec_bytes / (1024 * 1024)

        total_mb = kv_mb + conv_mb + rec_mb

        print(f"  Chunk {ci}: layers {start}-{end-1} ({local_layers}L, {full_attn} full_attn, {linear} linear)")
        print(f"    KV cache: {kv_mb:.1f} MB | Conv: {conv_mb:.2f} MB | Rec: {rec_mb:.1f} MB | Total state: {total_mb:.1f} MB")

print("\n=== Per-model weight sizes (approx, LUT4) ===")
# Each layer: q_proj + k_proj + v_proj + o_proj + gate_proj + up_proj + down_proj
# Full attn: q=2560->2048, k=2560->2048, v=2560->2048, o=2048->2560, gate=2560->8960, up=2560->8960, down=8960->2560
# Linear attn: similar but different head configs
params_per_layer = (2560 * 2048 * 4 + 2560 * 8960 * 3)  # approximate
for num_chunks in [4, 6, 8]:
    base, rem = divmod(num_hidden_layers, num_chunks)
    print(f"\n--- {num_chunks} chunks ---")
    for ci in range(num_chunks):
        start = ci * base + min(ci, rem)
        end = start + base + (1 if ci < rem else 0)
        local_layers = end - start
        weight_bytes_fp16 = local_layers * params_per_layer * 2
        weight_bytes_lut4 = weight_bytes_fp16 / 4  # ~4x compression
        print(f"  Chunk {ci}: {local_layers} layers, ~{weight_bytes_lut4/(1024*1024):.0f} MB weights (LUT4)")

print("\n=== KV cache state tensor dimensions ===")
for num_chunks in [4, 6, 8]:
    base, rem = divmod(num_hidden_layers, num_chunks)
    print(f"\n--- {num_chunks} chunks ---")
    for ci in range(num_chunks):
        start = ci * base + min(ci, rem)
        end = start + base + (1 if ci < rem else 0)
        local_layers = end - start
        print(f"  Chunk {ci}: k_cache shape=({local_layers}, {num_key_value_heads}, {state_length}, {head_dim})")
        elements = local_layers * num_key_value_heads * state_length * head_dim
        print(f"           elements per cache: {elements:,} ({elements*2/(1024*1024):.1f} MB)")

print("\n=== ANE conv_state reshaping ===")
ANE_MAX = 1024
if conv_dim > ANE_MAX:
    group = (conv_dim + ANE_MAX - 1) // ANE_MAX
    ane_dim1 = conv_dim // group
    ane_dim2 = linear_conv_kernel_dim * group
    print(f"  conv_dim={conv_dim} > {ANE_MAX}, group={group}")
    print(f"  Reshaped: ({ane_dim1}, {ane_dim2}) per layer")
else:
    print(f"  conv_dim={conv_dim} <= {ANE_MAX}, no reshape needed")
