#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick end-to-end inference test for Qwen3.5-2B ctx4096 models."""
import coremltools as ct
import numpy as np
from transformers import AutoTokenizer

MODEL_DIR = "/Volumes/MySSD/Anemll/qwen3_5_2B_milestone_3.4_ctx4096_fkv"
COMBINED = f"{MODEL_DIR}/combined_LUT4_dedup"
NUM_CHUNKS = 7
CTX = 4096
BS = 256  # prefill batch size
HIDDEN = 2048
COMPUTE = ct.ComputeUnit.CPU_AND_NE  # Test on ANE like the app
MAX_TOKENS = 100

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
print("Tokenizer loaded")

# Load embed + lmhead (combined multi-function model)
print("Loading embed_lmhead_combined...")
embed = ct.models.MLModel(f"{MODEL_DIR}/embed_lmhead_combined.mlpackage",
                          function_name="embed", compute_units=ct.ComputeUnit.CPU_AND_NE)
embed_prefill = ct.models.MLModel(f"{MODEL_DIR}/embed_lmhead_combined.mlpackage",
                                   function_name="embed_prefill", compute_units=ct.ComputeUnit.CPU_AND_NE)
lmhead = ct.models.MLModel(f"{MODEL_DIR}/embed_lmhead_combined.mlpackage",
                            function_name="lmhead", compute_units=ct.ComputeUnit.CPU_AND_NE)
print("  embed/lmhead OK")

# Load infer + prefill chunks
print("Loading chunks...")
infer_chunks = []
prefill_chunks = []
for ci in range(NUM_CHUNKS):
    path = f"{COMBINED}/chunk{ci}.mlpackage"
    infer_chunks.append(ct.models.MLModel(path, function_name="infer", compute_units=ct.ComputeUnit.CPU_AND_NE))
    prefill_chunks.append(ct.models.MLModel(path, function_name="prefill", compute_units=ct.ComputeUnit.CPU_AND_NE))
    print(f"  chunk{ci} OK")

# Create states (one per chunk)
print("Creating states...")
states = [m.make_state() for m in infer_chunks]

# Detect linear state shapes from protobuf
from coremltools.proto import Model_pb2
def get_input_shapes(mlpackage_path, fn_name):
    with open(f"{mlpackage_path}/Data/com.apple.CoreML/model.mlmodel", 'rb') as f:
        spec = Model_pb2.Model()
        spec.ParseFromString(f.read())
    fn = spec.mlProgram.functions[fn_name]
    shapes = {}
    for inp in fn.inputs:
        if inp.type.HasField('tensorType'):
            tt = inp.type.tensorType
            dims = []
            for d in tt.dimensions:
                if d.HasField('constant'):
                    dims.append(d.constant.size)
            shapes[inp.name] = dims
    return shapes

shapes_all = [get_input_shapes(f"{COMBINED}/chunk{ci}.mlpackage", "infer") for ci in range(NUM_CHUNKS)]
for ci in range(NUM_CHUNKS):
    cs = shapes_all[ci].get("linear_conv_state", [3, 1024, 24])
    rs = shapes_all[ci].get("linear_recurrent_state", [3, 16, 128, 128])
    print(f"  chunk{ci}: conv={cs}, rec={rs}")

# Initialize linear states (per chunk, per-chunk shapes!)
conv_states = [np.zeros(shapes_all[ci].get("linear_conv_state", [3,1024,24]), dtype=np.float16) for ci in range(NUM_CHUNKS)]
rec_states = [np.zeros(shapes_all[ci].get("linear_recurrent_state", [3,16,128,128]), dtype=np.float16) for ci in range(NUM_CHUNKS)]

# Build prompt
messages = [
    {"role": "user", "content": "没问题，做红烧鱼（通常是鲤鱼或鲈鱼）。以下是红烧鱼的详细步骤：\n\n### 红烧鱼的做法\n\n#### 所需材料：\n- 鲤鱼或鲈鱼 1条（约500克）\n- 葱 2根\n- 姜 3片\n- 蒜 3瓣\n- 干辣椒 3-5个\n- 料酒 2勺\n- 生抽 3勺\n- 老抽 1勺\n- 糖 1勺\n- 盐 适量\n- 食用油 适量\n- 八角 2个\n- 花椒 少许\n\n#### 步骤：\n\n1. **处理鱼**\n   - 把鱼清洗干净，去掉内脏和鳞片\n   - 在鱼身两侧各划3-4刀（方便入味）\n   - 用少许盐和料酒腌制15分钟\n\n2. **煎鱼**\n   - 锅中倒入适量油，大火烧热\n   - 放入鱼煎至两面金黄（每面约3分钟）\n   - 煎好后取出备用\n\n3. **炒配料**\n   - 锅中留少许油\n   - 放入葱段、姜片、蒜末、干辣椒、八角、花椒\n   - 小火炒出香味\n\n4. **红烧**\n   - 加入料酒、生抽、老抽、糖\n   - 倒入适量热水（没过鱼身一半即可）\n   - 放入煎好的鱼\n   - 大火烧开后转中小火\n   - 盖锅盖焖煮15-20分钟\n\n5. **收汁**\n   - 打开锅盖，大火收汁\n   - 待汤汁浓稠后关火\n   - 撒上葱花即可\n\n请翻译以上内容为英文"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
token_ids = tokenizer.encode(prompt)
print(f"\nPrompt: {repr(prompt[:100])}...")
print(f"Token count: {len(token_ids)}")

# ---- PREFILL ----
print("\n--- PREFILL ---")
position = 0
first_token = None

# Process in blocks of BS
num_blocks = (len(token_ids) + BS - 1) // BS
for block_idx in range(num_blocks):
    start = block_idx * BS
    end = min(start + BS, len(token_ids))
    block_tokens = token_ids[start:end]
    valid_len = len(block_tokens)
    
    # Pad to BS
    padded = block_tokens + [0] * (BS - valid_len)
    input_ids = np.array([padded], dtype=np.int32)  # [1, BS]
    
    # Embed
    emb_out = embed_prefill.predict({"input_ids": input_ids})
    hidden = emb_out["hidden_states"]  # [1, BS, HIDDEN]
    
    # Causal mask [1, 1, BS, CTX]
    mask = np.full((1, 1, BS, CTX), -65504.0, dtype=np.float16)
    for i in range(valid_len):
        pos_i = start + i
        mask[0, 0, i, :pos_i + 1] = 0.0
    
    # Position IDs [BS]
    pos_ids = np.zeros(BS, dtype=np.int32)
    for i in range(valid_len):
        pos_ids[i] = start + i
    
    current_pos = np.array([start], dtype=np.int32)
    valid_len_arr = np.array([valid_len], dtype=np.int32)
    
    # Run through chunks
    for ci in range(NUM_CHUNKS):
        result = prefill_chunks[ci].predict(
            {
                "hidden_states": hidden,
                "position_ids": pos_ids,
                "causal_mask": mask,
                "current_pos": current_pos,
                "linear_conv_state": conv_states[ci],
                "linear_recurrent_state": rec_states[ci],
                "valid_len": valid_len_arr,
            },
            state=states[ci]
        )
        hidden = result["output_hidden_states"]
        if "linear_conv_state_out" in result:
            conv_states[ci] = result["linear_conv_state_out"]
        if "linear_recurrent_state_out" in result:
            rec_states[ci] = result["linear_recurrent_state_out"]
    
    # Extract last valid token hidden state
    print(f"  hidden shape after chunks: {np.array(hidden).shape}")
    hidden_np = np.array(hidden)
    if len(hidden_np.shape) >= 3 and hidden_np.shape[1] > 1:
        last_hidden = hidden_np[:, valid_len - 1:valid_len, :].copy()
    else:
        last_hidden = hidden_np
    print(f"  last_hidden shape: {last_hidden.shape}")
    
    # LM head
    lm_out = lmhead.predict({"hidden_states": last_hidden})
    logits = lm_out["logits"]
    first_token = int(np.argmax(logits.flat))
    
    position = start + valid_len
    print(f"  Block {block_idx}: pos={start}..{start+valid_len-1}, nextToken={first_token} ({repr(tokenizer.decode([first_token]))})")

# ---- DECODE ----
print(f"\n--- DECODE (max {MAX_TOKENS} tokens) ---")
generated = []
current_token = first_token
stop_ids = {248044, 248046}  # <|endoftext|>, <|im_end|>

for step in range(MAX_TOKENS):
    if current_token in stop_ids:
        print(f"  Stop token {current_token} at step {step}")
        break
    
    generated.append(current_token)
    
    # Embed single token
    tok_input = np.array([[current_token]], dtype=np.int32)  # [1, 1]
    emb_out = embed.predict({"input_ids": tok_input})
    hidden = emb_out["hidden_states"]  # [1, 1, HIDDEN]
    
    # Causal mask [1, 1, 1, CTX]
    mask = np.full((1, 1, 1, CTX), -65504.0, dtype=np.float16)
    mask[0, 0, 0, :position + 1] = 0.0
    
    pos_ids = np.array([position], dtype=np.int32)
    current_pos = np.array([position], dtype=np.int32)
    
    for ci in range(NUM_CHUNKS):
        result = infer_chunks[ci].predict(
            {
                "hidden_states": hidden,
                "position_ids": pos_ids,
                "causal_mask": mask,
                "current_pos": current_pos,
                "linear_conv_state": conv_states[ci],
                "linear_recurrent_state": rec_states[ci],
            },
            state=states[ci]
        )
        hidden = result["output_hidden_states"]
        if "linear_conv_state_out" in result:
            conv_states[ci] = result["linear_conv_state_out"]
        if "linear_recurrent_state_out" in result:
            rec_states[ci] = result["linear_recurrent_state_out"]
    
    lm_out = lmhead.predict({"hidden_states": hidden})
    logits = lm_out["logits"]
    current_token = int(np.argmax(logits.flat))
    position += 1
    
    if step < 5 or step % 10 == 0:
        print(f"  step={step} pos={position} token={current_token} ({repr(tokenizer.decode([current_token]))})")

output = tokenizer.decode(generated, skip_special_tokens=True)
print(f"\n=== OUTPUT ({len(generated)} tokens) ===")
print(output)
