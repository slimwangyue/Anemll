#!/usr/bin/env python3
"""Diagnose batch prefill issue with the Chinese cooking prompt."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'scripts_qwen3_5'))

PROMPT = """作为一个人工智能模型，我虽然不能亲自下厨（毕竟没有手和厨房），但我**非常擅长"云做饭"**！我可以为你提供：

🍳 **家常菜指南**  
比如「番茄炒蛋」的万能公式：鸡蛋打散加盐 → 热锅倒油滑熟盛出 → 爆香蒜末后倒入番茄煮出汁 → 回锅混合加糖调味。你只需要按步骤操作即可～  

📖 **菜谱百科库**  
从粤式早茶到川味火锅，再到国际风味料理（如西班牙海鲜饭、意大利披萨），甚至冷门地方菜系都有详细教程。需要特定食材或低卡/高热量版本也可以调整哦！  

⏱️ **时间管理大师**  
- 15 分钟快手菜（适合上班族）  
- 周末慢炖硬菜（如红烧牛腩配土豆）  
- 一人食迷你版（避免浪费食物）  

💡 **小贴士升级**  
比如「如何判断鱼是否煎透」「如何让蔬菜更脆爽」「剩菜怎么变新花样」。  

你想尝试哪类菜肴？或者告诉我你的偏好（辣度/口味/食材限制），我来为你定制专属食谱！ 😋 翻译成英文"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", required=True)
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--ffn-dir", default=None)
parser.add_argument("--cpu-only", action="store_true")
parser.add_argument("--chunks", type=int, default=9)
parser.add_argument("--ctx", type=int, default=4096)
parser.add_argument("--sequential", action="store_true", help="Force sequential prefill instead of batch")
args = parser.parse_args()

from chat_server import ChatEngine
import coremltools as ct
import numpy as np

compute_unit = ct.ComputeUnit.CPU_ONLY if args.cpu_only else None

engine = ChatEngine(
    model_dir=args.model_dir,
    hf_path=args.tokenizer,
    ctx=args.ctx,
    num_chunks=args.chunks,
    ffn_dir=args.ffn_dir,
    compute_unit=compute_unit,
)
engine.load()

# Tokenize
messages = [{"role": "user", "content": PROMPT}]
tokens = engine._tokenize_messages(messages, enable_thinking=False)
bs = engine._prefill_bs

n_blocks = (len(tokens) + bs - 1) // bs
tail_len = len(tokens) % bs or bs
padding = bs - tail_len if tail_len < bs else 0

print(f"\n{'='*70}")
print(f"PROMPT: Chinese cooking text + '翻译成英文'")
print(f"Tokens: {len(tokens)}, batch_size={bs}")
print(f"Blocks: {n_blocks}, tail={tail_len}, padding={padding}")
print(f"{'='*70}")

# Reset
engine._reset_states()
engine.pos = 0
engine.rope_offset = 0
engine.token_history.clear()

if args.sequential:
    # Force sequential by disabling prefill
    print("\n[MODE] SEQUENTIAL prefill (forced)")
    orig_has_prefill = engine.has_prefill
    engine.has_prefill = False
    first_token = engine._process_prompt(tokens)
    engine.has_prefill = orig_has_prefill
else:
    print("\n[MODE] BATCH prefill")
    first_token = engine._process_prompt(tokens)

print(f"\nFirst token: {first_token}")
if first_token is None:
    print("ERROR: _process_prompt returned None")
    sys.exit(1)

# Decode up to 256 tokens
generated = [first_token]
current = first_token
for i in range(255):
    if engine.pos >= engine.ctx - 1:
        break
    if current in engine.stop_ids and i > 0:
        break
    next_tok, _ = engine._step(current, engine.pos)
    engine.pos += 1
    current = next_tok
    if current in engine.stop_ids:
        generated.append(current)
        break
    generated.append(current)

text = engine.tokenizer.decode(generated, skip_special_tokens=True)
hit_eos = generated[-1] in engine.stop_ids

print(f"\nGenerated: {len(generated)} tokens, EOS={hit_eos}")
print(f"\n{'='*70}")
print(f"OUTPUT:")
print(f"{'='*70}")
print(text)
print(f"{'='*70}")
