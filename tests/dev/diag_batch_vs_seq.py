#!/usr/bin/env python3
"""Quick test: compare batch vs sequential output for the Chinese cooking prompt."""
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

import argparse, numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", required=True)
parser.add_argument("--tokenizer", required=True)
parser.add_argument("--ffn-dir", default=None)
parser.add_argument("--chunks", type=int, default=9)
parser.add_argument("--ctx", type=int, default=4096)
args = parser.parse_args()

import coremltools as ct
from chat_server import ChatEngine

engine = ChatEngine(
    model_dir=args.model_dir,
    hf_path=args.tokenizer,
    ctx=args.ctx,
    num_chunks=args.chunks,
    ffn_dir=args.ffn_dir,
    compute_unit=ct.ComputeUnit.CPU_ONLY,
)
engine.load()

messages = [{"role": "user", "content": PROMPT}]
tokens = engine._tokenize_messages(messages, enable_thinking=False)
bs = engine._prefill_bs
print(f"\nTokens: {len(tokens)}, bs={bs}")
print(f"Blocks: {(len(tokens)+bs-1)//bs}, tail={len(tokens)%bs or bs}, padding={bs-(len(tokens)%bs or bs)}")

def run_and_decode(engine, tokens, mode, max_gen=64):
    engine._reset_states()
    engine.pos = 0
    engine.rope_offset = 0
    engine.token_history.clear()
    
    if mode == "sequential":
        orig = engine.has_prefill
        engine.has_prefill = False
        first_token = engine._process_prompt(tokens)
        engine.has_prefill = orig
    else:
        first_token = engine._process_prompt(tokens)
    
    if first_token is None:
        return "ERROR: None"
    
    generated = [first_token]
    current = first_token
    for i in range(max_gen - 1):
        if engine.pos >= engine.ctx - 1:
            break
        if current in engine.stop_ids and i > 0:
            break
        next_tok, _ = engine._step(current, engine.pos)
        engine.pos += 1
        current = next_tok
        if current in engine.stop_ids:
            break
        generated.append(current)
    
    return engine.tokenizer.decode(generated, skip_special_tokens=True)

# Also compare first token logits to see divergence
def get_first_token_info(engine, tokens, mode):
    engine._reset_states()
    engine.pos = 0
    engine.rope_offset = 0
    engine.token_history.clear()
    
    if mode == "sequential":
        orig = engine.has_prefill
        engine.has_prefill = False
        first_token = engine._process_prompt(tokens)
        engine.has_prefill = orig
    else:
        first_token = engine._process_prompt(tokens)
    
    return first_token

print(f"\n{'='*70}")
print("SEQUENTIAL PREFILL")
print(f"{'='*70}")
seq_tok = get_first_token_info(engine, tokens, "sequential")
seq_text = run_and_decode(engine, tokens, "sequential")
print(f"First token: {seq_tok} = '{engine.tokenizer.decode([seq_tok])}'")
print(f"Output: {seq_text[:300]}")

print(f"\n{'='*70}")
print("BATCH PREFILL")
print(f"{'='*70}")
batch_tok = get_first_token_info(engine, tokens, "batch")
batch_text = run_and_decode(engine, tokens, "batch")
print(f"First token: {batch_tok} = '{engine.tokenizer.decode([batch_tok])}'")
print(f"Output: {batch_text[:300]}")

print(f"\n{'='*70}")
print(f"COMPARISON: first_token seq={seq_tok} batch={batch_tok} match={seq_tok==batch_tok}")
print(f"{'='*70}")
