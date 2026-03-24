#!/usr/bin/env python3
"""Analyze LUT4 lm_head group-size sweep results."""
import json, os

results_dir = '/tmp/lmhead_groupsize_sweep/results'
configs = ['LUT6_gs8_gpu', 'LUT4_gs1', 'LUT4_gs2', 'LUT4_gs4', 'LUT4_gs8']
sizes = {'LUT6_gs8_gpu': '462MB', 'LUT4_gs1': '311MB', 'LUT4_gs2': '321MB',
         'LUT4_gs4': '320MB', 'LUT4_gs8': '305MB'}

data = {}
for cfg in configs:
    path = os.path.join(results_dir, f'{cfg}.json')
    with open(path) as f:
        data[cfg] = json.load(f)

baseline = data['LUT6_gs8_gpu']
baseline_tokens = {t['turn']: t['tokens'] for t in baseline['results']}

print('=' * 80)
print('LUT4 LM HEAD GROUP-SIZE SWEEP RESULTS')
print('=' * 80)
print('Model: Qwen3.5-4B, CTX=1024, 3 conversation turns, 20 tokens each')
print('Compute: CPU_AND_GPU (ANE stuck from previous tests)')
print()

# Token match analysis
header = f"{'Config':>15} {'Size':>8} {'T1':>6} {'T2':>6} {'T3':>6} {'Avg':>7}"
print(header)
print('-' * len(header))

for cfg in configs:
    d = data[cfg]
    turn_matches = []
    for res in d['results']:
        turn = res['turn']
        tokens = res['tokens'][:20]
        base_tokens = baseline_tokens[turn][:20]
        match = sum(1 for a, b in zip(tokens, base_tokens) if a == b)
        pct = match / min(len(tokens), len(base_tokens)) * 100
        turn_matches.append(pct)

    overall = sum(turn_matches) / len(turn_matches)
    t1, t2, t3 = turn_matches
    tag = ' *BASE*' if cfg == 'LUT6_gs8_gpu' else ''
    print(f'{cfg:>15} {sizes[cfg]:>8} {t1:>5.0f}% {t2:>5.0f}% {t3:>5.0f}% {overall:>6.1f}%{tag}')

print()
print('Token-by-token comparison per turn:')
for turn_idx in range(3):
    print(f'\n--- Turn {turn_idx+1} ---')
    for cfg in configs:
        d = data[cfg]
        tokens = d['results'][turn_idx]['tokens'][:20]
        base = baseline_tokens[turn_idx + 1][:20]
        matches = sum(1 for a, b in zip(tokens, base) if a == b)
        first_diff = next((i for i, (a, b) in enumerate(zip(tokens, base)) if a != b), None)
        label = cfg.replace('LUT6_gs8_gpu', 'LUT6(base)')
        if first_diff is None:
            print(f'  {label:>12}: all 20 match')
        else:
            print(f'  {label:>12}: first diff at pos {first_diff} (base={base[first_diff]} vs {tokens[first_diff]}), {matches}/20 match')

print()
print('Text comparison:')
for turn_idx in range(3):
    print(f'\n--- Turn {turn_idx+1} ---')
    for cfg in configs:
        d = data[cfg]
        text = d['results'][turn_idx]['text'][:80]
        label = cfg.replace('LUT6_gs8_gpu', 'LUT6(base)')
        print(f'  {label:>12}: {text}')

print()
print('Model sizes:')
for cfg in configs:
    tag = ' (baseline)' if 'LUT6' in cfg else ''
    print(f'  {cfg:>15}: {sizes[cfg]}{tag}')
print()
savings = (462 - 305) / 462 * 100
print(f'Max savings: LUT4_gs8 = {savings:.0f}% smaller than LUT6_gs8')
savings_gs1 = (462 - 311) / 462 * 100
print(f'LUT4_gs1 savings: {savings_gs1:.0f}% smaller than LUT6_gs8')
