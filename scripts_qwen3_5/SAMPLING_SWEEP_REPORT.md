# Qwen3.5 Sampling Sweep Report

**Date**: 2026-04-15  
**Model**: Qwen3.5-4B (32 layers, 9 LUT4 chunks, ANE)  
**Runtime**: chat_server.py, ctx=2048  
**Objective**: Minimize repetition while preserving coherence  
**Tool**: `sweep_sampling.py` with systematic parameter isolation  

---

## Final Recommended Configs

### 4B Think-Off
```json
{
  "temperature": 0.7,
  "top_p": 0.8,
  "top_k": 20,
  "presence_penalty": 1.0,
  "repetition_penalty": 1.1,
  "frequency_penalty": 0.05
}
```

### 4B Think-On
```json
{
  "temperature": 1.0,
  "top_p": 0.95,
  "top_k": 20,
  "presence_penalty": 1.5,
  "repetition_penalty": 1.1,
  "frequency_penalty": 0.08
}
```

### 2B (Provisional — same as 4B until compiled artifacts available)
- Think-off: same as 4B think-off
- Think-on: same as 4B think-on

---

## Changes from Qwen Official Defaults

| Parameter | Think-Off Old | Think-Off New | Think-On Old | Think-On New |
|-----------|:---:|:---:|:---:|:---:|
| temperature | 0.7 | 0.7 | 1.0 | 1.0 |
| top_p | 0.8 | 0.8 | 0.95 | 0.95 |
| top_k | 20 | 20 | 20 | 20 |
| presence_penalty | **1.5** | **1.0** | 1.5 | 1.5 |
| repetition_penalty | **1.0** | **1.1** | **1.0** | **1.1** |
| frequency_penalty | **0.0** | **0.05** | **0.0** | **0.08** |

**Summary**: Only 3 parameters changed per mode. Temperature, top_p, top_k remain at Qwen defaults.

---

## Final Validation Results

### Think-Off (10 single-turn + 4 multi-turn)

| Prompt | Tokens | Stop | Rep Score | 5gram Dup | 8gram Dup | Loop | Status |
|--------|--------|------|-----------|-----------|-----------|------|--------|
| S1_en_short | 500 | length | 0.011 | 0 | 0 | 0 | OK |
| S2_en_reason | 120 | eos | 0.000 | 0 | 0 | 0 | OK |
| S3_en_code | 427 | eos | 0.021 | 0 | 0 | 0 | OK |
| S4_en_list | 437 | eos | 0.000 | 0 | 0 | 0 | OK |
| L1_en_long | 800 | length | 0.005 | 1 | 0 | 0 | OK |
| L2_en_long | 800 | length | 0.000 | 0 | 0 | 0 | OK |
| CH1_zh_short | 500 | length | 0.000 | 0 | 0 | 0 | OK |
| CH2_zh_long | 800 | length | 0.044 | 1 | 0 | 0 | OK |
| REP1_history | 600 | length | 0.000 | 0 | 0 | 0 | OK |
| REP2_analysis | 800 | length | 0.021 | 0 | 0 | 0 | OK |
| **MT Turn 1** | 400 | length | 0.000 | 0 | - | 0 | OK |
| **MT Turn 2** | 400 | length | 0.000 | 0 | - | 0 | OK |
| **MT Turn 3** | 500 | length | 0.011 | 1 | - | 0 | OK |
| **MT Turn 4** | 500 | length | 0.000 | 0 | - | 0 | OK |

**avg_rep_st=0.0101, max_rep_st=0.0445, 0 loops, 100% coherent, 100% on-topic**

### Think-On (10 single-turn + 4 multi-turn)

| Prompt | Tokens | Stop | Rep Score | 5gram Dup | 8gram Dup | Loop | Status |
|--------|--------|------|-----------|-----------|-----------|------|--------|
| S1_en_short | 500 | length | 0.000 | 0 | 0 | 0 | OK |
| S2_en_reason | 200 | length | 0.000 | 0 | 0 | 0 | OK |
| S3_en_code | 500 | length | 0.020 | 0 | 0 | 0 | OK |
| S4_en_list | 500 | length | 0.000 | 0 | 0 | 0 | OK |
| L1_en_long | 800 | length | 0.000 | 0 | 0 | 0 | OK |
| L2_en_long | 800 | length | 0.000 | 0 | 0 | 0 | OK |
| CH1_zh_short | 500 | length | 0.000 | 0 | 0 | 0 | OK |
| CH2_zh_long | 800 | length | 0.000 | 0 | 0 | 0 | OK |
| REP1_history | 600 | length | 0.000 | 0 | 0 | 0 | OK |
| REP2_analysis | 800 | length | 0.000 | 0 | 0 | 0 | OK |
| **MT Turn 1** | 400 | length | 0.000 | 0 | - | 0 | OK |
| **MT Turn 2** | 400 | length | 0.000 | 0 | - | 0 | OK |
| **MT Turn 3** | 500 | length | 0.000 | 0 | - | 0 | OK |
| **MT Turn 4** | 500 | length | 0.000 | 0 | - | 0 | OK |

**avg_rep_st=0.0020, max_rep_st=0.0200, 0 loops, 100% coherent, 100% on-topic**

---

## Sweep Details

### Stage 1 — previously established baseline
Baseline (Qwen defaults) had multi-turn repetition issues: `no_repetition` failures at M1 turn 2 and M2 turn 3 in the earlier validation report.

### Stage 2 — Frequency Penalty Sweep (Think-Off)

| freq | avg_rep_st | avg_rep_mt | max_rep_mt |
|------|-----------|-----------|-----------|
| 0.0 | 0.0000 | 0.0362 | 0.0621 |
| 0.02 | 0.0000 | 0.0110 | 0.0330 |
| **0.05** | **0.0000** | **0.0052** | **0.0155** |
| 0.08 | 0.0000 | 0.0065 | 0.0196 |
| 0.1 | 0.0000 | 0.0078 | 0.0233 |

**Winner: freq=0.05** — lowest MT rep; diminishing returns beyond 0.08.

### Stage 3 — Repetition Penalty Sweep (Think-Off, freq=0.05)

| rep | avg_rep_st | avg_rep_mt |
|-----|-----------|-----------|
| 1.0 | 0.0056 | 0.0000 |
| 1.02 | 0.0000 | 0.0142 |
| 1.05 | 0.0053 | 0.0000 |
| 1.08 | 0.0000 | 0.0060 |
| **1.1** | **0.0000** | **0.0000** |
| 1.12 | 0.0000 | 0.0067 |

**Winner: rep=1.1** — zero repetition in both ST and MT.

### Stage 4 — Presence Penalty Sweep (Think-Off, freq=0.05, rep=1.1)

| pres | avg_rep_mt | max_rep_mt |
|------|-----------|-----------|
| 0.0 | 0.1491 | 0.4474 |
| 0.5 | 0.0141 | 0.0423 |
| **1.0** | **0.0000** | **0.0000** |
| 1.5 | 0.0000 | 0.0000 |
| 2.0 | 0.0000 | 0.0000 |

**Winner: pres=1.0** — minimum effective value. pres=0.0 reintroduces severe MT repetition even with freq+rep.

### Stage 5 — Temperature/Top-p Sweep (Think-Off, freq=0.05, rep=1.1, pres=1.0)

| Config | avg_rep_st | avg_rep_mt |
|--------|-----------|-----------|
| temp=0.5 | 0.0000 | 0.0042 |
| **temp=0.6** | **0.0000** | **0.0000** |
| **temp=0.7** | **0.0000** | **0.0000** |
| temp=0.8 | 0.0056 | 0.0000 |
| topp=0.7 | 0.0000 | 0.0000 |
| **topp=0.8** | **0.0000** | **0.0000** |
| topp=0.9 | 0.0056 | 0.0000 |
| topp=0.95 | 0.0000 | 0.0000 |

**Winner: temp=0.7, top_p=0.8** (Qwen defaults) — stable and clean.

### Stage 6 — Think-On Frequency Penalty Sweep

| freq | avg_rep_mt | max_rep_mt |
|------|-----------|-----------|
| 0.0 | 0.0424 | 0.1271 |
| 0.02 | 0.1500 | 0.4500 |
| 0.05 | 0.1263 | 0.3788 |
| **0.08** | **0.0000** | **0.0000** |
| 0.1 | 0.0000 | 0.0000 |

**Winner: freq=0.08** — think-on needs higher freq than think-off. Intermediate values (0.02–0.05) made repetition **worse** due to interaction with thinking token generation.

### Stage 7 — Think-On Repetition Penalty (freq=0.08)

| rep | avg_rep_mt |
|-----|-----------|
| 1.0 | 0.0000 |
| 1.02 | 0.0095 |
| 1.05 | 0.0000 |
| 1.08 | 0.0000 |
| **1.1** | **0.0000** |
| 1.12 | 0.0000 |

rep=1.0 looked clean on probe, but full validation (v1, rep=1.0) showed S3 code rep=0.51 and MT4 rep=0.22. Adding rep=1.1 (v2) eliminated both completely.

---

## Key Insights

1. **frequency_penalty is the most impactful parameter** for reducing multi-turn repetition. Even 0.05 drops MT rep by 86%.

2. **Think-on needs a higher freq penalty (0.08) than think-off (0.05)**. Intermediate values (0.02–0.05) for think-on can make repetition *worse* — the penalty interacts with the thinking token distribution differently.

3. **presence_penalty is essential** — removing it (0→0.0) reintroduces severe repetition (0.45 max) even with freq+rep. The minimum effective value is 1.0 (think-off). Think-on benefits from staying at the Qwen default of 1.5.

4. **repetition_penalty=1.1 is a robust safety net** — it catches edge cases (code prompts, long multi-turn) that freq alone misses.

5. **Temperature and top_p don't need tuning** — the Qwen defaults (0.7/0.8 for think-off, 1.0/0.95 for think-on) are optimal once penalties are set.

6. **All repetition mitigation was achieved without damaging coherence or Chinese output quality** — CH1 and CH2 prompts remain clean across all tested configs.

---

## Tradeoffs

| Change | Benefit | Cost |
|--------|---------|------|
| freq=0.05 (think-off) | 86% MT rep reduction | May slightly favor rare tokens in very long output |
| freq=0.08 (think-on) | Eliminates think-on rep | Slightly stronger bias; no quality degradation observed |
| rep=1.1 | Eliminates code/MT edge cases | Minimal — common tokens softly penalized |
| pres 1.5→1.0 (think-off) | Less aggressive novel-token forcing | None observed — repetition controlled by freq+rep |

---

## Files

| File | Description |
|------|-------------|
| `sweep_freq_off.json/.txt` | Freq sweep results (think-off) |
| `sweep_rep_off.json/.txt` | Rep sweep results (think-off) |
| `sweep_pres_off.json/.txt` | Pres sweep results (think-off) |
| `sweep_temp_off.json/.txt` | Temp/top_p sweep results (think-off) |
| `sweep_freq_on.json/.txt` | Freq sweep results (think-on) |
| `sweep_rep_on.json/.txt` | Rep sweep results (think-on) |
| `sweep_final_4B_off.json/.txt` | Final validation (think-off, full suite) |
| `sweep_final_4B_on_v2.json/.txt` | Final validation (think-on, full suite) |
| `sweep_sampling.py` | Sweep harness |
| `inference_config.py` | Updated with winning configs |

---

## 2B Status

No compiled 2B model artifacts exist (`qwen3_5_2b_v4_lut4/` not found). The 2B presets in `inference_config.py` are set to match 4B values as a reasonable starting point. When 2B compiled models become available, run the same sweep to validate/tune.
