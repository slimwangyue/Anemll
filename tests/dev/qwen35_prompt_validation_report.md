# Qwen3.5 Prompt-Level Validation (ANEMLL vs HF)

- model_path: `/home/yue/local_llm/models/Qwen__Qwen3.5-4B`
- fixed_state_length: `256`
- max_new_tokens: `64`
- input policy: both HF and ANEMLL use the same fixed-window-truncated prompt

| mode | prompt | full_tokens | used_tokens | similarity |
|---|---:|---:|---:|---:|
| normal | short | 25 | 25 | 1.0000 |
| normal | medium | 32 | 32 | 1.0000 |
| normal | long | 3089 | 192 | 1.0000 |
| think | short | 23 | 23 | 1.0000 |
| think | medium | 30 | 30 | 0.5542 |
| think | long | 3087 | 192 | 1.0000 |

## normal / short

- full_tokens: `25`
- used_tokens: `25`
- similarity: `1.0000`
- HF answer:
```text
4
```
- ANEMLL answer:
```text
4
```

## normal / medium

- full_tokens: `32`
- used_tokens: `32`
- similarity: `1.0000`
- HF answer:
```text
### Stack vs. Heap: The Simple Analogy

Imagine you are running a busy restaurant kitchen.

**The Stack is like the "Order Tickets" counter.**
*   **How it works:** When a new order comes in (a function is called), the chef immediately writes it down on a fresh ticket and
```
- ANEMLL answer:
```text
### Stack vs. Heap: The Simple Analogy

Imagine you are running a busy restaurant kitchen.

**The Stack is like the "Order Tickets" counter.**
*   **How it works:** When a new order comes in (a function is called), the chef immediately writes it down on a fresh ticket and
```

## normal / long

- full_tokens: `3089`
- used_tokens: `192`
- similarity: `1.0000`
- HF answer:
```text
42
```
- ANEMLL answer:
```text
42
```

## think / short

- full_tokens: `23`
- used_tokens: `23`
- similarity: `1.0000`
- HF answer:
```text
Thinking Process:

1.  **Analyze the Request:** The user is asking "What is 2 + 2?" and explicitly instructs "Return only the number."

2.  **Calculate the Answer:** 2 + 2 = 4.

3.  **Format the Output:** The
```
- ANEMLL answer:
```text
Thinking Process:

1.  **Analyze the Request:** The user is asking "What is 2 + 2?" and explicitly instructs "Return only the number."

2.  **Calculate the Answer:** 2 + 2 = 4.

3.  **Format the Output:** The
```

## think / medium

- full_tokens: `30`
- used_tokens: `30`
- similarity: `0.5542`
- HF answer:
```text
Thinking Process:

1.  **Analyze the Request:**
    *   Topic: Stack vs. Heap memory.
    *   Requirement 1: Explain the difference in simple terms.
    *   Requirement 2: Provide one practical debugging tip.
    *   Tone: Simple, clear,
```
- ANEMLL answer:
```text
Thinking Process:

1.  **Analyze the Request:**
    *   Topic: Difference between stack and heap memory.
    *   Constraint 1: Explain in "simple terms".
    *   Constraint 2: Provide "one practical debugging tip".

2.  **Deconstruct the Topic
```

## think / long

- full_tokens: `3087`
- used_tokens: `192`
- similarity: `1.0000`
- HF answer:
```text
Thinking Process:

1.  **Analyze the Request:**
    *   Input: A long string of repetitive text ("Project note filler: latency target, memory budget, throughput estimate, and deployment checklist.") followed by a final question.
    *   Final Question: "what number comes after 41
```
- ANEMLL answer:
```text
Thinking Process:

1.  **Analyze the Request:**
    *   Input: A long string of repetitive text ("Project note filler: latency target, memory budget, throughput estimate, and deployment checklist.") followed by a final question.
    *   Final Question: "what number comes after 41
```
