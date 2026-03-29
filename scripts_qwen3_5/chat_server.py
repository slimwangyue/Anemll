#!/usr/bin/env python3
"""Browser-based multi-round chat console for Qwen3.5-4B on ANE.

Architecture:
  - 10 CoreML models loaded at startup:
      embed (1) + lm_head (1) + FFN-infer chunks (4) + FFN-prefill chunks (4)
  - Prefill (prompt processing) uses the prefill-function models,
    which process up to 256 tokens at once with valid_len gating.
  - Decode (generation) uses the infer-function models (1 token at a time).
  - KV cache state is shared between infer and prefill instances.
  - KV cache is fixed at 1024 positions.

Cache overflow policy:
  When accumulated context (pos + new_prompt + generation_reserve) would exceed
  1024, the engine resets the KV cache and rebuilds from the most recent
  conversation turns that fit. Oldest turn pairs are dropped first.
  After overflow, chat continues seamlessly with reduced history.

Usage:
    python scripts_qwen3_5/chat_server.py [--port 8080]
    python scripts_qwen3_5/chat_server.py --model-dir /path/to/models --tokenizer /path/to/hf

Then open http://localhost:8080 in your browser.
"""
import sys, os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _SCRIPT_DIR)  # must be first for config.py

import gc, time, json, argparse, threading
import numpy as np
import coremltools as ct
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from transformers import AutoTokenizer
from config import FFN_LABEL

# ── HTML / JS / CSS ──────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Qwen3.5-4B ANE Chat</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #1a1a2e; color: #e0e0e0; height: 100vh;
    display: flex; flex-direction: column;
  }
  .header {
    background: #16213e; padding: 12px 20px; display: flex;
    align-items: center; justify-content: space-between;
    border-bottom: 1px solid #0f3460;
  }
  .header h1 { font-size: 16px; color: #e94560; }
  .header .info { font-size: 12px; color: #888; }
  .header button {
    background: #0f3460; color: #e0e0e0; border: 1px solid #e94560;
    padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 12px;
  }
  .header button:hover { background: #e94560; color: #fff; }
  .header button.active { background: #e94560; color: #fff; border-color: #e94560; }
  #chat {
    flex: 1; overflow-y: auto; padding: 16px 20px;
    display: flex; flex-direction: column; gap: 12px;
  }
  .msg { max-width: 85%; padding: 10px 14px; border-radius: 12px; line-height: 1.5; }
  .msg.user {
    align-self: flex-end; background: #0f3460; border-bottom-right-radius: 4px;
  }
  .msg.assistant {
    align-self: flex-start; background: #16213e; border: 1px solid #333;
    border-bottom-left-radius: 4px;
  }
  .msg .think {
    color: #888; font-style: italic; font-size: 13px;
    border-left: 2px solid #444; padding-left: 8px; margin-bottom: 6px;
  }
  .msg .think summary { cursor: pointer; color: #aaa; }
  .msg .answer { white-space: pre-wrap; }
  .msg .answer code {
    background: #0d1117; padding: 2px 5px; border-radius: 3px;
    font-family: 'SF Mono', Monaco, monospace; font-size: 13px;
  }
  .msg .answer pre {
    background: #0d1117; padding: 10px; border-radius: 6px;
    overflow-x: auto; margin: 6px 0;
  }
  .msg .answer pre code { padding: 0; background: none; }
  .msg .meta {
    font-size: 11px; color: #666; margin-top: 4px; text-align: right;
  }
  .typing { color: #e94560; font-size: 13px; padding: 4px 20px; min-height: 20px; }
  .input-area {
    display: flex; gap: 8px; padding: 12px 20px;
    background: #16213e; border-top: 1px solid #0f3460;
  }
  #input {
    flex: 1; background: #1a1a2e; color: #e0e0e0; border: 1px solid #333;
    padding: 10px 14px; border-radius: 8px; font-size: 14px;
    font-family: inherit; resize: none; min-height: 42px; max-height: 120px;
  }
  #input:focus { outline: none; border-color: #e94560; }
  #send {
    background: #e94560; color: #fff; border: none; padding: 10px 20px;
    border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 600;
    align-self: flex-end;
  }
  #send:hover { background: #d63851; }
  #send:disabled { background: #555; cursor: not-allowed; }
  .settings {
    display: none; background: #16213e; padding: 12px 20px;
    border-bottom: 1px solid #0f3460; font-size: 13px;
  }
  .settings.open { display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }
  .settings label { color: #aaa; }
  .settings input, .settings select {
    background: #1a1a2e; color: #e0e0e0; border: 1px solid #333;
    padding: 4px 8px; border-radius: 4px; width: 70px;
  }
  .settings input[type="checkbox"] { width: auto; }
</style>
</head>
<body>
<div class="header">
  <h1>Qwen3.5-4B on ANE</h1>
  <span class="info" id="status">Loading models...</span>
  <div>
    <button id="thinkBtn" class="active" onclick="toggleThink()">Think: ON</button>
    <button onclick="toggleSettings()">Settings</button>
    <button onclick="resetChat()">New Chat</button>
  </div>
</div>
<div class="settings" id="settings">
  <label>Max tokens: <input type="number" id="maxTokens" value="512" min="16" max="2048"></label>
  <label>Show thinking: <input type="checkbox" id="showThink" checked></label>
  <label>Thinking mode: <input type="checkbox" id="enableThinking" checked></label>
  <label>Repetition guard: <input type="checkbox" id="repGuard" checked></label>
  <label>Temperature: <input type="number" id="temperature" value="0.7" min="0.0" max="2.0" step="0.05"></label>
  <label>Top-p: <input type="number" id="topP" value="0.9" min="0.0" max="1.0" step="0.05"></label>
  <label>Rep penalty: <input type="number" id="repPenalty" value="1.1" min="1.0" max="2.0" step="0.05"></label>
  <label>Pres penalty: <input type="number" id="presPenalty" value="0.0" min="0.0" max="2.0" step="0.1"></label>
  <label>Freq penalty: <input type="number" id="freqPenalty" value="0.2" min="0.0" max="2.0" step="0.1"></label>
</div>
<div id="chat"></div>
<div class="typing" id="typing"></div>
<div class="input-area">
  <textarea id="input" placeholder="Type a message..." rows="1"
    onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendMsg();}"></textarea>
  <button id="send" onclick="sendMsg()">Send</button>
</div>

<script>
let generating = false;

async function checkStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    document.getElementById('status').textContent =
      d.ready ? `Ready | CTX=${d.ctx} | pos=${d.pos} (${d.cache_pct}%) | ${d.turns} turns` : 'Loading models...';
    document.getElementById('send').disabled = !d.ready;
    if (!d.ready) setTimeout(checkStatus, 2000);
  } catch(e) { setTimeout(checkStatus, 2000); }
}

function toggleThink() {
  const cb = document.getElementById('enableThinking');
  cb.checked = !cb.checked;
  const btn = document.getElementById('thinkBtn');
  btn.textContent = cb.checked ? 'Think: ON' : 'Think: OFF';
  btn.classList.toggle('active', cb.checked);
}

function toggleSettings() {
  document.getElementById('settings').classList.toggle('open');
}

function addMsg(role, html, meta) {
  const chat = document.getElementById('chat');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = html + (meta ? `<div class="meta">${meta}</div>` : '');
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function escapeHtml(t) {
  return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function formatAnswer(text) {
  text = escapeHtml(text);
  text = text.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
  text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
  text = text.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  return text;
}

async function sendMsg() {
  if (generating) return;
  const input = document.getElementById('input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  input.style.height = 'auto';

  addMsg('user', escapeHtml(text));

  generating = true;
  document.getElementById('send').disabled = true;
  document.getElementById('typing').textContent = 'Generating...';

  const maxTokens = parseInt(document.getElementById('maxTokens').value) || 512;
  const showThink = document.getElementById('showThink').checked;
  const enableThinking = document.getElementById('enableThinking').checked;

  try {
    const resp = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text, max_tokens: maxTokens, enable_thinking: enableThinking, repetition_guard: document.getElementById('repGuard').checked, temperature: parseFloat(document.getElementById('temperature').value) || 0.7, top_p: parseFloat(document.getElementById('topP').value) || 0.9, repetition_penalty: parseFloat(document.getElementById('repPenalty').value) || 1.1, presence_penalty: parseFloat(document.getElementById('presPenalty').value) || 0.0, frequency_penalty: parseFloat(document.getElementById('freqPenalty').value) || 0.2})
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let thinkText = '';
    let answerText = '';
    let inThink = false;
    let msgDiv = null;
    let tokCount = 0;
    let startTime = Date.now();

    readLoop:
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream: true});

      let lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6);
        if (payload === '[DONE]') { break readLoop; }

        let ev;
        try { ev = JSON.parse(payload); } catch(e) { continue; }

        if (ev.type === 'token') {
          tokCount++;
          let tok = ev.text;
          // Detect <think> tag entering think mode
          if (!inThink && tok.includes('<think>')) {
            inThink = true;
            tok = tok.replace('<think>', '');
          }
          if (inThink && tok.includes('</think>')) {
            const parts = tok.split('</think>');
            thinkText += parts[0];
            answerText += parts.slice(1).join('');
            inThink = false;
          } else if (inThink) {
            thinkText += tok;
          } else {
            answerText += tok;
          }

          let html = '';
          if (showThink && thinkText.trim()) {
            html += `<details class="think"><summary>Thinking (${thinkText.length} chars)</summary>${escapeHtml(thinkText.trim())}</details>`;
          }
          html += `<div class="answer">${formatAnswer(answerText.trim() || (inThink && enableThinking ? '...' : ''))}</div>`;

          if (!msgDiv) msgDiv = addMsg('assistant', html);
          else {
            const metaHtml = msgDiv.querySelector('.meta')?.outerHTML || '';
            msgDiv.innerHTML = html + metaHtml;
          }

          const elapsed = (Date.now() - startTime) / 1000;
          document.getElementById('typing').textContent =
            `Generating... ${tokCount} tokens (${(tokCount/elapsed).toFixed(1)} tok/s)`;
          document.getElementById('chat').scrollTop = document.getElementById('chat').scrollHeight;
        } else if (ev.type === 'done') {
          const elapsed = (Date.now() - startTime) / 1000;
          const meta = `${ev.decode_tokens} tokens | ${elapsed.toFixed(1)}s | ${(ev.decode_tokens/elapsed).toFixed(1)} tok/s | pos=${ev.end_pos}` + (ev.stop_reason === 'repetition' ? ' | ⚠ stopped: repetition' : '');
          if (msgDiv) {
            let html = '';
            if (showThink && thinkText.trim()) {
              html += `<details class="think"><summary>Thinking (${thinkText.length} chars)</summary>${escapeHtml(thinkText.trim())}</details>`;
            }
            html += `<div class="answer">${formatAnswer(answerText.trim())}</div>`;
            html += `<div class="meta">${meta}</div>`;
            msgDiv.innerHTML = html;
          }
        } else if (ev.type === 'error') {
          addMsg('assistant', `<div class="answer" style="color:#e94560">Error: ${escapeHtml(ev.message)}</div>`);
        }
      }
    }
  } catch(e) {
    addMsg('assistant', `<div class="answer" style="color:#e94560">Network error: ${escapeHtml(e.message)}</div>`);
  } finally {
    generating = false;
    document.getElementById('send').disabled = false;
    document.getElementById('typing').textContent = '';
    checkStatus();
  }
}

async function resetChat() {
  await fetch('/api/reset', {method: 'POST'});
  document.getElementById('chat').innerHTML = '';
  checkStatus();
}

document.getElementById('input').addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

checkStatus();
document.getElementById('input').focus();
</script>
</body>
</html>
"""

# ── Constants ────────────────────────────────────────────────────────

BLOCK_SIZE = 256        # logical prefill block size (matches model export)
BATCH_SIZE = 256        # prefill batch size (must match compiled model)
MIN_GEN_RESERVE = 100   # minimum tokens reserved for generation after prefill
PREFILL_CROSSOVER = 32  # below this many tokens, sequential is faster than batch


# ── Repetition Detection ─────────────────────────────────────────────

class RepetitionDetector:
    """Sliding-window n-gram repetition detector.

    Maintains a window of recent token IDs and checks for repeated
    n-grams.  When any n-gram appears >= threshold times within the
    window, `is_repeating()` returns True.

    This works with argmax-only LM heads (no logits required).
    Matches the ANEMLLChat app's RepetitionDetector design.
    """

    def __init__(self, window_size=80, ngram_size=5, threshold=3):
        self.window_size = window_size
        self.ngram_size = ngram_size
        self.threshold = threshold
        self._window = []  # recent token IDs

    def reset(self):
        self._window.clear()

    def add_token(self, token_id):
        """Add a token and return True if repetition detected."""
        self._window.append(token_id)
        if len(self._window) > self.window_size:
            self._window.pop(0)
        return self._check()

    def _check(self):
        tokens = self._window
        n = self.ngram_size
        if len(tokens) < n:
            return False
        # Count n-gram occurrences
        counts = {}
        for i in range(len(tokens) - n + 1):
            gram = tuple(tokens[i:i + n])
            counts[gram] = counts.get(gram, 0) + 1
            if counts[gram] >= self.threshold:
                return True
        return False


# ── Helpers ──────────────────────────────────────────────────────────

def _load_model(path, compute_unit, function_name=None):
    """Load a CoreML model from .mlpackage or .mlmodelc."""
    if path.endswith(".mlmodelc"):
        return ct.models.CompiledMLModel(path, compute_unit)
    kwargs = {"compute_units": compute_unit}
    if function_name:
        kwargs["function_name"] = function_name
    return ct.models.MLModel(path, **kwargs)


def _find_model(base_dir, name):
    """Find model path, preferring .mlmodelc over .mlpackage."""
    for ext in (".mlmodelc", ".mlpackage"):
        p = os.path.join(base_dir, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model found for {name} in {base_dir}")


def _chunk_tokens(tokens, block_size=BLOCK_SIZE):
    """Split token list into block_size chunks. Last chunk may be shorter."""
    return [tokens[i:i + block_size] for i in range(0, len(tokens), block_size)]


# ── Chat Engine ──────────────────────────────────────────────────────

class ChatEngine:
    """Manages model inference, conversation state, and KV cache.

    All prompt processing (prefill) and response generation (decode) use
    the same set of infer-function models. No model switching at runtime.
    Models are loaded once at startup and kept in memory.
    """

    def __init__(self, model_dir, hf_path, ctx=1024, num_chunks=4):
        self.model_dir = model_dir
        self.hf_path = hf_path
        self.ctx = ctx
        self.num_chunks = num_chunks
        self.ready = False
        self.lock = threading.Lock()

        # Conversation state
        self.messages = []     # list of {"role": ..., "content": ...}
        self.pos = 0           # next write position in KV cache (0..ctx-1)
        self.states = None     # CoreML model states (KV cache)
        self.lin_convs = None  # linear conv states per chunk
        self.lin_recs = None   # linear recurrent states per chunk

        # Template token IDs (filled after tokenizer loads)
        self.tpl_tokens = {}

        # Detect combined dedup directory
        self.combined_dir = os.path.join(model_dir, f"combined_{FFN_LABEL}_dedup")
        self.use_combined = os.path.isdir(self.combined_dir)

    # ── Model loading (called once at startup) ───────────────────────

    def load(self):
        """Load all models synchronously.

        Loads separate infer and prefill MLModel instances for each
        FFN chunk.  The two instances share KV-cache state but accept
        different input shapes (seq_len=1 vs seq_len=256).
        """
        cu = ct.ComputeUnit.CPU_AND_NE

        print("[engine] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_path, use_fast=False)
        self._build_stop_ids()
        self._build_template_tokens()

        print("[engine] Loading embeddings...")
        self.embed = _load_model(
            _find_model(self.model_dir, "embeddings"), cu)

        print("[engine] Loading lm_head...")
        # Prefer logits lm_head (enables logit-space penalties)
        try:
            lmhead_path = _find_model(self.model_dir, "lm_head_logits")
            self.lmhead = _load_model(lmhead_path, cu)
            self.lmhead_mode = "logits"
            print(f"  Loaded logits lm_head (penalties enabled)")
        except FileNotFoundError:
            self.lmhead = _load_model(
                _find_model(self.model_dir, "lm_head"), cu)
            # Detect output type
            spec = self.lmhead.get_spec()
            out_names = [o.name for o in spec.description.output]
            self.lmhead_mode = "logits" if ("logits" in out_names
                                            or "output_logits" in out_names
                                            ) else "argmax"
            print(f"  Loaded lm_head (mode={self.lmhead_mode})")
        # Detect split logits (logits1..logitsN) vs single output
        if self.lmhead_mode == "logits":
            spec = self.lmhead.get_spec()
            out_names = [o.name for o in spec.description.output]
            split_keys = sorted([n for n in out_names if n.startswith("logits")
                                 and n[6:].isdigit()])
            if split_keys:
                self.logits_keys = split_keys  # ["logits1", ..., "logits16"]
                self.logits_key = None
                print(f"  Split logits: {len(split_keys)}-way ({split_keys[0]}..{split_keys[-1]})")
            else:
                self.logits_keys = None
                self.logits_key = ("output_logits" if "output_logits" in out_names
                                  else "logits")

        print("[engine] Loading FFN chunks (infer + prefill)...")
        self.ffns = []       # infer instances  (seq_len=1)
        self.prefills = []   # prefill instances (seq_len=256)
        self.has_prefill = False

        for ci in range(self.num_chunks):
            # --- infer instance ---
            if self.use_combined:
                path = _find_model(self.combined_dir, f"chunk{ci}")
                if path.endswith(".mlmodelc"):
                    # compiled multi-function models can't select function
                    self.use_combined = False
            if self.use_combined:
                print(f"  chunk {ci} infer  (combined)...", end="", flush=True)
                import time as _t; _t0 = _t.time()
                m_infer = _load_model(path, cu, function_name="infer")
                print(f" {_t.time()-_t0:.0f}s")
            else:
                path = _find_model(self.model_dir, f"ffn_{FFN_LABEL}_chunk{ci}")
                print(f"  chunk {ci} infer  (separate)...", end="", flush=True)
                import time as _t; _t0 = _t.time()
                m_infer = _load_model(path, cu)
                print(f" {_t.time()-_t0:.0f}s")
            self.ffns.append(m_infer)

            # --- prefill instance ---
            m_prefill = None
            if self.use_combined:
                print(f"  chunk {ci} prefill (combined)...", end="", flush=True)
                _t0 = _t.time()
                m_prefill = _load_model(path, cu, function_name="prefill")
                print(f" {_t.time()-_t0:.0f}s")
            else:
                try:
                    pf_path = _find_model(
                        self.model_dir, f"prefill_{FFN_LABEL}_chunk{ci}")
                    print(f"  chunk {ci} prefill (separate)...", end="", flush=True)
                    import time as _t; _t0 = _t.time()
                    m_prefill = _load_model(pf_path, cu)
                    print(f" {_t.time()-_t0:.0f}s")
                except FileNotFoundError:
                    print(f"  chunk {ci} prefill — not found, batch disabled")
            self.prefills.append(m_prefill)

        self.has_prefill = all(p is not None for p in self.prefills)

        self._detect_shapes()

        # Pre-allocate reusable buffers to avoid per-token allocation
        self._tok_buf = np.zeros((1, 1), dtype=np.int32)
        self._mask_buf = np.full(
            (1, 1, 1, self.ctx), -65504.0, dtype=np.float16)
        self._pos_buf = np.zeros(1, dtype=np.int32)

        # Pre-allocate batch prefill buffers
        self._batch_tok_buf = np.zeros((1, BATCH_SIZE), dtype=np.int32)
        self._batch_embed_buf = np.zeros((1, BATCH_SIZE), dtype=np.int32)
        self._valid_len_buf = np.zeros((1,), dtype=np.int32)
        self._batch_mask_buf = np.full(
            (1, 1, BATCH_SIZE, self.ctx), -65504.0, dtype=np.float16)
        self._batch_pos_buf = np.zeros(BATCH_SIZE, dtype=np.int32)
        self._batch_cur_buf = np.zeros(1, dtype=np.int32)

        self._reset_states()
        self.ready = True
        mode = "combined-dedup" if self.use_combined else "separate"
        prefill_mode = "batch-256" if self.has_prefill else "sequential"
        print(f"[engine] Ready! CTX={self.ctx}, BATCH={BATCH_SIZE}, "
              f"crossover={PREFILL_CROSSOVER}, "
              f"mode={mode}, prefill={prefill_mode}, "
              f"stop_ids={self.stop_ids}")

    def _build_stop_ids(self):
        self.stop_ids = set()
        if self.tokenizer.eos_token_id is not None:
            self.stop_ids.add(self.tokenizer.eos_token_id)
        for name in ["<|im_end|>", "<|endoftext|>", "<|end|>"]:
            tok = self.tokenizer.convert_tokens_to_ids(name)
            if tok is not None and tok != self.tokenizer.unk_token_id:
                self.stop_ids.add(tok)

    def _build_template_tokens(self):
        t = self.tokenizer
        self.tpl_tokens = {
            "im_start": t.convert_tokens_to_ids("<|im_start|>"),
            "im_end": t.convert_tokens_to_ids("<|im_end|>"),
            "nl": t.encode("\n", add_special_tokens=False),
            "think": t.convert_tokens_to_ids("<think>"),
            "endthink": t.convert_tokens_to_ids("</think>"),
            "user": t.encode("user", add_special_tokens=False),
            "assistant": t.encode("assistant", add_special_tokens=False),
        }

    def _detect_shapes(self):
        """Read input shapes from model spec for state initialization."""
        self.inp_map = {}
        try:
            spec = self.ffns[0].get_spec()
            fn_inputs = None
            if self.use_combined and hasattr(spec.description, 'functions'):
                for fn in spec.description.functions:
                    if fn.name == "infer":
                        fn_inputs = fn.input
                        break
            if fn_inputs is None:
                fn_inputs = spec.description.input
            for inp in fn_inputs:
                try:
                    self.inp_map[inp.name] = tuple(
                        inp.type.multiArrayType.shape)
                except Exception:
                    pass
        except Exception:
            print("[engine] Using default input shapes")
            self.inp_map = {
                'linear_conv_state': (8, 1024, 32),
                'linear_recurrent_state': (8, 32, 128, 128),
            }

    def _reset_states(self):
        """Reset KV cache, linear states, and position to zero."""
        self.states = [m.make_state() for m in self.ffns]
        self.lin_convs = [
            np.zeros(self.inp_map['linear_conv_state'], dtype=np.float16)
            for _ in range(self.num_chunks)]
        self.lin_recs = [
            np.zeros(self.inp_map['linear_recurrent_state'], dtype=np.float16)
            for _ in range(self.num_chunks)]
        self.pos = 0

    def reset(self):
        """Full reset: clear KV cache and conversation history."""
        with self.lock:
            self._reset_states()
            self.messages = []
            print(f"[cache] Full reset. pos=0, messages=0")

    # ── Core model step ──────────────────────────────────────────────

    def _extract_logits(self, lm_out):
        """Extract flattened fp32 logits from lm_head output.

        Handles both split (logits1..logitsN) and single (logits/output_logits) formats.
        """
        if self.logits_keys:
            parts = [lm_out[k].flatten().astype(np.float32) for k in self.logits_keys]
            return np.concatenate(parts)
        return lm_out[self.logits_key].flatten().astype(np.float32)

    def _step_kv_only(self, tok_id, pos):
        """Run one token through embed -> all FFN chunks (NO lm_head).

        Updates KV cache / state but does not compute logits.
        Used for all prefill tokens except the last one.
        """
        tok = self._tok_buf
        tok[0, 0] = tok_id
        hidden = list(
            self.embed.predict({"input_ids": tok}).values())[0]

        mask = self._mask_buf
        mask[:, :, :, :] = -65504.0
        mask[:, :, :, :pos + 1] = 0

        pos_arr = self._pos_buf
        pos_arr[0] = pos

        for ci in range(self.num_chunks):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_arr,
                "causal_mask": mask,
                "current_pos": pos_arr,
                "linear_conv_state": self.lin_convs[ci],
                "linear_recurrent_state": self.lin_recs[ci],
            }
            out = self.ffns[ci].predict(inp, state=self.states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                self.lin_convs[ci] = out['linear_conv_state_out']
                self.lin_recs[ci] = out['linear_recurrent_state_out']

    def _step(self, tok_id, pos):
        """Run one token through embed -> all FFN chunks -> lm_head.
        Returns (next_token_id, logits_or_None).

        If lm_head outputs logits, returns raw logits as np.float32 array.
        If lm_head uses fused argmax, returns None for logits.
        """
        tok = self._tok_buf
        tok[0, 0] = tok_id
        hidden = list(
            self.embed.predict({"input_ids": tok}).values())[0]

        mask = self._mask_buf
        mask[:, :, :, :] = -65504.0
        mask[:, :, :, :pos + 1] = 0

        pos_arr = self._pos_buf
        pos_arr[0] = pos

        for ci in range(self.num_chunks):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_arr,
                "causal_mask": mask,
                "current_pos": pos_arr,
                "linear_conv_state": self.lin_convs[ci],
                "linear_recurrent_state": self.lin_recs[ci],
            }
            out = self.ffns[ci].predict(inp, state=self.states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                self.lin_convs[ci] = out['linear_conv_state_out']
                self.lin_recs[ci] = out['linear_recurrent_state_out']

        lm_out = self.lmhead.predict(
            {"hidden_states": hidden.astype(np.float16)})
        if self.lmhead_mode == "logits":
            logits = self._extract_logits(lm_out)
            return int(np.argmax(logits)), logits
        return int(lm_out["argmax_idx"].flatten()[0]), None

    def _batch_prefill(self, token_ids, block_start):
        """Process up to BATCH_SIZE tokens through the prefill function.

        Pads short sequences to BATCH_SIZE with zeros.  The valid_len
        input ensures padding tokens do not corrupt linear-attention
        state (conv_state, recurrent_state) and the correct last-token
        hidden state is returned by the model.

        block_start is the KV cache write position.
        Returns next_token_id for the last valid token in the batch.
        Updates self.pos to block_start + len(token_ids).
        """
        valid_len = len(token_ids)
        assert 1 <= valid_len <= BATCH_SIZE

        # Batch embedding: (1, BATCH_SIZE)  — pad with token 0
        input_ids = self._batch_tok_buf
        input_ids[0, :] = 0
        input_ids[0, :valid_len] = token_ids
        hidden = list(
            self.embed.predict({"input_ids": input_ids}).values())[0]

        # Build causal mask: (1, 1, BATCH_SIZE, CTX)
        # Valid positions get normal causal mask; padding rows get all -inf.
        mask = self._batch_mask_buf
        mask[:, :, :, :] = -65504.0
        for i in range(valid_len):
            mask[0, 0, i, :block_start + i + 1] = 0
        # Padding rows (valid_len..BATCH_SIZE-1) stay all -inf → no attention.

        pos_ids = self._batch_pos_buf
        pos_ids[:valid_len] = np.arange(
            block_start, block_start + valid_len, dtype=np.int32)
        pos_ids[valid_len:] = 0  # padding positions — meaningless

        cur_pos = self._batch_cur_buf
        cur_pos[0] = block_start

        valid_len_arr = self._valid_len_buf
        valid_len_arr[0] = valid_len

        for ci in range(self.num_chunks):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": pos_ids,
                "causal_mask": mask,
                "current_pos": cur_pos,
                "linear_conv_state": self.lin_convs[ci],
                "linear_recurrent_state": self.lin_recs[ci],
                "valid_len": valid_len_arr,
            }
            out = self.prefills[ci].predict(inp, state=self.states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                self.lin_convs[ci] = out['linear_conv_state_out']
                self.lin_recs[ci] = out['linear_recurrent_state_out']

        # Last chunk already extracts the last valid token's hidden
        # state and applies RMSNorm → shape (1, 1, hidden_size).
        lm_out = self.lmhead.predict(
            {"hidden_states": hidden.astype(np.float16)})
        if self.lmhead_mode == "logits":
            logits = self._extract_logits(lm_out)
            next_id = int(np.argmax(logits))
        else:
            next_id = int(lm_out["argmax_idx"].flatten()[0])

        self.pos = block_start + valid_len
        return next_id

    # ── Logit-space penalties ────────────────────────────────────────

    def _apply_penalties(self, logits, generated_ids,
                         repetition_penalty=1.1,
                         presence_penalty=0.0,
                         frequency_penalty=0.2,
                         temperature=0.7,
                         top_p=0.9):
        """Apply penalties + temperature/top-p sampling, return token ID.

        Modifies logits in-place and returns sampled token ID.
        """
        # 1. Repetition / presence / frequency penalties
        token_counts = {}
        for tid in generated_ids:
            token_counts[tid] = token_counts.get(tid, 0) + 1

        for tid, count in token_counts.items():
            if repetition_penalty != 1.0:
                if logits[tid] > 0:
                    logits[tid] /= repetition_penalty
                else:
                    logits[tid] *= repetition_penalty
            if presence_penalty != 0.0:
                logits[tid] -= presence_penalty
            if frequency_penalty != 0.0:
                logits[tid] -= frequency_penalty * count

        # 2. Temperature + top-p (nucleus) sampling
        if temperature <= 0 or top_p <= 0:
            return int(np.argmax(logits))

        logits_f = logits.astype(np.float64)
        logits_f /= temperature

        # Numerical stability
        logits_f -= np.max(logits_f)
        probs = np.exp(logits_f)
        probs /= probs.sum()

        # Top-p filtering
        if top_p < 1.0:
            sorted_idx = np.argsort(-probs)
            sorted_probs = probs[sorted_idx]
            cumsum = np.cumsum(sorted_probs)
            # Keep tokens until cumulative prob exceeds top_p
            cutoff = np.searchsorted(cumsum, top_p) + 1
            mask = np.zeros_like(probs, dtype=bool)
            mask[sorted_idx[:cutoff]] = True
            probs[~mask] = 0.0
            probs /= probs.sum()

        return int(np.random.choice(len(probs), p=probs))

    # ── Prefill: process prompt tokens ───────────────────────────────
    #
    # Uses true batched prefill (256 tokens at once) for all chunks
    # when prefill models are available and input length exceeds the
    # crossover threshold.  Below the threshold, sequential is faster
    # because the batch always processes a full 256-token frame.

    def _process_prompt(self, prompt_tokens):
        """PREFILL PHASE: process all prompt tokens through the model.

        Strategy (when has_prefill=True):
          - If total tokens >= PREFILL_CROSSOVER:
              Process all tokens in 256-token batched blocks.
              Full blocks use valid_len=256 (no padding).
              The final tail block uses valid_len=len(tail) with zero-padding.
          - If total tokens < PREFILL_CROSSOVER:
              Sequential token-by-token (skip lm_head except last).

        Fallback (has_prefill=False):
          - All tokens sequential.

        Returns: predicted next-token ID (first decode token),
                 or None if cache overflows during prefill.
        """
        n_total = len(prompt_tokens)
        if n_total == 0:
            return None

        last_next = None
        t_total = time.time()
        n_batched = 0
        use_batch = self.has_prefill and n_total >= PREFILL_CROSSOVER

        # ── Batch prefill path ──
        if use_batch:
            chunks = _chunk_tokens(prompt_tokens, BATCH_SIZE)
            for block in chunks:
                block_len = len(block)
                if self.pos + block_len > self.ctx:
                    print(f"[prefill] OVERFLOW at pos={self.pos} "
                          f"during batch prefill")
                    return None
                t0 = time.time()
                last_next = self._batch_prefill(block, self.pos)
                elapsed = time.time() - t0
                tps = block_len / max(elapsed, 1e-9)
                n_batched += block_len
                tag = "full" if block_len == BATCH_SIZE else f"tail({block_len})"
                print(f"[prefill] batch-{tag}: {block_len} tok, "
                      f"{elapsed*1000:.0f}ms ({tps:.0f} tok/s), "
                      f"pos={self.pos}/{self.ctx}")

        # ── Sequential fallback ──
        remaining = prompt_tokens[n_batched:]
        n_remaining = len(remaining)
        if n_remaining > 0:
            t0 = time.time()
            for ti, tok_id in enumerate(remaining):
                if self.pos >= self.ctx - 1:
                    print(f"[prefill] OVERFLOW at pos={self.pos} "
                          f"during sequential prefill")
                    return None
                is_last = (ti == n_remaining - 1)
                if is_last:
                    last_next, _ = self._step(tok_id, self.pos)
                else:
                    self._step_kv_only(tok_id, self.pos)
                self.pos += 1
            elapsed = time.time() - t0
            tps = n_remaining / max(elapsed, 1e-9)
            print(f"[prefill] seq: {n_remaining} tok, "
                  f"{elapsed*1000:.0f}ms ({tps:.0f} tok/s), "
                  f"pos={self.pos}/{self.ctx}")

        total_elapsed = time.time() - t_total
        total_tps = n_total / max(total_elapsed, 1e-9)
        print(f"[prefill] total: {n_total} tok in "
              f"{total_elapsed*1000:.0f}ms ({total_tps:.0f} tok/s) "
              f"[batch={n_batched}, seq={n_remaining}]")
        return last_next

    # ── Cache overflow management ────────────────────────────────────
    #
    # Overflow policy (deterministic):
    #   1. Before each turn, check: pos + prompt_len + MIN_GEN_RESERVE > ctx
    #   2. If no overflow, proceed with incremental continuation
    #   3. If overflow:
    #      a. Drop oldest user+assistant turn pairs from self.messages
    #      b. Re-tokenize retained messages with apply_chat_template
    #      c. Reset KV cache and all states (pos=0)
    #      d. Return new prompt tokens for full re-prefill from scratch
    #   4. Always keep at least the current user message
    #   5. Chat continues seamlessly — user sees no disruption

    def _handle_overflow(self, prompt_tokens, enable_thinking):
        """Check cache space and handle overflow if needed.

        Returns: (prompt_tokens, did_overflow)
          - If no overflow: returns original tokens, False
          - If overflow: trims messages, resets cache, returns
            re-tokenized prompt from apply_chat_template, True
        """
        space_needed = self.pos + len(prompt_tokens) + MIN_GEN_RESERVE
        if space_needed <= self.ctx:
            return prompt_tokens, False

        print(f"[cache] OVERFLOW: pos={self.pos} + prompt="
              f"{len(prompt_tokens)} + reserve={MIN_GEN_RESERVE} = "
              f"{space_needed} > ctx={self.ctx}")

        # Try dropping oldest turn pairs until prompt fits
        retained = list(self.messages)
        while len(retained) > 1:
            if len(retained) >= 3:
                retained = retained[2:]  # drop first user+assistant pair
            else:
                break

            test_tokens = self._tokenize_messages(
                retained, enable_thinking)
            if len(test_tokens) + MIN_GEN_RESERVE <= self.ctx:
                n_dropped = (len(self.messages) - len(retained)) // 2
                print(f"[cache] Trimmed {n_dropped} old turn(s), "
                      f"retained {len(retained)} msg(s), "
                      f"rebuilt: {len(test_tokens)} tok")
                self.messages = retained
                self._reset_states()
                return test_tokens, True

        # Fallback: keep only the current user message
        retained = [self.messages[-1]]
        test_tokens = self._tokenize_messages(retained, enable_thinking)
        n_total = (len(self.messages) - 1) // 2
        print(f"[cache] Trimmed ALL {n_total} old turn(s), "
              f"current message only: {len(test_tokens)} tok")
        self.messages = retained
        self._reset_states()
        return test_tokens, True

    def _tokenize_messages(self, messages, enable_thinking):
        """Tokenize message list using apply_chat_template."""
        input_ids = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=enable_thinking)
        if hasattr(input_ids, 'input_ids'):
            input_ids = input_ids.input_ids
        return input_ids[0].tolist()

    # ── Multi-round conversation ─────────────────────────────────────

    def _build_continuation_tokens(self, user_msg, enable_thinking=True):
        """Build incremental token sequence for a follow-up turn.

        Produces: <|im_end|> NL <|im_start|>user NL {msg} <|im_end|>
                  NL <|im_start|>assistant NL [<think> NL]

        The leading <|im_end|> closes the previous assistant turn.
        Always included because the stop token from the previous
        decode was predicted but never fed back into the model.
        """
        t = self.tpl_tokens
        tk = self.tokenizer
        nl = t["nl"]
        tokens = []
        tokens += [t["im_end"]] + nl            # close previous turn
        tokens += [t["im_start"]] + t["user"] + nl  # new user turn
        tokens += tk.encode(user_msg, add_special_tokens=False)
        tokens += [t["im_end"]] + nl
        tokens += [t["im_start"]] + t["assistant"] + nl  # assistant prompt
        if enable_thinking:
            tokens += [t["think"]] + nl
        else:
            # Empty think block: <think>\n\n</think>\n\n
            tokens += [t["think"]] + nl + nl + [t["endthink"]] + nl + nl
        return tokens

    # ── Main chat stream ─────────────────────────────────────────────

    def chat_stream(self, user_msg, max_tokens=512,
                     enable_thinking=True, repetition_guard=True,
                     temperature=0.7, top_p=0.9,
                     repetition_penalty=1.1, presence_penalty=0.0,
                     frequency_penalty=0.2):
        """Generator yielding SSE events for a streaming response.

        Two-phase pipeline:
          1. PREFILL — process prompt tokens (sequentially, in 256-tok blocks)
          2. DECODE  — generate response tokens one at a time
        Both phases use the same infer-function models. No switching.

        repetition_guard: if True, stops generation early when n-gram
        repetition is detected (sliding-window 5-gram, threshold=3).
        """
        with self.lock:
            is_first_turn = len(self.messages) == 0

            self.messages.append({"role": "user", "content": user_msg})

            # Build prompt tokens
            if is_first_turn:
                prompt_tokens = self._tokenize_messages(
                    self.messages, enable_thinking)
            else:
                prompt_tokens = self._build_continuation_tokens(
                    user_msg, enable_thinking)

            turn_num = (len(self.messages) + 1) // 2
            print(f"\n{'='*60}")
            print(f"[chat] Turn {turn_num}: "
                  f"\"{user_msg[:60]}{'...' if len(user_msg)>60 else ''}\"")
            print(f"[chat] prompt={len(prompt_tokens)} tok, "
                  f"cache={self.pos}/{self.ctx} "
                  f"({self.pos*100//self.ctx}%), "
                  f"history={len(self.messages)} msg(s)")

            # ── Check for cache overflow ──
            prompt_tokens, overflowed = self._handle_overflow(
                prompt_tokens, enable_thinking)
            if overflowed:
                print(f"[cache] After rebuild: "
                      f"prompt={len(prompt_tokens)} tok, "
                      f"cache=0/{self.ctx}")

            # Adjust max_tokens to fit remaining cache
            remaining = self.ctx - self.pos - len(prompt_tokens)
            if remaining < 10:
                yield {"type": "error",
                       "message": f"Context full ({self.pos}/{self.ctx}). "
                                  f"Please reset the chat."}
                return
            max_tokens = min(max_tokens, remaining)

            # ── PREFILL PHASE ──
            t0 = time.time()
            last_next = self._process_prompt(prompt_tokens)
            if last_next is None:
                yield {"type": "error",
                       "message": "Context overflow during prefill."}
                return

            # ── DECODE PHASE ──
            rep_detector = RepetitionDetector() if repetition_guard else None
            generated_ids = [last_next]
            if rep_detector:
                rep_detector.add_token(last_next)
            text_so_far = self.tokenizer.decode(
                [last_next], skip_special_tokens=False)
            yield {"type": "token", "text": text_so_far, "id": last_next}

            has_penalties = (repetition_penalty > 1.0
                             or presence_penalty != 0.0
                             or frequency_penalty != 0.0
                             or (temperature > 0 and temperature != 1.0)
                             or (top_p > 0 and top_p < 1.0))
            if has_penalties and self.lmhead_mode == "logits":
                print(f"[decode] Sampling: temp={temperature}, top_p={top_p}, "
                      f"rep={repetition_penalty}, pres={presence_penalty}, "
                      f"freq={frequency_penalty}")
            stopped_by_rep = False
            for gi in range(max_tokens - 1):
                if self.pos >= self.ctx - 1:
                    break
                next_id, logits = self._step(generated_ids[-1], self.pos)
                self.pos += 1
                if logits is not None and has_penalties:
                    next_id = self._apply_penalties(
                        logits, generated_ids,
                        repetition_penalty, presence_penalty,
                        frequency_penalty, temperature, top_p)
                generated_ids.append(next_id)

                if rep_detector and rep_detector.add_token(next_id):
                    stopped_by_rep = True
                    print(f"[decode] Repetition detected at token {gi+2}, "
                          f"stopping generation")
                    break

                if next_id in self.stop_ids:
                    break

                new_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=False)
                delta = new_text[len(text_so_far):]
                text_so_far = new_text
                if delta:
                    yield {"type": "token", "text": delta, "id": next_id}

            elapsed = time.time() - t0
            decode_count = len(generated_ids)

            # Store assistant response
            full_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True)
            if enable_thinking:
                self.messages.append({
                    "role": "assistant",
                    "content": "<think>\n" + full_text})
            else:
                self.messages.append({
                    "role": "assistant",
                    "content": full_text})

            stop_reason = ("repetition" if stopped_by_rep
                          else "eos" if (generated_ids
                                         and generated_ids[-1]
                                         in self.stop_ids)
                          else "length")
            print(f"[decode] {decode_count} tok in {elapsed:.1f}s, "
                  f"pos={self.pos}/{self.ctx} "
                  f"({self.pos*100//self.ctx}%), "
                  f"stop={stop_reason}")

            yield {
                "type": "done",
                "decode_tokens": decode_count,
                "end_pos": self.pos,
                "elapsed": round(elapsed, 1),
                "stop_reason": stop_reason,
            }


# ── HTTP Handler ─────────────────────────────────────────────────────

engine = None


class ChatHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            body = HTML_PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/status":
            self._send_json({
                "ready": engine.ready,
                "ctx": engine.ctx,
                "pos": engine.pos,
                "turns": len(engine.messages) // 2,
                "cache_pct": (engine.pos * 100 // engine.ctx
                              if engine.ctx else 0),
                "mode": ("combined-dedup" if engine.use_combined
                         else "separate"),
            })
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/reset":
            engine.reset()
            self._send_json({"ok": True})

        elif path == "/api/chat/stream":
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._send_json({"error": "Invalid JSON"}, 400)
                return

            message = data.get("message", "").strip()
            if not message:
                self._send_json({"error": "Empty message"}, 400)
                return

            max_tokens = min(int(data.get("max_tokens", 512)), 2048)
            enable_thinking = data.get("enable_thinking", True)
            repetition_guard = data.get("repetition_guard", True)
            temperature = float(data.get("temperature", 0.7))
            top_p = float(data.get("top_p", 0.9))
            repetition_penalty = float(data.get("repetition_penalty", 1.1))
            presence_penalty = float(data.get("presence_penalty", 0.0))
            frequency_penalty = float(data.get("frequency_penalty", 0.2))

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            try:
                for event in engine.chat_stream(
                        message, max_tokens, enable_thinking,
                        repetition_guard, temperature, top_p,
                        repetition_penalty, presence_penalty,
                        frequency_penalty):
                    line = f"data: {json.dumps(event)}\n\n"
                    self.wfile.write(line.encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    global engine

    from config import DEFAULT_OUTPUT, DEFAULT_HF_MODEL

    parser = argparse.ArgumentParser(
        description="Qwen3.5-4B ANE Chat Server")
    parser.add_argument("--model-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer dir (default: same as --model-dir)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ctx", type=int, default=1024)
    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model_dir

    engine = ChatEngine(args.model_dir, args.tokenizer, ctx=args.ctx)

    print(f"\n  Model dir: {args.model_dir}")
    print(f"  CTX: {args.ctx}, BLOCK_SIZE: {BLOCK_SIZE}, "
          f"GEN_RESERVE: {MIN_GEN_RESERVE}\n")
    engine.load()

    server = HTTPServer(("0.0.0.0", args.port), ChatHandler)
    print(f"\n  Chat server ready on http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
