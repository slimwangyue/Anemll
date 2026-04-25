#!/usr/bin/env python3
"""Browser-based multi-round chat console for Qwen3.5-4B on ANE.

Architecture:
  - CoreML models loaded at startup:
      embed_lmhead_combined (1 file, 2 functions: embed + lmhead)
      + FFN-infer chunks (N) + FFN-prefill chunks (N)
  - Embeddings and lm_head share tied weights via cross-model dedup
    in a single multifunction mlpackage (half the size).
  - lm_head outputs full logits as a single tensor (no 16-way split).
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
from collections import deque
import numpy as np
import coremltools as ct
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from transformers import AutoTokenizer
from config import FFN_LABEL
from inference_config import get_sampling_config

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
    <button id="thinkBtn" onclick="toggleThink()">Think: ON</button>
    <button onclick="toggleSettings()">Settings</button>
    <button onclick="resetChat()">New Chat</button>
  </div>
</div>
<div class="settings" id="settings">
  <label>Max tokens: <input type="number" id="maxTokens" value="4096" min="16" max="4096"></label>
  <label>Show thinking: <input type="checkbox" id="showThink" checked></label>
  <label>Thinking mode: <input type="checkbox" id="enableThinking" checked></label>
  <label>Repetition guard: <input type="checkbox" id="repGuard"></label>
  <label>Temperature: <input type="number" id="temperature" value="1.0" min="0.0" max="2.0" step="0.05"></label>
  <label>Top-p: <input type="number" id="topP" value="0.95" min="0.0" max="1.0" step="0.05"></label>
  <label>Top-k: <input type="number" id="topK" value="20" min="0" max="200" step="1"></label>
  <label>Rep penalty: <input type="number" id="repPenalty" value="1.1" min="1.0" max="2.0" step="0.05"></label>
  <label>Pres penalty: <input type="number" id="presPenalty" value="1.5" min="0.0" max="2.0" step="0.1"></label>
  <label>Freq penalty: <input type="number" id="freqPenalty" value="0.0" min="0.0" max="2.0" step="0.1"></label>
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
      d.ready ? `Ready | CTX=${d.ctx} | pos=${d.pos}/${d.ctx} (${d.cache_pct}%)` +
        (d.compactions > 0 ? ` | ${d.compactions} compactions` : '') +
        ` | ${d.turns} turns` : 'Loading models...';
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
  // Qwen3.5 official recommended defaults per mode
  if (cb.checked) {
    document.getElementById('temperature').value = '1.0';
    document.getElementById('topP').value = '0.95';
    document.getElementById('topK').value = '20';
    document.getElementById('repPenalty').value = '1.1';
    document.getElementById('presPenalty').value = '1.5';
    document.getElementById('freqPenalty').value = '0.0';
  } else {
    document.getElementById('temperature').value = '0.7';
    document.getElementById('topP').value = '0.8';
    document.getElementById('topK').value = '20';
    document.getElementById('repPenalty').value = '1.1';
    document.getElementById('presPenalty').value = '1.5';
    document.getElementById('freqPenalty').value = '0.0';
  }
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

  const maxTokens = parseInt(document.getElementById('maxTokens').value) || 4096;
  const showThink = document.getElementById('showThink').checked;
  const enableThinking = document.getElementById('enableThinking').checked;

  try {
    const resp = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text, max_tokens: maxTokens, enable_thinking: enableThinking, repetition_guard: document.getElementById('repGuard').checked, temperature: parseFloat(document.getElementById('temperature').value), top_p: parseFloat(document.getElementById('topP').value) || 0.9, top_k: parseInt(document.getElementById('topK').value) || 20, repetition_penalty: parseFloat(document.getElementById('repPenalty').value) || 1.0, presence_penalty: parseFloat(document.getElementById('presPenalty').value) || 0.0, frequency_penalty: parseFloat(document.getElementById('freqPenalty').value) || 0.0})
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

BLOCK_SIZE = 512        # logical prefill block size (matches model export)
BATCH_SIZE = 512        # prefill batch size (must match compiled model: config.py BATCH_SIZE=512)
MIN_GEN_RESERVE = 100   # minimum tokens reserved for generation after prefill
SYSTEM_PROMPT = None    # no system prompt by default (better quality for quantized models)

# Batch prefill uses _chunk_gated_delta_rule (parallelised) while sequential
# infer uses _recurrent_gated_delta_rule (token-by-token).  On ANE fp16 the
# different accumulation order produces a per-layer error (~0.006 rec_state)
# that cascades through 32 layers to produce significant hidden/logit
# divergence (max_diff ≈ 8.3 at output).  Disable batch prefill until the
# model export is fixed to use force_recurrent=True in the prefill path.
PREFILL_CROSSOVER = 32  # batch prefill for prompts >= 32 tokens (cs=32 validated on ANE)


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

def _cleanup_ane_temp():
    """Remove stale ANE compilation temps from boot drive."""
    import glob, tempfile
    tmp = tempfile.gettempdir()
    for p in glob.glob(os.path.join(tmp, "*.mlmodelc")):
        try:
            import shutil; shutil.rmtree(p)
        except Exception:
            pass
    for p in glob.glob(os.path.join(tmp, "TemporaryItems", "NSIRD_Python_*")):
        try:
            import shutil; shutil.rmtree(p)
        except Exception:
            pass


def _load_model(path, compute_unit, function_name=None):
    """Load a CoreML model from .mlpackage or .mlmodelc."""
    _cleanup_ane_temp()
    if path.endswith(".mlmodelc"):
        return ct.models.CompiledMLModel(path, compute_unit)
    kwargs = {"compute_units": compute_unit}
    if function_name:
        kwargs["function_name"] = function_name
    return ct.models.MLModel(path, **kwargs)


def _find_model(base_dir, name):
    """Find model path, preferring .mlpackage over .mlmodelc.

    coremltools cannot load .mlmodelc directly — use .mlpackage for
    models loaded via ct.models.MLModel.  .mlmodelc is only usable
    via ct.models.CompiledMLModel (no function_name support).
    """
    for ext in (".mlpackage", ".mlmodelc"):
        p = os.path.join(base_dir, name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No model found for {name} in {base_dir}")


def _find_combined_dir(model_dir):
    """Find the combined_*_dedup FFN directory inside model_dir."""
    if not os.path.isdir(model_dir):
        return None
    for name in sorted(os.listdir(model_dir)):
        if name.startswith("combined_") and name.endswith("_dedup"):
            p = os.path.join(model_dir, name)
            if os.path.isdir(p):
                return p
    return None


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

    def __init__(self, model_dir, hf_path, ctx=1024, num_chunks=4,
                 embed_lmhead_path=None, ffn_dir=None,
                 system_prompt=None, compute_unit=None):
        self.model_dir = model_dir
        self.hf_path = hf_path
        self.ctx = ctx
        self.num_chunks = num_chunks
        self.embed_lmhead_path = embed_lmhead_path
        self.ffn_dir = ffn_dir
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.compute_unit = compute_unit or ct.ComputeUnit.CPU_AND_NE
        self.ready = False
        self.lock = threading.Lock()

        # Conversation state
        self._init_messages()
        self.pos = 0           # physical write position in KV cache (0..ctx-1)
        self.rope_offset = 0   # logical_pos = pos + rope_offset (for RoPE)
        self.token_history = deque(maxlen=ctx * 2)  # tokens written to cache
        self.compaction_count = 0
        self.states = None     # CoreML model states (KV cache)
        self.lin_convs = None  # linear conv states per chunk
        self.lin_recs = None   # linear recurrent states per chunk

        # Special token IDs (filled after tokenizer loads)
        self.think_token_id = None
        self.endthink_token_id = None

        # Detect combined dedup directory (auto-find combined_*_dedup)
        self.combined_dir = _find_combined_dir(model_dir)
        self.use_combined = self.combined_dir is not None

    # ── Model loading (called once at startup) ───────────────────────

    def load(self):
        """Load all models synchronously.

        Loads separate infer and prefill MLModel instances for each
        FFN chunk.  The two instances share KV-cache state but accept
        different input shapes (seq_len=1 vs seq_len=512).
        """
        cu = self.compute_unit

        print("[engine] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_path, use_fast=False)
        self._build_stop_ids()
        self._build_special_token_ids()

        # ── Load embed + lm_head ──
        # Try separate pre-compiled .mlmodelc first (no runtime compilation),
        # then fall back to combined multifunction .mlpackage.
        _sep_embed = os.path.join(self.model_dir, "embed_single.mlmodelc")
        _sep_embed_pf = os.path.join(self.model_dir, "embed_prefill.mlmodelc")
        _sep_lmhead = os.path.join(self.model_dir, "lm_head_nosplit.mlmodelc")
        if (not self.embed_lmhead_path
                and os.path.isdir(_sep_embed)
                and os.path.isdir(_sep_embed_pf)
                and os.path.isdir(_sep_lmhead)):
            print("[engine] Loading embed + lmhead from separate .mlmodelc files...")
            self.embed = _load_model(_sep_embed, cu)
            print(f"  embed_single loaded (seq_len=1)")
            self.embed_prefill = _load_model(_sep_embed_pf, cu)
            print(f"  embed_prefill loaded (seq_len={BATCH_SIZE})")
            self.lmhead = _load_model(_sep_lmhead, cu)
            self.lmhead_mode = "logits"
            self.logits_key = "logits"
            print("  lm_head_nosplit loaded (single logits output)")
        else:
            embed_lmhead = self.embed_lmhead_path or _find_model(
                self.model_dir, "embed_lmhead_combined")
            print(f"[engine] Loading embed + lmhead from {os.path.basename(embed_lmhead)}...")
            # Try both naming conventions: combine.py uses "embed"/"embed_prefill",
            # older builds may use "embedding_decode"/"embedding_prefill"
            try:
                self.embed = _load_model(embed_lmhead, cu, function_name="embedding_decode")
                embed_fn = "embedding_decode"
            except (ValueError, RuntimeError):
                self.embed = _load_model(embed_lmhead, cu, function_name="embed")
                embed_fn = "embed"
            print(f"  {embed_fn} function loaded (seq_len=1)")
            try:
                self.embed_prefill = _load_model(embed_lmhead, cu, function_name="embedding_prefill")
                embed_pf_fn = "embedding_prefill"
            except (ValueError, RuntimeError):
                self.embed_prefill = _load_model(embed_lmhead, cu, function_name="embed_prefill")
                embed_pf_fn = "embed_prefill"
            print(f"  {embed_pf_fn} function loaded (seq_len={BATCH_SIZE})")
            self.lmhead = _load_model(embed_lmhead, cu, function_name="lmhead")
            self.lmhead_mode = "logits"
            self.logits_key = "logits"
            print("  lmhead function loaded (single logits output)")

        print("[engine] Loading FFN chunks (infer + prefill)...")
        self.ffns = []       # infer instances  (seq_len=1)
        self.prefills = []   # prefill instances (seq_len=512)
        self.has_prefill = False

        # Override combined_dir if --ffn-dir was given
        if self.ffn_dir:
            self.combined_dir = self.ffn_dir
            self.use_combined = os.path.isdir(self.ffn_dir)

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

        # Auto-detect prefill batch size from embed_prefill model
        # (overrides module-level BATCH_SIZE if model was compiled differently)
        detected_bs = self._detect_prefill_batch_size()
        if detected_bs is not None and detected_bs != BATCH_SIZE:
            print(f"[engine] ⚠️ Prefill batch size auto-detected={detected_bs} "
                  f"(overriding module BATCH_SIZE={BATCH_SIZE})")
        self._prefill_bs = detected_bs if detected_bs is not None else BATCH_SIZE

        # Pre-allocate reusable buffers to avoid per-token allocation
        self._tok_buf = np.zeros((1, 1), dtype=np.int32)
        self._mask_buf = np.full(
            (1, 1, 1, self.ctx), -65504.0, dtype=np.float16)
        self._pos_buf = np.zeros(1, dtype=np.int32)
        self._rope_buf = np.zeros(1, dtype=np.int32)  # logical RoPE position

        # Pre-allocate batch prefill buffers using detected batch size
        bs = self._prefill_bs
        self._batch_tok_buf = np.zeros((1, bs), dtype=np.int32)
        self._batch_embed_buf = np.zeros((1, bs), dtype=np.int32)
        self._valid_len_buf = np.zeros((1,), dtype=np.int32)
        self._batch_mask_buf = np.full(
            (1, 1, bs, self.ctx), -65504.0, dtype=np.float16)
        self._batch_pos_buf = np.zeros(bs, dtype=np.int32)
        self._batch_cur_buf = np.zeros(1, dtype=np.int32)

        self._reset_states()
        self.ready = True
        mode = "combined-dedup" if self.use_combined else "separate"
        prefill_mode = f"batch-{self._prefill_bs}" if self.has_prefill else "sequential"
        print(f"[engine] Ready! CTX={self.ctx}, BATCH={self._prefill_bs}, "
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

    def _build_special_token_ids(self):
        """Look up <think> and </think> token IDs for logit suppression."""
        t = self.tokenizer
        self.think_token_id = t.convert_tokens_to_ids("<think>")
        self.endthink_token_id = t.convert_tokens_to_ids("</think>")
        if self.think_token_id == t.unk_token_id:
            self.think_token_id = None
        if self.endthink_token_id == t.unk_token_id:
            self.endthink_token_id = None

    def _shapes_from_metadata(self, chunk_idx):
        """Read linear state shapes from .mlmodelc/metadata.json.

        Used when CompiledMLModel (no get_spec()) is loaded.
        Returns (conv_shape, rec_shape) tuples.
        """
        import json as _json
        conv_shape = (6, 1024, 32)   # fallback
        rec_shape = (6, 32, 128, 128)
        # Try to find metadata.json next to the loaded model
        for label in (FFN_LABEL, "LUT4", "LUT6"):
            meta_path = os.path.join(
                self.model_dir, f"ffn_{label}_chunk{chunk_idx}.mlmodelc",
                "metadata.json")
            if os.path.isfile(meta_path):
                break
        else:
            return conv_shape, rec_shape
        try:
            with open(meta_path) as f:
                meta_list = _json.load(f)
            for entry in meta_list:
                for inp in entry.get("inputSchema", []):
                    name = inp.get("name", "")
                    shp_str = inp.get("shape", "")
                    if name == "linear_conv_state" and shp_str:
                        conv_shape = tuple(
                            int(x) for x in shp_str.strip("[]").split(","))
                    elif name == "linear_recurrent_state" and shp_str:
                        rec_shape = tuple(
                            int(x) for x in shp_str.strip("[]").split(","))
        except Exception:
            pass
        return conv_shape, rec_shape

    def _detect_shapes(self):
        """Read per-chunk input shapes from model spec for state initialization.

        Layer counts may differ across chunks (e.g. 6/6/5/5/5/5),
        so linear state shapes must be detected per-chunk.
        """
        self.per_chunk_conv_shapes = []
        self.per_chunk_rec_shapes = []
        for ci in range(self.num_chunks):
            conv_shape = (6, 1024, 32)   # fallback
            rec_shape = (6, 32, 128, 128)
            try:
                spec = self.ffns[ci].get_spec()
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
                        name = inp.name
                        shp = tuple(inp.type.multiArrayType.shape)
                        if name == 'linear_conv_state':
                            conv_shape = shp
                        elif name == 'linear_recurrent_state':
                            rec_shape = shp
                    except Exception:
                        pass
            except Exception:
                # CompiledMLModel has no get_spec() — read from
                # .mlmodelc/metadata.json instead.
                conv_shape, rec_shape = self._shapes_from_metadata(ci)
            self.per_chunk_conv_shapes.append(conv_shape)
            self.per_chunk_rec_shapes.append(rec_shape)
            print(f"    chunk{ci} state: conv={conv_shape}, rec={rec_shape}")
        # Keep inp_map for backward compat (use chunk 0 shapes)
        self.inp_map = {
            'linear_conv_state': self.per_chunk_conv_shapes[0],
            'linear_recurrent_state': self.per_chunk_rec_shapes[0],
        }

        # Discover KV cache state names from model spec
        self.kv_state_names = []
        try:
            spec = self.ffns[0].get_spec()
            if hasattr(spec.description, 'state'):
                for s in spec.description.state:
                    if 'cache' in s.name.lower() or 'kv' in s.name.lower():
                        self.kv_state_names.append(s.name)
        except Exception:
            pass
        if not self.kv_state_names:
            self.kv_state_names = ['k_cache', 'v_cache']  # fallback
        print(f"    KV state names: {self.kv_state_names}")

    def _detect_prefill_batch_size(self):
        """Auto-detect prefill batch size from model specs.

        Tries embed_prefill first, then falls back to prefill chunk models,
        then to reading .mlpackage spec from disk.
        Returns detected batch size or None if detection fails.
        """
        # Strategy 1: Try embed_prefill model (works for .mlpackage)
        try:
            if hasattr(self, 'embed_prefill') and self.embed_prefill is not None:
                spec = self.embed_prefill.get_spec()
                fn_inputs = None
                if hasattr(spec.description, 'functions'):
                    for fn in spec.description.functions:
                        if fn.name in ('embed_prefill', 'embedding_prefill'):
                            fn_inputs = fn.input
                            break
                if fn_inputs is None:
                    fn_inputs = spec.description.input
                for inp in fn_inputs:
                    if inp.name == 'input_ids':
                        bs = inp.type.multiArrayType.shape[1]
                        print(f"    prefill batch_size auto-detected (embed_prefill): {bs}")
                        return bs
        except Exception:
            pass

        # Strategy 2: Try prefill chunk model (combined .mlpackage with functions)
        try:
            if hasattr(self, 'prefills') and self.prefills and self.prefills[0] is not None:
                spec = self.prefills[0].get_spec()
                fn_inputs = None
                if hasattr(spec.description, 'functions'):
                    for fn in spec.description.functions:
                        if fn.name == 'prefill':
                            fn_inputs = fn.input
                            break
                if fn_inputs is None:
                    fn_inputs = spec.description.input
                for inp in fn_inputs:
                    if inp.name == 'hidden_states':
                        shp = tuple(inp.type.multiArrayType.shape)
                        if len(shp) >= 2:
                            bs = shp[1]
                            print(f"    prefill batch_size auto-detected (chunk0 prefill): {bs}")
                            return bs
        except Exception:
            pass

        # Strategy 3: Read embed_prefill.mlpackage spec from disk
        try:
            import coremltools as ct
            for name in ("embed_prefill.mlpackage",):
                path = os.path.join(self.model_dir, name)
                if os.path.exists(path):
                    spec = ct.utils.load_spec(path)
                    for inp in spec.description.input:
                        if inp.name == 'input_ids':
                            bs = inp.type.multiArrayType.shape[1]
                            print(f"    prefill batch_size auto-detected (spec file): {bs}")
                            return bs
        except Exception:
            pass

        print("    prefill batch_size: could not auto-detect, using module default")
        return None

    def _reset_states(self):
        """Reset KV cache, linear states, and position to zero."""
        self.states = [m.make_state() for m in self.ffns]
        self.lin_convs = [
            np.zeros(self.per_chunk_conv_shapes[ci], dtype=np.float16)
            for ci in range(self.num_chunks)]
        self.lin_recs = [
            np.zeros(self.per_chunk_rec_shapes[ci], dtype=np.float16)
            for ci in range(self.num_chunks)]
        self.pos = 0
        self.rope_offset = 0
        self.token_history.clear()

    def compact_cache(self):
        """Discard left half of KV cache, keep right half via direct shift.

        Uses MLState.read_state() / write_state() to directly read the
        KV cache arrays, shift the right half to position 0, and write
        back.  This is O(cache_size) memcpy — much faster than the
        O(kept_tokens) sequential re-prefill alternative.

        After compaction:
          - Physical pos resets to keep_count (ctx // 2).
          - rope_offset increases so logical positions (for RoPE) remain
            monotonically increasing and correct.
          - token_history is trimmed to just the kept tokens.
          - Linear-attention states (conv/rec) are KEPT AS-IS.
            They are recurrent summaries that accumulate all history,
            so they still contain information from discarded tokens
            (which is actually beneficial vs rebuild-from-scratch).

        Returns True if compaction succeeded, False if nothing to compact.
        """
        keep_count = min(self.ctx // 2, self.pos, len(self.token_history))
        if keep_count <= 0:
            return False

        old_logical = self.pos + self.rope_offset
        old_physical = self.pos
        discard_count = old_physical - keep_count
        new_rope_offset = old_logical - keep_count

        print(f"[compact] Shifting KV cache: keep {keep_count}/{old_physical} "
              f"cached tokens, discard {discard_count}, "
              f"logical={old_logical}, new_offset={new_rope_offset}")
        t0 = time.time()

        # Direct KV cache shift via read_state/write_state
        for ci in range(self.num_chunks):
            for sname in self.kv_state_names:
                kv = self.states[ci].read_state(name=sname)
                # KV shape: (..., state_length, head_dim) with seq_len at axis -2
                # Standard layouts:
                #   (layers, kv_heads, CTX, head_dim) — split k_cache/v_cache
                #   (2*layers, kv_heads, CTX, head_dim) — combined kv_cache_0
                #   (kv_heads, CTX, head_dim) — per-layer k_cache_i/v_cache_i
                seq_axis = kv.ndim - 2  # always second-to-last
                shifted = np.zeros_like(kv)
                src = [slice(None)] * kv.ndim
                dst = [slice(None)] * kv.ndim
                src[seq_axis] = slice(discard_count, old_physical)
                dst[seq_axis] = slice(0, keep_count)
                shifted[tuple(dst)] = kv[tuple(src)]
                self.states[ci].write_state(name=sname, value=shifted)

        # Update position tracking
        self.pos = keep_count
        self.rope_offset = new_rope_offset

        # Trim token history to just the kept tokens
        kept_tokens = list(self.token_history)[-keep_count:]
        self.token_history = deque(kept_tokens, maxlen=self.ctx * 2)
        self.compaction_count += 1

        elapsed = time.time() - t0
        print(f"[compact] Done in {elapsed*1000:.0f}ms: "
              f"physical {old_physical}->{self.pos}, "
              f"freed {discard_count} slots, "
              f"logical_pos={self.pos + self.rope_offset}, "
              f"compactions={self.compaction_count}")
        return True

    def _init_messages(self):
        """Initialize message list, optionally with system prompt."""
        if self.system_prompt:
            self.messages = [{"role": "system", "content": self.system_prompt}]
        else:
            self.messages = []

    def reset(self):
        """Full reset: clear KV cache and conversation history."""
        with self.lock:
            self._reset_states()
            self._init_messages()
            print(f"[cache] Full reset. pos=0, messages=0")

    # ── Core model step ──────────────────────────────────────────────

    def _extract_logits(self, lm_out):
        """Extract flattened fp32 logits from lm_head output."""
        return lm_out[self.logits_key].flatten().astype(np.float32)

    def _step_kv_only(self, tok_id, pos):
        """Run one token through embed -> all FFN chunks (NO lm_head).

        Updates KV cache / state but does not compute logits.
        Used for all prefill tokens except the last one.

        pos is the physical KV-cache write index.  The logical RoPE
        position is computed as ``pos + self.rope_offset``.
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

        rope_arr = self._rope_buf
        rope_arr[0] = pos + self.rope_offset

        for ci in range(self.num_chunks):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": rope_arr,
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

        pos is the physical KV-cache write index.  The logical RoPE
        position is computed as ``pos + self.rope_offset``.
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

        rope_arr = self._rope_buf
        rope_arr[0] = pos + self.rope_offset

        for ci in range(self.num_chunks):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": rope_arr,
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
        assert 1 <= valid_len <= self._prefill_bs

        # Batch embedding: (1, _prefill_bs)  — pad with token 0
        input_ids = self._batch_tok_buf
        input_ids[0, :] = 0
        input_ids[0, :valid_len] = token_ids
        hidden = list(
            self.embed_prefill.predict({"input_ids": input_ids}).values())[0]
        # Zero-fill padding hidden states to prevent padding embeddings from
        # contributing any numerical signal through the network.
        if valid_len < self._prefill_bs:
            hidden[:, valid_len:, :] = 0.0

        # Build causal mask: (1, 1, BATCH_SIZE, CTX)
        # Valid positions get normal causal mask; padding rows get all -inf.
        mask = self._batch_mask_buf
        mask[:, :, :, :] = -65504.0
        for i in range(valid_len):
            mask[0, 0, i, :block_start + i + 1] = 0
        # Padding rows: unmask position 0 so softmax sees at least one finite
        # value.  Without this, softmax(all -inf) = 0/0 = NaN, which propagates
        # through residual connections into the linear-attention recurrent state
        # (NaN * 0 = NaN in IEEE 754), corrupting all subsequent decode tokens.
        for i in range(valid_len, self._prefill_bs):
            mask[0, 0, i, 0] = 0.0

        pos_ids = self._batch_pos_buf
        pos_ids[:valid_len] = np.arange(
            block_start + self.rope_offset,
            block_start + self.rope_offset + valid_len, dtype=np.int32)
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

        # Extract last valid token's hidden state for lm_head.
        # Prefill outputs hidden [1, BATCH_SIZE, 2560] but lm_head expects
        # [1, 1, 2560].  Slice out the (valid_len-1)-th token.
        if hidden.ndim >= 3 and hidden.shape[1] > 1:
            hidden = hidden[:, valid_len - 1:valid_len, :]

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
                         frequency_penalty=0.0,
                         temperature=0.7,
                         top_p=0.9,
                         top_k=20):
        """Apply penalties + temperature/top-p/top-k sampling, return token ID.

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

        # 2. Temperature + top-k + top-p (nucleus) sampling
        if temperature <= 0 or top_p <= 0:
            return int(np.argmax(logits))

        logits_f = logits.astype(np.float64)
        logits_f /= temperature

        # Top-k filtering (applied before softmax)
        if top_k is not None and top_k > 0 and top_k < len(logits_f):
            top_k_idx = np.argpartition(logits_f, -top_k)[-top_k:]
            mask_k = np.full_like(logits_f, -np.inf)
            mask_k[top_k_idx] = logits_f[top_k_idx]
            logits_f = mask_k

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
    # Uses true batched prefill (512 tokens at once) for all chunks
    # when prefill models are available and input length exceeds the
    # crossover threshold.  Below the threshold, sequential is faster
    # because the batch always processes a full 512-token frame.

    def _process_prompt(self, prompt_tokens):
        """PREFILL PHASE: process all prompt tokens through the model.

        Strategy (when has_prefill=True):
          - If total tokens >= PREFILL_CROSSOVER:
              Process all tokens in 512-token batched blocks.
              Full blocks use valid_len=512 (no padding).
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
        bs = self._prefill_bs
        # Batch prefill requires a full bs state write, so we
        # can only use it while pos + bs <= ctx.  When fewer
        # than bs slots remain the sequential fallback handles
        # the rest token-by-token.
        use_batch = (self.has_prefill
                     and n_total >= PREFILL_CROSSOVER
                     and self.pos + bs <= self.ctx)

        # ── Batch prefill path ──
        # Only FULL blocks (valid_len == bs) are processed via batch prefill.
        # Partial tail blocks fall through to sequential because the batch
        # prefill computation with heavy padding (>50% zeros) diverges
        # numerically from sequential on both CPU and ANE, causing wrong
        # output for some prompts.
        if use_batch:
            chunks = _chunk_tokens(prompt_tokens, bs)
            for block in chunks:
                block_len = len(block)
                if block_len < bs:
                    # Tail (partial) block — let sequential fallback handle
                    # it to avoid padding-induced numerical divergence.
                    break
                if self.pos + bs > self.ctx:
                    # Not enough state slots for a full block — hand
                    # the remaining tokens to the sequential fallback.
                    break
                t0 = time.time()
                last_next = self._batch_prefill(block, self.pos)
                elapsed = time.time() - t0
                tps = block_len / max(elapsed, 1e-9)
                n_batched += block_len
                print(f"[prefill] batch-full: {block_len} tok, "
                      f"{elapsed*1000:.0f}ms ({tps:.0f} tok/s), "
                      f"pos={self.pos}/{self.ctx}")

        # ── Sequential fallback ──
        remaining = prompt_tokens[n_batched:]
        n_remaining = len(remaining)
        if n_remaining > 0:
            t0 = time.time()
            for ti, tok_id in enumerate(remaining):
                if self.pos >= self.ctx:
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
    # Overflow policy (layered):
    #   1. Before each turn, check: pos + prompt_len + MIN_GEN_RESERVE > ctx
    #   2. If no overflow, proceed with incremental continuation
    #   3. If overflow, try strategies in order:
    #      a. KV-cache compaction: discard left half, keep right half,
    #         re-prefill from token_history.  FAST — no re-tokenization.
    #      b. Message trimming: drop oldest turn pairs, re-tokenize,
    #         full cache reset + re-prefill from scratch.  SLOWER.
    #   4. Always keep at least the current user message
    #   5. Chat continues seamlessly — user sees no disruption

    def _handle_overflow(self, prompt_tokens, enable_thinking):
        """Check cache space and handle overflow if needed.

        Strategies (tried in order):
          1. KV-cache compaction — discard left half, keep right half.
             Fast: replays exact token IDs, no re-tokenization.
          2. Message trimming — drop oldest turns, re-tokenize.
             Slower but handles cases where compaction isn't enough.

        Returns: (prompt_tokens, did_overflow)
        """
        space_needed = self.pos + len(prompt_tokens) + MIN_GEN_RESERVE
        if space_needed <= self.ctx:
            return prompt_tokens, False

        print(f"[cache] OVERFLOW: pos={self.pos} + prompt="
              f"{len(prompt_tokens)} + reserve={MIN_GEN_RESERVE} = "
              f"{space_needed} > ctx={self.ctx}")

        # Strategy 1: KV-cache compaction (fast, no re-tokenization)
        keep_count = self.ctx // 2
        freed = self.pos - keep_count
        if (freed > 0
                and keep_count + len(prompt_tokens) + MIN_GEN_RESERVE <= self.ctx):
            self.compact_cache()
            print(f"[cache] After compaction: pos={self.pos}, "
                  f"space={self.ctx - self.pos - len(prompt_tokens)}")
            return prompt_tokens, True

        # Strategy 2: message trimming (slower, re-tokenizes)
        retained = list(self.messages)
        # Preserve system message (index 0) when trimming
        sys_msg = retained[0] if retained and retained[0]["role"] == "system" else None
        while len(retained) > (2 if sys_msg else 1):
            start = 1 if sys_msg else 0  # skip system msg
            if len(retained) - start >= 3:
                retained = (retained[:1] if sys_msg else []) + retained[start + 2:]
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

        # Fallback: keep system prompt + current user message
        retained = ([sys_msg] if sys_msg else []) + [self.messages[-1]]
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
            messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking)
        if hasattr(input_ids, 'input_ids'):
            input_ids = input_ids.input_ids
            return input_ids[0].tolist()
        return list(input_ids)

    # ── Multi-round conversation ─────────────────────────────────────

    def _get_continuation_delta(self, user_msg, enable_thinking):
        """Compute continuation tokens for a follow-up turn.

        Uses apply_chat_template to render the new user message as a
        single-turn prompt, then prepends <|im_end|>\\n to close the
        previous assistant turn (whose stop token was predicted but
        not fed back into the model).

        This guarantees the prompt always matches the official Jinja
        template exactly — no manual token construction.
        """
        # Render just the new user message through the official template
        new_turn = [{"role": "user", "content": user_msg}]
        new_turn_tokens = self._tokenize_messages(
            new_turn, enable_thinking)
        # Prepend <|im_end|>\n to close the previous assistant turn
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        nl_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        return [im_end_id] + nl_ids + new_turn_tokens

    # ── Main chat stream ─────────────────────────────────────────────

    def chat_stream(self, user_msg, max_tokens=4096,
                     enable_thinking=True, repetition_guard=False,
                     temperature=0.7, top_p=0.9, top_k=20,
                     repetition_penalty=1.1, presence_penalty=0.0,
                     frequency_penalty=0.0):
        """Generator yielding SSE events for a streaming response.

        Two-phase pipeline:
          1. PREFILL — process prompt tokens (sequentially, in 512-tok blocks)
          2. DECODE  — generate response tokens one at a time
        Both phases use the same infer-function models. No switching.

        repetition_guard: if True, stops generation early when n-gram
        repetition is detected (sliding-window 5-gram, threshold=3).
        """
        with self.lock:
            is_first_turn = not any(
                m["role"] == "user" for m in self.messages)

            self.messages.append({"role": "user", "content": user_msg})

            # Build prompt tokens — always via apply_chat_template
            if is_first_turn:
                prompt_tokens = self._tokenize_messages(
                    self.messages, enable_thinking)
            else:
                prompt_tokens = self._get_continuation_delta(
                    user_msg, enable_thinking)

            turn_num = (len(self.messages) + 1) // 2
            print(f"\n{'='*60}")
            print(f"[chat] Turn {turn_num}: "
                  f"\"{user_msg[:60]}{'...' if len(user_msg)>60 else ''}\"")
            print(f"[chat] prompt_tokens ({len(prompt_tokens)}): {prompt_tokens[:30]}{'...' if len(prompt_tokens)>30 else ''}")
            print(f"[chat] prompt decoded: {repr(self.tokenizer.decode(prompt_tokens)[:200])}")
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
            # Record prompt tokens in history (for future compaction)
            self.token_history.extend(prompt_tokens)

            # ── DECODE PHASE ──
            rep_detector = RepetitionDetector() if repetition_guard else None
            generated_ids = [last_next]
            print(f"[decode] first_token={last_next} = {repr(self.tokenizer.decode([last_next]))}")
            if rep_detector:
                rep_detector.add_token(last_next)
            text_so_far = self.tokenizer.decode(
                [last_next], skip_special_tokens=True)
            # Strip trailing U+FFFD (incomplete byte-fallback tokens)
            text_so_far = text_so_far.rstrip('\ufffd')
            if text_so_far:
                yield {"type": "token", "text": text_so_far, "id": last_next}

            # Get <think> and </think> token IDs to suppress when thinking is OFF
            think_token_id = self.think_token_id
            endthink_token_id = self.endthink_token_id

            has_penalties = (repetition_penalty > 1.0
                             or presence_penalty != 0.0
                             or frequency_penalty != 0.0
                             or (temperature > 0 and temperature != 1.0)
                             or (top_p > 0 and top_p < 1.0)
                             or (top_k is not None and top_k > 0))
            if has_penalties and self.lmhead_mode == "logits":
                print(f"[decode] Sampling: temp={temperature}, top_p={top_p}, "
                      f"top_k={top_k}, rep={repetition_penalty}, "
                      f"pres={presence_penalty}, freq={frequency_penalty}")
            # Build cross-turn penalty context from recent token history
            # (last 256 tokens from previous turns provide cross-turn
            # repetition awareness)
            PENALTY_HISTORY_WINDOW = 256
            history_prefix = list(self.token_history)[-PENALTY_HISTORY_WINDOW:]

            stopped_by_rep = False
            for gi in range(max_tokens - 1):
                if self.pos >= self.ctx:
                    # KV cache full — try mid-decode compaction
                    if not self.compact_cache():
                        break
                fed_tok = generated_ids[-1]
                next_id, logits = self._step(fed_tok, self.pos)
                self.pos += 1
                self.token_history.append(fed_tok)
                if logits is not None and has_penalties:
                    # Suppress <think> and </think> tokens when thinking is OFF
                    if not enable_thinking:
                        if think_token_id is not None:
                            logits[think_token_id] = -1e9
                        if endthink_token_id is not None:
                            logits[endthink_token_id] = -1e9
                    # Include recent history from prior turns for cross-turn
                    # repetition penalty
                    penalty_ids = history_prefix + generated_ids
                    next_id = self._apply_penalties(
                        logits, penalty_ids,
                        repetition_penalty, presence_penalty,
                        frequency_penalty, temperature, top_p,
                        top_k)
                generated_ids.append(next_id)

                if len(generated_ids) <= 10:
                    print(f"[decode] tok[{len(generated_ids)-1}]={next_id} = {repr(self.tokenizer.decode([next_id]))}")

                if rep_detector and rep_detector.add_token(next_id):
                    stopped_by_rep = True
                    print(f"[decode] Repetition detected at token {gi+2}, "
                          f"stopping generation")
                    break

                if next_id in self.stop_ids:
                    break

                new_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True)
                # Strip trailing U+FFFD from incomplete byte-fallback
                # sequences (e.g. emoji split across multiple tokens).
                # Hold them back until the full character resolves.
                stable_text = new_text.rstrip('\ufffd')
                if len(stable_text) > len(text_so_far):
                    delta = stable_text[len(text_so_far):]
                    text_so_far = stable_text
                    yield {"type": "token", "text": delta, "id": next_id}

            elapsed = time.time() - t0
            decode_count = len(generated_ids)

            # Store assistant response — the official template handles
            # <think> parsing automatically via the content field.
            # When thinking=ON the model generates: reasoning\n</think>\n\nanswer
            # When thinking=OFF the model generates just the answer.
            full_text = self.tokenizer.decode(
                generated_ids, skip_special_tokens=True)
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
_model_size = "4B"


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
                "logical_pos": engine.pos + engine.rope_offset,
                "rope_offset": engine.rope_offset,
                "compactions": engine.compaction_count,
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

            max_tokens = min(int(data.get("max_tokens", 4096)), 4096)
            enable_thinking = data.get("enable_thinking", True)
            repetition_guard = data.get("repetition_guard", False)

            # ── Sampling defaults from inference_config ──
            _sc = get_sampling_config(_model_size, enable_thinking)
            def_temp = _sc["temperature"]
            def_top_p = _sc["top_p"]
            def_top_k = _sc["top_k"]
            def_rep = _sc["repetition_penalty"]
            def_pres = _sc["presence_penalty"]
            def_freq = _sc["frequency_penalty"]

            temperature = float(data.get("temperature", def_temp))
            top_p = float(data.get("top_p", def_top_p))
            top_k_val = data.get("top_k", def_top_k)
            top_k = int(top_k_val) if top_k_val is not None else def_top_k
            repetition_penalty = float(data.get("repetition_penalty", def_rep))
            presence_penalty = float(data.get("presence_penalty", def_pres))
            frequency_penalty = float(data.get("frequency_penalty", def_freq))

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            try:
                for event in engine.chat_stream(
                        message, max_tokens, enable_thinking,
                        repetition_guard, temperature, top_p, top_k,
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
    global engine, _model_size

    from config import DEFAULT_OUTPUT, DEFAULT_HF_MODEL

    parser = argparse.ArgumentParser(
        description="Qwen3.5 ANE Chat Server")
    parser.add_argument("--model-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--model-size", default="4B",
                        choices=["4B", "2B"],
                        help="Model size for sampling defaults (default: 4B)")
    parser.add_argument("--tokenizer", default=None,
                        help="Tokenizer dir (default: same as --model-dir)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--num-chunks", type=int, default=None,
                        help="Number of FFN chunks (auto-detected if not set)")
    parser.add_argument("--embed-lmhead", default=None,
                        help="Path to combined embed_lmhead_combined.mlpackage")
    parser.add_argument("--ffn-dir", default=None,
                        help="Directory containing chunk{i} combined models")
    parser.add_argument("--system-prompt", default=None,
                        help="System prompt (default: none)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Prefill batch size (default: auto-detect from model, fallback 512)")
    parser.add_argument("--compute-unit", default="all",
                        choices=["all", "cpu", "cpu_and_gpu", "cpu_and_ne"],
                        help="CoreML compute unit (default: all = CPU_AND_NE)")
    args = parser.parse_args()
    _model_size = args.model_size
    if args.tokenizer is None:
        args.tokenizer = args.model_dir

    # Auto-detect num_chunks if not specified
    num_chunks = args.num_chunks
    if num_chunks is None:
        ffn_base = args.ffn_dir or _find_combined_dir(args.model_dir)
        if ffn_base and os.path.isdir(ffn_base):
            num_chunks = sum(
                1 for f in os.listdir(ffn_base)
                if f.startswith("chunk") and (f.endswith(".mlpackage") or f.endswith(".mlmodelc"))
            )
            # Each chunk has mlpackage+mlmodelc, deduplicate
            if num_chunks > 6:
                num_chunks //= 2
        if not num_chunks:
            num_chunks = 4  # fallback
        print(f"  Auto-detected {num_chunks} FFN chunks")

    _cu_map = {
        "all": ct.ComputeUnit.CPU_AND_NE,
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
    }
    compute_unit = _cu_map[args.compute_unit]
    print(f"  Compute unit: {compute_unit}")

    # Override BATCH_SIZE / BLOCK_SIZE from CLI or auto-detect from model
    global BATCH_SIZE, BLOCK_SIZE
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
        BLOCK_SIZE = args.batch_size
        print(f"  Batch size (CLI): {BATCH_SIZE}")
    else:
        # Auto-detect from embed_prefill model input shape
        _detect_dir = args.model_dir
        for _ep_name in ("embed_prefill.mlpackage", "embed_prefill.mlmodelc"):
            _ep_path = os.path.join(_detect_dir, _ep_name)
            if os.path.exists(_ep_path):
                try:
                    _ep_spec = ct.utils.load_spec(_ep_path)
                    for inp in _ep_spec.description.input:
                        if inp.name == "input_ids":
                            _detected = inp.type.multiArrayType.shape[1]
                            BATCH_SIZE = _detected
                            BLOCK_SIZE = _detected
                            print(f"  Batch size (auto-detected): {BATCH_SIZE}")
                            break
                except Exception as e:
                    print(f"  Warning: could not auto-detect batch size: {e}")
                break

    engine = ChatEngine(args.model_dir, args.tokenizer, ctx=args.ctx,
                        num_chunks=num_chunks,
                        embed_lmhead_path=args.embed_lmhead,
                        ffn_dir=args.ffn_dir,
                        system_prompt=args.system_prompt,
                        compute_unit=compute_unit)

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
