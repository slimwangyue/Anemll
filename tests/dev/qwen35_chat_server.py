#!/usr/bin/env python3
"""Browser-based multi-round chat console for Qwen3.5-4B on ANE.

Usage:
    python tests/dev/qwen35_chat_server.py [--port 8080]
    python tests/dev/qwen35_chat_server.py --model-dir /path/to/models --tokenizer /path/to/hf

Then open http://localhost:8080 in your browser.
"""
import sys
sys.path.insert(0, "/Users/yw68/Anemll")

import os, gc, time, json, argparse, threading
import numpy as np
import coremltools as ct
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse
from transformers import AutoTokenizer

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
      d.ready ? `Ready | CTX=${d.ctx} | pos=${d.pos} | ${d.turns} turns` : 'Loading models...';
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
  // Simple markdown: code blocks, inline code, bold
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
      body: JSON.stringify({message: text, max_tokens: maxTokens, enable_thinking: enableThinking})
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let thinkText = '';
    let answerText = '';
    let inThink = enableThinking;  // only parse thinking block when thinking mode is on
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
          const tok = ev.text;
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

          // Update display
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
          const meta = `${ev.decode_tokens} tokens | ${elapsed.toFixed(1)}s | ${(ev.decode_tokens/elapsed).toFixed(1)} tok/s | pos=${ev.end_pos}`;
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

async function resetChat() {
  await fetch('/api/reset', {method: 'POST'});
  document.getElementById('chat').innerHTML = '';
  checkStatus();
}

// Auto-resize textarea
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

# ── Model Engine ─────────────────────────────────────────────────────

class ChatEngine:
    def __init__(self, model_dir, hf_path, ctx=1024, num_chunks=4):
        self.model_dir = model_dir
        self.hf_path = hf_path
        self.ctx = ctx
        self.num_chunks = num_chunks
        self.ready = False
        self.lock = threading.Lock()

        # Conversation state
        self.messages = []
        self.pos = 0
        self.states = None
        self.lin_convs = None
        self.lin_recs = None

        # Template token IDs (filled after tokenizer loads)
        self.tpl_tokens = {}

    def load(self):
        """Load models (call in background thread)."""
        print("[engine] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_path, use_fast=False)
        self._build_stop_ids()
        self._build_template_tokens()

        print("[engine] Loading embeddings...")
        self.embed = ct.models.MLModel(
            os.path.join(self.model_dir, "embeddings.mlpackage"),
            compute_units=ct.ComputeUnit.CPU_AND_NE)

        print("[engine] Loading lm_head...")
        self.lmhead = ct.models.MLModel(
            os.path.join(self.model_dir, "lm_head.mlpackage"),
            compute_units=ct.ComputeUnit.CPU_AND_NE)

        print("[engine] Loading FFN chunks...")
        self.ffns = []
        for ci in range(self.num_chunks):
            print(f"  chunk {ci}...")
            m = ct.models.MLModel(
                os.path.join(self.model_dir, f"ffn_LUT4_chunk{ci}.mlpackage"),
                compute_units=ct.ComputeUnit.CPU_AND_NE)
            self.ffns.append(m)

        # Get input shapes
        spec = self.ffns[0].get_spec()
        self.inp_map = {}
        for inp in spec.description.input:
            try:
                self.inp_map[inp.name] = tuple(inp.type.multiArrayType.shape)
            except Exception:
                pass

        self._reset_states()
        self.ready = True
        print(f"[engine] Ready! CTX={self.ctx}, stop_ids={self.stop_ids}")

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
            "nl": t.convert_tokens_to_ids("\n"),
            "think": t.convert_tokens_to_ids("<think>"),
            "user": t.encode("user", add_special_tokens=False),
            "assistant": t.encode("assistant", add_special_tokens=False),
        }

    def _reset_states(self):
        self.states = [m.make_state() for m in self.ffns]
        self.lin_convs = [np.zeros(self.inp_map['linear_conv_state'], dtype=np.float16)
                          for _ in range(self.num_chunks)]
        self.lin_recs = [np.zeros(self.inp_map['linear_recurrent_state'], dtype=np.float16)
                         for _ in range(self.num_chunks)]
        self.pos = 0

    def reset(self):
        with self.lock:
            self._reset_states()
            self.messages = []

    def _step(self, tok_id, pos):
        """Run one token through the full pipeline. Returns next token id."""
        tok = np.array([[tok_id]], dtype=np.int32)
        hidden = list(self.embed.predict({"input_ids": tok}).values())[0]

        mask = np.full((1, 1, 1, self.ctx), -65504.0, dtype=np.float16)
        mask[:, :, :, :pos + 1] = 0

        for ci in range(self.num_chunks):
            inp = {
                "hidden_states": hidden.astype(np.float16),
                "position_ids": np.array([pos], dtype=np.int32),
                "causal_mask": mask,
                "current_pos": np.array([pos], dtype=np.int32),
                "linear_conv_state": self.lin_convs[ci],
                "linear_recurrent_state": self.lin_recs[ci],
            }
            out = self.ffns[ci].predict(inp, state=self.states[ci])
            hidden = out["output_hidden_states"]
            if 'linear_conv_state_out' in out:
                self.lin_convs[ci] = out['linear_conv_state_out']
                self.lin_recs[ci] = out['linear_recurrent_state_out']

        lm_out = self.lmhead.predict({"hidden_states": hidden.astype(np.float16)})
        if "logits" in lm_out:
            return int(np.argmax(lm_out["logits"].flatten()))
        return int(lm_out["argmax_idx"].flatten()[0])

    def _build_continuation_tokens(self, user_msg, has_prev_stop, enable_thinking=True):
        """Build token sequence for a new turn (incremental mode)."""
        t = self.tpl_tokens
        tk = self.tokenizer
        tokens = []
        if not has_prev_stop:
            tokens += [t["im_end"], t["nl"]]
        else:
            tokens += [t["nl"]]
        tokens += [t["im_start"]] + t["user"] + [t["nl"]]
        tokens += tk.encode(user_msg, add_special_tokens=False)
        tokens += [t["im_end"], t["nl"]]
        tokens += [t["im_start"]] + t["assistant"] + [t["nl"]]
        if enable_thinking:
            tokens += [t["think"], t["nl"]]
        return tokens

    def chat_stream(self, user_msg, max_tokens=512, enable_thinking=True):
        """Generator that yields SSE events for streaming response."""
        with self.lock:
            is_first_turn = len(self.messages) == 0

            if is_first_turn:
                # First turn: use full template
                self.messages.append({"role": "user", "content": user_msg})
                input_ids = self.tokenizer.apply_chat_template(
                    self.messages, return_tensors="pt",
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking)
                if hasattr(input_ids, 'input_ids'):
                    input_ids = input_ids.input_ids
                prompt_tokens = input_ids[0].tolist()
            else:
                # Subsequent turns: incremental continuation
                self.messages.append({"role": "user", "content": user_msg})
                prompt_tokens = self._build_continuation_tokens(user_msg, has_prev_stop=True, enable_thinking=enable_thinking)

            # Check context budget
            total_needed = self.pos + len(prompt_tokens) + max_tokens
            if total_needed > self.ctx:
                remaining = self.ctx - self.pos - len(prompt_tokens)
                if remaining < 10:
                    yield {"type": "error", "message": f"Context full ({self.pos}/{self.ctx}). Reset the chat."}
                    return
                max_tokens = min(max_tokens, remaining)

            # Prefill prompt tokens
            t0 = time.time()
            for tok_id in prompt_tokens:
                if self.pos >= self.ctx - 1:
                    yield {"type": "error", "message": "Context limit reached during prefill."}
                    return
                last_next = self._step(tok_id, self.pos)
                self.pos += 1

            # Decode
            generated = []
            generated_ids = [last_next]
            text_so_far = self.tokenizer.decode([last_next], skip_special_tokens=False)
            yield {"type": "token", "text": text_so_far, "id": last_next}

            for gi in range(max_tokens - 1):
                if self.pos >= self.ctx - 1:
                    break
                next_id = self._step(generated_ids[-1], self.pos)
                self.pos += 1
                generated_ids.append(next_id)

                # Decode incrementally
                new_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
                delta = new_text[len(text_so_far):]
                text_so_far = new_text

                if delta:
                    yield {"type": "token", "text": delta, "id": next_id}

                if next_id in self.stop_ids:
                    break

            elapsed = time.time() - t0
            decode_count = len(generated_ids)

            # Save assistant response for conversation history
            full_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            if enable_thinking:
                self.messages.append({"role": "assistant", "content": "<think>\n" + full_text})
            else:
                self.messages.append({"role": "assistant", "content": full_text})

            yield {
                "type": "done",
                "decode_tokens": decode_count,
                "end_pos": self.pos,
                "elapsed": round(elapsed, 1),
            }


# ── HTTP Handler ─────────────────────────────────────────────────────

engine = None  # Global engine instance

class ChatHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress default access logs
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

            # SSE response
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            try:
                for event in engine.chat_stream(message, max_tokens, enable_thinking):
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

    parser = argparse.ArgumentParser(description="Qwen3.5-4B ANE Chat Server")
    parser.add_argument("--model-dir", default="/Users/yw68/Anemll_remote_run/qwen35_milestone1")
    parser.add_argument("--tokenizer", default="/Users/yw68/local_llm/models/Qwen__Qwen3.5-4B")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ctx", type=int, default=1024)
    args = parser.parse_args()

    engine = ChatEngine(args.model_dir, args.tokenizer, ctx=args.ctx)

    # Load models in background thread
    def load_models():
        engine.load()
    load_thread = threading.Thread(target=load_models, daemon=True)
    load_thread.start()

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    server = ThreadedHTTPServer(("0.0.0.0", args.port), ChatHandler)
    print(f"\n  Chat server starting on http://localhost:{args.port}")
    print(f"  Model dir: {args.model_dir}")
    print(f"  CTX: {args.ctx}")
    print(f"  Press Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == "__main__":
    main()
