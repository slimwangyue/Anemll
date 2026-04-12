#!/usr/bin/env python3
"""
Reproduce Apple's ml-ane-transformers MultiHeadAttention and measure ANE utilization.

Directly implements Apple's reference code from:
  https://github.com/apple/ml-ane-transformers/blob/main/ane_transformers/reference/multihead_attention.py

Tests:
  A) Apple's exact MHA (per-head einsum, channels-first BC1S)
  B) Standard PyTorch MHA (batched matmul, BHSD format)
  C) Apple MHA with causal mask (generative LM style)

For each, exports CoreML mlpackage and measures ANE% at various seq_len / ctx.

Usage:
  python tests/dev/apple_ane_mha_repro.py
  python tests/dev/apple_ane_mha_repro.py --embed-dim 2560 --n-head 16 --ctx 128 256 512 1024 2048
  python tests/dev/apple_ane_mha_repro.py --skip-export
"""
import sys, os, time, gc, math, warnings, argparse
import numpy as np

REPO_ROOT = '/Volumes/MySSD/Anemll'
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
warnings.filterwarnings('ignore', category=UserWarning)
os.environ.setdefault('TMPDIR', '/Volumes/MySSD/tmp')

import torch
import torch.nn as nn
torch.set_grad_enabled(False)
import coremltools as ct
from collections import Counter

ARTIFACT_DIR = os.path.join(REPO_ROOT, 'artifacts', 'apple_ane_mha_repro')
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# Apple's LayerNormANE (verbatim from ml-ane-transformers)
# ═══════════════════════════════════════════════════════════════════════

class LayerNormANE(nn.Module):
    """LayerNorm optimized for ANE. Expects BC1S input format."""

    def __init__(self, num_channels, clip_mag=None, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.expected_rank = 4  # BC1S
        self.num_channels = num_channels
        self.eps = eps
        self.clip_mag = clip_mag
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, inputs):
        input_rank = len(inputs.size())
        if input_rank == 3 and inputs.size(2) == self.num_channels:
            inputs = inputs.transpose(1, 2).unsqueeze(2)
            input_rank = len(inputs.size())
        assert input_rank == self.expected_rank
        assert inputs.size(1) == self.num_channels
        if self.clip_mag is not None:
            inputs.clamp_(-self.clip_mag, self.clip_mag)
        channels_mean = inputs.mean(dim=1, keepdims=True)
        zero_mean = inputs - channels_mean
        zero_mean_sq = zero_mean * zero_mean
        denom = (zero_mean_sq.mean(dim=1, keepdims=True) + self.eps).rsqrt()
        out = zero_mean * denom
        if self.elementwise_affine:
            out = (out + self.bias.view(1, self.num_channels, 1, 1)
                   ) * self.weight.view(1, self.num_channels, 1, 1)
        return out


# ═══════════════════════════════════════════════════════════════════════
# Apple's MultiHeadAttention (verbatim from ml-ane-transformers)
# ═══════════════════════════════════════════════════════════════════════

class AppleMultiHeadAttention(nn.Module):
    """Multi-Head Attention optimized for efficient ANE deployment.
    Directly from Apple's ml-ane-transformers reference."""

    def __init__(self, embed_dim, d_qk=None, d_v=None, d_out=None,
                 n_head=8, dropout=0.0):
        super().__init__()
        self.d_qk = d_qk or embed_dim
        self.d_v = d_v or embed_dim
        self.d_out = d_out or embed_dim
        self.n_head = n_head
        if self.d_qk % self.n_head != 0 or self.d_v % self.n_head != 0:
            raise ValueError(
                f"Either query-key dimensions ({self.d_qk}) or the value embeddings "
                f"dimensions ({self.d_v}) is not divisible by n_head ({self.n_head})"
            )
        self.q_normalize_fact = float(self.d_qk // self.n_head) ** -0.5
        self.q_proj = nn.Conv2d(embed_dim, self.d_qk, 1)
        self.v_proj = nn.Conv2d(embed_dim, self.d_v, 1)
        self.k_proj = nn.Conv2d(embed_dim, self.d_qk, 1)
        self.out_proj = nn.Conv2d(self.d_v, self.d_out, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        self.apply(self._reset_parameters)

    @staticmethod
    def _reset_parameters(module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.)

    def _attention_fn(self, q, k, v, qk_mask, k_mask, return_weights):
        """Core per-head einsum attention.
        q: (B, d_qk, 1, tgt_seq_len)
        k: (B, d_qk, 1, src_seq_len)
        v: (B, d_v, 1, src_seq_len)
        """
        # Split into per-head chunks
        mh_q = q.split(self.d_qk // self.n_head, dim=1)
        mh_k = k.transpose(1, 3).split(self.d_qk // self.n_head, dim=3)
        mh_v = v.split(self.d_v // self.n_head, dim=1)

        # Per-head Q·K^T via einsum
        attn_weights = [
            torch.einsum('bchq,bkhc->bkhq', [qi, ki]) * self.q_normalize_fact
            for qi, ki in zip(mh_q, mh_k)
        ]

        # Apply masks
        if qk_mask is not None:
            for head_idx in range(self.n_head):
                attn_weights[head_idx] = attn_weights[head_idx] + qk_mask
        if k_mask is not None:
            for head_idx in range(self.n_head):
                attn_weights[head_idx] = attn_weights[head_idx] + k_mask

        # Per-head softmax
        attn_weights = [aw.softmax(dim=1) for aw in attn_weights]
        mh_w = [self.dropout(aw) for aw in attn_weights]

        # Per-head attn·V via einsum
        attn = [
            torch.einsum('bkhq,bchk->bchq', wi, vi)
            for wi, vi in zip(mh_w, mh_v)
        ]
        attn = torch.cat(attn, dim=1)

        if return_weights:
            return attn, attn_weights
        return attn, None

    def forward(self, q, k, v, qk_mask=None, k_mask=None, return_weights=False):
        assert len(q.size()) == 4 and len(k.size()) == 4 and len(v.size()) == 4
        b, ct, ht, wt = q.size()
        b, cs, hs, ws = k.size()
        tgt_seq_len = ht * wt
        src_seq_len = hs * ws

        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        attn, attn_weights = self._attention_fn(q, k, v, qk_mask, k_mask, return_weights)
        attn = attn.contiguous().view(b, self.d_v, ht, wt)
        attn = self.out_proj(attn)

        if return_weights:
            return attn, attn_weights
        return attn, None


# ═══════════════════════════════════════════════════════════════════════
# Standard PyTorch MHA (batched matmul, for comparison)
# ═══════════════════════════════════════════════════════════════════════

class StandardMultiHeadAttention(nn.Module):
    """Standard MHA using torch.matmul in (B,H,S,D) format."""

    def __init__(self, embed_dim, n_head=8, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_head = n_head
        self.head_dim = embed_dim // n_head
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, 1)
        self.apply(self._reset_parameters)

    @staticmethod
    def _reset_parameters(module):
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.)

    def forward(self, q, k, v, qk_mask=None, k_mask=None, return_weights=False):
        B, C, _, S_q = q.shape
        _, _, _, S_k = k.shape

        q = self.q_proj(q)  # (B, C, 1, S_q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Reshape to (B, H, S, D)
        q = q.squeeze(2).view(B, self.n_head, self.head_dim, S_q).permute(0, 1, 3, 2)  # (B,H,S_q,D)
        k = k.squeeze(2).view(B, self.n_head, self.head_dim, S_k).permute(0, 1, 3, 2)  # (B,H,S_k,D)
        v = v.squeeze(2).view(B, self.n_head, self.head_dim, S_k).permute(0, 1, 3, 2)  # (B,H,S_k,D)

        # Batched matmul attention
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B,H,S_q,S_k)

        if qk_mask is not None:
            # Convert ANE mask format (B,S_k,1,S_q) → (B,1,S_q,S_k) for standard format
            attn_weights = attn_weights + qk_mask.squeeze(2).permute(0, 2, 1).unsqueeze(1)

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)  # (B,H,S_q,D)

        # Back to channels-first: (B,H,S_q,D) → (B,C,1,S_q)
        attn_out = attn_out.permute(0, 1, 3, 2).reshape(B, C, 1, S_q)
        attn_out = self.out_proj(attn_out)

        return attn_out, None


# ═══════════════════════════════════════════════════════════════════════
# Wrapper modules for tracing
# ═══════════════════════════════════════════════════════════════════════

class AppleSelfAttnWrapper(nn.Module):
    """Self-attention wrapper: query = key = value. Channels-first BC1S."""
    def __init__(self, embed_dim, n_head):
        super().__init__()
        self.mha = AppleMultiHeadAttention(embed_dim, n_head=n_head, dropout=0.0)

    def forward(self, x, qk_mask=None):
        out, _ = self.mha(x, x, x, qk_mask=qk_mask)
        return out


class AppleCrossAttnWrapper(nn.Module):
    """Decode-style: Q is (B,C,1,1) from single token, KV is (B,C,1,CTX) from cache."""
    def __init__(self, embed_dim, n_head):
        super().__init__()
        self.mha = AppleMultiHeadAttention(embed_dim, n_head=n_head, dropout=0.0)

    def forward(self, q, kv, qk_mask=None):
        out, _ = self.mha(q, kv, kv, qk_mask=qk_mask)
        return out


class StandardSelfAttnWrapper(nn.Module):
    def __init__(self, embed_dim, n_head):
        super().__init__()
        self.mha = StandardMultiHeadAttention(embed_dim, n_head=n_head, dropout=0.0)

    def forward(self, x, qk_mask=None):
        out, _ = self.mha(x, x, x, qk_mask=qk_mask)
        return out


class StandardCrossAttnWrapper(nn.Module):
    def __init__(self, embed_dim, n_head):
        super().__init__()
        self.mha = StandardMultiHeadAttention(embed_dim, n_head=n_head, dropout=0.0)

    def forward(self, q, kv, qk_mask=None):
        out, _ = self.mha(q, kv, kv, qk_mask=qk_mask)
        return out


# ═══════════════════════════════════════════════════════════════════════
# Export & Measurement
# ═══════════════════════════════════════════════════════════════════════

def export_model(wrapper, inputs, input_specs, output_name, label, path):
    """Trace and convert to CoreML."""
    traced = torch.jit.trace(wrapper, inputs, check_trace=False)
    mlmodel = ct.convert(
        traced,
        inputs=input_specs,
        outputs=[ct.TensorType(name=output_name, dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS18,
        convert_to="mlprogram",
    )
    mlmodel.save(path)
    del mlmodel, traced; gc.collect()
    return path


def analyze_mil(path, label):
    """Count MIL ops."""
    spec = ct.utils.load_spec(path)
    prog = spec.mlProgram
    for fname in prog.functions.keys():
        func = prog.functions[fname]
        oc = Counter()
        hostile = []
        for _, blk in func.block_specializations.items():
            for op in blk.operations:
                oc[op.type] += 1
                if op.type in ('gather', 'gather_along_axis', 'greater_equal', 'select',
                               'read_state', 'coreml_update_state'):
                    hostile.append(op.type)
        h = dict(Counter(hostile)) if hostile else "NONE"
        mm = oc.get('matmul', 0) + oc.get('einsum', 0)
        sm = oc.get('softmax', 0)
        conv_cnt = oc.get('conv', 0)
        total = sum(oc.values())
        print(f"  {label:50s} ops={total:4d} conv={conv_cnt:2d} "
              f"matmul/einsum={mm:2d} softmax={sm:2d} hostile={h}")
        sys.stdout.flush()
        return oc


def measure_ane(path, label, cu=ct.ComputeUnit.CPU_AND_NE, warmup=10, runs=30):
    """Measure ANE utilization."""
    ml = ct.models.MLModel(path, compute_units=cu)
    spec = ct.utils.load_spec(path)
    fn = list(spec.mlProgram.functions.keys())[0]
    func = spec.mlProgram.functions[fn]
    inputs_dict = {}
    for inp in func.inputs:
        if inp.type.WhichOneof('type') == 'tensorType':
            tt = inp.type.tensorType
            shape = tuple(d.constant.size for d in tt.dimensions)
            dt_map = {1: np.float32, 3: np.float16, 5: np.int32}
            inputs_dict[inp.name] = np.random.randn(*shape).astype(
                dt_map.get(tt.dataType, np.float16))
    for _ in range(warmup):
        ml.predict(inputs_dict)
    walls, cpus = [], []
    for _ in range(runs):
        tw = time.perf_counter(); tc = time.process_time()
        ml.predict(inputs_dict)
        cpus.append(time.process_time() - tc)
        walls.append(time.perf_counter() - tw)
    mw = sorted(walls)[len(walls) // 2] * 1000
    mc = sorted(cpus)[len(cpus) // 2] * 1000
    cf = mc / mw if mw > 0 else 1.0
    af = max(0, 1 - cf)
    cu_s = 'ANE' if cu == ct.ComputeUnit.CPU_AND_NE else 'CPU'
    print(f"  {label:50s} [{cu_s}] wall={mw:7.2f}ms "
          f"cpu={mc:7.2f}ms  CPU%={cf * 100:5.1f}%  ANE%={af * 100:5.1f}%")
    sys.stdout.flush()
    del ml; gc.collect()
    return mw, mc, af


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--embed-dim", type=int, default=None,
                        help="Embedding dim (default: test 256 and 2560)")
    parser.add_argument("--n-head", type=int, default=None,
                        help="Number of heads (default: matches embed-dim)")
    parser.add_argument("--ctx", type=int, nargs='+', default=None,
                        help="Context lengths (default: 32 64 128 256 512 1024 2048)")
    args = parser.parse_args()

    # ── Test Configurations ──
    if args.embed_dim:
        configs = [(args.embed_dim, args.n_head or args.embed_dim // 64)]
    else:
        configs = [
            (256, 8),      # Small: DistilBERT-like (Apple's reference)
            (2560, 16),    # Qwen3.5-4B sized
        ]

    CTX_LIST = args.ctx or [32, 64, 128, 256, 512, 1024, 2048]

    # Test modes:
    #  "self_attn"  — full self-attention: q=k=v all (B,C,1,S) same
    #  "cross_attn" — decode-style: q=(B,C,1,1), kv=(B,C,1,CTX)
    MODES = ["self_attn", "cross_attn"]

    print("=" * 90)
    print("  APPLE ml-ane-transformers MHA REPRODUCTION")
    print("=" * 90)
    print(f"  Configs: {configs}")
    print(f"  CTX: {CTX_LIST}")
    print(f"  Modes: {MODES}")
    print(f"  Variants: Apple (per-head einsum) vs Standard (batched matmul)")
    print("=" * 90)
    sys.stdout.flush()

    # ── Export Phase ──
    all_paths = {}  # (variant, mode, embed_dim, n_head, ctx) → path

    if not args.skip_export:
        for embed_dim, n_head in configs:
            for ctx in CTX_LIST:
                for variant in ["apple", "standard"]:
                    for mode in MODES:
                        tag = f"{variant}_{mode}_e{embed_dim}_h{n_head}_ctx{ctx}"
                        path = os.path.join(ARTIFACT_DIR, f"{tag}.mlpackage")
                        all_paths[(variant, mode, embed_dim, n_head, ctx)] = path

                        if os.path.exists(path):
                            print(f"  SKIP (exists): {tag}")
                            sys.stdout.flush()
                            continue

                        print(f"  Exporting {tag}...", end=" ", flush=True)
                        t0 = time.time()

                        if mode == "self_attn":
                            x = torch.randn(1, embed_dim, 1, ctx, dtype=torch.float32)
                            if variant == "apple":
                                wrapper = AppleSelfAttnWrapper(embed_dim, n_head).eval()
                                inputs = (x,)
                                specs = [ct.TensorType("x", shape=x.shape, dtype=np.float16)]
                            else:
                                wrapper = StandardSelfAttnWrapper(embed_dim, n_head).eval()
                                inputs = (x,)
                                specs = [ct.TensorType("x", shape=x.shape, dtype=np.float16)]

                        elif mode == "cross_attn":
                            q = torch.randn(1, embed_dim, 1, 1, dtype=torch.float32)
                            kv = torch.randn(1, embed_dim, 1, ctx, dtype=torch.float32)
                            if variant == "apple":
                                wrapper = AppleCrossAttnWrapper(embed_dim, n_head).eval()
                                inputs = (q, kv)
                                specs = [
                                    ct.TensorType("q", shape=q.shape, dtype=np.float16),
                                    ct.TensorType("kv", shape=kv.shape, dtype=np.float16),
                                ]
                            else:
                                wrapper = StandardCrossAttnWrapper(embed_dim, n_head).eval()
                                inputs = (q, kv)
                                specs = [
                                    ct.TensorType("q", shape=q.shape, dtype=np.float16),
                                    ct.TensorType("kv", shape=kv.shape, dtype=np.float16),
                                ]

                        export_model(wrapper, inputs, specs, "output", tag, path)
                        print(f"{time.time() - t0:.1f}s")
                        gc.collect()
    else:
        # Build paths for existing exports
        for embed_dim, n_head in configs:
            for ctx in CTX_LIST:
                for variant in ["apple", "standard"]:
                    for mode in MODES:
                        tag = f"{variant}_{mode}_e{embed_dim}_h{n_head}_ctx{ctx}"
                        path = os.path.join(ARTIFACT_DIR, f"{tag}.mlpackage")
                        all_paths[(variant, mode, embed_dim, n_head, ctx)] = path

    # ── MIL Analysis (one representative CTX per config) ──
    print("\n" + "=" * 90)
    print("  MIL OP ANALYSIS")
    print("=" * 90)
    rep_ctx = CTX_LIST[len(CTX_LIST) // 2]  # middle CTX
    for embed_dim, n_head in configs:
        for variant in ["apple", "standard"]:
            for mode in MODES:
                key = (variant, mode, embed_dim, n_head, rep_ctx)
                p = all_paths.get(key)
                if p and os.path.exists(p):
                    tag = f"{variant}_{mode}_e{embed_dim}_h{n_head}_ctx{rep_ctx}"
                    analyze_mil(p, tag)
    sys.stdout.flush()

    # ── ANE Measurement ──
    print("\n" + "=" * 90)
    print("  ANE UTILIZATION (CPU_AND_NE, 30 runs, 10 warmup)")
    print("=" * 90)

    results = {}  # key → (wall, cpu, ane_frac)
    for embed_dim, n_head in configs:
        print(f"\n  ═══ embed_dim={embed_dim}, n_head={n_head} ═══")
        for mode in MODES:
            print(f"\n  --- {mode} ---")
            for variant in ["apple", "standard"]:
                for ctx in CTX_LIST:
                    key = (variant, mode, embed_dim, n_head, ctx)
                    p = all_paths.get(key)
                    if p and os.path.exists(p):
                        tag = f"{variant:8s} {mode:10s} ctx={ctx}"
                        w, c, a = measure_ane(p, tag)
                        results[key] = (w, c, a)

    # CPU_ONLY reference (just smallest and largest CTX)
    print(f"\n  --- CPU_ONLY reference ---")
    results_cpu = {}
    for embed_dim, n_head in configs:
        for mode in MODES:
            for variant in ["apple", "standard"]:
                for ctx in [CTX_LIST[0], CTX_LIST[-1]]:
                    key = (variant, mode, embed_dim, n_head, ctx)
                    p = all_paths.get(key)
                    if p and os.path.exists(p):
                        tag = f"{variant:8s} {mode:10s} e{embed_dim} ctx={ctx}"
                        w, c, a = measure_ane(p, tag, cu=ct.ComputeUnit.CPU_ONLY)
                        results_cpu[key] = (w, c, a)

    # ── Summary Tables ──
    for embed_dim, n_head in configs:
        for mode in MODES:
            print(f"\n{'=' * 90}")
            print(f"  SUMMARY: embed_dim={embed_dim}, n_head={n_head}, mode={mode}")
            print(f"{'=' * 90}")
            print(f"  {'CTX':>5s}  "
                  f"{'Apple wall':>11s} {'ANE%':>6s}  "
                  f"{'Std wall':>11s} {'ANE%':>6s}  "
                  f"{'Apple/Std':>9s}")
            print(f"  {'-' * 5}  {'-' * 11} {'-' * 6}  {'-' * 11} {'-' * 6}  {'-' * 9}")

            for ctx in CTX_LIST:
                ka = ("apple", mode, embed_dim, n_head, ctx)
                ks = ("standard", mode, embed_dim, n_head, ctx)
                print(f"  {ctx:5d}  ", end="")
                if ka in results:
                    wa, ca, aa = results[ka]
                    print(f"{wa:9.2f}ms {aa * 100:5.1f}%  ", end="")
                else:
                    print(f"{'N/A':>11s} {'N/A':>6s}  ", end="")
                if ks in results:
                    ws, cs, sa = results[ks]
                    print(f"{ws:9.2f}ms {sa * 100:5.1f}%  ", end="")
                else:
                    print(f"{'N/A':>11s} {'N/A':>6s}  ", end="")
                if ka in results and ks in results:
                    ratio = results[ka][0] / results[ks][0]
                    print(f"{ratio:7.2f}x", end="")
                print()

    # ── Final Verdict ──
    print(f"\n{'=' * 90}")
    apple_ane = [(k, v) for k, v in results.items() if k[0] == "apple" and v[2] > 0.05]
    std_ane = [(k, v) for k, v in results.items() if k[0] == "standard" and v[2] > 0.05]
    total_apple = len([k for k in results if k[0] == "apple"])
    total_std = len([k for k in results if k[0] == "standard"])
    print(f"  Apple ANE einsum:  {len(apple_ane)}/{total_apple} configs with >5% ANE")
    print(f"  Standard matmul:   {len(std_ane)}/{total_std} configs with >5% ANE")
    if apple_ane:
        best = max(apple_ane, key=lambda x: x[1][2])
        k, v = best
        print(f"  Best Apple ANE%:   {v[2]*100:.1f}% at e={k[2]}, h={k[3]}, ctx={k[4]}, mode={k[1]}")
    if std_ane:
        best = max(std_ane, key=lambda x: x[1][2])
        k, v = best
        print(f"  Best Standard ANE%: {v[2]*100:.1f}% at e={k[2]}, h={k[3]}, ctx={k[4]}, mode={k[1]}")
    print(f"{'=' * 90}")
    print("  Done!")
