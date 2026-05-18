#!/usr/bin/env python3
"""unified_repr.py - Anchor-bias analysis for code-switched QA.

Computes per-example, per-layer hidden-state similarity between EN/CS/TGT
question variants and derives the **anchor bias**

    AB_l(q_cs) = s_l(q_src, q_cs) - s_l(q_tgt, q_cs)

over the question-content token span, with per-layer anisotropy correction
and whitening-based cosine. Late-layer (last 25 percent) and upper-layer
(final 30 percent) means are reported as the primary diagnostic.

Usage:
  python unified_repr.py analyze --data_jsonl data/pairs_with_tgt.jsonl \\
      --out_dir outputs/unified_repr --tag aya_8b [--max_samples N]
  python unified_repr.py report  --out_dir outputs/unified_repr
"""

from __future__ import annotations
import argparse, gc, json, math, os, sys, warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import Cache

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*attention.*")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS
# ══════════════════════════════════════════════════════════════════

QA_SYSTEM = (
    "You are a QA assistant. "
    "Answer with a short phrase or named entity only. "
    "Do not explain or add extra words."
)
QA_SYSTEM_TGT = QA_SYSTEM + " Answer in English."
QA_SYSTEM_PHI = (
    "You are a QA assistant. "
    "Answer with a short phrase or named entity ONLY. "
    "Never refuse, apologize, or correct the question. "
    "Output only the answer, nothing else."
)
QA_SYSTEM_PHI_TGT = (
    "You are a QA assistant. "
    "Answer with a short phrase or named entity ONLY. "
    "Never refuse, apologize, or correct the question. "
    "Output only the answer in English, nothing else."
)

CS_METHOD_NORM: Dict[str, str] = {
    "random":               "Random",
    "selective":            "Selective",
    "grammarforce_source":  "GF_Src",
    "grammarforce_target":  "GF_Tgt",
}

# ── Language metadata (lang_name, family, word_order) ─────────────────────
LANG_META: Dict[str, Tuple[str, str, str]] = {
    "bbc-eng": ("Burmese",  "Sino-Tibetan", "SOV"),
    "ben-eng": ("Bengali",  "Indo-Aryan",   "SOV"),
    "esp-eng": ("Spanish",  "Romance",      "SVO"),
    "fra-eng": ("French",   "Romance",      "SVO"),
    "hin-eng": ("Hindi",    "Indo-Aryan",   "SOV"),
    "kor-eng": ("Korean",   "Koreanic",     "SOV"),
    "mar-eng": ("Marathi",  "Indo-Aryan",   "SOV"),
    "urd-eng": ("Urdu",     "Indo-Aryan",   "SOV"),
    "zho-eng": ("Chinese",  "Sino-Tibetan", "SVO"),
}


# ══════════════════════════════════════════════════════════════════
# SECTION 2 — MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════

@dataclass
class ModelCfg:
    tag:               str
    name:              str
    params_b:          float
    active_b:          float
    is_moe:            bool
    trust_remote_code: bool = False
    qwen_thinking:     Optional[bool] = None
    no_system_prompt:  bool = False
    attn_impl:         Optional[str] = None
    use_phi_gen:       bool = False   # phi35_moe: bypass generate(), use model() loop
    min_max_len:       int  = 0


ALL_MODELS: List[ModelCfg] = [
    # Dense models
    ModelCfg("aya_8b",      "CohereLabs/aya-expanse-8b",           8.0,  8.0,  False),
    ModelCfg("llama31_70b", "meta-llama/Llama-3.1-70B-Instruct",  70.0, 70.0, False),
    ModelCfg("llama33_70b", "meta-llama/Llama-3.3-70B-Instruct",  70.0, 70.0, False),
    ModelCfg("qwen35_4b",   "Qwen/Qwen3.5-4B",                     4.0,  4.0, False, qwen_thinking=False),
    ModelCfg("qwen35_9b",   "Qwen/Qwen3.5-9B",                     9.0,  9.0, False, qwen_thinking=False),
    ModelCfg("qwen35_27b",  "Qwen/Qwen3.5-27B",                   27.0, 27.0, False, qwen_thinking=False),
    ModelCfg("qwen3_06b",   "Qwen/Qwen3-0.6B",                    0.75, 0.75, False, qwen_thinking=False),
    ModelCfg("tiny_aya_3b", "CohereLabs/tiny-aya-global",          3.35, 3.35, False, min_max_len=600),
    ModelCfg("llama32_1b",  "meta-llama/Llama-3.2-1B-Instruct",   1.24, 1.24, False),
    ModelCfg("llama31_8b",  "meta-llama/Llama-3.1-8B-Instruct",   8.03, 8.03, False),
    ModelCfg("mistral_7b",  "mistralai/Mistral-7B-Instruct-v0.1", 7.24, 7.24, False),
    # MoE models (anchor-bias analysis only; no routing-extraction code here)
    ModelCfg("qwen35_35b_moe", "Qwen/Qwen3.5-35B-A3B",   35.0,  3.0, True, qwen_thinking=False),
    ModelCfg("qwen36_35b_moe", "Qwen/Qwen3.6-35B-A3B",   35.0,  3.0, True, qwen_thinking=False),
    ModelCfg("qwen3_30b_moe",  "Qwen/Qwen3-30B-A3B",     30.0,  3.0, True, qwen_thinking=False),
    ModelCfg("mixtral_8x7b",   "mistralai/Mixtral-8x7B-Instruct-v0.1", 46.7, 12.9, True),
    ModelCfg("phi35_moe",      "microsoft/Phi-3.5-MoE-instruct", 41.9, 6.6, True, use_phi_gen=True),
]

MODEL_MAP = {m.tag: m for m in ALL_MODELS}


# ══════════════════════════════════════════════════════════════════
# SECTION 3 — I/O HELPERS
# ══════════════════════════════════════════════════════════════════

def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: print(f"[warn] bad JSON line {i}: {line[:60]}")
    return rows

def append_jsonl(path: Path, obj: Dict) -> None:
    with open(path, "a") as f: f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def jdump(path: Path, obj: Any) -> None:
    with open(path, "w") as f: json.dump(obj, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════
# SECTION 4 — F1 / EM
# ══════════════════════════════════════════════════════════════════

import re, string

def _normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(s.split())

def em(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))

def f1(pred: str, gold: str) -> float:
    p_toks = _normalize(pred).split()
    g_toks = _normalize(gold).split()
    if not p_toks or not g_toks: return float(p_toks == g_toks)
    common = sum(min(p_toks.count(t), g_toks.count(t)) for t in set(g_toks))
    if common == 0: return 0.0
    pr = common / len(p_toks); rc = common / len(g_toks)
    return 2 * pr * rc / (pr + rc)


# ══════════════════════════════════════════════════════════════════
# SECTION 5 — GPU / LOAD UTILITIES
# ══════════════════════════════════════════════════════════════════

def flush_gpu(model=None) -> None:
    if model is not None:
        try: del model
        except: pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def _total_free_gib() -> float:
    total = 0
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        total += free
    return total / (1024 ** 3)

def _max_mem(factor: float = 0.92, cap_gib: Optional[int] = None) -> Dict[Any, str]:
    mem: Dict[Any, str] = {}
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        avail = int(free * factor) // (1024 ** 3)
        if cap_gib is not None: avail = min(avail, cap_gib)
        mem[i] = f"{max(avail, 0)}GiB"
    mem["cpu"] = "0GiB"
    print(f"  [max_mem] {mem}", flush=True)
    return mem

def _quant_max_mem(params_b: float, bits: int = 4) -> Dict[Any, str]:
    n = torch.cuda.device_count()
    if n == 0: return {"cpu": "0GiB"}
    quant_gib  = params_b * (bits / 8) * 1.1
    per_gpu_cap = max(int(quant_gib / n) + 6, 6)
    return _max_mem(factor=0.95, cap_gib=per_gpu_cap)


def load_hf(cfg: ModelCfg):
    tok = AutoTokenizer.from_pretrained(
        cfg.name, trust_remote_code=cfg.trust_remote_code, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    bnb4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    bnb8 = BitsAndBytesConfig(load_in_8bit=True)
    _attn_kw = {"attn_implementation": cfg.attn_impl} if cfg.attn_impl else {}
    device_map = os.environ.get("INCAS_DEVICE_MAP", "auto")
    base_bf16   = {"device_map": device_map, "max_memory": _max_mem(0.92),
                   "trust_remote_code": cfg.trust_remote_code, "low_cpu_mem_usage": True, **_attn_kw}
    base_quant4 = {"device_map": device_map, "max_memory": _quant_max_mem(cfg.params_b, 4),
                   "trust_remote_code": cfg.trust_remote_code, "low_cpu_mem_usage": True, **_attn_kw}
    base_quant8 = {"device_map": device_map, "max_memory": _quant_max_mem(cfg.params_b, 8),
                   "trust_remote_code": cfg.trust_remote_code, "low_cpu_mem_usage": True, **_attn_kw}

    bf16_needed_gib = cfg.params_b * 2.0
    free_gib = _total_free_gib()
    can_bf16 = free_gib >= bf16_needed_gib * 1.05

    if cfg.params_b <= 12:
        strats = [("bf16", {**base_bf16,   "torch_dtype": torch.bfloat16}),
                  ("4bit", {**base_quant4, "quantization_config": bnb4})]
    elif can_bf16:
        print(f"  [info] free {free_gib:.0f}GiB >= bf16 {bf16_needed_gib:.0f}GiB → try bf16")
        strats = [("bf16", {**base_bf16,   "torch_dtype": torch.bfloat16}),
                  ("4bit", {**base_quant4, "quantization_config": bnb4}),
                  ("8bit", {**base_quant8, "quantization_config": bnb8})]
    else:
        strats = [("4bit", {**base_quant4, "quantization_config": bnb4}),
                  ("8bit", {**base_quant8, "quantization_config": bnb8})]

    model = None
    for label, kw in strats:
        flush_gpu(model); model = None
        try:
            print(f"  [load] {label} ...", end="", flush=True)
            model = AutoModelForCausalLM.from_pretrained(cfg.name, **kw)
            for _, p in model.named_parameters():
                if p.device.type == "cpu":
                    raise RuntimeError("CPU offload detected — not enough GPU RAM")
            print(" OK"); break
        except Exception as e:
            print(f" FAIL: {type(e).__name__}: {str(e)[:120]}")
            try: del model
            except: pass
            model = None

    if model is None:
        raise RuntimeError(f"All load strategies failed for {cfg.name}")
    model.eval()
    return model, tok


# ══════════════════════════════════════════════════════════════════
# SECTION 6 — CHAT / TOKENIZATION HELPERS
# ══════════════════════════════════════════════════════════════════

def chat_text(tok, question: str, qwen_thinking: Optional[bool] = None,
              system: str = QA_SYSTEM, no_system_prompt: bool = False) -> str:
    if no_system_prompt:
        msgs = [{"role": "user", "content": question}]
    else:
        msgs = [{"role": "system", "content": system},
                {"role": "user",   "content": question}]
    if not hasattr(tok, "apply_chat_template"):
        return question if no_system_prompt else f"{system}\n\n{question}"
    kw: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if qwen_thinking is not None: kw["enable_thinking"] = bool(qwen_thinking)
    try:
        return tok.apply_chat_template(msgs, **kw)
    except TypeError:
        kw.pop("enable_thinking", None)
        return tok.apply_chat_template(msgs, **kw)

def _enc(tok, text: str, max_len: int) -> Dict[str, torch.Tensor]:
    add_sp = not bool(getattr(tok, "add_bos_token", False))
    return tok(text, return_tensors="pt", truncation=True,
               max_length=max_len, add_special_tokens=add_sp)

def _question_prefix_len(tok, system: str, qwen_thinking: Optional[bool],
                          no_system_prompt: bool) -> int:
    """Return number of tokens in the prompt template WITHOUT the question content.

    We use a dummy question 'X' and measure T_prefix so that question-only
    pooling can slice hidden states from T_prefix to T_full.
    The dummy must be short enough not to trigger truncation.
    """
    dummy_text = chat_text(tok, "X", qwen_thinking,
                           system=system, no_system_prompt=no_system_prompt)
    enc = tok(dummy_text, return_tensors="pt", add_special_tokens=False)
    # subtract 1 for the dummy 'X' token itself
    return max(0, int(enc["input_ids"].shape[1]) - 1)

def _generation_prompt_len(tok, system: str, qwen_thinking: Optional[bool],
                            no_system_prompt: bool) -> int:
    """Return number of tokens in the generation-prompt suffix (e.g. '<|im_start|>assistant\\n').
    These tokens are identical across EN/CS/TGT and should be excluded from
    question-content pooling.
    """
    if no_system_prompt:
        msgs = [{"role": "user", "content": "X"}]
    else:
        msgs = [{"role": "system", "content": system},
                {"role": "user",   "content": "X"}]
    kw_gen    = {"tokenize": False, "add_generation_prompt": True}
    kw_no_gen = {"tokenize": False, "add_generation_prompt": False}
    if qwen_thinking is not None:
        kw_gen["enable_thinking"]    = bool(qwen_thinking)
        kw_no_gen["enable_thinking"] = bool(qwen_thinking)
    try:
        try:
            text_with = tok.apply_chat_template(msgs, **kw_gen)
            text_no   = tok.apply_chat_template(msgs, **kw_no_gen)
        except TypeError:
            kw_gen.pop("enable_thinking",    None)
            kw_no_gen.pop("enable_thinking", None)
            text_with = tok.apply_chat_template(msgs, **kw_gen)
            text_no   = tok.apply_chat_template(msgs, **kw_no_gen)
        enc_with = tok(text_with, return_tensors="pt", add_special_tokens=False)
        enc_no   = tok(text_no,   return_tensors="pt", add_special_tokens=False)
        return max(0, enc_with.input_ids.shape[1] - enc_no.input_ids.shape[1])
    except Exception:
        return 0  # fallback: don't exclude generation prompt


# ══════════════════════════════════════════════════════════════════
# SECTION 7 — FORWARD PASS + HIDDEN STATE EXTRACTION
# ══════════════════════════════════════════════════════════════════

def forward_hs(model, inputs: Dict[str, torch.Tensor]):
    with torch.inference_mode():
        out = model(**inputs, output_hidden_states=True,
                    output_attentions=False, use_cache=False, return_dict=True)
    hs = getattr(out, "hidden_states", None)
    if hs is None:
        raise RuntimeError("hidden_states not available for this model")
    return hs   # tuple[n_layers+1], each (1, T, D)

def get_hs(model, tok, q: str, max_len: int,
           qwen_thinking: Optional[bool],
           system: str = QA_SYSTEM,
           no_system_prompt: bool = False) -> Tuple[Any, int]:
    text   = chat_text(tok, q, qwen_thinking, system=system, no_system_prompt=no_system_prompt)
    inputs = _enc(tok, text, max_len)
    dev    = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    T      = int(inputs["attention_mask"][0].sum())
    hs     = forward_hs(model, inputs)
    return hs, T


# ══════════════════════════════════════════════════════════════════
# SECTION 8 — POOLING STRATEGIES
# ══════════════════════════════════════════════════════════════════

def pool_full(hs_layer: torch.Tensor, T: int) -> torch.Tensor:
    """Mean pool over all T tokens (full-input, existing behavior)."""
    return hs_layer[0, :T, :].float().mean(0)

def pool_question(hs_layer: torch.Tensor, T: int, T_prefix: int) -> torch.Tensor:
    """Mean pool over question-only tokens (T_prefix : T), excluding system prompt."""
    start = min(T_prefix, T - 1)
    return hs_layer[0, start:T, :].float().mean(0)

def pool_last(hs_layer: torch.Tensor, T: int) -> torch.Tensor:
    """Last-token representation."""
    return hs_layer[0, T - 1, :].float()

def pool_whitened(hs_layer: torch.Tensor, T: int, mean_dir: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean pool then subtract global mean direction (whitening-lite)."""
    v = pool_full(hs_layer, T)
    if mean_dir is not None:
        v = v - mean_dir
    return v

def pool_q_pure(hs_layer: torch.Tensor, T: int, T_prefix: int, T_gen: int = 0) -> torch.Tensor:
    """Mean pool over pure question-content tokens only: [T_prefix : T - T_gen].
    Excludes both the system-prompt prefix AND the generation-prompt suffix.
    This is the cleanest representation of the question content itself.
    """
    start = min(T_prefix, T - 1)
    end   = max(start + 1, T - T_gen) if T_gen > 0 else T
    return hs_layer[0, start:end, :].float().mean(0)

def pool_q_normed(hs_layer: torch.Tensor, T: int, T_prefix: int, T_gen: int = 0) -> torch.Tensor:
    """Direction mean: L2-normalize each question token, then mean pool.
    Removes token-magnitude bias (special tokens, BOS/EOS often have large magnitude).
    Equivalent to the mean of unit vectors on the unit sphere.
    """
    start = min(T_prefix, T - 1)
    end   = max(start + 1, T - T_gen) if T_gen > 0 else T
    vecs = hs_layer[0, start:end, :].float()      # (n_q_tokens, D)
    vecs = F.normalize(vecs, dim=-1)              # unit vectors per token
    return vecs.mean(0)                           # direction mean


# ══════════════════════════════════════════════════════════════════
# SECTION 9 — REPRESENTATION METRICS
# ══════════════════════════════════════════════════════════════════

def cka_linear(X: torch.Tensor, Y: torch.Tensor) -> float:
    if X.shape[0] < 4 or Y.shape[0] < 4: return float("nan")
    Xc = X.float() - X.float().mean(0, keepdim=True)
    Yc = Y.float() - Y.float().mean(0, keepdim=True)
    K, L  = Xc @ Xc.T, Yc @ Yc.T
    hsic  = (K * L).sum()
    denom = torch.sqrt((K*K).sum() * (L*L).sum() + 1e-12)
    return float((hsic / denom).detach().cpu())

def cosine_vecs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

def repr_metrics_at_layer(
    hsA, hsB, T_A: int, T_B: int, layer: int,
    Tp_A: int = 0, Tp_B: int = 0,
    Tg_A: int = 0, Tg_B: int = 0,
    mean_dir_A: Optional[torch.Tensor] = None,
    mean_dir_B: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute all repr metrics between two conditions at a given layer.

    Returns keys:
      cos_full    — cosine of full-input mean pools (all tokens including system+gen prompt)
      cos_qonly   — cosine of [T_prefix:T] mean pools (excludes system prompt)
      cos_q_pure  — cosine of [T_prefix:T-T_gen] mean pools (excludes system AND gen prompt)
                    ← PRIMARY: purest question-content representation
      cos_q_normed— cosine of direction-mean over [T_prefix:T-T_gen]
                    (L2-normalize each token before mean → removes magnitude bias)
      cos_last    — cosine of last tokens (generation-readiness signal)
      cos_white   — cosine after mean-direction subtraction (anisotropy correction)
      cka         — linear CKA (full tokens)
      l2          — L2 on unit sphere
    """
    hA = hsA[layer][0]  # (T_A, D)
    hB = hsB[layer][0]

    T = min(T_A, T_B)
    hA_ = hA.unsqueeze(0) if hA.dim() == 2 else hA
    hB_ = hB.unsqueeze(0) if hB.dim() == 2 else hB

    # full-input pool (all tokens)
    mA_f = pool_full(hA_, T_A)
    mB_f = pool_full(hB_, T_B)
    # question-only pool ([T_prefix:T], incl. gen-prompt tokens)
    mA_q = pool_question(hA_, T_A, Tp_A)
    mB_q = pool_question(hB_, T_B, Tp_B)
    # pure question-content pool ([T_prefix:T-T_gen], excl. gen-prompt)
    pA_pure = pool_q_pure(hA_, T_A, Tp_A, Tg_A)
    pB_pure = pool_q_pure(hB_, T_B, Tp_B, Tg_B)
    # direction mean: L2-normalized tokens, then mean
    pA_norm = pool_q_normed(hA_, T_A, Tp_A, Tg_A)
    pB_norm = pool_q_normed(hB_, T_B, Tp_B, Tg_B)
    # last token
    lA = pool_last(hA_, T_A)
    lB = pool_last(hB_, T_B)
    # whitened
    wA = mA_f - mean_dir_A if mean_dir_A is not None else mA_f
    wB = mB_f - mean_dir_B if mean_dir_B is not None else mB_f

    hA_mat = hA[:T_A, :].float()
    hB_mat = hB[:T_B, :].float()
    cka = cka_linear(hA_mat[:T], hB_mat[:T])

    return {
        "cos_full":    cosine_vecs(mA_f,    mB_f),
        "cos_qonly":   cosine_vecs(mA_q,    mB_q),
        "cos_q_pure":  cosine_vecs(pA_pure, pB_pure),   # ← primary
        "cos_q_normed":cosine_vecs(pA_norm, pB_norm),
        "cos_last":    cosine_vecs(lA,      lB),
        "cos_white":   cosine_vecs(wA,      wB),
        "cka":         cka,
        "l2":          float((F.normalize(mA_f, dim=0) - F.normalize(mB_f, dim=0)).norm()),
    }


# ══════════════════════════════════════════════════════════════════
# SECTION 10 — PER-LAYER ANISOTROPY BASELINE (Reservoir per layer)
# ══════════════════════════════════════════════════════════════════

class LayerReservoir:
    """Reservoir sampler that tracks vectors per layer for anisotropy estimation.

    Stores a small reservoir per layer to estimate random_baseline_cos[layer]
    independently — more accurate than one global baseline.
    """
    def __init__(self, n_layers: int, capacity: int = 200, seed: int = 42):
        self.cap = capacity
        self.bufs: List[List[np.ndarray]] = [[] for _ in range(n_layers)]
        self.ns:   List[int]              = [0] * n_layers
        self.rng   = np.random.default_rng(seed)

    def add_layer(self, layer: int, v: np.ndarray) -> None:
        self.ns[layer] += 1
        n = self.ns[layer]
        buf = self.bufs[layer]
        if len(buf) < self.cap:
            buf.append(v.copy())
        else:
            j = int(self.rng.integers(0, n))
            if j < self.cap: buf[j] = v.copy()

    def baseline(self, layer: int, n_pairs: int = 500) -> float:
        buf = self.bufs[layer]
        if len(buf) < 2: return float("nan")
        V = np.stack(buf)
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        idx = list(range(len(V)))
        sims = []
        for _ in range(n_pairs):
            i, j = self.rng.choice(idx, size=2, replace=False)
            sims.append(float(np.dot(V[i], V[j])))
        return float(np.mean(sims))

    def all_baselines(self, n_pairs: int = 500) -> List[float]:
        return [self.baseline(l, n_pairs) for l in range(len(self.bufs))]

    def mean_direction(self, layer: int) -> Optional[np.ndarray]:
        """Return the mean of the reservoir vectors (for whitening)."""
        buf = self.bufs[layer]
        if not buf: return None
        return np.stack(buf).mean(0)


# ══════════════════════════════════════════════════════════════════
# SECTION 12 — GENERATION
# ══════════════════════════════════════════════════════════════════

def gen_hf(model, tok, q: str, cfg: ModelCfg,
           max_len: int, max_new: int,
           system: str = QA_SYSTEM) -> str:
    text   = chat_text(tok, q, cfg.qwen_thinking, system=system,
                       no_system_prompt=cfg.no_system_prompt)
    inputs = _enc(tok, text, max_len)
    dev    = next(model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    _mtype = getattr(getattr(model, "config", None), "model_type", "").lower()

    # phi35_moe: bypass model.generate() → DynamicCache(config) bug
    if cfg.use_phi_gen or "phi" in _mtype:
        from transformers import DynamicCache as _DynCache
        _ids  = inputs["input_ids"]
        _mask = inputs.get("attention_mask")
        # Collect all EOS-like tokens: eos_token_id + any special token with "end" in name
        _eos_ids: set = set()
        _raw_eos = tok.eos_token_id
        if isinstance(_raw_eos, list): _eos_ids.update(_raw_eos)
        elif _raw_eos is not None:     _eos_ids.add(_raw_eos)
        for _ts, _tid in tok.get_added_vocab().items():
            if "end" in _ts.lower(): _eos_ids.add(_tid)
        _gen: List[int] = []
        _past: Any = _DynCache()
        with torch.inference_mode():
            _out  = model(input_ids=_ids, attention_mask=_mask,
                          past_key_values=_past, use_cache=True, return_dict=True)
            _nid  = int(_out.logits[:, -1, :].argmax(-1).item())
            _past = _out.past_key_values
            if _nid not in _eos_ids:
                _gen.append(_nid)
                _cid  = torch.tensor([[_nid]], dtype=torch.long, device=dev)
                _cmsk = (torch.cat([_mask, torch.ones(1, 1, dtype=_mask.dtype, device=dev)], 1)
                         if _mask is not None else None)
                for _ in range(max_new - 1):
                    _out  = model(input_ids=_cid, attention_mask=_cmsk,
                                  past_key_values=_past, use_cache=True, return_dict=True)
                    _nid  = int(_out.logits[:, -1, :].argmax(-1).item())
                    _past = _out.past_key_values
                    if _nid in _eos_ids: break
                    _gen.append(_nid)
                    _cid  = torch.tensor([[_nid]], dtype=torch.long, device=dev)
                    if _cmsk is not None:
                        _cmsk = torch.cat(
                            [_cmsk, torch.ones(1, 1, dtype=_cmsk.dtype, device=dev)], 1)
        raw = tok.decode(_gen, skip_special_tokens=True).strip()
        if "</think>" in raw: raw = raw.split("</think>", 1)[1].strip()
        return raw

    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new,
                             do_sample=False, temperature=None, top_p=None,
                             use_cache=True, pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id)
    ids = out[0, inputs["input_ids"].shape[1]:]
    raw = tok.decode(ids, skip_special_tokens=True).strip()
    if "</think>" in raw: raw = raw.split("</think>", 1)[1].strip()
    return raw


# ══════════════════════════════════════════════════════════════════
# SECTION 13 — PER-EXAMPLE ANALYSIS
# ══════════════════════════════════════════════════════════════════

def analyze_example(
    model, tok, ex: Dict, cfg: ModelCfg,
    max_len: int, layer_ids: List[int],
    do_repr: bool, do_qa: bool, max_new: int,
    reservoir: LayerReservoir,
    T_prefix_en: int, T_prefix_cs: int, T_prefix_tgt: int,
    T_gen_en: int, T_gen_cs: int, T_gen_tgt: int,
    system_en: str, system_cs: str, system_tgt: str,
) -> Tuple[Dict, List[Dict], Dict[str, np.ndarray]]:
    """
    Returns: (row, layer_rows, probe_vecs)
      row           per-example summary
      layer_rows    per-layer repr metrics
      probe_vecs    {cond: mean-pooled last-layer vec for anisotropy reservoir}
    """
    q_en   = str(ex.get("question_en",  "") or "")
    q_cs   = str(ex.get("question_cs",  "") or "")
    q_tgt  = str(ex.get("question_tgt", "") or "")
    gold   = str(ex.get("gold_answer",  "") or "")
    uid        = ex.get("uid",        "")
    pair_code  = ex.get("pair_code",  "")
    cs_method  = ex.get("cs_method",  "")
    cs_display = CS_METHOD_NORM.get(cs_method, cs_method)
    lang_meta  = LANG_META.get(pair_code, (pair_code, "Unknown", "UNK"))
    lang_name, lang_family, word_order = lang_meta
    nan = float("nan")

    row: Dict[str, Any] = {
        "uid": uid, "pair_code": pair_code,
        "lang_name": lang_name, "lang_family": lang_family, "word_order": word_order,
        "cs_method": cs_method, "cs_display": cs_display,
        "gold_answer": gold, "model_tag": cfg.tag,
    }

    # Forward passes
    hs_cache:   Dict[str, Any] = {}
    T_cache:    Dict[str, int] = {}
    probe_vecs: Dict[str, np.ndarray] = {}

    cond_specs = [
        ("EN", q_en, T_prefix_en, system_en),
        ("CS", q_cs, T_prefix_cs, system_cs),
        ("TGT", q_tgt, T_prefix_tgt, system_tgt),
    ]

    if do_repr:
        for cond, q, _Tp, sys_str in cond_specs:
            if not q: continue
            try:
                hs_out, T_out = get_hs(model, tok, q, max_len, cfg.qwen_thinking,
                                       system=sys_str,
                                       no_system_prompt=cfg.no_system_prompt)
                hs_cache[cond] = hs_out
                T_cache[cond]  = T_out
                # last-layer mean pool for anisotropy reservoir
                probe_vecs[cond] = hs_out[-1][0, :T_out, :].float().mean(0).cpu().numpy()
                # feed into per-layer reservoir
                for li in layer_ids:
                    if li < len(hs_out):
                        v = hs_out[li][0, :T_out, :].float().mean(0).cpu().numpy()
                        reservoir.add_layer(li, v)
            except Exception as e:
                print(f"    [WARN] fwd {cond} uid={uid}: {e}", flush=True)

    # ── Per-layer metrics ───────────────────────────────────────────
    layer_rows: List[Dict] = []
    if do_repr and "EN" in hs_cache and "CS" in hs_cache:
        T_EN  = T_cache["EN"];  T_CS  = T_cache["CS"]
        T_TGT = T_cache.get("TGT", 0)
        n_hs  = len(hs_cache["EN"])
        Tp_EN = T_prefix_en;   Tp_CS = T_prefix_cs
        Tp_TGT = T_prefix_tgt
        Tg_EN = T_gen_en; Tg_CS = T_gen_cs; Tg_TGT = T_gen_tgt

        for li in [l for l in layer_ids if l < n_hs]:
            lr: Dict[str, Any] = {
                "uid": uid, "layer": li,
                "pair_code": pair_code, "lang_family": lang_family, "word_order": word_order,
                "cs_method": cs_method, "cs_display": cs_display,
            }
            m_ec = repr_metrics_at_layer(hs_cache["EN"], hs_cache["CS"],
                                         T_EN, T_CS, li, Tp_EN, Tp_CS,
                                         Tg_A=Tg_EN, Tg_B=Tg_CS)
            lr.update({f"EN_CS_{k}": v for k, v in m_ec.items()})

            if "TGT" in hs_cache and T_TGT > 0:
                m_tc = repr_metrics_at_layer(hs_cache["TGT"], hs_cache["CS"],
                                             T_TGT, T_CS, li, Tp_TGT, Tp_CS,
                                             Tg_A=Tg_TGT, Tg_B=Tg_CS)
                m_et = repr_metrics_at_layer(hs_cache["EN"], hs_cache["TGT"],
                                             T_EN, T_TGT, li, Tp_EN, Tp_TGT,
                                             Tg_A=Tg_EN, Tg_B=Tg_TGT)
                lr.update({f"TGT_CS_{k}": v for k, v in m_tc.items()})
                lr.update({f"EN_TGT_{k}": v for k, v in m_et.items()})

                # Anchor bias (difference) and ratio for each pooling strategy
                for pool in ("cos_full", "cos_qonly", "cos_q_pure", "cos_q_normed",
                             "cos_last", "cos_white", "cka"):
                    en_v = m_ec.get(pool, nan)
                    tg_v = m_tc.get(pool, nan)
                    ab_val = en_v - tg_v
                    lr[f"anchor_bias_{pool}"] = ab_val if not math.isnan(ab_val) else nan
                    # ratio: cos(EN,CS) / cos(TGT,CS) — >1 EN-anchored, <1 TGT-anchored
                    if not math.isnan(en_v) and not math.isnan(tg_v) and tg_v > 1e-6:
                        lr[f"anchor_ratio_{pool}"] = en_v / tg_v
                    else:
                        lr[f"anchor_ratio_{pool}"] = nan
            else:
                for pool in ("cos_full", "cos_qonly", "cos_last", "cos_white", "cka"):
                    lr[f"anchor_bias_{pool}"] = nan

            layer_rows.append(lr)

        # Aggregate to per-example row: last-layer values + layermean + late-layer mean
        if layer_rows:
            ll = layer_rows[-1]
            for pool in ("cos_full", "cos_qonly", "cos_q_pure", "cos_q_normed",
                         "cos_last", "cos_white", "cka"):
                row[f"anchor_bias_{pool}"]  = ll.get(f"anchor_bias_{pool}", nan)
                row[f"anchor_ratio_{pool}"] = ll.get(f"anchor_ratio_{pool}", nan)
                row[f"EN_CS_{pool}"]        = ll.get(f"EN_CS_{pool}", nan)
                row[f"TGT_CS_{pool}"]       = ll.get(f"TGT_CS_{pool}", nan)
            # layer-mean (all layers)
            for pool in ("cos_full", "cos_qonly", "cos_q_pure", "cos_q_normed",
                         "cos_last", "cos_white"):
                for metric in ("anchor_bias", "anchor_ratio"):
                    vals = [lr.get(f"{metric}_{pool}", nan) for lr in layer_rows]
                    valid = [v for v in vals if not math.isnan(v)]
                    row[f"{metric}_layermean_{pool}"] = float(np.mean(valid)) if valid else nan
            # late-layer mean (last 25% of layers) — more robust than single last layer
            late_start = max(0, int(len(layer_rows) * 0.75))
            late_rows  = layer_rows[late_start:]
            for pool in ("cos_full", "cos_qonly", "cos_q_pure", "cos_q_normed"):
                for metric in ("anchor_bias", "anchor_ratio"):
                    vals = [lr.get(f"{metric}_{pool}", nan) for lr in late_rows]
                    valid = [v for v in vals if not math.isnan(v)]
                    row[f"{metric}_latelayer_{pool}"] = float(np.mean(valid)) if valid else nan
        else:
            for k in ["anchor_bias_cos_full", "anchor_bias_cos_qonly",
                      "anchor_bias_cos_q_pure", "anchor_bias_cos_q_normed",
                      "anchor_bias_cos_last", "anchor_bias_cos_white", "anchor_bias_cka",
                      "anchor_ratio_cos_q_pure", "anchor_ratio_cos_q_normed",
                      "anchor_ratio_cos_full", "anchor_ratio_cos_qonly"]:
                row[k] = nan

    # QA generation
    for cond, q, _, sys_str in cond_specs:
        row[f"pred_{cond}"] = ""
        row[f"f1_{cond}"]   = nan
        row[f"em_{cond}"]   = nan
        if do_qa and q and gold:
            try:
                pred = gen_hf(model, tok, q, cfg, max_len, max_new, system=sys_str)
                row[f"pred_{cond}"] = pred
                row[f"f1_{cond}"]   = f1(pred, gold)
                row[f"em_{cond}"]   = em(pred, gold)
            except Exception as e:
                print(f"    [WARN] gen {cond} uid={uid}: {e}", flush=True)

    for a, b in [("CS", "EN"), ("CS", "TGT"), ("EN", "TGT")]:
        va = row.get(f"f1_{a}", nan); vb = row.get(f"f1_{b}", nan)
        row[f"delta_f1_{a}_vs_{b}"] = (va - vb
                                        if not (math.isnan(va) or math.isnan(vb))
                                        else nan)
    return row, layer_rows, probe_vecs


# ══════════════════════════════════════════════════════════════════
# SECTION 14 — MODEL RUNNER
# ══════════════════════════════════════════════════════════════════

def run_one_model(cfg: ModelCfg, data: List[Dict], out_dir: Path,
                  layer_mode: str, do_repr: bool, do_qa: bool,
                  max_len: int, max_new: int, max_samples: int = 0,
                  rerun_repr: bool = False) -> None:
    """Run analysis for one model.

    rerun_repr=True: re-run repr extraction even for already-processed UIDs.
      - Loads existing per_example.jsonl to preserve QA results (f1/em/pred).
      - Ignores done_uids for repr computation.
      - Merges saved QA back into new rows, then overwrites per_example.jsonl.
      - Use this when the repr pipeline has been updated but QA is already done.
    """

    tag      = cfg.tag
    mdir     = mkdir(out_dir / "model_runs" / tag)
    raw_path = mdir / "per_example.jsonl"
    done_uids: set = set()

    # Saved QA cache: keyed by "uid|cs_display", stores pred/f1/em columns
    QA_COLS = ("pred_EN", "f1_EN", "em_EN",
               "pred_CS", "f1_CS", "em_CS",
               "pred_TGT", "f1_TGT", "em_TGT",
               "delta_f1_CS_vs_EN", "delta_f1_CS_vs_TGT", "delta_f1_EN_vs_TGT")
    saved_qa: Dict[str, Dict] = {}

    if raw_path.exists():
        with open(raw_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if "uid" not in obj or "cs_display" not in obj:
                        continue
                    key = f"{obj['uid']}|{obj['cs_display']}"
                    if not rerun_repr:
                        done_uids.add(key)
                    else:
                        # Preserve existing QA results
                        saved_qa[key] = {k: obj[k] for k in QA_COLS if k in obj}
                except: pass
        if rerun_repr:
            print(f"  [rerun_repr] re-computing repr for all examples; "
                  f"QA preserved from {len(saved_qa)} cached rows")
        else:
            print(f"  [resume] {len(done_uids)} examples already done")

    # Determine sample limit
    examples = data if max_samples <= 0 else data[:max_samples]
    todo = [ex for ex in examples
            if f"{ex.get('uid','')}|{CS_METHOD_NORM.get(ex.get('cs_method',''), ex.get('cs_method',''))}"
            not in done_uids]
    if not todo:
        print(f"  [skip] all examples done for {tag}"); return

    # When rerunning repr, overwrite per_example.jsonl from scratch
    if rerun_repr and raw_path.exists():
        raw_path.rename(raw_path.with_suffix(".jsonl.bak"))
        print(f"  [rerun_repr] backed up old per_example.jsonl → per_example.jsonl.bak")

    # Load model
    print(f"\n{'='*60}\n  TAG: {tag}   START\n{'='*60}")
    model, tok = load_hf(cfg)

    # Select layers
    n_hs = None
    if layer_mode == "all":
        dummy_text = chat_text(tok, "test", cfg.qwen_thinking,
                               no_system_prompt=cfg.no_system_prompt)
        dummy_in   = _enc(tok, dummy_text, max_len)
        dev        = next(model.parameters()).device
        dummy_in   = {k: v.to(dev) for k, v in dummy_in.items()}
        with torch.inference_mode():
            out = model(**dummy_in, output_hidden_states=True,
                        use_cache=False, return_dict=True)
        n_hs = len(out.hidden_states) - 1  # exclude embedding layer
        layer_ids = list(range(1, n_hs + 1))
        print(f"  [layers] mode=all  n_transformer_layers={n_hs}")
    else:
        raise ValueError(f"Unknown layer_mode: {layer_mode}")

    # Question prefix lengths for question-only pooling
    sys_en   = QA_SYSTEM_PHI     if cfg.use_phi_gen else QA_SYSTEM
    sys_cs   = QA_SYSTEM_PHI     if cfg.use_phi_gen else QA_SYSTEM
    sys_tgt  = QA_SYSTEM_PHI_TGT if cfg.use_phi_gen else QA_SYSTEM_TGT
    T_prefix_en  = _question_prefix_len(tok, sys_en,  cfg.qwen_thinking, cfg.no_system_prompt)
    T_prefix_cs  = T_prefix_en   # same system prompt for EN and CS
    T_prefix_tgt = _question_prefix_len(tok, sys_tgt, cfg.qwen_thinking, cfg.no_system_prompt)
    T_gen_en  = _generation_prompt_len(tok, sys_en,  cfg.qwen_thinking, cfg.no_system_prompt)
    T_gen_cs  = T_gen_en   # same generation prompt for EN and CS
    T_gen_tgt = _generation_prompt_len(tok, sys_tgt, cfg.qwen_thinking, cfg.no_system_prompt)
    print(f"  [prefix_len] EN/CS={T_prefix_en}  TGT={T_prefix_tgt}"
          f"  [gen_prompt] EN/CS={T_gen_en}  TGT={T_gen_tgt}")

    # Per-layer reservoir
    reservoir = LayerReservoir(n_layers=len(layer_ids) + 2, capacity=200)

    all_rows, all_layer_rows = [], []

    for ex in tqdm(todo, desc=f"[{tag}]"):
        try:
            row, lrows, probe_vecs = analyze_example(
                model, tok, ex, cfg,
                max_len, layer_ids,
                do_repr, do_qa, max_new,
                reservoir,
                T_prefix_en, T_prefix_cs, T_prefix_tgt,
                T_gen_en, T_gen_cs, T_gen_tgt,
                sys_en, sys_cs, sys_tgt,
            )
            if rerun_repr:
                key = f"{row.get('uid','')}|{row.get('cs_display','')}"
                if key in saved_qa:
                    row.update(saved_qa[key])
            all_rows.append(row)
            all_layer_rows.extend(lrows)
            append_jsonl(raw_path, row)
        except Exception as e:
            print(f"  [ERROR] example uid={ex.get('uid','?')}: {e}", flush=True)

    # ── Compute per-layer anisotropy baselines ──────────────────────
    print("  [anisotropy] computing per-layer baselines ...", flush=True)
    per_layer_baselines: List[float] = reservoir.all_baselines(n_pairs=500)
    # Also compute mean direction per layer for whitening validation
    mean_dirs = [reservoir.mean_direction(l) for l in range(len(per_layer_baselines))]

    # Save per-layer baselines
    jdump(mdir / "per_layer_baseline.json", {
        "model_tag": tag,
        "layer_ids": layer_ids,
        "random_baseline_per_layer": per_layer_baselines,
        # Global (last layer) for compatibility
        "random_baseline_cos": per_layer_baselines[-1] if per_layer_baselines else float("nan"),
        "n_reservoir": reservoir.cap,
    })

    # ── Reload all rows and add normed anchor_bias ───────────────────
    if all_rows:
        df = pd.DataFrame(all_rows)

        # anchor_bias_normed using median of non-nan per-layer baselines
        valid_baselines = [b for b in per_layer_baselines
                           if b is not None and not math.isnan(b)]
        global_baseline = float(np.median(valid_baselines)) if valid_baselines else float("nan")
        if not math.isnan(global_baseline) and (1 - global_baseline) > 1e-6:
            denom = 1 - global_baseline
            for pool in ("cos_full", "cos_qonly", "cos_last",
                         "cos_q_pure", "cos_q_normed"):
                col = f"anchor_bias_{pool}"
                if col in df.columns:
                    df[f"{col}_normed"] = df[col] / denom

        # Save
        df.to_csv(mdir / "metrics_per_example.csv", index=False)
        _save_summaries(df, mdir)

    # ── Layer-level CSV ──────────────────────────────────────────────
    if all_layer_rows:
        ldf = pd.DataFrame(all_layer_rows)
        # Attach per-layer normed values
        if per_layer_baselines:
            baseline_map = {layer_ids[i]: per_layer_baselines[i]
                            for i in range(min(len(layer_ids), len(per_layer_baselines)))}
            ldf["random_baseline_layer"] = ldf["layer"].map(baseline_map)
            denom_col = 1 - ldf["random_baseline_layer"]
            for pool in ("cos_full", "cos_qonly", "cos_last",
                         "cos_q_pure", "cos_q_normed"):
                col = f"anchor_bias_{pool}"
                if col in ldf.columns:
                    ldf[f"{col}_normed"] = ldf[col] / denom_col.replace(0, float("nan"))
        ldf.to_csv(mdir / "layer_metrics.csv", index=False)

        # Layer-convergence summary
        _layer_convergence(ldf, mdir, tag)

    print(f"  DONE: {tag}", flush=True)


def _save_summaries(df: pd.DataFrame, mdir: Path) -> None:
    for col, fname in [("cs_display", "summary_by_cs_display.csv"),
                       ("pair_code",  "summary_by_pair_code.csv"),
                       ("lang_family","summary_by_lang_family.csv")]:  # NEW: family summary
        if col not in df.columns: continue
        agg_cols = [c for c in [
                        "f1_EN", "f1_CS", "f1_TGT",
                        "delta_f1_CS_vs_EN", "delta_f1_CS_vs_TGT",
                        # Primary repr metrics (pure question-content, excl. system+gen prompt)
                        "anchor_bias_cos_q_pure",   "anchor_ratio_cos_q_pure",
                        "anchor_bias_cos_q_normed", "anchor_ratio_cos_q_normed",
                        "EN_CS_cos_q_pure", "TGT_CS_cos_q_pure",
                        # Late-layer means (last 25% of layers)
                        "anchor_bias_latelayer_cos_q_pure",
                        "anchor_bias_latelayer_cos_q_normed",
                        # Legacy (kept for backward compat)
                        "anchor_bias_cos_qonly",   "anchor_ratio_cos_qonly",
                        "anchor_bias_cos_full",    "anchor_ratio_cos_full",
                        "anchor_bias_cos_full_normed", "anchor_bias_cos_qonly_normed",
                        "EN_CS_cos_full", "TGT_CS_cos_full",
                    ] if c in df.columns]
        if not agg_cols: continue
        summary = df.groupby(col)[agg_cols].mean().reset_index()
        summary.to_csv(mdir / fname, index=False)


def _layer_convergence(ldf: pd.DataFrame, mdir: Path, tag: str) -> None:
    """Compute and save layer convergence metrics (both cos_full and cos_qonly)."""
    out_rows = []
    for pool in ("cos_full", "cos_qonly"):
        ab_col = f"anchor_bias_{pool}"
        if ab_col not in ldf.columns: continue
        for cond in ldf["cs_display"].unique():
            sub = ldf[ldf["cs_display"] == cond].groupby("layer")[ab_col].mean()
            if sub.empty: continue
            # convergence layer: last layer where sign changes to final sign
            final_sign = np.sign(sub.iloc[-1]) if not np.isnan(sub.iloc[-1]) else 0
            conv_layer = int(sub.index[-1])
            for i, (l, v) in enumerate(sub.items()):
                if not np.isnan(v) and np.sign(v) == final_sign:
                    conv_layer = int(l)
                    break
            peak_layer = int(sub.abs().idxmax())
            pct_en = float((sub > 0).mean())
            n_layers = int(sub.index[-1]) + 1  # total layer count (0-indexed last + 1)
            out_rows.append({
                "cs_display": cond, "pool": pool,
                "n_layers": n_layers,
                "lc_convergence_layer": conv_layer,
                "lc_convergence_layer_pct": round(conv_layer / max(n_layers - 1, 1) * 100, 1),
                "lc_peak_bias_layer": peak_layer,
                "lc_peak_bias_layer_pct": round(peak_layer / max(n_layers - 1, 1) * 100, 1),
                "lc_pct_EN_anchored": pct_en,
                "model_tag": tag,
            })
    if out_rows:
        mkdir(mdir / "layer_convergence")
        pd.DataFrame(out_rows).to_csv(
            mdir / f"layer_convergence/layer_conv_{tag}.csv", index=False)


# ══════════════════════════════════════════════════════════════════
# SECTION 15 — REPORT
# ══════════════════════════════════════════════════════════════════

def cmd_report(out_dir: Path) -> None:
    from scipy import stats as sp_stats

    mrun_dir = out_dir / "model_runs"
    if not mrun_dir.exists():
        print(f"[error] {mrun_dir} not found"); return

    all_dfs = []
    for mdir in sorted(mrun_dir.iterdir()):
        csv = mdir / "metrics_per_example.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            all_dfs.append(df)

    if not all_dfs:
        print("[report] no metrics_per_example.csv found"); return

    df_all = pd.concat(all_dfs, ignore_index=True)
    cdir   = mkdir(out_dir / "compare")

    # Spearman correlation table
    targets = ["delta_f1_CS_vs_EN", "delta_f1_CS_vs_TGT"]
    features = [c for c in df_all.columns if "anchor_bias" in c]

    corr_rows = []
    for tag, gdf in df_all.groupby("model_tag"):
        for feat in features:
            for tgt in targets:
                sub = gdf[[feat, tgt]].dropna()
                if len(sub) < 30: continue
                r, p = sp_stats.spearmanr(sub[feat], sub[tgt])
                corr_rows.append({"model_tag": tag, "feature": feat,
                                   "target": tgt, "spearman_r": round(r, 4),
                                   "p_value": round(p, 6), "n": len(sub)})
    if corr_rows:
        corr_df = pd.DataFrame(corr_rows)
        corr_df.to_csv(cdir / "spearman_correlations.csv", index=False)
        print(f"[report] saved {len(corr_df)} Spearman correlations")

    # Pivot: anchor_bias by model × condition (full + qonly)
    for pool in ("cos_full", "cos_qonly"):
        ab_col = f"anchor_bias_{pool}_normed"
        if ab_col not in df_all.columns:
            ab_col = f"anchor_bias_{pool}"
        if ab_col not in df_all.columns: continue
        pivot = df_all.pivot_table(index="model_tag", columns="cs_display",
                                   values=ab_col, aggfunc="mean")
        pivot.to_csv(cdir / f"anchor_bias_{pool}_pivot.csv")
        print(f"[report] saved anchor_bias_{pool}_pivot")

    # delta_f1 pivot
    pivot_f1 = df_all.pivot_table(index="model_tag", columns="cs_display",
                                   values="delta_f1_CS_vs_EN", aggfunc="mean")
    pivot_f1.to_csv(cdir / "delta_f1_pivot.csv")

    # Language family analysis (new)
    if "lang_family" in df_all.columns:
        fam_agg = df_all.groupby(["model_tag", "lang_family"])[
            ["delta_f1_CS_vs_EN", "anchor_bias_cos_full", "anchor_bias_cos_qonly"]
        ].mean().reset_index()
        fam_agg.to_csv(cdir / "by_lang_family.csv", index=False)
        print("[report] saved by_lang_family.csv")

    # cos_full vs cos_qonly comparison (new)
    if "anchor_bias_cos_full" in df_all.columns and "anchor_bias_cos_qonly" in df_all.columns:
        comp = df_all.groupby(["model_tag", "cs_display"])[
            ["anchor_bias_cos_full", "anchor_bias_cos_qonly"]
        ].mean().reset_index()
        comp["diff_qonly_minus_full"] = (comp["anchor_bias_cos_qonly"] -
                                          comp["anchor_bias_cos_full"])
        comp.to_csv(cdir / "pooling_comparison.csv", index=False)
        print("[report] saved pooling_comparison.csv (full vs question-only)")

    print(f"[report] complete → {cdir}")


# ══════════════════════════════════════════════════════════════════
# SECTION 16 — CLI
# ══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Anchor-bias analysis")
    sub = parser.add_subparsers(dest="cmd")

    # analyze
    p_analyze = sub.add_parser("analyze")
    p_analyze.add_argument("--data_jsonl",    required=True)
    p_analyze.add_argument("--out_dir",       required=True)
    p_analyze.add_argument("--tag",           required=True,
                           help=f"Model tag: {list(MODEL_MAP)}")
    p_analyze.add_argument("--layer_mode",    default="all", choices=["all"])
    p_analyze.add_argument("--eval_repr",     action="store_true", default=True)
    p_analyze.add_argument("--eval_qa",       action="store_true")
    p_analyze.add_argument("--max_len",       type=int, default=256)
    p_analyze.add_argument("--max_new_tokens",type=int, default=48)
    p_analyze.add_argument("--max_samples",   type=int, default=0)
    p_analyze.add_argument("--rerun_repr",    action="store_true",
                           help="Re-run repr extraction even for already-processed UIDs. "
                                "Preserves existing QA (f1/em/pred) from per_example.jsonl. "
                                "Use when repr pipeline has been updated.")

    # report
    p_report = sub.add_parser("report")
    p_report.add_argument("--out_dir", required=True)

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help(); return

    if args.cmd == "analyze":
        if args.tag not in MODEL_MAP:
            print(f"[error] unknown tag '{args.tag}'. Valid: {list(MODEL_MAP)}"); return
        cfg  = MODEL_MAP[args.tag]
        data = read_jsonl(Path(args.data_jsonl))
        print(f"[info] loaded {len(data)} examples from {args.data_jsonl}")
        effective_max_len = max(args.max_len, cfg.min_max_len)
        if cfg.min_max_len > args.max_len:
            print(f"  [max_len] {args.max_len} → {effective_max_len} (model min_max_len override)")
        run_one_model(
            cfg, data,
            out_dir     = Path(args.out_dir),
            layer_mode  = args.layer_mode,
            do_repr     = args.eval_repr,
            do_qa       = args.eval_qa,
            max_len     = effective_max_len,
            max_new     = args.max_new_tokens,
            max_samples = args.max_samples,
            rerun_repr  = args.rerun_repr,
        )

    elif args.cmd == "report":
        cmd_report(Path(args.out_dir))


if __name__ == "__main__":
    main()
