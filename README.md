# CANVAS: Contextual Anchor-based Neural Vector Alignment Steering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **"Code-Switching Reveals Language Anchoring in
Multilingual LLMs"**.

Jeonghyun Park\* (Chung-Ang University), Seunghyun Yoon (Adobe Research),
Yonghyun Jun (Chung-Ang University), Hwanhee Lee† (Chung-Ang University)

*\*Work done at Chung-Ang University. †Corresponding author.*

---

## Overview

Multilingual Large Language Models (MLLMs) are increasingly expected to
handle inputs that mix multiple languages in a single interaction, yet how
they internally organize **code-switching (CS)** remains unclear. We use CS
as a controlled probe of **language anchoring** inside MLLMs: matched
question variants that share the same information need but differ in
linguistic form let us ask whether a CS hidden state is internally closer
to its source-language or target-language counterpart.

We introduce **anchor bias**, a representation-level measure that compares
the cosine similarity of a CS question's hidden state to its source-side
and target-side counterparts. Across MLLMs we observe a consistent
pattern: **source-framed CS is more source-anchored**, while
**target-framed CS is more target-anchored**, and stronger target
anchoring is associated with **larger CS QA degradation**.

Building on this signal, we propose **CANVAS** (Contextual Anchor-based
Neural Vector Alignment Steering), an inference-time intervention that
estimates an in-context source-side "canvas" from the same CS input and
softly steers target-language token hidden states toward that canvas
during prefill. CANVAS requires **no retraining, no text rewriting, no
translation, and no calibration data**, and it recovers QA F1 across
multiple MLLMs and CS conditions.

## Key findings

- **Frame-dependent anchoring.** Across 10 MLLMs (6 dense + 4 MoE),
  upper-layer anchor bias is positive for grammar-forced source-frame CS
  (mean AB_upper = +0.246) and negative for target-frame CS (mean = -0.320),
  a swing of Δ_AB = +0.566. The fraction of upper layers that are
  source-anchored is 75.9% under **GF-Src** vs. only 36.2% under **GF-Tgt**.
- **Anchor bias predicts QA degradation.** Stronger source anchoring is
  associated with smaller F1 drops; the per-example correlation between
  anchor bias and ΔF1 is **Spearman ρ = +0.485, p < 0.01**.
- **CANVAS improves CS QA across the board.** With adaptive interpolation
  strength α = clip(0.45 − 1.5·γ, [0.05, 0.75]), CANVAS improves average
  F1 for **every** evaluated MLLM, with the largest gains under
  target-framed CS (GF-Tgt).
- **CANVAS shifts representations source-ward.** Across the 32
  model × condition cells, CANVAS shifts AB_upper toward the source side
  by +0.136 on average; the signed projection ratio η has mean +0.198,
  and 93.0% of examples satisfy η > 0.

## Repository layout

```
CANVAS/
├── canvas.py                  # CANVAS steering engine (paper Section 4)
├── canvas_eval.py             # BASE / SRC-oracle / CANVAS evaluation
├── unified_repr.py            # Anchor-bias analysis (paper Section 3)
├── compute_canvas_stats.py    # Wilcoxon, bootstrap CIs, BH-FDR, paper tables
├── data/
│   └── pairs_with_tgt.jsonl   # 2,880 matched QA comparisons (9 target languages)
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/jeonghyunpark2002/CANVAS.git
cd CANVAS
pip install -r requirements.txt
```

### Dependencies

- Python 3.10+
- PyTorch 2.5+
- Transformers 4.46+
- bitsandbytes (4-bit quantization for 70B-class models)
- scipy, scikit-learn
- lingua-language-detector (token-level language tagging used in CANVAS)

GPU required. 70B-class models load in 4-bit; smaller models in bf16/fp16.

## Quick start

### 1. Anchor-bias analysis (Section 3)

Compute per-layer hidden-state similarity bias and per-example QA F1:

```bash
python unified_repr.py analyze \
    --data_jsonl data/pairs_with_tgt.jsonl \
    --out_dir    outputs/unified_repr \
    --tag        aya_8b \
    --layer_mode all --eval_qa \
    --max_len 256 --max_new_tokens 48

python unified_repr.py report --out_dir outputs/unified_repr
```

Per-model outputs in `outputs/unified_repr/model_runs/{tag}/`:

- `metrics_per_example.csv` — per-example anchor bias (raw + normalized,
  late-layer mean, multiple pooling variants), per-condition F1, and ΔF1.
- `layer_metrics.csv` — full per-layer breakdown.
- `per_layer_baseline.json` — per-layer random-cosine anisotropy baselines.

### 2. CANVAS mitigation (Section 4)

Run BASE / SRC-oracle / CANVAS_adapt for one model:

```bash
python canvas_eval.py analyze \
    --data_jsonl data/pairs_with_tgt.jsonl \
    --out_dir    outputs/canvas \
    --tag        aya_8b \
    --max_len 256 --max_new_tokens 48 \
    --interp_alpha 0.45 \
    --interp_alpha_k 1.5 \
    --interp_alpha_min 0.05 \
    --interp_alpha_max 0.75

python canvas_eval.py report --out_dir outputs/canvas
```

CANVAS uses adaptive α by default. Pass `--no_adaptive_alpha` for a fixed
α = `--interp_alpha`.

### 3. Paper statistics

After running anchor-bias analysis and CANVAS evaluation across the model
pool:

```bash
python compute_canvas_stats.py \
    --canvas_dir outputs/canvas/model_runs \
    --anchor_dir outputs/unified_repr/model_runs \
    --out_dir    outputs/canvas_stats \
    --tbl_dir    tables
```

Produces per-cell Wilcoxon (one-sided, alternative='greater'), Hedges-
corrected paired d_z, 10,000 paired bootstrap CIs, Benjamini-Hochberg FDR,
cross-model sign tests, and anchor-bias ↔ ΔF1 Spearman correlation with
bootstrap CI.

## Method

### Anchor bias (Section 3)

For a matched question set (q_src, q_tgt, q_cs) and an MLLM M, let
H_M(q) ∈ R^{L × T × d} be the hidden states. We pool over the
question-content token span C(q) only (excluding system and chat-template
tokens):

```
r_l(q) = mean over t in C(q) of H_M(q)[l, t, :]
s_l(q_i, q_j) = cos(r_l(q_i), r_l(q_j))
AB_l(q_cs)    = s_l(q_src, q_cs) − s_l(q_tgt, q_cs)
```

We apply per-model, per-layer **random-cosine normalization** estimated
from English question pairs with disjoint reference answers to remove
background anisotropy, and aggregate over the upper half of layers
(AB^upper). Positive AB^upper indicates source anchoring; negative
indicates target anchoring.

### CANVAS (Section 4)

CANVAS runs on the same CS input — no rewriting, no translation. It uses
the **upper control layers** L_ctrl = [⌊0.7L⌋, …, L − 1] (the final 30%
of transformer layers).

**Stage 1: Token partition.** Split the question span
C(q_cs) = C_src ∪ C_tgt ∪ C_oth using lightweight token-level language
tagging (`canvas.tag_tokens`: Unicode-script for non-Latin TGT, lingua
language detection for Latin-script TGT).

**Stage 2: Source canvas construction.** A clean prefill pass over q_cs
gives, for each l ∈ L_ctrl:

```
c_src_l = mean over t in C_src of H_M(q_cs)[l, t, :]
a_tgt_l = mean over t in C_tgt of H_M(q_cs)[l, t, :]
```

The source canvas c_src serves as the in-context source-side reference.

**Stage 3: Canvas-alignment scoring.** With u_l = (c_src_l − a_tgt_l) /
‖c_src_l − a_tgt_l‖₂ and m_l = mean of H_M(q_cs)[l, t, :] over
C_mix = C_src ∪ C_tgt, the canvas alignment is

```
γ(q_cs) = mean over l ∈ L_ctrl of cos(m_l, u_l)
```

γ > 0 means the CS representation is already source-canvas aligned;
γ < 0 means it is target-oriented.

**Stage 4: Adaptive interpolation.** With γ_CS = γ(q_cs):

```
α = clip_{[0.05, 0.75]}( 0.45 − 1.5 · γ_CS )
```

The negative-slope rule means target-leaning inputs receive **stronger**
correction.

**Stage 5: Online source-canvas interpolation.** A second prefill pass
applies, for each layer l ∈ L_ctrl and each target-language token
position t ∈ C_tgt:

```
h_{l, t} ← (1 − α) · h_{l, t} + α · c_src_l
```

The model decodes greedily from the modified KV cache. Only target-language
token positions are touched; chat-template tokens, the source-framed
context, and the generation suffix are left unchanged.

## Data

`data/pairs_with_tgt.jsonl` — 2,880 matched (SRC, CS, TGT) comparisons
derived from **955 unique source questions** (SimpleQA Verified) crossed
with four CS conditions and nine target languages.

```json
{
  "uid": "3296-selective-Toba Batak",
  "pair_code": "bbc-eng",
  "language_label": "Toba Batak",
  "cs_method": "selective",
  "question_en":  "In what key was \"I Offer My Life\" by Don Moen composed?",
  "question_cs":  "Di aha key do lagu \"I Offer My Life\" ni Don Moen disusun?",
  "question_tgt": "...",
  "gold_answer":  "F Major"
}
```

- **9 target languages** (paired with English as the source): Bengali,
  Spanish, French, Hindi, Korean, Marathi, Toba Batak, Urdu, Chinese.
- **4 CS conditions** (CodeMixQA): `grammarforce_source` (**GF-Src**),
  `grammarforce_target` (**GF-Tgt**), `random`, `selective`.
- Target-language counterparts q_tgt are produced by neural translation
  with Qwen3-235B-A22B.

## Models

| Tag | HuggingFace ID | Dense/MoE |
|---|---|:-:|
| `aya_8b`         | CohereLabs/aya-expanse-8b              | Dense |
| `qwen35_4b`      | Qwen/Qwen3.5-4B                        | Dense |
| `qwen35_9b`      | Qwen/Qwen3.5-9B                        | Dense |
| `qwen35_27b`     | Qwen/Qwen3.5-27B                       | Dense |
| `llama31_8b`     | meta-llama/Llama-3.1-8B-Instruct       | Dense |
| `llama31_70b`    | meta-llama/Llama-3.1-70B-Instruct      | Dense |
| `llama33_70b`    | meta-llama/Llama-3.3-70B-Instruct      | Dense |
| `mistral_7b`     | mistralai/Mistral-7B-Instruct-v0.1     | Dense |
| `qwen35_35b_moe` | Qwen/Qwen3.5-35B-A3B                   | MoE   |
| `qwen36_35b_moe` | Qwen/Qwen3.6-35B-A3B                   | MoE   |
| `qwen3_30b_moe`  | Qwen/Qwen3-30B-A3B                     | MoE   |
| `mixtral_8x7b`   | mistralai/Mixtral-8x7B-Instruct-v0.1   | MoE   |
| `phi35_moe`      | microsoft/Phi-3.5-MoE-instruct         | MoE   |

The 10-model anchor-bias pool in Table 1 is a subset (excludes the small
1B/0.6B/3B/7B-8B reference models, which are added in Table 2 for the
CANVAS intervention).

## License

This project is released under the [MIT License](LICENSE).
