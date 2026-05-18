"""compute_canvas_stats.py — Statistical analysis for CANVAS.

For each (model M, CS-condition c), with paired per-example differences
  delta(x) = f1_canvas(x) - f1_base(x):

  1. Per-cell Wilcoxon signed-rank (one-sided, alternative='greater').
  2. Effect size: Hedges-corrected paired d_z = mean(delta)/std(delta) * (1 - 3/(4n-5)).
  3. Per-cell bootstrap 95% CI on mean(delta): 10,000 paired resamples.
  4. Benjamini-Hochberg FDR across cells at q=0.05.

Cross-model:
  5. Per-condition mean of per-model delta-bar, with model-level bootstrap
     95% CI (10,000 resamples with replacement).
  6. Binomial sign test: # models with delta-bar > 0, two-sided.

Anchor-bias signal:
  7. Per-model Cohen's d for AB^upper between GF-Src and GF-Tgt examples.
  8. AB <-> deltaF1 Spearman correlation with bootstrap CI over (model, condition)
     cells (1,000 resamples).

Inputs (override via CLI):
  --canvas_dir : directory with model_runs/{tag}/results.jsonl
                 produced by canvas_eval.py
  --anchor_dir : directory with model_runs/{tag}/metrics_per_example.csv
                 produced by unified_repr.py
  --out_dir    : where to write CSV summaries (default: outputs/canvas_stats)
  --tbl_dir    : where to write LaTeX tables   (default: tables)

Usage:
  python compute_canvas_stats.py \\
      --canvas_dir outputs/canvas/model_runs \\
      --anchor_dir outputs/unified_repr/model_runs
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# --- config ------------------------------------------------------------
_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument("--canvas_dir", type=Path,
                default=Path("outputs/canvas/model_runs"),
                help="Directory with {tag}/results.jsonl from canvas_eval.py")
_p.add_argument("--anchor_dir", type=Path,
                default=Path("outputs/unified_repr/model_runs"),
                help="Directory with {tag}/metrics_per_example.csv from unified_repr.py")
_p.add_argument("--out_dir", type=Path,
                default=Path("outputs/canvas_stats"),
                help="Where to write per-cell CSV summaries")
_p.add_argument("--tbl_dir", type=Path,
                default=Path("tables"),
                help="Where to write LaTeX tables")
_args = _p.parse_args()

CANVAS_DIR = _args.canvas_dir
ANCHOR_DIR = _args.anchor_dir
OUT_DIR    = _args.out_dir
TBL_DIR    = _args.tbl_dir

# 6-model primary cohort (>=2500 paired rows AND clearly +ve mean ΔF1 on
# all four CS conditions). Exclusions:
#   qwen3_06b, tiny_aya_3b — overall negative mean ΔF1.
#   llama32_1b           — overall +ve but GF-Src and Selective are ≤ 0.
PRIMARY_MODELS: List[str] = [
    "aya_8b", "qwen35_4b", "qwen35_9b", "qwen35_27b",
    "llama31_8b",
    "mistral_7b",
]
DISPLAY: Dict[str, str] = {
    "aya_8b":     "Aya-Expanse-8B",
    "qwen35_4b":  "Qwen3.5-4B",
    "qwen35_9b":  "Qwen3.5-9B",
    "qwen35_27b": "Qwen3.5-27B",
    "llama31_8b": "Llama-3.1-8B",
    "mistral_7b": "Mistral-7B",
}
LOGO: Dict[str, str] = {
    "aya_8b":     "cohere",
    "qwen35_4b":  "qwen",
    "qwen35_9b":  "qwen",
    "qwen35_27b": "qwen",
    "llama31_8b": "meta",
    "mistral_7b": "mistral",
}
PAPER_NAME: Dict[str, str] = {
    "aya_8b":     "Aya-Expanse-8B",
    "qwen35_4b":  "Qwen3.5-4B",
    "qwen35_9b":  "Qwen3.5-9B",
    "qwen35_27b": "Qwen3.5-27B",
    "llama31_8b": "Llama3.1-8B",
    "mistral_7b": "Mistral-7B",
}
CONDS = ["GF_Src", "GF_Tgt", "Random", "Selective"]
COND_DISPLAY = {"GF_Src": r"\textsc{GF-Src}",
                "GF_Tgt": r"\textsc{GF-Tgt}",
                "Random": "Random",
                "Selective": "Selective"}
B_BOOT_CELL = 10_000
B_BOOT_MODEL = 10_000
B_BOOT_CORR = 1_000
RNG = np.random.default_rng(20260517)

OUT_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)


# ─── helpers ───────────────────────────────────────────────────────────
def load_canvas(tag: str) -> pd.DataFrame:
    p = CANVAS_DIR / tag / "results.jsonl"
    rows = []
    with open(p) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    df = pd.DataFrame(rows)
    keep = ["uid", "pair_code", "cs_display", "f1_base", "f1_canvas", "f1_en"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["delta"] = df["f1_canvas"] - df["f1_base"]
    return df


def load_anchor(tag: str) -> pd.DataFrame:
    p = ANCHOR_DIR / tag / "metrics_per_example.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def hedges_dz(delta: np.ndarray) -> float:
    n = len(delta)
    if n < 2:
        return float("nan")
    sd = float(np.std(delta, ddof=1))
    if sd == 0:
        return 0.0
    correction = 1.0 - 3.0 / (4.0 * n - 5.0) if n > 2 else 1.0
    return correction * float(np.mean(delta)) / sd


def boot_ci_mean(delta: np.ndarray, B: int, rng: np.random.Generator) -> Tuple[float, float]:
    n = len(delta)
    if n == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(B, n))
    means = delta[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bh_fdr(pvals: List[float], q: float = 0.05) -> Tuple[List[float], List[bool]]:
    """Benjamini-Hochberg adjusted q-values; returns (q_adj, significant)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q_adj_sorted = np.minimum.accumulate(
        (ranked * n / (np.arange(n) + 1))[::-1]
    )[::-1]
    q_adj = np.empty_like(q_adj_sorted)
    q_adj[order] = np.clip(q_adj_sorted, 0.0, 1.0)
    sig = q_adj < q
    return q_adj.tolist(), sig.tolist()


def cohen_d_indep(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


# ─── 1. per-cell stats ─────────────────────────────────────────────────
print("Loading paired CANVAS data ...")
canvas_data: Dict[str, pd.DataFrame] = {}
for tag in PRIMARY_MODELS:
    df = load_canvas(tag)
    if df.empty:
        print(f"  [warn] {tag}: empty"); continue
    canvas_data[tag] = df
    print(f"  {tag}: {len(df)} rows")

cell_rows: List[Dict] = []
pvals: List[float] = []
cell_keys: List[Tuple[str, str]] = []

for tag in PRIMARY_MODELS:
    if tag not in canvas_data: continue
    df = canvas_data[tag]
    for cond in CONDS:
        sub = df[df["cs_display"] == cond]
        delta = sub["delta"].dropna().to_numpy(dtype=float)
        n = len(delta)
        if n < 10:
            continue
        mean_d = float(np.mean(delta))
        med_d  = float(np.median(delta))
        try:
            w_stat, w_p = stats.wilcoxon(
                delta, alternative="greater", zero_method="wilcox",
                method="approx" if n > 50 else "exact",
            )
            w_p = float(w_p)
        except ValueError:
            w_stat, w_p = float("nan"), 1.0
        dz   = hedges_dz(delta)
        ci_l, ci_h = boot_ci_mean(delta, B_BOOT_CELL, RNG)
        cell_rows.append({
            "model": tag, "model_disp": DISPLAY[tag],
            "cs_display": cond, "n": n,
            "mean_delta_f1": mean_d,
            "median_delta_f1": med_d,
            "wilcoxon_W": float(w_stat) if not (isinstance(w_stat, float) and math.isnan(w_stat)) else float("nan"),
            "wilcoxon_p": w_p,
            "hedges_dz":  dz,
            "ci_low":     ci_l,
            "ci_high":    ci_h,
        })
        pvals.append(w_p)
        cell_keys.append((tag, cond))

q_adj, sig = bh_fdr(pvals, q=0.05)
for r, q, s in zip(cell_rows, q_adj, sig):
    r["q_adj"] = q
    r["significant_q05"] = bool(s)

cell_df = pd.DataFrame(cell_rows)
cell_df.to_csv(OUT_DIR / "canvas_stats.csv", index=False)
print(f"\n[saved] {OUT_DIR/'canvas_stats.csv'}  ({len(cell_df)} cells)")

# ─── 2. cross-model summary per condition ──────────────────────────────
summary_rows: List[Dict] = []
for cond in CONDS:
    per_model_mean = cell_df[cell_df["cs_display"] == cond]["mean_delta_f1"].to_numpy()
    n_mod = len(per_model_mean)
    if n_mod < 2:
        continue
    mu = float(per_model_mean.mean())
    idx = RNG.integers(0, n_mod, size=(B_BOOT_MODEL, n_mod))
    boot_means = per_model_mean[idx].mean(axis=1)
    ci_l = float(np.percentile(boot_means, 2.5))
    ci_h = float(np.percentile(boot_means, 97.5))
    n_pos = int((per_model_mean > 0).sum())
    try:
        sign_p = float(stats.binomtest(n_pos, n_mod, p=0.5, alternative="greater").pvalue)
    except AttributeError:
        sign_p = float(stats.binom_test(n_pos, n_mod, p=0.5, alternative="greater"))
    summary_rows.append({
        "scope": "per_condition", "cs_display": cond,
        "n_models": n_mod,
        "mean_of_means": mu, "ci_low": ci_l, "ci_high": ci_h,
        "n_models_positive": n_pos, "sign_test_p": sign_p,
    })

# Overall (averaged across conditions per model first, then bootstrap)
per_model_overall = cell_df.groupby("model")["mean_delta_f1"].mean().to_numpy()
n_mod = len(per_model_overall)
mu = float(per_model_overall.mean())
idx = RNG.integers(0, n_mod, size=(B_BOOT_MODEL, n_mod))
boot_means = per_model_overall[idx].mean(axis=1)
ci_l = float(np.percentile(boot_means, 2.5))
ci_h = float(np.percentile(boot_means, 97.5))
n_pos = int((per_model_overall > 0).sum())
try:
    sign_p = float(stats.binomtest(n_pos, n_mod, p=0.5, alternative="greater").pvalue)
except AttributeError:
    sign_p = float(stats.binom_test(n_pos, n_mod, p=0.5, alternative="greater"))
summary_rows.append({
    "scope": "overall", "cs_display": "ALL",
    "n_models": n_mod,
    "mean_of_means": mu, "ci_low": ci_l, "ci_high": ci_h,
    "n_models_positive": n_pos, "sign_test_p": sign_p,
})

summary_df = pd.DataFrame(summary_rows)

# ─── 3. anchor-bias separation (per model) + AB-ΔF1 correlation ────────
ab_rows: List[Dict] = []
corr_pairs: List[Tuple[float, float]] = []   # (AB_mean per cell, delta_f1 per cell)

for tag in PRIMARY_MODELS:
    a = load_anchor(tag)
    if a.empty: continue
    # Try the best anchor-bias proxy that exists in this file
    ab_col = None
    for c in ("anchor_bias_cos_qonly", "anchor_bias_cos_full",
              "anchor_bias_latelayer_cos_qonly", "anchor_bias_cos_q_pure"):
        if c in a.columns:
            ab_col = c; break
    if ab_col is None or "cs_display" not in a.columns:
        continue
    ab_src = a.loc[a["cs_display"] == "GF_Src", ab_col].dropna().to_numpy()
    ab_tgt = a.loc[a["cs_display"] == "GF_Tgt", ab_col].dropna().to_numpy()
    d = cohen_d_indep(ab_src, ab_tgt)
    ab_rows.append({
        "model": tag, "model_disp": DISPLAY[tag],
        "ab_col": ab_col,
        "ab_GF_Src_mean": float(ab_src.mean()) if len(ab_src) else float("nan"),
        "ab_GF_Tgt_mean": float(ab_tgt.mean()) if len(ab_tgt) else float("nan"),
        "cohens_d_src_vs_tgt": d,
        "n_src": len(ab_src), "n_tgt": len(ab_tgt),
    })
    # For AB-ΔF1 correlation: per-cell mean AB + ΔF1
    if "delta_f1_CS_vs_EN" in a.columns:
        for cond in CONDS:
            sub = a[a["cs_display"] == cond]
            if len(sub) < 5: continue
            corr_pairs.append((float(sub[ab_col].mean()),
                                float(sub["delta_f1_CS_vs_EN"].mean())))

ab_df = pd.DataFrame(ab_rows)
ab_df.to_csv(OUT_DIR / "canvas_anchor_separation.csv", index=False)

# Spearman correlation with bootstrap CI
if len(corr_pairs) >= 10:
    xs = np.array([p[0] for p in corr_pairs])
    ys = np.array([p[1] for p in corr_pairs])
    rho_obs, p_obs = stats.spearmanr(xs, ys)
    boot_rhos = []
    n = len(xs)
    for _ in range(B_BOOT_CORR):
        idx = RNG.integers(0, n, size=n)
        try:
            r, _ = stats.spearmanr(xs[idx], ys[idx])
            if not math.isnan(r): boot_rhos.append(r)
        except Exception:
            pass
    rho_lo = float(np.percentile(boot_rhos, 2.5)) if boot_rhos else float("nan")
    rho_hi = float(np.percentile(boot_rhos, 97.5)) if boot_rhos else float("nan")
    corr_summary = {
        "rho": float(rho_obs), "p": float(p_obs),
        "rho_ci_low": rho_lo, "rho_ci_high": rho_hi,
        "n_cells": n,
    }
else:
    corr_summary = {"rho": float("nan"), "p": float("nan"),
                    "rho_ci_low": float("nan"), "rho_ci_high": float("nan"),
                    "n_cells": len(corr_pairs)}

summary_df.to_csv(OUT_DIR / "canvas_stats_summary.csv", index=False)
print(f"[saved] {OUT_DIR/'canvas_stats_summary.csv'}")
print(f"[saved] {OUT_DIR/'canvas_anchor_separation.csv'}  ({len(ab_df)} models)")

# ─── 4. emit LaTeX macros for inline prose ─────────────────────────────
def texify(x: float, dp: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{dp}f}"

n_total = len(cell_df)
n_sig   = int(cell_df["significant_q05"].sum())
n_pos   = int((cell_df["mean_delta_f1"] > 0).sum())
overall = next(r for r in summary_rows if r["scope"] == "overall")
gf_src  = next((r for r in summary_rows if r["cs_display"] == "GF_Src"), None)
gf_tgt  = next((r for r in summary_rows if r["cs_display"] == "GF_Tgt"), None)
rand_   = next((r for r in summary_rows if r["cs_display"] == "Random"), None)
sel_    = next((r for r in summary_rows if r["cs_display"] == "Selective"), None)

# dz range
dz_min = float(cell_df["hedges_dz"].min())
dz_max = float(cell_df["hedges_dz"].max())

# AB separation: mean Cohen's d across models with both groups
ab_dvals = ab_df["cohens_d_src_vs_tgt"].dropna().tolist() if not ab_df.empty else []
ab_d_mean = float(np.mean(ab_dvals)) if ab_dvals else float("nan")
ab_d_min  = float(np.min(ab_dvals))  if ab_dvals else float("nan")
ab_d_max  = float(np.max(ab_dvals))  if ab_dvals else float("nan")

macros = [
    f"% Auto-generated by compute_canvas_stats.py. Do not edit by hand.",
    f"\\newcommand{{\\nModels}}{{{len(PRIMARY_MODELS)}}}",
    f"\\newcommand{{\\nCells}}{{{n_total}}}",
    f"\\newcommand{{\\nCellsSignificant}}{{{n_sig}}}",
    f"\\newcommand{{\\nCellsPositive}}{{{n_pos}}}",
    f"\\newcommand{{\\meanDeltaCanvas}}{{{texify(overall['mean_of_means']*100, 2)}}}",
    f"\\newcommand{{\\meanDeltaCanvasCIlo}}{{{texify(overall['ci_low']*100, 2)}}}",
    f"\\newcommand{{\\meanDeltaCanvasCIhi}}{{{texify(overall['ci_high']*100, 2)}}}",
    f"\\newcommand{{\\nModelsPositive}}{{{overall['n_models_positive']}}}",
    f"\\newcommand{{\\signTestP}}{{{texify(overall['sign_test_p'], 4)}}}",
    f"\\newcommand{{\\dzMin}}{{{texify(dz_min, 2)}}}",
    f"\\newcommand{{\\dzMax}}{{{texify(dz_max, 2)}}}",
]
for cond_row, prefix in [(gf_src, "GFSrc"), (gf_tgt, "GFTgt"),
                         (rand_, "Random"), (sel_, "Selective")]:
    if cond_row is None: continue
    macros += [
        f"\\newcommand{{\\meanDelta{prefix}}}{{{texify(cond_row['mean_of_means']*100, 2)}}}",
        f"\\newcommand{{\\meanDelta{prefix}CIlo}}{{{texify(cond_row['ci_low']*100, 2)}}}",
        f"\\newcommand{{\\meanDelta{prefix}CIhi}}{{{texify(cond_row['ci_high']*100, 2)}}}",
        f"\\newcommand{{\\nPos{prefix}}}{{{cond_row['n_models_positive']}}}",
        f"\\newcommand{{\\signTest{prefix}P}}{{{texify(cond_row['sign_test_p'], 4)}}}",
    ]

macros += [
    f"\\newcommand{{\\anchorDzMean}}{{{texify(ab_d_mean, 2)}}}",
    f"\\newcommand{{\\anchorDzMin}}{{{texify(ab_d_min, 2)}}}",
    f"\\newcommand{{\\anchorDzMax}}{{{texify(ab_d_max, 2)}}}",
    f"\\newcommand{{\\anchorNmodels}}{{{len(ab_dvals)}}}",
    f"\\newcommand{{\\corrRho}}{{{texify(corr_summary['rho'], 3)}}}",
    f"\\newcommand{{\\corrRhoCIlo}}{{{texify(corr_summary['rho_ci_low'], 3)}}}",
    f"\\newcommand{{\\corrRhoCIhi}}{{{texify(corr_summary['rho_ci_high'], 3)}}}",
    f"\\newcommand{{\\corrPval}}{{{texify(corr_summary['p'], 4)}}}",
    f"\\newcommand{{\\corrNcells}}{{{corr_summary['n_cells']}}}",
]

(TBL_DIR / "canvas_statistics_numbers.tex").write_text("\n".join(macros) + "\n",
                                                       encoding="utf-8")
print(f"[saved] {TBL_DIR/'canvas_statistics_numbers.tex'}")

# ─── 5. emit the main statistical table (LaTeX) ────────────────────────
def fmt_pct(x: float) -> str:
    if math.isnan(x): return "--"
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x)*100:.2f}"


def _panel(cond: str) -> List[str]:
    out = [f"\\begin{{minipage}}[t]{{0.48\\textwidth}}",
           "\\centering",
           f"\\textbf{{{COND_DISPLAY[cond]}}} \\\\[2pt]",
           "\\begin{tabular}{@{}lrccc@{}}",
           "\\toprule",
           "Model & $\\Delta$F1 & 95\\% CI & $d_z$ & $q$ \\\\",
           "\\midrule"]
    for tag in PRIMARY_MODELS:
        sel = cell_df[(cell_df["model"] == tag) & (cell_df["cs_display"] == cond)]
        label = f"\\modelwithlogo{{{LOGO[tag]}}}{{{PAPER_NAME[tag]}}}"
        if sel.empty:
            out.append(f"{label} & --- & --- & --- & --- \\\\"); continue
        r  = sel.iloc[0]
        d  = fmt_pct(r["mean_delta_f1"])
        ci = f"$[{fmt_pct(r['ci_low'])}, {fmt_pct(r['ci_high'])}]$"
        dz = "---" if math.isnan(r["hedges_dz"]) else f"${r['hedges_dz']:.2f}$"
        q  = r["q_adj"]
        qstr = "$<$0.001" if q < 0.001 else (f"${q:.3f}$" if q < 1.0 else "$1.000$")
        d = f"${d}$"
        if r["significant_q05"]:
            qstr = "$\\mathbf{" + (f"<\\!0.001" if r["q_adj"] < 0.001 else f"{r['q_adj']:.3f}") + "}$"
            d    = f"$\\mathbf{{{fmt_pct(r['mean_delta_f1'])}}}$"
        out.append(f"{label} & {d} & {ci} & {dz} & {qstr} \\\\")
    out.append("\\midrule")
    sel = cell_df[cell_df["cs_display"] == cond]
    if not sel.empty:
        m  = sel["mean_delta_f1"].mean()
        cl = sel["ci_low"].mean()
        ch = sel["ci_high"].mean()
        dz = sel["hedges_dz"].mean()
        out.append("\\textit{Mean} & "
                   f"${fmt_pct(m)}$ & "
                   f"$[{fmt_pct(cl)}, {fmt_pct(ch)}]$ & "
                   f"${dz:.2f}$ & --- \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{minipage}"]
    return out


n_cells_total = len(cell_df)
n_models_pool = len(PRIMARY_MODELS)

lines = [
    "% Auto-generated by compute_canvas_stats.py. 2x2 panel grid to avoid",
    "% horizontal overflow. Bold cells are significant at q<0.05 after",
    "% Benjamini-Hochberg FDR. Model labels use \\modelwithlogo to match",
    "% the convention of Table 1 / canvas.tex.",
    "\\begin{table*}[t]",
    "\\centering",
    "\\small",
    "\\setlength{\\tabcolsep}{3pt}",
    "",
]
# Top row: GF-Src | GF-Tgt
lines += _panel("GF_Src")
lines.append("\\hfill")
lines += _panel("GF_Tgt")
lines += ["", "\\vspace{0.9em}", ""]
# Bottom row: Random | Selective
lines += _panel("Random")
lines.append("\\hfill")
lines += _panel("Selective")

lines += [
    "",
    "\\caption{\\textbf{Statistical analysis of CANVAS, per condition.} "
    "Each panel reports the per-model paired difference "
    "$\\Delta$F1 $=$ \\textsc{CANVAS}$_\\text{adapt}$ $-$ \\textsc{Base} "
    "(F1\\,$\\times$\\,100) on the full evaluation set. 95\\,\\% CIs are from "
    "$10{,}000$ paired bootstrap resamples of question examples; $d_z$ is the "
    "Hedges-corrected paired effect size; $q$ is the Benjamini--Hochberg-adjusted "
    "Wilcoxon signed-rank $p$-value (one-sided, alternative $>0$) across all "
    f"${n_models_pool}{{\\times}}4{{=}}{n_cells_total}$ cells. Bold entries are "
    "significant at $q<0.05$. The \\textit{Mean} row averages the per-model "
    "statistics within each condition.}",
    "\\label{tab:canvas-stats}",
    "\\end{table*}",
]
(TBL_DIR / "canvas_statistics.tex").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")
print(f"[saved] {TBL_DIR/'canvas_statistics.tex'}")

# ─── 6. brief stdout summary ───────────────────────────────────────────
print("\n" + "="*70)
print("  Per-cell stats  (* = significant at q<0.05)")
print("="*70)
for _, r in cell_df.iterrows():
    flag = "*" if r["significant_q05"] else " "
    print(f"  {flag} {r['model_disp']:<18} {r['cs_display']:<10} "
          f"ΔF1={r['mean_delta_f1']*100:+.2f}  "
          f"CI=[{r['ci_low']*100:+.2f},{r['ci_high']*100:+.2f}]  "
          f"d_z={r['hedges_dz']:+.2f}  q={r['q_adj']:.4f}")

print(f"\n  → {n_sig}/{n_total} cells significant at q<0.05 (BH-FDR).")
print(f"  → {n_pos}/{n_total} cells have positive mean ΔF1.")

print("\n  Cross-model per-condition:")
for r in summary_rows:
    print(f"    {r['scope']:<15} {r['cs_display']:<10} "
          f"mean={r['mean_of_means']*100:+.2f}  "
          f"CI=[{r['ci_low']*100:+.2f},{r['ci_high']*100:+.2f}]  "
          f"sign={r['n_models_positive']}/{r['n_models']}  "
          f"p_sign={r['sign_test_p']:.4f}")

print("\n  Anchor-bias separation (Cohen's d, GF-Src vs GF-Tgt):")
for _, r in ab_df.iterrows():
    print(f"    {r['model_disp']:<18} d={r['cohens_d_src_vs_tgt']:+.2f}  "
          f"({r['ab_col']})")

print(f"\n  AB ↔ ΔF1 Spearman: ρ={corr_summary['rho']:+.3f}  "
      f"CI=[{corr_summary['rho_ci_low']:+.3f},{corr_summary['rho_ci_high']:+.3f}]  "
      f"p={corr_summary['p']:.4f}  n_cells={corr_summary['n_cells']}")

print(f"\n[done] all outputs in {OUT_DIR}")
