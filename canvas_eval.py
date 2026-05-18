"""CANVAS evaluation driver.

For each (model, example) we compute three F1 scores:

  - Base    : direct CS answering (greedy)             [reference lower bound]
  - EN-ora  : answer the gold EN question              [reference upper bound]
  - CANVAS  : translation-free decoding-time steering  [our method]

Usage:
  python canvas_eval.py analyze \\
      --data_jsonl data/pairs_with_tgt.jsonl \\
      --out_dir    outputs/canvas \\
      --tag        qwen35_4b \\
      --max_len 256 --max_new_tokens 48
  # smoke: --max_samples 50

  python canvas_eval.py report --out_dir outputs/canvas
"""

from __future__ import annotations
import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).parent))
from unified_repr import (
    ModelCfg, MODEL_MAP,
    QA_SYSTEM, QA_SYSTEM_PHI, CS_METHOD_NORM,
    load_hf, flush_gpu, chat_text,
    f1, em, gen_hf,
)
from canvas import CANVASConfig, steered_generate


# Per-model transition_frac: relative depth where steering begins.
# Empirically derived from per-layer anchor-bias curves (unified_repr.py);
# models without measured data fall back to 0.70.
TRANSITION_FRAC: Dict[str, float] = {
    "qwen35_4b":      0.69,
    "qwen35_9b":      0.78,
    "qwen35_27b":     0.84,
    "aya_8b":         0.50,
    "llama31_70b":    0.72,
    "llama33_70b":    0.78,
    "qwen35_35b_moe": 0.93,
    "qwen3_30b_moe":  0.70,
    "phi35_moe":      0.70,
    "mixtral_8x7b":   0.70,
    "qwen3_06b":      0.70,
    "llama32_1b":     0.70,
    "tiny_aya_3b":    0.70,
    "mistral_7b":     0.70,
    "llama31_8b":     0.70,
    "qwen36_35b_moe": 0.70,
}


def run_canvas(cfg: ModelCfg,
               data: List[Dict],
               out_dir: Path,
               max_len: int = 256,
               max_new: int = 48,
               max_samples: int = 0,
               transition_frac: Optional[float] = None,
               interp_alpha: float = 0.45,
               interp_alpha_adaptive: bool = True,
               interp_alpha_k: float = 1.5,
               interp_alpha_min: float = 0.05,
               interp_alpha_max: float = 0.75) -> None:
    tag = cfg.tag
    mdir = out_dir / "model_runs" / tag
    mdir.mkdir(parents=True, exist_ok=True)
    raw_path = mdir / "results.jsonl"

    if transition_frac is None:
        transition_frac = TRANSITION_FRAC.get(tag, 0.70)

    canvas_cfg = CANVASConfig(
        transition_frac=transition_frac,
        interp_alpha=interp_alpha,
        interp_alpha_adaptive=interp_alpha_adaptive,
        interp_alpha_k=interp_alpha_k,
        interp_alpha_min=interp_alpha_min,
        interp_alpha_max=interp_alpha_max,
    )

    # Resume from any prior partial run
    done: set = set()
    if raw_path.exists():
        with open(raw_path) as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    done.add(f"{obj['uid']}|{obj['cs_display']}")
                except Exception:
                    pass
        print(f"  [resume] {len(done)} examples done")

    examples = data if max_samples <= 0 else data[:max_samples]
    todo = [
        ex for ex in examples
        if f"{ex.get('uid', '')}|"
           f"{CS_METHOD_NORM.get(ex.get('cs_method', ''), ex.get('cs_method', ''))}"
           not in done
    ]
    if not todo:
        print(f"  [skip] all done for {tag}")
        return

    print(f"\n{'='*60}\n  TAG: {tag}   CANVAS  "
          f"(transition={transition_frac:.2f}, alpha_base={interp_alpha}, "
          f"k={interp_alpha_k}, alpha_range=[{interp_alpha_min},{interp_alpha_max}], "
          f"adaptive={interp_alpha_adaptive})\n{'='*60}")

    model, tok = load_hf(cfg)
    sys_cs = QA_SYSTEM_PHI if cfg.use_phi_gen else QA_SYSTEM
    sys_en = sys_cs

    nan = float("nan")

    for ex in tqdm(todo, desc=f"[{tag}]"):
        uid       = ex.get("uid", ex.get("original_index", "?"))
        cs_method = CS_METHOD_NORM.get(ex.get("cs_method", ""), ex.get("cs_method", ""))
        q_cs      = str(ex.get("question_cs", "") or "")
        q_en_gold = str(ex.get("question_en", "") or "")
        pair_code = ex.get("pair_code", "")
        gold      = str(ex.get("gold_answer", "") or "")
        if not q_cs or not gold:
            continue

        row: Dict = {
            "uid": uid, "pair_code": pair_code,
            "cs_display": cs_method, "gold_answer": gold,
        }

        # Base CS answering
        t0 = time.perf_counter()
        try:
            pred = gen_hf(model, tok, q_cs, cfg, max_len, max_new, system=sys_cs)
            row["pred_base"] = pred
            row["f1_base"]   = f1(pred, gold)
            row["em_base"]   = em(pred, gold)
        except Exception as e:
            print(f"  [WARN] baseline uid={uid}: {e}")
            row.update({"pred_base": "", "f1_base": nan, "em_base": nan})
        row["base_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # EN oracle
        if q_en_gold:
            t0 = time.perf_counter()
            try:
                pred = gen_hf(model, tok, q_en_gold, cfg, max_len, max_new, system=sys_en)
                row["pred_en"] = pred
                row["f1_en"]   = f1(pred, gold)
                row["em_en"]   = em(pred, gold)
            except Exception as e:
                print(f"  [WARN] EN oracle uid={uid}: {e}")
                row.update({"pred_en": "", "f1_en": nan, "em_en": nan})
            row["en_oracle_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        # CANVAS (translation-free steering)
        t0 = time.perf_counter()
        try:
            prompt = chat_text(tok, q_cs, cfg.qwen_thinking,
                               system=sys_cs,
                               no_system_prompt=cfg.no_system_prompt)
            pred_canvas, diag = steered_generate(
                model, tok, prompt, q_cs, pair_code,
                max_len=max_len, max_new=max_new, cfg=canvas_cfg)
            row["pred_canvas"] = pred_canvas
            row["f1_canvas"]   = f1(pred_canvas, gold)
            row["em_canvas"]   = em(pred_canvas, gold)
            row["canvas_alignment"] = round(float(diag.get("alignment", 0.0)), 6)
            row["canvas_alpha"]     = round(float(diag.get("eff_alpha", canvas_cfg.interp_alpha)), 4)
            row["canvas_n_en"]      = int(diag.get("n_en", 0))
            row["canvas_n_tgt"]     = int(diag.get("n_tgt", 0))
            row["canvas_n_layers_steered"] = int(diag.get("n_upper", 0))
        except Exception as e:
            import traceback
            print(f"  [WARN] CANVAS uid={uid}: {e}")
            traceback.print_exc()
            row.update({"pred_canvas": "", "f1_canvas": nan, "em_canvas": nan,
                        "canvas_alignment": nan, "canvas_alpha": nan,
                        "canvas_n_en": 0, "canvas_n_tgt": 0,
                        "canvas_n_layers_steered": 0})
        row["canvas_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        with open(raw_path, "a") as fp:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    flush_gpu(model)
    del model, tok
    print(f"  DONE: {tag}")
    _save_summary(raw_path, mdir, tag)


def _save_summary(raw_path: Path, mdir: Path, tag: str) -> None:
    rows = []
    with open(raw_path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    if not rows:
        print("  [warn] no rows to summarize")
        return
    df = pd.DataFrame(rows)
    df.to_csv(mdir / "results.csv", index=False)

    f1_cols = [c for c in df.columns if c.startswith("f1_")]
    by_cond = df.groupby("cs_display")[f1_cols].mean().round(4)
    by_cond.to_csv(mdir / "summary_by_cs_display.csv")

    methods = [
        ("Base",   "f1_base"),
        ("EN-ora", "f1_en"),
        ("CANVAS", "f1_canvas"),
    ]
    conds = ["GF_Src", "GF_Tgt", "Random", "Selective"]
    rows_out = []
    for name, col in methods:
        if col not in df.columns:
            continue
        per_c = df.groupby("cs_display")[col].mean().reindex(conds)
        avg = float(np.nanmean(per_c.values))
        rows_out.append([name] + [round(per_c.get(c, np.nan), 4) for c in conds] + [round(avg, 4)])
    cmp_df = pd.DataFrame(rows_out, columns=["method"] + conds + ["avg"])
    cmp_df.to_csv(mdir / "comparison_table.csv", index=False)

    print(f"\n{'='*82}\n  {tag} - CANVAS vs Base / EN-ora\n{'='*82}")
    print(f"  {'Method':<8s}  {'GF_Src':>7s} {'GF_Tgt':>7s} {'Random':>7s} {'Selectiv':>8s}   {'Avg':>7s}")
    print("  " + "-"*72)
    for _, r in cmp_df.iterrows():
        print(f"  {r['method']:<8s}  {r['GF_Src']*100:7.2f} {r['GF_Tgt']*100:7.2f} "
              f"{r['Random']*100:7.2f} {r['Selective']*100:8.2f}   {r['avg']*100:7.2f}")
    if "canvas_alpha" in df.columns:
        alpha_stats = df["canvas_alpha"].dropna()
        align_stats = df["canvas_alignment"].dropna()
        if len(alpha_stats):
            print(f"\n  [diag] alpha: mean={alpha_stats.mean():.3f}  "
                  f"min={alpha_stats.min():.2f}  max={alpha_stats.max():.2f}")
            print(f"  [diag] alignment: mean={align_stats.mean():+.4f}  "
                  f"std={align_stats.std():.4f}")


def cmd_report(out_dir: Path) -> None:
    runs_dir = out_dir / "model_runs"
    if not runs_dir.exists():
        print(f"  [error] {runs_dir} not found")
        return
    rows = []
    for mrun in sorted(runs_dir.iterdir()):
        ct = mrun / "comparison_table.csv"
        if not ct.exists():
            continue
        df = pd.read_csv(ct)
        df["model"] = mrun.name
        rows.append(df)
    if not rows:
        print("  [warn] no per-model comparisons found")
        return
    all_df = pd.concat(rows, ignore_index=True)
    all_df.to_csv(out_dir / "all_models_comparison.csv", index=False)
    print(f"  [saved] {out_dir/'all_models_comparison.csv'}")

    agg = all_df.groupby(["method"])[["GF_Src", "GF_Tgt", "Random", "Selective", "avg"]].mean().round(4)
    print("\n=== Cross-model CANVAS aggregate ===")
    print(agg)
    agg.to_csv(out_dir / "all_models_summary.csv")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="run CANVAS for one model")
    a.add_argument("--data_jsonl", type=Path, required=True)
    a.add_argument("--out_dir",    type=Path, required=True)
    a.add_argument("--tag",        type=str, required=True,
                   help=f"Model tag, one of: {list(MODEL_MAP)}")
    a.add_argument("--max_len",        type=int, default=256)
    a.add_argument("--max_new_tokens", type=int, default=48)
    a.add_argument("--max_samples",    type=int, default=0)
    a.add_argument("--transition_frac", type=float, default=None,
                   help="Override per-model transition layer (relative depth).")
    a.add_argument("--interp_alpha",     type=float, default=0.45,
                   help="Base interpolation strength (paper default 0.45).")
    a.add_argument("--no_adaptive_alpha", action="store_true",
                   help="Disable adaptive alpha; use --interp_alpha as fixed value.")
    a.add_argument("--interp_alpha_k",   type=float, default=1.5,
                   help="Slope of adaptive alpha vs canvas-alignment gamma "
                        "(alpha = base - k*gamma).")
    a.add_argument("--interp_alpha_min", type=float, default=0.05,
                   help="Floor for adaptive alpha (paper default 0.05).")
    a.add_argument("--interp_alpha_max", type=float, default=0.75,
                   help="Ceiling for adaptive alpha (paper default 0.75).")

    r = sub.add_parser("report", help="aggregate across models")
    r.add_argument("--out_dir", type=Path, required=True)

    args = p.parse_args()

    if args.cmd == "analyze":
        if args.tag not in MODEL_MAP:
            print(f"[error] unknown tag '{args.tag}'. Valid: {list(MODEL_MAP)}")
            return
        cfg = MODEL_MAP[args.tag]
        with open(args.data_jsonl) as f:
            data = [json.loads(line) for line in f if line.strip()]
        run_canvas(cfg, data, args.out_dir,
                   max_len=args.max_len, max_new=args.max_new_tokens,
                   max_samples=args.max_samples,
                   transition_frac=args.transition_frac,
                   interp_alpha=args.interp_alpha,
                   interp_alpha_adaptive=not args.no_adaptive_alpha,
                   interp_alpha_k=args.interp_alpha_k,
                   interp_alpha_min=args.interp_alpha_min,
                   interp_alpha_max=args.interp_alpha_max)
    elif args.cmd == "report":
        cmd_report(args.out_dir)


if __name__ == "__main__":
    main()
