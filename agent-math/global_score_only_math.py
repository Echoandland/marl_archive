#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
global_score_only_math.py

Purpose:
- Read interaction logs (problem_*_interaction*.json) and corresponding results (problem_*.json).
- Compute ONLY the global score using the exact same method as interaction_to_score_async_math.py:
  * Use LLM-based equivalence judgement between final_answer and golden_answer.
  * Answer match -> +5.0, mismatch -> -5.0.
  * Preferred sources (when available): result.predict_answer / result.golden_answer.
  * If result file is missing, or fields are empty, fallback to interaction['final_decision'].

Outputs:
- Writes one JSON per case: <prefix>_global.json in data_dir, with shape:
  {
    "global": {"score": float, "match": bool},
    "source": {"interaction": "...", "result": "..."}
  }
"""

import os
import re
import glob
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from typing import Dict, Any, Optional

# Reuse the exact global scoring method
from interaction_to_score_async_math import _evaluate_global_with_llm


def _prefix_from_interact(path: str) -> str:
    """
    Given a path like problem_12_interaction_useexperience.json,
    return 'problem_12_useexperience' (remove the first occurrence of '_interaction').
    """
    base = os.path.basename(path)
    name = base[:-5] if base.lower().endswith(".json") else base
    return name.replace("_interaction", "", 1)


def _result_path_from_prefix(base_dir: str, prefix: str) -> str:
    return os.path.join(base_dir, f"{prefix}.json")


def _compute_global_result(interaction: Dict[str, Any], result_dict: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Mirror selection in interaction_to_score_async_math.run_one:
    - final_answer from result.predict_answer, else fallback to interaction.final_decision.final_answer
    - golden_answer from result.golden_answer
    - compute global via _evaluate_global_with_llm
    """
    final_answer = ""
    golden = ""

    if result_dict:
        final_answer = (result_dict.get("predict_answer") or "").strip()
        golden = (result_dict.get("golden_answer") or "").strip()

    # Fallbacks from interaction
    fd = interaction.get("final_decision") if isinstance(interaction, dict) else None
    if isinstance(fd, dict):
        if not final_answer:
            final_answer = (fd.get("final_answer") or "").strip()
        if not golden:
            golden = (fd.get("golden_answer") or "").strip()

    g = _evaluate_global_with_llm(final_answer, golden)
    # Return only the standardized global section, keep raw internals minimal
    return {
        "global": {"score": float(g.get("score", 0.0)), "match": bool(g.get("match", False))}
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Directory containing problem_*_interaction*.json and optionally problem_*.json")
    ap.add_argument("--results_dir", default=None, help="Optional directory where problem_*.json are stored (if different from data_dir)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    interact_files = sorted(glob.glob(os.path.join(args.data_dir, "problem_*_interaction*.json")))
    if not interact_files:
        raise FileNotFoundError(f"No interaction files in {args.data_dir}")

    base_results_dir = args.results_dir if args.results_dir else args.data_dir
    os.makedirs(args.data_dir, exist_ok=True)

    def _run_one(ipath: str):
        prefix = _prefix_from_interact(ipath)
        rpath = _result_path_from_prefix(base_results_dir, prefix)
        interaction = json.load(open(ipath, "r", encoding="utf-8-sig"))
        result_dict = None
        if os.path.exists(rpath):
            result_dict = json.load(open(rpath, "r", encoding="utf-8-sig"))
        out = _compute_global_result(interaction, result_dict)
        # Include provenance
        out["source"] = {
            "interaction": os.path.basename(ipath),
            "result": os.path.basename(rpath) if os.path.exists(rpath) else None
        }
        out_path = os.path.join(args.data_dir, f"{prefix}_global.json")
        json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return out_path, bool(out.get("global", {}).get("match", False))

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(_run_one, ip): ip for ip in interact_files}
        total = 0
        correct = 0
        for fut in tqdm(as_completed(futs), total=len(futs), disable=not args.verbose, desc="global"):
            out_path, is_match = fut.result()
            total += 1
            correct += 1 if is_match else 0
            print(f"[OK] {out_path}")

    # Aggregate accuracy over all processed items
    if total > 0:
        accuracy = correct / total
        metrics = {"total": total, "correct": correct, "accuracy": accuracy}
        metrics_path = os.path.join(args.data_dir, "global_metrics.json")
        json.dump(metrics, open(metrics_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"[METRICS] Accuracy: {accuracy*100:.2f}% ({correct}/{total}) -> {metrics_path}")
    else:
        print("[METRICS] No files processed; accuracy not computed.")


if __name__ == "__main__":
    main()


