#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_to_experience_math.py
从 *score.json + 对应 result(problem_x.json) 中，抽取“可检索经验”：
- OPINION：专家如何组织/修正解题步骤(S-steps)、处理同伴意见(delta)、选择策略/不变量
- REVIEW：同行审阅如何识别错误类别（algebra_error/logic_gap/missing_case...）、如何给出可执行修正

输出：按角色分桶，写入 out_dir/<Role>.experiences.jsonl （逐条 JSON 行）
"""

import argparse
import json
import os
import glob
import re
from typing import Any, Dict, List, Tuple, Optional
from tqdm.auto import tqdm

# 用你的 Agent（和医学版一致）
try:
    from utils_with_experience_math import Agent
except Exception:
    from utils_with_experience import Agent


EXTRACTION_PROMPT_TEMPLATE = """You are an expert math reasoning analyzer.

GOAL
From a {{ROLE, PROMPT, RESPONSE, GOLDEN_ANSWER, EVAL_ANALYSIS, FINAL_SCORE}} bundle,
produce retrieval-ready key:value experiences that are ABSTRACT and TRANSFERABLE across problems.

OUTPUT TYPES (two only)
1) OPINION — how a specialist handles received context/constraints/peer feedback to build solution steps.
2) REVIEW — how a specialist peer-reviews a step or a set of related steps.

STRICT GENERALIZATION RULES
- Make each item generalizable. Do NOT copy any problem-specific symbols, constants, indices, labels, or named objects.
  * Avoid quoting variables (like x, y, S2, (3,8), E_r etc.), numeric values, or bespoke geometry/topology objects from the case.
  * Refer abstractly instead (e.g., “destination group”, “boundary cases”, “invertibility check”, “case split criterion”).
- Do NOT introduce new theorems/facts/data not present in the inputs.
- Prefer transferable reasoning guidance: invariant choice, step ordering, safe algebraic transforms, case coverage, counterexample checks, error classes.

SCORING LANGUAGE
- Each EXPERIENCE must start with:
  * “Good practice: …” when behavior aligns with GOLDEN_ANSWER/EVAL_ANALYSIS or clearly improves derivation.
  * “Pitfall: …” when it conflicts, introduces errors, or reduces robustness.
- End each EXPERIENCE with one tag: [helpful] / [harmful] / [neutral] / [insufficient].

SELECTION HINTS
- Favor items about pruning rules, invariant-preserving moves, preconditions (e.g., invertibility), propagation justifications, case-split discipline, and auditability.
- REVIEW items should name error class if possible: algebra_error / logic_gap / missing_case / theorem_misuse / unsafe_transformation.

REQUIRED OUTPUT FORMAT
Return ONE JSON object ONLY where each entry is "KEY": "EXPERIENCE".

KEY (single line, natural English, no problem-specific tokens):
- For OPINION:
  "OPINION — <ResponderRole> handles <Info/Constraint/PeerFeedback>: <concise, abstract content>"
- For REVIEW:
  "REVIEW — <ReviewerRole> evaluates <Step/Claim>: <concise, abstract content>"

EXPERIENCE (value):
- One or two sentences, fully abstract (no copied symbols/values), then the tag.

INPUTS
PROMPT ROLE:
<<<
{prompt_role}
>>>

INPUT PROMPT:
<<<
{input_prompt}
>>>

INPUT RESPONSE:
<<<
{input_response}
>>>

GOLDEN_ANSWER (may be empty):
<<<
{golden}
>>>

EVAL_ANALYSIS (may be empty):
<<<
{eval_analysis}
>>>

FINAL_SCORE (may be empty or a number):
<<<
{final_score}
>>>

OUTPUT EXAMPLE (purely illustrative; keep abstract, no symbols/IDs/numbers)
{{
  "OPINION — Algebra (member) handles peer feedback about possible differentials: screens out any move whose destination is known to be trivial": "Good practice: Pruning transformations that must land in zero prevents wasted reasoning and keeps focus on impactful branches. [helpful]",
  "REVIEW — Reviewer (member) evaluates a claim of nonzero differential: the argument ignores that the target object vanishes": "Pitfall: Asserting a nonzero map into a trivial target is a logic gap; verify target viability before claiming effects. [harmful]",
  "OPINION — Strategy Planner (leader) handles auditability constraint: fixes an order of invariants to track before performing algebra": "Good practice: Establishing a stable evaluation order reduces backtracking and aligns each step with the final invariant. [helpful]",
  "REVIEW — Algebra (member) evaluates a cancellation step: cancellation is performed without confirming invertibility": "Pitfall: Canceling without a validity check is an unsafe transformation and can create spurious equivalences. [harmful]",
  "OPINION — Geometry (member) handles edge-case reminders: inserts boundary and degenerate cases before drawing a global conclusion": "Good practice: Early coverage of edge cases prevents overgeneralization and stabilizes conclusions. [helpful]"
}}
"""



# --------------- 抽取器 -----------------

def _safe_float(x: Any, default: float = float("-inf")) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _parse_json_maybe(text: str) -> Dict[str, str]:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        pass
    return {}

class ScoreToExperience:
    def __init__(self, agent: Agent):
        self.agent = agent

    @staticmethod
    def select_top_k_steps(steps: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        filtered = []
        for s in steps:
            if not isinstance(s, dict):
                continue
            if "prompt" not in s or "response" not in s or "final_score" not in s:
                continue
            s["_score_float"] = _safe_float(s.get("final_score"))
            filtered.append(s)
        filtered.sort(key=lambda x: x["_score_float"], reverse=True)
        return filtered[:max(0, top_k)]

    @staticmethod
    def build_prompt(role, input_prompt: str, input_response: str, golden, eval_analysis, final_score) -> str:
        return EXTRACTION_PROMPT_TEMPLATE.format(
            prompt_role=(role or "").strip(),
            input_prompt=(input_prompt or "").strip(),
            input_response=(input_response or "").strip(),
            golden=(golden or "").strip(),
            eval_analysis=(eval_analysis or "").strip(),
            final_score=final_score
        )

    def step_to_experiences(self, step: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, str]:
        prompt = self.build_prompt(
            step.get("role", "") or "",
            step.get("prompt", "") or "",
            step.get("response", "") or "",
            result.get("golden_answer", ""),
            step.get("analysis", ""),        # 来自 _eval_one_response 的 analysis
            step.get("final_score", 0)
        )
        raw = self.agent.temp_responses(prompt)
        return _parse_json_maybe(raw)

    def convert(self, score_obj: Dict[str, Any], result_obj: Dict[str, Any],
                top_k: int = 25, dedup_keys: bool = True) -> Dict[str, Any]:
        steps = score_obj.get("steps", []) or []
        top_steps = self.select_top_k_steps(steps, top_k)

        aggregated = {}
        per_step = []
        seen = set()

        for s in top_steps:
            experiences = self.step_to_experiences(s, result_obj)
            per_step.append({
                "turn": s.get("turn"),
                "step": s.get("step"),
                "role": s.get("role"),
                "phase": s.get("phase"),
                "final_score": s.get("final_score"),
                "experiences": experiences
            })
            for k, v in experiences.items():
                key = (k or "").strip()
                if not key: continue
                if dedup_keys and key in seen: continue
                aggregated[key] = (v or "").strip()
                seen.add(key)

        return {"aggregated_experiences": aggregated, "stepwise_outputs": per_step}

# --------------- CLI：批量从 score → experiences.jsonl -----------------

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _sanitize_filename(s: str) -> str:
    s = re.sub(r"\([^)]*\)", "", s)
    s = s.strip()
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "", s)
    return s or "UnknownRole"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="包含 problem_*_score.json 的目录")
    parser.add_argument("--results_dir", default=None, help="可选：problem_*.json 所在目录（若与 data_dir 不同）")
    parser.add_argument("--out_dir", required=True, help="输出每角色 *.experiences.jsonl")
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--no_dedup", action="store_true")
    parser.add_argument("--unified_experience", action="store_true", help="将所有经验写入一个统一文件 Unified.experiences.jsonl")
    args = parser.parse_args()

    _ensure_dir(args.out_dir)

    score_files = sorted(glob.glob(os.path.join(args.data_dir, "problem_*_score.json")))
    if not score_files:
        raise FileNotFoundError(f"No score files in {args.data_dir}")

    agent = Agent("Convert math interaction to retrieval experiences", "experience extractor", model_info="gpt-5", unified_experience=getattr(args, "unified_experience", False))

    for sp in tqdm(score_files):
        prefix = os.path.basename(sp).replace("_score.json", "")
        base_dir = args.results_dir if args.results_dir else args.data_dir
        rp = os.path.join(base_dir, f"{prefix}.json")
        if not os.path.exists(rp):
            print(f"[WARN] Missing result for {sp}: {rp}")
            continue

        score_obj = json.load(open(sp, "r", encoding="utf-8-sig"))
        result_obj = json.load(open(rp, "r", encoding="utf-8-sig"))

        conv = ScoreToExperience(agent).convert(
            score_obj, result_obj, top_k=args.top_k, dedup_keys=not args.no_dedup
        )

        stepwise = conv.get("stepwise_outputs", []) or []
        agg = conv.get("aggregated_experiences", {}) or {}

        # 统一写入模式：所有经验写到一个文件
        if getattr(args, "unified_experience", False):
            outp = os.path.join(args.out_dir, "Unified.experiences.jsonl")
            with open(outp, "a", encoding="utf-8-sig") as fw:
                # 优先逐步条目中的 experiences；若为空则落 agg
                wrote = 0
                for r in stepwise:
                    exps = r.get("experiences", {}) or {}
                    for k, v in exps.items():
                        fw.write(json.dumps({
                            "turn": r.get("turn"),
                            "step": r.get("step"),
                            "role": r.get("role"),
                            "key": k,
                            "experience": v
                        }, ensure_ascii=False) + "\n")
                        wrote += 1
                if wrote == 0 and agg:
                    for k, v in agg.items():
                        fw.write(json.dumps({"key": k, "experience": v}, ensure_ascii=False) + "\n")
            print(f"[OK] {sp} -> unified file append")
            continue

        # 按 role 写入
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for row in stepwise:
            role = _sanitize_filename(row.get("role") or "UnknownRole")
            buckets.setdefault(role, []).append(row)

        for role, rows in buckets.items():
            outp = os.path.join(args.out_dir, f"{role}.experiences.jsonl")
            with open(outp, "a", encoding="utf-8-sig") as fw:
                for r in rows:
                    exps = r.get("experiences", {}) or {}
                    for k, v in exps.items():
                        fw.write(json.dumps({
                            "turn": r.get("turn"),
                            "step": r.get("step"),
                            "role": role,
                            "key": k,
                            "experience": v
                        }, ensure_ascii=False) + "\n")

        print(f"[OK] {sp} -> {len(stepwise)} step-entries")

if __name__ == "__main__":
    main()
