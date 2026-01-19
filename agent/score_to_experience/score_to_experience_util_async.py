#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_to_experience.py  (parallel & resumable)

- Parallelizes per-prefix processing with ThreadPoolExecutor (safe via per-file locks)
- Skips already-processed prefixes using out_dir/.done/<prefix>.done markers
- Preserves the exact LLM prompt template (DO NOT CHANGE)
- Minimal logging; tqdm progress
"""

import argparse
import json
import os
import glob
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple, Optional, Union

from tqdm import tqdm

# ---------- LLM Prompt Template (unchanged) ----------
EXTRACTION_PROMPT_TEMPLATE = """You are an expert medical reasoning analyzer.

TASK
From a {{ROLE, PROMPT, RESPONSE, GROUND_TRUTH, EVAL_ANALYSIS, FINAL_SCORE}} pair, produce retrieval-ready key:value experiences in TWO classes:
1) OPINION — how an expert handles received information/opinion.
2) REVIEW — how an expert peer-reviews a specific claim or set of related information.

INPUTS
- PROMPT_ROLE: prompt_role
- INPUT_PROMPT: source text containing instructions, constraints, agreements, disagreements, claims.
- INPUT_RESPONSE: produced reasoning/diagnosis.
- GROUND_TRUTH (optional)
- EVAL_ANALYSIS (optional)
- FINAL_SCORE (optional numeric)

STRICT CONSTRAINTS
- Use ONLY the provided INPUT_PROMPT, INPUT_RESPONSE, GROUND_TRUTH, EVAL_ANALYSIS, FINAL_SCORE.
- Do NOT introduce external facts, disease names, tests, or treatments.
- Experiences must be generalizable, instructional reasoning guidance (how to prioritize, exclude, scope, weigh evidence, handle peer input).
- Skip sentences without actionable content.

OUTPUT
Return ONE JSON object ONLY where each entry is: "KEY": "EXPERIENCE".

KEY CONSTRUCTION (single line, natural English, no brackets):
- For OPINION (expert receives information/opinion and acts on it):
  "OPINION — <ResponderRole> handles <Info/Opinion from <SourceRole>>: <concise content>"
  * <ResponderRole> is typically RESPONSE_ROLE.
  * <SourceRole> is typically PROMPT_ROLE (or a named section like Peer_feedback, Constraint).
  * <Info/Opinion> is a short, standalone summary of the received point.

- For REVIEW (expert peer-reviews a specific claim or related info set):
  "REVIEW — <ReviewerRole> evaluates <Claim/Info from <TargetRole>>: <concise content>"
  * <ReviewerRole> is the role performing the review (often RESPONSE_ROLE).
  * <TargetRole> is the role that made the claim/info (often PROMPT_ROLE).
  * <Claim/Info> is a short, standalone summary of what is being reviewed.

EXPERIENCE (value) — 1–2 sentences, must:
- Be generalizable, instructional reasoning guidance (how to prioritize, exclude, scope, weigh evidence, integrate peer input).
- Include a clear judgment prefix:
  * "Good practice: ..." when behavior aligns with ground truth/evidence signals.
  * "Pitfall: ..." when behavior conflicts with ground truth/evidence or adds noise.
- Optionally append a simple outcome tag at the end: [helpful] / [harmful] / [neutral] / [insufficient].
- Justify judgment ONLY by contrasts observable between INPUT_RESPONSE and GROUND_TRUTH / EVAL_ANALYSIS / FINAL_SCORE. If these are absent or inconclusive, use "Pitfall/Good practice" only if you can justify from INPUT_RESPONSE behavior; otherwise end with [insufficient].

SELECTION GUIDELINES
- Extract OPINION items when RESPONSE_ROLE incorporates or reacts to a specific piece of information/opinion from PROMPT_ROLE (or section).
- Extract REVIEW items when RESPONSE_ROLE scrutinizes, supports, or rejects a specific claim or related info set from PROMPT_ROLE (or section).
- Prefer items that materially affect prioritization, exclusion, scoping, or evidence weighting.
- Aim for precision over quantity (3–10 items typical).

PROMPT ROLE:
<<<
{prompt_role}
>>>

INPUT_PROMPT:
<<<
{input_prompt}
>>>

INPUT_RESPONSE:
<<<
{input_response}
>>>

GROUND_TRUTH (may be empty):
<<<
{ground_truth}
>>>

EVAL_ANALYSIS (may be empty):
<<<
{eval_analysis}
>>>

FINAL_SCORE (may be empty or a number):
<<<
{final_score}
>>>

OUTPUT EXAMPLE
{{
  "OPINION — Neurology_expert handles peer feedback from Peer_feedback_review: narrows diagnosis to multiple subtypes without distinctive anchors": "Pitfall: Enumerating subtypes without distinguishing features dilutes clarity and creates false precision, which harmed alignment with the reference. [harmful]",
  "REVIEW — Peer_feedback_review evaluates constraint from HARD_CONSTRAINTS: insists on exactly ten numbered diagnoses": "Good practice: Enforcing strict format and scope improves comparability and prevents drift across disciplines. [helpful]",
  "OPINION — Neurology_expert handles peer feedback from Pediatrics_leader: integrates metabolic and neurological features as a unified signal": "Good practice: Combining convergent evidence streams strengthens prioritization and increases consistency. [helpful]",
  "REVIEW — Neurology_expert evaluates peer claim of excluding a candidate due to missing signals": "Pitfall: Over-reliance on absence of one signal caused premature exclusion despite partial alignment; this reduced sensitivity. [harmful]"
}}
"""

# ---------- Utilities ----------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _safe_float(x: Any, default: float = float("-inf")) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _parse_llm_json(text: str) -> Dict[str, str]:
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
    except Exception:
        pass
    return {}

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def _sanitize_filename(s: str) -> str:
    s = re.sub(r"\([^)]*\)", "", s)
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "", s)
    return s or "UnknownRole"

def _prefix_from(path: str, suffix: str) -> str:
    base = os.path.basename(path)
    if base.endswith(suffix):
        return base[:-len(suffix)]
    return os.path.splitext(base)[0]

# ---------- Agent Loader ----------
def build_default_agent() -> Any:
    """
    Build the default Agent as requested by the user.
    Uses the same signature you outlined.
    """
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils import Agent  # noqa: E402
        return Agent("Convert the communication history to experiences", "Convert communication to experience", examplers=None, model_info="gpt-5")
    except Exception:
        try:
            Agent  # type: ignore[name-defined]
        except NameError as e:
            raise RuntimeError(
                "Agent class not found. Ensure Agent is importable or supply --agent_module."
            ) from e
        return Agent("Convert the communication history to experiences", "Convert communication to experience", examplers=None, model_info="gpt-5")  # type: ignore[name-defined]

def build_agent_from_module(agent_module: str) -> Any:
    pkg = __import__(agent_module, fromlist=["Agent"])
    Agent = getattr(pkg, "Agent")
    # Keep params identical to your original usage
    return Agent(None, "Convert communication to experience", examplers=None, model_info="gpt-5")

# ---------- Core Converter (unchanged prompt usage) ----------
class ScoreToExperience:
    def __init__(self, agent: Any):
        self.agent = agent

    @staticmethod
    def select_top_k_steps(steps: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        filtered = []
        for s in steps:
            if not isinstance(s, dict):
                continue
            if "prompt" not in s or "response" not in s or "final_score" not in s:
                continue
            score = _safe_float(s.get("final_score"))
            s["_score_float"] = score
            filtered.append(s)
        filtered.sort(key=lambda x: x["_score_float"], reverse=True)
        return filtered[:max(0, top_k)]

    @staticmethod
    def build_prompt(role, input_prompt: str, input_response: str, gt, eval_analysis, final_score) -> str:
        # DO NOT change the template text
        return EXTRACTION_PROMPT_TEMPLATE.format(
            prompt_role=role.strip(),
            input_prompt=input_prompt.strip(),
            input_response=input_response.strip(),
            ground_truth=(gt or "").strip(),
            eval_analysis=(eval_analysis or "").strip(),
            final_score=final_score
        )

    def step_to_experiences(self, step: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, str]:
        prompt = self.build_prompt(
            step.get("role", "") or "",
            step["prompt"],
            step["response"],
            result.get("ground_truth", ""),
            step.get("eval_analysis", ""),
            step.get("final_score", "")
        )
        # NOTE: do NOT modify the prompt itself.
        # Call style stays as in your snippet (temp_responses), to avoid changing your Agent API.
        raw = self.agent.temp_responses(prompt)  # Expecting pure JSON string
        return _parse_llm_json(raw)

    def convert(self, data: Dict[str, Any], result: Dict[str, Any], top_k: int = 5, dedup_keys: bool = True
               ) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        steps = data.get("steps", [])
        top_steps = self.select_top_k_steps(steps, top_k)

        aggregated: Dict[str, str] = {}
        per_step: List[Dict[str, Any]] = []
        seen_keys = set()

        for s in top_steps:
            experiences = self.step_to_experiences(s, result)
            per_step.append({
                "turn": s.get("turn"),
                "step": s.get("step"),
                "name": s.get("name"),
                "role": s.get("role"),
                "phase": s.get("phase"),
                "final_score": s.get("final_score"),
                "experiences": experiences
            })
            for k, v in experiences.items():
                key_norm = (k or "").strip()
                if not key_norm:
                    continue
                if dedup_keys and key_norm in seen_keys:
                    continue
                aggregated[key_norm] = (v or "").strip()
                seen_keys.add(key_norm)

        return aggregated, per_step

def score_to_experience(score_obj: Dict[str, Any], gt_obj: Dict[str, Any], agent: Any,
                        top_k: int = 5, dedup_keys: bool = True) -> Dict[str, Any]:
    converter = ScoreToExperience(agent)
    aggregated, stepwise = converter.convert(score_obj, gt_obj, top_k=top_k, dedup_keys=dedup_keys)
    return {"aggregated_experiences": aggregated, "stepwise_outputs": stepwise}

# ---------- Parallel Orchestration ----------
_file_locks: "defaultdict[str, threading.Lock]" = defaultdict(threading.Lock)

def _append_jsonl(role_file: str, line_obj: Dict[str, Any]) -> None:
    lock = _file_locks[role_file]
    with lock:
        with open(role_file, "a", encoding="utf-8") as fw:
            fw.write(json.dumps(line_obj, ensure_ascii=False) + "\n")

def _done_marker_path(out_dir: str, prefix: str) -> str:
    done_dir = os.path.join(out_dir, ".done")
    _ensure_dir(done_dir)
    return os.path.join(done_dir, f"{prefix}.done")

def _is_done(out_dir: str, prefix: str) -> bool:
    return os.path.exists(_done_marker_path(out_dir, prefix))

def _mark_done(out_dir: str, prefix: str) -> None:
    path = _done_marker_path(out_dir, prefix)
    with open(path, "w", encoding="utf-8") as f:
        f.write("OK\n")

def _process_one_prefix(prefix: str,
                        score_path: str,
                        gt_path: str,
                        out_dir: str,
                        top_k: int,
                        dedup: bool,
                        agent_module: Optional[str],
                        quiet: bool) -> Tuple[str, bool, Optional[str]]:
    """
    Returns: (prefix, success, error_message_if_any)
    """
    try:
        # Build a fresh Agent per worker for thread-safety
        agent = build_agent_from_module(agent_module) if agent_module else build_default_agent()

        score_obj = _load_json(score_path)
        gt_obj = _load_json(gt_path)

        result = score_to_experience(
            score_obj=score_obj,
            gt_obj=gt_obj,
            agent=agent,
            top_k=top_k,
            dedup_keys=dedup
        )

        stepwise = result.get("stepwise_outputs", []) or []
        if not stepwise:
            # fall back to aggregated
            agg = result.get("aggregated_experiences", {}) or {}
            if agg:
                role_file = os.path.join(out_dir, _sanitize_filename("UnknownRole") + ".experiences.jsonl")
                for k, v in agg.items():
                    _append_jsonl(role_file, {
                        "source_prefix": prefix,
                        "role": "UnknownRole",
                        "key": k,
                        "experience": v
                    })
            _mark_done(out_dir, prefix)
            return (prefix, True, None)

        for step in stepwise:
            role = step.get("role") or "UnknownRole"
            role_file = os.path.join(out_dir, _sanitize_filename(role) + ".experiences.jsonl")
            exp_map = step.get("experiences", {}) or {}
            if isinstance(exp_map, dict) and exp_map:
                for k, v in exp_map.items():
                    _append_jsonl(role_file, {
                        "source_prefix": prefix,
                        "turn": step.get("turn"),
                        "step": step.get("step"),
                        "role": role,
                        "key": k,
                        "experience": v
                    })

        _mark_done(out_dir, prefix)
        return (prefix, True, None)
    except Exception as e:
        return (prefix, False, str(e))

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Folder containing *score.json and *A.json")
    parser.add_argument("--out_dir", required=True, help="Folder to append per-expert experiences as JSONL")
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--no_dedup", action="store_true")
    parser.add_argument("--agent_module", default=None)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--force", action="store_true", help="Reprocess even if .done marker exists")
    parser.add_argument("--quiet", action="store_true", help="Reduce logs; keep tqdm only")
    args = parser.parse_args()

    _ensure_dir(args.out_dir)

    score_files = sorted(
        glob.glob(os.path.join(args.data_dir, "*.score.json"))
        + glob.glob(os.path.join(args.data_dir, "*_score.json"))
    )
    gt_files = sorted(glob.glob(os.path.join(args.data_dir, "*.A.json")))

    if not score_files:
        raise FileNotFoundError(f"No *(_).score.json found in {args.data_dir}")
    if not gt_files:
        raise FileNotFoundError(f"No *.A.json found in {args.data_dir}")

    def _score_prefix(score_path: str) -> str:
        """根据文件名得到统一的 prefix（例如 patient_36）"""
        base = os.path.basename(score_path)
        if base.endswith(".A_score.json"):
            # 适配 patient_36.A_score.json  -> prefix = patient_36
            return base[:-len(".A_score.json")]
        if base.endswith(".score.json"):
            # 适配旧的 xxx.score.json -> 去掉 .score.json
            return base[:-len(".score.json")]
        if base.endswith("_score.json"):
            # 兜底： xxx_score.json -> 去掉 _score.json
            return base[:-len("_score.json")]
        # 再兜底：去扩展名
        return os.path.splitext(base)[0]

    def _gt_prefix(gt_path: str) -> str:
        """GT 格式是 patient_36.A.json -> prefix = patient_36"""
        base = os.path.basename(gt_path)
        if base.endswith(".A.json"):
            return base[:-len(".A.json")]
        return os.path.splitext(base)[0]

    # 前缀 -> groundtruth 路径
    gt_map = { _gt_prefix(p): p for p in gt_files }

    # 构造待处理任务（支持断点跳过）
    tasks = []
    for score_path in score_files:
        prefix = _score_prefix(score_path)
        if prefix not in gt_map:
            if not args.quiet:
                print(f"[WARN] GT not found for {os.path.basename(score_path)} (need {prefix}.A.json)")
            continue
        if not args.force and _is_done(args.out_dir, prefix):
            continue
        tasks.append((prefix, score_path, gt_map[prefix]))

    if not tasks:
        if not args.quiet:
            print("[INFO] Nothing to do (all done or no pairs).")
        return

    if not args.quiet:
        print(f"[INFO] Scheduling {len(tasks)} prefixes with {args.workers} workers...")

    successes = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                _process_one_prefix,
                prefix,
                score_path,
                gt_path,
                args.out_dir,
                args.top_k,
                not args.no_dedup,
                args.agent_module,
                args.quiet
            )
            for (prefix, score_path, gt_path) in tasks
        ]

        for fut in tqdm(as_completed(futs), total=len(futs), desc="Processing", unit="file"):
            prefix, ok, err = fut.result()
            if ok:
                successes += 1
            else:
                errors += 1
                if not args.quiet:
                    print(f"[ERROR] {prefix}: {err}")

    if not args.quiet:
        print(f"[DONE] Success: {successes}, Errors: {errors}, Output: {args.out_dir}")

if __name__ == "__main__":
    main()
