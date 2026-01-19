#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_to_experience.py

Convert high-scoring MDT steps into generalizable, hallucination-safe clinician experiences.

Assumptions:
- You have an Agent class available in your environment:
    Agent(system_prompt: Optional[str], name: str, examplers: Optional[Any] = None, model_info: str = "gpt-5")
    Agent.chat(prompt: str) -> str
- The input JSON has the schema:
{
  "steps": [
    {
      "turn": "turn_0",
      "step": 1,
      "name": "Pediatrics (leader)",
      "role": "Pediatrics (leader)",
      "phase": "opinion",
      "prompt": "...",        # REQUIRED
      "response": "...",      # REQUIRED
      "eval_analysis": "...", # OPTIONAL
      "final_score": 4.7      # REQUIRED (number)
    },
    ...
  ]
}

Usage:
    python score_to_experience.py --input steps.json --output experiences.json --top_k 5
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple, Optional, Union


# ---------- LLM Prompt Template (focused, generalizable, hallucination-safe) ----------

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
  "REVIEW — Neurology_expert evaluates peer claim of excluding a candidate due to missing signals": "Pitfall: Over-reliance on absence of one signal caused premature exclusion despite partial alignment; this reduced sensitivity. [harmful]"=
}}
"""


# ---------- Utility Functions ----------

def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _safe_float(x: Any, default: float = float("-inf")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _parse_llm_json(text: str) -> Dict[str, str]:
    """
    Try to parse LLM output as a JSON object of {str: str}.
    If parsing fails, return an empty dict to be safe.
    """
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            # filter to str->str only
            return {str(k): str(v) for k, v in parsed.items()}
    except Exception:
        pass
    return {}


# ---------- Core Converter ----------

class ScoreToExperience:
    """
    Convert top-K highest scoring MDT steps into key:value experiences via an LLM Agent.
    """

    def __init__(self, agent: Any):
        """
        Parameters
        ----------
        agent : Any
            An Agent instance with a .chat(prompt: str) -> str method.
            Example: Agent(None, "Convert communication to experience", examplers=None, model_info='gpt-5')
        """
        self.agent = agent

    @staticmethod
    def select_top_k_steps(steps: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """
        Select the top_k steps sorted by final_score (descending).
        Only include steps that have required fields.
        """
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
        """
        Construct the LLM prompt for a single step using the template.
        """
        return EXTRACTION_PROMPT_TEMPLATE.format(
            prompt_role=role.strip(),
            input_prompt=input_prompt.strip(),
            input_response=input_response.strip(),
            ground_truth=gt.strip(),
            eval_analysis=eval_analysis.strip(),
            final_score=final_score
        )

    def step_to_experiences(self, step: Dict[str, Any], result) -> Dict[str, str]:
        """
        Convert a single step into experiences via the LLM.
        Returns a dict mapping prompt-sentences -> experience sentences.
        """
        prompt = self.build_prompt(step["role"], step["prompt"], step["response"], result.get("ground_truth", ""), step.get("eval_analysis", ""), step.get("final_score", ""))
        print(f"[DEBUG] LLM Prompt:\n{prompt}\n---")
        raw = self.agent.temp_responses(prompt)  # Expecting pure JSON string
        parsed = _parse_llm_json(raw)
        return parsed

    def convert(
        self,
        data: Dict[str, Any],
        result,
        top_k: int = 5,
        dedup_keys: bool = True
    ) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
        """
        Convert the highest scoring steps into experiences.

        Returns
        -------
        aggregated_experiences : Dict[str, str]
            Merged key:value experiences across selected steps.
        stepwise_outputs : List[Dict[str, Any]]
            Per-step outputs: metadata + the raw JSON experiences from the LLM.
        """
        steps = data.get("steps", [])
        top_steps = self.select_top_k_steps(steps, top_k)

        aggregated: Dict[str, str] = {}
        per_step: List[Dict[str, Any]] = []

        seen_keys = set()

        for s in top_steps:
            experiences = self.step_to_experiences(s, result)

            # Keep also the metadata for traceability
            per_step.append({
                "turn": s.get("turn"),
                "step": s.get("step"),
                "name": s.get("name"),
                "role": s.get("role"),
                "phase": s.get("phase"),
                "final_score": s.get("final_score"),
                "experiences": experiences
            })

            # Merge into aggregated (optionally deduplicate keys)
            for k, v in experiences.items():
                key_norm = k.strip()
                if not key_norm:
                    continue
                if dedup_keys and key_norm in seen_keys:
                    # Skip duplicate keys (first come, first served)
                    continue
                aggregated[key_norm] = v.strip()
                seen_keys.add(key_norm)

        return aggregated, per_step


# ---------- Functional Wrapper ----------

def score_to_experience(
    score_obj: Dict[str, Any],
    gt_obj,
    agent: Any,
    top_k: int = 5,
    dedup_keys: bool = True
) -> Dict[str, Any]:
    """
    Functional interface: from in-memory score object -> experiences result.

    Returns
    -------
    result : Dict[str, Any]
        {
          "aggregated_experiences": { ... },
          "stepwise_outputs": [ ... ]
        }
    """
    converter = ScoreToExperience(agent)
    aggregated, stepwise = converter.convert(score_obj, gt_obj, top_k=top_k, dedup_keys=dedup_keys)
    return {
        "aggregated_experiences": aggregated,
        "stepwise_outputs": stepwise
    }


# ---------- CLI / Main ----------

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _save_json(obj: Union[Dict[str, Any], List[Any]], path: str) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def build_default_agent() -> Any:
    """
    Build the default Agent as requested by the user.
    Change this if your Agent constructor differs.
    """
    # Lazy import in case user provides their own Agent in environment
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils import Agent
    except Exception:
        # Minimal shim for environments where Agent is already in global namespace
        try:
            Agent  # type: ignore # noqa
        except NameError as e:
            raise RuntimeError(
                "Agent class not found. Please ensure Agent is importable or modify build_default_agent()."
            ) from e

    return Agent("Convert the communication history to experiences", "Convert communication to experience", examplers=None, model_info="gpt-5")


import argparse
import json
import os
import glob
import re

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _load_json(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)

def _sanitize_filename(s: str) -> str:
    # 用于文件名：去除括号内容，空白->下划线，只保留常见安全字符
    s = re.sub(r"\([^)]*\)", "", s)        # 去括号及内部
    s = s.strip()
    s = re.sub(r"\s+", "_", s)             # 空白 -> _
    s = re.sub(r"[^A-Za-z0-9_\-\.]", "", s)
    return s or "UnknownRole"

def _prefix_from(path: str, suffix: str) -> str:
    # 去掉末尾指定的后缀，得到前缀
    if path.endswith(suffix):
        return os.path.basename(path)[:-len(suffix)]
    # 兜底：去最后一个扩展名
    return os.path.splitext(os.path.basename(path))[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Folder containing *.score.json and *.A.json")
    parser.add_argument("--out_dir", required=True, help="Folder to append per-expert experiences as JSONL")
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--no_dedup", action="store_true")
    parser.add_argument("--agent_module", default=None)
    # 其他可选权重参数，传递给下游 run/score_to_experience 如需要可自行扩展
    args = parser.parse_args()

    _ensure_dir(args.out_dir)

    # 收集文件
    score_files = sorted(glob.glob(os.path.join(args.data_dir, "*.score.json")))
    gt_files    = sorted(glob.glob(os.path.join(args.data_dir, "*.A.json")))
    if not score_files:
        raise FileNotFoundError(f"No *.score.json found in {args.data_dir}")
    if not gt_files:
        raise FileNotFoundError(f"No *.A.json found in {args.data_dir}")

    # 前缀 -> groundtruth 路径
    gt_map = { _prefix_from(p, ".A.json"): p for p in gt_files }

    # 构建 agent
    if args.agent_module:
        pkg = __import__(args.agent_module, fromlist=["Agent"])
        Agent = getattr(pkg, "Agent")
        agent = Agent(None, "Convert communication to experience", examplers=None, model_info="gpt-5")
    else:
        agent = build_default_agent()  # 你已有的函数

    processed = 0
    for score_path in score_files:
        prefix = _prefix_from(score_path, ".score.json")
        if prefix not in gt_map:
            print(f"[WARN] groundtruth not found for {os.path.basename(score_path)} (looking for {prefix}.A.json)")
            continue

        # 读取 score 与 ground truth
        score_obj = _load_json(score_path)
        gt_obj    = _load_json(gt_map[prefix])

        # 调用你已有的转换函数（注意签名：移除 result=）
        result = score_to_experience(
            score_obj=score_obj,
            gt_obj=gt_obj,
            agent=agent,
            top_k=args.top_k,
            dedup_keys=not args.no_dedup
        )
        # result 预期形如：
        # {
        #   "aggregated_experiences": { key -> experience_str, ... },
        #   "stepwise_outputs": [
        #       { "turn":..., "step":..., "name":..., "role":..., "experiences": { key->exp, ... }, ... },
        #       ...
        #   ]
        # }

        stepwise = result.get("stepwise_outputs", [])
        if not stepwise:
            # 若没有分步信息，则把 aggregated 按 UnknownRole 归档
            agg = result.get("aggregated_experiences", {})
            if agg:
                out_file = os.path.join(args.out_dir, _sanitize_filename("UnknownRole") + ".experiences.jsonl")
                with open(out_file, "a", encoding="utf-8") as fw:
                    for k, v in agg.items():
                        line = {
                            "source_prefix": prefix,
                            "role": "UnknownRole",
                            "key": k,
                            "experience": v
                        }
                        fw.write(json.dumps(line, ensure_ascii=False) + "\n")
            processed += 1
            continue

        # 将每个 step 的经验按 role 追加写入各自文件
        for step in stepwise:
            role = step.get("role") or "UnknownRole"
            role_file = os.path.join(args.out_dir, _sanitize_filename(role) + ".experiences.jsonl")
            exp_map = step.get("experiences", {}) or {}
            if not isinstance(exp_map, dict):
                continue
            with open(role_file, "a", encoding="utf-8") as fw:
                for k, v in exp_map.items():
                    # 写入一行 JSON，包含最小必要溯源
                    line = {
                        "source_prefix": prefix,
                        "turn": step.get("turn"),
                        "step": step.get("step"),
                        "role": role,
                        "key": k,
                        "experience": v
                    }
                    fw.write(json.dumps(line, ensure_ascii=False) + "\n")

        processed += 1
        print(f"[OK] {os.path.basename(score_path)} -> appended experiences per expert")

    print(f"[DONE] Processed {processed} score files; per-expert JSONL written to {args.out_dir}")


if __name__ == "__main__":
    main()
