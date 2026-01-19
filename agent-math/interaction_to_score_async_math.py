#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
interaction_to_score_async_math.py
将一次数学 RARE/MDT 交互日志 (problem_x_interaction.json) + 对应结果 (problem_x.json)
打分为:
- global: 最终答案是否正确（可选: LLM 严格判断），得到一个全局分
- steps: 每条 agent 输出（opinion/peer_review）对最终质量的“影响分”(0..5)
并把 global 分按时间衰减分摊到各轮，再线性合成 final_score
输出：<prefix>._score.json
"""

import json, argparse, re, os, glob
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
from tqdm.auto import tqdm

# 与数学 utils 兼容的 Agent
try:
    from utils_with_experience_math import Agent  # 如果你文件叫 utils_with_experience.py，请改这个 import
except Exception:
    from utils_with_experience import Agent  # 兜底

_TURN_PAT = re.compile(r"(?:^|_)(\d+)$")

def _turn_index(k: str) -> int:
    s = str(k); m = _TURN_PAT.search(s)
    if m:
        try: return int(m.group(1))
        except Exception: return 0
    try: return int(s)
    except Exception: return 0

def _unwrap_interaction_root(interaction: Dict[str, Any]) -> Dict[str, Any]:
    """取到包含 turn_0/turn_1/... 的那层字典。"""
    if any(str(k).startswith("turn_") for k in interaction.keys()):
        return interaction
    for _, v in interaction.items():
        if isinstance(v, dict) and any(str(k).startswith("turn_") for k in v.keys()):
            return v
    return interaction

def _agents_from_step(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    统一抽取一个 step 中的“一个 agent 的输入/输出”。
    数学流程里 step 可能是 opinion 或 peer_review：
      - opinion: 有 prompt, response(raw), solution_outline(提取块)
      - peer_review: 有 prompt, response(JSON)
    """
    role = step.get("role") or step.get("name") or "Unknown"
    phase = step.get("phase") or ""
    prompt = (step.get("prompt") or "").strip()
    # 优先展示 solution_outline；其次 response
    out = step.get("solution_outline") or step.get("response") or ""
    return [{
        "name": role,
        "role": role,
        "phase": phase,
        "prompt": prompt,
        "response": (out or "").strip()
    }]

def _format_agents_block_from_step(step: Dict[str, Any]) -> str:
    agents = _agents_from_step(step)
    lines = []
    for a in agents:
        head = f'- {a["name"]}'
        body = a["response"] or a["prompt"]
        lines.append(f"{head}:\n{(body or '').strip()}\n")
    return ("\n".join(lines)).strip() or "(empty)"

# ---------------- Global 评估：最终答案是否匹配 golden ----------------

_GLOBAL_SYSTEM = "You are a mathematician who judges equivalence of final answers."

def _math_answer_evaluate(pred: str, golden: str, *, agent: Optional[Agent] = None):
    """
    使用 LLM 做稳健比对（数值、代数等价），输出 Yes/No。
    若你不想用 LLM，也可直接做字符串/浮点近似比对。
    """
    pred = (pred or "").strip()
    golden = (golden or "").strip()
    if pred == "" or golden == "":
        return "No", "<answer>No</answer>"

    _agent = agent or Agent(_GLOBAL_SYSTEM, "math answer judger", model_info="gpt-5", use_experience=False)
    # 严格输出 Yes/No
    user_prompt = (
        "Decide if two final math answers are equivalent.\n"
        "Accept algebraic equivalence (simplify), numeric equality (within 1e-9), "
        "and conventional forms (e.g., 1/2 vs 0.5). Reject if they are materially different.\n\n"
        "Output ONLY: <answer>Yes</answer> or <answer>No</answer>.\n\n"
        f"Pred: {pred}\nGolden: {golden}\n"
    )
    raw = _agent.temp_responses(user_prompt)
    m = re.search(r"<answer>(Yes|No)</answer>", raw, re.I)
    ans = (m.group(1).title() if m else "No")
    return ans, raw

def _evaluate_global_with_llm(final_answer: str, golden: str,
                              hit_score: float = 5.0, miss_penalty: float = -5.0) -> Dict[str, Any]:
    ok, raw = _math_answer_evaluate(final_answer, golden)
    score = hit_score if ok == "Yes" else miss_penalty
    return {"match": (ok == "Yes"), "score": float(score), "raw": raw}

# ---------------- Step 影响度评估（与医学版一致的协议） ----------------

_STEP_SYSTEM_PROMPT = (
    "You are a strict adjudicator for math response influence. "
    "Judge how the TARGET RESPONSE influenced final correctness, relative to STEP 0 baseline. "
    "Be specific about mathematical content (identities, invariants, bounds), not step IDs. Output strict JSON."
)

_STEP_USER_TEMPLATE = """Evaluate one TARGET RESPONSE within full context.

[STEP 0 — Baseline]
{step0_block}

[STEP {i} — Target container]
{stepi_block}

[TARGET RESPONSE to score ONLY]
Agent: {agent_name} {role_desc}
Phase: {phase_desc}
Prompt:
{agent_prompt}

Response:
{agent_response}

Consider ONLY:
- Final Answer (chair): {final_answer}
- Golden Answer: {golden}

Guidance:
- Reward reasoning that correctly derives key lemmas, applies valid transformations, or fixes critical issues.
- Penalize algebra/logic errors, missing cases, misuse of theorems, or misleading reviews.
- If response is empty/boilerplate, score low.

Output STRICT JSON:
{{
  "analysis": "<5-10 short sentences on THIS response's impact; cite concrete math content (not step IDs)>",
  "score": 0..5
}}
"""

def _parse_step_json(raw: str) -> Tuple[str, int]:
    try:
        obj = json.loads(raw)
        analysis = (obj.get("analysis") or "").strip() or "empty analysis"
        sc = int(obj.get("score", 0))
        if sc < -5 or sc > 5: sc = 0
        return analysis, sc
    except Exception:
        return "parse error", 0

def _eval_one_response(step0_block: str, stepi_block: str, final_answer: str, golden: str, i: int,
                       target: Dict[str, Any]) -> Dict[str, Any]:
    role_desc = f"({target['role']})" if target.get("role") and target["role"] != target.get("name") else ""
    phase_desc = (target.get("phase") or "unknown")
    user_prompt = _STEP_USER_TEMPLATE.format(
        step0_block=step0_block, stepi_block=stepi_block,
        final_answer=final_answer, golden=golden, i=i,
        agent_name=target.get("name") or "Unknown",
        role_desc=role_desc, phase_desc=phase_desc,
        agent_prompt=target.get("prompt") or "",
        agent_response=target.get("response") or ""
    )
    agent = Agent("Return strict JSON with 'analysis' and integer 'score'.", "math step influence judger", model_info="gpt-5", use_experience=False)
    raw = agent.temp_responses(user_prompt)
    analysis, score = _parse_step_json(raw)
    return {"analysis": analysis, "score": score, "eval_user_prompt": user_prompt, "eval_response": raw}

# ---------------- 分摊/合成 ----------------

def _allocate_global(global_score: float, grouped: Dict[str, List[Dict[str, Any]]], decay: float = 0.9) -> None:
    turns = [k for k, lst in grouped.items() if lst]
    if not turns: return
    turns = sorted(turns, key=_turn_index)
    T = len(turns)
    raw_weights = [decay ** ((T - 1) - idx) for idx in range(T)]
    S = sum(raw_weights) or 1.0
    norm_w = [w / S for w in raw_weights]
    for idx, tk in enumerate(turns):
        rows = grouped[tk]
        per_turn_total = global_score * norm_w[idx]
        weights = [max(0.0, float(r.get("step_score", 0))) for r in rows]
        wsum = sum(weights)
        if wsum > 0:
            for r, w in zip(rows, weights):
                r["global_per_turn_total"] = per_turn_total
                r["global_contrib"] = per_turn_total * (w / wsum)
        else:
            even = per_turn_total / len(rows) if rows else 0.0
            for r in rows:
                r["global_per_turn_total"] = per_turn_total
                r["global_contrib"] = even

def _combine_scores(grouped: Dict[str, List[Dict[str, Any]]], w_global: float, w_step: float) -> None:
    for rows in grouped.values():
        for r in rows:
            r["final_score"] = w_global * float(r.get("global_contrib", 0.0)) + w_step * float(r.get("step_score", 0.0))

# ---------------- 主流程：单文件 ----------------

def _collect_prompts_responses(turns: Dict[str, Any]) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    for turn_key, turn in turns.items():
        if not str(turn_key).startswith("turn_"): continue
        for idx, step in enumerate(turn.get("steps", []) or []):
            for a in _agents_from_step(step):
                recs.append({
                    "turn": turn_key, "step": idx, "name": a.get("name"),
                    "role": a.get("role"), "phase": a.get("phase"),
                    "prompt": a.get("prompt"), "response": a.get("response"),
                })
    return recs

def run_one(interaction: Dict[str, Any], result_dict: Dict[str, Any],
            w_global: float, w_step: float, decay: float,
            eval_workers: int = 8, verbose: bool = False) -> Dict[str, Any]:

    # 终局答案 + 标准答案，从 result(problem_x.json) 读取更稳妥
    final_answer = (result_dict.get("predict_answer") or "").strip()
    golden = (result_dict.get("golden_answer") or "").strip()
    if not final_answer and isinstance(interaction.get("final_decision"), dict):
        final_answer = interaction["final_decision"].get("final_answer", "").strip()

    # GLOBAL 评估
    g = _evaluate_global_with_llm(final_answer, golden)
    global_score, is_match = g["score"], g["match"]

    # 收集并发评估每条 step（与医学版一致）
    turns = _unwrap_interaction_root(interaction)
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for turn_key, turn in sorted(turns.items(), key=lambda kv: _turn_index(kv[0])):
        if not str(turn_key).startswith("turn_"): continue
        steps = turn.get("steps", []) or []
        grouped[turn_key] = []
        if not steps: continue

        def _first_non_system_step(steps_list):
            for s in steps_list:
                if (s.get("role", "").upper() != "SYSTEM"):
                    return s
            return steps_list[0] if steps_list else {}

        base = _first_non_system_step(steps)
        step0_block = _format_agents_block_from_step(base)

        # 目标条目数量
        total_targets = sum(len(_agents_from_step(steps[idx])) for idx in range(1, len(steps)))
        if total_targets == 0: continue

        with ThreadPoolExecutor(max_workers=max(1, eval_workers)) as ex, \
             tqdm(total=total_targets, disable=not verbose, desc=f"{turn_key} eval") as pbar:
            futures, metas = [], []
            for idx in range(1, len(steps)):
                step_i = steps[idx]
                stepi_block = _format_agents_block_from_step(step_i)
                for a in _agents_from_step(step_i):
                    if (a.get("role") or "").upper() == "SYSTEM": continue
                    futures.append(ex.submit(_eval_one_response, step0_block, stepi_block, final_answer, golden, idx, a))
                    metas.append((idx, a))
            for fut, (idx, a) in zip(futures, metas):
                ev = fut.result()
                row = OrderedDict()
                row["turn"] = turn_key; row["step"] = idx
                row["name"] = a.get("name"); row["role"] = a.get("role"); row["phase"] = a.get("phase")
                row["prompt"] = a.get("prompt"); row["response"] = a.get("response")
                row["analysis"] = ev["analysis"]; row["step_score"] = int(ev["score"])
                row["global_contrib"] = 0.0; row["final_score"] = 0.0
                row["eval_user_prompt"] = ev["eval_user_prompt"]; row["eval_response"] = ev["eval_response"]
                grouped[turn_key].append(row)
                pbar.update(1)

    _allocate_global(global_score, grouped, decay=decay)
    _combine_scores(grouped, w_global=w_global, w_step=w_step)

    steps_flat = [r for tk in sorted(grouped.keys(), key=_turn_index) for r in grouped[tk]]

    result = {
        "global": {"score": global_score, "match": is_match},
        "weights": {"w_global": w_global, "w_step": w_step, "decay": decay},
        "steps": steps_flat
    }
    return result

# ---------------- CLI：批量处理一个目录 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="必须包含 problem_*_interaction.json 与 problem_*.json")
    ap.add_argument("--w_global", type=float, default=0.6)
    ap.add_argument("--w_step", type=float, default=0.4)
    ap.add_argument("--decay", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-workers", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    # NEW: allow results stored in a different directory
    ap.add_argument("--results_dir", default=None, help="可选：problem_*.json 所在目录（若与 data_dir 不同）")
    args = ap.parse_args()

    # 匹配对：problem_X_interaction[可选后缀].json 与 problem_X[同后缀].json
    interact_files = sorted(glob.glob(os.path.join(args.data_dir, "problem_*_interaction*.json")))
    if not interact_files:
        raise FileNotFoundError(f"No interaction files in {args.data_dir}")

    def _prefix_from_interact(path: str) -> str:
        base = os.path.basename(path)
        # 去掉 .json
        if base.lower().endswith(".json"):
            name = base[:-5]
        else:
            name = base
        # 将首次出现的 "_interaction" 去掉，保留后续后缀（如 _useexperience/_free_recruited）
        name = name.replace("_interaction", "", 1)
        return name

    def _result_path_from_prefix(prefix: str) -> str:
        base_dir = args.results_dir if args.results_dir else args.data_dir
        return os.path.join(base_dir, f"{prefix}.json")

    jobs = []
    for f in interact_files:
        prefix = _prefix_from_interact(f)  # problem_12
        rpath = _result_path_from_prefix(prefix)
        if not os.path.exists(rpath):
            raise FileNotFoundError(f"Missing result: {rpath} for {f}")
        jobs.append((prefix, f, rpath))

    os.makedirs(args.data_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {}
        for prefix, ipath, rpath in jobs:
            def _run(ip=ipath, rp=rpath, px=prefix):
                interaction = json.load(open(ip, "r", encoding="utf-8-sig"))
                result_dict = json.load(open(rp, "r", encoding="utf-8-sig"))
                out = run_one(interaction, result_dict,
                              w_global=args.w_global, w_step=args.w_step,
                              decay=args.decay, eval_workers=args.eval_workers,
                              verbose=args.verbose)
                out_path = os.path.join(args.data_dir, f"{px}_score.json")
                json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                return out_path
            futs[ex.submit(_run)] = (prefix, ipath)

        for fut in tqdm(as_completed(futs), total=len(futs), desc="files"):
            out_path = fut.result()
            print(f"[OK] {out_path}")

if __name__ == "__main__":
    main()
