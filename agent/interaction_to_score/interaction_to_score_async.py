#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
score_runner.py (concurrent + tqdm + quiet)
"""

import json, argparse, re, sys, os, glob
from typing import List, Dict, Any, Optional, Tuple
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm.auto import tqdm  # NEW

# add parent folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import Agent  # noqa: E402

# -----------------------
# Utils
# -----------------------

_TURN_PAT = re.compile(r"(?:^|_)(\d+)$")

def _turn_index(k: str) -> int:
    s = str(k); m = _TURN_PAT.search(s)
    if m:
        try: return int(m.group(1))
        except Exception: return 0
    try: return int(s)
    except Exception: return 0

def _unwrap_interaction_root(interaction: Dict[str, Any]) -> Dict[str, Any]:
    if any(str(k).startswith("turn_") for k in interaction.keys()):
        return interaction
    for _, v in interaction.items():
        if isinstance(v, dict) and any(str(k).startswith("turn_") for k in v.keys()):
            return v
    return interaction

def _extract_top10_text(interaction: Dict[str, Any]) -> str:
    fd = interaction.get("final_decision", {}) or {}
    return fd.get("top10") or fd.get("raw_decision", "") or ""

def _agents_from_step(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    prompt = (step.get("prompt") or step.get("diagnosis") or step.get("opinion_text") or "").strip()
    response = (step.get("response") or step.get("output") or step.get("answer") or "").strip()
    return [{
        "name": step.get("name") or step.get("role") or "Unknown",
        "role": step.get("role"),
        "phase": step.get("phase"),
        "prompt": prompt,
        "response": response,
    }]

def _format_agents_block_from_step(step: Dict[str, Any]) -> str:
    agents = _agents_from_step(step)
    lines = []
    for a in agents:
        head = f'- {a["name"]}' + (f' ({a["role"]})' if a.get("role") and a["role"] != a["name"] else "")
        body = a["response"] or a["prompt"]
        lines.append(f"{head}:\n{(body or '').strip()}\n")
    return ("\n".join(lines)).strip() or "(empty)"

# -----------------------
# GLOBAL：LLM 判 rank（No|1..10）—— Prompt 未更改
# -----------------------

_GLOBAL_SYSTEM = "You are a specialist in the field of rare diseases."
_GLOBAL_ROLE    = "diagnosis judger"

def _diagnosis_evaluate(predict_diagnosis: str, golden_diagnosis: str, *, agent: Optional[Agent] = None):
    if predict_diagnosis is None:
        raise ValueError("Predict diagnosis is None")

    _agent = agent or Agent(_GLOBAL_SYSTEM, _GLOBAL_ROLE)
    predict_diagnosis = predict_diagnosis.replace("\n\n\n", "").strip()

    # === 原文 prompt：逐字未改 ===
    medium_prompt = (
        "I will now give you ten predicted diseases. "
        "If the predicted diagnosis is in the standard diagnosis list, output its rank (1–10); "
        "otherwise, output \"No\". Output exactly one value—either \"No\" or a single number from 1 to 10. "
        "If multiple match, choose the highest rank.\n\n"
        "Decide whether the predicted disease and a standard diagnosis are the SAME medical condition. "
        "Be moderately strict: allow synonyms and parent↔unspecified subtype matches, "
        "but do not accept clearly distinct subtypes or genetic forms as the same.\n\n"
        "Matching rules:\n"
        "- ACCEPT if they are synonyms, eponyms, abbreviations, or different wording for the same condition.\n"
        "- ACCEPT if one is a broad parent category and the other is an unspecified form within that category.\n"
        "- REJECT if they specify different subtypes (e.g., type I vs type II), different enzyme defects, or different genes.\n"
        "- REJECT if they are unrelated conditions, or only partially overlapping.\n\n"
        "Output format:\n"
        "Write your reasoning inside <think>…</think> and the final decision inside <answer>…</answer>.\n\n"
        "<think>\nStep-by-step reasoning:\n</think>\n"
        "<answer>No|1-10</answer>\n"
    )
    # ============================

    prompt = medium_prompt + f'Predicted diseases: {predict_diagnosis}\nStandard diagnosis: {golden_diagnosis}\n'
    raw = _agent.temp_responses(prompt)
    m = re.findall(r"<answer>(.*?)</answer>", raw, re.IGNORECASE | re.DOTALL)
    rank = (m[-1].strip() if m else raw.strip())
    if re.fullmatch(r"(?i)no", rank): return "No", raw
    m2 = re.search(r"\b(10|[1-9])\b", rank)
    return (int(m2.group(1)) if m2 else "No"), raw

def _evaluate_global_with_llm(final_top10_text: str, golden: str,
                              gamma: float = 0.85, hit_max: float = 5.0, miss_penalty: float = -5.0) -> Dict[str, Any]:
    rank, raw = _diagnosis_evaluate(final_top10_text, golden)
    if isinstance(rank, int):
        score = hit_max * (gamma ** (rank - 1))
        hit_rank = rank
    else:
        score = miss_penalty
        hit_rank = None
    return {"rank": hit_rank, "score": float(score), "raw": raw}

# -----------------------
# STEP 评分类——Prompt 未更改；严格 analysis -> score
# -----------------------

# === 原文 system prompt：逐字未改 ===
_STEP_SYSTEM_PROMPT = (
    "You are a strict adjudicator for MDT step influence. "
    "Judge how STEP i (all agents) influenced final Top10 quality, relative to STEP 0 (all agents). "
    "Be concise and fair; do not invent facts."
)
# === 原文 user 模板：逐字未改 ===
_STEP_USER_TEMPLATE = """Evaluate one TARGET RESPONSE within full context.

[STEP 0 — All agents]
{step0_block}

[STEP {i} — All agents]
{stepi_block}

[TARGET RESPONSE to score ONLY]
Agent: {agent_name} {role_desc}
Phase: {phase_desc}
Prompt:
{agent_prompt}

Response:
{agent_response}

Only consider:
- Final Top10 (chair): {final_top10}
- Golden diagnosis (synonyms allowed): {golden}

Output STRICT JSON with fields in this order:
{{
  "analysis": "<5-10 short sentences focusing on THIS target response's influence, comparison to step 0, strengths/weaknesses>",
  "score": 0..5
}}
"""
# ===================================

def _parse_step_json(raw: str) -> Tuple[str, int]:
    try:
        obj = json.loads(raw)
        analysis = (obj.get("analysis") or "").strip() or "empty analysis"
        sc = int(obj.get("score", 0))
        if sc < -5 or sc > 5: sc = 0
        return analysis, sc
    except Exception:
        return "parse error", 0

def _eval_one_response(step0_block: str, stepi_block: str, final_top10: str, golden: str, i: int,
                       target: Dict[str, Any]) -> Dict[str, Any]:
    role_desc = f"({target['role']})" if target.get("role") and target["role"] != target.get("name") else ""
    phase_desc = (target.get("phase") or "unknown")
    user_prompt = _STEP_USER_TEMPLATE.format(
        step0_block=step0_block, stepi_block=stepi_block,
        final_top10=final_top10, golden=golden, i=i,
        agent_name=target.get("name") or "Unknown",
        role_desc=role_desc, phase_desc=phase_desc,
        agent_prompt=target.get("prompt") or "",
        agent_response=target.get("response") or ""
    )
    # 线程安全：每次新建 Agent
    agent = Agent("You are a strict judge. Output strict JSON with 'analysis' then 'score'.", "step influence judger")
    raw = (agent.temp_responses_with_system(_STEP_SYSTEM_PROMPT, user_prompt)
           if hasattr(agent, "temp_responses_with_system") else agent.temp_responses(user_prompt))
    analysis, score = _parse_step_json(raw)
    return {"analysis": analysis, "score": score, "eval_user_prompt": user_prompt, "eval_response": raw}

# -----------------------
# 分摊 + 合成
# -----------------------

def _allocate_global(global_score: float, grouped: Dict[str, List[Dict[str, Any]]], decay: float = 0.9) -> None:
    turns = [k for k, lst in grouped.items() if lst]
    if not turns: return
    turns = sorted(turns, key=_turn_index)
    T = len(turns)
    raw_weights = [decay ** ((T - 1) - idx) for idx in range(T)]
    for idx, tk in enumerate(turns):
        rows = grouped[tk]
        per_turn_total = global_score * raw_weights[idx]
        weights = [max(0.0, float(r.get("step_score", 0))) for r in rows]
        wsum = sum(weights)
        if wsum > 0:
            for r, w in zip(rows, weights):
                r["global_per_turn_total"] = per_turn_total
                r["global_contrib"] = per_turn_total * (w / wsum)
        else:
            even = per_turn_total / len(rows)
            for r in rows:
                r["global_per_turn_total"] = per_turn_total
                r["global_contrib"] = even

def _combine_scores(grouped: Dict[str, List[Dict[str, Any]]], w_global: float, w_step: float) -> None:
    for rows in grouped.values():
        for r in rows:
            r["final_score"] = w_global * float(r.get("global_contrib", 0.0)) + w_step * float(r.get("step_score", 0.0))

# -----------------------
# 单文件主流程
# -----------------------

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

def run_one(interaction: Dict[str, Any], groundtruth: Dict[str, Any],
            w_global: float, w_step: float, gamma: float, hit_max: float,
            miss_penalty: float, decay: float, eval_workers: int = 8, verbose: bool = False) -> Dict[str, Any]:

    final_top10 = _extract_top10_text(interaction)
    golden = groundtruth.get("golden_diagnosis", "")

    # GLOBAL
    if "predict_rank" not in groundtruth or groundtruth["predict_rank"] is None:
        g = _evaluate_global_with_llm(final_top10, golden, gamma=gamma, hit_max=hit_max, miss_penalty=miss_penalty)
        global_score, hit_rank = g["score"], g["rank"]
    else:
        hit_rank = int(groundtruth["predict_rank"])
        global_score = hit_max * (gamma ** (hit_rank - 1))

    # 收集原始
    turns = _unwrap_interaction_root(interaction)
    original_recs = _collect_prompts_responses(turns)

    # 并发评估（带 tqdm）
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    by_role_step_sum: Dict[str, float] = {}

    for turn_key, turn in sorted(turns.items(), key=lambda kv: _turn_index(kv[0])):
        if not str(turn_key).startswith("turn_"): continue
        steps = turn.get("steps", []) or []
        grouped[turn_key] = []
        if not steps: continue

        step0_block = _format_agents_block_from_step(steps[0])

        # 先统计要评的条目数以设置进度条
        total_targets = sum(len(_agents_from_step(steps[idx])) for idx in range(1, len(steps)))
        if total_targets == 0: continue

        with ThreadPoolExecutor(max_workers=max(1, eval_workers)) as ex, \
             tqdm(total=total_targets, disable=not verbose, desc=f"{turn_key} eval") as pbar:  # NEW
            futures, metas = [], []
            for idx in range(1, len(steps)):
                step_i = steps[idx]
                stepi_block = _format_agents_block_from_step(step_i)
                for a in _agents_from_step(step_i):
                    if (a.get("role") or "").upper() == "SYSTEM": continue
                    futures.append(ex.submit(_eval_one_response, step0_block, stepi_block, final_top10, golden, idx, a))
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
                if row["role"]:
                    by_role_step_sum[row["role"]] = by_role_step_sum.get(row["role"], 0.0) + row["step_score"]
                pbar.update(1)  # NEW

    _allocate_global(global_score, grouped, decay=decay)
    _combine_scores(grouped, w_global=w_global, w_step=w_step)

    steps_flat, by_role_final_sum = [], {}
    for tk in sorted(grouped.keys(), key=_turn_index):
        for r in grouped[tk]:
            steps_flat.append(r)
            rr = r.get("role")
            if rr: by_role_final_sum[rr] = by_role_final_sum.get(rr, 0.0) + float(r["final_score"])

    result = {
        "global": {"score": global_score, "rank": hit_rank},
        "weights": {"w_global": w_global, "w_step": w_step, "decay": decay},
        "by_role": {
            "step_influence_sum": by_role_step_sum,
            "final_score_sum": {k: round(v, 4) for k, v in by_role_final_sum.items()}
        },
        "steps": steps_flat
    }

    # 文件输出交给调用方；同时保存两份全局文件（与原逻辑一致）
    with open("./score.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open("./original_prompts_responses.json", "w", encoding="utf-8") as f:
        json.dump(_collect_prompts_responses(turns), f, ensure_ascii=False, indent=2)

    return result

# -----------------------
# CLI（文件级并发 + tqdm）
# -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--w_global", type=float, default=0.6)
    ap.add_argument("--w_step", type=float, default=0.4)
    ap.add_argument("--gamma", type=float, default=0.85)
    ap.add_argument("--hit_max", type=float, default=5.0)
    ap.add_argument("--miss_penalty", type=float, default=-5.0)
    ap.add_argument("--decay", type=float, default=0.9)
    ap.add_argument("--workers", type=int, default=10, help="file-level concurrency")
    ap.add_argument("--eval-workers", type=int, default=8, help="per-file response-level concurrency")
    ap.add_argument("--verbose", action="store_true", help="show per-turn tqdm and extra logs")
    args = ap.parse_args()

    interact_files = sorted(glob.glob(os.path.join(args.data_dir, "*interact*.json")))
    gt_files = sorted(glob.glob(os.path.join(args.data_dir, "*.A.json")))
    if not interact_files: raise FileNotFoundError(f"No interaction files found in {args.data_dir}")
    if not gt_files: raise FileNotFoundError(f"No groundtruth .A.json files found in {args.data_dir}")

    gt_map = {os.path.basename(f).replace(".A.json", ""): f for f in gt_files}

    def _process_one(ifile: str) -> Tuple[str, Optional[str], Optional[str]]:
        iname = os.path.basename(ifile); prefix = iname.split(".")[0]

        out_name = prefix + ".A_score.json"
        out_path = os.path.join(args.data_dir, out_name)
        if os.path.exists(out_path):
            return iname, out_path, None

        if prefix not in gt_map:
            return iname, None, f"[WARN] No groundtruth file matching {iname}"
        with open(ifile, "r", encoding="utf-8-sig") as f:
            interaction = json.load(f)
        with open(gt_map[prefix], "r", encoding="utf-8-sig") as f:
            gt = json.load(f)

        out = run_one(
            interaction, gt,
            w_global=args.w_global, w_step=args.w_step,
            gamma=args.gamma, hit_max=args.hit_max,
            miss_penalty=args.miss_penalty, decay=args.decay,
            eval_workers=max(1, args.eval_workers), verbose=args.verbose
        )
        
        with open(out_path, "w", encoding="utf-8") as wf:
            json.dump(out, wf, ensure_ascii=False, indent=2)
        return iname, out_path, None

    results, errors = {}, []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex, \
         tqdm(total=len(interact_files), disable=not args.verbose, desc="files") as pbar:  # NEW
        futs = {ex.submit(_process_one, f): f for f in interact_files}
        for fut in as_completed(futs):
            iname, out_path, err = fut.result()
            if err:
                errors.append(err)
            else:
                results[iname] = out_path
            pbar.update(1)

    # 更安静的总结输出
    for k, v in results.items():
        print(f"[OK] {k} -> {v}")
    if errors:
        print("[WARN] Some files skipped:")
        for e in errors:
            print("  -", e)

if __name__ == "__main__":
    main()
