# score_runner.py
# 读取 interaction.json / groundtruth.json
# 1) global 用 LLM 判 rank（No|1..10）并映射为分数
# 2) 每个 step_i 用「step0 + stepi 全体对话」为上下文，对 stepi 中“每条 response”逐条打分（-5..5）
#    且严格先输出 analysis 再输出 score（JSON 字段顺序）
# 3) global 先按“从最后一轮向前”几何衰减分配到各 turn（默认 decay=0.9，归一化），
#    再在 turn 内按 response 的分数比例进行分摊；最后与本地分线性合成最终分
# 4) 额外保存：
#    - 最终汇总 score.json
#    - 原始 turn/step（含 per-agent）prompt/response 列表 original_prompts_responses.json

import json
import argparse
import re
from typing import List, Dict, Any, Optional, Tuple
import sys
import os

# add parent folder to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import Agent
import os

# -----------------------
# 通用工具
# -----------------------

_TURN_PAT = re.compile(r"(?:^|_)(\d+)$")

def _turn_index(k: str) -> int:
    s = str(k)
    m = _TURN_PAT.search(s)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    try:
        return int(s)
    except Exception:
        return 0

def _is_system_agent(a: Dict[str, Any]) -> bool:
    """剔除每轮中的 SYSTEM 消息（name/role/type 任一为 system 时）。"""
    for key in ("role", "name", "type"):
        v = (a.get(key) or "").strip().lower()
        if v == "system":
            return True
    return False

def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _split_golden(golden: str) -> List[str]:
    if not isinstance(golden, str):
        return []
    return [x.strip() for x in re.split(r"[;/,]|，|；", golden) if x.strip()]

def _extract_top10_text(interaction: Dict[str, Any]) -> str:
    fd = interaction.get("final_decision", {}) or {}
    return fd.get("top10") or fd.get("raw_decision", "") or ""

def _unwrap_interaction_root(interaction: Dict[str, Any]) -> Dict[str, Any]:
    if any(str(k).startswith("turn_") for k in interaction.keys()):
        return interaction
    for _, v in interaction.items():
        if isinstance(v, dict) and any(str(k).startswith("turn_") for k in v.keys()):
            return v
    return interaction

def _agents_from_step(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    解析一个 step 的“多 agent 结构”。若无 agents，则回退到单条文本。
    统一产出：[{name, role, phase, prompt, response}]
    """
    out: List[Dict[str, Any]] = []

    # 回退（无 agents）：将该 step 视作单一“匿名 agent”
    prompt = (
        step.get("prompt")
        or step.get("diagnosis")
        or step.get("opinion_text")
        or ""
    )
    response = (
        step.get("response")
        or step.get("output")
        or step.get("answer")
        or ""
    )
    out.append({
        "name": step.get("name") or step.get("role") or "Unknown",
        "role": step.get("role"),
        "phase": step.get("phase"),
        "prompt": prompt.strip(),
        "response": response.strip(),
    })
    return out

def _format_agents_block_from_step(step: Dict[str, Any]) -> str:
    """把一个 step 的“非 SYSTEM 的所有人”输出拼成评估上下文块。"""
    agents = _agents_from_step(step)
    lines = []
    for a in agents:
        head = f'- {a["name"]}' + (f' ({a["role"]})' if a.get("role") and a["role"] != a["name"] else "")
        body = a["response"] or a["prompt"]  # 优先展示 response，空则回退 prompt
        lines.append(f"{head}:\n{(body or '').strip()}\n")
    return ("\n".join(lines)).strip() or "(empty)"

# -----------------------
# GLOBAL：用 LLM 判 rank（No|1..10）
# -----------------------

def diagnosis_evaluate(predict_diagnosis: str, golden_diagnosis: str, *, agent: Optional[Agent] = None):
    if predict_diagnosis is None:
        raise Exception("Predict diagnosis is None")

    _agent = agent or Agent("You are a specialist in the field of rare diseases.", "diagnosis judger")
    predict_diagnosis = predict_diagnosis.replace("\n\n\n", "").strip()

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

    prompt = medium_prompt + f'Predicted diseases: {predict_diagnosis}\nStandard diagnosis: {golden_diagnosis}\n'
    raw_result = _agent.temp_responses(prompt)

    matches = re.findall(r"<answer>(.*?)</answer>", raw_result, re.IGNORECASE | re.DOTALL)
    rank = (matches[-1].strip() if matches else raw_result.strip())
    if re.fullmatch(r"(?i)no", rank.strip()):
        parsed = "No"
    else:
        m = re.search(r"\b(10|[1-9])\b", rank)
        parsed = int(m.group(1)) if m else "No"
    return parsed, raw_result

def evaluate_global_with_llm(agent: Agent, final_top10_text: str, golden_diagnosis: str,
                             gamma: float = 0.85, hit_max: float = 5.0, miss_penalty: float = -2.0) -> Dict[str, Any]:
    rank, raw = diagnosis_evaluate(final_top10_text, golden_diagnosis, agent=agent)
    if isinstance(rank, int):
        score = hit_max * (gamma ** (rank - 1))
        hit_rank = rank
    else:
        score = miss_penalty
        hit_rank = None
    return {"rank": hit_rank, "score": float(score), "raw": raw}

# -----------------------
# STEP：step0 + stepi（全体）上下文，对 stepi 的“每条 response”打分
# -----------------------

_STEP_SYSTEM_PROMPT = (
    "You are a strict adjudicator for MDT step influence. "
    "Judge how STEP i (all agents) influenced final Top10 quality, relative to STEP 0 (all agents). "
    "Be concise and fair; do not invent facts."
)

_STEP_USER_TEMPLATE_PER_AGENT = """Evaluate one TARGET RESPONSE within full context.

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

def _parse_step_json(raw: str) -> Tuple[str, int]:
    try:
        obj = json.loads(raw)
        analysis = (obj.get("analysis") or "").strip()
        sc = int(obj.get("score", 0))
        if sc < -5 or sc > 5:
            sc = 0
        if not analysis:
            analysis = "empty analysis"
        return analysis, sc
    except Exception:
        return "parse error", 0

def evaluate_target_response(agent: Agent, step0_block: str, stepi_block: str,
                             final_top10: str, golden: str, i: int,
                             target: Dict[str, Any]) -> Dict[str, Any]:
    role_desc = f"({target['role']})" if target.get("role") and target["role"] != target.get("name") else ""
    phase_desc = (target.get("phase") or "unknown")
    user_prompt = _STEP_USER_TEMPLATE_PER_AGENT.format(
        step0_block=step0_block,
        stepi_block=stepi_block,
        final_top10=final_top10,
        golden=golden,
        i=i,
        agent_name=target.get("name") or "Unknown",
        role_desc=role_desc,
        phase_desc=phase_desc,
        agent_prompt=target.get("prompt") or "",
        agent_response=target.get("response") or ""
    )
    # 不注入每轮的 SYSTEM 消息；这里只是 system-level 指令（若有接口）
    raw = agent.temp_responses_with_system(_STEP_SYSTEM_PROMPT, user_prompt) \
          if hasattr(agent, "temp_responses_with_system") else agent.temp_responses(user_prompt)
    analysis, score = _parse_step_json(raw)
    return {
        "analysis": analysis,
        "score": score,
        "eval_user_prompt": user_prompt,
        "eval_response": raw
    }

# -----------------------
# 分摊 global（turn 衰减）→ response，并合成
# -----------------------

def allocate_global(global_score: float,
                    grouped: Dict[str, List[Dict[str, Any]]],
                    decay: float = 0.9) -> None:
    """
    先将 global_score 按“从最后一轮向前”几何衰减分配到各 turn（末轮=1，倒二=decay¹，…），
    归一化后得到 per_turn_total，再在 turn 内按 response 的分数比例（与符号一致）分配。
    """
    valid_turns = [k for k, lst in grouped.items() if len(lst) > 0]
    if not valid_turns:
        return
    valid_turns = sorted(valid_turns, key=_turn_index)

    T = len(valid_turns)
    raw_weights = []
    for idx, _tk in enumerate(valid_turns):
        j = (T - 1) - idx  # 与末轮的距离
        raw_weights.append(decay ** j)
   
    for idx, tk in enumerate(valid_turns):
        rows = grouped[tk]
        print(f'Processing turn {tk}, rows: {rows}')
        if not rows:
            continue
        per_turn_total = global_score * raw_weights[idx]

        weights = [max(0, r.get("step_score", 0)) for r in rows]
        
        wsum = sum(weights)
        if wsum > 0:
            for i, r in enumerate(rows):
                r["global_per_turn_total"] = per_turn_total
                r["global_contrib"] = per_turn_total * (weights[i] / wsum)
        else:
            even = per_turn_total / len(rows)
            for r in rows:
                r["global_per_turn_total"] = per_turn_total
                r["global_contrib"] = even

def combine_scores(grouped: Dict[str, List[Dict[str, Any]]], w_global: float = 0.6, w_step: float = 0.4) -> None:
    for _, rows in grouped.items():
        for r in rows:
            g = float(r.get("global_contrib", 0.0))
            loc = float(r.get("step_score", 0.0))
            r["final_score"] = w_global * g + w_step * loc

# -----------------------
# 主流程
# -----------------------

def run(interaction: Dict[str, Any], groundtruth: Dict[str, Any],
        w_global: float = 0.6, w_step: float = 0.4,
        gamma: float = 0.85, hit_max: float = 5.0, miss_penalty: float = -5.0,
        decay: float = 0.9) -> Dict[str, Any]:

    final_top10 = _extract_top10_text(interaction)
    golden = groundtruth.get("golden_diagnosis", "")
    print(f'final top10 {final_top10},\ngolden {golden}')

    # 1) GLOBAL
    if "predict_rank" not in groundtruth or groundtruth["predict_rank"] is None:
        global_agent = Agent("You are a specialist in the field of rare diseases.", "diagnosis judger")
        g = evaluate_global_with_llm(global_agent, final_top10, golden, gamma=gamma, hit_max=hit_max, miss_penalty=miss_penalty)
        global_score, hit_rank = g["score"], g["rank"]
    else:
        global_score = hit_max * (gamma ** (int(groundtruth["predict_rank"]) - 1))
        hit_rank = int(groundtruth["predict_rank"])

    # 2) 收集原始 prompt/response（局部函数）
    turns = _unwrap_interaction_root(interaction)

    def collect_prompts_responses(_turns: Dict[str, Any]) -> List[Dict[str, Any]]:
        recs: List[Dict[str, Any]] = []
        for turn_key, turn in _turns.items():
            if not str(turn_key).startswith("turn_"):
                continue
            steps = turn.get("steps", [])
            for idx, step in enumerate(steps):
                agents = _agents_from_step(step)
                if agents:
                    for a in agents:
                        recs.append({
                            "turn": turn_key,
                            "step": idx,
                            "name": a.get("name"),
                            "role": a.get("role"),
                            "phase": a.get("phase"),
                            "prompt": a.get("prompt"),
                            "response": a.get("response"),
                        })
                else:
                    # 理论上不会进入（_agents_from_step 已有回退）
                    recs.append({
                        "turn": turn_key,
                        "step": idx,
                        "name": step.get("name") or step.get("role") or "Unknown",
                        "role": step.get("role"),
                        "phase": step.get("phase"),
                        "prompt": step.get("prompt"),
                        "response": step.get("response"),
                    })
        return recs

    original_recs = collect_prompts_responses(turns)

    # 3) 对每个次轮（i>=1）的每条 response 打分
    step_agent = Agent("You are a strict judge. Output strict JSON with 'analysis' then 'score'.", "step influence judger")
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    by_role_step_sum: Dict[str, float] = {}

    for turn_key, turn in turns.items():
        if not str(turn_key).startswith("turn_"):
            continue
        steps = turn.get("steps", [])
        grouped[turn_key] = []
        if not steps:
            continue

        step0_block = _format_agents_block_from_step(steps[0])

        for idx in range(1, len(steps)):
            step_i = steps[idx]
            stepi_block = _format_agents_block_from_step(step_i)
            agents = _agents_from_step(step_i)

            for a in agents:
                if a.get("role") == "SYSTEM":
                    continue
                # 对“每条 response”单独打分
                eval_res = evaluate_target_response(
                    step_agent, step0_block, stepi_block,
                    final_top10, golden, i=idx, target=a
                )
                row = {
                    "turn": turn_key,
                    "step": idx,
                    "name": a.get("name"),
                    "role": a.get("role"),
                    "phase": a.get("phase"),
                    "prompt": a.get("prompt"),       # 保存该条的原始 prompt
                    "response": a.get("response"),   # 保存该条的原始 response
                    "analysis": eval_res["analysis"],            # 先展示
                    "step_score": int(eval_res["score"]),        # 再展示
                    "global_contrib": 0.0,
                    "final_score": 0.0,
                    "eval_user_prompt": eval_res["eval_user_prompt"],
                    "eval_response": eval_res["eval_response"],
                }
                grouped[turn_key].append(row)

                if row["role"]:
                    by_role_step_sum[row["role"]] = by_role_step_sum.get(row["role"], 0.0) + row["step_score"]

    # 4) 分摊 global（turn 衰减 → response 比例）
    allocate_global(global_score, grouped, decay=decay)

    # 5) 合成 global + local
    combine_scores(grouped, w_global=w_global, w_step=w_step)

    # 6) 汇总
    steps_flat = []
    by_role_final_sum: Dict[str, float] = {}
    for tk in sorted(grouped.keys(), key=_turn_index):
        for r in grouped[tk]:
            steps_flat.append(r)
            rr = r.get("role")
            if rr:
                by_role_final_sum[rr] = by_role_final_sum.get(rr, 0.0) + r["final_score"]

    result = {
        "global": {"score": global_score, "rank": hit_rank},
        "weights": {"w_global": w_global, "w_step": w_step, "decay": decay},
        "by_role": {
            "step_influence_sum": by_role_step_sum,
            "final_score_sum": {k: round(v, 4) for k, v in by_role_final_sum.items()}
        },
        "steps": steps_flat
    }

    # 保存两个 JSON 文件
    with open("./score.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open("./original_prompts_responses.json", "w", encoding="utf-8") as f:
        json.dump(original_recs, f, ensure_ascii=False, indent=2)

    return result

# -----------------------
# CLI
# -----------------------
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to folder containing interaction and groundtruth JSONs")
    parser.add_argument("--w_global", type=float, default=0.6)
    parser.add_argument("--w_step", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=0.85)
    parser.add_argument("--hit_max", type=float, default=5.0)
    parser.add_argument("--miss_penalty", type=float, default=-5.0)
    parser.add_argument("--decay", type=float, default=0.9)
    args = parser.parse_args()

    # Find all files
    interact_files = sorted(glob.glob(os.path.join(args.data_dir, "*interact*.json")))
    gt_files = sorted(glob.glob(os.path.join(args.data_dir, "*.A.json")))

    if not interact_files:
        raise FileNotFoundError(f"No interaction files found in {args.data_dir}")
    if not gt_files:
        raise FileNotFoundError(f"No groundtruth .A.json files found in {args.data_dir}")

    # Map ground truth files by prefix (filename before .A.json)
    gt_map = {os.path.basename(f).replace(".A.json", ""): f for f in gt_files}

    results = {}
    for ifile in interact_files:
        iname = os.path.basename(ifile)
        prefix = iname.split(".")[0]  # everything before first dot, adjust if needed
        if prefix not in gt_map:
            print(f"[WARN] No groundtruth file matching {iname}")
            continue

        with open(ifile, "r", encoding="utf-8-sig") as f:
            interaction = json.load(f)
        with open(gt_map[prefix], "r", encoding="utf-8-sig") as f:
            gt = json.load(f)

        out = run(
            interaction,
            gt,
            w_global=args.w_global,
            w_step=args.w_step,
            gamma=args.gamma,
            hit_max=args.hit_max,
            miss_penalty=args.miss_penalty,
            decay=args.decay
        )
        results[iname] = out

        out_name = prefix + ".A_score.json"
        out_path = os.path.join(args.data_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        print(f"[OK] Saved score for {iname} -> {out_name}")

if __name__ == "__main__":
    main()
