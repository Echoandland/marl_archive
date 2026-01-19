# utils_with_experience_math.py  (math version, per-expert KB + global fallback)

from datetime import datetime
import os
import json
import random
from tqdm import tqdm
from prettytable import PrettyTable
from termcolor import cprint
from pptree import Node
try:
    from openai import OpenAI  # optional
except Exception:
    OpenAI = None
from pptree import *
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

# ----------------------------
# Small utilities
# ----------------------------
def _now_ts() -> str:
    return datetime.utcnow().isoformat()

def _truncate(s: str, n: int) -> str:
    if not isinstance(s, str):
        s = str(s)
    return s[:n]

def _strip_fenced(text: str) -> str:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = text.lstrip("`")
        if text.startswith("json"):
            text = text[len("json"):].lstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
    return text

def _safe_json_loads(s: str, default: Any) -> Any:
    s = _strip_fenced(str(s))
    try:
        obj = json.loads(s)
        if isinstance(default, dict) and not isinstance(obj, dict):
            return default, False
        if isinstance(default, list) and not isinstance(obj, list):
            return default, False
        return obj, True
    except Exception:
        return default, False

def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for it in items or []:
        if not it:
            continue
        it = str(it).strip()
        if not it:
            continue
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out

# ----------------------------
# Math-specific parsing helpers
# ----------------------------
def parse_solution_block(text: str) -> str:
    """
    Extract <solution_outline> ... </solution_outline> if present,
    else return the whole text.
    """
    if not text:
        return ""
    m = re.search(r"<solution_outline>(.*?)</solution_outline>", text, flags=re.S | re.I)
    return (m.group(1).strip() if m else str(text).strip())

def parse_math_final(text: str):
    """
    Extract <analysis>, <final_answer>, <formal_proof> blocks.
    Return empty strings if not present.
    """
    if not text:
        return "", "", ""
    def _grab(tag):
        m = re.search(fr"<{tag}>(.*?)</{tag}>", text, flags=re.S | re.I)
        return m.group(1).strip() if m else ""
    return _grab("analysis"), _grab("final_answer"), _grab("formal_proof")

# ----------------------------
# Experience (per-expert KB with global fallback)
# ----------------------------
try:
    # 你刚改好的数学经验库实现
    from experience_api_math import ExperienceKB
except Exception:
    ExperienceKB = None

# 经验库根目录（按专家拆分的子库会放在这里）
# 例如：experience_to_db/exp_index_math/Algebra/, Number_Theory/, ...
MATH_KB_ROOT = os.environ.get("MATH_KB_ROOT", "experience_to_db/exp_index_math")

# 全局共享库（可选）：如果找不到某专家的子库，就回退到它
# 例如：kb_math/ 里是一个合并好的单库（index.faiss/meta.json/config.json）
GLOBAL_MATH_KB_DIR = os.environ.get("MATH_KB_DIR", "kb_math")

def _normalize_expert_name_from_role(role: str) -> str:
    """
    从 role 中抽出“专家名”，规范到目录名：
    - 先取 'specialty (role)' 里的 specialty 部分（括号前）
    - 去首尾空白
    - 将空格/斜杠替换成下划线
    例如：
      "Number Theory (leader)" -> "Number_Theory"
      "Calculus/Analysis (member)" -> "Calculus_Analysis"
    """
    if not isinstance(role, str) or not role.strip():
        return "Unknown"
    base = role.split("(")[0].strip()        # 去掉括号和其中的“leader/member”等
    base = base.split(":")[0].strip()        # 避免 "Algebra: xxx" 这样的后缀
    base = base.replace("/", "_").replace("\\", "_")
    base = "_".join(x for x in base.split() if x)  # 空格 -> 下划线（保序）
    return base or "Unknown"

# ----------------------------
# Agent
# ----------------------------
class Agent:
    def __init__(self, instruction, role, examplers=None, model_info='gpt-5', port=8000, img_path=None, use_experience: bool = True, unified_experience: bool = False):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path
        self.use_experience = use_experience
        self.unified_experience = unified_experience

        self.experience = None
        self.init_experience()

        self.prompt_tokens = []
        self.completion_tokens = []
        self.total_tokens = []

        # ---- Client setup (replace with your own keys or env vars) ----
        if self.model_info == 'gemini-pro' or "gemini" in self.model_info:
            try:
                from google import genai  # lazy import to avoid import-time crash
            except Exception as e:
                raise RuntimeError("Gemini selected but google-genai not installed") from e
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Missing GEMINI_API_KEY (or GENAI_API_KEY). See ../env.example")
            self.client = genai.Client(api_key=api_key)
            self._chat = self.client.chats.create(model=self.model_info)
        elif self.model_info in [
            'gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini',
            'o3-2025-04-16', 'gpt-5', 'gpt-4.1-2025-04-14'
        ] or 'qwen3' in self.model_info.lower():
            # Use standard OpenAI-compatible endpoint
            if OpenAI is None:
                raise RuntimeError("OpenAI selected but openai package not installed")
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                base_url=os.environ.get("OPENAI_BASE_URL") or f"http://localhost:{port}/v1",
                default_headers=({"X-Api-Key": os.environ["X_API_KEY"]} if os.environ.get("X_API_KEY") else None),
            )
            self.messages = [{"role": "system", "content": instruction}]
            if examplers is not None:
                for ex in examplers:
                    self.messages.append({"role": "user", "content": ex.get('question', '')})
                    self.messages.append({"role": "assistant", "content": ex.get('answer', '') + "\n\n" + ex.get('reason', '')})
        else:
            # local server fallback
            if OpenAI is not None:
                self.client = OpenAI(api_key='EMPTY', base_url=f"http://localhost:{port}/v1")
            else:
                # Minimal client using requests to call OpenAI-compatible endpoints
                import requests  # may raise if not installed
                base_url = f"http://localhost:{port}/v1"
                class _Resp:
                    def __init__(self, content):
                        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})]
                        self.usage = None
                class _Chat:
                    def __init__(self, base): self._base = base
                    class _Completions:
                        def __init__(self, base): self._base = base
                        def create(self, **kwargs):
                            url = f"{self._base}/chat/completions"
                            headers = {"Content-Type": "application/json"}
                            res = requests.post(url, json=kwargs, headers=headers, timeout=60)
                            data = res.json()
                            # assume OpenAI compatible response
                            return type("R", (), {
                                "choices": data.get("choices", [{"message": {"content": data.get("content", "")}}]),
                                "usage": data.get("usage")
                            })
                    @property
                    def completions(self): return self._comp
                class _Client:
                    def __init__(self, base):
                        self.chat = type("Chat", (), {"completions": type("Comp", (), {"create": _Chat(base)._Completions(base).create})()})()
                self.client = _Client(base_url)
            self.messages = [{"role": "system", "content": instruction}]
            if examplers is not None:
                for ex in examplers:
                    self.messages.append({"role": "user", "content": ex.get('question', '')})
                    self.messages.append({"role": "assistant", "content": ex.get('answer', '') + "\n\n" + ex.get('reason', '')})

    def init_experience(self):
        """
        当 unified_experience=True 时，总是加载全局共享库 GLOBAL_MATH_KB_DIR。
        否则：优先按专家名加载 MATH_KB_ROOT/<ExpertName>/，失败再回退到全局库。
        都不存在则禁用经验增强。
        """
        if not getattr(self, "use_experience", True) or ExperienceKB is None:
            self.experience = None
            return

        # 统一经验库：忽略专家名，直接用全局库
        if getattr(self, "unified_experience", False):
            global_dir = Path(GLOBAL_MATH_KB_DIR)
            if global_dir.exists() and (global_dir / "meta.json").exists():
                try:
                    self.experience = ExperienceKB(storage_dir=str(global_dir))
                    cprint(f"[KB] Using unified global KB: {global_dir}", "cyan")
                    return
                except Exception as e:
                    cprint(f"[KB] Failed to load unified global KB {global_dir}: {e}", "red")
            self.experience = None
            cprint("[KB] Unified KB not found. Experience augmentation disabled.", "magenta")
            return

        # 分角色优先，其次全局回退
        expert_name = _normalize_expert_name_from_role(self.role)
        per_expert_dir = Path(MATH_KB_ROOT) / expert_name

        if per_expert_dir.exists() and (per_expert_dir / "meta.json").exists():
            try:
                self.experience = ExperienceKB(storage_dir=str(per_expert_dir))
                cprint(f"[KB] Loaded per-expert KB: {per_expert_dir}", "cyan")
                return
            except Exception as e:
                cprint(f"[KB] Failed to load per-expert KB {per_expert_dir}: {e}", "red")

        global_dir = Path(GLOBAL_MATH_KB_DIR)
        if global_dir.exists() and (global_dir / "meta.json").exists():
            try:
                self.experience = ExperienceKB(storage_dir=str(global_dir))
                cprint(f"[KB] Fallback to global KB: {global_dir}", "yellow")
                return
            except Exception as e:
                cprint(f"[KB] Failed to load global KB {global_dir}: {e}", "red")

        self.experience = None
        cprint(f"[KB] No KB found for role={expert_name}. Experience augmentation disabled.", "magenta")

    def augment_prompt(self, message, top_k_per_query=3, max_hints=3):
        if (not self.use_experience) or (not self.experience):
            return message
        return self.experience.augment_prompt(message, top_k_per_query=top_k_per_query, max_hints=max_hints)

    def chat(self, message, img_path=None, chat_mode=True, temperature=0.3):
        if self.model_info == 'gemini-pro' or "gemini" in self.model_info:
            for _ in range(5):
                try:
                    response = self._chat.send_message(message)
                    return response.text
                except Exception as e:
                    cprint(f"[Gemini] retry due to {e}", 'red')
            return "Error: Gemini failed."
        else:
            self.messages.append({"role": "user", "content": message})
            model_name = "gpt-3.5-turbo" if self.model_info == 'gpt-3.5' else self.model_info
            kwargs = dict(model=model_name, messages=self.messages)
            if "gpt-5" not in model_name and "o3" not in model_name and "gpt-4.1-2025-04-14" not in model_name:
                kwargs["temperature"] = temperature
            if "gpt-5" in model_name:
                kwargs["model"] = "gpt-5"
                kwargs["reasoning_effort"] = "low"
                kwargs["verbosity"] = "low"

            response = self.client.chat.completions.create(**kwargs)
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            if hasattr(response, "usage") and response.usage:
                self.prompt_tokens.append(getattr(response, "usage").prompt_tokens if hasattr(response, "usage") else 0)
                self.completion_tokens.append(getattr(response, "usage").completion_tokens if hasattr(response, "usage") else 0)
                self.total_tokens.append(getattr(response, "usage").total_tokens if hasattr(response, "usage") else 0)
            return response.choices[0].message.content

    def temp_responses(self, message, img_path=None):
        # identical to chat but does not mutate self.messages permanently
        messages = getattr(self, "messages", [])[:]
        messages.append({"role": "user", "content": message})

        if self.model_info == 'gemini-pro' or "gemini" in self.model_info:
            for _ in range(5):
                try:
                    response = self._chat.send_message(message)
                    return response.text
                except Exception as e:
                    cprint(f"[Gemini] retry due to {e}", 'red')
            return "Error: Gemini failed."
        else:
            model_name = "gpt-3.5-turbo" if self.model_info == 'gpt-3.5' else self.model_info
            kwargs = dict(model=model_name, messages=messages)
            if "gpt-5" in model_name:
                kwargs["model"] = "gpt-5"
                kwargs["reasoning_effort"] = "low"
                kwargs["verbosity"] = "low"
            response = self.client.chat.completions.create(**kwargs)
            if hasattr(response, "usage") and response.usage:
                self.prompt_tokens.append(getattr(response, "usage").prompt_tokens if hasattr(response, "usage") else 0)
                self.completion_tokens.append(getattr(response, "usage").completion_tokens if hasattr(response, "usage") else 0)
                self.total_tokens.append(getattr(response, "usage").total_tokens if hasattr(response, "usage") else 0)
            return response.choices[0].message.content

# ----------------------------
# Group for Math
# ----------------------------
class Group:
    def __init__(self, goal, members, model, question, port=8000, examplers=None, use_experience: bool = True, unified_experience: bool = False):
        self.goal = goal
        self.members = []
        for member_info in members:
            sys_prompt = 'You are a {} who {}.'.format(
                f"{member_info['specialty']} ({member_info['role']})",
                member_info['expertise_description'].lower()
            )
            _agent = Agent(sys_prompt, role=f"{member_info['specialty']} ({member_info['role']})",
                           model_info=random.choice(model), port=port, use_experience=use_experience, unified_experience=unified_experience)
            self.members.append(_agent)
        self.question = question
        self.examplers = examplers
        self._state = None

    def interact(self, comm_type, message=None, img_path=None):
        # legacy (kept for API parity)
        interaction_log = {
            "timestamp": _now_ts(),
            "comm_type": comm_type,
            "goal": getattr(self, "goal", None),
            "question": getattr(self, "question", None),
            "members": [m.role for m in getattr(self, "members", [])],
            "steps": [],
            "final_response": None
        }
        if comm_type == 'external':
            interaction_log["message"] = message
            interaction_log["img_path"] = img_path
            return "External interaction logged.", interaction_log
        else:
            return "Use rare_interact for math workflow.", interaction_log

    def rare_interact(
        self,
        comm_type: str,
        message: Optional[str] = None,
        first_turn=False,
        img_path: Optional[str] = None
    ):
        """
        Math (RARE-like) multi-round convergence:
        - Each specialist outputs Reflection + <solution_outline> (with [S1].. steps) + <result> + <gaps>.
        - Peer review returns a JSON {analysis, verdict, validated, issues[], tests, ...}.
        - Acceptance: no issues at any severity and peer verdicts all 'accept' → record into snapshot..
        """
        interaction_log: Dict[str, Any] = {
            "timestamp": _now_ts(),
            "comm_type": comm_type,
            "goal": getattr(self, "goal", None),
            "question": getattr(self, "question", None),
            "members": [f"{getattr(m, 'role', 'Unknown')}_{getattr(m, 'model_info', 'Unknown')}" for m in getattr(self, "members", [])],
            "steps": [],
            "final_response": None
        }

        if comm_type == "external":
            interaction_log["message"] = message
            interaction_log["img_path"] = img_path
            return "External interaction logged.", interaction_log

        members: List[Any] = getattr(self, "members", [])
        if not members:
            interaction_log["final_response"] = "No members."
            return "No members.", interaction_log

        # init persistent state
        if not hasattr(self, "_state") or not isinstance(self._state, dict) or not self._state:
            roles = [getattr(m, "role", "Unknown") for m in members]
            self._state = {
                "r": 0,
                "fc_global": False,
                "fc_s": {role: False for role in roles},
                "delta": {role: {} for role in roles},  # aggregated structured feedback per role
                "O_rounds": [],
                "last_opinion": {}  # role -> latest solution outline (string)
            }

        state: Dict[str, Any] = self._state
        r: int = state["r"]

        interaction_log["steps"].append({
            "role": "SYSTEM", "phase": "round_start",
            "prompt": f"Round {r} start", "response": ""
        })

        # accumulate accepted across all past rounds for fuller context
        acc_all: Dict[str, Any] = {}
        for rr in state.get("O_rounds", []):
            try:
                acc_all.update(rr.get("accepted", {}) or {})
            except Exception:
                pass
        accepted_snapshot: Dict[str, Any] = acc_all
        round_O: Dict[str, Any] = {}

        def _json(obj): return json.dumps(obj, ensure_ascii=False)

        # pick leader (optional, not used for math prompts seriously)
        lead_member = None
        assist_members = []
        for member in self.members:
            if 'lead' in str(member.role).lower():
                lead_member = member
            else:
                assist_members.append(member)
        if lead_member is None and assist_members:
            lead_member = assist_members[0]

        team_snapshot_json = json.dumps(accepted_snapshot, ensure_ascii=False)

        # (1) Specialists produce solution attempt
        for m in members:
            role = getattr(m, "role", "Unknown")
            model = getattr(m, "model_info", "Unknown")
            if state["fc_s"].get(role, False):
                continue

            delta_struct = state["delta"].get(role, {})
            delta_text = _json(delta_struct) if delta_struct else "(no prior peer feedback)"

            if first_turn:
                opinion_prompt = (
                    f"You are {role} in a math problem-solving team.\n"
                    f"Goal: {getattr(self, 'goal', '')}\n"
                    f"Problem: {getattr(self, 'question', '')}\n\n"
                    "Please think step by step from your expert perspective, and produce ONE integrated, concise solution message.\n"
                    "Include: brief reflection (1–2 sentences), core reasoning leading to the result, and the final answer (one line if known).\n"
                    "Keep it rigorous and self-contained; avoid decorative tags.\n"
                )
            else:
                opinion_prompt = (
                    f"You are {role} in a math team.\n"
                    f"Problem: {getattr(self, 'question', '')}\n\n"
                    f"Last accepted snapshot (may be empty):\n{team_snapshot_json}\n\n"
                    f"Peer feedback for YOU (JSON):\n{delta_text}\n\n"
                    "Please think step by step from your expert perspective, revise with ONE integrated, concise message addressing feedback (no step numbering).\n"
                    "State the refined reasoning and the final answer if applicable.\n"
                    "Be precise and minimal; no special tags.\n"
                )

            try:
                opinion_prompt = m.augment_prompt(opinion_prompt, top_k_per_query=5, max_hints=5)
                raw_opinion = m.chat(opinion_prompt)
                solution_outline = parse_solution_block(raw_opinion)
            except Exception as e:
                raw_opinion = ""
                solution_outline = ""
                interaction_log["steps"].append({
                    "role": role, "model": model, "phase": "opinion_error",
                    "prompt": opinion_prompt, "response": "", "error": str(e), "target": role
                })
            else:
                interaction_log["steps"].append({
                    "role": role, "model": model, "phase": "opinion",
                    "prompt": opinion_prompt, "response": raw_opinion,
                    "solution_outline": solution_outline, "target": role
                })

            state["last_opinion"][role] = solution_outline

            # (2) Peer reviews (structured JSON)
            pack_for_reviewers = _json({
                "target_role": role,
                "solution_outline": solution_outline if solution_outline else raw_opinion,
                "team_snapshot": accepted_snapshot,
                "problem": getattr(self, 'question', '')
            })

            all_issues = []
            all_validated = []
            verdicts = []
            analyses = []
            any_fatal_major = False

            for reviewer in members:
                r_role = getattr(reviewer, "role", "Unknown")
                if r_role == role:
                    continue

                peer_prompt = (
                    f"You are {r_role} reviewing the solution of {role}.\n\n"
                    f"Context bundle (JSON):\n{pack_for_reviewers}\n\n"
                    "REVIEW STRICTLY FROM YOUR OWN SPECIALTY PERSPECTIVE.\n"
                    "Anchor judgements to concrete mathematical content (definitions, identities, invariants, bounds)\n"
                    "Avoid generic praise/objections; be precise and discipline-specific.\n\n"
                    "DISCIPLINE:\n"
                    "- Keep 'analysis' to 3–4 concise sentences: start with grounded agreements, then targeted critiques.\n"
                    "- Put parts you believe correct in 'validated' as short phrases.\n"
                    "- For EACH critique, add an entry in 'issues' with type, severity, concrete note, and a minimal fix.\n"
                    "- Use 'tests' to note quick sanity checks (dimension counts, toy examples, boundary cases, etc.).\n"
                    "- Verdict rule: 'accept' only if there are NO issues at any severity (including minor) and the solution is essentially complete;\n"
                    "'revise' if ANY fixable issue remains (even 'minor'); 'reject' if there is a fatal flaw.\n"
                    "OUTPUT (RAW JSON ONLY):\n"
                    "{\n"
                    '  "analysis": "2-4 sentences overall appraisal, specialty-anchored: start with concrete agreements, then targeted critiques.",\n'
                    '  "verdict": "accept | revise | reject",\n'
                    '  "validated": ["sound manipulation of invariants", "boundary cases handled"],\n'
                    '  "issues": [\n'
                    '    {"type":"algebra_error|logic_gap|missing_case|theorem_misuse|undefined_symbol|diagram_dependence...,\n'
                    '     "severity":"fatal|major|minor","note":"Exactly what is wrong.",\n'
                    '     "fix":"Concrete minimal correction consistent with the current approach."}\n'
                    "  ],\n"
                    '  "tests": {"random_checks": 2, "counterexample": {"found": false, "example": null}},\n'
                    '  "alt_hint": "Optional short hint to guide a revision.",\n'
                    '  "confidence": 0.0\n'
                    "}\n"
                )

                try:
                    peer_prompt = reviewer.augment_prompt(peer_prompt, top_k_per_query=5, max_hints=5)
                    peer_raw = reviewer.temp_responses(peer_prompt)
                    parsed, ok = _safe_json_loads(peer_raw, {"analysis": "", "verdict": "", "validated": [], "issues": [], "tests": {}, "alt_hint": "", "confidence": 0})
                    if not ok or not isinstance(parsed, dict):
                        # attempt a simple rewriter pass with explicit schema
                        schema = """{
                                    "analysis": "...",
                                    "verdict": "accept | revise | reject",
                                    "validated": [],
                                    "issues": [{"type":"algebra_error|logic_gap|missing_case|theorem_misuse|undefined_symbol|diagram_dependence",
                                                "severity":"fatal|major|minor","note":"...", "fix":"..."}],
                                    "tests": {"random_checks": 0, "counterexample": {"found": false, "example": null}},
                                    "alt_hint": "", "confidence": 0
                                    }"""
                        rewriter = Agent(instruction="Rewrite content into VALID JSON for math peer review.", role='json rewriter', model_info="gpt-5", use_experience=False)
                        peer_raw = rewriter.temp_responses(
                            "Rewrite into VALID JSON matching EXACTLY this schema (no extra fields):\n"
                            f"```json\n{schema}\n```\n"
                            "Here is the content to fix:\n"
                            f"```\n{peer_raw}\n```"
                        )
                        parsed, ok = _safe_json_loads(peer_raw, {"analysis": "", "verdict": "", "validated": [], "issues": [], "tests": {}, "alt_hint": "", "confidence": 0})

                    analyses.append(str(parsed.get("analysis", "")))
                    verdicts.append(str(parsed.get("verdict", "")).lower())
                    vset = parsed.get("validated", []) or []
                    vset = [str(x) for x in vset]
                    all_validated.extend(vset)

                    issues = parsed.get("issues", []) or []
                    for it in issues:
                        try:
                            sev = str(it.get("severity", "")).lower()
                            if sev in ("fatal", "major"):
                                any_fatal_major = True
                        except Exception:
                            pass
                    all_issues.extend(issues)

                except Exception as e:
                    interaction_log["steps"].append({
                        "role": r_role, "phase": "peer_review_error",
                        "prompt": peer_prompt, "response": "{}", "error": str(e), "target": role
                    })
                else:
                    interaction_log["steps"].append({
                        "role": r_role, "model": reviewer.model_info, "phase": "peer_review",
                        "prompt": peer_prompt, "response": json.dumps(parsed, ensure_ascii=False),
                        "target": role
                    })

            # aggregate Δ for next round
            aggregated_delta = {
                "summary": " | ".join([a for a in analyses if a])[:2000],
                "verdicts": verdicts,
                "validated": _dedup_keep_order(all_validated),
                "issues": all_issues
            }
            state["delta"][role] = aggregated_delta

            # acceptance rule
            has_any_issue = bool(all_issues)
            if (not any_fatal_major) and (not has_any_issue) and verdicts and all(v == "accept" for v in verdicts):
                state["fc_s"][role] = True
                round_O[role] = {
                    "owner": role,
                    "solution_outline": solution_outline,
                    "validated_steps": aggregated_delta["validated"],
                    "notes": "Accepted by peer consensus (no issues reported)."
                }

        # round end
        state["O_rounds"].append({"r": r, "accepted": round_O})
        state["r"] = r + 1
        state["fc_global"] = all(state["fc_s"].values()) if state["fc_s"] else True

        round_summary = f"[Round {r}] accepted={list(round_O.keys())}; converged={state['fc_global']}"
        interaction_log["steps"].append({
            "role": "SYSTEM", "phase": "round_summary", "prompt": "", "response": round_summary
        })
        interaction_log["final_response"] = round_summary

        return round_summary, interaction_log

# ----------------------------
# Specialist pool (Math)
# ----------------------------
MATH_POOL = [
    "Problem Parser", "Strategy Planner",
    "Algebra", "Number Theory", "Combinatorics", "Geometry",
    "Calculus/Analysis", "Linear Algebra", "Probability",
    "Functional Equations", "Inequalities", "Graph Theory",
    "Computation Verifier", "Counterexample Hunter", "Proof Formalizer"
]

def _extract_json_list(text: str):
    text = _strip_fenced(text)
    try:
        obj = json.loads(text)
        if isinstance(obj, list): return obj, True
        if isinstance(obj, dict): return [obj], True
    except Exception:
        # rough fallback: find the first [...] block
        m = re.search(r"\[\s*\{.*\}\s*\]", text, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, list): return obj, True
            except Exception:
                pass
    return [], False

def _pick_rare_specialists(tmp_agent, problem: str, n: int) -> List[Dict]:
    pool_str = ", ".join(MATH_POOL)
    prompt = (
        f"You are the Chief Problem Solving Officer assembling a small math team.\n\n"
        f"Problem: {problem}\n\n"
        f"From this specialist pool:\n{pool_str}\n\n"
        f"Pick no more than {n} distinct specialties best suited for this problem. "
        "Exactly ONE item must have role 'leader'.\n\n"
        "Return ONLY a valid JSON array. Each item must be:\n"
        "{\n"
        '  "specialty": "<one from the pool>",\n'
        '  "role": "<short role name, exactly one item has \'leader\'>",\n'
        '  "description": "<2-4 sentences on how this specialist will reason for THIS problem: key invariants, typical tactics, how to communicate>"\n'
        "}\n"
    )
    raw = tmp_agent.chat(prompt)
    items, ok = _extract_json_list(raw)

    if not ok:
        # force rewriting to strict JSON if needed, feed the model output
        rewriter = Agent(instruction="Rewrite the following into a valid JSON array exactly, do not add/remove fields.", role='json rewriter', model_info="gpt-5", use_experience=False)
        raw = rewriter.temp_responses(raw)
        items, ok = _extract_json_list(raw)

    seen = set()
    cleaned = []
    for it in items:
        spec = str(it.get("specialty", "")).strip()
        if spec in MATH_POOL and spec not in seen:
            cleaned.append({
                "specialty": spec,
                "role": it.get("role", spec),
                "expertise_description": it.get("description", ""),
                "system_prompt": it.get("system_prompt", f"You are a {spec} contributing to this problem.")
            })
            seen.add(spec)
        if len(cleaned) >= n:
            break

    if len(cleaned) == 0:
        # minimal fallback
        cleaned = [
            {"specialty": "Problem Parser", "role": "leader", "expertise_description": "Parse the problem, define variables, constraints, and targets."},
            {"specialty": "Algebra", "role": "member", "expertise_description": "Perform algebraic manipulations and identities."},
            {"specialty": "Computation Verifier", "role": "member", "expertise_description": "Double-check calculations and simple enumerations."}
        ][:n]

    # ensure exactly one leader
    leaders = [m for m in cleaned if str(m.get("role", "")).lower() == "leader"]
    if not leaders and cleaned:
        cleaned[0]["role"] = "leader"
        leaders = [cleaned[0]]
    if len(leaders) > 1:
        first = leaders[0]
        for m in cleaned:
            if m is not first and str(m.get("role", "")).lower() == "leader":
                m["role"] = "member"
    return cleaned[:n]

# 新增：自由招募（无预设池）
def _pick_free_specialists(tmp_agent, problem: str, n: int) -> List[Dict]:
    prompt = (
        f"You are the Chief Problem Solving Officer assembling a small math team.\n\n"
        f"Problem: {problem}\n\n"
        f"Design up to {n} distinct specialties best suited for this problem. \n"
        "You are NOT restricted to any preset pool. Create precise specialist titles that reflect reasoning roles (e.g., invariant designer, structure normalizer, edge-case auditor).\n"
        "Exactly ONE item must have role 'leader'.\n\n"
        "Return ONLY a valid JSON array. Each item must be:\n"
        "{\n"
        '  "specialty": "<your created specialist title>",\n'
        '  "role": "<short role name, exactly one item has \'leader\'>",\n'
        '  "description": "<2-4 sentences on how this specialist will reason for THIS problem: key invariants, typical tactics, how to communicate>"\n'
        "}\n"
    )
    raw = tmp_agent.chat(prompt)
    items, ok = _extract_json_list(raw)

    if not ok:
        rewriter = Agent(instruction="Rewrite the following into a valid JSON array exactly, do not add/remove fields.", role='json rewriter', model_info="gpt-5", use_experience=False)
        raw = rewriter.temp_responses(raw)
        items, ok = _extract_json_list(raw)

    seen = set()
    cleaned = []
    for it in items:
        spec = str(it.get("specialty", "")).strip()
        if not spec or spec in seen:
            continue
        cleaned.append({
            "specialty": spec,
            "role": it.get("role", spec),
            "expertise_description": it.get("description", ""),
            "system_prompt": it.get("system_prompt", f"You are a {spec} contributing to this problem.")
        })
        seen.add(spec)
        if len(cleaned) >= n:
            break

    if len(cleaned) == 0:
        cleaned = [
            {"specialty": "Strategy Architect", "role": "leader", "expertise_description": "Orchestrates the solution plan and validates milestone invariants."},
            {"specialty": "Algebraic Surgeon", "role": "member", "expertise_description": "Performs safe transformations with precondition checks (invertibility, domains)."},
            {"specialty": "Edge-Case Auditor", "role": "member", "expertise_description": "Surfaces boundary/degenerate scenarios to prevent overgeneral claims."}
        ][:n]

    # ensure exactly one leader
    leaders = [m for m in cleaned if str(m.get("role", "")).lower() == "leader"]
    if not leaders and cleaned:
        cleaned[0]["role"] = "leader"
        leaders = [cleaned[0]]
    if len(leaders) > 1:
        first = leaders[0]
        for m in cleaned:
            if m is not first and str(m.get("role", "")).lower() == "leader":
                m["role"] = "member"
    return cleaned[:n]

# ----------------------------
# Final Chair prompt (Math)
# ----------------------------
def build_mdt_prompt_math(assessment_report: str, problem: str) -> str:
    return (
        "You are the chair of a math MDT. Integrate accepted outlines and produce a concise, rigorous write-up.\n\n"
        f"Snapshot summary (accepted/unresolved):\n{assessment_report}\n\n"
        f"Problem:\n{problem}\n\n"
        "OUTPUT FORMAT (STRICT):\n"
        "<analysis>\n"
        "- Key ideas and why this route is effective.\n"
        "- Note any edge cases handled.\n"
        "</analysis>\n\n"
        "<final_answer>\n"
        "- Single line with the final numeric/symbolic answer; if unknown, write 'N/A'.\n"
        "</final_answer>\n\n"
        "<formal_proof>\n"
        "Provide a clean, step-by-step solution/proof.\n"
        "</formal_proof>\n"
        "No extra text outside these tags."
    )

# ----------------------------
# Data helpers (load & question)
# ----------------------------
def load_data(dataset):
    test_qa = []
    examplers = []
    test_path = f'./data/{dataset}/test.jsonl'
    if os.path.exists(test_path):
        with open(test_path, 'r', encoding="utf-8") as f:
            for line in f:
                test_qa.append(json.loads(line))
    train_path = f'./data/{dataset}/train.jsonl'
    if os.path.exists(train_path):
        with open(train_path, 'r', encoding="utf-8") as f:
            for line in f:
                examplers.append(json.loads(line))
    return test_qa, examplers

def create_question(sample, dataset, few_shot="none"):
    """
    For math datasets, expect fields like:
    - sample['problem']       (core problem statement)
    - sample['none'] / 'random' / 'dynamic' (optional few-shot variants text)
    """
    if dataset.lower().startswith("math"):
        if few_shot in sample:
            return sample[few_shot], None
        return sample.get('problem', ''), None
    else:
        if few_shot in sample:
            return sample[few_shot], None
        return sample.get('problem', sample.get('question', '')), None

# ----------------------------
# Top-level pipeline (RARE-like for math)
# ----------------------------
def process_rareagent(sample, question, model, args, max_turn: int = 6, num_members: int = 5,
                      result_file: str = 'result.json', interaction_history_file: str = 'interaction_history.json'):
    """
    Math RARE pipeline:
    1) Recruit specialists (<= num_members).
    2) Multi-round rare_interact until convergence or max_turn.
    3) Chair composes final <analysis>/<final_answer>/<formal_proof>.
    """
    # single vs multi expert models
    if getattr(args, "multi_expert", False):
        model = ["gemini-2.5-pro", "gpt-5"]
    else:
        model = [model]

    print("[STEP 1] Team Recruitment")
    recruit_prompt = (
        "You are an experienced math expert. Build ONE multidisciplinary math team "
        "tailored to the problem. Do not solve the problem yet."
    )
    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=random.choice(model), port=args.port, use_experience=getattr(args, "use_experience", True), unified_experience=getattr(args, "unified_experience", False))

    if getattr(args, "free_recruitment", False):
        print("[Recruitment] Free mode: recruiter designs specialties without preset pool.")
        members = _pick_free_specialists(tmp_agent, question, num_members)
    else:
        members = _pick_rare_specialists(tmp_agent, question, num_members)
    print(f"Recruited team members: {members}")

    group_goal = "Math MDT – integrate specialist solutions and converge to a rigorous solution."
    group_instance = Group(group_goal, members, model, question, args.port, use_experience=getattr(args, "use_experience", True), unified_experience=getattr(args, "unified_experience", False))

    interaction_history: Dict[str, Any] = {group_instance.goal: {}}
    assessments: List[str] = []

    print("[STEP 2] Multi-turn reviews")
    for i in range(max_turn):
        if i == 0:
            assessment, assessment_log = group_instance.rare_interact(comm_type='internal', first_turn=True)
        else:
            assessment, assessment_log = group_instance.rare_interact(comm_type='internal')
        assessments.append(assessment)
        interaction_history[group_instance.goal][f"turn_{i}"] = assessment_log
        print(f"[Turn {i+1}] {assessment}...")
        if getattr(group_instance, "_state", {}).get("fc_global"):
            print("[MDT] Converged early.")
            break

    # Summarize snapshot
    if hasattr(group_instance, "_state"):
        O_rounds = group_instance._state.get("O_rounds", [])
        last_ops = group_instance._state.get("last_opinion", {}) or {}
        final_brief = []
        accepted_cnt = 0
        accepted_roles = set()

        for item in O_rounds:
            rr = item["r"]
            accepted = item.get("accepted", {})
            if accepted:
                accepted_cnt += len(accepted)
                accepted_roles.update(accepted.keys())
                final_brief.append(f"Round {rr} accepted: " + json.dumps(accepted, ensure_ascii=False))

        if accepted_cnt == 0:
            if last_ops:
                compact = {role: str(op) for role, op in last_ops.items()}
                assessment_report = "No consensus reached. Using latest specialist outlines:\n" + json.dumps(compact, ensure_ascii=False)
            else:
                assessment_report = "No consensus; no specialist outline available."
        else:
            unresolved = {}
            if last_ops:
                for role, op in last_ops.items():
                    if role not in accepted_roles:
                        unresolved[role] = str(op)
            if unresolved:
                final_brief.append("Unresolved (latest outlines): " + json.dumps(unresolved, ensure_ascii=False))
            assessment_report = "\n".join(final_brief)
    else:
        assessment_report = "\n".join(f"Turn {i+1}: {t}" for i, t in enumerate(assessments))

    # Final decision
    print("[STEP 3] Final Decision")
    decision_prompt = (
        "You are an experienced chair mathematician. Given accepted outlines, produce the final answer and proof."
    )
    summary_model = model if not getattr(args, "o3_summary", False) else ["o3-2025-04-16"]
    decider = Agent(instruction=decision_prompt, role='decision maker', model_info=random.choice(summary_model), port=args.port, use_experience=getattr(args, "use_experience", True), unified_experience=getattr(args, "unified_experience", False))
    decider.chat(decision_prompt)

    prompt = build_mdt_prompt_math(assessment_report=assessment_report, problem=question)
    final_decision = decider.temp_responses(prompt, img_path=None)
    analysis, final_answer, formal_proof = parse_math_final(final_decision)

    if final_answer == "" and formal_proof == "":
        # attempt rewrite to target format if needed
        rewriter = Agent(instruction=decision_prompt, role='decision rewriter', model_info='gpt-5', port=args.port, use_experience=False)
        rewrite_prompt = (
            "Rewrite the following into STRICT format <analysis>/<final_answer>/<formal_proof> (do not change math content):\n"
            f"{final_decision}\n"
            "<analysis>...</analysis>\n"
            "<final_answer>...</final_answer>\n"
            "<formal_proof>...</formal_proof>"
        )
        final_decision_rewrited = rewriter.temp_responses(rewrite_prompt, img_path=None)
        analysis, final_answer, formal_proof = parse_math_final(final_decision_rewrited)

    # Token metrics
    token_cost_metric = {
        agent.role: {
            'prompt_tokens_sum': sum(agent.prompt_tokens) if agent.prompt_tokens else 0,
            'completion_tokens_sum': sum(agent.completion_tokens) if agent.completion_tokens else 0,
            'total_tokens_sum': sum(agent.total_tokens) if agent.total_tokens else 0,
            'prompt_tokens_max': max(agent.prompt_tokens) if agent.prompt_tokens else 0,
            'completion_tokens_max': max(agent.completion_tokens) if agent.completion_tokens else 0,
            'total_tokens_max': max(agent.total_tokens) if agent.total_tokens else 0,
            'prompt_tokens': agent.prompt_tokens,
            'completion_tokens': agent.completion_tokens,
            'total_tokens': agent.total_tokens
        } for agent in [tmp_agent] + group_instance.members + [decider]
    }

    final_decision_history = {
        'model': decider.model_info,
        'prompt': prompt,
        'analysis': analysis,
        'golden_answer': sample.get('golden_answer', None),
        'final_answer': final_answer,
        'formal_proof': formal_proof,
        'raw_decision': final_decision,
        'token_cost_metric': token_cost_metric
    }

    interaction_history['final_decision'] = final_decision_history
    print(f"Final Decision: {final_answer}")

    # build result record
    result = {
        'problem': question,
        'problem_idx': sample.get('problem_idx', None),
        'golden_answer': sample.get('golden_answer', None),
        'predict_answer': final_answer,
        'proof': formal_proof
    }

    # write files
    with open(result_file, 'w', encoding='utf-8-sig') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    with open(interaction_history_file, 'w', encoding='utf-8-sig') as f:
        json.dump(interaction_history, f, indent=4, ensure_ascii=False)

    cprint(f"saved to {result_file}", 'yellow')
    return final_answer, interaction_history
