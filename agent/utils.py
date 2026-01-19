from datetime import datetime
import os
import json
import random
from tqdm import tqdm
from prettytable import PrettyTable 
from termcolor import cprint
from pptree import Node
from google import genai
from google.genai import types
from openai import OpenAI
from pptree import *
from typing import List, Dict, Any, Optional

from typing import Any, Dict, List, Optional
from datetime import datetime
import json

# -----------------------------
# Public-safe config helpers
# -----------------------------
def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return default
    v = str(v).strip()
    return v if v else default

def _require_env(name: str) -> str:
    v = _env(name)
    if not v:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set it before running. See ../env.example."
        )
    return v

def _pick_gemini_key() -> str:
    keys = _env("GEMINI_API_KEYS")
    if keys:
        pool = [k.strip() for k in keys.split(",") if k.strip()]
        if pool:
            return random.choice(pool)
    # backward-compat: some older scripts used genai_api_key
    return _require_env("GEMINI_API_KEY") if _env("GEMINI_API_KEY") else _require_env("genai_api_key")

# 若你已有这些工具函数，可删除本段重复定义
def _now_ts() -> str:
    return datetime.utcnow().isoformat()

def _truncate(s: str, n: int) -> str:
    if not isinstance(s, str):
        s = str(s)
    return s[:n]

def _safe_json_loads(s: str, default: Any) -> Any:
    try:
        obj = json.loads(s)
        if isinstance(default, dict) and not isinstance(obj, dict):
            return default
        if isinstance(default, list) and not isinstance(obj, list):
            return default
        return obj
    except Exception:
        return default

def _dedup_keep_order(items: List[str]) -> List[str]:
    """仅做去重保序与长度截断（非字符串语义过滤）。"""
    seen = set()
    out = []
    for it in items:
        if not it:
            continue
        it = str(it).strip()
        if not it:
            continue
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


class Agent:
    def __init__(self, instruction, role, examplers=None, model_info='gpt-5', port=8000, img_path=None):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path

        if self.model_info == 'gemini-pro' or "gemini" in self.model_info:
            self.api_key = _pick_gemini_key()
            # genai.configure(api_key=self.api_key, transport='rest')
            # self.model = genai.GenerativeModel(self.model_info)
            # self._chat = self.model.start_chat(history=[])
            self.client = genai.Client(api_key=self.api_key)
            self._chat = self.client.chats.create(model=self.model_info)

        elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini', 'o3-2025-04-16', 'gpt-5', 'gpt-4.1-2025-04-14']:
            self.client = OpenAI(
                api_key=_require_env("OPENAI_API_KEY"),
                base_url=_env("OPENAI_BASE_URL"),
            )
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})
        elif 'qwen3-235b-a22b-instruct-2507' in self.model_info:
            self.client = OpenAI(
                            base_url=_env("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                            api_key=_require_env("DASHSCOPE_API_KEY"),
                            )
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})
        else:
            self.client = OpenAI(api_key='EMPTY', base_url=f"http://localhost:{port}/v1")
            self.messages = [
                {"role": "system", "content": instruction},
            ]
            if examplers is not None:
                for exampler in examplers:
                    self.messages.append({"role": "user", "content": exampler['question']})
                    self.messages.append({"role": "assistant", "content": exampler['answer'] + "\n\n" + exampler['reason']})            

    def chat(self, message, img_path=None, chat_mode=True, temperature=0.3):
        if self.model_info == 'gemini-pro' or "gemini" in self.model_info:
            for _ in range(10):
                try:
                    response = self._chat.send_message(message)
                    responses = response.text
                    return responses
                except Exception as e:
                    cprint(f"[Api-Key {self.api_key}] failed to get response from Gemini, {e}", 'red')
                    old_api_key = self.api_key
                    while self.api_key == old_api_key:
                        self.api_key = random.choice(self.api_key_list)
                    self.client = genai.Client(api_key=self.api_key)
                    self._chat = self.client.chats.create(model=self.model_info)
                    continue
            return "Error: Failed to get response from Gemini."

        # elif self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini', 'o3-2025-04-16']:
        else:
            self.messages.append({"role": "user", "content": message})
            
            if self.model_info == 'gpt-3.5':
                model_name = "gpt-3.5-turbo"
            else:
                model_name = self.model_info
            
            if 'Qwen3' in model_name:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=self.messages,
                    temperature=temperature,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
            elif 'o3' in model_name or 'gpt-4.1-2025-04-14' in model_name:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=self.messages,
                )
            elif 'gpt-5' in model_name:
                resp = self.client.chat.completions.create(
                    model="gpt-5",                    # gpt-5-mini / nano 亦可
                    messages=self.messages,
                    reasoning_effort="low",  # 可选：最快响应
                    verbosity="low"         # 可选：输出更简洁
                )
                self.messages.append({"role": "assistant", "content": resp.choices[0].message.content})
                return resp.choices[0].message.content
            else:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=self.messages,
                    temperature=temperature,
                )

            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            return response.choices[0].message.content

    def temp_responses(self, message, img_path=None):
        if self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini', 'o3-2025-04-16']:      
            self.messages.append({"role": "user", "content": message})
            
            temperatures = [0.7]
            
            responses = {}
            for temperature in temperatures:
                if self.model_info == 'gpt-3.5':
                    model_info = 'gpt-3.5-turbo'
                else:
                    model_info = self.model_info
                
                if 'Qwen3' in model_info:
                    response = self.client.chat.completions.create(
                        model=model_info,
                        messages=self.messages,
                        temperature=temperature,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                elif 'o3' in model_info or 'gpt-4.1-2025-04-14' in model_info:
                    response = self.client.chat.completions.create(
                        model=model_info,
                        messages=self.messages,
                    )
                elif 'gpt-5' in model_info:
                    resp = self.client.chat.completions.create(
                        model="gpt-5",                    # gpt-5-mini / nano 亦可
                        messages=self.messages,
                        reasoning_effort="low",  # 可选：最快响应
                        verbosity="low"         # 可选：输出更简洁
                    )
                    self.messages.append({"role": "assistant", "content": resp.choices[0].message.content})
                    responses[temperature] = resp.choices[0].message.content
                    return resp.choices[0].message.content
                else:
                    response = self.client.chat.completions.create(
                        model=model_info,
                        messages=self.messages,
                        temperature=temperature,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )

                responses[temperature] = response.choices[0].message.content
                
            return response.choices[0].message.content
        
        elif self.model_info == 'gemini-pro' or "gemini" in self.model_info:
            for _ in range(10):
                try:
                    response = self._chat.send_message(message)
                    responses = response.text
                    return responses
                except Exception as e:
                    cprint(f"[Api-Key {self.api_key}] failed to get response from Gemini, {e}", 'red')
                    old_api_key = self.api_key
                    while self.api_key == old_api_key:
                        self.api_key = random.choice(self.api_key_list)
                    self.client = genai.Client(api_key=self.api_key)
                    self._chat = self.client.chats.create(model=self.model_info)
                    continue
            return "Error: Failed to get response from Gemini."
        else:
            self.messages.append({"role": "user", "content": message})
            
            temperatures = [0.7]
            
            responses = {}
            for temperature in temperatures:
                if self.model_info == 'gpt-3.5':
                    model_info = 'gpt-3.5-turbo'
                else:
                    model_info = self.model_info
                
                if 'gpt-5' in model_info:
                    response = self.client.chat.completions.create(
                        model="gpt-5",                    # gpt-5-mini / nano 亦可
                        messages=self.messages,
                        reasoning_effort="low",  # 可选：最快响应
                        verbosity="low"         # 可选：输出更简洁
                    )
                    responses[temperature] = response.choices[0].message.content
                elif 'Qwen3' not in model_info:
                    response = self.client.chat.completions.create(
                        model=model_info,
                        messages=self.messages,
                        temperature=temperature,
                    )
                    responses[temperature] = response.choices[0].message.content
                else:
                    response = self.client.chat.completions.create(
                        model=model_info,
                        messages=self.messages,
                        temperature=temperature,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    
                    responses[temperature] = response.choices[0].message.content
                
            return response.choices[0].message.content

class Group:
    def __init__(self, goal, members, model, question, port=8000, examplers=None):
        self.goal = goal
        self.members = []
        for member_info in members:
            _agent = Agent('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()), role=member_info['role'], model_info=random.choice(model), port=port)
            _agent.chat('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()))
            self.members.append(_agent)
        self.question = question
        self.examplers = examplers
        self._state = None

    def interact(self, comm_type, message=None, img_path=None):
        # 用于保存交互记录
        interaction_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "comm_type": comm_type,
            "goal": getattr(self, "goal", None),
            "question": getattr(self, "question", None),
            "members": [m.role for m in getattr(self, "members", [])],
            "steps": [],
            "final_response": None
        }

        if comm_type == 'internal':
            lead_member = None
            assist_members = []
            for member in self.members:
                if 'lead' in member.role.lower():
                    lead_member = member
                else:
                    assist_members.append(member)

            if lead_member is None:
                lead_member = assist_members[0]

            delivery_prompt = (
                f"You are the lead of the medical group which aims to {self.goal}. "
                f"You have the following assistant clinicians who work for you:\n"
                + "\n".join(a_mem.role for a_mem in assist_members)
                + f"\n\nNow, given the medical query, provide a short answer to what kind investigations are needed from each assistant clinicians.\nQuestion: {self.question}"
            )

            try:
                delivery = lead_member.chat(delivery_prompt)
            except Exception as e:
                delivery = assist_members[0].chat(delivery_prompt)
                interaction_log["error"] = str(e)

            interaction_log["steps"].append({
                "role": lead_member.role,
                "prompt": delivery_prompt,
                "response": delivery
            })

            investigations = []
            for a_mem in assist_members:
                inv_prompt = (
                    f"You are in a medical group where the goal is to {self.goal}. "
                    f"Your group lead is asking for the following investigations:\n{delivery}\n\n"
                    f"Please remind your expertise and return your investigation summary that contains the core information."
                )
                investigation = a_mem.chat(inv_prompt)
                investigations.append([a_mem.role, investigation])
                interaction_log["steps"].append({
                    "role": a_mem.role,
                    "prompt": inv_prompt,
                    "response": investigation
                })

            gathered_investigation = "".join(
                f"[{role}]\n{inv}\n" for role, inv in investigations
            )

            if self.examplers is not None:
                investigation_prompt = (
                    f"The gathered investigation from your assistant clinicians is as follows:\n"
                    f"{gathered_investigation}\n\n"
                    f"Now, after reviewing the following example cases, return your answer to the medical query among the option provided:\n\n"
                    f"{self.examplers}\nQuestion: {self.question}"
                )
            else:
                investigation_prompt = (
                    f"The gathered investigation from your assistant clinicians is as follows:\n"
                    f"{gathered_investigation}\n\n"
                    f"Now, return your answer to the medical query among the option provided.\n\n"
                    f"Question: {self.question}"
                )

            response = lead_member.chat(investigation_prompt)
            interaction_log["steps"].append({
                "role": lead_member.role,
                "prompt": investigation_prompt,
                "response": response
            })

            interaction_log["final_response"] = response
            return response, interaction_log

        elif comm_type == 'external':
            # external 情况：直接保存传入的 message / img_path
            interaction_log["message"] = message
            interaction_log["img_path"] = img_path
            interaction_log["final_response"] = None  # 如果有实际处理可替换
            return "External interaction logged.", interaction_log

    def rare_interact(
        self,
        comm_type: str,
        message: Optional[str] = None,
        img_path: Optional[str] = None
    ):
        """
        MDT（结论版，易收敛；无字符串过滤）
        - 专家输出：Reflection（2–3 句） + 3–5 条编号诊断（每条 1–2 句理由） + 一行 Conclusion（高→低）
        * 不引入新检查/治疗/外部事实；仅基于 Goal/Question + Last snapshot + Peer feedback。
        - 同行审阅：analysis（1–3句） + agreements（≤3） + disagreements（≤3，若无异议用 null）
        - 下一轮把 agreements 和 disagreements 一并作为 Δ 喂回；
        当 **没有异议** 时，将 disagreements 记为 None，并视为该专家收敛。
        """
        interaction_log: Dict[str, Any] = {
            "timestamp": _now_ts(),
            "comm_type": comm_type,
            "goal": getattr(self, "goal", None),
            "question": getattr(self, "question", None),
            "members": [f"{getattr(m, "role", "Unknown")}_{getattr(m, "model_info", "Unknown")}" for m in getattr(self, "members", [])],
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

        # 初始化持久状态
        if not hasattr(self, "_state") or not isinstance(self._state, dict) or not self._state:
            roles = [getattr(m, "role", "Unknown") for m in members]
            self._state = {
                "r": 0,
                "fc_global": False,
                "fc_s": {role: False for role in roles},
                # 注意 disagreements 初始为 None（表示当前无异议）
                "delta": {role: {"agreements": [], "disagreements": None} for role in roles},
                "O_rounds": [],
                "last_opinion": {}  # role -> raw text
            }

        state: Dict[str, Any] = self._state
        r: int = state["r"]

        interaction_log["steps"].append({
            "role": "SYSTEM", "phase": "round_start",
            "prompt": f"Round {r} start", "response": ""
        })

        accepted_snapshot: Dict[str, Any] = state["O_rounds"][-1].get("accepted", {}) if state["O_rounds"] else {}
        round_O: Dict[str, Any] = {}

        # 辅助：把列表/None渲染成提示文本
        def _bullet(items: Optional[List[str]]) -> str:
            if items is None:
                return "(None)"
            return "\n".join(f"- {x}" for x in items) if items else "(empty)"

        # === 逐专家：意见 → 同行 → Δ（agreements + disagreements 或 None）===
        for m in members:
            role = getattr(m, "role", "Unknown")
            model = getattr(m, "model_info", "Unknown")
            if state["fc_s"].get(role, False):
                continue

            cur_delta: Dict[str, Optional[List[str]]] = state["delta"].get(role, {"agreements": [], "disagreements": None})
            ag_list: List[str] = (cur_delta.get("agreements") or [])
            dg_list_opt: Optional[List[str]] = cur_delta.get("disagreements")  # 可能为 None

            delta_text = (
                f"Peer agreements to consider:\n{_bullet(ag_list)}\n\n"
                f"Peer disagreements to address:\n{_bullet(dg_list_opt)}"
            )

            team_snapshot_json = json.dumps(accepted_snapshot, ensure_ascii=False)

            # (1) 专家个人意见：自由文本（无 JSON）
            opinion_prompt = (
                f"You are {role} in an MDT.\n"
                f"Goal: {getattr(self, 'goal', '')}\n"
                f"Clinical question: {getattr(self, 'question', '')}\n\n"
                f"Last accepted snapshot (may be empty):\n{team_snapshot_json}\n\n"
                f"Peer feedback (delta):\n{delta_text}\n\n"
                "TASK:\n"
                "- Update YOUR opinion using ONLY the goal/question, last snapshot, and peer feedback.\n"
                "- Do NOT introduce tests, treatments, or any new facts.\n"
                "- Keep to reasoning and diagnosis ranking only.\n\n"
                "OUTPUT (no extra text, no code fences):\n"
                "1) Reflection (2–3 sentences): briefly state how the question and peer feedback refine your view.\n"
                "2) 3–5 numbered diagnoses, each with a 1–2 sentence rationale tied to given evidence.\n"
                "3) One-line Conclusion ranking diagnoses from high → low.\n\n"
                "[reflecting on how the case details and peer feedback refine your diagnostic reasoning and ranking.]\n\n"
                "1. **[Diagnosis 1]**: [1–2 sentence rationale].\n"
                "2. **[Diagnosis 2]**: [1–2 sentence rationale].\n"
                "3. **[Diagnosis 3]**: [1–2 sentence rationale].\n"
                "4. **[Diagnosis 4]**: [optional, 1–2 sentence rationale].\n"
                "5. **[Diagnosis 5]**: [optional, 1–2 sentence rationale].\n\n"
                "Conclusion: [Dx1], [Dx2], [Dx3][, Dx4][, Dx5] (high → low).\n\n"
                "Start your response:"
            )
            try:
                raw_opinion = m.chat(opinion_prompt)
            except Exception as e:
                raw_opinion = ""
                interaction_log["steps"].append({
                    "role": role, "model": model, "phase": "opinion_error",
                    "prompt": opinion_prompt, "response": "", "error": str(e), "target": role
                })
            else:
                interaction_log["steps"].append({
                    "role": role, "model": model, "phase": "opinion",
                    "prompt": opinion_prompt, "response": raw_opinion, "target": role
                })

            # 存储专家原始意见文本
            state["last_opinion"][role] = raw_opinion

            # (2) 同行审阅：输入为“专家意见全文 + 团队快照”，输出结构化 JSON
            pack_for_reviewers = json.dumps(
                {"target_role": role, "opinion_text": raw_opinion, "team_snapshot": accepted_snapshot},
                ensure_ascii=False
            )

            collected_agreements: List[str] = []
            collected_disagreements: List[str] = []

            for reviewer in members:
                r_role = getattr(reviewer, "role", "Unknown")
                if r_role == role:
                    continue

                peer_prompt = (
                    f"You are {r_role} in an MDT. Review {role}'s latest opinion and the team snapshot below:\n"
                    f"{pack_for_reviewers}\n\n"
                    "Give structured peer feedback. Do NOT request tests or treatments.\n"
                    "Return ONLY valid JSON matching this schema:\n"
                    "{\n"
                    '  "analysis": "<As a peer, briefly comment on their reasoning. Clearly mention the main points you agree with, followed by any points you disagree with or think need clarification.>",\n'
                    '  "agreements": ["<point you agree with>", "..."],\n'
                    '  "disagreements": null | ["<brief point you disagree with and why>", "..."]\n'
                    "}\n"
                    "Rules:\n"
                    "- If you have NO objections (consensus), set disagreements to null (not an empty list).\n"
                    "- Do not include extra keys or any text outside the JSON.\n"
                    "- Do not start with ```json or any other code fence.\n"
                )
                try:
                    peer_raw = reviewer.chat(peer_prompt)
                    if reviewer.model_info.startswith("gemini"):
                        rewriter_prompt = (
                    f"Give structured peer feedback {peer_raw}.\n"
                    "Return ONLY valid JSON matching this schema:\n"
                    "{\n"
                    '  "analysis": "<As a peer, briefly comment on their reasoning. Clearly mention the main points you agree with, followed by any points you disagree with or think need clarification.>",\n'
                    '  "agreements": ["<point you agree with>", "..."],\n'
                    '  "disagreements": null | ["<brief point you disagree with and why>", "..."]\n'
                    "}\n"
                    "Rules:\n"
                    "- If you have NO objections (consensus), set disagreements to null (not an empty list).\n"
                    "- Do not include extra keys or any text outside the JSON.\n"
                    "- Do not start with ```json or any other code fence.\n"
                )
                        rewriter = Agent(instruction="You are a professional json rewriter. You need to rewrite a raw json to a standard JSON format", role='decision maker', model_info="o3-2025-04-16")
                        peer_raw = rewriter.temp_responses(rewriter_prompt)
                    parsed = _safe_json_loads(peer_raw, {"analysis": "", "agreements": [], "disagreements": []})
                    if isinstance(parsed.get("agreements"), list):
                        collected_agreements.extend([str(x) for x in parsed["agreements"]])
                    if isinstance(parsed.get("disagreements"), list):
                        collected_disagreements.extend([str(x) for x in parsed["disagreements"]])
                except Exception as e:
                    peer_raw = "{}"
                    interaction_log["steps"].append({
                        "role": r_role, "phase": "peer_review_error",
                        "prompt": peer_prompt, "response": "{}", "error": str(e), "target": role
                    })
                else:
                    interaction_log["steps"].append({
                        "role": r_role, "model":reviewer.model_info, "phase": "peer_review",
                        "prompt": peer_prompt, "response": peer_raw, "agreements": collected_agreements, "disagreements": collected_disagreements, "target": role
                    })

            # (3) 形成下一轮 Δ
            next_agreements = _dedup_keep_order(collected_agreements)
            next_disagreements_list = _dedup_keep_order(collected_disagreements)
            disagreements_field: Optional[List[str]] = next_disagreements_list if next_disagreements_list else None

            state["delta"][role] = {
                "agreements": next_agreements,
                "disagreements": disagreements_field
            }

            # **无异议（disagreements 为 None）即收敛**：接受该专家意见（存全文）
            if disagreements_field is None:
                state["fc_s"][role] = True
                round_O[role] = raw_opinion

        # 轮末：记录推进
        state["O_rounds"].append({"r": r, "accepted": round_O})
        state["r"] = r + 1
        state["fc_global"] = all(state["fc_s"].values()) if state["fc_s"] else True

        round_summary = f"[Round {r}] accepted={list(round_O.keys())}; converged={state['fc_global']}"
        interaction_log["steps"].append({
            "role": "SYSTEM", "phase": "round_summary",
            "prompt": "", "response": round_summary
        })
        interaction_log["final_response"] = round_summary

        return round_summary, interaction_log


def parse_hierarchy(info, emojis):
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]

    count = 0
    for expert, hierarchy in info:
        try:
            expert = expert.split('-')[0].split('.')[1].strip()
        except:
            expert = expert.split('-')[0].strip()
        
        if hierarchy is None:
            hierarchy = 'Independent'
        
        if 'independent' not in hierarchy.lower():
            parent = hierarchy.split(">")[0].strip()
            child = hierarchy.split(">")[1].strip()

            for agent in agents:
                if agent.name.split("(")[0].strip().lower() == parent.strip().lower():
                    child_agent = Node("{} ({})".format(child, emojis[count]), agent)
                    agents.append(child_agent)

        else:
            agent = Node("{} ({})".format(expert, emojis[count]), moderator)
            agents.append(agent)

        count += 1

    return agents

def parse_group_info(group_info):
    lines = group_info.split('\n')
    
    parsed_info = {
        'group_goal': '',
        'members': []
    }

    parsed_info['group_goal'] = "".join(lines[0].split('-')[1:])
    
    for line in lines[1:]:
        if line.startswith('Member') or line.startswith('- **Member') or line.startswith('**Member'):
            member_info = line.split(':')
            member_role_description = member_info[1].split('-')
            
            member_role = member_role_description[0].strip()
            member_expertise = member_role_description[1].strip() if len(member_role_description) > 1 else ''
            
            parsed_info['members'].append({
                'role': member_role,
                'expertise_description': member_expertise
            })
    
    return parsed_info

def setup_model(model_name):
    if 'gemini' in model_name:
        genai.configure(api_key=_pick_gemini_key())
        return genai, None
    elif 'gpt' in model_name or 'o3' in model_name:
        client = OpenAI(
            api_key=_require_env("OPENAI_API_KEY"),
            base_url=_env("OPENAI_BASE_URL"),
        )
        return None, client
    else:
        client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
        return None, client

def load_data(dataset):
    test_qa = []
    examplers = []

    test_path = f'./data/{dataset}/test.jsonl'
    with open(test_path, 'r') as file:
        for line in file:
            test_qa.append(json.loads(line))

    train_path = f'./data/{dataset}/train.jsonl'
    with open(train_path, 'r') as file:
        for line in file:
            examplers.append(json.loads(line))

    return test_qa, examplers

def create_question(sample, dataset, few_shot="none"):
    if dataset == 'medqa':
        question = sample['question'] + " Options: "
        options = []
        for k, v in sample['options'].items():
            options.append("({}) {}".format(k, v))
        random.shuffle(options)
        question += " ".join(options)
        return question, None
    elif 'rarebench' in dataset:
        question = sample[few_shot]
        return question, None
    return sample['question'], None

def determine_difficulty(question, difficulty):
    if difficulty != 'adaptive':
        return difficulty
    
    difficulty_prompt = f"""Now, given the medical query as below, you need to decide the difficulty/complexity of it:\n{question}.\n\nPlease indicate the difficulty/complexity of the medical query among below options:\n1) basic: a single medical agent can output an answer.\n2) intermediate: number of medical experts with different expertise should dicuss and make final decision.\n3) advanced: multiple teams of clinicians from different departments need to collaborate with each other to make final decision."""
    
    medical_agent = Agent(instruction='You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.', role='medical expert', model_info='gpt-3.5')
    medical_agent.chat('You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.')
    response = medical_agent.chat(difficulty_prompt)

    if 'basic' in response.lower() or '1)' in response.lower():
        return 'basic'
    elif 'intermediate' in response.lower() or '2)' in response.lower():
        return 'intermediate'
    elif 'advanced' in response.lower() or '3)' in response.lower():
        return 'advanced'

def process_basic_query(question, examplers, model, args):
    medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model, port=args.port)
    new_examplers = []
    if args.dataset == 'medqa':
        random.shuffle(examplers)
        for ie, exampler in enumerate(examplers[:5]):
            tmp_exampler = {}
            exampler_question = exampler['question']
            choices = [f"({k}) {v}" for k, v in exampler['options'].items()]
            random.shuffle(choices)
            exampler_question += " " + ' '.join(choices)
            exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}\n\n"
            exampler_reason = medical_agent.chat(f"You are a helpful medical agent. Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\nAnswer: {exampler_answer}")

            tmp_exampler['question'] = exampler_question
            tmp_exampler['reason'] = exampler_reason
            tmp_exampler['answer'] = exampler_answer
            new_examplers.append(tmp_exampler)
    
    single_agent = Agent(instruction='You are a helpful assistant that answers multiple choice questions about medical knowledge.', role='medical expert', examplers=new_examplers, model_info=model, port=args.port)
    single_agent.chat('You are a helpful assistant that answers multiple choice questions about medical knowledge.')
    final_decision = single_agent.temp_responses(f'''The following are multiple choice questions (with answers) about medical knowledge. Let's think step by step.\n\n**Question:** {question}\nAnswer: ''', img_path=None)
    
    return final_decision

def process_intermediate_query(question, examplers, model, args):
    cprint("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])
    recruit_prompt = f"""You are an experienced medical expert who recruits a group of experts with diverse identity and ask them to discuss and solve the given medical query."""
    
    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=model, port=args.port)
    tmp_agent.chat(recruit_prompt)
    
    num_agents = 5  # You can adjust this number as needed
    recruited = tmp_agent.chat(f"Question: {question}\nYou can recruit {num_agents} experts in different medical expertise. Considering the medical question and the options for the answer, what kind of experts will you recruit to better make an accurate answer?\nAlso, you need to specify the communication structure between experts (e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), or indicate if they are independent.\n\nFor example, if you want to recruit five experts, you answer can be like:\n1. Pediatrician - Specializes in the medical care of infants, children, and adolescents. - Hierarchy: Independent\n2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions. - Hierarchy: Pediatrician > Cardiologist\n3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders. - Hierarchy: Independent\n4. Neonatologist - Focuses on the care of newborn infants, especially those who are born prematurely or have medical issues at birth. - Hierarchy: Independent\n5. Medical Geneticist - Specializes in the study of genes and heredity. - Hierarchy: Independent\n\nPlease answer in above format, and do not include your reason.")
    print(f'recruited:\n {recruited}\n -------')
    agents_info = [agent_info.split(" - Hierarchy: ") for agent_info in recruited.split('\n') if agent_info]
    agents_data = [(info[0], info[1]) if len(info) > 1 else (info[0], None) for info in agents_info]

    agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
    random.shuffle(agent_emoji)

    hierarchy_agents = parse_hierarchy(agents_data, agent_emoji)

    agent_list = ""
    for i, agent in enumerate(agents_data):
        agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
        description = agent[0].split('-')[1].strip().lower()
        agent_list += f"Agent {i+1}: {agent_role} - {description}\n"

    agent_dict = {}
    medical_agents = []
    for agent in agents_data:
        try:
            agent_role = agent[0].split('-')[0].split('.')[1].strip().lower()
            description = agent[0].split('-')[1].strip().lower()
        except:
            continue
        
        inst_prompt = f"""You are a {agent_role} who {description}. Your job is to collaborate with other medical experts in a team."""
        _agent = Agent(instruction=inst_prompt, role=agent_role, model_info=model, port=args.port)
        
        _agent.chat(inst_prompt)
        agent_dict[agent_role] = _agent
        medical_agents.append(_agent)

    for idx, agent in enumerate(agents_data):
        try:
            print(f"Agent {idx+1} ({agent_emoji[idx]} {agent[0].split('-')[0].strip()}): {agent[0].split('-')[1].strip()}")
        except:
            print(f"Agent {idx+1} ({agent_emoji[idx]}): {agent[0]}")

    fewshot_examplers = ""
    medical_agent = Agent(instruction='You are a helpful medical agent.', role='medical expert', model_info=model, port=args.port)
    if args.dataset == 'medqa':
        random.shuffle(examplers)
        for ie, exampler in enumerate(examplers[:5]):
            exampler_question = f"[Example {ie+1}]\n" + exampler['question']
            options = [f"({k}) {v}" for k, v in exampler['options'].items()]
            random.shuffle(options)
            exampler_question += " " + " ".join(options)
            exampler_answer = f"Answer: ({exampler['answer_idx']}) {exampler['answer']}"
            exampler_reason = tmp_agent.chat(f"Below is an example of medical knowledge question and answer. After reviewing the below medical question and answering, can you provide 1-2 sentences of reason that support the answer as you didn't know the answer ahead?\n\nQuestion: {exampler_question}\n\nAnswer: {exampler_answer}")
            
            exampler_question += f"\n{exampler_answer}\n{exampler_reason}\n\n"
            fewshot_examplers += exampler_question

    print()
    cprint("[INFO] Step 2. Collaborative Decision Making", 'yellow', attrs=['blink'])
    cprint("[INFO] Step 2.1. Hierarchy Selection", 'yellow', attrs=['blink'])
    print_tree(hierarchy_agents[0], horizontal=False)
    print()

    num_rounds = 5
    num_turns = 5
    num_agents = len(medical_agents)

    interaction_log = {f'Round {round_num}': {f'Turn {turn_num}': {f'Agent {source_agent_num}': {f'Agent {target_agent_num}': None for target_agent_num in range(1, num_agents + 1)} for source_agent_num in range(1, num_agents + 1)} for turn_num in range(1, num_turns + 1)} for round_num in range(1, num_rounds + 1)}

    cprint("[INFO] Step 2.2. Participatory Debate", 'yellow', attrs=['blink'])

    round_opinions = {n: {} for n in range(1, num_rounds+1)}
    round_answers = {n: None for n in range(1, num_rounds+1)}
    initial_report = ""
    for k, v in agent_dict.items():
        opinion = v.chat(f'''Given the examplers, please return your answer to the medical query among the option provided.\n\n{fewshot_examplers}\n\nQuestion: {question}\n\nYour answer should be like below format.\n\nAnswer: ''', img_path=None)
        initial_report += f"({k.lower()}): {opinion}\n"
        round_opinions[1][k.lower()] = opinion

    final_answer = None
    for n in range(1, num_rounds+1):
        print(f"== Round {n} ==")
        round_name = f"Round {n}"
        agent_rs = Agent(instruction="You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.", role="medical assistant", model_info=model, port=args.port)
        agent_rs.chat("You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.")
        
        assessment = "".join(f"({k.lower()}): {v}\n" for k, v in round_opinions[n].items())

        report = agent_rs.chat(f'''Here are some reports from different medical domain experts.\n\n{assessment}\n\nYou need to complete the following steps\n1. Take careful and comprehensive consideration of the following reports.\n2. Extract key knowledge from the following reports.\n3. Derive the comprehensive and summarized analysis based on the knowledge\n4. Your ultimate goal is to derive a refined and synthesized report based on the following reports.\n\nYou should output in exactly the same format as: Key Knowledge:; Total Analysis:''')
        
        for turn_num in range(num_turns):
            turn_name = f"Turn {turn_num + 1}"
            print(f"|_{turn_name}")

            num_yes = 0
            for idx, v in enumerate(medical_agents):
                all_comments = "".join(f"{_k} -> Agent {idx+1}: {_v[f'Agent {idx+1}']}\n" for _k, _v in interaction_log[round_name][turn_name].items())
                
                participate = v.chat("Given the opinions from other medical experts in your team, please indicate whether you want to talk to any expert (yes/no)\n\nOpinions:\n{}".format(assessment if n == 1 else all_comments))
                
                if 'yes' in participate.lower().strip():                
                    chosen_expert = v.chat(f"Enter the number of the expert you want to talk to:\n{agent_list}\nFor example, if you want to talk with Agent 1. Pediatrician, return just 1. If you want to talk with more than one expert, please return 1,2 and don't return the reasons.")
                    
                    chosen_experts = [int(ce) for ce in chosen_expert.replace('.', ',').split(',') if ce.strip().isdigit()]

                    for ce in chosen_experts:
                        specific_question = v.chat(f"Please remind your medical expertise and then leave your opinion to an expert you chose (Agent {ce}. {medical_agents[ce-1].role}). You should deliver your opinion once you are confident enough and in a way to convince other expert with a short reason.")
                        
                        print(f" Agent {idx+1} ({agent_emoji[idx]} {medical_agents[idx].role}) -> Agent {ce} ({agent_emoji[ce-1]} {medical_agents[ce-1].role}) : {specific_question}")
                        interaction_log[round_name][turn_name][f'Agent {idx+1}'][f'Agent {ce}'] = specific_question
                
                    num_yes += 1
                else:
                    print(f" Agent {idx+1} ({agent_emoji[idx]} {v.role}): \U0001f910")

            if num_yes == 0:
                break
        
        if num_yes == 0:
            break

        tmp_final_answer = {}
        for i, agent in enumerate(medical_agents):
            response = agent.chat(f"Now that you've interacted with other medical experts, remind your expertise and the comments from other experts and make your final answer to the given question:\n{question}\nAnswer: ")
            tmp_final_answer[agent.role] = response

        round_answers[round_name] = tmp_final_answer
        final_answer = tmp_final_answer

    print('\nInteraction Log')        
    myTable = PrettyTable([''] + [f"Agent {i+1} ({agent_emoji[i]})" for i in range(len(medical_agents))])

    for i in range(1, len(medical_agents)+1):
        row = [f"Agent {i} ({agent_emoji[i-1]})"]
        for j in range(1, len(medical_agents)+1):
            if i == j:
                row.append('')
            else:
                i2j = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {i}'][f'Agent {j}'] is not None
                          for k in range(1, len(interaction_log)+1)
                          for l in range(1, len(interaction_log['Round 1'])+1))
                j2i = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {j}'][f'Agent {i}'] is not None
                          for k in range(1, len(interaction_log)+1)
                          for l in range(1, len(interaction_log['Round 1'])+1))
                
                if not i2j and not j2i:
                    row.append(' ')
                elif i2j and not j2i:
                    row.append(f'\u270B ({i}->{j})')
                elif j2i and not i2j:
                    row.append(f'\u270B ({i}<-{j})')
                elif i2j and j2i:
                    row.append(f'\u270B ({i}<->{j})')

        myTable.add_row(row)
        if i != len(medical_agents):
            myTable.add_row(['' for _ in range(len(medical_agents)+1)])
    
    print(myTable)

    cprint("\n[INFO] Step 3. Final Decision", 'yellow', attrs=['blink'])
    
    moderator = Agent("You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.", "Moderator", model_info=model, port=args.port)
    moderator.chat('You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.')
    
    _decision = moderator.temp_responses(f"Given each agent's final answer, please review each agent's opinion and make the final answer to the question by taking majority vote. Your answer should be like below format:\nAnswer: C) 2th pharyngeal arch\n{final_answer}\n\nQuestion: {question}", img_path=None)
    final_decision = {'majority': _decision}

    print(f"{'\U0001F468\u200D\u2696\uFE0F'} moderator's final decision (by majority vote):", _decision)
    print()

    return _decision

def process_advanced_query(question, model, args):
    print("[STEP 1] Recruitment")
    group_instances = []

    recruit_prompt = f"""You are an experienced medical expert. Given the complex medical query, you need to organize Multidisciplinary Teams (MDTs) and the members in MDT to make accurate and robust answer."""

    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=model, port=args.port)
    tmp_agent.chat(recruit_prompt)

    num_teams = 3  # You can adjust this number as needed
    num_agents = 3  # You can adjust this number as needed

    recruited = tmp_agent.chat(f"""
Question: {question}

Instructions:
1. Do **not** directly answer the medical question.
2. Organize **{num_teams}** unique multidisciplinary teams (MDTs), each with **{num_agents}** clinicians, tailored to the question and its answer options.
3. Each MDT should have a distinct specialty or purpose.
4. **You must include** the Initial Assessment Team (IAT) and the Final Review and Decision Team (FRDT).
5. Follow this exact output format for each group:

Group 1 - [Team Name] ([Acronym])
Member 1: [Specialty] (Lead) - [Role description and why they’re chosen]
Member 2: [Specialty] - [Role description]
Member 3: [Specialty] - [Role description]

Group 2 - [Team Name] ([Acronym])
Member 1: [Specialty] (Lead) - [Role description and why they’re chosen]
Member 2: [Specialty] - [Role description]
Member 3: [Specialty] - [Role description]

… (continue until Group {num_teams})

Example:

Group 1 - Initial Assessment Team (IAT)  
Member 1: Otolaryngologist (ENT Surgeon) (Lead) – Specializes in ear, nose, and throat surgery, including thyroidectomy; leads the team for surgical intervention and complication management.  
Member 2: General Surgeon – Provides additional surgical expertise and supports overall complication management.  
Member 3: Anesthesiologist – Focuses on perioperative care, pain management, and anesthesia-related complication assessment.

Group 2 - Diagnostic Evidence Team (DET)  
Member 1: Endocrinologist (Lead) – Manages long-term hormonal therapy and monitors post-surgical complications.  
Member 2: Speech-Language Pathologist – Provides voice/swallow rehabilitation after nerve injury.  
Member 3: Neurologist – Assesses nerve damage and recovery strategies.

…  

Group {num_teams} - Final Review and Decision Team (FRDT)  
Member 1: Senior Consultant from each specialty (Lead) – Offers overarching expertise and guidance.  
Member 2: Clinical Decision Specialist – Integrates all recommendations into a unified treatment plan.  
Member 3: Advanced Diagnostic Support – Uses advanced tools to confirm the extent of nerve damage and advise final decisions.

Above is just an example. Please create your own MDTs accordingly.
""")
    groups = [group.strip() for group in recruited.split("Group") if group.strip()]
    print(f"Recruited {groups} groups.")
    group_strings = ["Group " + group for group in groups]
    
    print(f"Recruited {len(group_strings)} groups.")
    for i1, gs in enumerate(group_strings):
        res_gs = parse_group_info(gs)
        print(f"Group {i1+1} - {res_gs['group_goal']}")
        if len(res_gs['members']) == 0:
            print(" No members recruited.")
            continue    
        for i2, member in enumerate(res_gs['members']):
            print(f" Member {i2+1} ({member['role']}): {member['expertise_description']}")
        print(res_gs['group_goal'], res_gs['members'])

        group_instance = Group(res_gs['group_goal'], res_gs['members'], [model], question, args.port)
        group_instances.append(group_instance)

    # STEP 2. initial assessment from each group
    # STEP 2.1. IAP Process
    interaction_history = {}
    initial_assessments = []
    for group_instance in group_instances:
        if 'initial' in group_instance.goal.lower() or 'iap' in group_instance.goal.lower():
            init_assessment, init_assessment_iteration_log = group_instance.interact(comm_type='internal')
            initial_assessments.append([group_instance.goal, init_assessment])
            interaction_history[group_instance.goal] = init_assessment_iteration_log

    initial_assessment_report = ""
    for idx, init_assess in enumerate(initial_assessments):
        initial_assessment_report += f"Group {idx+1} - {init_assess[0]}\n{init_assess[1]}\n\n"
    print(f'Initial Assessment {initial_assessment_report}')
    # STEP 2.2. other MDTs Process
    assessments = []
    for group_instance in group_instances:
        if 'initial' not in group_instance.goal.lower() and 'iap' not in group_instance.goal.lower():
            assessment, assessment_iteration_log = group_instance.interact(comm_type='internal')
            assessments.append([group_instance.goal, assessment])
            interaction_history[group_instance.goal] = assessment_iteration_log

    assessment_report = ""
    for idx, assess in enumerate(assessments):
        assessment_report += f"Group {idx+1} - {assess[0]}\n{assess[1]}\n\n"
    print(f'Assessment Report {assessment_report}')

    # STEP 2.3. FRDT Process
    final_decisions = []
    for group_instance in group_instances:
        if 'review' in group_instance.goal.lower() or 'decision' in group_instance.goal.lower() or 'frdt' in group_instance.goal.lower():
            decision, decision_iteration_log = group_instance.interact(comm_type='internal')
            final_decisions.append([group_instance.goal, decision])
            interaction_history[group_instance.goal] = decision_iteration_log
    print(f'Final Decisions Report {final_decisions}')

    compiled_report = ""
    for idx, decision in enumerate(final_decisions):
        compiled_report += f"Group {idx+1} - {decision[0]}\n{decision[1]}\n\n"
    print(f'Compiled Report {compiled_report}')

    # STEP 3. Final Decision
    decision_prompt = f"""You are an experienced medical expert. Now, given the investigations from multidisciplinary teams (MDT), please review them very carefully and return your final decision to the medical query."""
    summery_model = model if not args.o3_summary else 'o3-2025-04-16'
    tmp_agent = Agent(instruction=decision_prompt, role='decision maker', model_info=summery_model, port=args.port)
    tmp_agent.chat(decision_prompt)

    final_decision = tmp_agent.temp_responses(f"""Investigation:\n{initial_assessment_report}\n\nQuestion: {question}.Enumerate the top 10 most likely diagnoses. Be precise, listing one diagnosis per line, and try to cover many unique possibilities (at least 10). The top 10 diagnoses are:""", img_path=None)
    print(f"Final Decision: {final_decision}")
    interaction_history['final_decision'] = final_decision
    return final_decision, interaction_history

from typing import List, Dict
import json
import re

SPECIALIST_POOL: List[str] = [
    # 41 departments from the provided table
    "Pediatrics","Urology","Hematology","Radiology","Neurosurgery","Rheumatology",
    "Psychiatry","Pulmonology","Dentistry","Endocrinology","Allergy and Immunology",
    "Nuclear Medicine","Hepatobiliary Surgery","Plastic Surgery","Interventional Radiology",
    "Cardiology","Thoracic Surgery","Clinical Nutrition","Vascular Surgery","Anesthesiology",
    "Laboratory Medicine","Pathology","Neurology","Obstetrics and Gynecology","Ophthalmology",
    "General Surgery","Dermatology","Geriatrics","Orthopedic Surgery","Cardiac Surgery",
    "Traditional Chinese Medicine","Nephrology","Oncology","General Practice","Gastroenterology",
    "Infectious Diseases","Rehabilitation Medicine","Pharmacy","Ultrasound Medicine",
    "Otorhinolaryngology","Breast Surgery"
]

JSON_FINDER = re.compile(r"\[\s*\{.*\}\s*\]", re.S)

def _extract_json_list(text: str) -> List[Dict]:
    """Robustly extract a JSON list of dicts from LLM output."""
    m = JSON_FINDER.search(text)
    blob = m.group(0) if m else text.strip()
    try:
        data = json.loads(blob)
        assert isinstance(data, list)
        return data
    except Exception:
        # last-ditch: wrap single dict
        try:
            d = json.loads(blob)
            return [d] if isinstance(d, dict) else []
        except Exception as e:
            print(f"Failed to parse JSON from: {blob}\nError: {e}")
            return []

def _pick_specialists(tmp_agent, question: str, n: int) -> List[Dict]:
    pool_str = ", ".join(SPECIALIST_POOL)
    prompt = f"""
You are the Chief Medical Officer assembling a single MDT.

Case: {question}

From this specialist pool:
{pool_str}

Pick EXACTLY {n} distinct specialties best suited for this case.
Return ONLY a valid JSON array. Each item must be:
{{
  "specialty": "<one from the pool>",
  "role": "<short role name>",
  "why": "<1-2 lines on why this specialty is needed>",
  "system_prompt": "<2-4 sentences instructing an LLM acting as this specialist on how to reason for THIS case. Include differential focus, key tests, red flags, and how to communicate with other members.>"
}}
No prose outside JSON.
"""
    raw = tmp_agent.chat(prompt)
    items = _extract_json_list(raw)

    # Normalize & guardrail to pool + uniqueness + length n
    seen = set()
    cleaned = []
    for it in items:
        spec = it.get("specialty", "").strip()
        if spec in SPECIALIST_POOL and spec not in seen:
            cleaned.append({
                "specialty": spec,
                "role": it.get("role", spec),
                "expertise_description": it.get("why", ""),
                "system_prompt": it.get("system_prompt", f"You are a {spec} contributing to this case.")
            })
            seen.add(spec)
        if len(cleaned) >= n:
            break
    # if LLM under-picked, pad with generic General Practice etc.
    if len(cleaned) < n:
        for s in SPECIALIST_POOL:
            if s not in seen:
                cleaned.append({
                    "specialty": s,
                    "role": s,
                    "expertise_description": "Added to complete the MDT size.",
                    "system_prompt": f"You are a {s} contributing to this case."
                })
                if len(cleaned) >= n:
                    break
    return cleaned[:n]

POOL = [
    "Pediatrics", "Urology", "Hematology", "Radiology", "Neurosurgery",
    "Rheumatology", "Psychiatry", "Pulmonology", "Dentistry", "Endocrinology",
    "Allergy and Immunology",
    "Nuclear Medicine", "Hepatobiliary Surgery", "Plastic Surgery",
    "Interventional Radiology", "Cardiology", "Thoracic Surgery",
    "Clinical Nutrition", "Vascular Surgery", "Anesthesiology",
    "Laboratory Medicine",
    "Pathology", "Neurology", "Obstetrics and Gynecology", "Ophthalmology",
    "General Surgery", "Dermatology", "Geriatrics", "Orthopedic Surgery",
    "Cardiac Surgery", "Traditional Chinese Medicine",
    "Nephrology", "Oncology", "General Practice", "Gastroenterology",
    "Infectious Diseases", "Rehabilitation Medicine", "Pharmacy",
    "Ultrasound Medicine", "Otorhinolaryngology", "Breast Surgery"
]

def _pick_rare_specialists(tmp_agent, question: str, n: int) -> List[Dict]:
    pool_str = ", ".join(POOL)
    prompt = f"""
You are the Chief Medical Officer assembling a single MDT.

Case: {question}

From this specialist pool:
{pool_str}

Pick no more than {n} distinct specialties best suited for this case. Never add unnecessary specialties just to complete the size.
Return ONLY a valid JSON array. Each item must be:\n\n
{{
  "specialty": "<one from the pool>",
  "role": "<short role name>",
  "description": "<2-4 sentences instructing an LLM acting as this specialist on how to reason for THIS case. Include differential focus, red flags, and how to communicate with other members.>",
}}\n\n
No prose outside JSON. No special characters.\n
"""
    raw = tmp_agent.chat(prompt)
    if tmp_agent.model_info.startswith("gemini"):
        rewriter_prompt = f"""
Based on the raw input {raw}, rewrite it to a standard JSON format.
Return ONLY a valid JSON array. Each item must be:\n
{{
  "specialty": "<one from the pool>",
  "role": "<short role name>",
  "description": "<2-4 sentences instructing an LLM acting as this specialist on how to reason for THIS case. Include differential focus, red flags, and how to communicate with other members.>",
}}\n\n
No prose outside JSON. No special characters.\n
"""
        rewriter = Agent(instruction="You are a professional json rewriter. You need to rewrite a raw json to a standard JSON format", role='json rewriter', model_info="o3-2025-04-16")
        raw = rewriter.temp_responses(rewriter_prompt)

    items = _extract_json_list(raw)

    # Normalize & guardrail to pool + uniqueness + length n
    seen = set()
    cleaned = []
    for it in items:
        spec = it.get("specialty", "").strip()
        if spec in SPECIALIST_POOL and spec not in seen:
            cleaned.append({
                "specialty": spec,
                "role": it.get("role", spec),
                "expertise_description": it.get("description", ""),
                "system_prompt": it.get("system_prompt", f"You are a {spec} contributing to this case.")
            })
            seen.add(spec)
        if len(cleaned) >= n:
            break
    
    if len(cleaned) == 0:
        recruit_prompt = (
            "You are an experienced medical expert. Build ONE multidisciplinary team (MDT) "
            "tailored to the case. Do not answer the case yet."
        )
        tmp_recruit_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info='o3-2025-04-16')
        tmp_recruit_agent.chat(recruit_prompt)
        cleaned = _pick_rare_specialists(tmp_recruit_agent, question, n)
        
    return cleaned[:n]

def build_mdt_prompt(assessment_report: str, question: str) -> str:
    return (
        "You are the chair of an MDT. Read the case snapshot and reason step by step.\n\n"
        f"MDT Investigations Summary:\n{assessment_report}\n\n"
        f"Question: {question}\n\n"
        "Output format (STRICT):\n"
        "<analysis>\n"
        "- Summarize key findings and red flags.\n"
        "- Differential logic: why certain classes of diseases are prioritized.\n"
        "- Tie-breakers and evidence weighting.\n"
        "</analysis>\n\n"
        "<top10>\n"
        "[1] <Diagnosis name>\n"
        "[2] <Diagnosis name>\n"
        "[3] <Diagnosis name>\n"
        "[4] <Diagnosis name>\n"
        "[5] <Diagnosis name>\n"
        "[6] <Diagnosis name>\n"
        "[7] <Diagnosis name>\n"
        "[8] <Diagnosis name>\n"
        "[9] <Diagnosis name>\n"
        "[10] <Diagnosis name>\n"
        "</top10>\n\n"
        "Rules:\n"
        "- Provide exactly 10 diagnoses, one per line, each starting with [rank].\n"
        "- Be precise and avoid variants on the same concept unless clinically distinct.\n"
        "- No extra text outside <analysis> and <top10>."
    )

def parse_top10_block(text: str):
    """
    解析 <analysis> 与 <top10>。对 <top10> 的每行 [n] 诊断名 做健壮提取。
    若不满10行，会返回已解析到的部分；并做去重（保序）。
    """
    # 提取 analysis
    m_analysis = re.search(r"<analysis>(.*?)</analysis>", text, flags=re.S | re.I)
    analysis = m_analysis.group(1).strip() if m_analysis else ""

    # 提取 top10 区块
    m_top = re.search(r"<top10>(.*?)</top10>", text, flags=re.S | re.I)
    top_block = m_top.group(1).strip() if m_top else ""

    return analysis, top_block

def process_rareagent(sample, question, model, args, max_turn: int = 10, num_members: int = 4, result_file: str = 'result.json', interaction_history_file: str = 'interaction_history.json'):
    if args.multi_expert:
        # "qwen3-235b-a22b-instruct-2507"
        # model = ["o3-2025-04-16", "gemini-2.5-pro", "gpt-5"]
        model = ["gemini-2.5-pro"]
    else:
        model = [model]
    print("[STEP 1] MDT Recruitment")
    recruit_prompt = (
        "You are an experienced medical expert. Build ONE multidisciplinary team (MDT) "
        "tailored to the case. Do not answer the case yet."
    )
    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=random.choice(model), port=args.port)
    tmp_agent.chat(recruit_prompt)

    # === Pick specialists and craft system prompts ===
    members = _pick_rare_specialists(tmp_agent, question, num_members)

    print(f"Recruited MDT with {len(members)} members:")
    # for i, m in enumerate(members, 1):
    #     print(f" Member {i} [{m['specialty']}] ({m['role']}): {m['expertise_description']}")

    # === Create one Group ===
    group_goal = "Multidisciplinary Review Team (MDT) – integrate specialty opinions and converge on a final diagnosis/plan."
    group_instance = Group(group_goal, members, model, question, args.port)

    # === Internal discussion turns ===
    assessments: List[str] = []
    interaction_history = {}
    interaction_history[group_instance.goal] = {}
    for i in range(max_turn):
        assessment, assessment_iteration_log = group_instance.rare_interact(comm_type='internal')
        assessments.append(assessment)
        interaction_history[group_instance.goal][f'turn_{i}'] = assessment_iteration_log
        print(f"[Turn {i+1}] {assessment[:200]}...")
        # 早停
        if getattr(group_instance, "_state", {}).get("fc_global"):
            print("[MDT] Converged early.")
            break

    if hasattr(group_instance, "_state"):
        O_rounds = group_instance._state.get("O_rounds", [])
        final_brief = []
        accepted_cnt = 0

        for item in O_rounds:
            rr = item["r"]
            accepted = item.get("accepted", {})
            if accepted:
                accepted_cnt += len(accepted)
                final_brief.append(
                    f"Round {rr} accepted: " + json.dumps(accepted, ensure_ascii=False)
                )

        if accepted_cnt == 0:
            # 没有任何共识 → 回退到最近一轮各专科的意见
            last_ops = group_instance._state.get("last_opinion", {})
            if last_ops:
                compact = {
                    role: (op.get("findings", op) if isinstance(op, dict) else str(op))
                    for role, op in last_ops.items()
                }
                assessment_report = (
                    "No consensus reached. Using latest specialist opinions:\n"
                    + json.dumps(compact, ensure_ascii=False)
                )
        else:
            assessment_report = "\n".join(final_brief)
    else:
        assessment_report = "\n".join(f"Turn {i+1}: {t}" for i, t in enumerate(assessments))

    # Build assessment report for the decider
    # assessment_report = ""
    # for idx, txt in enumerate(assessments, 1):
    #     assessment_report += f"Turn {idx} - {txt}\n\n"

    # === Final decision ===
    decision_prompt = (
        "You are an experienced medical expert. Given the MDT investigations below, "
        "review carefully and produce a final, concise decision to the medical query. "
        "Include key reasoning, differentials ruled in/out, and next steps."
    )
    summary_model = model if not getattr(args, "o3_summary", False) else "o3-2025-04-16"
    decider = Agent(instruction=decision_prompt, role='decision maker', model_info=random.choice(summary_model), port=args.port)
    decider.chat(decision_prompt)

    prompt = build_mdt_prompt(assessment_report=assessment_report, question=question)
    final_decision = decider.temp_responses(prompt, img_path=None)
    analysis, top10 = parse_top10_block(final_decision)
    if top10 == "":
        rewriter = Agent(instruction=decision_prompt, role='decision rewriter', model_info='o3-2025-04-16', port=args.port)
        rewrite_prompt = (
            "You are the chair of an MDT. Read the case snapshot and reason step by step.\n\n"
            f"MDT analysis and top10 choices:\n{final_decision}\n\n"
            "Rewrite the answer to this output format (STRICT), please don't change original words:\n"
            "<analysis>\n"
            "- Summarize key findings and red flags.\n"
            "- Differential logic: why certain classes of diseases are prioritized.\n"
            "- Tie-breakers and evidence weighting.\n"
            "</analysis>\n\n"
            "<top10>\n"
            "[1] <Diagnosis name>\n"
            "[2] <Diagnosis name>\n"
            "[3] <Diagnosis name>\n"
            "[4] <Diagnosis name>\n"
            "[5] <Diagnosis name>\n"
            "[6] <Diagnosis name>\n"
            "[7] <Diagnosis name>\n"
            "[8] <Diagnosis name>\n"
            "[9] <Diagnosis name>\n"
            "[10] <Diagnosis name>\n"
            "</top10>\n\n"
            "Rules:\n"
            "- Provide exactly 10 diagnoses, one per line, each starting with [rank].\n"
            "- Be precise and avoid variants on the same concept unless clinically distinct.\n"
            "- No extra text outside <analysis> and <top10>."
        )

        final_decision_rewrited = rewriter.temp_responses(rewrite_prompt, img_path=None)
        analysis, top10 = parse_top10_block(final_decision_rewrited)

    final_decision_history = {
        'model': decider.model_info,
        'prompt': prompt,
        'analysis': analysis,
        'top10': top10,
        'raw_decision': final_decision
    }

    interaction_history['final_decision'] = final_decision_history
    print(f"Final Decision: {final_decision}")


    result = {
        'question': question,
        'patient_info': sample['patient_idx'],
        'golden_diagnosis': sample['golden_diagnosis'],
        'predict_diagnosis': top10,
        'predict_rank': None
    }

    # write result
    with open(result_file, 'w', encoding='utf-8-sig') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    with open(interaction_history_file, 'w', encoding='utf-8-sig') as f:
        json.dump(interaction_history, f, indent=4, ensure_ascii=False)

    cprint(f"saved to {result_file}", 'yellow')
    return top10, interaction_history