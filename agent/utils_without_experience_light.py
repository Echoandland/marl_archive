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
    return _require_env("GEMINI_API_KEY") if _env("GEMINI_API_KEY") else _require_env("genai_api_key")

# 若你已有这些工具函数，可删除本段重复定义
def _now_ts() -> str:
    return datetime.utcnow().isoformat()

def _truncate(s: str, n: int) -> str:
    if not isinstance(s, str):
        s = str(s)
    return s[:n]

def _safe_json_loads(s: str, default: Any) -> Any:
        # 去掉首尾空格和换行
    s = s.strip()

    # 处理 fenced block 的情况：```json ... ``` 或 ``` ... ```
    if s.startswith("```"):
        # 去掉开头的 ``` 或 ```json
        s = s.lstrip("`")
        if s.startswith("json"):
            s = s[len("json"):].lstrip()
        # 去掉末尾的 ```
        if s.endswith("```"):
            s = s[:-3].rstrip()

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

def parse_diagnosis_block(text: str):
    """
    解析 <diagnosis>。对 <diagnosis> 的每行 [n] 诊断名 做健壮提取。
    若不满10行，会返回已解析到的部分；并做去重（保序）。
    """
    # 提取 diagnosis 区块
    diagnosis = re.search(r"<diagnosis>(.*?)</diagnosis>", text, flags=re.S | re.I)
    diagnosis_block = diagnosis.group(1).strip() if diagnosis else text

    return diagnosis_block


from experience_to_db.experience_api import ExperienceKB
from pathlib import Path


class Agent:
    def __init__(self, instruction, role, examplers=None, model_info='gpt-5', port=8000, img_path=None):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path
        # self.init_experience()
        self.experience = None
        
        self.prompt_tokens = []
        self.completion_tokens = []
        self.total_tokens = []

        self.review_history = {}

        if self.model_info == 'gemini-pro' or "gemini" in self.model_info:
            # Set GEMINI_API_KEY (or GEMINI_API_KEYS) in environment.
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
            # Set OPENAI_API_KEY in environment.
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

    def init_experience(self):
        # role 安全提取前缀
        if not self.role or not isinstance(self.role, str):
            expert_name = "Unknown"
        else:
            expert_name = self.role.split(" ")[0].strip() or "Unknown"

        # 构建路径
        # storage_dir = Path(f"./experience_to_db/old/exp_index/{expert_name}")
        storage_dir = Path(f"./experience_to_db/api/qwen3/30_top3_not_top1/exp_index/{expert_name}")
        if not storage_dir.exists():
            print(f"[WARN] 目录不存在: {storage_dir} → 设置为空")
            self.experience = None
            return

        print(f"[INFO] 使用专家目录: {storage_dir}")

        # 初始化 ExperienceKB
        self.experience = ExperienceKB(storage_dir=str(storage_dir))
    
    def save_message(self, prompt, response):
        self.messages.append({"role": "user", "content": prompt})
        self.messages.append({"role": "assistant", "content": response})

    def augment_prompt(self, message, top_k_per_query=3, max_hints=3):
        if not self.experience:
            return message

        return self.experience.augment_prompt(message, top_k_per_query=top_k_per_query, max_hints=max_hints)

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
            messages = self.messages[:]
            messages.append({"role": "user", "content": message})
            if self.model_info == 'gpt-3.5':
                model_name = "gpt-3.5-turbo"
            else:
                model_name = self.model_info
            
            if 'Qwen3' in model_name:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                )

                self.prompt_tokens.append(response.usage.prompt_tokens)
                self.completion_tokens.append(response.usage.completion_tokens)
                self.total_tokens.append(response.usage.total_tokens)
                # print("Prompt tokens:", response.usage.prompt_tokens)
                # print("Completion tokens:", response.usage.completion_tokens)
                # print("Total tokens:", response.usage.total_tokens)
            elif 'o3' in model_name or 'gpt-4.1-2025-04-14' in model_name:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                )
                self.prompt_tokens.append(response.usage.prompt_tokens)
                self.completion_tokens.append(response.usage.completion_tokens)
                self.total_tokens.append(response.usage.total_tokens)
            elif 'gpt-5' in model_name:
                response = self.client.chat.completions.create(
                    model="gpt-5",                    # gpt-5-mini / nano 亦可
                    messages=messages,
                    reasoning_effort="medium",  # 可选：最快响应
                    verbosity="medium"         # 可选：输出更简洁
                )
                #self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
                self.prompt_tokens.append(response.usage.prompt_tokens)
                self.completion_tokens.append(response.usage.completion_tokens)
                self.total_tokens.append(response.usage.total_tokens)
            else:
                print(model_name)
                print(messages)
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                )
                print(response)
                self.prompt_tokens.append(response.usage.prompt_tokens)
                self.completion_tokens.append(response.usage.completion_tokens)
                self.total_tokens.append(response.usage.total_tokens)

            #self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            return response.choices[0].message.content

    def temp_responses(self, message, img_path=None):
        if self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini', 'o3-2025-04-16', 'gpt-5']:      
            messages = self.messages[:]
            messages.append({"role": "user", "content": message})
            
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
                        messages=messages,
                        temperature=temperature,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    self.prompt_tokens.append(response.usage.prompt_tokens)
                    self.completion_tokens.append(response.usage.completion_tokens)
                    self.total_tokens.append(response.usage.total_tokens)
                elif 'o3' in model_info or 'gpt-4.1-2025-04-14' in model_info:
                    response = self.client.chat.completions.create(
                        model=model_info,
                        messages=messages,
                    )
                    self.prompt_tokens.append(response.usage.prompt_tokens)
                    self.completion_tokens.append(response.usage.completion_tokens)
                    self.total_tokens.append(response.usage.total_tokens)
                elif 'gpt-5' in model_info:
                    response = self.client.chat.completions.create(
                        model="gpt-5",                    # gpt-5-mini / nano 亦可
                        messages=messages,
                        reasoning_effort="low",  # 可选：最快响应
                        verbosity="low"         # 可选：输出更简洁
                    )
                    self.prompt_tokens.append(response.usage.prompt_tokens)
                    self.completion_tokens.append(response.usage.completion_tokens)
                    self.total_tokens.append(response.usage.total_tokens)
                else:
                    response = self.client.chat.completions.create(
                        model=model_info,
                        messages=messages,
                        temperature=temperature,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    self.prompt_tokens.append(response.usage.prompt_tokens)
                    self.completion_tokens.append(response.usage.completion_tokens)
                    self.total_tokens.append(response.usage.total_tokens)

                responses[temperature] = response.choices[0].message.content
                
            return response.choices[0].message.content
        
        elif self.model_info == 'gemini-pro' or "gemini" in self.model_info:
            for _ in range(10):
                try:
                    response = self._chat.send_message(message)
                    responses = response.text
                    print(f"[{self.role}], model {self.model_info}, Response: {responses}")
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
            messages = self.messages[:]
            messages.append({"role": "user", "content": message})
            
            temperatures = [0.7]
            
            responses = {}
            for temperature in temperatures:
                if self.model_info == 'gpt-3.5':
                    model_info = 'gpt-3.5-turbo'
                else:
                    model_info = self.model_info
                
                if 'Qwen3' not in model_info:
                    response = self.client.chat.completions.create(
                        model=model_info,
                        messages=messages,
                        temperature=temperature,
                    )
                    responses[temperature] = response.choices[0].message.content
                    self.prompt_tokens.append(response.usage.prompt_tokens)
                    self.completion_tokens.append(response.usage.completion_tokens)
                    self.total_tokens.append(response.usage.total_tokens)
                else:
                    response = self.client.chat.completions.create(
                        model=model_info,
                        messages=messages,
                        temperature=temperature,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    self.prompt_tokens.append(response.usage.prompt_tokens)
                    self.completion_tokens.append(response.usage.completion_tokens)
                    self.total_tokens.append(response.usage.total_tokens)
                    responses[temperature] = response.choices[0].message.content
                
            return response.choices[0].message.content

    def msgs_response(self, messages):
        if self.model_info in ['gpt-3.5', 'gpt-4', 'gpt-4o', 'gpt-4o-mini', 'o3-2025-04-16', 'gpt-5']:      
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
                        messages=messages,
                        temperature=temperature,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    self.prompt_tokens.append(response.usage.prompt_tokens)
                    self.completion_tokens.append(response.usage.completion_tokens)
                    self.total_tokens.append(response.usage.total_tokens)
                elif 'o3' in model_info or 'gpt-4.1-2025-04-14' in model_info:
                    response = self.client.chat.completions.create(
                        model=model_info,
                        messages=messages,
                    )
                    self.prompt_tokens.append(response.usage.prompt_tokens)
                    self.completion_tokens.append(response.usage.completion_tokens)
                    self.total_tokens.append(response.usage.total_tokens)
                elif 'gpt-5' in model_info:
                    response = self.client.chat.completions.create(
                        model="gpt-5",                    # gpt-5-mini / nano 亦可
                        messages=messages,
                        reasoning_effort="low",  # 可选：最快响应
                        verbosity="low"         # 可选：输出更简洁
                    )
                    self.prompt_tokens.append(response.usage.prompt_tokens)
                    self.completion_tokens.append(response.usage.completion_tokens)
                    self.total_tokens.append(response.usage.total_tokens)
                else:
                    response = self.client.chat.completions.create(
                        model=model_info,
                        messages=messages,
                        temperature=temperature,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    self.prompt_tokens.append(response.usage.prompt_tokens)
                    self.completion_tokens.append(response.usage.completion_tokens)
                    self.total_tokens.append(response.usage.total_tokens)

                responses[temperature] = response.choices[0].message.content
                
            return response.choices[0].message.content
        
        elif self.model_info == 'gemini-pro' or "gemini" in self.model_info:
            for _ in range(10):
                try:
                    response = self._chat.send_message(messages)
                    responses = response.text
                    print(f"[{self.role}], model {self.model_info}, Response: {responses}")
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
            temperatures = [0.7]
            responses = {}
            for temperature in temperatures:
                if self.model_info == 'gpt-3.5':
                    model_info = 'gpt-3.5-turbo'
                else:
                    model_info = self.model_info
                
                if 'Qwen3' not in model_info:
                    response = self.client.chat.completions.create(
                        model=model_info,
                        messages=messages,
                        temperature=temperature,
                    )
                    responses[temperature] = response.choices[0].message.content
                    self.prompt_tokens.append(response.usage.prompt_tokens)
                    self.completion_tokens.append(response.usage.completion_tokens)
                    self.total_tokens.append(response.usage.total_tokens)
                else:
                    response = self.client.chat.completions.create(
                        model=model_info,
                        messages=messages,
                        temperature=temperature,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                    )
                    self.prompt_tokens.append(response.usage.prompt_tokens)
                    self.completion_tokens.append(response.usage.completion_tokens)
                    self.total_tokens.append(response.usage.total_tokens)
                    responses[temperature] = response.choices[0].message.content
                
            return response.choices[0].message.content

    def review(self, target_role, message):
        if target_role not in self.review_history:
            self.review_history[target_role] = []

        # 合并历史
        msgs = self.messages[-1:] + self.review_history[target_role][-1:]
        msgs.append({"role": "user", "content": message})
        response = self.msgs_response(msgs)
        self.review_history[target_role].append({"role": "user", "content": message})
        self.review_history[target_role].append({"role": "assistant", "content": response})
        return response

class Group:
    def __init__(self, goal, members, model, question, port=8000, examplers=None):
        self.goal = goal
        self.members = []
        for member_info in members:
            _agent = Agent('You are a {}.'.format(f"{member_info['specialty']} ({member_info['role']})"), role=f"{member_info['specialty']} ({member_info['role']})", model_info=random.choice(model), port=port)
            # response = _agent.chat('You are a {} who {}.'.format(f"{member_info['specialty']} ({member_info['role']})", member_info['expertise_description'].lower()))
            # _agent.save_message('You are a {} who {}.'.format(f"{member_info['specialty']} ({member_info['role']})", member_info['expertise_description'].lower()), response)
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
                    f"Now, after reviewing the following example cases, return your top10 diseases to the symptoms.\n\n"
                    f"{self.examplers}\nSymptoms: {self.question}"
                )
            else:
                investigation_prompt = (
                    f"The gathered investigation from your assistant clinicians is as follows:\n"
                    f"{gathered_investigation}\n\n"
                    f"Now, return your top10 diseases to the symptoms.\n\n"
                    f"Symptoms: {self.question}"
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
        first_turn = False,
        img_path: Optional[str] = None
    ):
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
        lead_member = None
        assist_members = []
        for member in self.members:
            if 'lead' in member.role.lower():
                lead_member = member
            else:
                assist_members.append(member)

        if lead_member is None:
            lead_member = assist_members[0]

        team_snapshot_json = json.dumps(accepted_snapshot, ensure_ascii=False)
        # (1) 专家个人意见：自由文本（无 JSON）
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

            if first_turn:
                opinion_prompt = (
                    f"You are {role} in an MDT. "
                    "You will collaborate with other specialists to develop a diagnosis in the field of rare diseases based on the patient's info. "
                    "You will be provided and asked about a complicated clinical case; "
                    "read it carefully and then provide a diverse and comprehensive differential diagnosis.\n\n"
                    #f"Goal: {getattr(self, 'goal', '')}\n"

                    f"Patient info: {getattr(self, 'question', '')}\n\n"

                    #"TASK:\n"
                    #"- Give your opinion using ONLY the Goal and Patient info.\n"
                    #"- Base reasoning on your specialist expertise.\n"
                    #"- If information is missing, say \"insufficient evidence.\"\n"
                    #"- List the most likely diagnoses (up to 10).\n"
                    #"- Stop if you feel no further reasonable diagnoses remain.\n"
                    #"- For each diagnosis, add a 1–2 sentence rationale tied to the #patient info.\n\n"

                    "OUTPUT (no extra text):\n"
                    "Reflection (2–3 sentences): Describe how the patient’s information shapes your differential diagnosis.\n\n"

                    "<diagnosis>\n"
                    "1. Diagnosis 1: rationale\n"
                    "2. Diagnosis 2: rationale\n"
                    "...\n"
                    "</diagnosis>\n\n"

                    "Start your response:"
                )
            else:
                opinion_prompt = (
                    f"You are {role} in an MDT. "
                    "You will collaborate with other specialists to develop a diagnosis in the field of rare diseases based on the patient's info. "
                    "You will be provided and asked about a complicated clinical case; "
                    "read it carefully and then provide a diverse and comprehensive differential diagnosis.\n"
                    # f"Goal: {getattr(self, 'goal', '')}\n"
                    f"Patient info: {getattr(self, 'question', '')}\n\n"
                    f"Last accepted snapshot (may be empty):\n{team_snapshot_json}\n\n"
                    f"Peer feedback (delta):\n{delta_text}\n\n"

                    "TASK:\n"
                    "- Update YOUR opinion using Snapshot, and Delta.\n"
                    # "- Base reasoning on your specialist expertise.\n"
                    # "- If Snapshot and Delta conflict, prefer Snapshot unless Delta provides strong justification.\n"
                    "- Rebut peer feedback or deltas if you disagree, give your reasons clearly.\n" 
                    #"- If a point lacks support from these sources, write 'insufficient evidence' and stop.\n"
                    "- List the most likely diagnoses (up to 10). Stop if no further reasonable diagnoses remain.\n"
                    "- For each diagnosis, add a 1–2 sentence rationale tied strictly to Snapshot and/or Delta.\n\n"

                    "OUTPUT (no extra text, no code fences):\n"
                    "Reflection (2–3 sentences): Explain how Snapshot and Delta refine or shift your view.\n\n"

                    "<diagnosis>\n"
                    "1. Diagnosis 1: rationale\n"
                    "2. Diagnosis 2: rationale\n"
                    "...\n"
                    "</diagnosis>\n\n"

                    "Start your response:"
                )
            simple_peer_prompt = (f"You are {role} in an MDT.\n"
                f"Goal: {getattr(self, 'goal', '')}\n"
                f"Patient info: {getattr(self, 'question', '')}\n\n"
                "TASK:\n"
                #"- Give your opinion using ONLY the Goal and Patient info.\n"
                #"- Base reasoning on your specialist expertise.\n"
                #"- If information is missing, say \"insufficient evidence.\"\n"
                "- List the most likely diagnoses (up to 10).\n"
                #"- Stop if you feel no further reasonable diagnoses remain.\n"
                "- For each diagnosis, add a 1–2 sentence rationale tied to the patient info.\n\n"
                "Start your response:")
            try:
                opinion_prompt = m.augment_prompt(opinion_prompt, top_k_per_query=5, max_hints=5)
                raw_opinion = m.chat(opinion_prompt)
                diagnosis_block = parse_diagnosis_block(raw_opinion)
                m.save_message(opinion_prompt, diagnosis_block if diagnosis_block not in (None, "") else raw_opinion)
            except Exception as e:
                raw_opinion = ""
                interaction_log["steps"].append({
                    "role": role, "model": model, "phase": "opinion_error",
                    "prompt": opinion_prompt, "response": "", "error": str(e), "target": role
                })
                diagnosis_block = ""
            else:
                interaction_log["steps"].append({
                    "role": role, "model": model, "phase": "opinion",
                    "prompt": opinion_prompt, "response": raw_opinion, "diagnosis":diagnosis_block,  "target": role
                })

            # 存储专家原始意见文本
            state["last_opinion"][role] = diagnosis_block

            # (2) 同行审阅：输入为“专家意见全文 + 团队快照”，输出结构化 JSON
            pack_for_reviewers = json.dumps(
                {
                    "target_role": role,
                    "opinion_text": diagnosis_block if diagnosis_block not in (None, "") else raw_opinion,
                    "team_snapshot": accepted_snapshot,
                },
                ensure_ascii=False
            )

            collected_agreements: List[str] = []
            collected_disagreements: List[str] = []

            for reviewer in members:
                r_role = getattr(reviewer, "role", "Unknown")
                if r_role == role:
                    continue
                
                peer_prompt = (
                    f"You are a {r_role} in a multidisciplinary team (MDT).\n"
                    f"Review the {role}'s opinion on the case below, using ONLY the provided materials.\n"
                    f"{pack_for_reviewers}\n\n"
                    f"Provided phenotype/features: {getattr(self, 'question', '')}\n\n"

                    "Guidelines (flexible):\n"
                    "- When the TARGET is the leader, usually put mild differences into 'agreements' as suggestions, "
                    "and only log 'disagreements' when there are clear, well-grounded contradictions (ideally with ≥2 distinct anchors).\n"
                    "- Avoid repeating earlier review points; bring in fresh, non-overlapping insights.\n"
                    "- Each agreement or disagreement must include a list of 'anchors' (specific features) supporting the point.\n\n"

                    "Return STRICT JSON, with the following schema:\n"
                    "{{\n"
                    '  "analysis": "Free-form summary of key reasoning (multiple sentences allowed).",\n'
                    '  "agreements": [\n'
                    '    {{\n'
                    '      "role": "<your MDT role>",\n'
                    '      "content": "Detailed explanation of agreement or supportive suggestion.",\n'
                    '      "anchors": ["feature1", "feature2"]\n'
                    '    }}\n'
                    '  ],\n'
                    '  "disagreements": null | [\n'
                    '    {{\n'
                    '      "role": "<your MDT role>",\n'
                    '      "content": "Detailed explanation of disagreement or challenge.",\n'
                    '      "anchors": ["featureA", "featureB"]\n'
                    '    }}\n'
                    '  ]\n'
                    "}}"
                )
                
                simple_peer_prompt = (f"You are a {r_role} in a multidisciplinary team (MDT).\n"
                    f"Review the {role}'s opinion on the case below, using ONLY the provided materials.\n"
                    f"{pack_for_reviewers}\n\n"
                    f"Provided phenotype/features: {getattr(self, 'question', '')}\n\n")
                
                try:
                    peer_prompt = m.augment_prompt(peer_prompt, top_k_per_query=5, max_hints=5)
                    peer_raw = reviewer.review(role, peer_prompt)
                    parsed, ok = _safe_json_loads(peer_raw, {"analysis": "", "agreements": [], "disagreements": []})
                    reviewer.save_message(simple_peer_prompt, peer_raw)

                    if reviewer.model_info.startswith("gemini") and not ok:
                        rewriter_prompt = (
                            f"Peer feedback input:\n{peer_raw}\n\n"
                            "Task: Reformat this input into valid JSON ONLY, matching exactly the schema below:\n\n"
                            "{{\n"
                            '  "analysis": "<As a peer, briefly comment on their reasoning. First mention the main points you agree with, then highlight any disagreements or points needing clarification.>",\n'
                            '  "agreements": ["<point you agree with>", "..."],\n'
                            '  "disagreements": null | ["<brief point you disagree with and why>", "..."]\n'
                            "}}\n\n"
                            "Rules:\n"
                            "1. Preserve the wording from the Peer feedback input exactly — do NOT paraphrase, shorten, or modify its descriptions.\n"
                            "2. If you have NO objections (consensus), set disagreements to null (not an empty list).\n"
                            "3. Do not add extra keys, metadata, or any text outside the JSON.\n"
                            "4. Do not include ```json or any other code fence.\n"
                        )
                        rewriter = Agent(instruction="You are a professional json rewriter. You need to rewrite a raw json to a standard JSON format", role='decision maker', model_info="o3-2025-04-16")
                        peer_raw = rewriter.temp_responses(rewriter_prompt)
                        parsed, ok = _safe_json_loads(peer_raw, {"analysis": "", "agreements": [], "disagreements": []})
                    if isinstance(parsed.get("agreements"), list):
                        collected_agreements.extend([str(x) for x in parsed["agreements"]])
                    if isinstance(parsed.get("disagreements"), list):
                        collected_disagreements.extend([str(x) for x in parsed["disagreements"]])
                except Exception as e:
                    import traceback
                    print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
                    peer_raw = "{}"
                    interaction_log["steps"].append({
                        "role": r_role, "phase": "peer_review_error",
                        "prompt": peer_prompt, "response": "{}", "error": str(e), "target": role
                    })
                    raise Exception(f"Peer review error: {e}")
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
                    round_O[role] = diagnosis_block

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

def determine_multiagents(question, difficulty):
    # if difficulty != 'adaptive':
    #    return difficulty
    
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

def process_cot(question, examplers, model, args):
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
    # single_agent.chat('You are a helpful assistant that answers multiple choice questions about medical knowledge.')
    final_decision = single_agent.temp_responses(f'''The following are multiple choice questions (with answers) about medical knowledge. Let's think step by step.\n\n**Question:** {question}\nAnswer: ''', img_path=None)
    
    return final_decision

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

from typing import List, Dict
import json
import re

JSON_FINDER = re.compile(r"\[\s*\{.*\}\s*\]", re.S)

def _strip_fenced(text: str) -> str:
    """去掉 ```json ... ``` 或 ``` 包裹"""
    text = text.strip()
    if text.startswith("```"):
        text = text.lstrip("`")
        if text.startswith("json"):
            text = text[len("json"):].lstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
    return text

def _extract_json_list(text: str):
    """Robustly extract a JSON list of dicts from LLM output."""
    text = _strip_fenced(text)

    # 优先匹配数组
    m = JSON_FINDER.search(text)
    blob = m.group(0) if m else text

    try:
        data = json.loads(blob)
        if isinstance(data, list):
            return data, True
        if isinstance(data, dict):  # 单个 dict → 包装成 list
            return [data], True
    except Exception:
        pass

    # 兜底：截取第一个 {...} 或 [...]
    try:
        start = min((i for i in [text.find("["), text.find("{")] if i >= 0), default=-1)
        end = max((i for i in [text.rfind("]"), text.rfind("}")]), default=-1)
        if start >= 0 and end >= 0 and end > start:
            snippet = text[start:end+1]
            data = json.loads(snippet)
            if isinstance(data, list):
                return data, True
            if isinstance(data, dict):
                return [data], True
    except Exception:
        pass

    print(f"⚠️ Failed to parse JSON from: {text[:200]}...")
    return [], False

POOL = [
    "Pediatrics","Urology","Hematology","Neurosurgery","Rheumatology",
    "Psychiatry","Pulmonology","Dentistry","Endocrinology","Allergy and Immunology",
    "Hepatobiliary Surgery","Plastic Surgery","Cardiology","Thoracic Surgery",
    "Vascular Surgery","Neurology","Obstetrics and Gynecology","Ophthalmology",
    "General Surgery","Dermatology","Geriatrics","Orthopedic Surgery","Cardiac Surgery",
    "Traditional Chinese Medicine","Nephrology","Oncology","General Practice","Gastroenterology",
    "Infectious Diseases","Rehabilitation Medicine","Otorhinolaryngology","Breast Surgery"
]

def _pick_rare_specialists(tmp_agent, question: str, n: int) -> List[Dict]:
    pool_str = ", ".join(POOL)
    prompt = (
        f"You are the Chief Medical Officer assembling a single MDT.\n\n"
        f"Case: {question}\n\n"
        f"From this specialist pool:\n{pool_str}\n\n"
        f"Pick no more than {n} distinct specialties best suited for this case. "
        "Never add unnecessary specialties just to complete the size.\n\n"
        "Return ONLY a valid JSON array. Each item must be:\n\n"
        "{\n"
        '  "specialty": "<one from the pool>",\n'
        '  "role": "<short role name, exactly one item has \'leader\'>",\n'
        '  "description": "<2-4 sentences instructing an LLM acting as this specialist on how to reason for THIS case. '
        'Include differential focus, red flags, and how to communicate with other members>"\n'
        "}\n\n"
        "No prose outside JSON. No special characters.\n"
    )
    raw = tmp_agent.chat(prompt)
    items, ok = _extract_json_list(raw)

    if tmp_agent.model_info.startswith("gemini") and not ok:
        rewriter_prompt = (
            f"Given the raw input: {raw}\n"
            "Your task: Reformat it into a valid JSON array ONLY.\n"
            "Do not change, shorten, or paraphrase ANY of the text inside `raw` — keep all descriptions exactly as they are.\n\n"
            "The output must be a JSON array, where each element follows this schema:\n\n"
            "{\n"
            '  "specialty": "<one value from the predefined pool>",\n'
            '  "role": "<short role name, exactly one item must be \'leader\'>",\n'
            '  "description": "<use the corresponding raw description verbatim — do not edit wording>"\n'
            "}\n\n"
            "Return ONLY the JSON array. No prose, no comments, no special characters outside JSON."
        )
        rewriter = Agent(instruction="You are a professional json rewriter. You need to rewrite a raw json to a standard JSON format", role='json rewriter', model_info="o3-2025-04-16")
        raw = rewriter.temp_responses(rewriter_prompt)

        items, ok = _extract_json_list(raw)

    # Normalize & guardrail to pool + uniqueness + length n
    seen = set()
    cleaned = []
    for it in items:
        spec = it.get("specialty", "").strip()
        if spec in POOL and spec not in seen:
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
        # "- Be precise and avoid variants on the same concept unless clinically distinct.\n"
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
        model = ["gemini-2.5-pro", "gpt-5"]
        # model = ["gemini-2.5-pro"]
    else:
        model = [model]

    print("[STEP 1] MDT Recruitment")
    recruit_prompt = (
        "You are an experienced medical expert. Build ONE multidisciplinary team (MDT) "
        "tailored to the case. Do not answer the case yet."
    )
    tmp_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info=random.choice(model), port=args.port)

    # === Pick specialists and craft system prompts ===
    members = _pick_rare_specialists(tmp_agent, question, num_members)

    print(f"Recruited MDT with members:  {members}")
    # for i, m in enumerate(members, 1):
    #     print(f" Member {i} [{m['specialty']}] ({m['role']}): {m['expertise_description']}")

    # === Create one Group ===
    group_goal = "Multidisciplinary Review Team (MDT) – integrate specialty opinions and converge on a final diagnosis/plan."
    group_instance = Group(group_goal, members, model, question, args.port)
    
    assessments = []
    interaction_history = {}
    interaction_history[group_instance.goal] = {}
    
    # === Internal discussion turns ===
    print("[STEP 2] Multiturn Critics")
    assessments: List[str] = []

    for i in range(max_turn):
        if i == 0:
            assessment, assessment_iteration_log = group_instance.rare_interact(comm_type='internal', first_turn=True)
        else:
            assessment, assessment_iteration_log = group_instance.rare_interact(comm_type='internal')
        assessments.append(assessment)
        interaction_history[group_instance.goal][f'turn_{i}'] = assessment_iteration_log
        print(f"[Turn {i+1}] {assessment}...")
        # 早停
        if getattr(group_instance, "_state", {}).get("fc_global"):
            print("[MDT] Converged early.")
            break

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
                final_brief.append(
                    f"Round {rr} accepted: " + json.dumps(accepted, ensure_ascii=False)
                )

        if accepted_cnt == 0:
            # 没有任何共识 → 回退到最近一轮各专科的意见
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
            # 已有共识：在共识摘要后追加“未达成共识角色的最近结论”
            unresolved = {}
            if last_ops:
                for role, op in last_ops.items():
                    if role not in accepted_roles:
                        unresolved[role] = (
                            op.get("findings", op) if isinstance(op, dict) else str(op)
                        )

            if unresolved:
                final_brief.append(
                    "Unresolved (latest specialist opinions): "
                    + json.dumps(unresolved, ensure_ascii=False)
                )

            assessment_report = "\n".join(final_brief)
    else:
        assessment_report = "\n".join(f"Turn {i+1}: {t}" for i, t in enumerate(assessments))

    # Build assessment report for the decider
    # assessment_report = ""
    # for idx, txt in enumerate(assessments, 1):
    #     assessment_report += f"Turn {idx} - {txt}\n\n"

    # === Final decision ===
    print("[STEP 3] Final Decision")
    decision_prompt = (
        "You are an experienced medical expert. Given the MDT investigations below, "
        "review carefully and produce a final, concise decision to the medical query. "
        "Include key reasoning, differentials ruled in/out, and next steps."
    )
    summary_model = model if not getattr(args, "o3_summary", False) else "o3-2025-04-16"
    decider = group_instance.members[0]
    # decider.chat(decision_prompt)

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

    token_cost_metric = {
        agent.role: {
            'prompt_tokens_sum': sum(agent.prompt_tokens),
            'completion_tokens_sum': sum(agent.completion_tokens),
            'total_tokens_sum': sum(agent.total_tokens),
            'prompt_tokens_max': max(agent.prompt_tokens),
            'completion_tokens_max': max(agent.completion_tokens),
            'total_tokens_max': max(agent.total_tokens),
            'prompt_tokens': agent.prompt_tokens,
            'completion_tokens': agent.completion_tokens,
            'total_tokens': agent.total_tokens
        } for agent in [tmp_agent] + group_instance.members + [decider]
    }

    final_decision_history = {
        'model': decider.model_info,
        'prompt': prompt,
        'analysis': analysis,
        'golden_diagnosis': sample['golden_diagnosis'],
        'top10': top10,
        'raw_decision': final_decision,
        'token_cost_metric': token_cost_metric
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