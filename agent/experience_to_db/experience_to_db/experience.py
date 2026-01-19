# experience_kb.py
# Requirements:
#   pip install faiss-cpu transformers torch openai tiktoken
# Notes:
#   - OpenAI 后端需设置环境变量 OPENAI_API_KEY
#   - 默认使用 L2 归一化 + IndexFlatIP 实现余弦相似
#   - Transformers 默认模型: BAAI/bge-large-en-v1.5（可改为 gte-large 等）
#   - OpenAI 默认模型: text-embedding-3-small

from __future__ import annotations
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import faiss

# 延迟导入，避免不使用某后端时的依赖报错
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModel = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class ExperienceKB:
    """
    A unified KB for MDT 'experience hints' with pluggable embedding backends.

    Backends
    --------
    - backend="transformers":
        Uses Hugging Face Transformers (e.g., BAAI/bge-large-en-v1.5).
        Default pooling: mean pooling with attention_mask, then L2 normalize.

    - backend="openai":
        Uses OpenAI Embeddings API (text-embedding-3-small by default).
        Requires OPENAI_API_KEY env var.

    Storage layout (if storage_dir is provided)
    -------------------------------------------
    storage_dir/
      ├── index.faiss
      ├── meta.json
      └── config.json      # saves backend, model_name, normalize flag

    Metadata format
    ---------------
    meta.json: List[{"key": str, "value": str, "text": str}]
    (order aligns with FAISS vectors)
    """

    ROLE_RE = re.compile(r'You are\s+([A-Za-z\s\(\)]+?)\.\s*Goal:', re.IGNORECASE)

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        *,
        backend: str = "transformers",                 # "transformers" | "openai"
        model_name: Optional[str] = None,              # HF model id or OpenAI model name
        index: Optional[faiss.Index] = None,           # externally provided index (advanced)
        metadata: Optional[List[Dict]] = None,         # externally provided metadata
        auto_load: bool = True,
        normalize: bool = True,
        device: Optional[str] = None,                  # e.g., "cuda", "cpu"
        batch_size: int = 32
    ):
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.backend = backend
        self.model_name = model_name or (
            "Qwen/Qwen3-Embedding-4B" if backend == "transformers" else "text-embedding-3-small"
        )
        self._index: Optional[faiss.Index] = index
        self._meta: List[Dict] = metadata or []
        self._normalize = normalize
        self._device = device or ("cuda" if (torch and torch.cuda.is_available()) else "cpu")
        self._batch_size = batch_size

        # backend-specific state
        self._hf_tokenizer = None
        self._hf_model = None
        self._openai_client = None

        # auto-load if requested
        if self._index is None and auto_load and self.storage_dir:
            idx_fp = self.storage_dir / "index.faiss"
            meta_fp = self.storage_dir / "meta.json"
            cfg_fp = self.storage_dir / "config.json"
            if idx_fp.exists() and meta_fp.exists() and cfg_fp.exists():
                self._index = faiss.read_index(str(idx_fp))
                self._meta = json.loads(meta_fp.read_text(encoding="utf-8"))
                cfg = json.loads(cfg_fp.read_text(encoding="utf-8"))
                self.backend = cfg.get("backend", self.backend)
                self.model_name = cfg.get("model_name", self.model_name)
                self._normalize = cfg.get("normalize", self._normalize)
                # no need to load model here; lazy init in _encode

    # ----------------------------- backend init ------------------------------
    def _ensure_backend(self):
        if self.backend == "transformers":
            if AutoTokenizer is None or AutoModel is None or torch is None:
                raise RuntimeError("Transformers backend requires: pip install transformers torch")
            if self._hf_model is None or self._hf_tokenizer is None:
                self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._hf_model = AutoModel.from_pretrained(self.model_name)
                self._hf_model.to(self._device)
                self._hf_model.eval()
        elif self.backend == "openai":
            if OpenAI is None:
                raise RuntimeError("OpenAI backend requires: pip install openai")
            if self._openai_client is None:
                # relies on OPENAI_API_KEY
                self._openai_client = OpenAI()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    # ---------------------------- embedding encode ---------------------------
    def _encode_transformers(self, texts: List[str]) -> np.ndarray:
        """
        Mean pooling with attention mask, then L2 normalize (if self._normalize).
        """
        self._ensure_backend()
        embs = []
        bs = self._batch_size
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = texts[i:i+bs]
                inputs = self._hf_tokenizer(
                    batch, padding=True, truncation=True, return_tensors="pt"
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                outputs = self._hf_model(**inputs)
                last_hidden = outputs.last_hidden_state  # [B, T, H]
                mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
                masked = last_hidden * mask
                sum_embed = masked.sum(dim=1)
                lengths = mask.sum(dim=1).clamp(min=1)
                mean_embed = sum_embed / lengths
                if self._normalize:
                    mean_embed = torch.nn.functional.normalize(mean_embed, p=2, dim=1)
                embs.append(mean_embed.detach().cpu().numpy())
        return np.vstack(embs) if embs else np.zeros((0, 768), dtype=np.float32)

    def _encode_openai(self, texts: List[str]) -> np.ndarray:
        """
        OpenAI embeddings: default text-embedding-3-small (1536 dims).
        """
        self._ensure_backend()
        vectors: List[List[float]] = []
        bs = self._batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            resp = self._openai_client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            for item in resp.data:
                vectors.append(item.embedding)
        arr = np.array(vectors, dtype=np.float32)
        if self._normalize and arr.size > 0:
            arr = arr / np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)
        return arr

    def _encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        if self.backend == "transformers":
            return self._encode_transformers(texts)
        elif self.backend == "openai":
            return self._encode_openai(texts)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    # ------------------------------ index helpers ----------------------------
    def _ensure_index(self, dim: int):
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)

    # ----------------------------- public ingestion --------------------------
    def ingest_kv(self, kv: Dict[str, str]) -> int:
        """
        Ingest {key: value} experiences. Each item is embedded as: "key\n---\nvalue".
        Returns number of items added.
        """
        items = [{"key": k, "experience": v, "text": f"{k}\n---\n{v}"} for k, v in kv.items()]
        embs = self._encode([it["text"] for it in items])
        self._ensure_index(embs.shape[1])
        self._index.add(embs)
        self._meta.extend(items)
        return len(items)

    def add_or_update_items(self, items: List[Dict[str, str]]) -> int:
        """
        Batch ingest from a list of {"key": ..., "value": ...}.
        Returns number of items added.
        """
        kv = {it["key"]: it["value"] for it in items}
        return self.ingest_kv(kv)

    # ------------------------------- persistence -----------------------------
    def save(self):
        if not self.storage_dir:
            raise ValueError("storage_dir not set; cannot save.")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.storage_dir / "index.faiss"))
        (self.storage_dir / "meta.json").write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.storage_dir / "config.json").write_text(
            json.dumps(
                {"backend": self.backend, "model_name": self.model_name, "normalize": self._normalize},
                ensure_ascii=False, indent=2
            ),
            encoding="utf-8"
        )

    # ------------------------------ prompt parsing ---------------------------
    @staticmethod
    def _extract_role(prompt: str) -> str:
        m = ExperienceKB.ROLE_RE.search(prompt)
        return (m.group(1).strip() if m else "").replace(" ", "_")

    @staticmethod
    def _extract_patient_info(prompt: str) -> str:
        m = re.search(r"Patient info:\s*(.+?)(?:\n\s*\n|HARD CONSTRAINTS|TASK:|OUTPUT)", prompt, re.S | re.I)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _extract_symptoms(patient_info: str) -> List[str]:
        toks = re.split(r"[,\n;]+", patient_info)
        return [t.strip() for t in toks if t.strip()]

    @staticmethod
    def _detect_sections(prompt: str) -> List[str]:
        section_tokens = ["HARD CONSTRAINTS", "Reflection", "<diagnosis>", "OUTPUT (STRICT", "Patient info"]
        present = []
        for tok in section_tokens:
            if tok.lower() in prompt.lower():
                present.append(tok)
        return present

    @staticmethod
    def _prioritize_symptoms(symptoms: List[str]) -> List[str]:
        priority_keywords = {
            "methylmalonic", "homocyst", "alanin", "carnitine", "carboxylic",
            "optic", "nystagmus", "strabismus", "ventricular", "septal", "death", "infancy"
        }
        prioritized = [s for s in symptoms if any(pk in s.lower() for pk in priority_keywords)]
        return prioritized or symptoms[:6]

    @staticmethod
    def _make_queries(role: str, symptoms: List[str], sections: List[str]) -> List[str]:
        qs = []
        if role:
            qs.append(f"role={role} global MDT action rules HARD_CONSTRAINTS diagnosis_block reflection")
        for sec in sections:
            qs.append(f"role={role} section={sec} action best_practices errors")
        for s in ExperienceKB._prioritize_symptoms(symptoms)[:8]:
            qs.append(f"role={role} symptom={s} differential ranking rationale usage pitfalls strengths")
        return qs

    # -------------------------------- retrieval ------------------------------
    def _search(self, queries: List[str], top_k_per_query: int = 8) -> List[Dict]:
        if self._index is None or len(self._meta) == 0:
            return []
        q_embs = self._encode(queries)
        scores: Dict[int, float] = {}
        for qe in q_embs:
            qe = np.expand_dims(qe, 0)
            sims, idxs = self._index.search(qe, top_k_per_query)
            for score, idx in zip(sims[0], idxs[0]):
                if idx < 0:
                    continue
                scores[idx] = scores.get(idx, 0.0) + float(score)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for idx, acc in ranked:
            it = self._meta[idx].copy()
            it["score"] = acc
            results.append(it)
        # dedup by key, preserve order
        seen = set()
        dedup = []
        for it in results:
            k = it["key"]
            if k in seen:
                continue
            seen.add(k)
            dedup.append(it)
        return dedup

    @staticmethod
    def _compose_hints(results: List[Dict], max_items: int = 8) -> str:
        lines = []
        for it in results[:max_items]:
            lines.append(f"- ACTION: {it.get('key','')}\n  EXPERIENCE: {it.get('experience','')}")
        return "\n".join(lines)

    @staticmethod
    def _append_hints(prompt: str, hints: str) -> str:
        sep = "\n\n===== EXPERIENCE HINTS (for the role to consult; do not quote verbatim) =====\n"
        tail = "\n\n===== END OF EXPERIENCE HINTS =====\n"
        return f"{prompt}{sep}{hints}{tail}"

    # ------------------------------ high-level API ---------------------------
    def augment_prompt(
        self,
        prompt: str,
        *,
        top_k_per_query: int = 8,
        max_hints: int = 8,
    ) -> str:
        role = self._extract_role(prompt)
        patient_info = self._extract_patient_info(prompt)
        symptoms = self._extract_symptoms(patient_info)
        sections = self._detect_sections(prompt)
        queries = self._make_queries(role, symptoms, sections)

        results = self._search(queries, top_k_per_query=top_k_per_query)
        hints = self._compose_hints(results, max_items=max_hints)
        return self._append_hints(prompt, hints)

    @classmethod
    def from_json(
        cls,
        json_path: str,
        storage_dir: Optional[str] = None,
        *,
        backend: str = "transformers",
        model_name: Optional[str] = None,
        normalize: bool = True,
        device: Optional[str] = None,
        batch_size: int = 32
    ) -> "ExperienceKB":
        kb = cls(
            storage_dir=storage_dir,
            backend=backend,
            model_name=model_name,
            auto_load=False,
            normalize=normalize,
            device=device,
            batch_size=batch_size
        )
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        kb.ingest_kv(data)
        if storage_dir:
            kb.save()
        return kb


def main():
    input_dir = Path(os.getenv("EXPERIENCE_INPUT_DIR", "./score_to_experience/full_top1_top25"))
    out_base = Path(os.getenv("EXPERIENCE_OUT_DIR", "./old/exp_index"))

    out_base.mkdir(parents=True, exist_ok=True)

    for fp in input_dir.glob("*.experiences.jsonl"):
        expert_name = fp.stem.replace(".experiences", "")  # 去掉后缀，作为 expert 名
        storage_dir = out_base / expert_name
        storage_dir.mkdir(parents=True, exist_ok=True)

        # 读取 jsonl，生成 {key: experience}
        experiences = {}
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                k = obj.get("key")
                v = obj.get("experience")
                if k and v:
                    experiences[k] = v

        # 存入 vectordb
        kb = ExperienceKB(storage_dir=str(storage_dir))
        kb.ingest_kv(experiences)
        kb.save()
        print(f"[OK] {expert_name}: {len(experiences)} entries → {storage_dir}")

# import json
# 
# # 1) Build once
# experiences = json.loads(Path(os.getenv("EXPERIENCES_MAP_JSON", "./experiences.map.json")).read_text(encoding="utf-8"))
# kb = ExperienceKB(storage_dir="./exp_index")
# kb.ingest_kv(experiences)
# kb.save()

# 2) Later: load and augment a prompt
def test():
    kb2 = ExperienceKB(storage_dir="./exp_index/Pediatrics")  # auto-loads
    prompt = "You are Pediatrics (leader) in an MDT.\nGoal: Multidisciplinary Review Team (MDT) – integrate specialty opinions and converge on a final # diagnosis/plan.\nPatient info: Patient's phenotype: Strabismus,Nystagmus,Optic atrophy,Abnormality of prenatal development or birth,Motor delay,# Generalized hypotonia,Death in infancy,Ventricular septal defect,Hyperammonemia,Hyperhomocystinemia,Hyperalaninemia,Neonatal death,Death in childhood,# Abnormal circulating carnitine concentration,Methylmalonic aciduria,Elevated urinary carboxylic acid\n\n\nHARD CONSTRAINTS — READ CAREFULLY:\n- Base # reasoning strictly on Goal and Patient info above.\n- DO NOT introduce or infer external facts, literature, or assumptions.\n- DO NOT mention tests, # treatments, management, or propose new data.\n- DO NOT copy reasoning patterns from other roles or prior agents.\n- DO NOT invent missing details # (labs, imaging, history, etc.).\n- If information is missing, state 'insufficient evidence' rather than guess.\n- If you notice content requiring # external facts, STOP and refocus on Goal/Patient info only.\n\nTASK:\n- Update YOUR opinion using ONLY the Goal and Patient info.\n- Ground reasoning # in your own specialist expertise and perspective.\n- Focus strictly on reasoning and diagnosis ranking.\n- Final output MUST include a <diagnosis> ... # </diagnosis> block.\n\nOUTPUT (STRICT, no extra text, no code fences):\n1) Reflection (2–3 sentences): briefly explain how Goal/Patient info constrains # your view and note key uncertainties.\n2) Then produce a <diagnosis> block with exactly 10 numbered diagnoses.\n - Each line starts with a number and # diagnosis name.\n - Each has a 1–2 sentence rationale tied strictly to Goal/Patient info.\n\n<diagnosis>\n1. [Diagnosis 1]: [1–2 sentence rationale].# \n2. [Diagnosis 2]: [1–2 sentence rationale].\n...\n10. [Diagnosis 10]: [1–2 sentence rationale].\n</diagnosis>\n\nSELF-AUDIT (do not output this # section):\n- Did I avoid adding facts, tests, or treatments?\n- Is every rationale explicitly linked to Goal/Patient info?\n- Did I avoid copying other # roles' perspectives?\n- If evidence was insufficient, did I write 'insufficient evidence'?\n\nStart your response:"
    augmented = kb2.augment_prompt(prompt, top_k_per_query=8, max_hints=8)
    print(augmented)

if __name__ == "__main__":
    main()
    # test()
