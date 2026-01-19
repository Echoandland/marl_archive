# -*- coding: utf-8 -*-
"""
experience_kb.py
Requirements:
  pip install faiss-cpu openai tiktoken
Notes:
  - backend 固定为 "openai"
  - 支持 OpenAI/vLLM Embeddings API（需设置 base_url）
  - 默认余弦相似：L2 归一化 + FAISS IndexFlatIP
"""

from __future__ import annotations
import os, json, re
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import faiss

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


class ExperienceKB:
    ROLE_RE = re.compile(r'You are\s+([A-Za-z\s\(\)]+?)\.\s*Goal:', re.IGNORECASE)

    def __init__(
        self,
        storage_dir: Optional[str] = None,
        *,
        model_name: str = "Qwen/Qwen3-Embedding-4B",    # 默认 OpenAI embedding 模型
        openai_base_url: Optional[str] = "http://localhost:8000/v1",
        index: Optional[faiss.Index] = None,
        metadata: Optional[List[Dict]] = None,
        auto_load: bool = True,
        normalize: bool = True,
        batch_size: int = 32,
    ):
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.model_name = model_name
        self._index: Optional[faiss.Index] = index
        self._meta: List[Dict] = metadata or []
        self._normalize = normalize
        self._batch_size = batch_size
        self._openai_base_url = openai_base_url
        self._openai_client = None

        if self._index is None and auto_load and self.storage_dir:
            idx_fp = self.storage_dir / "index.faiss"
            meta_fp = self.storage_dir / "meta.json"
            cfg_fp = self.storage_dir / "config.json"
            if idx_fp.exists() and meta_fp.exists() and cfg_fp.exists():
                self._index = faiss.read_index(str(idx_fp))
                self._meta = json.loads(meta_fp.read_text(encoding="utf-8"))
                cfg = json.loads(cfg_fp.read_text(encoding="utf-8"))
                self.model_name = cfg.get("model_name", self.model_name)
                self._normalize = cfg.get("normalize", self._normalize)
                self._openai_base_url = cfg.get("openai_base_url", self._openai_base_url)

    # ----------------------------- backend init ------------------------------
    def _ensure_backend(self):
        if OpenAI is None:
            raise RuntimeError("Requires: pip install openai")
        if self._openai_client is None:
            self._openai_client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
                base_url=self._openai_base_url or os.getenv("OPENAI_BASE_URL"),
            )

    # ---------------------------- embedding encode ---------------------------
    def _encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        self._ensure_backend()
        vectors: List[List[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            resp = self._openai_client.embeddings.create(model=self.model_name, input=batch)
            for item in resp.data:
                vectors.append(item.embedding)
        arr = np.array(vectors, dtype=np.float32)
        if self._normalize and arr.size > 0:
            arr = arr / np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)
        return arr

    # ------------------------------ index helpers ----------------------------
    def _ensure_index(self, dim: int):
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)

    # ----------------------------- ingestion ---------------------------------
    def ingest_kv(self, kv: Dict[str, str]) -> int:
        items = [{"key": k, "experience": v, "text": f"{k}\n---\n{v}"} for k, v in kv.items()]
        embs = self._encode([it["text"] for it in items])
        if embs.shape[0] == 0: return 0
        self._ensure_index(embs.shape[1])
        self._index.add(embs)
        self._meta.extend(items)
        return len(items)

    # ------------------------------- persistence -----------------------------
    def save(self):
        if not self.storage_dir:
            raise ValueError("storage_dir not set; cannot save.")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.storage_dir / "index.faiss"))
        (self.storage_dir / "meta.json").write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.storage_dir / "config.json").write_text(json.dumps(
            {"model_name": self.model_name, "normalize": self._normalize, "openai_base_url": self._openai_base_url},
            ensure_ascii=False, indent=2), encoding="utf-8")

    # ------------------------------ retrieval --------------------------------
    def _search(self, queries: List[str], top_k_per_query: int = 8) -> List[Dict]:
        if self._index is None or not self._meta:
            return []
        q_embs = self._encode(queries)
        scores: Dict[int, float] = {}
        for qe in q_embs:
            sims, idxs = self._index.search(np.expand_dims(qe, 0), top_k_per_query)
            for sc, ix in zip(sims[0], idxs[0]):
                if ix < 0: continue
                scores[ix] = scores.get(ix, 0.0) + float(sc)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        seen, results = set(), []
        for ix, acc in ranked:
            it = self._meta[ix].copy()
            k = it.get("key", f"idx-{ix}")
            if k in seen: continue
            seen.add(k)
            it["score"] = acc
            results.append(it)
        return results

    @staticmethod
    def _compose_hints(results: List[Dict], max_items: int = 8) -> str:
        return "\n".join([f"- ACTION: {it.get('key','')}\n  EXPERIENCE: {it.get('experience','')}" for it in results[:max_items]])

    @staticmethod
    def _append_hints(prompt: str, hints: str) -> str:
        if not hints.strip(): return prompt
        return f"{prompt}\n\n===== EXPERIENCE HINTS =====\n{hints}\n\n===== END OF EXPERIENCE HINTS =====\n"

    def augment_prompt(self, prompt: str, *, top_k_per_query: int = 8, max_hints: int = 8) -> str:
        results = self._search([prompt], top_k_per_query=top_k_per_query)
        hints = self._compose_hints(results, max_items=max_hints)
        return self._append_hints(prompt, hints)

def main():
    input_dir = Path(os.getenv("EXPERIENCE_INPUT_DIR", "./score_to_experience/30_top3_not_top1"))
    out_base = Path(os.getenv("EXPERIENCE_OUT_DIR", "./api/qwen3/30_top3_not_top1/exp_index"))

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
        kb = ExperienceKB(
            storage_dir=storage_dir,
            model_name=os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-4B"),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
        )
        kb.ingest_kv(experiences)
        kb.save()
        print(f"[OK] {expert_name}: {len(experiences)} entries → {storage_dir}")

def test():
    kb = ExperienceKB(storage_dir=os.getenv("EXPERIENCE_TEST_KB_DIR", "./api/qwen3/30_top3_not_top1/exp_index/Nephrology"))
    prompt = "You are Pediatrics (leader) in an MDT.\nGoal: Converge on a diagnosis."
    print(kb.augment_prompt(prompt))

# ------------------------------ simple test ---------------------------------
if __name__ == "__main__":
    main()
    test()
