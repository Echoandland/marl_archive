# experience_api_math.py
# 数学版 ExperienceKB：支持“单库构建 / 增强”、以及“按专家拆分建库（每个 *.experiences.jsonl -> 一个子库）”

from __future__ import annotations
import os, json, re, argparse, glob
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

# 可选后端：transformers / openai
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


# ------------------------- I/O helpers -------------------------
def _read_jsonl(fp: str) -> List[dict]:
    rows = []
    with open(fp, "r", encoding="utf-8-sig") as fr:
        for line in fr:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def _merge_jsonl_dir(dir_path: str) -> Dict[str, str]:
    """
    合并目录下所有 *.experiences.jsonl 为一个 {key: experience}，
    若同 key 冲突，取“经验文本更长”的那个。
    """
    kv: Dict[str, str] = {}
    for fp in sorted(glob.glob(os.path.join(dir_path, "*.experiences.jsonl"))):
        for obj in _read_jsonl(fp):
            k = (obj.get("key") or "").strip()
            v = (obj.get("experience") or "").strip()
            if not k or not v:
                continue
            if k not in kv or len(v) > len(kv[k]):
                kv[k] = v
    return kv


def _kv_from_jsonl_file(fp: str) -> Dict[str, str]:
    """
    读取单个 *.experiences.jsonl 为 {key: experience}
    """
    kv: Dict[str, str] = {}
    for obj in _read_jsonl(fp):
        k = (obj.get("key") or "").strip()
        v = (obj.get("experience") or "").strip()
        if k and v and (k not in kv or len(v) > len(kv[k])):
            kv[k] = v
    return kv


def _normalize_expert_name(name: str) -> str:
    """
    与运行时保持一致的专家目录名规范：
    - 去除括号内容
    - 将斜杠/反斜杠替换为下划线
    - 合并空白为下划线
    - 仅保留安全字符 [A-Za-z0-9_-.]
    """
    name = str(name or "").strip()
    name = re.sub(r"\([^)]*\)", "", name)
    name = name.replace("/", "_").replace("\\", "_")
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_\-\.]", "", name)
    return name or "Unknown"


# ------------------------- Core KB -------------------------
class ExperienceKB:
    """
    Math Experience KB with pluggable embedding backends.
    存储项格式： {"key": "...", "experience": "...", "text": "key\\n---\\nexperience"}
    """

    ROLE_RE = re.compile(r"You are\s+([A-Za-z\s\/\-\(\)]+?)\s+in a math", re.I)
    ROLE_FALLBACK = re.compile(r"You are\s+([A-Za-z\s\/\-\(\)]+?)\.", re.I)

    def __init__(self,
                 storage_dir: Optional[str] = None,
                 *,
                 backend: str = "transformers",  # "transformers" | "openai"
                 model_name: Optional[str] = None,
                 index: Optional[faiss.Index] = None,
                 metadata: Optional[List[Dict]] = None,
                 auto_load: bool = True,
                 normalize: bool = True,
                 device: Optional[str] = None,
                 batch_size: int = 32):
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.backend = backend
        self.model_name = model_name or ("BAAI/bge-large-en-v1.5" if backend=="transformers" else "text-embedding-3-small")
        self._index = index
        self._meta = metadata or []
        self._normalize = normalize
        self._device = device or ("cuda" if (torch and torch.cuda.is_available()) else "cpu")
        self._batch_size = batch_size

        self._hf_tokenizer = None
        self._hf_model = None
        self._openai_client = None

        if self._index is None and auto_load and self.storage_dir:
            idx_fp = self.storage_dir / "index.faiss"
            meta_fp = self.storage_dir / "meta.json"
            cfg_fp = self.storage_dir / "config.json"
            if idx_fp.exists() and meta_fp.exists() and cfg_fp.exists():
                if _HAS_FAISS:
                    self._index = faiss.read_index(str(idx_fp))
                else:
                    self._index = None  # 用到时会 fallback 到 numpy 实现
                self._meta = json.loads(meta_fp.read_text(encoding="utf-8"))
                cfg = json.loads(cfg_fp.read_text(encoding="utf-8"))
                self.backend = cfg.get("backend", self.backend)
                self.model_name = cfg.get("model_name", self.model_name)
                self._normalize = cfg.get("normalize", self._normalize)

    # ---------------- backend init/encode ----------------
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
                self._openai_client = OpenAI(
                    api_key="dummy",
                    base_url="https://gateway.salesforceresearch.ai/openai/process/v1/",
                    default_headers={"X-Api-Key": os.environ.get("X_API_KEY", "81cb41741637a7b8a772ef63cc760123")}
                )
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _encode_transformers(self, texts: List[str]) -> np.ndarray:
        self._ensure_backend()
        embs = []
        bs = self._batch_size
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                batch = texts[i:i+bs]
                inputs = self._hf_tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                outputs = self._hf_model(**inputs)
                last = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                masked = last * mask
                mean = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                if self._normalize:
                    mean = torch.nn.functional.normalize(mean, p=2, dim=1)
                embs.append(mean.detach().cpu().numpy())
        return np.vstack(embs) if embs else np.zeros((0, 768), dtype=np.float32)

    def _encode_openai(self, texts: List[str]) -> np.ndarray:
        self._ensure_backend()
        vectors = []
        bs = self._batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            resp = self._openai_client.embeddings.create(model=self.model_name, input=batch)
            for item in resp.data:
                vectors.append(item.embedding)
        arr = np.array(vectors, dtype=np.float32)
        if self._normalize and arr.size > 0:
            arr = arr / np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-12)
        return arr

    def _encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return self._encode_transformers(texts) if self.backend=="transformers" else self._encode_openai(texts)

    def _ensure_index(self, dim: int):
        if self._index is None:
            if _HAS_FAISS:
                self._index = faiss.IndexFlatIP(dim)
            else:
                class _NumpyFlatIP:
                    def __init__(self): self._vecs = None
                    def add(self, arr):
                        self._vecs = arr if self._vecs is None else np.vstack([self._vecs, arr])
                    def search(self, q, k):
                        if self._vecs is None:
                            sims = np.zeros((q.shape[0], 0), dtype=np.float32)
                            return sims, -np.ones((q.shape[0], k), dtype=int)
                        sims = q @ self._vecs.T
                        idx = np.argsort(-sims, axis=1)[:, :k]
                        top = np.take_along_axis(sims, idx, axis=1)
                        return top, idx
                self._index = _NumpyFlatIP()

    # ---------------- ingest/save ----------------
    def ingest_kv(self, kv: Dict[str, str]) -> int:
        # 1) 组装条目
        items = [{"key": k, "experience": v, "text": f"{k}\n---\n{v}"} 
                 for k, v in kv.items() if (k and v)]
        if not items:
            return 0

        # 2) 编码
        texts = [it["text"] for it in items]
        embs = self._encode(texts)

        # 3) 统一为 float32、C-contiguous、2D；空则仅存 meta
        if not isinstance(embs, np.ndarray):
            embs = np.array(embs, dtype=np.float32)
        if embs.size == 0:
            self._meta.extend(items)
            return 0
        if embs.dtype != np.float32:
            embs = embs.astype(np.float32, copy=False)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        if (not embs.flags["C_CONTIGUOUS"]) or (not embs.flags["OWNDATA"]):
            embs = np.array(embs, dtype=np.float32, order="C", copy=True)

        # 4) 建索引并写入（faiss 不可用则自动退回 numpy 实现）
        self._ensure_index(embs.shape[1])
        try:
            self._index.add(embs)
        except Exception:
            class _NumpyFlatIP:
                def __init__(self): self._vecs = None
                def add(self, arr):
                    self._vecs = arr if self._vecs is None else np.vstack([self._vecs, arr])
                def search(self, q, k):
                    if self._vecs is None:
                        sims = np.zeros((q.shape[0], 0), dtype=np.float32)
                        return sims, -np.ones((q.shape[0], k), dtype=int)
                    sims = q @ self._vecs.T
                    idx = np.argsort(-sims, axis=1)[:, :k]
                    top = np.take_along_axis(sims, idx, axis=1)
                    return top, idx
            self._index = _NumpyFlatIP()
            self._index.add(embs)

        self._meta.extend(items)
        return len(items)

    def save(self):
        if not self.storage_dir:
            raise ValueError("storage_dir not set")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        # numpy 后备索引不写 faiss 文件
        if _HAS_FAISS and isinstance(self._index, faiss.Index):
            faiss.write_index(self._index, str(self.storage_dir / "index.faiss"))
        (self.storage_dir / "meta.json").write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (self.storage_dir / "config.json").write_text(
            json.dumps({"backend": self.backend, "model_name": self.model_name, "normalize": self._normalize},
                       ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    # ---------------- prompt 解析（数学） ----------------
    @staticmethod
    def _extract_role(prompt: str) -> str:
        m = ExperienceKB.ROLE_RE.search(prompt) or ExperienceKB.ROLE_FALLBACK.search(prompt)
        role = (m.group(1).strip() if m else "")
        return role.replace(" ", "_")

    @staticmethod
    def _extract_problem(prompt: str) -> str:
        m = re.search(r"Problem:\s*(.+?)(?:\n\s*\n|OUTPUT|HARD CONSTRAINTS|<solution_outline>|</solution_outline>)",
                      prompt, re.S | re.I)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _detect_sections(prompt: str) -> List[str]:
        toks = ["HARD CONSTRAINTS", "<solution_outline>", "</solution_outline>", "<gaps>", "<result>"]
        return [t for t in toks if t.lower() in prompt.lower()]

    @staticmethod
    def _key_tokens(problem: str) -> List[str]:
        # 取变量/关键字/数字（只是弱特征，以便 query 多样性）
        cands = re.findall(r"[A-Za-z]{1,10}|\d+|sin|cos|tan|log|exp|mod|gcd|lcm", problem)
        seen, out = set(), []
        for t in cands:
            if t not in seen:
                seen.add(t); out.append(t)
        return out[:12]

    @staticmethod
    def _make_queries(role: str, problem: str, sections: List[str]) -> List[str]:
        qs = []
        if role:
            qs.append(f"role={role} global math MDT action rules solution_outline steps")
        for s in sections:
            qs.append(f"role={role} section={s} best_practices pitfalls")
        for k in ExperienceKB._key_tokens(problem):
            qs.append(f"role={role} token={k} step_structuring invariants edge_cases review_classes")
        return qs

    # ---------------- search/augment ----------------
    def _ensure_lazy_rebuild(self):
        """Rebuild an in-memory index from meta if index file is missing."""
        if self._index is None and self._meta:
            texts = [m.get("text", "") for m in self._meta if m.get("text")]
            if texts:
                embs = self._encode(texts)
                self._ensure_index(embs.shape[1])
                try:
                    self._index.add(embs)
                except Exception:
                    # fallback to numpy inside _ensure_index already handled
                    self._ensure_index(embs.shape[1])
                    self._index.add(embs)

    def _search(self, queries: List[str], top_k_per_query: int = 8) -> List[Dict]:
        if not self._meta:
            return []
        self._ensure_lazy_rebuild()
        if self._index is None:
            return []
        qv = self._encode(queries)
        scores: Dict[int, float] = {}
        for qe in qv:
            qe = np.expand_dims(qe, 0)
            sims, idxs = self._index.search(qe, top_k_per_query)
            for score, idx in zip(sims[0], idxs[0]):
                if idx < 0: continue
                scores[idx] = scores.get(idx, 0.0) + float(score)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        seen = set()
        for idx, acc in ranked:
            it = self._meta[idx].copy()
            if it["key"] in seen: continue
            it["score"] = acc
            results.append(it)
            seen.add(it["key"])
        return results

    @staticmethod
    def _compose_hints(results: List[Dict], max_items: int = 8) -> str:
        lines = []
        for it in results[:max_items]:
            lines.append(f"- ACTION: {it.get('key','')}\n  EXPERIENCE: {it.get('experience','')}")
        return "\n".join(lines)

    @staticmethod
    def _append_hints(prompt: str, hints: str) -> str:
        if not hints: return prompt
        sep = "\n\n===== EXPERIENCE HINTS (consult; do not quote verbatim) =====\n"
        tail = "\n===== END OF EXPERIENCE HINTS =====\n"
        return f"{prompt}{sep}{hints}{tail}"

    def augment_prompt(self, prompt: str, *, top_k_per_query: int = 8, max_hints: int = 8) -> str:
        role = self._extract_role(prompt)
        prob = self._extract_problem(prompt)
        secs = self._detect_sections(prompt)
        queries = self._make_queries(role, prob, secs)
        results = self._search(queries, top_k_per_query=top_k_per_query)
        hints = self._compose_hints(results, max_items=max_hints)
        return self._append_hints(prompt, hints)

    # ---------------- 方便的构建器 ----------------
    @classmethod
    def from_json(cls, json_path: str, storage_dir: Optional[str] = None,
                  *, backend: str = "transformers", model_name: Optional[str] = None,
                  normalize: bool = True, device: Optional[str] = None, batch_size: int = 32) -> "ExperienceKB":
        kb = cls(storage_dir=storage_dir, backend=backend, model_name=model_name,
                 auto_load=False, normalize=normalize, device=device, batch_size=batch_size)
        kv = json.loads(Path(json_path).read_text(encoding="utf-8-sig"))
        kb.ingest_kv(kv)
        if storage_dir: kb.save()
        return kb

    @classmethod
    def from_jsonl_dir(cls, dir_path: str, storage_dir: Optional[str] = None,
                       *, backend: str = "transformers", model_name: Optional[str] = None,
                       normalize: bool = True, device: Optional[str] = None, batch_size: int = 32) -> "ExperienceKB":
        """
        把整个目录聚合为一个库（老功能）。
        """
        kb = cls(storage_dir=storage_dir, backend=backend, model_name=model_name,
                 auto_load=False, normalize=normalize, device=device, batch_size=batch_size)
        kv = _merge_jsonl_dir(dir_path)
        kb.ingest_kv(kv)
        if storage_dir: kb.save()
        return kb

    @classmethod
    def from_jsonl_per_expert(cls, dir_path: str, out_root: str,
                              *, backend: str = "transformers", model_name: Optional[str] = None,
                              normalize: bool = True, device: Optional[str] = None, batch_size: int = 32) -> Dict[str, str]:
        """
        新功能：按专家拆分建库。
        - dir_path: 包含多个 *.experiences.jsonl 的目录
        - out_root: 输出根目录，每个文件生成 out_root/<ExpertName>/
        返回 {expert_dir_name: abs_path}
        """
        out_map: Dict[str, str] = {}
        Path(out_root).mkdir(parents=True, exist_ok=True)
        for fp in sorted(glob.glob(os.path.join(dir_path, "*.experiences.jsonl"))):
            base = os.path.basename(fp)                     # e.g., "Algebra.experiences.jsonl"
            expert = base.replace(".experiences.jsonl", "") # "Algebra"
            expert = _normalize_expert_name(expert)         # "Algebra" / "Number_Theory" ...
            storage_dir = str(Path(out_root) / expert)
            kv = _kv_from_jsonl_file(fp)
            kb = cls(storage_dir=storage_dir, backend=backend, model_name=model_name,
                     auto_load=False, normalize=normalize, device=device, batch_size=batch_size)
            kb.ingest_kv(kv)
            kb.save()
            out_map[expert] = storage_dir
        return out_map


# ------------------------- CLI -------------------------
def _cli():
    ap = argparse.ArgumentParser(description="Math ExperienceKB builder & augmenter")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # 单库：把整个目录聚合为一个库，或给定聚合后的 json
    b = sub.add_parser("build", help="Build ONE KB from a jsonl dir (merged) or an aggregated json")
    b.add_argument("--source_dir", help="Directory containing *.experiences.jsonl")
    b.add_argument("--source_json", help="Aggregated KV json file")
    b.add_argument("--storage_dir", required=True, help="Output directory for the single KB")
    b.add_argument("--backend", choices=["transformers","openai"], default="transformers")
    b.add_argument("--model_name", default=None)
    b.add_argument("--normalize", action="store_true", default=True)
    b.add_argument("--no-normalize", dest="normalize", action="store_false")
    b.add_argument("--device", default=None)
    b.add_argument("--batch_size", type=int, default=32)

    # 新增：按专家拆分，每个文件 -> 一个子库
    p = sub.add_parser("build-per-expert", help="Build per-expert KBs from a dir of *.experiences.jsonl")
    p.add_argument("--source_dir", required=True, help="Directory containing *.experiences.jsonl")
    p.add_argument("--out_root", required=True, help="Output root; each expert becomes a subfolder here")
    p.add_argument("--backend", choices=["transformers","openai"], default="transformers")
    p.add_argument("--model_name", default=None)
    p.add_argument("--normalize", action="store_true", default=True)
    p.add_argument("--no-normalize", dest="normalize", action="store_false")
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=32)

    a = sub.add_parser("augment", help="Augment a prompt file using a saved KB")
    a.add_argument("--storage_dir", required=True)
    a.add_argument("--prompt_file", required=True)
    a.add_argument("--top_k_per_query", type=int, default=8)
    a.add_argument("--max_hints", type=int, default=8)

    args = ap.parse_args()

    if args.cmd == "build":
        if not args.source_dir and not args.source_json:
            raise SystemExit("Please provide --source_dir OR --source_json")
        if args.source_dir:
            ExperienceKB.from_jsonl_dir(
                args.source_dir, storage_dir=args.storage_dir,
                backend=args.backend, model_name=args.model_name,
                normalize=args.normalize, device=args.device, batch_size=args.batch_size
            )
        else:
            ExperienceKB.from_json(
                args.source_json, storage_dir=args.storage_dir,
                backend=args.backend, model_name=args.model_name,
                normalize=args.normalize, device=args.device, batch_size=args.batch_size
            )
        print(f"[OK] Single KB saved to {args.storage_dir}")

    elif args.cmd == "build-per-expert":
        out_map = ExperienceKB.from_jsonl_per_expert(
            args.source_dir, args.out_root,
            backend=args.backend, model_name=args.model_name,
            normalize=args.normalize, device=args.device, batch_size=args.batch_size
        )
        print("[OK] Built per-expert KBs:")
        for k, v in out_map.items():
            print(f"  - {k}: {v}")

    elif args.cmd == "augment":
        kb = ExperienceKB(storage_dir=args.storage_dir, auto_load=True)
        raw = Path(args.prompt_file).read_text(encoding="utf-8-sig")
        out = kb.augment_prompt(raw, top_k_per_query=args.top_k_per_query, max_hints=args.max_hints)
        print(out)


if __name__ == "__main__":
    _cli()
