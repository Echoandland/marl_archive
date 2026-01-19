# Math-MDT (RARE) Pipeline — End-to-End

This repo runs a full loop: **multi-agent solving → scoring → experience extraction → KB build → solving with KB**.

---

## 1) First pass (no experience)

```bash
cd agent-math
nohup python main_async_math.py \
  --num_samples 1 \
  --free_recruitment \
  > 23_Oct_baseline.log 2>&1 &
```

Outputs:
- Results: `output/gpt-5_HLE_none_RARE_free_recruited/problem_0_free_recruited.json`
- Interaction logs: `output/interact/gpt-5_HLE_none_RARE_free_recruited/problem_0_interaction_free_recruited.json`

---

## 2) Score interactions (global correctness + per-step influence)

```bash
python interaction_to_score_async_math.py \
  --data_dir    output/interact/gpt-5_HLE_none_RARE_free_recruited \
  --results_dir output/gpt-5_HLE_none_RARE_free_recruited \
  --w_global 0.6 --w_step 0.4 --decay 0.9 --verbose
```

Produces per-problem `*_score.json`, e.g.  
`output/interact/gpt-5_HLE_none_RARE_free_recruited/problem_0_free_recruited_score.json`.

---

## 3) Extract transferable experiences from scores

Per-expert JSONL files:

```bash
python score_to_experience_util_async_math.py \
  --data_dir    output/interact/gpt-5_HLE_none_RARE_free_recruited \
  --results_dir output/gpt-5_HLE_none_RARE_free_recruited \
  --out_dir     output/math_experiences
```

Optional unified (single file) experience set:

```bash
python score_to_experience_util_async_math.py \
  --data_dir    output/interact/gpt-5_HLE_none_RARE_free_recruited \
  --results_dir output/gpt-5_HLE_none_RARE_free_recruited \
  --out_dir     output/math_experiences \
  --unified_experience
```

---

## 4) Build the Experience KB

### 4.1 Unified (single) KB
```bash
python experience_api_math.py build \
  --source_dir  output/math_experiences \
  --storage_dir kb/experience_extractor
```

### 4.2 Per-expert KBs
```bash
python experience_api_math.py build-per-expert \
  --source_dir output/math_experiences \
  --out_root   kb/math_experts
```

Artifacts are written under:
- `kb/experience_extractor/` (single KB)
- `kb/math_experts/<ExpertName>/` (per-expert)

---

## 5) Point the runtime to your KB

Set paths (used by `utils_with_experience_math.py`):

```bash
export MATH_KB_ROOT="kb/math_experts"         # per-expert KB root (preferred)
export MATH_KB_DIR="kb/experience_extractor"  # unified KB (fallback or unified mode)
```

At runtime you’ll see messages like:
- `[KB] Loaded per-expert KB: ...`
- `[KB] Fallback to global KB: ...`
- `[KB] No KB found ... Experience augmentation disabled.`

---

## 6) Second pass (with experience)

### 6.1 Unified experience (all agents share one KB)
```bash
nohup python main_async_math.py \
  --num_samples 1 \
  --free_recruitment \
  --use_experience \
  --unified_experience \
  > 23_Oct_with_unified_kb.log 2>&1 &
```

### 6.2 Per-expert experience (each role uses its own KB，useful only when free_recruitment is disabled during interaction construction)
```bash
nohup python main_async_math.py \
  --num_samples 1 \
  --free_recruitment \
  --use_experience \
  > 23_Oct_with_per_expert_kb.log 2>&1 &
```

---

## What gets written where

```
output/
  gpt-5_HLE_none_RARE_free_recruited/
    problem_0_free_recruited.json               # final result
  interact/gpt-5_HLE_none_RARE_free_recruited/
    problem_0_interaction_free_recruited.json   # interaction log
    problem_0_free_recruited_score.json         # scored steps
  math_experiences/
    <Role>.experiences.jsonl                    # per-role experiences
    Unified.experiences.jsonl                   # (optional) unified set
kb/
  experience_extractor/                         # unified KB (index/meta/config)
  math_experts/
    <ExpertA>/                                  # per-expert KBs
    <ExpertB>/
```

---

## Script cheat-sheet

- `main_async_math.py` — Multi-round RARE flow (recruit → interact → chair).
- `interaction_to_score_async_math.py` — Global correctness + per-step influence + temporal decay allocation.
- `score_to_experience_util_async_math.py` — Turn top-K steps → transferable OPINION/REVIEW experiences.
- `experience_api_math.py` — Build/serve KBs (unified or per-expert).
- `utils_with_experience_math.py` — Agent & Group; loads KBs and augments prompts.

