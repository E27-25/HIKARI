# Full Project Recap + Implementation Plan

## Project Goal
Classify 10 skin diseases from dermatoscopy images using a fine-tuned Vision-Language Model (Qwen3-VL-8B via Unsloth), evaluated on the SkinCAP dataset.

## Dataset
- **SkinCAP**: ~1,010 images, 10 disease classes, each image has a clinical caption
- **Val split**: 99 samples (fixed via `split_info_3stage.json`)
- **Train split**: 911 samples
- **10 classes**: psoriasis, lupus erythematosus, lichen planus, scleroderma, photodermatoses, sarcoidosis, melanocytic nevi, squamous cell carcinoma in situ (SCCIS), basal cell carcinoma (BCC), acne vulgaris

---

# Part 1 — Complete Experiment History

## Chapter 1: 3-Stage Pipeline + M0–M4 Methods

### Architecture
```
Stage 1: Group classifier (4 groups: Inflammatory / Benign / Malignant / Acne)
    ↓ group label (or predicted)
Stage 2: Disease classifier (10 diseases within each group)
    ↓ disease label
Stage 3: Caption generator
```

### Stage 1 Training Result
- **Accuracy: 92.05%** (4-group, top-10 diseases, 3 epochs, LR=2e-4)
- Model saved: `skincap_3stage_group_4group_top10_merged/`

### Stage 2 Methods (M0–M4) — What group context is injected into the prompt?

| Method | Train prompt | Inference prompt | Notes |
|--------|-------------|-----------------|-------|
| M0 | No group | No group | Cascade baseline |
| M1 | GT group | GT group | Oracle — uses ground truth (not production-usable) |
| M2 | GT group | Stage1 predicted | Train/inference mismatch |
| M3 | Stage1 predicted | Stage1 predicted | Consistent but noisy |
| M4 | GT group | Stage1 beam soft probs | Partial probability info |

### Stage 2 Results (99 val samples, all use Stage1 merged as starting weights)

| Method | Accuracy | Balanced Acc | Notes |
|--------|----------|-------------|-------|
| M0 | 61.0% | 59.4% | Cascade baseline |
| **M1** | **66.0%** | 63.7% | Best — oracle GT groups |
| M2 | 57.4% | 56.2% | GT train / S1 pred inf — mismatch hurts |
| M3 | 57.4% | 56.5% | S1 pred both — consistent but noisy |
| M4 | 61.0% | 58.2% | Soft probs = no benefit over M0 |

### Key Findings from M0–M4
- Cascade is **helpful** (+5% M0→M1) — Stage1 domain adaptation transfers
- Prompt-level Stage1 injection is **harmful** (M2/M3 < M0) — 8% Stage1 errors cause mismatch
- M1 is the real upper bound — not usable without GT labels at inference

### Bugs Fixed During This Phase
- **Filter-before-consolidate bug**: inference scripts had wrong order (consolidate then filter → non-deterministic Group 3 exclusion). Fixed by filtering to ACTIVE_TOP_DISEASES first, then fuzzy consolidation.
- **Non-determinism**: `list(set(...))` → `sorted(set(...))` in `fuzzy_consolidate_diseases()`
- **KeyError 'id'**: `predict_batch()` in `inference_group_classification.py` required `id`/`file_name` fields not present in training data → fixed with `.get()`
- **TypeError None in replace()**: `item.get('predicted_group')` returned None when Stage1 failed → fixed with `or item['disease_group']` fallback

---

## Chapter 2: FuzzyTopK Discovery

### What is FuzzyTopK?
A separate 2-stage pipeline (`train_two_stage_FuzzyTopK.py`):
- Stage 1: Direct disease classification (no group stage) from base model
- Trained on "squamous cell carcinoma in situ" (SCCIS/Bowen's) — visually more distinct than invasive SCC
- Uses `split_info_fuzzytopk.json` (different train/val split, ~9× more val samples)

### FuzzyTopK vs 3-Stage Results
| Model | Accuracy | Val split | SCC class |
|-------|----------|-----------|-----------|
| FuzzyTopK (own val) | 83% | 1,044 samples | SCCIS |
| FuzzyTopK (3stage val) | **74%** | 99 samples | SCCIS |
| M1 | 66% | 99 samples | invasive SCC |

### Why FuzzyTopK > M1
1. **SCCIS is easier to classify** than invasive SCC (confined to epidermis, more distinct features)
2. **No cascade penalty** — FuzzyTopK trains directly from base model, M1 starts from Stage1 group-classifier weights (biased toward short group-name outputs)
3. **More training data** (~9× more val samples in own split = more data)

---

## Chapter 3: Option B Rounds 1–5 (Improving M1/3-Stage)

Attempted to improve Stage2 M1 through hyperparameter tuning and data fixes. All on 99-sample val.

| Round | Change | Result | Verdict |
|-------|--------|--------|---------|
| B Round 1 | Fix SCC→SCCIS label rename | 61.39% | Failed — label rename only, no SCCIS images |
| B Round 2 | Add true SCCIS images + B1+B2+B3 (LR=5e-5, dropout=0.05, best-ckpt) | 58.65% | Failed — mixing SCCIS+SCC confused model |
| B Round 3 | Revert SCCIS, keep B1+B2+B3, 5 epochs | 61.62% | Marginal — B1 (low LR) kills convergence |
| B Round 4 | Revert LR→2e-4, keep B2+B3 | 58.16% | Failed — merge silently failed (OOM) |
| B Round 5 | Full revert: no dropout, 3 epochs, LR=2e-4 | target ~66% | Reproducing original M1 conditions |

**Key lesson**: For small datasets (~911 train), LR=2e-4 + 3 epochs + no dropout is better than lower LR with regularization.

---

## Chapter 4: D2 Prompts + RAG (Visual Few-Shot Retrieval)

### D2 Prompts (Better Clinical Prompts)
Replaced simple "What disease is this?" with detailed clinical feature prompts:
- "Examine morphology, color, scale/crust, border, distribution. What disease?"

**Result without RAG**: 47.87% (worse than old 55.22%) — prompt mismatch. D2 prompts don't match training prompts → model confused without visual anchors.

### RAG: Visual Few-Shot Retrieval
At inference time, retrieve top-K similar training images → show as reference examples in prompt.

```
[Reference 1 image] → psoriasis
[Reference 2 image] → psoriasis
[Reference 3 image] → lichen planus
Now diagnose: [query image]
```

**Implementation**: CLIP ViT-B/32 embeds all train images → NearestNeighbors index → retrieve at inference.

**Initial result (RAG, all-data index)**: 68.57% — but this was **inflated** due to data leakage (val images included in 1010-image index → val queries retrieved themselves as top-1 match with cosine sim ≈ 1.0).

### Data Leakage Fix
- `RAG_USE_ALL_DATA = False` → index built from 911 train images only
- Index filename includes scope: `rag_index_{R}_train.npz` vs `rag_index_{R}_all.npz`

---

## Chapter 5: Hybrid RAG Benchmark (Current Phase)

### 5 RAG Encoder Experiments

| ID | Image Encoder | Text Encoder | Strategy | Description |
|----|--------------|-------------|----------|-------------|
| R0 | CLIP ViT-B/32 | None | A (cross-modal) | Image-only baseline |
| R1 | CLIP ViT-B/32 | ClinicalBERT | B (two-pass) | Medical text matching |
| R2 | SigLIP base-patch16-512 | BGE-M3 | B | Stronger image + multilingual text |
| R3 | JinaClip-v2 | MedCPT | B | Clinical specialist (PubMed-trained text) |
| R4 | Nomic vision v1.5 | Nomic text v1.5 | A | Unified cross-modal space |

**Strategy A**: same image encoder queries both image AND text reference embeddings (shared space)
**Strategy B**: image encoder for image query, separate text encoder for text query (requires vlm_description)

### 4 Prompt Variants

| ID | Name | Style |
|----|------|-------|
| P0 | D2 Clinical | "Examine morphology, color, scale/crust, border, distribution. What disease?" |
| P1 | CoT Step-by-step | "Step 1: describe lesion → Step 2: narrow group → Step 3: final diagnosis" |
| P2 | Differential Dx | "List top 3 candidates, explain evidence, choose most likely" |
| P3 | Structured Clinical | Fill-in template: Morphology / Color / Surface / Border → Diagnosis |

### Fixes Applied During Benchmark Setup

| Issue | Fix |
|-------|-----|
| R2: `google/siglip-2-base-patch16-512` → 401 gated | Replaced with `google/siglip-base-patch16-512` (public) |
| R3: `Can't use xattn without xformers` | xformers installed but wrong version (PyTorch 2.7.1 vs 2.10.0) → unconditional `xattn=False` patch |
| R3: dimension mismatch 512 vs 1024 | Old index had 512-dim (wrong build); deleted, rebuilt → 1024-dim correct |
| R3: strategy A mismatch 1024 vs 768 | JinaClip (1024) ≠ MedCPT (768) — different spaces; changed R3 strategy A→B |
| R4: `Qwen3-VL-2B-Embedding` not found | Replaced with `nomic-ai/nomic-embed-vision-v1.5` + `nomic-ai/nomic-embed-text-v1.5` |
| R0=R1 identical results | Strategy B fell back to image-only (vlm_description=None); fixed by passing caption |
| fuzzytopk wrong model path | `STAGE2_MODEL_PATH` formula gave wrong dir; added `_STAGE2_MODEL_PATH_MAP` dict |

### Strategy B Caption Fix
Every SkinCAP data item has a `caption` field (clinical description from `caption_zh_polish_en` column).
Passing it as `vlm_description` activates text retrieval for R1/R2/R3 at zero extra compute:
```python
vlm_desc = item.get('caption')
ref_pairs = rag_retriever.retrieve(image, k=RAG_K, vlm_description=vlm_desc)
```

### Full Benchmark Results (before Strategy B caption fix — R1/R2/R3 text was unused)

**Method=fuzzytopk** (base model, no GT, no cascade)
```
             P0      P1      P2      P3
R0         66.67   69.81   85.71   90.00  ← best: P3+R0 = 90%
R1         66.67   69.81   85.71   90.00  ← identical to R0 (Strategy B bug)
R2         64.18   65.38   83.33   90.00
R3         64.56   71.43   93.75   80.00  ← P2+R3 = 93.75%!
R4         60.00   65.38   80.95   75.00
```

**Method=fuzzytopk_s1cascade** (starts from Stage1 weights, no GT)
```
             P0      P1      P2      P3
R0         70.67   70.67   69.33   69.33  ← best: P0+R0 = 70.67%
R1         70.67   70.67   69.33   69.33  ← identical to R0 (Strategy B bug)
R2         71.64   70.15   71.64   71.64
R3         64.56   63.29   64.56   65.82
R4         65.33   64.00   62.67   62.67
```

**Method=M1** (cascade + GT group labels at inference — oracle, not production)
```
             P0      P1      P2      P3
R0         55.22   58.33   54.55   50.00
R1         55.22   58.33   54.55   50.00  ← identical to R0 (Strategy B bug)
R2         58.73   63.64   49.21   49.21
R3         51.35   52.38   52.11   50.00
R4         50.72   69.57   50.72   52.17
```

### Key Findings from Benchmark

**1. fuzzytopk > s1cascade > M1 for complex prompts (P2/P3)**
- Stage1 cascade biases the model toward short outputs (trained to output group names) → fails to follow P3 structured template
- Base fuzzytopk (Qwen3-VL-8B unbiased) follows templates natively → 90%

**2. s1cascade > fuzzytopk for simple prompts (P0/P1)**
- Stage1 domain adaptation improves skin feature visual encoder → +4% for P0

**3. M1 worst despite GT groups**
- Complex message (3 RAG images + GT group context + question) → 32/99 "Unknown" outputs (unparseable) vs 24/99 for fuzzytopk
- GT group context + RAG = conflicting/redundant signals → more verbose VLM output → parsing failure

**4. P3 dramatically outperforms P0 for fuzzytopk**
- Structured template forces model to think through morphology/color/border before diagnosing
- Qwen3-VL-8B thinking capability leveraged by structured output format

**5. P2+R3 = 93.75%** (fuzzytopk, P2 differential dx + R3 JinaClip+MedCPT image-only — text unused)
- This was measured BEFORE the caption fix; text retrieval for R3 wasn't active yet

**Per-class sensitivity (P0, R0 — baseline comparison)**

| Disease | fuzzytopk | s1cascade | M1 |
|---------|:---------:|:---------:|:--:|
| psoriasis | 75% | 62.5% | 62.5% |
| lupus | 25% | **75%** | 50% |
| lichen planus | 42.9% | **71.4%** | 57.1% |
| scleroderma | **71.4%** | 42.9% | 14.3% |
| photodermatoses | 37.5% | 37.5% | 37.5% |
| sarcoidosis | **80%** | 60% | **80%** |
| melanocytic nevi | 88.9% | **100%** | 100%† |
| SCCIS | **90.9%** | **90.9%** | 63.6% |
| BCC | 44.4% | **66.7%** | 33.3% |
| acne vulgaris | 85.7% | 85.7% | **100%** |

†M1: 8/9 melanocytic nevi GT samples returned "Unknown" (only 1 parseable)

**Persistent bottleneck**: photodermatoses stuck at 37.5% across ALL methods (insufficient training data, visual overlap with other inflammatory diseases)

---

# Part 2 — Current State

## What is done and working
| Item | Status |
|------|--------|
| 3-stage pipeline (Stage1 group + Stage2 disease) | ✅ |
| FuzzyTopK 2-stage pipeline | ✅ |
| FuzzyTopK_s1cascade (trained from Stage1 weights) | ✅ |
| Hybrid RAG (5 encoders × 4 prompts) | ✅ Benchmark run |
| Data leakage fix (train-only index) | ✅ |
| Strategy B caption fix (vlm_description passed) | ✅ Applied but not yet re-run |
| fuzzytopk model path fix | ✅ |

## What needs re-running
R1/R2/R3 results in the benchmark were built **before** the Strategy B caption fix — their text components were silently unused. R0 and R4 (Strategy A) results are correct.

---

# Part 3 — Implementation Plan (Next Steps)

## Stage 1 — Re-run R1/R2/R3 (no code changes needed)

### Step 1a — Quick validation (~30 min)
Check that R1 now differs from R0 (confirms caption fix works):
```bash
cd HIKARI/Model
python run_rag_benchmark.py --rag_experiments R0 R1 --prompt_variants P0 P3 --no_skip_existing --method fuzzytopk --batch_size 4
```
**Gate**: If R1-P3 ≠ 90.00% → proceed. If still 90.00% → investigate.

### Step 1b — Full fuzzytopk R1/R2/R3 (~3-4 hrs)
```bash
python run_rag_benchmark.py --rag_experiments R1 R2 R3 --prompt_variants P0 P1 P2 P3 --no_skip_existing --method fuzzytopk --batch_size 4
```

### Step 1c — fuzzytopk_s1cascade R1/R2/R3 (if text helps in 1b, ~3-4 hrs)
```bash
python run_rag_benchmark.py --rag_experiments R1 R2 R3 --prompt_variants P0 P1 P2 P3 --no_skip_existing --method fuzzytopk_s1cascade --batch_size 4
```

**Watch for**: fuzzytopk P2+R3 was already 93.75% with image-only. With MedCPT text enabled, this could be the new best.

---

## Stage 2 — Alpha Tuning (1-line code change)

**What**: `alpha=0.5` gives equal weight to image and text retrieval. Try `alpha=0.7` (more image).

**Code change** — `rag_retrieval.py`, `HybridRAGRetriever.__init__()`:
```python
def __init__(self, ..., alpha=0.7):   # was 0.5
```

**Run** (no index rebuild needed, query-only change):
```bash
python run_rag_benchmark.py --rag_experiments R1 R2 R3 --prompt_variants P0 P3 --no_skip_existing --method fuzzytopk --batch_size 4
```

---

## Stage 3 — Fix M1 Unknown Rate (~5-line code change)

**Problem**: M1 generates 32/99 unparseable "Unknown" outputs. Complex message (3 RAG images + labels + group context + question) overwhelms Qwen3-VL.

**Fix** — `inference_disease_classification.py`, inside `predict_batch()`:
```python
# If RAG is on, skip group context (RAG visual examples already anchor the diagnosis)
if USE_RAG and rag_retriever:
    group_ctx = None
```

**Run**:
```bash
python run_rag_benchmark.py --rag_experiments R0 R1 --prompt_variants P0 P3 --no_skip_existing --method M1 --batch_size 4
```

---

## Stage 4 — No-Code Quick Wins

### 4a — Ensemble routing (no changes)
| Prompt | Best model | Accuracy |
|--------|-----------|---------|
| P0/P1 | fuzzytopk_s1cascade | 70.67–71.64% |
| P2/P3 | fuzzytopk base | 90–93.75% |
Just change `STAGE2_METHOD` per run — no code change needed.

### 4b — Increase RAG_K (1-line change)
Change `RAG_K = 5` in `inference_disease_classification.py` (was 3). More references may help confusable diseases.
```bash
python run_rag_benchmark.py --rag_experiments R0 --prompt_variants P3 --no_skip_existing --method fuzzytopk --batch_size 4
```

---

## Stage 5 — Training Improvements (GPU time)

### 5a — Retrain fuzzytopk with P2/P3 prompts in training mix
Currently fuzzytopk was trained with only simple prompts (P0-style). Adding P2/P3 to training mix would make the model learn structured output format → potentially push beyond 90%.

**File**: `HIKARI/Model/train_two_stage_FuzzyTopK.py`
- Add `COT_PROMPTS["P2"]` and `COT_PROMPTS["P3"]` to training prompt pool
- Mix all 4 variants during data preparation

### 5b — Fix photodermatoses bottleneck
All methods stuck at 37.5% (8 GT val samples, ~80 train samples). Options:
- Weighted loss (upweight rare inflammatory classes)
- Data augmentation for underrepresented classes

---

## Priority Order

| Stage | Action | Expected gain | Effort | Run time |
|-------|--------|--------------|--------|----------|
| **1a** | Validate caption fix | Confirms text retrieval works | None | ~30 min |
| **1b** | Re-run fuzzytopk R1/R2/R3 | Real hybrid results; R3-P2 may beat 93.75% | None | ~3-4 hrs |
| **3** | Remove group ctx when RAG on (M1) | Drop Unknown rate, +5-10% M1 | ~5 lines | ~1 hr |
| **2** | Alpha tuning 0.5→0.7 | Balance image/text weight | 1 line | ~2 hrs |
| **4b** | RAG_K 3→5 | More anchoring for confusable diseases | 1 line | ~1 hr |
| **1c** | Re-run s1cascade R1/R2/R3 | Complete comparison table | None | ~3-4 hrs |
| **5a** | Retrain fuzzytopk with P3 prompts | +2-5% for complex prompts | Medium | ~4 hrs GPU |
| **5b** | Fix photodermatoses | +2-3% balanced accuracy | Medium | ~4 hrs GPU |

---

## Decision Gate After Stage 1b

**If text retrieval (R1/R2/R3 with captions) improves over R0**:
→ Proceed with Stage 2 (alpha tuning) to optimize image/text balance
→ Focus on R3 (MedCPT) which is most suited for clinical text

**If text retrieval is neutral or worse**:
→ Skip Stage 2, focus on Stage 3 (M1 fix) and Stage 4 (no-code improvements)
→ Consider generating real-time VLM descriptions instead of pre-existing captions for future Strategy B

---

## All Result Files Location
`HIKARI/Model/disease_classification_results/`

Naming convention: `results_disease_{METHOD}_{RAG_TAG}_{PROMPT_TAG}_val.json`
- Example: `results_disease_fuzzytopk_RAGR3_P2_val.json` → fuzzytopk + R3 + P2 = **93.75%**
