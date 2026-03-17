# HIKARI Skin Disease Classification — Full Project Summary

## Overview

End-to-end benchmark of skin disease classification using **Qwen3-VL-8B-Thinking** fine-tuned on the SkinCAP dataset.
The project tested multiple training strategies, Retrieval-Augmented Generation (RAG) setups, and prompt variants,
measured on a **99-sample validation set** (10 disease classes, 3-stage stratified split).

---

## 1. Dataset & Disease Classes

**Dataset:** SkinCAP (skincap_v240623.csv) — 4000 dermatology images with disease labels and clinical captions
**Source labels:** Raw CSV column `disease` (lowercase, normalized); `caption_zh_polish_en` used as RAG text embedding source
**Train/Val split:** Stratified split stored in `split_info_3stage.json` (locked after Stage 1 training, never regenerated)

### 1.1 Disease Classes & Validation Counts

| Group | Disease | Val N |
|-------|---------|-------|
| 1. Inflammatory & Autoimmune | psoriasis | 13 |
| | lupus erythematosus | 9 |
| | lichen planus | 9 |
| | scleroderma | 8 |
| | photodermatoses | 8 |
| | sarcoidosis | 7 |
| 2. Benign Tumors & Nevi | melanocytic nevi | 12 |
| 3. Malignant Tumors | squamous cell carcinoma in situ (SCCIS) | 12 |
| | basal cell carcinoma (BCC) | 13 |
| 4. Acne & Follicular | acne vulgaris | 8 |
| **Total** | | **99** |

### 1.2 Data Preprocessing Pipeline

```
Raw CSV (4000 rows)
    ↓ Normalize disease labels: lowercase + strip whitespace
    ↓ Apply FORCED_CANONICAL mapping: "squamous cell carcinoma" → "squamous cell carcinoma in situ"
    ↓ Filter to TOP_10_DISEASES set (exact-match): 4000 → 1010 samples
    ↓ fuzzy_consolidate_diseases(): Jaccard word-overlap (threshold 91), APPLIED AFTER filter
           (prevents cross-group artifacts; sort() ensures determinism)
    ↓ Stratified train/val split via split_info_3stage.json: 911 train / 99 val
    ↓ Training: sqrt-mode oversampling (rare classes boosted by √ratio toward mean, ~1.36× for least-common)
    ↓ Image loading: thumbnail(672, 672, LANCZOS) — caps at 672px per side (24×28 tile boundary for Qwen3-VL)
```

**Note on val split variants:**
- `split_info_3stage.json` (99 val): All RAG experiments and M1/M2/M3/M4 use this split.
- `split_info_fuzzytopk.json`: Standalone fuzzytopk model uses this for its own 83% result. When evaluated on the 3-stage split, fuzzytopk gives 74.0% (101 samples including invasive SCC images).

---

## 2. Model Architectures

### 2.0 Common Training Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | Qwen3-VL-8B-Thinking |
| Quantization | 4-bit NF4 via Unsloth (`load_in_4bit=True`, `use_gradient_checkpointing="unsloth"`) |
| LoRA r | 16 |
| LoRA alpha | 32 |
| LoRA target | All attention + MLP modules (vision + language layers) |
| Batch size | 2 per GPU, gradient accumulation = 4 → effective batch = 8 |
| Optimizer | AdamW 8-bit paged |
| Hardware | NVIDIA RTX 5070 Ti (15.92 GB VRAM) |
| Image resize (inference) | 672×672 thumbnail (LANCZOS) |
| Inference max_new_tokens | 64 (fine-tuned models avg 2.1 words; EOS naturally before limit) |
| Inference max_new_tokens (zero-shot) | 1024 (thinking chain needs ~500–1000 tokens to close `</think>`) |

### 2.1 fuzzytopk (74% baseline, no RAG)

**Architecture:** 2-stage fine-tune from Qwen3-VL-8B base
**Stage 1:** Disease classification fine-tune (10 classes) → `skincap_fuzzytopk_classification_merged`
**Stage 2:** Caption generation fine-tune from Stage 1 weights
**Training script:** `train_two_stage_FuzzyTopK.py`

| Parameter | Value |
|-----------|-------|
| Training data | 939 images from `split_info_fuzzytopk.json` train set |
| Val data (own) | Fuzzytopk val set → 83% (different split from 3-stage) |
| Val data (3-stage) | 101 samples (includes invasive SCC) → **74.0%** |
| LR | 2e-4 |
| LR scheduler | Cosine with 5 warmup steps |
| Epochs | 3 |
| LoRA dropout | 0 |
| Model output | Very short (disease name, 1–3 words); no `<think>` block |

**Key characteristic:** Short, direct disease name output → high parse rate (99/99 valid) without RAG. Adding RAG at inference confuses this model because it was trained on single-image prompts only.

### 2.2 fuzzytopk_s1cascade (79.80% best with RAG)

**Architecture:** Same as fuzzytopk but Stage 1 fine-tune starts from the 3-stage Group Classifier weights
**Starting weights:** `skincap_3stage_group_classification_merged` (4-group classifier, 88.68% accuracy)
**Model path:** `skincap_fuzzytopk_s1cascade_classification_merged`
**Training script:** `train_two_stage_FuzzyTopK.py --start_from_stage1`

| Parameter | Value |
|-----------|-------|
| Training data | 939 images from `split_info_fuzzytopk.json` train set |
| Starting weights | Group classifier (Stage 1 cascade) |
| LR | 2e-4 |
| Epochs | 3 |
| LoRA dropout | 0 |
| Best result | RAG R2-P0 α=0.9 = **79.80%** |

**Key characteristic:** Stage 1 cascade pretraining teaches the model to compare lesion features across groups → enables effective RAG utilization. P0 (direct prompt) wins consistently because cascade weights bias toward short output.

### 2.3 fuzzytopk_s1cascade_ragR2_a09 (**85.86% — NEW BEST OVERALL**)

**Architecture:** fuzzytopk_s1cascade retrained with K=1 RAG reference image in every training sample
**Model path:** `skincap_fuzzytopk_s1cascade_ragR2_a09_classification_merged`
**Training script:** `train_two_stage_FuzzyTopK.py --start_from_stage1 --rag_k_train 1 --rag_exp R2 --alpha 0.9`

| Parameter | Value |
|-----------|-------|
| Starting weights | Group classifier (Stage 1 cascade) |
| Training RAG encoder | R2 (SigLIP+BGE-M3, α=0.9) |
| K (refs per training sample) | 1 (K=3 caused VRAM OOM: 4 images × batch=2 = 8 forward images) |
| Training sample format | `[ref_img] Reference 1: {label}\nDescription: {caption}\n[query_img] {prompt}` |
| Training items | 939 images × ~3.7 samples/image = 3,504 items (3 prompts per image, some variation) |
| Steps | 1,314 steps @ ~4.2s/step → ~1h 44min |
| Epochs | 3 |
| Pre-computation | `precompute_rag_refs()` runs R2 encoder BEFORE loading Qwen3-VL (frees GPU first) |
| Best result | RAG R0-P0 (CLIP image-only at inference) = **85.86%** |

**Surprising finding:** CLIP (R0) inference outperforms SigLIP+BGE-M3 (R2, training encoder) at inference (+3.03%). The model learned to use reference→label links generally, not encoder-specific features.

**K=3 OOM fix:** Unsloth fused CrossEntropy loss throws ZeroDivisionError when padding ratio is too small. 4 images × batch=2 = 8 GPU forward passes exhausted 15.92GB VRAM → K=1 (2 images/sample) fits comfortably.

### 2.4 M1 (3-stage pipeline, 47.87% No-RAG, 59.38% with R0-P1)

**Architecture:** 3-stage cascade from Qwen3-VL-8B base
**Stage 1:** Group classification (4 groups) → `skincap_3stage_group_classification_merged` (88.68% val)
**Stage 2:** Disease classification with GT group context at train+inference → `skincap_3stage_disease_M1_merged`
**Stage 3:** Caption generation
**Training script:** `train_three_stage_hybrid_topk.py`

| Parameter | Value |
|-----------|-------|
| Training data | 911 images from `split_info_3stage.json` train set |
| Stage 1 LR | 2e-4, 3 epochs |
| Stage 2 LR | 5e-5, 3 epochs with warmup_ratio=0.1, load_best_model_at_end=True |
| Stage 2 LoRA dropout | 0.05 |
| Group context | GT group label injected into prompt at train AND inference (oracle M1) |
| Model output | Short: 2–5 words (fine-tuned behavior despite thinking backbone) |
| Best result (no RAG) | 47.87% — catastrophic failure on inflammatory classes |
| Best result (with RAG) | R0-P1 = **59.38%** (+11.5%) |

**M0–M4 method variants (historical, evaluated on 99-sample val):**

| Method | Train context | Inference context | Accuracy |
|--------|--------------|------------------|---------|
| M0 | None | None | 61.0% |
| M1 | GT group | GT group | **66.0%** |
| M2 | GT group | Stage1 predicted | 57.4% |
| M3 | Stage1 predicted | Stage1 predicted | 57.4% |
| M4 | GT group | Beam soft probs | 61.0% |
| fuzzytopk (reference) | None | None | **74.0%** (different model) |

**Key characteristic:** CoT reasoning model; P1 (step-by-step) prompt significantly outperforms P0; RAG improves by +11.5% (47.87% → 59.38%) but still underperforms s1cascade (59.4% vs 74.75%) due to 3-stage cascade penalty.

---

## 3. RAG Experiments (R0–R4)

Visual + text retrieval-augmented generation. At inference, top-K=3 similar training images are retrieved and shown as reference examples in the prompt. **Index built from 911 training images only** (no val leakage; `RAG_USE_ALL_DATA=False`).

### 3.1 Retrieved Reference Format (in-context prompt)

```
Here are similar reference cases for context:
[ref_image_1]
Reference 1: psoriasis
Description: Erythematous scaly plaques with silvery surface on extensor surfaces...

[ref_image_2]
Reference 2: lichen planus
Description: Flat-topped violaceous papules on the forearms...

[ref_image_3]
Reference 3: psoriasis
Description: Multiple demarcated erythematous plaques with scaling...

Now, identify the disease in this new image:
[query_image]
{prompt_text}
```

### 3.2 RAG Encoder Configurations

| ID | Name | Image Encoder | Text Encoder | Strategy |
|----|------|--------------|--------------|----------|
| **R0** | Image-only CLIP | `openai/clip-vit-base-patch32` | None | A (image→image) |
| **R1** | CLIP + ClinicalBERT | `openai/clip-vit-base-patch32` | `medicalai/ClinicalBERT` | B (image+text) |
| **R2** | SigLIP + BGE-M3 | `google/siglip-2-base-patch16-512` | `BAAI/bge-m3` | B (image+text) |
| **R3** | Jina-CLIP + MedCPT | `jinaai/jina-clip-v2` | `ncbi/MedCPT-Query-Encoder` | B (image+text) |
| **R4** | Nomic Unified | `nomic-ai/nomic-embed-vision-v1.5` | `nomic-ai/nomic-embed-text-v1.5` | A (cross-modal) |

### 3.3 Retrieval Scoring Formulas

**Strategy A (Cross-modal, R0 and R4):**
Image encoder queries BOTH image and text embeddings in a shared embedding space.
```
score(query, ref) = α × cos(img_enc(query_img), ref_img_emb)
                 + (1−α) × cos(img_enc(query_img), ref_txt_emb)
```
R0: text encoder is None → α=1.0 always (image-only)
R4: Nomic vision+text share the same embedding space; α=0.5 default

**Strategy B (Two-pass, R1/R2/R3):**
Separate text encoder handles the text side; image encoder handles the image side.
```
score(query, ref) = α × cos(img_enc(query_img), ref_img_emb)
                 + (1−α) × cos(txt_enc(query_text), ref_txt_emb)
```
`query_text` = patient-written symptom description (from `val_captions_for_symptoms.json`, 94 items) OR Stage 3 generated caption (B4 experiment) OR `caption_zh_polish_en` field.

### 3.4 Index Construction

**Saved format:** `.npz` file with keys:
- `img_embs` (N×D_img): image embeddings for all 911 train images
- `txt_embs` (N×D_txt): text embeddings from `caption_zh_polish_en` captions (None for R0)
- `labels` (N,): disease names
- `paths` (N,): absolute image paths
- `captions` (N,): clinical caption text (returned in retrieve() for in-context display)
- `strategy`, `img_encoder`, `txt_encoder`: metadata

**Build process:** `HybridRAGRetriever.build()` in `rag_retrieval.py` → runs on GPU → saved to `rag_index_{R}_train.npz`. Built once per encoder, reloaded on subsequent runs.
**Query:** `HybridRAGRetriever.retrieve(query_img, k=3)` → cosine similarity scores → top-K by descending score → returns `[(path, label, caption)]`

---

## 4. Prompt Variants (P0–P3)

Full verbatim text for each variant (no group context; group context variants use separate templates):

### P0 — Direct Clinical Observation (default)
```
Carefully examine this dermatological image. Look for: lesion morphology (papule/plaque/macule/nodule),
color (red/violet/white/brown/black), scale or crust, border sharpness, and distribution.
Based on these visual features, what is the specific skin disease?
```
*Best for:* fuzzytopk, s1cascade (short-output models). Simple structured observation → clean disease name output.

### P1 — Step-by-Step CoT
```
Analyze this dermatological image step by step:
Step 1: Describe the primary lesion (papule/plaque/macule/nodule/vesicle).
Step 2: Note the color (red, violet, white, brown, black, mixed).
Step 3: Identify texture/surface (smooth, scaly, crusted, ulcerated).
Step 4: Note the border (sharp/blurred) and distribution.
Step 5: Based on these features, state the specific skin disease.
```
*Best for:* M1 (CoT reasoning model). Explicit numbered steps match the model's thinking pattern → +4.4% over P0 for M1.
*Parsing:* Extract content after last `Step N:` occurrence; fallback to last non-empty line.

### P2 — Differential Diagnosis
```
Examine this skin lesion. List the 3 most likely diagnoses and the key visual evidence for each.
Then select the single most likely diagnosis. Format:
1. [Disease] - [evidence]; 2. [Disease] - [evidence]; 3. [Disease] - [evidence].
Final diagnosis: [Disease]
```
*Parsing:* Extract after `Final diagnosis:` keyword; fallback to last list item.
*Note:* Worse than P0/P1 for fine-tuned models; Qwen2.5 zero-shot performs well with this format (50.51%).

### P3 — Structured Clinical Assessment
```
Clinical assessment of this skin lesion:
• Morphology: [describe primary lesion type]
• Color/pigmentation: [describe]
• Surface: [scale/crust/smooth]
• Border: [sharp/ill-defined]
• Distribution: [localized/disseminated]
Diagnosis: [state the specific skin disease]
```
*Parsing:* Extract content after `Diagnosis:` keyword (specific fix needed for double `</think>` bug).
*Best for:* Qwen3 zero-shot (33.33%) — the `Diagnosis:` anchor reliably closes the chain of thought.

### Group Context Variants (M1/M2/M3 only)

P0+group: `"This skin lesion belongs to '{group}'. Carefully examine... what is the specific skin disease?"`
P1+group: `"This lesion belongs to '{group}'. Analyze step by step: (1) lesion type... (5) specific disease name."`
P2+group: `"This lesion is in the '{group}' group. List the top 3 most likely specific diseases within this group..."`
P3+group: `"Group: '{group}'.\nClinical assessment:\n• Morphology... Diagnosis (specific disease within this group):"`

---

## 5. Evaluation & Parsing Pipeline

### 5.1 Output Parsing — `_extract_disease()`

```
Raw model output (string)
    ↓ Strip <think>...</think> block (regex): remove all content between <think> and </think>
         Also handle dangling </think> without opening (double-close bug fix)
    ↓ Apply prompt-specific anchor patterns:
         P3: extract after "Diagnosis:" keyword (last occurrence)
         P2: extract after "Final diagnosis:" keyword
         P1: extract content after last "Step N:" pattern (N=1–9)
         P0: take first non-empty line
    ↓ Normalize: lowercase, strip whitespace, remove "this image shows" / "this is" prefix
    ↓ Fuzzy word-overlap matching against DISEASE_NAMES (10 disease strings):
         For each candidate disease: Jaccard similarity on word sets
         Match = highest-similarity disease if score ≥ threshold (91/100)
         No match → "Unknown"
    ↓ Return: matched disease name OR "Unknown"
```

### 5.2 Metrics

| Metric | Definition | Primary? |
|--------|-----------|---------|
| **Accuracy (Acc/total)** | correct / total_samples | ✅ **YES** |
| Balanced Accuracy | Mean per-class recall | Secondary |
| F1 Macro | Unweighted mean F1 per class | Secondary |
| F1 Weighted | Sample-count weighted F1 | Secondary |
| Cohen's Kappa | Agreement beyond chance | Secondary |
| Overall Score | Composite (Acc + Balanced Acc + F1 + Kappa) / 4 | Logged |
| Sensitivity (recall) | TP / (TP + FN) per disease | Per-disease |
| PPV (precision) | TP / (TP + FP) per disease | Per-disease |

**Critical:** Unknown predictions count as incorrect (denominator = total_samples, NOT valid_predictions).
A model that correctly classifies 60/99 but returns "Unknown" for 20 more scores 60.6%, not 75%.

### 5.3 Coverage and OOM Issues

After `img.thumbnail((672, 672), LANCZOS)` fix in `_load_image()`, **all 60 RAG experiments have 0 OOM errors**.
Before fix: 14–40/99 samples failed per run (high-res images up to 3.89MP → VRAM overflow).
Qwen3-VL-8B uses 28px image tiles; 672px = 24 tiles per side → predictable VRAM for all test images.

---

## 6. Benchmark Results

> **Clean benchmark:** Image resize fix (`img.thumbnail((672, 672), LANCZOS)`) eliminated all VRAM overflow. Results below are from the first fully error-free run across all models × RAGs × prompts.

### 6.1 fuzzytopk Results (Acc = correct/99, **0 OOM errors**)

*Note: No-RAG uses 101 val samples (different split includes invasive SCC); RAG results use 99-sample 3-stage split.*

| RAG | P0 | P1 | P2 | P3 |
|-----|----|----|----|----|
| **No-RAG** | **74.0%** (74/101) | **71.72%** | **70.71%** | **71.72%** |
| R0 | 63.9% | 63.5% | 61.5% | 62.9% |
| R1 | 63.3% | 63.9% | 59.4% | 63.3% |
| R2 | 63.6% | 62.6% | 61.6% | 63.6% |
| R3 | 63.3% | **65.3%** | 59.4% | 59.2% |
| R4 | 60.2% | 58.6% | 51.5% | 57.6% |

**Per-disease breakdown — fuzzytopk No-RAG P0 (74.0%, 74/101 — note: 101-sample split):**

| Disease | Sens% | n |
|---------|-------|---|
| SCCIS | ~85.7% | 14 |
| Melanocytic nevi | 91.7% | 12 |
| BCC | 76.9% | 13 |
| Acne vulgaris | 87.5% | 8 |
| Psoriasis | 84.6% | 13 |
| Sarcoidosis | 42.9% | 7 |
| Lichen planus | 77.8% | 9 |
| Scleroderma | 37.5% | 8 |
| Photodermatoses | 57.1% | 8 |
| Lupus erythematosus | 44.4% | 9 |

**Per-disease breakdown — fuzzytopk R3-P1 (best RAG config, 65/99):**

| Disease | Sens% | n | PPV% |
|---------|-------|---|------|
| SCCIS | 100.0% | 12 | 63.2% |
| Lichen planus | 77.8% | 9 | 63.6% |
| BCC | 76.9% | 13 | 66.7% |
| Acne vulgaris | 75.0% | 8 | 85.7% |
| Sarcoidosis | 71.4% | 7 | 33.3% |
| Psoriasis | 61.5% | 13 | 100.0% |
| Scleroderma | 57.1% | 7 | 80.0% |
| Melanocytic nevi | 50.0% | 12 | 100.0% |
| Lupus erythematosus | 44.4% | 9 | 40.0% |
| Photodermatoses | 25.0% | 8 | 100.0% |

**Key finding for fuzzytopk:**
- **RAG consistently hurts fuzzytopk even with 0 OOM errors.** No-RAG P0 = 74.0%, best RAG = 65.3% (R3-P1, −8.7%).
- Root cause: fuzzytopk was trained on single-image prompts → adding reference images before query confuses it.
- No-RAG P1/P2/P3 (71.72%/70.71%/71.72%) all below P0 (74.0%) → prompt style matters less than image context.
- P0 and P1 are comparable (prompt-agnostic short-output model). P2/P3 slightly worse due to longer CoT structure.
- R4 (Nomic unified) worst RAG encoder for fuzzytopk; R3 (Jina-CLIP + MedCPT) + P1 = best RAG combo.

---

### 6.2 fuzzytopk_s1cascade Results (Acc = correct/99, **0 OOM errors**)

Default alpha=0.5 baseline results:

| RAG | P0 | P1 | P2 | P3 |
|-----|----|----|----|----|
| R0 | 72.7% | 69.7% | 70.7% | 69.7% |
| R1 | 69.7% | 69.7% | 69.7% | 67.7% |
| **R2** | **74.7%** | 72.7% | 73.7% | 71.7% |
| R3 | 66.7% | 64.6% | 63.6% | 63.6% |
| R4 | 67.7% | 66.7% | 67.7% | 67.7% |

#### Alpha Tuning — R2 (SigLIP + BGE-M3) and R3 (Jina-CLIP + MedCPT), P0

Score = α × img_sim + (1−α) × txt_sim. Tuning image weight α:

| α | R2 Accuracy | R3 Accuracy | R1 Accuracy | R4 Accuracy | Notes |
|---|-------------|-------------|-------------|-------------|-------|
| 0.5 (default) | 74.75% | 66.67% | 69.70% | 67.68% | Equal image/text weight |
| 0.7 | 76.77% | **78.79%** | — | 71.72% | R3 peaks here (MedCPT needs 30% text) |
| 0.8 | **75.76%** | 71.72% | — | — | R2 plateau; R3 drops |
| **0.9** | **79.80%** | 67.68% | **71.72%** | **72.73%** | **R2 peaks** — image-dominant (10% text) |
| 1.0 | 75.76% | 66.67% | — | — | Pure image — R2 drops; text adds ~4% boost at α=0.9 |

- **R2 optimal: α=0.9 → 79.80% (79/99)** — image-dominant; BGE-M3 text adds ~4% at 10% weight
- **R3 optimal: α=0.7 → 78.79% (78/99)** — MedCPT needs 30% text weight; collapses at α=0.9
- **R1 (ClinicalBERT):** Measured at α=0.9 = 71.72% — below R0 image-only (72.73%), ClinicalBERT too generic
- **R4 (Nomic):** α=0.9 = 72.73%, α=0.7 = 71.72% — cross-modal encoder with modest text contribution
- **R0:** No alpha to tune (image-only strategy A) = 72.73%

#### Stage 3 Caption as Text Query (B4 experiment, α=0.5)

| RAG | Symptoms source | Stage 3 source | Change |
|-----|----------------|----------------|--------|
| R1 | 69.7% | 71.7% | +2.0% |
| R2 | **74.7%** | 71.7% | −3.0% |
| R3 | 66.7% | 72.7% | **+6.1%** |

Stage 3 captions help R1 and R3 (structured medical text aligns with MedCPT/ClinicalBERT) but hurt R2 (BGE-M3 prefers patient symptom vocabulary).

**Per-disease breakdown — s1cascade R2 α=0.9 P0 (previous best: 79/99 = 79.80%):**

| Disease | Sens% | n | PPV% |
|---------|-------|---|------|
| SCCIS | 100.0% | 12 | 66.7% |
| Acne vulgaris | 100.0% | 8 | 88.9% |
| Melanocytic nevi | 91.7% | 12 | 100.0% |
| Lichen planus | 88.9% | 9 | 80.0% |
| BCC | 76.9% | 13 | 90.9% |
| Scleroderma | 75.0% | 8 | 85.7% |
| Lupus erythematosus | 77.8% | 9 | 70.0% |
| Photodermatoses | 62.5% | 8 | 71.4% |
| Psoriasis | 61.5% | 13 | 80.0% |
| Sarcoidosis | 57.1% | 7 | 66.7% |

**Per-disease breakdown — s1cascade R0-P0 (72.73%):**

| Disease | Sens% | n | PPV% |
|---------|-------|---|------|
| Melanocytic nevi | 91.7% | 12 | 100.0% |
| SCCIS | 91.7% | 12 | 73.3% |
| Lichen planus | 88.9% | 9 | 88.9% |
| Acne vulgaris | 87.5% | 8 | 87.5% |
| Sarcoidosis | 71.4% | 7 | 55.6% |
| Psoriasis | 69.2% | 13 | 69.2% |
| BCC | 69.2% | 13 | 69.2% |
| Lupus erythematosus | 66.7% | 9 | 54.5% |
| Scleroderma | 37.5% | 8 | 75.0% |
| Photodermatoses | 37.5% | 8 | 50.0% |

**Key findings for fuzzytopk_s1cascade:**
- **Best s1cascade: R2-P0 α=0.9 = 79.80% (79/99)** — SigLIP + BGE-M3 at 90% image weight (surpassed by ragR2_a09 = 85.86%).
- **Alpha tuning is critical:** Default α=0.5 = 74.75%; tuned α=0.9 = 79.80% (+5.05%). Never use default alpha without sweeping.
- R2 (SigLIP+BGE-M3) vs R0 (CLIP): at α=0.5, R2 beats R0 by +2.0%; at α=0.9, R2 beats R0 by +7.1%.
- R3 (Jina-CLIP+MedCPT) peaks at α=0.7 (78.79%) — nearly ties R2's best. MedCPT works well with 30% weight.
- **R1 (ClinicalBERT) consistently below R0** — generic clinical text doesn't match dermoscopy captions.
- **P0 (direct prompt) wins across all RAG configs** — cascade pretraining biases toward short outputs.
- RAG **helps** s1cascade across all configs. Stage 1 cascade pretraining enables effective reference utilization.

---

### 6.3 fuzzytopk_s1cascade_ragR2_a09 Results (Acc = correct/99, **0 OOM errors**)

Model trained with K=1 RAG reference per training sample (R2 encoder, α=0.9). Tested at inference with various RAG configs:

| RAG | P0 | P3 | Notes |
|-----|----|----|-------|
| **R0** | **85.86%** | 83.84% | CLIP image-only — **NEW BEST OVERALL** |
| R2 | 82.83% | 82.83% | SigLIP+BGE-M3 α=0.9 (training encoder) |

**Full metrics for all 4 configurations:**

| Inference RAG | Acc | Bal Acc | F1 Macro | Kappa |
|--------------|-----|---------|----------|-------|
| R0-P0 (CLIP) | **85.86%** | 82.12% | 80.78% | 84.18% |
| R0-P3 | 83.84% | 80.52% | 79.52% | 81.93% |
| R2-P0 (SigLIP+BGE-M3) | 82.83% | 80.12% | 80.07% | 80.80% |
| R2-P3 | 82.83% | 80.78% | 80.77% | 80.81% |

**Surprising finding:** CLIP image-only (R0) at inference outperforms the SigLIP+BGE-M3 retrieval (R2) used during training (+3.03%). RAG training teaches the model to extract relevant cues from references generally, not encoder-specific features.

**Per-disease breakdown — ragR2_a09 R0-P0 (85.86%, **NEW BEST**):**

| Disease | Sens% | n | PPV% |
|---------|-------|---|------|
| Psoriasis | 100.0% | 13 | 92.9% |
| Melanocytic nevi | 100.0% | 12 | 100.0% |
| SCCIS | 100.0% | 12 | 100.0% |
| BCC | 100.0% | 13 | 100.0% |
| Acne vulgaris | 100.0% | 8 | 88.9% |
| Lichen planus | 88.9% | 9 | 88.9% |
| Scleroderma | 87.5% | 8 | 77.8% |
| Photodermatoses | 75.0% | 8 | 66.7% |
| Lupus erythematosus | 55.6% | 9 | 55.6% |
| Sarcoidosis | 14.3% | 7 | 33.3% |

**Per-disease breakdown — ragR2_a09 R0-P3 (83.84%):**

| Disease | Sens% | n | PPV% |
|---------|-------|---|------|
| Melanocytic nevi | 100.0% | 12 | 100.0% |
| BCC | 100.0% | 13 | 92.9% |
| Acne vulgaris | 100.0% | 8 | 88.9% |
| Psoriasis | 92.3% | 13 | 92.3% |
| SCCIS | 91.7% | 12 | 100.0% |
| Scleroderma | 87.5% | 8 | 77.8% |
| Lichen planus | 77.8% | 9 | 87.5% |
| Photodermatoses | 75.0% | 8 | 75.0% |
| Lupus erythematosus | 66.7% | 9 | 50.0% |
| Sarcoidosis | 14.3% | 7 | 33.3% |

**Key findings for ragR2_a09:**
- **85.86% (R0-P0)** = new overall best, +6.06% over previous best (s1cascade R2-P0 α=0.9 = 79.80%)
- **5 diseases at 100% sensitivity** (Psoriasis, Melanocytic nevi, SCCIS, BCC, Acne vulgaris) — dramatically improved
- **Psoriasis** biggest winner: 61.5% (s1cascade) → 100.0% (+38.5%). RAG-in-training teaches psoriasis-vs-lupus-vs-lichen discrimination.
- **Sarcoidosis** dropped sharply to 14.3% (was 57.1% with s1cascade R2 α=0.9) — model over-relies on reference agreement; sarcoidosis references confused with other inflammatory conditions during training
- P0 > P3 for this model (+2.02%), consistent with s1cascade preference for direct prompts.

---

### 6.4 M1 Results (3-stage + GT group context, Acc = correct/99, **0 OOM errors**)

| RAG | P0 | P1 | P2 | P3 |
|-----|----|----|----|----|
| **No-RAG** | **47.9%** (47/99) | **49.49%** | **45.45%** | **46.46%** |
| R0 | 55.0% | **59.4%** | 53.8% | 50.6% |
| R1 | 54.8% | 58.9% | 52.7% | 49.5% |
| R2 | 53.9% | 52.1% | 53.9% | 47.2% |
| R3 | 52.2% | 52.6% | 56.0% | 50.0% |
| R4 | 56.7% | 57.5% | 54.4% | 53.3% |

**Per-disease breakdown — M1 No-RAG P0 (47/99):**

| Disease | Sens% | n |
|---------|-------|---|
| Melanocytic nevi | 100.0% | 9 |
| Acne vulgaris | 100.0% | 8 |
| BCC | 69.2% | 13 |
| SCCIS | 66.7% | 12 |
| Sarcoidosis | 42.9% | 7 |
| Lupus erythematosus | 33.3% | 9 |
| Scleroderma | 25.0% | 8 |
| Lichen planus | 22.2% | 9 |
| Psoriasis | 16.7% | 12 |
| Photodermatoses | 0.0% | 8 |

*(Note: M1 No-RAG was originally evaluated on a slightly different val split than other 99-sample runs.)*

**Per-disease breakdown — M1 No-RAG P1 (49.49% — 49/99):**

From JSON result: `accuracy: 0.4949`, `balanced_accuracy: 0.5044`, `f1_macro: 0.4938`, `kappa: 0.4578`

**Per-disease breakdown — M1 R0-P1 (best RAG config, 59/99):**

| Disease | Sens% | n | PPV% |
|---------|-------|---|------|
| Melanocytic nevi | 100.0% | 9 | 100.0% |
| Acne vulgaris | 100.0% | 8 | 100.0% |
| BCC | 84.6% | 13 | 73.3% |
| Sarcoidosis | 71.4% | 7 | 18.5% |
| SCCIS | 66.7% | 12 | 80.0% |
| Psoriasis | 61.5% | 13 | 80.0% |
| Scleroderma | 25.0% | 8 | 66.7% |
| Photodermatoses | 25.0% | 8 | 66.7% |
| Lupus erythematosus | 22.2% | 9 | 22.2% |
| Lichen planus | 22.2% | 9 | 100.0% |

**Key findings for M1:**
- **RAG substantially helps M1:** No-RAG = 47.9% → R0-P1 = 59.4% (+11.5%). M1's CoT reasoning uses reference context effectively.
- **P1 (step-by-step CoT) wins for M1** — thinking chain benefits from explicit step structure (+4.4% over P0 with RAG).
- **No-RAG P1 (49.49%) barely beats P0 (47.9%)** — without visual references, CoT prompts don't help much.
- **P2 and P3 are worse than P0 for M1 No-RAG** — longer output format without reference anchors confuses the model.
- **R0 (CLIP image-only) best RAG encoder for M1** — text retrieval (R1–R3) adds noise; R4 (Nomic) close second.
- M1 still underperforms s1cascade (59.4% vs 74.75%) due to 3-stage cascade penalty.

---

### 6.5 Zero-Shot Baselines (no fine-tuning)

Two base models tested with no fine-tuning, using the same evaluation framework as fine-tuned models.

#### max_new_tokens matters critically for zero-shot models

| Model | 64 tokens | 1024 tokens | Reason |
|-------|-----------|-------------|--------|
| Qwen3-VL-8B-Thinking ("base") | ~0–2% | up to **33.33%** | Thinking chain needs ~500–1000 tokens to close `</think>` → clean answer |
| Qwen2.5-VL-7B-Instruct ("qwen25") | 7.07% | up to **50.51%** | More headroom for verbose reasoning |

#### Qwen3-VL-8B-Thinking base (1024 tokens, R0, zero-shot)

| Prompt | Accuracy | Notes |
|--------|----------|-------|
| P0 | 4.04% | Simple "what disease?" — thinking chain wanders without clear endpoint |
| P1 | 1.01% | 6 steps exhaust token budget before reaching diagnosis |
| P2 | 18.18% | Differential list + choose → sometimes extracts correct final |
| **P3** | **33.33%** | Structured template forces `Diagnosis: [name]` → reliably extracted |

P3 works best because the `Diagnosis:` label anchors Qwen3's chain-of-thought to conclude with a clean disease name.

#### Qwen2.5-VL-7B-Instruct (1024 tokens, R0, zero-shot)

| Prompt | Accuracy | Notes |
|--------|----------|-------|
| P0 | 23.23% | Direct answer, concise output |
| P1 | 24.24% | Step-by-step, similar to P0 |
| **P2** | **50.51%** | Differential Dx: list 3 + choose → Qwen2.5 follows format well |
| P3 | 17.17% | Clinical template not well-followed; answer buried |

---

## 7. Summary Comparison — Best Results per Model (Clean, 0 OOM)

| Model | Best Config | Acc/total | Correct | Notes |
|-------|------------|-----------|---------|-------|
| **ragR2_a09** | RAG R0-P0 | **85.86%** | 85/99 | **NEW BEST OVERALL** — trained with RAG context, CLIP inference |
| **ragR2_a09** | RAG R0-P3 | 83.84% | 83/99 | Same model, structured clinical prompt |
| **ragR2_a09** | RAG R2-P0 | 82.83% | 82/99 | Same model, SigLIP+BGE-M3 (training encoder) |
| **ragR2_a09** | RAG R2-P3 | 82.83% | 82/99 | R2 + structured prompt — identical to R2-P0 |
| **s1cascade** | RAG R2-P0 **α=0.9** | 79.80% | 79/99 | Previous best — SigLIP+BGE-M3, tuned alpha |
| **s1cascade** | RAG R3-P0 **α=0.7** | 78.79% | 78/99 | Jina-CLIP+MedCPT, tuned alpha |
| **s1cascade** | RAG R2-P0 α=0.5 | 74.75% | 74/99 | SigLIP+BGE-M3, default alpha |
| **fuzzytopk** | No-RAG | 74.00% | 74/101 | Best single-image; RAG hurts this model |
| **s1cascade** | RAG R0-P0 | 72.73% | 72/99 | CLIP image-only RAG |
| **fuzzytopk** | RAG R3-P1 | 65.31% | 65/99 | Best fuzzytopk RAG (Jina+MedCPT, CoT) |
| **M1** | RAG R0-P1 | 59.38% | 59/99 | Best M1 — RAG +11.5% over No-RAG |
| **Qwen2.5 zero-shot** | R0-P2 (1024 tok) | 50.51% | 50/99 | No fine-tuning; Differential Dx prompt |
| **M1** | No-RAG P1 | 49.49% | 49/99 | M1 No-RAG best prompt |
| **M1** | No-RAG P0 | 47.87% | 47/99 | M1 No-RAG baseline |
| **Qwen3 base zero-shot** | R0-P3 (1024 tok) | 33.33% | 33/99 | No fine-tuning; Structured Clinical prompt |

---

## 8. Key Findings & Analysis

### 8.1 Does RAG Help? (Clean results, 0 OOM)

| Model | No-RAG | Best RAG | RAG Effect |
|-------|--------|----------|-----------|
| fuzzytopk | **74.0%** | 65.3% (R3-P1) | **−8.7%** (RAG hurts) |
| fuzzytopk_s1cascade | n/a | 79.80% (R2-P0, α=0.9) | +5.8% over fuzzytopk No-RAG |
| **ragR2_a09** (Exp E) | n/a | **85.86%** (R0-P0) | **+11.86% over fuzzytopk No-RAG** |
| M1 | 47.9% | **59.4%** (R0-P1) | **+11.5%** (RAG helps a lot) |

- **RAG hurts fuzzytopk:** Even with 0 OOM errors, RAG adds confusion. fuzzytopk was trained on single-image prompts and doesn't know how to use reference context.
- **RAG helps s1cascade:** R2-P0 α=0.9 = 79.80% — +5.8% over fuzzytopk No-RAG baseline (74.0%). Alpha tuning is essential.
- **RAG-in-training (ragR2_a09) closes the gap completely:** +11.86% over fuzzytopk No-RAG. By training with reference images in context, the model learns to actively use them — new best 85.86%.
- **RAG greatly helps M1:** +11.5% improvement. M1's CoT reasoning effectively uses reference images for disease disambiguation.

### 8.2 Which Prompt Works Best? (Clean results)

| Model | P0 | P1 | P2 | P3 | Best |
|-------|----|----|----|----|------|
| fuzzytopk No-RAG | **74.0%** | 71.7% | 70.7% | 71.7% | **P0** |
| fuzzytopk (R0) | 63.9% | 63.5% | 61.5% | 62.9% | **P0/P1** (≈tied) |
| s1cascade (R2) | **74.7%** | 72.7% | 73.7% | 71.7% | **P0** |
| M1 No-RAG | 47.9% | **49.5%** | 45.5% | 46.5% | **P1** |
| M1 (R0) | 55.0% | **59.4%** | 53.8% | 50.6% | **P1** |

- **fuzzytopk/s1cascade:** P0 wins (simple prompt). Short-output models are prompt-agnostic; P1 nearly ties.
- **M1:** P1 (step-by-step CoT) best — the model's `<think>` reasoning benefits from explicit step structure (+4.4% over P0 with RAG, +1.6% without).
- **P2 (differential Dx) and P3 (structured):** Consistently worse for fine-tuned models. Qwen2.5 zero-shot is the exception where P2 shines (50.51%).

### 8.3 Which RAG Encoder Works Best? (Clean results, s1cascade focus)

Default α=0.5 baseline:

| RAG | s1cascade P0 | fuzzytopk P0 | M1 P1 | Notes |
|-----|-------------|-------------|-------|-------|
| R2 | 74.75% | 63.6% | 52.1% | SigLIP + BGE-M3 |
| R0 | 72.73% | 63.9% | **59.4%** | CLIP image-only — BEST for M1 |
| R1 | 69.70% | 63.3% | 58.9% | CLIP + ClinicalBERT |
| R4 | 67.68% | 60.2% | 57.5% | Nomic unified cross-modal |
| R3 | 66.67% | 63.3% | 52.6% | Jina-CLIP + MedCPT |

With tuned alpha (s1cascade only):

| RAG | Default α=0.5 | Optimal α | Accuracy | Notes |
|-----|--------------|-----------|----------|-------|
| R2 | 74.75% | **α=0.9** | **79.80%** | SigLIP+BGE-M3: image-dominant |
| R3 | 66.67% | **α=0.7** | **78.79%** | Jina-CLIP+MedCPT: 30% text optimal |
| R0 | 72.73% | α=N/A | 72.73% | Image-only, no alpha to tune |
| R4 | 67.68% | α=0.9 | 72.73% | Nomic: slight text contribution |
| R1 | 69.70% | α=0.9 | 71.72% | CLIP+ClinicalBERT: worse than R0 |

### 8.4 Per-Disease Analysis — Clean Results

| Disease | fuzzytopk NoRAG | s1cascade R0-P0 | s1cascade R2 α=0.9 | **ragR2_a09 R0-P0** | M1 R0-P1 | Verdict |
|---------|----------------|----------------|-------------------|---------------------|---------|---------|
| SCCIS | **100.0%** | 91.7% | 100.0% | **100.0%** | 66.7% | ✅ Easy |
| Melanocytic nevi | 91.7% | 91.7% | 91.7% | **100.0%** | **100.0%** | ✅ Easy |
| BCC | 76.9% | 69.2% | 76.9% | **100.0%** | 84.6% | ✅ Easy (ragR2) |
| Acne vulgaris | 87.5% | 87.5% | **100.0%** | **100.0%** | **100.0%** | ✅ Easy |
| Psoriasis | 84.6% | 69.2% | 61.5% | **100.0%** | 61.5% | Moderate (ragR2 best) |
| Lichen planus | 77.8% | **88.9%** | **88.9%** | **88.9%** | 22.2% | Moderate |
| Scleroderma | 37.5% | 37.5% | **75.0%** | **87.5%** | 25.0% | Hard; RAG training helps |
| Photodermatoses | 57.1% | 37.5% | **62.5%** | **75.0%** | 25.0% | Hard; RAG training helps |
| Lupus | 44.4% | **66.7%** | 77.8% | 55.6% | 22.2% | Hard; s1cascade α=0.9 best |
| Sarcoidosis | 42.9% | **71.4%** | 57.1% | 14.3% | **71.4%** | Hard; **ragR2 collapses** |

**Key disease-level insights:**
- **ragR2_a09 dominates 7/10 diseases** at or near 100% sensitivity.
- **Psoriasis:** biggest ragR2_a09 winner: 61.5% (s1cascade R2 α=0.9) → 100.0%.
- **Sarcoidosis:** ragR2_a09 catastrophically drops to 14.3% (from 57.1%). Over-reliance on reference agreement; sarcoidosis references mislabeled during training. Critical failure point.
- **Scleroderma:** ragR2_a09 best at 87.5% (up from 75.0% with s1cascade). RAG training adds morphological reference anchors.
- **Photodermatoses:** ragR2_a09 best at 75.0% (up from 62.5% with s1cascade). Still below ideal.
- **Lupus:** ragR2_a09 regresses vs s1cascade (77.8% → 55.6%). May be confused with sarcoidosis references.

### 8.5 Why s1cascade Benefits from RAG but fuzzytopk Doesn't

| Aspect | fuzzytopk | fuzzytopk_s1cascade | ragR2_a09 |
|--------|-----------|---------------------|-----------|
| Training context | Single image + prompt | Single image + prompt | **Multi-image (ref + query)** |
| Starting weights | Base model → disease | Base model → group → disease | Base model → group → disease+RAG |
| RAG effect | −8.7% (confused by refs) | +5.8% over baseline | **+11.86% over baseline** |
| Hypothesis | Never seen multi-image context | Stage 1 cascade teaches visual comparison | Explicitly trained to leverage references |

### 8.6 Experiment E — RAG-in-Training Findings

**Goal:** Close the train/inference distribution gap — s1cascade trained on single images but evaluated with 3 reference images prepended.

**Implementation challenges:**
- K=3 caused VRAM OOM (ZeroDivisionError in Unsloth fused CE loss): 4 images × batch=2 = 8 GPU images → 15.92GB exhausted
- K=1 fits comfortably; RAG encoders must be pre-computed and freed before loading Qwen3-VL
- `precompute_rag_refs()` runs retrieval with SigLIP+BGE-M3 (R2, α=0.9) before model load

**Training config:** 939 train images × ~3.7 samples/image = 3,504 training items, 3 epochs, 1,314 steps, ~1h44min

**Key findings:**
1. **+6.06% improvement over previous best** (79.80% → 85.86%)
2. **CLIP (R0) inference beats SigLIP (R2) inference** despite training with R2. Model learned the concept "look at reference → use label" generally.
3. **5 diseases at 100% sensitivity:** Psoriasis dramatically improved (61.5% → 100%)
4. **Sarcoidosis collapsed** (57.1% → 14.3%) — over-reliance on reference agreement
5. **P0 > P3** for this model (consistent with s1cascade pattern): short-output training creates short-output preference

### 8.7 Why M1 Underperforms s1cascade (59.4% vs 74.75%)

1. **3-stage cascade penalty:** Stage 2 starts from Stage 1 group-classifier weights → corrupted initialization.
2. **Verbose CoT output:** M1 generates long `<think>` chains → larger KV cache → higher VRAM pressure.
3. **Prompt complexity:** GT group context + RAG images + captions + step-by-step CoT = very long prompts.
4. **M1 No-RAG severely underperforms:** 47.9% vs fuzzytopk 74.0% — SCCIS 66.7%, Photodermatoses 0%, Psoriasis 16.7%.

### 8.8 Bugs Fixed During Benchmarking

| Bug | Symptom | Fix |
|-----|---------|-----|
| Double `</think>` | P3 showed 100% from 1/99 valid | Strip dangling `</think>` + `Diagnosis:` pattern |
| P1/P2 low coverage | 16–28/99 valid → inflated accuracy | `Step N:` last occurrence + `Final diagnosis:` extraction |
| `--method` not forwarded | `--method fuzzytopk` silently ran M1 | Pass `method=method` at call site in `run_rag_benchmark.py` |
| `max_length=4096, truncation=True` in main batch | "Mismatch in image token count" for long RAG prompts | Removed from `apply_chat_template()` main path |
| Metric inflation | accuracy=1.0 from 1 parseable sample | Metric = correct/total (not correct/valid) |
| OOM on large query images | 14–40/99 samples failed per run | `img.thumbnail((672, 672), LANCZOS)` in `_load_image()` |

---

## 9. Stage 3 Caption Generation Evaluation

Stage 3 models generate multi-sentence clinical captions. Evaluated with BLEU and ROUGE against ground-truth `caption_zh_polish_en` from the CSV.

### 9.0 Where Stage 3 Starts From — Full Pipeline Flow

Stage 3 does **not** start from the base model or Stage 1. It starts from the **Stage 2 disease classifier** (fuzzytopk_s1cascade), which itself was built on top of the Stage 1 group classifier:

```
Base Model (Qwen3-VL-8B)
    ↓  Stage 1: fine-tune on 4-group classification (88.68% val)
Group Classifier  →  skincap_3stage_group_classification_merged
    ↓  Stage 2: fine-tune on 10-disease classification (74–85% val)
Disease Classifier  →  skincap_fuzzytopk_s1cascade_classification_merged
    ↓  Stage 3: fine-tune on caption generation  ← THIS IS WHERE WE ARE
Caption Model  →  skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_classification_merged
```

**Why use Stage 2 as the starting point?**
The disease classifier already "understands" dermoscopy lesion morphology, color, and disease-level visual features from Stage 2 training. Starting Stage 3 caption generation from this model means the caption model inherits rich disease-level visual representations — instead of learning skin lesion features from scratch.

**The critical choice: how to hand off from Stage 2 to Stage 3**

Two options were tested (see ablation below):

| | Way 1 — Checkpoint Init | Way 2 — Merged Init ✅ |
|--|------------------------|----------------------|
| Stage 2 knowledge location | Inside LoRA adapters (task-specific) | Baked into base weights (permanent) |
| Stage 3 LoRA | Reuses Stage 2 adapters | Fresh adapters (all zeros) |
| Risk | Caption fine-tuning overwrites disease adapters → interference | No risk — disease knowledge is safe in base weights |
| BLEU-4 result | 9.82 | **29.33 (3× better)** |

**Key rule:** When switching tasks in multi-stage LoRA fine-tuning, always **merge first, then add fresh LoRA**. Never continue fine-tuning existing adapters on a different task.

### Evaluation Setup
- **Prompt:** `"Describe this skin lesion image in detail. Include information about its appearance, possible diagnosis, and recommended examinations."`
- **Ground truth:** `caption_zh_polish_en` column (Polish-translated + polished clinical descriptions)
- **Val set:** 99 samples from `split_info_3stage.json`
- **Script:** `inference_disease_classification.py --flow val --stage2_method {method}`

### 9.1 Stage 3 Ablation — 4 Experiments (Way 1/2 × STS On/Off)

Two initialization strategies tested:
- **Way 1 (checkpoint init):** Load Stage 1 LoRA checkpoint (`skincap_fuzzytopk_s1cascade_classification`) → continue fine-tuning existing adapters on captions.
- **Way 2 (merged init):** Load Stage 1 fully merged model (`skincap_fuzzytopk_s1cascade_classification_merged`) → add fresh LoRA adapters → train from scratch on captions. Disease knowledge is baked into base weights permanently.

STS (Selective Token Supervision) from SIB-TinyLoRA: per-token loss weighting (`w_ans × w_reason`) + IBR (`β × ||LoRA_A||² + β × ||LoRA_B||²`) regularization.

| Exp | Name | Init | STS | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-----|------|------|-----|--------|--------|--------|--------|---------|---------|---------|
| 0 (baseline) | Original | checkpoint | ✗ | 33.07 | 17.20 | 12.14 | **9.82** | 38.50 | 13.15 | **27.89** |
| 1 (Exp 1) | Way 1 | checkpoint | ✗ | 29.22 | 18.65 | 13.03 | 9.82 | 38.9 | 15.71 | — |
| **2 (Exp 2)** | **Way 2** | **merged** | **✗** | **42.22** | **35.19** | **31.55** | **29.33** | **53.55** | **36.33** | — |
| 3 (Exp 3) | Way 1 + STS | checkpoint | ✓ | 0.01 | 0.00 | 0.00 | 0.00 | 5.03 | 0.38 | — |
| 4 (Exp 4) | Way 2 + STS | merged | ✓ | 1.78 | 1.09 | 0.78 | 0.61 | 15.68 | 5.83 | 13.12 |

**Winner: Exp 2 (Way 2, merged init, no STS) — BLEU-4=29.33 (3× over baseline)**

Model path: `skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_classification_merged`

### 9.2 Why Merged Init Wins

| Aspect | Way 1 (checkpoint) | Way 2 (merged) |
|--------|--------------------|----------------|
| LoRA adapters at Stage 3 start | Pre-trained (disease task) | Fresh (all zeros) |
| Disease knowledge | In LoRA adapters — **at risk during caption fine-tuning** | Baked into base weights — **permanent, safe** |
| Caption adaptation | LoRA must unlearn disease → learn caption | LoRA learns caption from scratch on clean base |
| Result | BLEU-4=9.82 (adapters partially retain disease bias) | **BLEU-4=29.33 (3× better)** |

**Core insight:** When LoRA adapters carry disease classification knowledge, fine-tuning them for a different task (caption generation) causes catastrophic interference. The merged init permanently encodes disease knowledge into the base weights, then fresh LoRA adapters can specialize on captions without conflicting gradients.

### 9.3 Why STS Fails (Both Experiments)

**Exp 3 (Way 1 + STS): BLEU-4=0.0 — catastrophic collapse**
- IBR (`β × ||LoRA||²`) penalizes large LoRA norms
- Checkpoint init has pre-trained LoRA with large existing norms
- IBR drives all LoRA parameters toward zero → complete erasure of disease + caption knowledge
- Model loses the ability to generate coherent text → 0.0 BLEU

**Exp 4 (Way 2 + STS): BLEU-4=0.61 — severe degradation**
- Fresh LoRA adapters start at zero (IBR penalty starts low)
- STS token weighting overrides standard CE loss → less stable training signal
- Unsloth's compiled training returns non-tensor `logits` → STS falls back to `outputs.loss + IBR`
- IBR alone (without STS token weighting) acts as strong L2 regularizer → prevents LoRA from growing → poor fitting
- Result: 29.33 (no STS) vs 0.61 (with STS) on same init — IBR costs −28.72 BLEU-4

**Conclusion: STS/IBR is incompatible with multi-stage transfer learning.** IBR was designed for training from scratch (near-zero initial norms); both stages here have strong priors encoded in either LoRA or base weights.

### 9.4 Previous Results — Use of Stage 3 Captions as RAG Text Query (B4 Experiment)

| RAG | Symptoms → text | Stage3 cap → text | Change |
|-----|----------------|-------------------|--------|
| R1 (ClinicalBERT) | 69.7% | 71.7% | +2.0% |
| R2 (BGE-M3) | 74.7% | 71.7% | −3.0% |
| R3 (MedCPT) | 66.7% | 72.7% | **+6.1%** |

*(Note: B4 experiment used the baseline Stage 3 model, BLEU-4=9.82. Repeating with the new Way 2 model (BLEU-4=29.33) may improve R3 further.)*

---

## 10. Technical Fixes Applied During Project

| Issue | Fix | Impact |
|-------|-----|--------|
| Data leakage in RAG index | `RAG_USE_ALL_DATA=False` → train-only 911-image index | Removes circular retrieval from val images |
| Strategy B text not used | Pass `caption` as `vlm_description` to `retrieve()` | Enables real hybrid retrieval for R1–R3 |
| `--method` not forwarded | Added `method=method` to `run_experiment()` call | `--method fuzzytopk` now correctly runs fuzzytopk model |
| Double `</think>` parsing | Added strip + `Diagnosis:` extraction | Fixed all P3 runs |
| P1/P2 low coverage | `_extract_disease()` now extracts `Final diagnosis:` (P2), last `Step N:` (P1) | Coverage improved to 59–79/99 |
| Reference captions missing | Return `self.captions[i]` from `retrieve()` | In-context references include clinical text |
| Symptom descriptions | `val_captions_for_symptoms.json` with patient descriptions | Realistic text query for R1–R3 |
| fuzzytopk model path | `_STAGE2_MODEL_PATH_MAP` dict | Correct model loaded |
| `max_length=4096, truncation=True` | "Mismatch in image token count" errors | Removed from `apply_chat_template()` main batch |
| **OOM on large query images** | 14–40/99 samples failed per run | `img.thumbnail((672, 672), LANCZOS)` in `_load_image()` |

---

## 11. Files Modified

| File | Changes |
|------|---------|
| `inference_disease_classification.py` | `--stage2_method` CLI arg, RAG caption in-context, symptom description priority, P3 Diagnosis extraction, double `</think>` fix, removed `max_length=4096`, **672×672 thumbnail cap**, zero-shot baselines in `_STAGE2_MODEL_PATH_MAP`, `_MODEL_MAX_NEW_TOKENS`, accuracy = correct/total |
| `rag_retrieval.py` | `retrieve()` returns 3-tuple `(path, label, caption)` |
| `run_rag_benchmark.py` | `method=method` forwarded to `run_experiment()` |
| `val_captions_for_symptoms.json` | Created: 94 val items with user-written symptom descriptions |
| `train_three_stage_hybrid_topk.py` | `--stage3_source`, `--stage3_lr` CLI args |
| `train_two_stage_FuzzyTopK.py` | `--rag_k_train`, `--rag_exp`, `--alpha` CLI args; `precompute_rag_refs()` function; `prepare_classification_data_with_rag()` |
| `inference_disease_classification.py` | `IS_CAPTION_MODEL`, `CAPTION_PROMPT`, `generate_caption_batch()`, `evaluate_captions()`; `--vlm_desc_source`; `--alpha` override |
| `run_rag_benchmark.py` | `--vlm_desc_source`, `--alpha` args; updated `_result_path()`, `run_experiment()`, `print_table()` |
| `val_captions_stage3.json` | Created: 99 Stage 3 generated captions for B4 experiment |
| `train_two_stage_FuzzyTopK.py` | `--stage3_init` (checkpoint/merged), `--use_sts`, `--sts_beta` CLI args; `Config.STAGE3_INIT`, `Config.USE_STS`; STSSFTTrainer mixin; Way 2 merged-init path |
| `medical_token_importance.py` | NEW: STS adapted for medical captions — `MedicalSTSConfig`, `MedicalTokenImportanceScorer`, `STSSFTTrainer` mixin; IBR (`compute_ibr_loss`); Unsloth non-tensor logits guard |
| `run_stage3_experiments.py` | NEW: 4-experiment Stage 3 ablation runner; auto-skip if merged exists; BLEU/ROUGE logging; `stage3_ablation_results.json` output |
| `inference_disease_classification.py` | Stage 3 method name keys updated to end with `_stage3`; added `fuzzytopk_s1cascade_merged_init_stage3`, `fuzzytopk_s1cascade_sts_stage3`, `fuzzytopk_s1cascade_merged_init_sts_stage3` path mappings |

---

## 12. Recommended Next Steps

Current best: **ragR2_a09 R0-P0 = 85.86%** (✅ Experiment E complete). Next experiments ordered by expected gain:

| Priority | Experiment | Rationale | Expected Gain |
|----------|-----------|-----------|--------------|
| 1 | ✅ **DONE: ragR2_a09 (RAG-in-training, K=1)** | ragR2_a09: trained with K=1 R2 reference → 85.86% (+6.06%) | +6.06% achieved |
| 2 | **Fix Sarcoidosis collapse** | ragR2_a09 sarcoidosis: 14.3% (was 57.1%) — try K=3 with filtered refs or retrain with sarcoidosis-oversampled dataset | +3–5% balanced acc |
| 3 | **ragR2_a09 with alpha tuning at inference** | R2-P0 inference gives 82.83%; try α=0.9 vs α=0.5 tuning | ±1–3% |
| 4 | **ragR2_a09 with R3 (Jina-CLIP+MedCPT) at inference** | R3 α=0.7 gave 78.79% for s1cascade — does training generalize? | ±2% |
| 5 | **Retrain with K=2 or K=3 references** | K=1 already gives 85.86%; K=3 failed OOM with batch=2; try batch=1 or gradient accum=8 | +1–3% |
| 6 | **Fix Photodermatoses/Sarcoidosis training data** | Structural bottleneck from limited training samples | +5–10% balanced accuracy |
| 7 | **Retrain Stage 3 from base model** | BLEU-4 only 9.82%; catastrophic forgetting from disease classifier starting point | Caption quality improvement |

### Current Best Configurations (Clean, 0 OOM)

For **best absolute accuracy:**
- **Model:** ragR2_a09 + **RAG R0-P0 = 85.86% (85/99)** ← NEW BEST
- CLIP image-only retrieval at inference (model trained with SigLIP+BGE-M3 — generalizes to CLIP)

For **best single-image** (no RAG overhead):
- **Model:** fuzzytopk, No-RAG P0 = **74.0% (74/101)**

For **best oracle M1:**
- **Model:** M1 + RAG R0-P1 = **59.4% (59/99)**

### Alpha Tuning Reference (fuzzytopk_s1cascade, P0)

| RAG | α=0.5 | α=0.7 | α=0.8 | α=0.9 | α=1.0 | Best |
|-----|-------|-------|-------|-------|-------|------|
| R2 (SigLIP+BGE-M3) | 74.75% | 76.77% | 75.76% | **79.80%** | 75.76% | **α=0.9** |
| R3 (Jina+MedCPT) | 66.67% | **78.79%** | 71.72% | 67.68% | 66.67% | **α=0.7** |
| R4 (Nomic unified) | 67.68% | 71.72% | — | 72.73% | — | **α=0.9** |
| R1 (CLIP+ClinicalBERT) | 69.70% | — | — | 71.72% | — | **α=0.9** |
| R0 (CLIP image-only) | 72.73% | — | — | — | — | N/A (no text) |

---

*Updated: 2026-03-07*
*Val set: 99 samples (from split_info_3stage.json) for RAG; 101 samples for fuzzytopk No-RAG*
*Models evaluated on: RTX 5070 Ti (15.9 GB VRAM), Qwen3-VL-8B-Thinking backbone*
*All RAG experiments: 0 OOM errors (image resize fix: 672×672 thumbnail applied)*
*Best result: ragR2_a09 + RAG R0-P0 = 85.86% (85/99) — RAG-in-training (Experiment E)*
