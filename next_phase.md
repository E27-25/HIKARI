<div align="center">

# 🌸 HIKARI — Phase 2 Roadmap

**Building on 85.86% accuracy · 14 published models · RAG-in-Training**

*Phase 1 Complete · Phase 2 Planning Document*

---

</div>

## Overview

Phase 1 proved that **RAG-in-Training** and **Merged-Init** work.  
Phase 2 asks: *how far can we push it?*

Six development axes — each independent, each extendable.

```
┌─────────────────────────────────────────────────────────────┐
│                    HIKARI Phase 2                           │
│                                                             │
│  🔭 Model      🔬 Technique    🧩 Pipeline                  │
│  🔍 XAI        📊 Evaluation   🏥 Application               │
│                                                             │
│  All axes share one automated pipeline:                     │
│  train → eval → benchmark → report → upload                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔭 Axis 1 — Model Comparison

> *"Does RAG-in-Training work across every VLM architecture?"*

The core claim of Phase 1 is that the **training methodology** matters more than the model.  
Phase 2 verifies this by running the same pipeline across multiple model families.

### Models

**Priority 1 — Try First**

| Model | Params | Family | Why |
|:------|:------:|:------:|:----|
| `Qwen2.5-VL-3B-Instruct` | 3B | Qwen | Same family as Phase 1, easiest migration |
| `Gemma3-4B-IT` (vision) | 4B | Google | Strong baseline, popular in research |
| `InternVL2.5-4B` | 4B | InternVL | Known strong vision encoder for medical |

**Priority 2 — If Time Allows**

| Model | Params | Family | Why |
|:------|:------:|:------:|:----|
| `Phi-4-multimodal` | 5B | Microsoft | Efficient, edge-friendly |
| `Qwen2.5-VL-7B-Instruct` | 7B | Qwen | Direct comparison to Phase 1 |
| `InternVL2.5-8B` | 8B | InternVL | Size-matched comparison |

### Experiments Per Model

Only **3 runs per model** (M-series ablation already answered in Phase 1):

```
1. Single-Image FT      ←  baseline (no cascade, no RAG)
2. Cascaded FT          ←  intermediate
3. RAG-in-Training      ←  key contribution — does it generalize?
```

### Research Questions

- Is RAG-in-Training architecture-agnostic?
- Can a 4B model + RAG outperform an 8B model without RAG?
- Which model family is most suited for medical image understanding?

### Expected Output

```
Model                | Params | Baseline | +RAG    | BLEU-4 | ms/img
---------------------|--------|----------|---------|--------|--------
SmolVLM-2-2B         |   2B   |  ~60%?   |  ~72%?  |  ~15?  |  ~200
Gemma3-4B            |   4B   |  ~72%?   |  ~80%?  |  ~22?  |  ~350
InternVL2.5-4B       |   4B   |  ~75%?   |  ~82%?  |  ~24?  |  ~400
Qwen2.5-VL-7B        |   7B   |  ~80%?   |  ~84%?  |  ~27?  |  ~500
HIKARI-Sirius (8B) ⭐ |   8B   | 79.80%   | 85.86%  | 29.33  |   584  ← Phase 1
```

---

## 🔬 Axis 2 — Technique

> *"Same model, same pipeline — but smarter training."*

**Status: Options under consideration — choose 1–2**

### Option A — Dynamic RAG-k ⭐ *Recommended*

Phase 1 used fixed `k=1`. Phase 2 lets the model decide how many references it needs.

```
Easy image  →  k=0  (model is confident, no reference needed)
Medium      →  k=1  (Phase 1 default)
Hard image  →  k=3  (uncertain, needs more context)

Decision basis: entropy of prediction distribution
High entropy = uncertain = request more references
```

**Why this matters:** Some diseases are visually distinct (k=0 is fine).  
Others look identical to neighboring diseases (need k=3 for disambiguation).

### Option B — Hard Negative Mining

```
Phase 1 RAG: retrieve closest image by visual similarity
Phase 2 add: also inject 1 "hard negative"
             — image that looks similar but is a DIFFERENT disease

Goal: teach the model to distinguish subtle visual differences
```

### Option C — Multi-Reference Fusion

```
Phase 1: k=1 → 1 reference image in context
Phase 2: k=3 → 3 reference images, model attends differently to each

Research question: how much does k=3 improve over k=1?
Hypothesis: diminishing returns after k=2
```

### Option D — Self-Training / Pseudo-Label

```
1. Use HIKARI-Sirius to predict unlabeled skin images
2. Keep only high-confidence predictions (> 0.90)
3. Add as training data → retrain
4. Expand dataset without manual labeling cost
```

---

## 🧩 Axis 3 — Pipeline

> *"Add preprocessing steps before the model sees the image."*

**Status: Options under consideration — choose 1**

### Option A — Segmentation → Classify ⭐ *Recommended*

```
Current pipeline:
[Full image] ──────────────────────→ HIKARI → diagnosis

New pipeline:
[Full image] → SAM2 (segment) → [Lesion crop only] → HIKARI → diagnosis
```

**Research question:** Does isolating the lesion before classification help?  
Or is HIKARI already attending to the right region (GradCAM suggests yes)?

**Contribution:** First systematic study of segmentation preprocessing on SkinCAP.

### Option B — Detection → Multi-lesion

```
Current: assumes 1 lesion per image → 1 diagnosis
New:     detect ALL lesions in image first
         → classify each lesion independently
         → output: { "lesion_A": "SCCIS", "lesion_B": "Melanoma" }
```

### Option C — Quality Filter + Preprocessing

```
Before every inference:
├── Blur detection     → reject image if not sharp enough
├── Skin region crop   → remove non-skin background
└── Color normalization → standardize lighting conditions
```

### Option D — 3-Level Cascade

```
Phase 1: Group (4 classes) → Disease (23 classes)
Phase 2: Group → Subgroup  → Disease    (3 levels)

Add intermediate layer to reduce cross-group confusion
```

---

## 🔍 Axis 4 — Explainability (XAI)

> *"Why did the model make this decision? — clinically trustworthy AI"*

Phase 1 implemented GradCAM. Phase 2 goes deeper.

### Level 1 — Richer Visual Explanation

```
Already done: GradCAM
Add:
├── GradCAM++          ← sharper, more precise localization
├── Attention Rollout  ← trace attention flow through all transformer layers
└── Token-level saliency ← which pixels contributed to which output tokens
```

### Level 2 — Uncertainty Score

Every prediction gets a confidence score:

```
Output example:
{
  "disease":     "SCCIS",
  "confidence":  0.87,
  "status":      "HIGH — reliable prediction"
}

{
  "disease":     "Melanoma",
  "confidence":  0.51,
  "status":      "LOW — recommend dermatologist review"
}
```

Implementation: Temperature Scaling + MC Dropout (N forward passes, measure variance)

### Level 3 — Counterfactual Explanation

```
"If this region were absent, what would the model predict?"

Method: patch masking → re-predict → compare
Output: highlight regions that are decision-critical
```

### Level 4 — Per-class Activation Atlas

```
Collect GradCAM from all SCCIS predictions
→ Average → find consistent pattern
→ "For SCCIS, the model always focuses on lesion borders"

Repeat for all 23 disease classes
→ Build a visual dictionary of what each disease "looks like" to the model
```

### Output — XAI Dashboard

```
┌──────────────────────────────────────────────────────┐
│  Input Image  │  GradCAM  │  Attention Map           │
│───────────────────────────────────────────────────────│
│  Diagnosis: SCCIS         Confidence: 87%  ██████░░  │
│  Caption: "Raised erythematous lesion..."             │
│  Critical region: lesion border (top-right)           │
│  If removed: prediction changes to BCC (51%)          │
└──────────────────────────────────────────────────────┘
```

---

## 📊 Axis 5 — Evaluation

> *"One accuracy number is not enough for medical AI."*

Phase 1 measured: accuracy, BLEU-4, BERTScore, Disease Correctness, inference speed.  
Phase 2 measures everything that matters clinically.

### Dimension 1 — Subgroup Analysis

```
Break accuracy down by:
├── Skin tone       (Fitzpatrick scale I–VI)  ← bias detection
├── Image quality   (sharp / blurry / noisy)
├── Lesion size     (small / medium / large)
└── Body location   (face / hand / back / etc.)

Goal: find where the model underperforms and why
```

### Dimension 2 — Calibration

```
Question: when the model says 90% confident — is it right 90% of the time?

Metrics:
├── Reliability diagram (confidence vs actual accuracy curve)
├── Expected Calibration Error (ECE)
└── Fix if miscalibrated: Temperature Scaling
```

### Dimension 3 — Out-of-Distribution (OOD) Robustness

```
Test on images the model has never seen:
├── HAM10000 dataset      ← different distribution, same diseases
├── Smartphone photos      ← vs dermoscope images used in training
└── Images with artifacts  ← hair, ruler, ink marks
```

### Dimension 4 — Clinical Agreement

```
Compare model predictions with real dermatologist diagnoses:
├── Cohen's Kappa score
├── Sensitivity / Specificity per disease
└── "Model vs Doctor A vs Doctor B" agreement matrix
```

### Dimension 5 — Richer Caption Evaluation

| Metric | Phase 1 | Phase 2 |
|:-------|:-------:|:-------:|
| BLEU-4 | ✅ | ✅ |
| BERTScore | ✅ | ✅ |
| Disease Correctness | ✅ | ✅ |
| ROUGE-L | ❌ | ✅ |
| CIDEr | ❌ | ✅ |
| Clinical Relevance (GPT-4o judge) | ❌ | ✅ |
| Human Evaluation | ❌ | ✅ |

### Output — Auto Evaluation Report

Every training run automatically generates a full HTML/PDF report:
accuracy breakdown · calibration curve · per-disease table · OOD results · caption scores

---

## 🏥 Axis 6 — Application

> *"Make HIKARI usable without knowing Python."*

### Phase 2A — REST API

```python
POST /api/diagnose
Content-Type: multipart/form-data

{
  "image": <file>,
  "mode": "disease" | "caption" | "full"
}

Response:
{
  "disease":    "SCCIS",
  "confidence": 0.87,
  "caption":    "Raised erythematous lesion with irregular border...",
  "heatmap":    "<base64 image>",
  "latency_ms": 584
}
```

### Phase 2B — Web UI

```
┌─────────────────────────────────────────────────────┐
│  🌸 HIKARI Skin Disease AI                          │
│─────────────────────────────────────────────────────│
│  [ Upload Image ]  or  [ Take Photo ]               │
│                                                     │
│  ┌─────────────┐  ┌─────────────┐                  │
│  │  Input      │  │  GradCAM    │                  │
│  │  Image      │  │  Heatmap    │                  │
│  └─────────────┘  └─────────────┘                  │
│                                                     │
│  Diagnosis:  SCCIS             Confidence: 87%      │
│  Caption:    Raised erythematous lesion...          │
│                                                     │
│  ⚠️ For clinical reference only.                    │
└─────────────────────────────────────────────────────┘
```

---

## Automation Infrastructure

> *All axes share one automated pipeline — train once, get everything.*

```bash
python run_all.py --model google/gemma-3-4b-it --wandb --upload-if-best

# Automatically runs:
# 1. train_stage1.py      → group classifier
# 2. train_stage2.py      → baseline, cascade, RAG-in-Training
# 3. merge.py             → merge LoRA into base
# 4. train_stage3.py      → Way 1 (checkpoint), Way 2 (merged-init)
# 5. eval_all.py          → full benchmark (all RAG × prompt configs)
# 6. xai_report.py        → GradCAM++, attention, uncertainty
# 7. speed_bench.py       → Unsloth / vLLM / SGLang
# 8. eval_report.py       → calibration, OOD, subgroup
# 9. generate_report.py   → HTML + PDF summary
# 10. upload_hf.py        → auto-upload if beats current best
```

**Experiment tracking:** Weights & Biases — every run logged automatically.  
**Config:** `models.yaml` + `experiments.yaml` — add a new model in 3 lines.

---

## Execution Order

```
Phase 2 recommended sequence:

MONTH 1   ── Build automation infrastructure (run_all.py + W&B)
          ── Model Axis: Qwen2.5-VL-3B (smoke test)

MONTH 2   ── Model Axis: Gemma3-4B + InternVL2.5-4B
          ── Technique Axis: Dynamic RAG-k experiment

MONTH 3   ── Pipeline Axis: Segmentation → Classify
          ── XAI Axis: uncertainty score + attention maps

MONTH 4   ── Evaluation Axis: OOD + subgroup + calibration
          ── Application Axis: REST API + Web UI

ONGOING   ── W&B dashboard always live
          ── Auto-upload best models to HuggingFace
```

---

## Summary

| Axis | Status | Key Output |
|:-----|:------:|:-----------|
| 🔭 Model Comparison | Ready | Multi-architecture benchmark table |
| 🔬 Technique | Choosing option | Dynamic RAG-k recommended |
| 🧩 Pipeline | Choosing option | Segmentation-first recommended |
| 🔍 XAI | Ready | Confidence score + attention dashboard |
| 📊 Evaluation | Ready | Full clinical-grade eval report |
| 🏥 Application | After research | REST API + Web UI |

---

<div align="center">

*HIKARI Phase 1 — Complete 🌸*  
*Phase 2 — In Planning*

**光 · Qwen3-VL-8B-Thinking · KMITL · 2026**

</div>
