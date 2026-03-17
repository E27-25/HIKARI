<!-- ✿ HIKARI - Healthcare-oriented Intelligent Knowledge-Augmented Retrieval and Inference system ✿ -->

<div align="center">

<!-- 🌸 Logo 🌸 -->
<img src="logo/HIKARI logo.png" alt="HIKARI Logo" width="800"/>

<br/>

<!-- ✿ Sakura Divider ✿ -->
```
✿ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✿
```

### 🌸 *A RAG-in-Training Vision-Language Model for Fine-Grained Skin Lesion Diagnosis with Retrieval-Augmented Generation and Evidence-Based Clinical Recommendations* 🌸

<br/>

<!-- ✿ Badges ✿ -->
![Python](https://img.shields.io/badge/Python-3.10+-FFB7C5?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-FADADD?style=for-the-badge&logo=pytorch&logoColor=white)
[![Accuracy](https://img.shields.io/badge/Accuracy-85.86%25-brightgreen?style=for-the-badge)](Model/README.md)
[![Model](https://img.shields.io/badge/Backbone-Qwen3--VL--8B-blue?style=for-the-badge)](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking)
![License](https://img.shields.io/badge/License-MIT-FFE4E9?style=for-the-badge)
[![Paper](https://img.shields.io/badge/Paper-ITC--CSCC_2025-red?style=for-the-badge)](Model/Conference_Paper.tex)

<br/>

```
                    🌸                          🌸                          🌸
              ✿           ✿              ✿           ✿              ✿           ✿
                    🌸                          🌸                          🌸
```

</div>

<!-- ✿ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✿ -->

## 🌸 About HIKARI

<div align="center">

| | |
|:---:|:---|
| **H** | **H**ealthcare-oriented |
| **I** | **I**ntelligent |
| **K** | **K**nowledge- |
| **A** | **A**ugmented |
| **R** | **R**etrieval and |
| **I** | **I**nference system |

</div>

```
✿ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✿
```

<div align="center">

## 🌸 Implementation Results 🌸

</div>

> The HIKARI model has been built and validated on the **SkinCAP** dermatology dataset.
> See [`Model/README.md`](Model/README.md) for the full technical reference.

<div align="center">

| Model Variant | Accuracy |
|:---|:---:|
| 🥇 **HIKARI RAG-in-Training** (`fuzzytopk_s1cascade_ragR2_a09`) | **85.86%** |
| 🥈 Cascaded FT + Inference RAG | 79.80% |
| 🥉 Single-Image Fine-Tune | 74.00% |
| Zero-Shot Frontier (best) | 50.51% |
| Base Qwen3-VL-8B (no fine-tuning) | 33.33% |

| Component | Detail |
|:---|:---|
| Backbone | Qwen3-VL-8B-Thinking |
| Dataset | SkinCAP — 4,000 dermatology images, 10 disease classes |
| RAG Encoder (training) | SigLIP-2 + BGE-M3 (R2), K=1 reference per sample |
| RAG Encoder (inference) | CLIP ViT-B/32 (R0), K=3 references |
| GPU | NVIDIA RTX 5070 Ti · 4-bit NF4 quantization |
| Training time | ~1h 44min · LoRA rank 16 |

</div>

```
✿ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✿
```

<div align="center">

## 🌸 Training Pipeline 🌸

</div>

### ✿ Data Preparation

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  🌸 DATA PREPARATION                                                                 │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  SkinCAP Dataset (4,000 images)       Fuzzy Label Consolidation                   │
│  Dermatology photos + captions ──────►  (thefuzz — merge near-duplicate labels)   │
│  CSV: skincap_v240623.csv                            │                             │
│                                                       ▼                             │
│                                            Top-K Class Filtering                   │
│                                            (Top 10 most frequent diseases)         │
│                                                       │                             │
│                                                       ▼                             │
│  ┌────────────────────────────────────────────────────────────────────────┐       │
│  │  10 Skin Disease Classes                                               │       │
│  ├────────────────────────────────────────────────────────────────────────┤       │
│  │  Psoriasis · Melanocytic Nevi · SCCIS · Basal Cell Carcinoma          │       │
│  │  Acne Vulgaris · Lichen Planus · Scleroderma · Photodermatoses        │       │
│  │  Lupus Erythematosus · Sarcoidosis                                     │       │
│  └────────────────────────────────────────────────────────────────────────┘       │
│                                                       │                             │
│                                                       ▼                             │
│                             Stratified 911 train / 99 val split (locked)           │
│                             Sqrt oversampling for class imbalance (train only)     │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

<br/>

### ✿ Stage 1: Group Classification

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  🌸 STAGE 1: GROUP CLASSIFICATION                                                    │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Input Image ───────►  Qwen3-VL-8B  ──────────────────────────────► Group Label   │
│  (Skin photo)          (4-bit NF4)                                   (4 classes)   │
│                                                                                      │
│  Prompt: "What disease group? Inflammatory / Benign / Malignant / Acne"            │
│                                                                                      │
│  Groups:                                                                            │
│  ├── Inflammatory: Psoriasis, Lichen Planus, Lupus, Photodermatoses, Scleroderma  │
│  ├── Benign:       Melanocytic Nevi, SCCIS                                         │
│  ├── Malignant:    Basal Cell Carcinoma, Sarcoidosis                               │
│  └── Acne:         Acne Vulgaris                                                   │
│                                                                                      │
│  Training: 5 epochs · LoRA r=16 · Accuracy: 88.68%                                │
│  Weights saved → initialize Stage 2                                                 │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

<br/>

### ✿ Stage 2: RAG-in-Training Disease Classification

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  🌸 STAGE 2: RAG-IN-TRAINING DISEASE CLASSIFICATION (Key Contribution)              │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  RAG Index (911 training images, encoded by SigLIP-2 + BGE-M3, R2, α=0.9)         │
│                                              │                                      │
│  For each training sample ───────────────► Retrieve K=1 most similar reference    │
│                                              │                                      │
│                                              ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐        │
│  │  Multi-image Prompt (per sample)                                     │        │
│  │  [ref_img]   Reference: "psoriasis"                                  │        │
│  │              Erythematous plaques with silver scaling...             │        │
│  │  [query_img] What skin disease does this patient have?              │        │
│  └──────────────────────────────────────────────────────────────────────┘        │
│                                              │                                      │
│                                              ▼                                      │
│                               Qwen3-VL-8B → Disease Name                           │
│                               Loss: Cross-Entropy on predicted label token          │
│                                                                                      │
│  Training: 3 epochs · Accuracy: 85.86% (best: CLIP R0 at inference)               │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

<br/>

### ✿ Stage 3: Clinical Caption Generation (Merged-Init)

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  🌸 STAGE 3: CLINICAL CAPTION GENERATION (Merged-Init)                              │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Stage 2 LoRA adapters ────► merge_and_unload() ────► Merged base weights          │
│  (classification weights)      (Merged-Init trick)      (fresh LoRA for captions)  │
│                                              │                                      │
│                                              ▼                                      │
│  Input Image + Prompt ───► Qwen3-VL-8B (Merged-Init) ──────────► Caption Text     │
│  "Describe the skin                                                                  │
│   condition and                                                                      │
│   recommend treatment"                                                               │
│                                                                                      │
│  Optional STS (Selective Token Supervision):                                        │
│  • w_ans:    higher weight for diagnosis/recommendation sentences                   │
│  • w_reason: higher weight for clinical terminology                                 │
│  • w_surp:   higher weight for tokens the base model finds surprising               │
│  • IBR:      L2 regularization on LoRA parameters                                  │
│                                                                                      │
│  Training: 3 epochs · BLEU-4: 29.33 (vs 9.82 checkpoint init — 3× gain)           │
│  Loss: Cross-Entropy with STS token weighting                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

<br/>

```
      🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸
```

<br/>

<div align="center">

## 🌸 ReGrounding Module Development 🌸

</div>

### ✿ Building Knowledge Base (Week 4)

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  🌸 BUILDING KNOWLEDGE BASE (Week 4)                                                │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Training Cases ──────► Vision Encoder ────────► Image Features                    │
│  (Images + Masks)      (CLIP/DINOv2)                  │                            │
│                                                        │                            │
│                                                        ├────► Vector Database       │
│  Segmentation ───────► Mask Encoder ──────────► Mask Features  (FAISS Index)      │
│  Masks                                                 │                            │
│                                                        │                            │
│  Clinical Text ──────► Text Encoder ──────────► Text Features                     │
│  (Captions +           (BioBERT)                       │                            │
│   Guidelines)                                          │                            │
│                                                        │                            │
│  Treatment ───────────────────────────────────────────┴────► Metadata             │
│  Outcomes                                                     (Disease, Outcome,    │
│                                                                Treatment, etc.)     │
│                                                                                      │
│  Result: Knowledge Base with ~1000+ indexed cases                                  │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

<br/>

```
      🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸
```

<br/>

<div align="center">

## 🌸 Inference Pipeline 🌸

</div>

### ✿ User Input (via Web or LINE API)

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  🌸 USER INPUT (via Web or LINE API)                                               │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Patient Image ────────┐                                                            │
│  (Oral Photo)          │                                                            │
│                        ├──────────────────────────────────────────────┐            │
│  User Instruction ─────┤                                              │            │
│  (Thai or English)     │                                              │            │
│  e.g., "Segment areas  │                                              │            │
│   requiring treatment" │                                              │            │
└─────────────────────────┴──────────────────────────────────────────────┼────────────┘
```

<br/>

### ✿ Step 1: Initial Prediction

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  🌸 STEP 1: INITIAL PREDICTION (Trained Model from Stage 1-3)                      │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Image + Instruction ────► OralGPT Model ─────► Initial Results                    │
│                            (Stage 1-2-3)              │                             │
│                                                       ├──► Disease Classification    │
│                                                       ├──► Segmentation Masks        │
│                                                       ├──► Clinical Caption          │
│                                                       └──► Uncertainty Score         │
│                                                              │                       │
└──────────────────────────────────────────────────────────────┼───────────────────────┘
```

<br/>

### ✿ Step 2: Uncertainty Estimation

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  🌸 STEP 2: UNCERTAINTY ESTIMATION (Monte Carlo Dropout)                           │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Multiple Forward Passes ────► Statistical Analysis ────► Uncertainty Score        │
│  (with dropout enabled)         (variance calculation)         │                   │
│                                                                 │                   │
│                                                                 ▼                   │
│                                                    ┌────────────────────────┐      │
│                                                    │ If Uncertainty > 0.7   │      │
│                                                    │ → Retrieve more cases  │      │
│                                                    │    (k=10)              │      │
│                                                    │ Else                   │      │
│                                                    │ → Retrieve fewer (k=3) │      │
│                                                    └────────────────────────┘      │
│                                                                 │                   │
└─────────────────────────────────────────────────────────────────┼───────────────────┘
```

<br/>

### ✿ Step 3: Visual-Semantic Retrieval (ReGrounding)

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  🌸 STEP 3: VISUAL-SEMANTIC RETRIEVAL (ReGrounding)                                │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Query Encoding:                                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐              │
│  │  Patient Image ────► Vision Encoder ────► Image Features        │              │
│  │                                                  │                │              │
│  │  Predicted Masks ──► Mask Encoder ──────► Mask Features         │              │
│  │                                                  │                │              │
│  │  Instruction ──────► Text Encoder ──────► Text Features          │              │
│  │                                                  │                │              │
│  │                                                  ▼                │              │
│  │                                        Combined Query Vector      │              │
│  └─────────────────────────────────────────────────────────────────┘              │
│                                                  │                                  │
│                                                  ▼                                  │
│  Similarity Search:                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐              │
│  │  Query Vector ──────► Vector Database (FAISS) ──► Top-k Cases  │              │
│  │                                                           │       │              │
│  │                                                           ▼       │              │
│  │                              Retrieved Cases with:                │              │
│  │                              • Similar images                     │              │
│  │                              • Segmentation masks                 │              │
│  │                              • Disease diagnosis                  │              │
│  │                              • Treatment used                     │              │
│  │                              • Clinical outcomes                  │              │
│  │                              • Follow-up duration                 │              │
│  └─────────────────────────────────────────────────────────────────┘              │
│                                                  │                                  │
│                                                  ▼                                  │
│  Cross-Modal Re-ranking:                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐              │
│  │  For each retrieved case:                                        │              │
│  │    Score = α × Visual_Similarity                                 │              │
│  │          + (1-α) × Semantic_Similarity                           │              │
│  │          + 0.2 × Language_Match_Bonus                            │              │
│  │                                                                   │              │
│  │  α = 0.7 if uncertainty > 0.5, else 0.3 (adaptive weighting)    │              │
│  │                                                                   │              │
│  │  Sort by final score ──────────────► Top-k Ranked Cases         │              │
│  └─────────────────────────────────────────────────────────────────┘              │
│                                                  │                                  │
└──────────────────────────────────────────────────┼──────────────────────────────────┘
```

<br/>

### ✿ Step 4: Evidence-Based Suggestion Generation

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  🌸 STEP 4: EVIDENCE-BASED SUGGESTION GENERATION                                   │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Build Context from Retrieved Cases:                                               │
│  ┌─────────────────────────────────────────────────────────────────┐              │
│  │  For each retrieved case i:                                      │              │
│  │    • Image_i + Segmentation_i (visual template)                  │              │
│  │    • Description_i (clinical narrative)                          │              │
│  │    • Treatment_i (what was done)                                 │              │
│  │    • Outcome_i (result: improved/stable/worsened)                │              │
│  │    • Similarity_Score_i                                          │              │
│  └─────────────────────────────────────────────────────────────────┘              │
│                          │                                                          │
│                          ▼                                                          │
│  Aggregate Evidence:                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐              │
│  │  • Successful treatments: 12/15 cases improved with Treatment X  │              │
│  │  • Average follow-up: 6 months                                   │              │
│  │  • Complication rate: 2/15 cases                                 │              │
│  │  • Evidence strength: 12/15 = 80%                                │              │
│  └─────────────────────────────────────────────────────────────────┘              │
│                          │                                                          │
│                          ▼                                                          │
│  Generate Clinical Suggestion:                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐              │
│  │  LLM Input:                                                      │              │
│  │    • Current case: [Image + Segmentation + Classification]       │              │
│  │    • Retrieved evidence: [Top-k cases + outcomes]                │              │
│  │    • Uncertainty score: 0.65                                     │              │
│  │    • Language: Thai/English                                      │              │
│  │                                                                   │              │
│  │  Prompt Template:                                                │              │
│  │    "Based on {k} similar cases with {evidence_strength}          │              │
│  │     success rate:                                                │              │
│  │     • Diagnosis: [Disease Name]                                  │              │
│  │     • Segmentation: [Areas marked]                               │              │
│  │     • Recommended treatment: [Treatment X]                       │              │
│  │     • Expected outcome: [Predicted based on evidence]            │              │
│  │     • Monitoring plan: [Based on similar cases]                  │              │
│  │     • Confidence: {1 - uncertainty}                              │              │
│  │     • Evidence: {similar_case_summaries}                         │              │
│  │     {IF uncertainty > 0.7:                                       │              │
│  │        Recommendation: Consult specialist}"                      │              │
│  │                                                                   │              │
│  │  Temperature: 0.3 if uncertain, 0.7 if confident                │              │
│  └─────────────────────────────────────────────────────────────────┘              │
│                          │                                                          │
└──────────────────────────┼──────────────────────────────────────────────────────────┘
```

<br/>

### ✿ Final Output

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  🌸 FINAL OUTPUT                                                                    │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐                │
│  │  1. Disease Classification:                                     │                │
│  │     • Primary: Oral Lichen Planus (OLP)                         │                │
│  │     • Confidence: 85%                                            │                │
│  │                                                                  │                │
│  │  2. Segmentation Masks:                                         │                │
│  │     • Erosive Area 1: [Polygon coordinates] <seg1>             │                │
│  │     • Erosive Area 2: [Polygon coordinates] <seg2>             │                │
│  │     • White Striae: [Polygon coordinates] <seg3>               │                │
│  │                                                                  │                │
│  │  3. Clinical Description:                                       │                │
│  │     "Clinical examination reveals bilateral reticular           │                │
│  │      white striae on buccal mucosa with erosive areas..."       │                │
│  │                                                                  │                │
│  │  4. Evidence-Based Recommendation:                              │                │
│  │     "Based on 12 similar cases (80% success rate):              │                │
│  │      • Treatment: Topical corticosteroids                       │                │
│  │      • Expected improvement: 4-6 weeks                          │                │
│  │      • Monitor erosive areas <seg1>, <seg2> monthly             │                │
│  │      • Biopsy if no improvement in 8 weeks                      │                │
│  │      Evidence: 12/15 similar cases showed improvement"          │                │
│  │                                                                  │                │
│  │  5. Similar Cases References:                                   │                │
│  │     • Case #234: 85% similarity, Improved after 6 weeks         │                │
│  │     • Case #567: 82% similarity, Complete healing               │                │
│  │     • Case #891: 78% similarity, Partial improvement            │                │
│  │                                                                  │                │
│  │  6. Uncertainty & Safety:                                       │                │
│  │     • Uncertainty Score: 0.35 (Low)                             │                │
│  │     • Recommendation: Suitable for monitoring                   │                │
│  │     [If high uncertainty: "Consult specialist recommended"]     │                │
│  └────────────────────────────────────────────────────────────────┘                │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

<br/>

```
      🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸
```

<br/>

<div align="center">

## 🌸 User Interface Delivery 🌸

</div>

<table align="center">
<tr>
<td align="center" width="50%">

### 🌸 Web Interface

</td>
<td align="center" width="50%">

### 🌸 LINE API

</td>
</tr>
<tr>
<td>

- 📤 Upload image
- ✍️ Enter instruction (Thai or English)
- 🔍 View segmentation overlay
- 📋 Read detailed report
- 📚 Access similar cases
- 👩‍⚕️ Medical professional use

</td>
<td>

- 📱 Send image via LINE chat
- 💬 Receive results in chat
- 🗣️ Interactive Q&A
- 🔗 Share with healthcare provider
- 🎯 Simple interface for public
- ⚡ Quick screening

</td>
</tr>
</table>

<br/>

```
      🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸
```

<br/>

<div align="center">

## 🌸 Key Innovations 🌸

| | |
|:---:|:---|
| 🌸 | **Multilingual Support** — Thai & English |
| ✿ | **Visual Grounding** — Complex instruction understanding |
| 🌸 | **ReGrounding (Novel)** — Retrieval-augmented visual grounding |
| ✿ | **Uncertainty-Aware** — Knows when to consult specialists |
| 🌸 | **Evidence-Based** — Recommendations backed by clinical outcomes |

</div>

<br/>

---

<div align="center">

```
✿ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✿
                              🌸 HIKARI Project 🌸
                   Healthcare-oriented Intelligent Knowledge-Augmented
                          Retrieval and Inference system
✿ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✿
```

<sub>Made with 💗 and 🌸</sub>

</div>