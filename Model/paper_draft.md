# RAG-in-Training for Multi-Stage Vision-Language Model Fine-Tuning in Skin Disease Classification

**[DRAFT — ITC-CSCC 2026 submission | Target: 3–4 pages IEEE two-column]**

---

## Abstract

Automated skin disease classification from dermatological images remains challenging due to high intra-class visual similarity among inflammatory conditions. While Retrieval-Augmented Generation (RAG) has been applied at inference time to provide visual few-shot context, a train-inference distribution gap persists when models are trained without reference images. We propose **RAG-in-Training**, a strategy that incorporates retrieved reference images directly into every training sample, enabling the model to learn how to exploit reference context. Built upon a three-stage cascaded fine-tuning pipeline using Qwen3-VL-8B-Thinking on the SkinCAP dataset (10 disease classes), our best model achieves **85.86% accuracy** on a 99-sample validation set — a gain of +11.86% over the single-image baseline (74.0%) and +6.06% over inference-only RAG (79.80%). We further show that for multi-stage LoRA fine-tuning, **merging adapter weights before switching tasks** (merged-init) yields 3× improvement in caption BLEU-4 over continuing existing adapters. These findings provide practical guidelines for multi-stage VLM fine-tuning in medical imaging.

**Keywords:** skin disease classification, retrieval-augmented generation, vision-language model, LoRA fine-tuning, dermoscopy

---

## 1. Introduction

Dermoscopy-based skin disease classification is a key step in clinical dermatology workflows. Deep learning models have achieved competitive performance on standard benchmarks [HAM10000, ISIC], yet fine-grained discrimination among visually similar inflammatory diseases (psoriasis, lupus, lichen planus) remains a persistent challenge due to limited labeled training data per class.

Vision-Language Models (VLMs) such as Qwen3-VL offer an alternative: their image-text co-training enables rich visual reasoning, and their instruction-following capability supports flexible prompt-based inference. However, naively fine-tuning a VLM for disease classification — even with domain-specific data — yields suboptimal results because the model lacks explicit mechanisms to compare the query image against known reference cases.

RAG addresses this by retrieving similar training images at inference and presenting them as few-shot examples in the prompt. Prior work on RAG for VLMs has focused on inference-time retrieval [cite]. A critical gap remains: **if the model was never trained to use multi-image reference context, it may ignore or be confused by the injected references at inference time.**

We address this gap with three contributions:

1. **RAG-in-Training:** Each training sample includes a retrieved reference image and its label, teaching the model to actively leverage visual comparisons. This achieves 85.86% vs. 74.0% single-image baseline (+11.86%).

2. **Cascaded Fine-Tuning Pipeline:** A three-stage progression (group classification → disease classification → caption generation) with domain-specific LoRA at each stage, achieving successive knowledge accumulation.

3. **Merged-Init for Task Switching:** We empirically show that merging LoRA adapters into base weights before introducing a new fine-tuning task (Stage 3 caption) yields 3× BLEU-4 improvement over continuing existing adapters — a practical guideline for multi-stage LoRA pipelines.

---

## 2. Related Work

**Skin disease classification.** CNN-based methods (ResNet, EfficientNet) achieve strong performance on HAM10000 and ISIC datasets [cite]. More recent approaches apply transformer backbones [cite] and contrastive learning [cite]. VLM-based methods remain underexplored for fine-grained classification beyond zero-shot prompting [cite].

**Retrieval-Augmented Generation.** RAG for language models [cite RAG paper] has been extended to multimodal settings [cite]. For medical VQA, retrieval of similar cases improves accuracy [cite]. However, retrieval at inference without corresponding training-time exposure creates a distribution gap [cite].

**LoRA fine-tuning for VLMs.** Low-Rank Adaptation [cite LoRA] enables efficient fine-tuning of large VLMs. Cascaded LoRA fine-tuning across multiple tasks can suffer from catastrophic interference [cite], which our merged-init strategy directly addresses.

---

## 3. Method

### 3.1 Three-Stage Cascaded Fine-Tuning

Our pipeline progressively specializes a Qwen3-VL-8B-Thinking backbone across three stages:

```
Base Model (Qwen3-VL-8B-Thinking)
    ↓ Stage 1: Group Classification (4 groups)
    ↓ Stage 2: Disease Classification (10 classes) ← RAG-in-Training applied here
    ↓ Stage 3: Caption Generation
```

**Stage 1** fine-tunes the model to classify skin lesions into four disease groups (Inflammatory & Autoimmune, Benign Tumors & Nevi, Malignant Tumors, Acne & Follicular). Starting Stage 2 from Stage 1 weights provides domain-adapted visual representations (+4.6% over base-model initialization).

**Stage 2** fine-tunes for 10-class disease classification. All LoRA hyperparameters: r=16, α=32, dropout=0, LR=2e-4, 3 epochs, AdamW 8-bit, batch=8 (2×GPU + 4× gradient accumulation), on RTX 5070 Ti (15.92 GB VRAM).

**Stage 3** fine-tunes for clinical caption generation from the Stage 2 merged model (see Section 3.3).

### 3.2 RAG-in-Training

**Hybrid RAG Index.** For each training image, we pre-compute embeddings using a hybrid encoder (image encoder + optional text encoder on clinical captions). The index covers all 911 training images; no validation images are included to prevent leakage.

The retrieval score for a query against reference $i$ is:

$$\text{score}(q, i) = \alpha \cdot \cos(\mathbf{e}^{img}_q, \mathbf{e}^{img}_i) + (1-\alpha) \cdot \cos(\mathbf{e}^{txt}_q, \mathbf{e}^{txt}_i)$$

We evaluate five encoder configurations (Table 1). For Stage 2 RAG-in-Training we use R2 (SigLIP-2 + BGE-M3) with α=0.9 — the configuration with the highest inference-only accuracy.

**Training Sample Format.** Each Stage 2 training sample includes K=1 retrieved reference:

```
[ref_image]
Reference 1: {disease_label}
Description: {clinical_caption}
[query_image]
{classification_prompt}
Answer: {ground_truth_disease}
```

K=1 is used because K=3 exhausts 15.92 GB VRAM (4 images × batch=2 = 8 concurrent GPU forward passes). Reference images are pre-computed before loading the VLM to avoid memory conflicts.

**Pre-computation.** `precompute_rag_refs()` runs retrieval using SigLIP+BGE-M3 on GPU, saves reference paths and labels per training image, then frees GPU memory before Qwen3-VL is loaded.

### 3.3 Merged-Init for Stage 3 Caption Generation

A key finding in our pipeline is that **how Stage 2 knowledge is transferred to Stage 3 determines caption quality**. We compared two initialization strategies:

- **Way 1 (Checkpoint Init):** Load Stage 2 LoRA checkpoint → continue fine-tuning existing adapters on captions. Disease knowledge resides in LoRA adapters → caption gradient updates overwrite disease representations → catastrophic interference.

- **Way 2 (Merged Init):** Merge Stage 2 (LoRA → full weights) → add fresh LoRA adapters → fine-tune on captions. Disease knowledge is permanently encoded in base weights → fresh LoRA adapters specialize freely on caption generation without conflicting gradients.

**Key rule:** When switching tasks in multi-stage LoRA fine-tuning, always merge adapter weights into the base model before introducing fresh adapters for the new task.

### 3.4 Output Parsing

Fine-tuned VLMs output short disease names (1–3 words). Raw outputs are parsed by stripping `<think>...</think>` reasoning blocks, applying prompt-specific anchor extraction (e.g., `Diagnosis:` for P3), then fuzzy word-overlap matching (Jaccard threshold 91/100) against the 10 disease name strings. Unmatched outputs are counted as "Unknown" and treated as incorrect (denominator = total samples).

---

## 4. Experiments

### 4.1 Dataset

**SkinCAP** [cite]: 4,000 dermatological images with disease labels and clinical captions. We filter to 10 disease classes across 4 groups (1,010 images), applying a stratified split: **911 train / 99 val** (locked in `split_info_3stage.json`). Rare classes are sqrt-oversampled during training.

### 4.2 RAG Encoder Comparison (Inference-Only RAG)

We first evaluate five encoder configurations at inference with the s1cascade model (no RAG-in-Training) to identify the best encoder for subsequent RAG-in-Training.

**Table 1: RAG Encoder Configurations**

| ID | Image Encoder | Text Encoder | Strategy | α=0.5 Acc | Best α Acc |
|----|--------------|--------------|----------|-----------|-----------|
| R0 | CLIP ViT-B/32 | — | Image-only | 72.73% | 72.73% |
| R1 | CLIP ViT-B/32 | ClinicalBERT | Hybrid | 69.70% | 71.72% (α=0.9) |
| **R2** | **SigLIP-2** | **BGE-M3** | **Hybrid** | **74.75%** | **79.80% (α=0.9)** |
| R3 | Jina-CLIP-v2 | MedCPT | Hybrid | 66.67% | 78.79% (α=0.7) |
| R4 | Nomic Vision | Nomic Text | Cross-modal | 67.68% | 72.73% (α=0.9) |

R2 (SigLIP-2 + BGE-M3) at α=0.9 achieves the best inference-only accuracy (79.80%) and is selected for RAG-in-Training.

### 4.3 Main Results

**Table 2: Classification Accuracy (99 val samples)**

| Model | RAG at Train | RAG at Inference | Prompt | Accuracy |
|-------|-------------|-----------------|--------|---------|
| fuzzytopk | — | — | P0 | 74.00% |
| Qwen2.5-VL zero-shot | — | R0 | P2 | 50.51% |
| Qwen3-VL zero-shot | — | R0 | P3 | 33.33% |
| s1cascade | — | R2 (α=0.5) | P0 | 74.75% |
| s1cascade | — | R2 (α=0.9) | P0 | 79.80% |
| **ragR2_a09** | **R2 (K=1)** | **R0 (CLIP)** | **P0** | **85.86%** |
| ragR2_a09 | R2 (K=1) | R2 (α=0.9) | P0 | 82.83% |
| M1 (3-stage) | — | R0 | P1 | 59.38% |

RAG-in-Training (ragR2_a09) achieves **85.86%**, outperforming:
- Single-image baseline (fuzzytopk): +11.86%
- Inference-only RAG best (s1cascade R2 α=0.9): +6.06%
- Zero-shot Qwen2.5: +35.35%

**Surprising finding:** CLIP (R0) at inference outperforms the SigLIP+BGE-M3 encoder used during training (+3.03%). This suggests that RAG-in-Training teaches the model a **general "reference-and-compare" reasoning skill** rather than encoder-specific feature matching.

### 4.4 Per-Disease Analysis

**Table 3: Per-disease sensitivity (%), ragR2_a09 R0-P0**

| Disease | s1cascade R2 α=0.9 | ragR2_a09 R0 | Δ |
|---------|-------------------|-------------|---|
| Psoriasis | 61.5% | **100.0%** | +38.5% |
| SCCIS | 100.0% | **100.0%** | — |
| BCC | 76.9% | **100.0%** | +23.1% |
| Melanocytic nevi | 91.7% | **100.0%** | +8.3% |
| Acne vulgaris | 100.0% | **100.0%** | — |
| Lichen planus | 88.9% | 88.9% | — |
| Scleroderma | 75.0% | 87.5% | +12.5% |
| Photodermatoses | 62.5% | 75.0% | +12.5% |
| Lupus erythematosus | 77.8% | 55.6% | −22.2% |
| **Sarcoidosis** | **57.1%** | **14.3%** | **−42.8%** |

5/10 diseases reach 100% sensitivity with RAG-in-Training. Psoriasis shows the largest gain (+38.5%), demonstrating improved discrimination of visually similar inflammatory conditions through reference-based learning. **Sarcoidosis is a notable failure** (57.1% → 14.3%): we hypothesize that over-reliance on reference agreement causes misclassification when sarcoidosis references are visually ambiguous. Addressing this via contrastive hard-negative training is reserved for future work.

### 4.5 Stage 3 Caption Generation Ablation

**Table 4: Stage 3 caption BLEU scores (Way 1/2 × STS)**

| Exp | Init | STS+IBR | BLEU-1 | BLEU-4 | ROUGE-1 |
|-----|------|---------|--------|--------|---------|
| Baseline | Checkpoint | ✗ | 33.07 | 9.82 | 38.50 |
| Way 1 | Checkpoint | ✗ | 29.22 | 9.82 | 38.90 |
| **Way 2** | **Merged** | **✗** | **42.22** | **29.33** | **53.55** |
| Way 1 + STS | Checkpoint | ✓ | 0.01 | 0.00 | 5.03 |
| Way 2 + STS | Merged | ✓ | 1.78 | 0.61 | 15.68 |

**Merged-init (Way 2) achieves BLEU-4=29.33**, a 3× improvement over checkpoint-init (9.82). This confirms that disease knowledge permanently encoded in base weights does not interfere with caption learning via fresh LoRA adapters.

**STS+IBR fails in both conditions.** IBR (L2 penalty on LoRA norms) was designed for near-zero initial norms; in multi-stage transfer learning, prior LoRA norms are large, and IBR drives them toward zero — erasing learned representations. This is a negative result that we report as a practical warning against applying IBR in cascaded fine-tuning settings.

### 4.6 Prompt Variant Analysis

| Prompt | fuzzytopk | s1cascade | M1 (R0) |
|--------|-----------|-----------|---------|
| P0 (direct) | **74.0%** | **74.7%** | 55.0% |
| P1 (CoT steps) | 71.7% | 72.7% | **59.4%** |
| P2 (differential) | 70.7% | 73.7% | 53.8% |
| P3 (structured) | 71.7% | 71.7% | 50.6% |

P0 is optimal for short-output fine-tuned models; P1 benefits the CoT-oriented M1 model (+4.4% with RAG).

---

## 5. Conclusion

We presented RAG-in-Training, a strategy that closes the train-inference distribution gap in VLM-based skin disease classification by incorporating retrieved reference images into every training sample. On the SkinCAP 10-class benchmark, our method achieves 85.86% accuracy (+11.86% over the single-image baseline, +6.06% over inference-only RAG). We further demonstrate that merged-init — merging prior-stage LoRA adapters before adding fresh adapters for a new task — yields 3× improvement in clinical caption BLEU-4, providing a practical guideline for multi-stage VLM fine-tuning. A key failure case (sarcoidosis sensitivity collapse to 14.3%) motivates future work on contrastive hard-negative RAG training. The full pipeline and ablation code is available at [repository URL].

---

## References

[1] Codella, N., et al. "Skin lesion analysis toward melanoma detection: ISIC 2018 challenge." arXiv:1902.03368, 2019.

[2] Tschandl, P., et al. "The HAM10000 dataset." Scientific Data, 5(1), 2018.

[3] Hu, E., et al. "LoRA: Low-rank adaptation of large language models." ICLR 2022.

[4] Lewis, P., et al. "Retrieval-augmented generation for knowledge-intensive NLP tasks." NeurIPS 2020.

[5] Bai, J., et al. "Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond." arXiv:2308.12966, 2023.

[6] Radford, A., et al. "Learning transferable visual models from natural language supervision." ICML 2021. [CLIP]

[7] Zhai, X., et al. "Sigmoid loss for language image pre-training." ICCV 2023. [SigLIP]

[8] Xiao, S., et al. "C-pack: Packaged resources to advance general Chinese embedding." arXiv:2309.07597, 2023. [BGE-M3]

[9] Jin, Q., et al. "MedCPT: Contrastive pre-trained transformers with large-scale PubMed search logs for zero-shot biomedical information retrieval." Bioinformatics, 2023.

[10] Méndez, D., et al. "SkinCAP: A multi-modal dermatology dataset annotated with rich medical captions." NeurIPS 2022 Datasets & Benchmarks.

---

*[TODO before submission:]*
- *[ ] Add 1–2 more SOTA comparison rows in Table 2 (find published numbers on SkinCAP or similar)*
- *[ ] Add confidence intervals to Table 2 main results (±CI at 95%)*
- *[ ] Replace [cite] placeholders with actual references*
- *[ ] Format to IEEE two-column template (.tex)*
- *[ ] Check page count in IEEE template (target: 3 pages)*
- *[ ] Add Figure 1: Pipeline diagram (Stage 1→2→3 + RAG-in-Training flow)*
- *[ ] Add Figure 2: Per-disease bar chart (Table 3 visualized)*
- *[ ] Confirm SkinCAP citation [10] is correct*
