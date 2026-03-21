# HIKARI — Total Experiment Summary

> Complete count of all training runs, inference benchmarks, and evaluations conducted across the project.

---

## 1. Training Experiments — 13 Runs

### Stage 1 (1 run)

| # | Model | HF Name | Accuracy |
|:-:|:------|:--------|:--------:|
| 1 | 4-class Group Classifier | HIKARI-Subaru-8B-SkinGroup | 88.68% |

### Stage 2 — Disease Classification (8 runs)

| # | Model | HF Name | Accuracy | Notes |
|:-:|:------|:--------|:--------:|:------|
| 2 | Single-Image FT | HIKARI-Altair-8B-SkinDx | 74.00% | Baseline — no RAG |
| 3 | Cascaded FT | HIKARI-Deneb-8B-SkinDx-Cascade | 79.80% | Cascade from Stage 1 weights |
| 4 | **RAG-in-Training** | **HIKARI-Sirius-8B-SkinDx-RAG** ⭐ | **85.86%** | **Key contribution** |
| 5 | M0 — No-Context 3-Stage | (ablation only) | 61.0% | No group context |
| 6 | M1 — GT-Oracle 3-Stage | HIKARI-Polaris-8B-SkinDx-Oracle | 59.38%* | GT group at inference |
| 7 | M2 — Oracle-Train / Cascade-Infer | (ablation only) | 57.4% | Train-inference mismatch |
| 8 | M3 — Full-Cascade 3-Stage | (ablation only) | 57.4% | Stage 1 predicted both |
| 9 | M4 — Soft-Probability 3-Stage | (ablation only) | 61.0% | Beam soft probs at inference |

*\* Requires ground-truth group at inference — oracle reference only*

### Stage 3 — Clinical Caption Generation (4 runs)

| # | Model | HF Name | BLEU-4 | Notes |
|:-:|:------|:--------|:------:|:------|
| 10 | Way 1 — Checkpoint-Init | HIKARI-Rigel-8B-SkinCaption | 9.82 | Catastrophic forgetting |
| 11 | **Way 2 — Merged-Init** | **HIKARI-Vega-8B-SkinCaption-Fused** ⭐ | **29.33** | **Best caption model** |
| 12 | Way 1 + STS | (collapsed, not published) | 0.00 | Complete training collapse |
| 13 | Way 2 + STS | HIKARI-Antares-8B-SkinCaption-STS | 0.61 | STS-induced collapse |

---

## 2. Inference Benchmark Configurations — ~300+ Combos

### RAG Encoders (R0–R4) × 5

| ID | Encoders | Role |
|:--:|:---------|:-----|
| R0 | CLIP ViT-B/32 (image only) | **Used at inference** |
| R1 | CLIP + ClinicalBERT | Clinical text variant |
| R2 | SigLIP-2 + BGE-M3 (α=0.9) | **Used during training** |
| R3 | Jina-CLIP + MedCPT | Medical text-visual fusion |
| R4 | Nomic Vision + Nomic Text | Cross-modal unified space |

### Alpha Tuning for R2 × ~4 values

`α ∈ {0.3, 0.5, 0.7, 0.9}` — controls image/text weight ratio (best: α=0.9)

### Prompt Templates (P0–P3) × 4

| ID | Style | Used With |
|:--:|:------|:----------|
| P0 | Direct Clinical Observation | fuzzytopk, s1cascade, ragR2 |
| P1 | Step-by-Step CoT | M1 oracle |
| P2 | Differential Diagnosis | Qwen2.5 zero-shot |
| P3 | Structured Clinical Assessment | Qwen3 zero-shot |

### Models Benchmarked × 5+

| Model | Type |
|:------|:-----|
| fuzzytopk | Fine-tuned |
| fuzzytopk_s1cascade | Fine-tuned |
| fuzzytopk_s1cascade_ragR2_a09 | Fine-tuned |
| M1 (GT-Oracle) | Fine-tuned |
| Qwen3-VL-8B (no fine-tune) | Zero-shot frontier |
| Qwen2.5-VL-7B (no fine-tune) | Zero-shot frontier |
| Gemini-2.5-Pro (no fine-tune) | Zero-shot frontier |

### Stage 3 Caption Evaluation × 2 modes

| Mode | BLEU-4 | BERTScore-F | Disease Correctness |
|:-----|:------:|:-----------:|:-------------------:|
| noguide (standard) | **29.33** | **91.12** | 62.63% |
| guided (disease label injected) | 13.11 | 88.57 | **78.79%** |

---

## 3. Special Experiments

| Experiment | Description | Result |
|:-----------|:------------|:-------|
| **B4 — Stage 3 as RAG Query** | Using Stage 3 caption output as text query for Stage 2 RAG | Explored cross-stage retrieval |
| **Alpha Tuning Grid** | Systematic α sweep for R2 hybrid scoring | Best: α=0.9 |
| **Encoder-Agnostic Test** | Train with R2, evaluate with R0 | +3.03% vs R2 inference — confirmed encoder-agnostic generalization |
| **STS Ablation** | Selective Token Supervision on Way 1 and Way 2 | Both collapsed — confirmed STS incompatible with SkinCAP scale |

---

## 4. Evaluation & Benchmarking

### Speed Benchmark — 3 Engines × 2 Stages × 2 Batch Sizes

| Engine | Stage 2 bs=1 | Stage 2 bs=4 | Stage 3 bs=1 | Stage 3 bs=4 |
|:-------|:-----------:|:-----------:|:-----------:|:-----------:|
| Unsloth 4-bit | baseline | baseline | 6,699 ms/img | 3,003 ms/img |
| vLLM BnB-4bit | ~2× | ~3× | 2,957 ms/img | 1,094 ms/img |
| SGLang FP8 | ~5× | **~10×** | 1,695 ms/img | **584 ms/img** ⚡ |

### SGLang FP8 Accuracy Evaluation

Full 99-sample validation set evaluation — confirmed accuracy trade-off: −5 pp vs BnB-4bit in exchange for ~10× throughput.

### BERTScore Multi-Model Comparison

Evaluated caption quality across multiple BERT checkpoints (roberta-large, bert-base, bert-large, etc.) to ensure metric robustness.

---

## 5. Summary Count

| Category | Count |
|:---------|:-----:|
| **Training runs** | **13** |
| RAG encoder configs | 5 |
| Alpha values tested | ~4 |
| Prompt templates | 4 |
| Models benchmarked at inference | 7+ |
| Inference benchmark combos (est.) | **~300+** |
| Special experiments | 4 |
| Speed benchmark configs | 12 |
| Caption eval modes | 2 |
| BERTScore BERT variants | 4+ |
| **Models published on HuggingFace** | **14** |
| **Documentation sections** | **17** |
| | |
| **Total experiments (approx.)** | **~330** |

---

## 6. Key Outcomes

| Milestone | Value |
|:----------|:------|
| Best disease accuracy | **85.86%** — HIKARI-Sirius (RAG-in-Training, R2 train → R0 inference) |
| Best caption BLEU-4 | **29.33** — HIKARI-Vega (Merged-Init, noguide) |
| Best caption BERTScore-F | **91.12** — HIKARI-Vega |
| Best inference throughput | **584 ms/img** — SGLang FP8 batch=4 (11.5× vs Unsloth) |
| Group classification | **88.68%** — HIKARI-Subaru |
| Models on HuggingFace | **14** (8 merged ~17 GB + 6 LoRA ~1.1 GB each) |

---

*Generated: 2026-03-21 · HIKARI Project · King Mongkut's Institute of Technology Ladkrabang (KMITL)*
