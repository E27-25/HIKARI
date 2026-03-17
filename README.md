<!-- ✿ HIKARI - Healthcare-oriented Intelligent Knowledge-Augmented Retrieval and Inference system ✿ -->

<div align="center">

<!-- 🌸 Logo 🌸 -->
<img src="logo/HIKARI logo.png" alt="HIKARI Logo" width="800"/>

<br/>

```
✿ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✿
```

### 🌸 *A RAG-in-Training Vision-Language Model for Fine-Grained Skin Lesion Diagnosis* 🌸

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

> **HIKARI** is a RAG-in-Training pipeline that fine-tunes **Qwen3-VL-8B-Thinking** on the SkinCAP dermatology dataset using a novel training strategy: injecting retrieved reference images *during training* so the model learns to leverage visual similarity evidence — achieving **85.86% accuracy** on 10-class skin disease classification.

```
✿ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✿
```

<div align="center">

## 🌸 Results 🌸

</div>

<table align="center">
<tr>
<td width="50%">

### 🏆 Model Comparison

| Model | Acc |
|:---|:---:|
| 🥇 **HIKARI (RAG-in-Training)** | **85.86%** |
| 🥈 Cascaded FT + Inference RAG | 79.80% |
| 🥉 Single-Image Fine-Tune | 74.00% |
| Zero-Shot Frontier (best) | 50.51% |
| Base Qwen3-VL-8B (no FT) | 33.33% |

</td>
<td width="50%">

### 🔬 Per-Disease Sensitivity

| Disease | Sensitivity |
|:---|:---:|
| 🟢 Psoriasis | **100%** (13/13) |
| 🟢 Melanocytic Nevi | **100%** (12/12) |
| 🟢 SCCIS | **100%** (12/12) |
| 🟢 Basal Cell Ca. | **100%** (13/13) |
| 🟢 Acne Vulgaris | **100%** (8/8) |
| 🟡 Lichen Planus | 88.9% (8/9) |
| 🟡 Scleroderma | 87.5% (7/8) |
| 🟡 Photodermatoses | 75.0% (6/8) |
| 🔴 Lupus Erythematosus | 55.6% (5/9) |
| 🔴 Sarcoidosis | 14.3% (1/7) |

</td>
</tr>
</table>

| Component | Detail |
|:---|:---|
| Backbone | Qwen3-VL-8B-Thinking |
| Dataset | SkinCAP — 4,000 dermatology images, 10 disease classes |
| RAG Encoder (training) | SigLIP-2 + BGE-M3 (R2), K=1 reference per sample |
| RAG Encoder (inference) | CLIP ViT-B/32 (R0), K=3 references |
| GPU | NVIDIA RTX 5070 Ti · 4-bit NF4 quantization |
| Training time | ~1h 44min · LoRA rank 16 |

```
✿ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✿
```

<div align="center">

## 🌸 HIKARI Pipeline 🌸

</div>

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  🌸 STAGE 1 — GROUP CLASSIFICATION                                                   │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Skin Image ───► Qwen3-VL-8B (LoRA) ────────────────────────────► Group Label      │
│                                                                    (4 classes)       │
│                                                                                      │
│  Groups: Inflammatory · Benign Tumor · Malignant Tumor · Acne                       │
│  Result: 88.68% group classification accuracy                                       │
│  Weights transferred → initialize Stage 2                                           │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  🌸 STAGE 2 — RAG-IN-TRAINING DISEASE CLASSIFICATION (Key Contribution)             │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  RAG Index (911 train images, SigLIP-2 + BGE-M3, R2, α=0.9)                        │
│                   │                                                                  │
│                   ▼  For each training sample → retrieve K=1 most similar           │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  [ref_img]   Reference: "psoriasis"                                         │    │
│  │              "Erythematous plaques with silver scaling..."                   │    │
│  │  [query_img] What skin disease does this patient have?                      │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                   │                                                                  │
│                   ▼  Qwen3-VL-8B → Disease Name (10 classes)                        │
│                                                                                      │
│  At inference: CLIP (R0) retrieves K=3 refs → 85.86% accuracy                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  🌸 STAGE 3 — CLINICAL CAPTION GENERATION (Merged-Init)                             │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Stage 2 LoRA ──► merge_and_unload() ──► Merged weights ──► Fresh LoRA adapters    │
│                    (Merged-Init trick)                                               │
│                                         │                                           │
│  Input Image + Prompt ──────────────────▼──► Clinical Caption                      │
│  "Describe and recommend treatment"         (BLEU-4: 29.33 vs 9.82 baseline)       │
│                                                                                      │
│  Optional STS: per-token loss weights for clinical terminology                      │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

```
      🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸 ✿ 🌸
```

<div align="center">

## 🌸 Attention Visualization 🌸

*LM Prefill Attention — HIKARI focuses on disease-relevant visual regions*

</div>

<table align="center">
<tr>
<th align="center">Melanocytic Nevi</th>
<th align="center">Basal Cell Carcinoma</th>
</tr>
<tr>
<td align="center">
<img src="Model/gradcam_outputs/melanocytic_nevi_2_comparison.png" width="380" alt="Melanocytic Nevi Attention"/>
</td>
<td align="center">
<img src="Model/gradcam_outputs/basal_cell_carcinoma_3_comparison.png" width="380" alt="BCC Attention"/>
</td>
</tr>
<tr>
<td align="center"><i>Network & border focus</i></td>
<td align="center"><i>Nodular lesion annotation</i></td>
</tr>
</table>

> **Left:** Base Qwen3-VL-8B (unfocused, scattered) → **Right:** HIKARI (disease-specific region focus)

```
✿ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✿
```

<div align="center">

## 🌸 Key Findings 🌸

</div>

| | |
|:---:|:---|
| 🌸 | **RAG-in-Training closes the train/inference gap** — Training with K=1 reference per sample teaches the model to use retrieval context; no architectural changes needed |
| ✿ | **Encoder-agnostic generalization** — Trained with SigLIP+BGE-M3 (R2) but best performance with CLIP (R0) at inference (+3.03 pp) — model learns the *concept*, not encoder-specific features |
| 🌸 | **Merged-Init prevents catastrophic forgetting** — Merging LoRA into base weights before Stage 3 → BLEU-4: 9.82 → **29.33 (3×)** |
| ✿ | **Group cascade hurts more than it helps** — 3-stage M-series (oracle: 66%) still underperforms 2-stage FuzzyTopK (74%) due to cascade penalty from weight initialization mismatch |

```
✿ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✿
```

<div align="center">

## 🌸 Quick Start 🌸

</div>

```bash
git clone https://github.com/E27-25/HIKARI.git
cd HIKARI/Model
pip install -r requirements.txt

# Train — RAG-in-Training (Stage 1 + 2)
python train_two_stage_FuzzyTopK.py \
    --start_from_stage1 --rag_k_train 1 --rag_exp R2 --alpha 0.9

# Evaluate — Full benchmark
python run_rag_benchmark.py \
    --method fuzzytopk_s1cascade_ragR2_a09 --rag_exp R0

# Attention visualization
python gradcam_visualization.py
```

> See [`Model/README.md`](Model/README.md) for the full technical reference — all scripts defined, all experiments documented.

```
✿ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✿
```

<div align="center">

## 🌸 RAG Encoder Configurations 🌸

</div>

| ID | Image Encoder | Text Encoder | Role |
|:--:|:---|:---|:---|
| **R0** | CLIP ViT-B/32 | — | ✅ Best at **inference** |
| R1 | CLIP | ClinicalBERT | Generic clinical text |
| **R2** | SigLIP-2 | BGE-M3 | ✅ Used during **training** |
| R3 | Jina-CLIP-v2 | MedCPT | Best with α=0.7 |
| R4 | Nomic Vision | Nomic Text | Cross-modal unified |

```
✿ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✿
```

<div align="center">

## 🌸 Citation 🌸

</div>

```bibtex
@inproceedings{hikari2025,
  title     = {HIKARI: RAG-in-Training for Fine-Grained Skin Lesion Diagnosis
               with Vision-Language Models},
  booktitle = {Proceedings of ITC-CSCC 2025},
  year      = {2025}
}
```

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
