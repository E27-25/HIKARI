<!-- ✿ HIKARI README ✿ -->

<div align="center">

<img src="logo/HIKARI logo.png" alt="HIKARI Logo" width="100%"/>

<br/>

<!-- Typing animation -->
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&duration=3000&pause=800&color=FF9EBC&center=true&vCenter=true&width=750&lines=HIKARI+%F0%9F%8C%B8;RAG-in-Training+%C3%97+Vision-Language+Model;85.86%25+Skin+Disease+Classification;Qwen3-VL-8B+%C3%97+SkinCAP+%C3%97+Hybrid+RAG;ITC-CSCC+2025" alt="Typing SVG" />

<br/><br/>

<!-- Badges -->
[![Accuracy](https://img.shields.io/badge/Accuracy-85.86%25-brightgreen?style=for-the-badge&logo=checkmarx&logoColor=white)](Model/README.md)
[![Model](https://img.shields.io/badge/Backbone-Qwen3--VL--8B--Thinking-4B9EFF?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking)
[![Dataset](https://img.shields.io/badge/Dataset-SkinCAP_4K-FF8C00?style=for-the-badge&logo=databricks&logoColor=white)](https://huggingface.co/datasets/joshuachou/SkinCAP)

[![GPU](https://img.shields.io/badge/GPU-RTX_5070_Ti-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](.)
[![Paper](https://img.shields.io/badge/Paper-ITC--CSCC_2025-E63946?style=for-the-badge&logo=arxiv&logoColor=white)](Model/Conference_Paper.tex)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](.)

<br/>

> **HIKARI** *(光)* — Healthcare-oriented Intelligent Knowledge-Augmented Retrieval and Inference system
>
> A RAG-in-Training pipeline that fine-tunes **Qwen3-VL-8B-Thinking** on dermatology images,
> injecting retrieved reference cases *during training* to achieve **85.86%** on 10-class skin disease classification.

</div>

---

## 📌 What is HIKARI?

| Letter | Meaning |
|:------:|:--------|
| **H** | **H**ealthcare-oriented |
| **I** | **I**ntelligent |
| **K** | **K**nowledge-Augmented |
| **A** | **A**ugmented |
| **R** | **R**etrieval and |
| **I** | **I**nference system |

HIKARI introduces **RAG-in-Training** — instead of using retrieval only at inference time, reference images are retrieved and injected into *every training sample*, so the model learns to reason about visual similarity evidence from the start. At inference, even switching to a different (simpler) retrieval encoder (CLIP) still yields best-in-class results, demonstrating true encoder-agnostic generalization.

---

## 🏆 Results at a Glance

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=18&duration=2500&pause=600&color=4ADE80&center=true&vCenter=true&width=600&lines=85.86%25+Overall+Accuracy;5+Diseases+at+100%25+Sensitivity;%2B6.06%25+over+previous+best+(79.80%25);MCC%3A+0.843+%C2%B7+Kappa%3A+0.842" alt="Results" />

</div>

<table>
<tr>
<td width="48%">

### 📊 Model Comparison

| Rank | Model | Accuracy |
|:----:|:------|:--------:|
| 🥇 | **HIKARI (RAG-in-Training)** | **85.86%** |
| 🥈 | Cascaded FT + Inference RAG | 79.80% |
| 🥉 | Single-Image Fine-Tune | 74.00% |
| 4 | Zero-Shot (Qwen2.5 + R0+P2) | 50.51% |
| 5 | Base Qwen3-VL-8B (no FT) | 33.33% |

</td>
<td width="52%">

### 🔬 Per-Disease Sensitivity

| Disease | Sensitivity |
|:--------|:-----------:|
| 🟢 Psoriasis | **100%** (13/13) |
| 🟢 Melanocytic Nevi | **100%** (12/12) |
| 🟢 SCCIS | **100%** (12/12) |
| 🟢 Basal Cell Carcinoma | **100%** (13/13) |
| 🟢 Acne Vulgaris | **100%** (8/8) |
| 🟡 Lichen Planus | 88.9% (8/9) |
| 🟡 Scleroderma | 87.5% (7/8) |
| 🟡 Photodermatoses | 75.0% (6/8) |
| 🔴 Lupus Erythematosus | 55.6% (5/9) |
| 🔴 Sarcoidosis | 14.3% (1/7) |

</td>
</tr>
</table>

---

## ⚙️ System Configuration

| Component | Detail |
|:----------|:-------|
| 🧠 Backbone | Qwen3-VL-8B-Thinking (4-bit NF4 quantization) |
| 📦 Dataset | SkinCAP — 4,000 dermatology images · 10 disease classes |
| 🔍 RAG Encoder (training) | SigLIP-2 + BGE-M3 (R2), α=0.9, K=1 reference per sample |
| 🔍 RAG Encoder (inference) | CLIP ViT-B/32 (R0), K=3 references |
| 🎛️ LoRA | rank=16, alpha=32 · AdamW 8-bit paged |
| 🖥️ GPU | NVIDIA RTX 5070 Ti (15.92 GB VRAM) |
| ⏱️ Training Time | ~1h 44min · 1,314 steps |
| 🖼️ Image Size | 672×672 thumbnail (LANCZOS) |

---

## 🏗️ HIKARI Pipeline

```
╔══════════════════════════════════════════════════════════════════════╗
║  STAGE 1 — GROUP CLASSIFICATION                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Skin Image ──► Qwen3-VL-8B ──► 4 Disease Groups  (88.68% acc)     ║
║                                                                      ║
║  ┌─ Inflammatory  (Psoriasis · Lichen · Lupus · Photo · Sclero)     ║
║  ├─ Benign Tumor  (Melanocytic Nevi · SCCIS)                        ║
║  ├─ Malignant     (Basal Cell Carcinoma · Sarcoidosis)              ║
║  └─ Acne          (Acne Vulgaris)                                   ║
║                                                                      ║
║  Weights → initialize Stage 2                                        ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║  STAGE 2 — RAG-IN-TRAINING  ⭐ Key Contribution                     ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  RAG Index  ←  911 train images encoded by SigLIP-2 + BGE-M3 (R2)  ║
║       │                                                              ║
║       ▼  For every training sample → retrieve K=1 most similar      ║
║                                                                      ║
║  ╔══════════════════════════════════════════╗                        ║
║  ║  [ref_img]   Reference: "psoriasis"      ║                       ║
║  ║              Erythematous plaques with   ║                       ║
║  ║              silver scaling...           ║                       ║
║  ║  [query_img] What skin disease is this? ║                       ║
║  ╚══════════════════════════════════════════╝                        ║
║       │                                                              ║
║       ▼  → Qwen3-VL-8B → Disease Name  (CE loss on label token)     ║
║                                                                      ║
║  At inference:  CLIP R0 (K=3) → 85.86%  ← encoder-agnostic!        ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║  STAGE 3 — CLINICAL CAPTION GENERATION  (Merged-Init)               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Stage 2 LoRA ──► merge_and_unload() ──► Merged base weights        ║
║                                                  │                   ║
║                                                  ▼                   ║
║  Image + Prompt ────────────────────────► Fresh LoRA ──► Caption    ║
║  "Describe condition & recommend treatment"                          ║
║                                                                      ║
║  BLEU-4: 9.82 (checkpoint init) → 29.33 (merged init)  ← 3× gain!  ║
║  Optional STS: per-token loss weights for clinical terminology       ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 🔭 Attention Visualization

> HIKARI learns to focus on **disease-relevant visual regions** — compared to the unfocused base model.

<table align="center">
<tr>
<th align="center" width="50%">🔵 Base Qwen3-VL-8B (unfocused)</th>
<th align="center" width="50%">🌸 HIKARI (disease-specific focus)</th>
</tr>
<tr>
<td align="center" colspan="2">
<img src="Model/gradcam_outputs/melanocytic_nevi_2_comparison.png" width="700" alt="Melanocytic Nevi — Base vs HIKARI"/>
<br/><i>Melanocytic Nevi — LM Prefill Attention (last token → image patches)</i>
</td>
</tr>
<tr>
<td align="center" colspan="2">
<img src="Model/gradcam_outputs/basal_cell_carcinoma_3_comparison.png" width="700" alt="BCC — Base vs HIKARI"/>
<br/><i>Basal Cell Carcinoma — Nodular lesion region annotation</i>
</td>
</tr>
</table>

---

## 🔑 Key Findings

| # | Finding |
|:-:|:--------|
| 1 | 🌸 **RAG-in-Training closes the train/inference gap** — Training with K=1 reference per sample teaches the model to use visual retrieval context. No architectural changes required. |
| 2 | ✿ **Encoder-agnostic generalization** — HIKARI trained with SigLIP+BGE-M3 (R2) but performs **best with CLIP (R0)** at inference (+3.03 pp). The model learns the *concept* of reference-guided diagnosis, not encoder-specific features. |
| 3 | 🌸 **Merged-Init prevents catastrophic interference** — Merging Stage 2 LoRA into base weights before Stage 3 training: BLEU-4 **9.82 → 29.33 (3×)**. Never fine-tune existing adapters on a different task in-place. |
| 4 | ✿ **Group cascade hurts more than it helps** — 3-stage M-series (oracle: 66%) still underperforms 2-stage FuzzyTopK (74%). Cascade penalty from weight initialization mismatch outweighs group-context benefit. |

---

## 🔍 RAG Encoder Reference (R0–R4)

| ID | Image Encoder | Text Encoder | α | Role |
|:--:|:-------------|:-------------|:-:|:-----|
| **R0** | `openai/clip-vit-base-patch32` | — | — | ✅ Best at **inference** |
| R1 | CLIP | `medicalai/ClinicalBERT` | 0.5 | Clinical text variant |
| **R2** | `google/siglip-2-base-patch16-512` | `BAAI/bge-m3` | 0.9 | ✅ Used during **training** |
| R3 | `jinaai/jina-clip-v2` | `ncbi/MedCPT-Query-Encoder` | 0.7 | Best text-visual fusion |
| R4 | `nomic-ai/nomic-embed-vision-v1.5` | `nomic-ai/nomic-embed-text-v1.5` | 0.5 | Cross-modal unified |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/E27-25/HIKARI.git
cd HIKARI/Model
pip install -r requirements.txt
huggingface-cli login  # Required for Qwen3-VL model access
```

### 2. Train — RAG-in-Training

```bash
# Stage 1 + 2: Group classification → RAG-in-Training disease classification
python train_two_stage_FuzzyTopK.py \
    --start_from_stage1 \
    --rag_k_train 1 \
    --rag_exp R2 \
    --alpha 0.9

# Stage 3: Clinical caption generation (Merged-Init, best config)
python train_two_stage_FuzzyTopK.py \
    --stage3_init merged \
    --use_sts False
```

### 3. Evaluate

```bash
# Full RAG benchmark — all encoder configs × prompts
python run_rag_benchmark.py \
    --method fuzzytopk_s1cascade_ragR2_a09 \
    --rag_exp R0

# Single model evaluation
python inference_disease_classification.py \
    --stage2_method fuzzytopk_s1cascade_ragR2_a09 \
    --rag_exp R0
```

### 4. Visualize Attention Maps

```bash
python gradcam_visualization.py
# Output → gradcam_outputs/*_comparison.png
```

### 5. Environment Check

```bash
python check_cuda.py  # Verify CUDA + GPU before training
```

> 📖 See [`Model/README.md`](Model/README.md) for the **full technical reference** — every script explained, all experiments documented, ablation tables.

---

## 📂 Repository Structure

```
HIKARI/
├── README.md                          ← You are here
├── logo/
│   └── HIKARI logo.png
└── Model/
    ├── README.md                      ← Full technical reference
    ├── Conference_Paper.tex           ← ITC-CSCC 2025 paper
    ├── summary.md                     ← Project summary (EN)
    ├── summary_Th.md                  ← Project summary (TH) + all experiment definitions
    │
    ├── 🚀 Training
    │   ├── train_two_stage_FuzzyTopK.py     ← MAIN: RAG-in-Training pipeline
    │   ├── train_three_stage_hybrid_topk.py ← M-series 3-stage ablation
    │   ├── train_qwen3_caption.py           ← Baseline caption SFT
    │   └── train_qwen3_thinking.py          ← Thinking-mode SFT variant
    │
    ├── 🔍 Inference & Evaluation
    │   ├── inference_disease_classification.py ← MAIN evaluation driver
    │   ├── run_rag_benchmark.py                ← Full benchmark sweep
    │   └── rag_retrieval.py                    ← HybridRAGRetriever (R0–R4)
    │
    ├── 🧠 Methods
    │   ├── SIB.py                       ← SIB-TinyLoRA adapter
    │   └── medical_token_importance.py  ← Selective Token Supervision (STS)
    │
    ├── 📈 Analysis
    │   ├── gradcam_visualization.py     ← LM Prefill Attention maps
    │   ├── plot_confusion_matrix.py     ← Confusion matrix plots
    │   └── EDA.ipynb                   ← Dataset EDA notebook
    │
    ├── 💾 Results
    │   ├── disease_classification_results/  ← Per-disease JSON results
    │   ├── gradcam_outputs/                 ← Attention comparison images
    │   └── stage3_ablation_results.json     ← Stage 3 ablation
    │
    └── ⚙️ requirements.txt
```

---

## 📄 Citation

```bibtex
@inproceedings{hikari2025,
  title     = {HIKARI: RAG-in-Training for Fine-Grained Skin Lesion Diagnosis
               with Vision-Language Models},
  booktitle = {Proceedings of ITC-CSCC 2025},
  year      = {2025}
}
```

---

## 📚 References

| Library / Resource | Role in HIKARI |
|:------------------|:---------------|
| [Unsloth](https://github.com/unslothai/unsloth) | Efficient 4-bit LoRA fine-tuning |
| [Qwen3-VL-8B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) | Vision-Language backbone |
| [SkinCAP Dataset](https://huggingface.co/datasets/joshuachou/SkinCAP) | 4,000 dermatology images + captions |
| [SigLIP-2](https://huggingface.co/google/siglip-2-base-patch16-512) | Image encoder for RAG training (R2) |
| [BGE-M3](https://huggingface.co/BAAI/bge-m3) | Text encoder for RAG training (R2) |
| [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) | Image encoder for RAG inference (R0) |
| [SIB-TinyLoRA](https://arxiv.org/abs/2410.10040) | Surprise-based token importance (STS) |

---

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=13&duration=4000&pause=500&color=AAAAAA&center=true&vCenter=true&width=650&lines=Validated+on+99-sample+locked+val+set+%E2%80%94+0+OOM+errors;RTX+5070+Ti+%C2%B7+Qwen3-VL-8B-Thinking+%C2%B7+SkinCAP+4%2C000+images;ITC-CSCC+2025+%C2%B7+RAG-in-Training+%C2%B7+Merged-Init+%C2%B7+STS" alt="footer" />

<br/>

```
✿ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✿
                     🌸 HIKARI Project 🌸
         Healthcare-oriented Intelligent Knowledge-Augmented
                  Retrieval and Inference system
✿ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✿
```

<sub>Made with 💗 and 🌸</sub>

</div>
