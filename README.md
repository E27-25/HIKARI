<!-- ✿ HIKARI README ✿ -->

<div align="center">

<img src="logo/HIKARI logo.png" alt="HIKARI Logo" width="100%"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=28&duration=3000&pause=800&color=FF9EBC&center=true&vCenter=true&width=800&lines=HIKARI+%F0%9F%8C%B8;RAG-in-Training+Pipeline;Vision-Language+Model+Fine-Tuning;3-Stage+Training+Architecture;Hybrid+Retrieval-Augmented+Generation;Qwen3-VL-8B+%C3%97+LoRA+%C3%97+SkinCAP" alt="Typing SVG" />

<br/>

<!-- Badges — tech stack only -->
[![Model](https://img.shields.io/badge/Backbone-Qwen3--VL--8B--Thinking-4B9EFF?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking)
[![Dataset](https://img.shields.io/badge/Dataset-SkinCAP_4K-FF8C00?style=for-the-badge&logo=databricks&logoColor=white)](https://huggingface.co/datasets/joshuachou/SkinCAP)
[![GPU](https://img.shields.io/badge/GPU-RTX_5070_Ti-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](.)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](.)
[![Framework](https://img.shields.io/badge/Framework-Unsloth+LoRA-A855F7?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/unslothai/unsloth)

<br/>

> **HIKARI** *(光・ヒカリ)* — Healthcare-oriented Intelligent Knowledge-Augmented Retrieval and Inference system
>
> A RAG-in-Training pipeline that fine-tunes **Qwen3-VL-8B-Thinking** on dermatology images,
> injecting retrieved reference cases *during training* so the model learns to reason with visual similarity evidence.

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

HIKARI introduces **RAG-in-Training** — instead of using retrieval only at inference time, reference images are retrieved and injected into *every training sample*, so the model learns to reason about visual similarity evidence from the start. At inference, even switching to a different (simpler) retrieval encoder still generalizes well, demonstrating true encoder-agnostic behavior.

---

## ⚙️ Techniques Used

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=18&duration=2500&pause=600&color=4ADE80&center=true&vCenter=true&width=700&lines=FuzzyTopK+Label+Consolidation;Stratified+Split+%C2%B7+Sqrt+Oversampling;SigLIP-2+%2B+BGE-M3+%E2%86%92+RAG+Training+Index;CLIP+ViT-B%2F32+%E2%86%92+Inference+Retrieval;LoRA+rank%3D16+%C2%B7+4-bit+NF4+%C2%B7+AdamW+8-bit;Merged-Init+%E2%86%92+Stage+3+Caption+Training" alt="Techniques" />

</div>

| Component | Technique | Detail |
|:----------|:----------|:-------|
| 🧠 Backbone | Qwen3-VL-8B-Thinking | Vision-Language Model with extended reasoning chain |
| 📦 Dataset | SkinCAP | 4,000 dermatology images · 10 disease classes |
| 🏷️ Label Prep | FuzzyTopK | `thefuzz` near-duplicate label consolidation → top-K class filtering |
| 📊 Sampling | Stratified + Sqrt Oversample | Class-balanced 911 train / 99 val split (locked) |
| 🔍 RAG Training | HybridRAGRetriever (R2) | SigLIP-2 + BGE-M3, α=0.9, K=1 reference per training sample |
| 🔍 RAG Inference | HybridRAGRetriever (R0) | CLIP ViT-B/32, K=3 references |
| 🎛️ Fine-Tuning | LoRA | rank=16, alpha=32 · 4-bit NF4 quantization (Unsloth) |
| ⚡ Optimizer | AdamW 8-bit paged | Memory-efficient optimizer for large VLM training |
| 🔗 Stage Transfer | Merged-Init | `merge_and_unload()` before Stage 3 — prevents catastrophic interference |
| 🖼️ Image | 672×672 LANCZOS | Thumbnail resize before tokenization |

---

## 🏗️ HIKARI Pipeline

```
╔══════════════════════════════════════════════════════════════════════╗
║  STAGE 1 — GROUP CLASSIFICATION                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Skin Image ──► Qwen3-VL-8B (LoRA) ──► 4 Disease Groups             ║
║                                                                      ║
║  ┌─ Inflammatory  (Psoriasis · Lichen · Lupus · Photo · Sclero)     ║
║  ├─ Benign Tumor  (Melanocytic Nevi · SCCIS)                        ║
║  ├─ Malignant     (Basal Cell Carcinoma · Sarcoidosis)              ║
║  └─ Acne          (Acne Vulgaris)                                   ║
║                                                                      ║
║  Trained weights → transferred to initialize Stage 2                 ║
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
║       ▼  Qwen3-VL-8B → Disease Name  (cross-entropy on label token) ║
║                                                                      ║
║  At inference: swap to CLIP R0 encoder → encoder-agnostic design    ║
╚══════════════════════════════════════════════════════════════════════╝
                              │
                              ▼
╔══════════════════════════════════════════════════════════════════════╗
║  STAGE 3 — CLINICAL CAPTION GENERATION  (Merged-Init)               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Stage 2 LoRA ──► merge_and_unload() ──► Merged base weights        ║
║                    (Merged-Init trick)           │                   ║
║                                                  ▼                   ║
║  Image + Prompt ──────────────────────► Fresh LoRA ──► Caption      ║
║  "Describe condition & recommend treatment"                          ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 🔑 Key Technical Insights

| # | Insight |
|:-:|:--------|
| 1 | 🌸 **RAG-in-Training** — Injecting K=1 retrieved reference per training sample teaches the model to leverage visual similarity context without any architectural changes to the VLM. |
| 2 | ✿ **Encoder-agnostic generalization** — Training with SigLIP-2+BGE-M3 (R2) but switching to CLIP (R0) at inference still generalizes well — the model learns the *concept* of reference-guided reasoning, not encoder-specific features. |
| 3 | 🌸 **Merged-Init** — Merging Stage 2 LoRA adapters into base weights via `merge_and_unload()` before Stage 3 caption training prevents catastrophic task interference between classification and generation objectives. |
| 4 | ✿ **Cascade penalty** — Stacking Stage 1 (group) → Stage 2 (disease) with weight transfer introduces initialization mismatch overhead that can outweigh the benefit of group-level context injection. |

---

## 🔭 Attention Visualization

> **LM Prefill Attention** — extracts attention from the last input token to `<|image_pad|>` positions, revealing where the model focuses during inference.

<table align="center">
<tr>
<th align="center" width="50%">🔵 Base Qwen3-VL-8B</th>
<th align="center" width="50%">🌸 HIKARI (fine-tuned)</th>
</tr>
<tr>
<td align="center" colspan="2">
<img src="Model/gradcam_outputs/melanocytic_nevi_2_comparison.png" width="700" alt="Melanocytic Nevi — Base vs HIKARI"/>
<br/><i>Melanocytic Nevi — attention shifts toward lesion network & border structure</i>
</td>
</tr>
<tr>
<td align="center" colspan="2">
<img src="Model/gradcam_outputs/basal_cell_carcinoma_3_comparison.png" width="700" alt="BCC — Base vs HIKARI"/>
<br/><i>Basal Cell Carcinoma — attention localizes to nodular lesion region</i>
</td>
</tr>
</table>

---

## 🔍 RAG Encoder Reference (R0–R4)

| ID | Image Encoder | Text Encoder | α | Role in HIKARI |
|:--:|:-------------|:-------------|:-:|:---------------|
| **R0** | `openai/clip-vit-base-patch32` | — | — | Used at **inference** |
| R1 | CLIP | `medicalai/ClinicalBERT` | 0.5 | Clinical text variant |
| **R2** | `google/siglip-2-base-patch16-512` | `BAAI/bge-m3` | 0.9 | Used during **training** |
| R3 | `jinaai/jina-clip-v2` | `ncbi/MedCPT-Query-Encoder` | 0.7 | Medical text-visual fusion |
| R4 | `nomic-ai/nomic-embed-vision-v1.5` | `nomic-ai/nomic-embed-text-v1.5` | 0.5 | Cross-modal unified space |

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

# Stage 3: Clinical caption generation (Merged-Init)
python train_two_stage_FuzzyTopK.py \
    --stage3_init merged
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

> 📖 See [`Model/README.md`](Model/README.md) for the **full technical reference** — every script explained, all experiments documented.

---

## 📂 Repository Structure

```
HIKARI/
├── README.md                          ← You are here
├── logo/
│   └── HIKARI logo.png
└── Model/
    ├── README.md                      ← Full technical reference (all scripts defined)
    ├── Conference_Paper.tex           ← Paper (LaTeX)
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

## 📚 References

| Library / Resource | Role in HIKARI |
|:------------------|:---------------|
| [Unsloth](https://github.com/unslothai/unsloth) | Efficient 4-bit LoRA fine-tuning |
| [Qwen3-VL-8B-Thinking](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) | Vision-Language backbone |
| [SkinCAP Dataset](https://huggingface.co/datasets/joshuachou/SkinCAP) | 4,000 dermatology images + captions |
| [SigLIP-2](https://huggingface.co/google/siglip-2-base-patch16-512) | Image encoder for RAG training (R2) |
| [BGE-M3](https://huggingface.co/BAAI/bge-m3) | Text encoder for RAG training (R2) |
| [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) | Image encoder for RAG inference (R0) |

---

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=13&duration=4000&pause=500&color=AAAAAA&center=true&vCenter=true&width=700&lines=Qwen3-VL-8B-Thinking+%C2%B7+Unsloth+%C2%B7+SkinCAP;SigLIP-2+%2B+BGE-M3+(train)+%C2%B7+CLIP+ViT-B%2F32+(inference);RAG-in-Training+%C2%B7+Merged-Init+%C2%B7+3-Stage+Pipeline;LoRA+rank%3D16+%C2%B7+4-bit+NF4+%C2%B7+FuzzyTopK;HybridRAGRetriever+%C2%B7+RTX+5070+Ti" alt="footer" />

<hr/>

<p>🌸 <b>HIKARI Project &nbsp;·&nbsp; 光（ヒカリ）</b> 🌸</p>
<p><i>Healthcare-oriented Intelligent Knowledge-Augmented Retrieval and Inference system</i></p>
<sub>Made with 💗 and 🌸</sub>

<hr/>

</div>
