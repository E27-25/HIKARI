<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=32&duration=3000&pause=1000&color=00D9FF&center=true&vCenter=true&width=700&lines=HIKARI+%F0%9F%8C%9F;RAG-in-Training+%C3%97+VLM;3-Stage+Training+Pipeline;Hybrid+Retrieval-Augmented+Generation" alt="HIKARI Typing SVG" />

<br/>

[![Model](https://img.shields.io/badge/Backbone-Qwen3--VL--8B--Thinking-4B9EFF?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking)
[![Dataset](https://img.shields.io/badge/Dataset-SkinCAP_4K-FF8C00?style=for-the-badge&logo=databricks&logoColor=white)](https://huggingface.co/datasets/joshuachou/SkinCAP)
[![GPU](https://img.shields.io/badge/GPU-RTX_5070_Ti-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](.)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](.)
[![Framework](https://img.shields.io/badge/Framework-Unsloth+LoRA-A855F7?style=for-the-badge&logo=pytorch&logoColor=white)](https://github.com/unslothai/unsloth)

<br/>

> **HIKARI** — A RAG-in-Training pipeline for fine-grained skin lesion diagnosis
> using Qwen3-VL-8B-Thinking × SkinCAP × Hybrid Retrieval-Augmented Generation

</div>

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HIKARI Pipeline                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Qwen3-VL-8B (Base)                                                │
│         │                                                           │
│         ▼  Stage 1 — Group Classifier (4 groups, 88.68%)           │
│   ┌──────────────┐      weights ──────────────────────────┐        │
│   │ Group Clf.   │                                         │        │
│   └──────────────┘                                         │        │
│         │                                                   ▼       │
│         ▼  Stage 2 — Disease Classifier (10 classes)               │
│   ┌──────────────────────────────────────────┐                     │
│   │  [ref_img] Reference 1: psoriasis        │  ← RAG-in-Training  │
│   │  Description: Erythematous plaques…      │    K=1 ref/sample   │
│   │  [query_img] What skin disease?          │    R2 encoder (α=0.9)│
│   └──────────────────────────────────────────┘                     │
│         │                                                           │
│         ▼  Stage 3 — Caption Generator (BLEU-4: 29.33)             │
│   ┌──────────────┐  Merged-Init  ┌──────────────────────────┐      │
│   │ Disease Clf. │ ────────────▶ │ Caption Model            │      │
│   │  (frozen)    │               │ (fresh LoRA adapters)    │      │
│   └──────────────┘               └──────────────────────────┘      │
│                                                                     │
│   ──────────────── Inference ─────────────────────────────────     │
│   RAG Index (911 train imgs) → CLIP R0 retrieval → K=3 refs        │
│   → Prompt → Qwen3-VL-8B → Disease Name                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Attention Map Visualization

<div align="center">

*LM Prefill Attention — last input token → image patch positions*

| Melanocytic Nevi | Basal Cell Carcinoma |
|:----------------:|:--------------------:|
| ![Melanocytic Nevi](gradcam_outputs/melanocytic_nevi_2_comparison.png) | ![BCC](gradcam_outputs/basal_cell_carcinoma_3_comparison.png) |
| Network & border focus | Nodular lesion annotation |

> **Left:** Base Qwen3-VL-8B (unfocused) → **Right:** HIKARI (disease-specific focus)

</div>

---

## 🗂️ Project Structure

```
HIKARI/Model/
│
├── 🚀 Training
│   ├── train_two_stage_FuzzyTopK.py      # Main training: Single-Image FT + RAG-in-Training
│   ├── train_three_stage_hybrid_topk.py  # M-series 3-stage pipeline
│   ├── train_qwen3_caption.py            # Stage 3 caption training (baseline SFT)
│   ├── train_qwen3_thinking.py           # Qwen3 thinking-mode SFT variant
│   ├── train_two_stage.py                # Original two-stage (pre-FuzzyTopK)
│   └── train.py                          # Earliest prototype caption SFT
│
├── 🔍 Inference & Evaluation
│   ├── inference_disease_classification.py   # Main eval (all models × RAG × prompts)
│   ├── run_rag_benchmark.py                  # Full RAG benchmark runner
│   ├── rag_retrieval.py                      # HybridRAGRetriever (R0–R4 encoders)
│   ├── inference_classification.py           # Binary classification inference
│   ├── inference_classification_FuzzyTopK.py # FuzzyTopK classification inference
│   ├── inference_group_classification.py     # Stage 1 group classifier inference
│   └── inference_qwen3.py                    # Raw Qwen3-VL caption inference
│
├── 🧪 Experiments & Ablation
│   ├── run_stage3_experiments.py     # Stage 3 ablation (4 experiments)
│   └── run_rag_benchmark.py          # Full RAG benchmark runner
│
├── 📈 Analysis & Visualization
│   ├── gradcam_visualization.py      # LM Prefill Attention maps
│   ├── visualize_attention.py        # Token-level attention heatmaps
│   ├── visualize.py                  # General result visualization
│   ├── analyze_benchmark.py          # RAG benchmark result analysis
│   ├── plot_confusion_matrix.py      # Confusion matrix plots
│   ├── results_EDA.py                # Exploratory data analysis on results
│   └── EDA.ipynb                     # Dataset EDA notebook
│
├── 🧪 Testing & Utilities
│   ├── test_callback.py              # Training callback tests
│   ├── test_data_pipeline.py         # Dataset pipeline smoke tests
│   ├── test_inference_1sample.py     # Single-sample inference smoke test
│   ├── test_parallel_formatting.py   # Parallel prompt formatting tests
│   ├── fix_emoji.py                  # Post-process to strip unwanted emoji from outputs
│   └── check_cuda.py                 # CUDA/GPU environment check
│
├── 📄 Paper & Docs
│   ├── Conference_Paper.tex          # Paper (LaTeX)
│   ├── summary.md                    # Full project summary (EN)
│   ├── summary_Th.md                 # Full project summary (TH) + experiment definitions
│   ├── paper_draft.md                # Draft paper notes
│   ├── answer.md                     # Q&A for paper review
│   └── plan.md                       # Development roadmap
│
├── 💾 Data Splits & Precomputed
│   ├── split_info_3stage.json              # Locked 911/99 stratified split (3-stage)
│   ├── split_info_fuzzytopk.json           # Standalone FuzzyTopK split
│   ├── top10_precomputed.json              # Top-10 RAG retrievals (precomputed)
│   ├── top10_precomputed_pure_vision.json  # Pure vision RAG retrievals
│   ├── top10_stage3_precomputed.json       # Stage 3 RAG retrievals
│   ├── rag_index*.npz                      # FAISS-compatible RAG embedding indices
│   ├── stage1_train_predictions.json       # Stage 1 predictions on train set
│   ├── val_captions_for_symptoms.json      # 94 patient symptom descriptions (val set)
│   └── SkinCAP/                            # Raw dataset (CSV + XLSX)
│
└── ⚙️ Config
    └── requirements.txt
```

---

## 📖 Program Reference — All Scripts Defined

### 🚀 Training Scripts

| Script | Purpose |
|--------|---------|
| `train.py` | **Prototype** — earliest SFT caption trainer for Qwen2-VL. Loads SkinCAP CSV, builds image+caption pairs, trains with Unsloth SFTTrainer. Single stage only. |
| `train_two_stage.py` | **Two-stage baseline** — Stage 1: classification, Stage 2: caption. Carries Stage 1 LoRA weights into Stage 2. No fuzzy matching, no top-K filtering. |
| `train_two_stage_FuzzyTopK.py` | **Main training script.** Adds fuzzy disease-name consolidation (`thefuzz`), top-K class filtering, stratified splits, sqrt oversampling, and the RAG-in-Training loop. Controls `--rag_k_train`, `--rag_exp`, `--alpha`, `--stage3_init` flags. Produces the `fuzzytopk` and `fuzzytopk_s1cascade_ragR2_a09` model families. |
| `train_three_stage_hybrid_topk.py` | **M-series 3-stage pipeline.** Stage 1: group classifier (4 or 3 groups). Stage 2: disease classifier per group. Stage 3: caption from Stage 2 checkpoint. Supports `GROUP_MODE` = `"4group"` / `"3group"` and `TOP_N` = 10/15. Produces `skincap_3stage_*` model families. |
| `train_qwen3_caption.py` | **Baseline SFT caption** using Qwen3-VL-8B-Thinking directly. Loads raw SkinCAP `caption_zh_polish_en` column; no classification stage. Used to establish the caption-only baseline. |
| `train_qwen3_thinking.py` | **Thinking-mode SFT variant.** Enables Qwen3's extended reasoning chain (`<think>…</think>`) during fine-tuning. Experimental; explores whether explicit reasoning improves diagnosis. |

---

### 🔍 Inference & Evaluation Scripts

| Script | Purpose |
|--------|---------|
| `inference.py` | **Basic caption inference.** Loads a trained model and runs caption generation on a single image or small batch. Used for quick sanity-checks. |
| `inference_classification.py` | **Binary/multi-class classification inference.** Given a model checkpoint, runs the classification prompt on a validation set and reports accuracy. |
| `inference_classification_FuzzyTopK.py` | **FuzzyTopK classification inference.** Same as above but uses the fuzzy label mapping to align predicted text to canonical disease names before scoring. |
| `inference_disease_classification.py` | **Main evaluation driver.** Evaluates any model variant (Stage 1 group → Stage 2 disease) across all RAG encoder configs (R0–R4) and prompt templates (P0–P2). Produces per-disease sensitivity, PPV, and overall accuracy. Writes `evaluation_results.json`. |
| `inference_group_classification.py` | **Stage 1 group classifier evaluation.** Runs only the group-level (4-class) head, reports per-group accuracy and confusion. Used to diagnose cascade bottlenecks. |
| `inference_qwen3.py` | **Raw Qwen3-VL caption inference.** Loads any Qwen3-VL checkpoint (merged or LoRA) and generates captions; computes BLEU-1/2/4 and ROUGE-L against ground-truth. |
| `run_rag_benchmark.py` | **Full RAG benchmark runner.** Sweeps over `(model, rag_exp, prompt_style, k)` combinations, calls `inference_disease_classification.py` for each, and aggregates results into a benchmark table. Produces `*.log` files and `stage3_ablation_results.json`. |

---

### 📡 RAG Retrieval

| Script | Purpose |
|--------|---------|
| `rag_retrieval.py` | **HybridRAGRetriever — core retrieval engine.** Supports 5 encoder configurations: **R0** (CLIP image-only, used at inference), **R1** (CLIP + ClinicalBERT), **R2** (SigLIP-2 + BGE-M3, used during RAG-in-Training), **R3** (Jina-CLIP-v2 + MedCPT), **R4** (Nomic unified vision+text). Implements two retrieval strategies: **Strategy A** (cross-modal: image query vs text index in shared space) and **Strategy B** (separate image and text spaces, alpha-weighted fusion). Builds and saves `.npz` index files; supports `top_k` retrieval with similarity scores. |

---

### 🧪 Experiments & Ablation

| Script | Purpose |
|--------|---------|
| `run_stage3_experiments.py` | **Stage 3 ablation orchestrator.** Runs 4 sequential experiments via subprocess comparing checkpoint-init vs merged-init for Stage 3 caption training. Each trains a Stage 3 caption model and evaluates BLEU/ROUGE. Supports `--skip N` and `--only N M` flags. |

---

### 📈 Analysis & Visualization

| Script | Purpose |
|--------|---------|
| `gradcam_visualization.py` | **LM Prefill Attention maps.** Extracts last-token → image-patch attention weights from Qwen3-VL's attention layers during a forward pass. Overlays a heatmap onto the original image. Generates side-by-side comparison (base model vs HIKARI) saved to `gradcam_outputs/`. |
| `visualize_attention.py` | **Token-level attention heatmaps.** Plots per-layer, per-head attention matrices for a given input. Used for qualitative inspection of which tokens the model attends to for disease-relevant features. |
| `visualize.py` | **General result visualization.** Bar charts and line plots of accuracy/BLEU across model variants and RAG configs. Reads `evaluation_results.json`. |
| `analyze_benchmark.py` | **RAG benchmark result analysis.** Parses benchmark log files and JSON outputs, computes rank tables, statistical summaries, and per-disease breakdowns. Outputs markdown-formatted tables. |
| `plot_confusion_matrix.py` | **Confusion matrix plots.** Reads model predictions and ground-truth from `evaluation_results.json`, plots normalized confusion matrices (per-disease and aggregated) using matplotlib/seaborn. |
| `results_EDA.py` | **Exploratory analysis of evaluation results.** Distribution of confidence scores, error analysis (which diseases are confused with which), sample-level inspection of high-error cases. |
| `EDA.ipynb` | **SkinCAP dataset EDA notebook.** Class distribution, image quality statistics, caption length analysis, train/val split verification, and label noise investigation. |

---

### 🧪 Testing & Utilities

| Script | Purpose |
|--------|---------|
| `test_callback.py` | **Training callback tests.** Verifies that custom `SFTTrainer` callbacks (checkpoint save, logging, RAG index refresh) fire at the correct training steps without errors. |
| `test_data_pipeline.py` | **Dataset pipeline smoke tests.** Loads a small subset of SkinCAP, runs the full fuzzy-matching and formatting pipeline, and asserts expected output shapes and label distributions. |
| `test_inference_1sample.py` | **Single-sample inference smoke test.** Loads a merged model checkpoint and runs inference on one image; verifies the output is a valid disease name string, not a crash. |
| `test_parallel_formatting.py` | **Parallel prompt formatting tests.** Tests multi-image (query + reference) prompt construction for RAG inputs; verifies correct token ordering and image placeholder positions under Qwen3-VL's chat template. |
| `fix_emoji.py` | **Post-processing utility.** Strips or replaces unwanted emoji characters from generated captions/outputs that interfere with BLEU scoring or downstream parsing. |
| `check_cuda.py` | **CUDA/GPU environment check.** Prints CUDA version, available GPU count, VRAM per device, and PyTorch build info. Run before training to confirm the environment is correctly configured. |

---

## 🧪 Experiments at a Glance

<div align="center">

| Model | Description | RAG Train | RAG Infer |
|-------|------------|:---------:|:---------:|
| `fuzzytopk` | Single-Image Fine-Tune | — | — |
| `M1` | 2-Stage Cascade FT | — | R0 |
| `fuzzytopk_s1cascade` | Cascaded FT (α=0.9) | — | R2 |
| `fuzzytopk_s1cascade` | Cascaded FT (α=0.5) | — | R2 |
| **`fuzzytopk_s1cascade_ragR2_a09`** | **RAG-in-Training (Ours)** | **R2 K=1** | **R0** |

</div>

### RAG Encoder Configurations (R0–R4)

| ID | Image Encoder | Text Encoder | Best Use |
|----|--------------|--------------|----------|
| **R0** | `openai/clip-vit-base-patch32` | — | ✅ Best at **inference** for HIKARI |
| **R1** | CLIP | `medicalai/ClinicalBERT` | Generic clinical text |
| **R2** | `google/siglip-2-base-patch16-512` | `BAAI/bge-m3` | ✅ Used during **training** |
| **R3** | `jinaai/jina-clip-v2` | `ncbi/MedCPT-Query-Encoder` | Best with α=0.7 |
| **R4** | `nomic-ai/nomic-embed-vision-v1.5` | `nomic-ai/nomic-embed-text-v1.5` | Cross-modal unified |

---

## 🚀 Quick Start

### 1. Install

```bash
git clone https://github.com/E27-25/HIKARI.git
cd HIKARI/Model
pip install -r requirements.txt
huggingface-cli login
```

### 2. Train — RAG-in-Training (HIKARI)

```bash
# Stage 1 + 2: RAG-in-Training (main contribution)
python train_two_stage_FuzzyTopK.py \
    --start_from_stage1 \
    --rag_k_train 1 \
    --rag_exp R2 \
    --alpha 0.9

# Stage 3: Caption generation (Merged-Init)
python train_two_stage_FuzzyTopK.py \
    --stage3_init merged \
    --use_sts False
```

### 3. Evaluate

```bash
# Full RAG benchmark (all models × encoders × prompts)
python run_rag_benchmark.py \
    --method fuzzytopk_s1cascade_ragR2_a09 \
    --rag_exp R0

# Single inference
python inference_disease_classification.py \
    --stage2_method fuzzytopk_s1cascade_ragR2_a09 \
    --rag_exp R0
```

### 4. Attention Visualization

```bash
python gradcam_visualization.py  # Generates comparison images in gradcam_outputs/
```

### 5. Environment Check

```bash
python check_cuda.py  # Verify CUDA and GPU setup before training
```

---

## 🔑 Key Findings

<table>
<tr><td>

**💡 RAG-in-Training closes the train/inference gap**
Training with K=1 reference image per sample teaches the model to use retrieval context — no architectural changes needed.

</td></tr>
<tr><td>

**💡 Encoder-agnostic generalization**
HIKARI trained with SigLIP+BGE-M3 (R2) but performs best with CLIP (R0) at inference (+3.03 pp) — the model learns the *concept* of reference-guided diagnosis, not encoder-specific features.

</td></tr>
<tr><td>

**💡 Merged-Init prevents catastrophic interference**
Merging LoRA into base weights before Stage 3 → BLEU-4: 9.82 → **29.33 (3×)**. Never fine-tune existing adapters on a different task.

</td></tr>
<tr><td>

**💡 Group cascade hurts more than it helps**
3-stage M-series (M1 oracle: 66%) still underperforms simple 2-stage fuzzytopk (74%) — cascade penalty from mismatched weight initialization outweighs group context benefit.

</td></tr>
</table>

---

## 📋 Per-Disease Results (HIKARI Best Config: R0-P0)

<div align="center">

| Disease | Sensitivity | n | PPV |
|---------|:-----------:|:---:|:---:|
| 🟢 Psoriasis | **100.0%** | 13 | 92.9% |
| 🟢 Melanocytic Nevi | **100.0%** | 12 | 100.0% |
| 🟢 SCCIS | **100.0%** | 12 | 100.0% |
| 🟢 Basal Cell Carcinoma | **100.0%** | 13 | 100.0% |
| 🟢 Acne Vulgaris | **100.0%** | 8 | 88.9% |
| 🟡 Lichen Planus | 88.9% | 9 | 88.9% |
| 🟡 Scleroderma | 87.5% | 8 | 77.8% |
| 🟡 Photodermatoses | 75.0% | 8 | 66.7% |
| 🔴 Lupus Erythematosus | 55.6% | 9 | 55.6% |
| 🔴 Sarcoidosis | 14.3% | 7 | 33.3% |

*🔴 Sarcoidosis collapse is a known limitation — over-reliance on reference agreement during training*

</div>

---

## ⚙️ Hardware & Training Config

| Parameter | Value |
|-----------|-------|
| Backbone | Qwen3-VL-8B-Thinking |
| Quantization | 4-bit NF4 (Unsloth) |
| LoRA rank / alpha | 16 / 32 |
| Effective batch size | 8 (2×GPU + grad_accum=4) |
| Optimizer | AdamW 8-bit paged |
| GPU | NVIDIA RTX 5070 Ti (15.92 GB VRAM) |
| Training time | ~1h 44min (1,314 steps) |
| Image size | 672×672 thumbnail (LANCZOS) |

---

## 📚 References

- [Unsloth](https://github.com/unslothai/unsloth) — Efficient LLM fine-tuning
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking) — Vision-Language backbone
- [SkinCAP Dataset](https://huggingface.co/datasets/joshuachou/SkinCAP) — 4,000 dermatology images
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) — Multilingual text embeddings
- [SigLIP-2](https://huggingface.co/google/siglip-2-base-patch16-512) — Image encoder

---

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=14&duration=4000&pause=500&color=888888&center=true&vCenter=true&width=700&lines=Qwen3-VL-8B-Thinking+%C2%B7+Unsloth+%C2%B7+SkinCAP;SigLIP-2+%2B+BGE-M3+(train)+%C2%B7+CLIP+ViT-B%2F32+(inference);RAG-in-Training+%C2%B7+Merged-Init+%C2%B7+3-Stage+Pipeline;LoRA+rank%3D16+%C2%B7+4-bit+NF4+%C2%B7+FuzzyTopK+%C2%B7+RTX+5070+Ti" alt="footer" />

</div>
