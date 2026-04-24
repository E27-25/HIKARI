<!-- ✿ HIKARI Phase 2 Roadmap ✿ -->

<div align="center">

<img src="logo/HIKARI logo.png" alt="HIKARI Logo" width="100%"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=26&duration=3000&pause=800&color=FF9EBC&center=true&vCenter=true&width=800&lines=HIKARI+Phase+2+%F0%9F%8C%B8;Multi-Architecture+Benchmark;Dynamic+RAG-k+%C3%97+Hard+Negative+Mining;Segmentation+%E2%86%92+Classify+Pipeline;Explainability+%2B+Uncertainty+Scores;Clinical-Grade+Evaluation" alt="Typing SVG" />

<br/>

[![Phase](https://img.shields.io/badge/Phase_1-Complete_✓-4ADE80?style=for-the-badge&logo=checkmarx&logoColor=white)](.)
[![Phase](https://img.shields.io/badge/Phase_2-In_Planning-FF9EBC?style=for-the-badge&logo=rocket&logoColor=white)](.)
[![Models](https://img.shields.io/badge/Models_to_Test-10+-4B9EFF?style=for-the-badge&logo=huggingface&logoColor=white)](.)
[![Runs](https://img.shields.io/badge/Training_Runs-~50_Automated-A855F7?style=for-the-badge&logo=pytorch&logoColor=white)](.)
[![Tracking](https://img.shields.io/badge/Tracking-Weights_&_Biases-FFBE00?style=for-the-badge&logo=weightsandbiases&logoColor=white)](.)

<br/>

> **Phase 1** proved that **RAG-in-Training** and **Merged-Init** work — **85.86% accuracy**, 14 published models.
>
> **Phase 2** asks: *does it generalize? how far can we push it?*

</div>

---

## 🗺️ Six Development Axes

```
╔══════════════════════════════════════════════════════════════════════╗
║                        HIKARI  Phase 2                               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   🔭 Model Axis      ──  Does RAG-in-Training work on every VLM?    ║
║   🔬 Technique Axis  ──  Smarter training: Dynamic-k, Hard Negative  ║
║   🧩 Pipeline Axis   ──  Segmentation-first preprocessing           ║
║   🔍 XAI Axis        ──  Confidence scores + Attention atlas         ║
║   📊 Eval Axis       ──  Clinical-grade multi-dimension evaluation   ║
║   🏥 App Axis        ──  REST API + Web UI for real-world use       ║
║                                                                      ║
║   All axes → one automated pipeline → train · eval · upload         ║
╚══════════════════════════════════════════════════════════════════════╝
```

| Axis | Status | Key Output | Paper? |
|:-----|:------:|:-----------|:------:|
| 🔭 Model Comparison | ✅ Ready | Multi-architecture benchmark table | ✅ Yes |
| 🔬 Technique | 🟡 Choosing option | Dynamic RAG-k algorithm + results | ✅ Yes |
| 🧩 Pipeline | 🟡 Choosing option | Segmentation-first ablation study | ✅ Yes |
| 🔍 Explainability | ✅ Ready | Confidence score + XAI dashboard | ⚠️ Supporting |
| 📊 Evaluation | ✅ Ready | Clinical-grade full eval report | ⚠️ Supporting |
| 🏥 Application | 🔵 After research | REST API + Web UI | 🔵 Demo |

---

## 🔭 Axis 1 — Model Comparison

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=16&duration=2500&pause=600&color=4B9EFF&center=true&vCenter=true&width=700&lines=Does+RAG-in-Training+work+on+every+VLM%3F;Qwen2.5-VL-3B+%C2%B7+Gemma3-4B+%C2%B7+InternVL2.5-4B;Can+4B+%2B+RAG+beat+8B+without+RAG%3F" alt="Model Axis" />

</div>

The core claim of Phase 1 is that **training methodology** matters more than model size.  
Phase 2 verifies this by running the same pipeline across multiple model families.

### Priority 1 — Try First

| Model | Params | Family | Why |
|:------|:------:|:------:|:----|
| `Qwen2.5-VL-3B-Instruct` | 3B | Qwen | Same family as Phase 1 — easiest migration |
| `Gemma3-4B-IT` (vision) | 4B | Google | Strong baseline, popular in research |
| `InternVL2.5-4B` | 4B | InternVL | Known strong medical vision encoder |

### Priority 2 — If Time Allows

| Model | Params | Family | Why |
|:------|:------:|:------:|:----|
| `Phi-4-multimodal` | 5B | Microsoft | Efficient, edge-friendly |
| `Qwen2.5-VL-7B-Instruct` | 7B | Qwen | Direct comparison to Phase 1 (8B) |
| `InternVL2.5-8B` | 8B | InternVL | Size-matched comparison |

### 3 Experiments Per Model

```
1. Single-Image FT      ←  baseline  (no cascade, no RAG)
2. Cascaded FT          ←  intermediate
3. RAG-in-Training ⭐   ←  key claim — does it generalize?
```

### Expected Benchmark Table

```
Model                | Params | Baseline | +RAG    | BLEU-4 | ms/img
---------------------|--------|----------|---------|--------|--------
Gemma3-4B            |   4B   |  ~72%?   |  ~80%?  |  ~22?  |  ~350
InternVL2.5-4B       |   4B   |  ~75%?   |  ~82%?  |  ~24?  |  ~400
Qwen2.5-VL-3B        |   3B   |  ~68%?   |  ~77%?  |  ~20?  |  ~300
Qwen2.5-VL-7B        |   7B   |  ~80%?   |  ~84%?  |  ~27?  |  ~500
HIKARI-Sirius (8B) ⭐ |   8B   | 79.80%   | 85.86%  | 29.33  |   584  ← Phase 1 baseline
```

---

## 🔬 Axis 2 — Technique

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=16&duration=2500&pause=600&color=B07CFF&center=true&vCenter=true&width=700&lines=Same+model%2C+same+pipeline+—+smarter+training;Dynamic+RAG-k+%C2%B7+Hard+Negative+Mining;Self-Training+%C2%B7+Multi-Reference+Fusion" alt="Technique Axis" />

</div>

> **Status: Options under consideration — choose 1–2**

### ⭐ Option A — Dynamic RAG-k *(Recommended)*

Phase 1 used fixed `k=1`. Phase 2 lets the model decide how many references it needs.

```
Easy image  →  k=0  (confident — no reference needed)
Medium      →  k=1  (Phase 1 default)
Hard image  →  k=3  (uncertain — needs more visual context)

Decision: entropy of prediction distribution
High entropy = uncertain = request more references
```

### Option B — Hard Negative Mining

```
Phase 1 RAG: retrieve the closest image (same disease)
Phase 2 add: also inject 1 "hard negative"
             — visually similar, but a DIFFERENT disease

Goal: teach subtle boundary discrimination between look-alike diseases
```

### Option C — Multi-Reference Fusion

```
Phase 1: k=1 → 1 reference in context window
Phase 2: k=3 → 3 references, model attends differently to each

Research question: does k=3 meaningfully outperform k=1?
Hypothesis: diminishing returns after k=2
```

### Option D — Self-Training / Pseudo-Label

```
1. HIKARI-Sirius predicts unlabeled skin images
2. Keep high-confidence predictions (conf > 0.90)
3. Add as training data → retrain
4. Expand dataset without manual annotation cost
```

---

## 🧩 Axis 3 — Pipeline

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=16&duration=2500&pause=600&color=7EB8FF&center=true&vCenter=true&width=700&lines=Add+preprocessing+steps+around+the+model;SAM2+Segmentation+%E2%86%92+Lesion+Crop+%E2%86%92+HIKARI;Multi-lesion+Detection+%C2%B7+Quality+Filter" alt="Pipeline Axis" />

</div>

> **Status: Options under consideration — choose 1**

### ⭐ Option A — Segmentation → Classify *(Recommended)*

```
╔══════════════════════════════════════════════════════════╗
║  CURRENT PIPELINE                                        ║
╠══════════════════════════════════════════════════════════╣
║  [Full Image] ──────────────────────► HIKARI ► Diagnosis ║
╚══════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════╗
║  PHASE 2 PIPELINE                                        ║
╠══════════════════════════════════════════════════════════╣
║  [Full Image] ► SAM2 ► [Lesion Crop] ► HIKARI ► Diagnosis║
╚══════════════════════════════════════════════════════════╝
```

**Research question:** Does isolating the lesion before classification help —  
or does HIKARI already attend to the right region? (GradCAM suggests it does.)

### Option B — Multi-lesion Detection

```
Current: 1 image → 1 diagnosis  (assumes single lesion)
Phase 2: detect all lesions → classify each independently
Output:  { "lesion_A": "SCCIS", "lesion_B": "Melanoma" }
```

### Option C — Quality Filter + Preprocessing

```
Before inference:
├── Blur detection      → reject images below sharpness threshold
├── Skin region crop    → remove non-skin background
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

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=16&duration=2500&pause=600&color=5EEAD4&center=true&vCenter=true&width=700&lines=Why+did+the+model+decide+this%3F;GradCAM%2B%2B+%C2%B7+Attention+Rollout+%C2%B7+Uncertainty;Per-class+Activation+Atlas+for+23+diseases" alt="XAI Axis" />

</div>

Phase 1 implemented GradCAM. Phase 2 goes deeper toward clinical trustworthiness.

| Level | Method | Output |
|:-----:|:-------|:-------|
| 1 | GradCAM++ · Attention Rollout · Token saliency | Sharper heatmaps per prediction |
| 2 | Temperature Scaling + MC Dropout | Confidence % with every prediction |
| 3 | Patch masking → re-predict | Counterfactual critical region |
| 4 | Average GradCAM across all samples per disease | Per-class activation atlas (23 diseases) |

### Uncertainty Score Output

```
╔══════════════════════════════════════════╗
║  HIGH CONFIDENCE                         ║
║  Diagnosis:   SCCIS                      ║
║  Confidence:  0.87  ██████████░░░        ║
║  Status:      Reliable prediction ✓      ║
╠══════════════════════════════════════════╣
║  LOW CONFIDENCE                          ║
║  Diagnosis:   Melanoma                   ║
║  Confidence:  0.51  █████░░░░░░░░        ║
║  Status:      ⚠️ Recommend dermatologist  ║
╚══════════════════════════════════════════╝
```

---

## 📊 Axis 5 — Evaluation

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=16&duration=2500&pause=600&color=FCD34D&center=true&vCenter=true&width=700&lines=One+accuracy+number+is+not+enough;Skin+tone+bias+%C2%B7+Calibration+%C2%B7+OOD+Robustness;Clinical+Agreement+vs+Dermatologist" alt="Eval Axis" />

</div>

Phase 1 measured: accuracy, BLEU-4, BERTScore, Disease Correctness, speed.  
Phase 2 measures everything that matters **clinically**.

| Dimension | What | Method |
|:----------|:-----|:-------|
| **Subgroup** | Accuracy by skin tone (Fitzpatrick I–VI), lesion size, body location | Disaggregated evaluation |
| **Calibration** | Does 90% confidence = 90% accuracy? | Reliability diagram · ECE |
| **OOD Robustness** | Performance on HAM10000, smartphone photos, artifact images | Distribution shift test |
| **Clinical Agreement** | Accuracy vs real dermatologist | Cohen's Kappa · Sensitivity/Specificity |

### Caption Evaluation — Phase 1 vs Phase 2

| Metric | Phase 1 | Phase 2 |
|:-------|:-------:|:-------:|
| BLEU-4 | ✅ | ✅ |
| BERTScore | ✅ | ✅ |
| Disease Correctness | ✅ | ✅ |
| ROUGE-L | ❌ | ✅ |
| CIDEr | ❌ | ✅ |
| Clinical Relevance (GPT-4o judge) | ❌ | ✅ |
| Human Evaluation | ❌ | ✅ |

---

## 🏥 Axis 6 — Application

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=16&duration=2500&pause=600&color=FB923C&center=true&vCenter=true&width=700&lines=Make+HIKARI+usable+without+Python;REST+API+%C2%B7+Web+UI+%C2%B7+Confidence+threshold;Upload+image+%E2%86%92+Diagnosis+%2B+Heatmap+instantly" alt="App Axis" />

</div>

### REST API

```python
POST /api/diagnose
Content-Type: multipart/form-data

{ "image": <file>, "mode": "disease" | "caption" | "full" }

# Response:
{
  "disease":    "SCCIS",
  "confidence": 0.87,
  "caption":    "Raised erythematous lesion with irregular border...",
  "heatmap":    "<base64 png>",
  "latency_ms": 584
}
```

### Web UI Mockup

```
╔══════════════════════════════════════════════════════╗
║  🌸 HIKARI Skin Disease AI                           ║
╠══════════════════════════════════════════════════════╣
║  [ Upload Image ]  or  [ Take Photo ]                ║
║                                                      ║
║  ┌────────────┐   ┌────────────┐                     ║
║  │  Original  │   │  GradCAM   │                     ║
║  │   Image    │   │  Heatmap   │                     ║
║  └────────────┘   └────────────┘                     ║
║                                                      ║
║  Diagnosis:  SCCIS          Confidence: 87% ████░    ║
║  Caption:    Raised erythematous lesion...            ║
║                                                      ║
║  ⚠️ For clinical reference only.                     ║
╚══════════════════════════════════════════════════════╝
```

---

## ⚙️ Automation Infrastructure

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=16&duration=2500&pause=600&color=4ADE80&center=true&vCenter=true&width=700&lines=One+command+—+full+pipeline;train+%E2%86%92+eval+%E2%86%92+benchmark+%E2%86%92+report+%E2%86%92+upload;W%26B+logging+%C2%B7+Auto-upload+if+beats+best" alt="Automation" />

</div>

```bash
python run_all.py --model google/gemma-3-4b-it --wandb --upload-if-best

# Automatically runs:
# 01. train_stage1.py      →  group classifier
# 02. train_stage2.py      →  baseline · cascade · RAG-in-Training
# 03. merge.py             →  LoRA → merged full weights
# 04. train_stage3.py      →  Way 1 (checkpoint) · Way 2 (merged-init)
# 05. eval_all.py          →  full benchmark (RAG × prompt matrix)
# 06. xai_report.py        →  GradCAM++ · attention · uncertainty
# 07. speed_bench.py       →  Unsloth · vLLM · SGLang
# 08. eval_report.py       →  calibration · OOD · subgroup
# 09. generate_report.py   →  HTML + PDF summary
# 10. upload_hf.py         →  auto-upload if beats current best ⭐
```

**Experiment tracking:** Weights & Biases — every run logged automatically.  
**Config:** `models.yaml` + `experiments.yaml` — add a new model in 3 lines.

---

## 📅 Execution Timeline

```
╔══════════════════════════════════════════════════════════════════════╗
║  MONTH 1  ── Foundation                                              ║
║            ── Build run_all.py + W&B + config system                ║
║            ── Smoke test: Qwen2.5-VL-3B (same family, easiest)      ║
╠══════════════════════════════════════════════════════════════════════╣
║  MONTH 2  ── Model Axis                                              ║
║            ── Gemma3-4B + InternVL2.5-4B full pipeline              ║
║            ── Technique Axis: Dynamic RAG-k implementation           ║
╠══════════════════════════════════════════════════════════════════════╣
║  MONTH 3  ── XAI + Pipeline                                          ║
║            ── SAM2 segmentation preprocessing                        ║
║            ── Uncertainty score + GradCAM++ + attention atlas        ║
╠══════════════════════════════════════════════════════════════════════╣
║  MONTH 4  ── Evaluation + Application                                ║
║            ── OOD robustness · subgroup · calibration                ║
║            ── Clinical agreement study vs dermatologist              ║
║            ── REST API + Web UI deployment                           ║
╠══════════════════════════════════════════════════════════════════════╣
║  ONGOING  ── W&B dashboard live · Auto-upload best to HuggingFace   ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 📄 Documentation

| Document | Description |
|:---------|:------------|
| [`next_phase.md`](next_phase.md) | This document — Phase 2 full plan |
| [`next_phase.html`](next_phase.html) | Interactive version with animations and charts |
| [`Model/Total_Exp.md`](Model/Total_Exp.md) | Phase 1 complete experiment count (~330 runs) |
| [`Model/README.md`](Model/README.md) | Full Phase 1 technical reference |

---

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=13&duration=4000&pause=500&color=AAAAAA&center=true&vCenter=true&width=700&lines=Phase+1+Complete+%F0%9F%8C%B8+%C2%B7+85.86%25+accuracy+%C2%B7+14+models+on+HuggingFace;Phase+2+%E2%80%94+Multi-Architecture+%C2%B7+Dynamic+RAG-k+%C2%B7+SAM2+Pipeline;Weights+%26+Biases+%C2%B7+Fully+Automated+%C2%B7+One+Command" alt="footer" />

<hr/>

<p>🌸 <b>HIKARI Phase 2 Roadmap &nbsp;·&nbsp; 光（ヒカリ）</b> 🌸</p>
<p><i>Healthcare-oriented Intelligent Knowledge-Augmented Retrieval and Inference system</i></p>
<sub>Made with 💗 and 🌸 at King Mongkut's Institute of Technology Ladkrabang (KMITL)</sub>

<hr/>

</div>
