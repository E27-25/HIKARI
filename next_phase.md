<!-- ✿ HIKARI Phase 2 Roadmap ✿ -->

<div align="center">

<img src="logo/HIKARI logo.png" alt="HIKARI Logo" width="100%"/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=26&duration=3000&pause=800&color=FF9EBC&center=true&vCenter=true&width=800&lines=HIKARI+Phase+2+%F0%9F%8C%B8;Multi-Architecture+Benchmark;Explainability+%2B+Uncertainty;Clinical-Grade+Evaluation;Fully+Automated+Pipeline" alt="Typing SVG" />

<br/>

[![Phase](https://img.shields.io/badge/Phase_1-Complete_✓-4ADE80?style=for-the-badge&logo=checkmarx&logoColor=white)](.)
[![Phase](https://img.shields.io/badge/Phase_2-Planning-FF9EBC?style=for-the-badge&logo=rocket&logoColor=white)](.)
[![Status](https://img.shields.io/badge/Details-TBD-94A3B8?style=for-the-badge&logoColor=white)](.)

<br/>

> **Phase 1** — 85.86% accuracy · 14 published models · RAG-in-Training ✓
>
> **Phase 2** — direction set, details still being finalized.

</div>

---

## 🗺️ Development Axes

Phase 2 will explore the following six axes.  
**Specific methods, experiments, and priorities within each axis are not yet finalized.**

```
╔══════════════════════════════════════════════════════════════════════╗
║                        HIKARI  Phase 2                               ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║   🔭 Model Axis      ──  try different VLM architectures             ║
║   🔬 Technique Axis  ──  new training methods and strategies         ║
║   🧩 Pipeline Axis   ──  add preprocessing steps around the model   ║
║   🔍 XAI Axis        ──  explainability and confidence scores        ║
║   📊 Eval Axis       ──  deeper, more clinical evaluation            ║
║   🏥 App Axis        ──  real-world deployment interface             ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
```

---

## 🔭 Model Axis

> Try the same HIKARI pipeline on different VLM architectures and sizes.

**Goal:** Understand whether RAG-in-Training and Merged-Init generalize beyond Qwen3-VL-8B.  
**Not yet decided:** which models, how many experiments per model, evaluation scope.

---

## 🔬 Technique Axis

> Improve the training methodology without changing the model or overall pipeline structure.

**Goal:** Find smarter ways to train — better RAG sampling, harder training examples, or expanded data.  
**Not yet decided:** specific technique (Dynamic RAG-k, Hard Negative Mining, Self-Training, etc.).

---

## 🧩 Pipeline Axis

> Add preprocessing or postprocessing steps around the model.

**Goal:** Improve input quality or output structure — e.g., segment the lesion before classification.  
**Not yet decided:** which preprocessing approach, whether to use SAM2 or another method.

---

## 🔍 Explainability Axis (XAI)

> Make the model's decisions interpretable and trustworthy for clinical use.

**Goal:** Every prediction comes with a confidence score and visual explanation of what the model focused on.  
**Not yet decided:** depth of implementation, which saliency methods beyond GradCAM to include.

---

## 📊 Evaluation Axis

> Measure model quality across multiple clinical dimensions, not just accuracy.

**Goal:** Evaluate fairness across skin tones, robustness to image quality, and agreement with dermatologists.  
**Not yet decided:** which datasets, whether clinical study is in scope, extent of caption evaluation expansion.

---

## 🏥 Application Axis

> Make HIKARI usable without writing code.

**Goal:** REST API and/or Web UI where a user uploads an image and gets diagnosis + caption + heatmap.  
**Not yet decided:** tech stack, deployment target, whether this is in Phase 2 or later.

---

## ⚙️ Shared Infrastructure

Regardless of which axes are pursued, all experiments will share one automated pipeline:

```bash
python run_all.py --model <model_id> --wandb --upload-if-best
# train → eval → benchmark → report → upload
```

Experiment tracking via **Weights & Biases**. Results auto-uploaded to HuggingFace if they beat the current best.

---

## 📄 Related Documents

| Document | Description |
|:---------|:------------|
| [`next_phase.html`](next_phase.html) | Interactive version with animations and charts |
| [`Model/Total_Exp.md`](Model/Total_Exp.md) | Phase 1 complete experiment summary (~330 runs) |
| [`Model/README.md`](Model/README.md) | Phase 1 full technical reference |
| [`Model/Deploy.md`](Model/Deploy.md) | Deployment guide — vLLM, SGLang, production setup |

---

<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=13&duration=4000&pause=500&color=AAAAAA&center=true&vCenter=true&width=700&lines=Phase+1+Complete+%F0%9F%8C%B8+%C2%B7+Phase+2+In+Planning;Six+axes+confirmed+%C2%B7+Details+TBD;HIKARI+%C2%B7+%E5%85%89%EF%BC%88%E3%83%92%E3%82%AB%E3%83%AA%EF%BC%89+%C2%B7+KMITL+2026" alt="footer" />

<hr/>

<p>🌸 <b>HIKARI Phase 2 &nbsp;·&nbsp; 光（ヒカリ）</b> 🌸</p>
<p><i>Healthcare-oriented Intelligent Knowledge-Augmented Retrieval and Inference system</i></p>
<sub>Made with 💗 and 🌸 at King Mongkut's Institute of Technology Ladkrabang (KMITL)</sub>

<hr/>

</div>
