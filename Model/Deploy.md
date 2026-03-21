```
██╗  ██╗██╗██╗  ██╗ █████╗ ██████╗ ██╗
██║  ██║██║██║ ██╔╝██╔══██╗██╔══██╗██║
███████║██║█████╔╝ ███████║██████╔╝██║
██╔══██║██║██╔═██╗ ██╔══██║██╔══██╗██║
██║  ██║██║██║  ██╗██║  ██║██║  ██║██║
╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝
  Skin Disease Classification · Production Deployment Guide
```

> **Hardware Tested:** NVIDIA RTX 5070 Ti · 16 GB VRAM · CUDA 12.8 (WSL2) · Blackwell sm_120
> **Best Model:** RAG-in-Training (R2 train → R0 infer) · **85.86% accuracy** on 99-sample val set

---

## ✨ What Gets Deployed

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                    HIKARI Inference Pipeline                    │
 │                                                                 │
 │   📷 Input Image                                                │
 │        │                                                        │
 │        ▼                                                        │
 │   ┌─────────────┐    CLIP RAG (K=3)    ┌──────────────────┐   │
 │   │  RAG Index  │ ──────────────────►  │                  │   │
 │   │  (911 imgs) │                      │   Stage 2 Model  │   │
 │   └─────────────┘                      │  Disease Classi- │   │
 │                                        │  fication (10cls)│   │
 │                                        └────────┬─────────┘   │
 │                                                 │              │
 │                                    🏷️  Predicted Disease       │
 │                                                 │              │
 │                                                 ▼              │
 │                                        ┌──────────────────┐   │
 │                                        │   Stage 3 Model  │   │
 │                                        │  Caption Genera- │   │
 │                                        │  tion (clinical) │   │
 │                                        └────────┬─────────┘   │
 │                                                 │              │
 │                                    📝  Clinical Report         │
 └─────────────────────────────────────────────────────────────────┘
```

| Model File | Task | Size (bf16) |
|-----------|------|-------------|
| `skincap_fuzzytopk_s1cascade_ragR2_a09_classification_merged` | Stage 2 — Disease Classification | ~16.78 GB |
| `skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_classification_merged` | Stage 3 — Caption Generation | ~16.78 GB |

---

## ⚡ Engine Comparison

```
                    ┌─────────────────────────────────────────┐
  Unsloth BnB-4bit  │████████████████████████░░░░░░░░░  1.0×  │  baseline
  ──────────────────┼─────────────────────────────────────────┤
  vLLM   BnB-4bit   │████████░░░░░░░░░░░░░░░░░░░░░░░░░  2.3×  │  bs=1
  vLLM   BnB-4bit   │███░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  6.1×  │  bs=4 ⚡
  ──────────────────┼─────────────────────────────────────────┤
  SGLang FP8        │██████░░░░░░░░░░░░░░░░░░░░░░░░░░░  3.3×  │  bs=1
  SGLang FP8        │██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ~10×  │  bs=4 🚀
                    └─────────────────────────────────────────┘
                                              (Stage 2 — shorter output)
```

| Engine | Stage 2 bs=1 | Stage 2 bs=4 | Stage 3 bs=1 | Stage 3 bs=4 | Accuracy |
|--------|-------------|-------------|-------------|-------------|----------|
| Unsloth BnB-4bit | 1095 ms | 500 ms | 6699 ms | 3003 ms | **82.83%** |
| vLLM BnB-4bit | 480 ms | 179 ms | 2957 ms | 1094 ms | **82.83%** |
| **SGLang FP8** | **331 ms** | **110 ms** | **1695 ms** | **584 ms** | 77.78% |

> **Trade-off:** SGLang FP8 is ~10× faster but −5 pp accuracy due to FP8 quantization mismatch with QLoRA training.
> **Recommendation:** Use **vLLM BnB-4bit** for best accuracy/speed balance; **SGLang FP8** for maximum throughput.

---

## 🛠️ Prerequisites

```bash
# System requirements
CUDA  >= 12.8
VRAM  >= 16 GB  (RTX 4090 / RTX 5070 Ti / A100 recommended)
RAM   >= 32 GB
WSL2  (Windows) or native Linux

# Python environment
python >= 3.10
```

```bash
# Install base dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate
pip install open-clip-torch          # for CLIP RAG retrieval
pip install fuzzywuzzy python-Levenshtein
pip install pillow numpy
```

---

## 🚀 Option A — vLLM BnB-4bit (Best Accuracy · Recommended)

```
  ┌──────────────────────────────────────────────────────────┐
  │  vLLM  ·  BitsAndBytes 4-bit  ·  Continuous Batching    │
  │                                                          │
  │  ✅  Same accuracy as training-time Unsloth             │
  │  ✅  ~6× faster at batch=4                              │
  │  ✅  No pre-quantized weights needed                    │
  │  ✅  torch ABI stable before SGLang torch upgrade       │
  │  ⚠️   Reinstall after any SGLang torch upgrade          │
  └──────────────────────────────────────────────────────────┘
```

### Install

```bash
pip install vllm
# ⚠️  If SGLang upgraded torch to 2.9.x, reinstall vLLM:
#     pip uninstall vllm -y && pip install vllm
```

### Stage 2 — Disease Classification

```python
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image

MODEL_PATH = "/path/to/skincap_fuzzytopk_s1cascade_ragR2_a09_classification_merged"

llm = LLM(
    model=MODEL_PATH,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    trust_remote_code=True,
    max_model_len=2048,
    gpu_memory_utilization=0.88,
)
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
sp = SamplingParams(max_tokens=64, temperature=0.0)

def classify_disease(image: Image.Image, disease_group: str) -> str:
    prompt_text = (
        f"This skin lesion belongs to the group '{disease_group}'. "
        "Examine the lesion morphology (papules, plaques, macules), "
        "color (red, violet, white, brown), scale/crust, border sharpness, "
        "and distribution pattern. Based on these visual features, "
        "what is the specific skin disease?"
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt_text},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    import re
    img_count = text.count("<|vision_start|>")
    output = llm.generate(
        {"prompt": text, "multi_modal_data": {"image": [image] * max(img_count, 1)}},
        sp
    )
    return output[0].outputs[0].text.strip()
```

### Stage 3 — Caption Generation

```python
CAPTION_MODEL_PATH = "/path/to/skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_classification_merged"

llm_caption = LLM(
    model=CAPTION_MODEL_PATH,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    trust_remote_code=True,
    max_model_len=2048,
    gpu_memory_utilization=0.88,
)
processor_caption = AutoProcessor.from_pretrained(CAPTION_MODEL_PATH, trust_remote_code=True)
sp_caption = SamplingParams(max_tokens=256, temperature=0.0)

def generate_caption(image: Image.Image) -> str:
    prompt_text = (
        "Describe this skin lesion image in detail. Include information about its "
        "appearance, possible diagnosis, and recommended examinations."
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt_text},
    ]}]
    text = processor_caption.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    import re
    img_count = text.count("<|vision_start|>")
    output = llm_caption.generate(
        {"prompt": text, "multi_modal_data": {"image": [image] * max(img_count, 1)}},
        sp_caption
    )
    return output[0].outputs[0].text.strip()
```

### Full Pipeline (vLLM)

```python
import gc, torch
from rag_retrieval import HybridRAGRetriever   # project RAG module

# ── Load RAG index ────────────────────────────────────────────────
rag = HybridRAGRetriever.load("rag_index_R0_train.npz")   # CLIP image-only

def run_pipeline(image_path: str) -> dict:
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((672, 672), Image.LANCZOS)

    # Step 1: retrieve K=3 reference cases
    refs = rag.retrieve(img, k=3)   # [(path, label, caption), ...]
    disease_group = refs[0][1]       # use top-1 label as group hint

    # Step 2: disease classification
    disease = classify_disease(img, disease_group)

    # Step 3: clinical caption
    caption = generate_caption(img)

    return {
        "predicted_disease": disease,
        "clinical_caption":  caption,
        "retrieved_refs":    [r[1] for r in refs],
    }
```

---

## 🔥 Option B — SGLang FP8 (Maximum Throughput)

```
  ┌──────────────────────────────────────────────────────────┐
  │  SGLang  ·  FP8 Online Quantization  ·  RadixAttention  │
  │                                                          │
  │  🚀  ~10× Stage 2,  ~11.5× Stage 3  vs Unsloth bs=1    │
  │  ✅  No pre-quantized weights — quantizes at load time  │
  │  ✅  RadixAttention for KV-cache prefix sharing         │
  │  ⚠️   −5 pp accuracy (77.78% vs 82.83%)                │
  │  ⚠️   Startup: ~2–4 min (loading shards over WSL2/NTFS) │
  └──────────────────────────────────────────────────────────┘
```

### Install

```bash
# Install SGLang (requires CUDA 12.x)
pip install sglang[all]

# Disable problematic extensions on Blackwell sm_120
export SGLANG_DISABLE_DEEPGEMM=1
export FLASHINFER_DISABLE_VERSION_CHECK=1
```

### Stage 2 — Disease Classification

```python
import os
os.environ["SGLANG_DISABLE_DEEPGEMM"] = "1"
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"

import sglang as sgl
from transformers import AutoProcessor
from PIL import Image

MODEL_PATH = "/path/to/skincap_fuzzytopk_s1cascade_ragR2_a09_classification_merged"

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
engine = sgl.Engine(
    model_path=MODEL_PATH,
    dtype="bfloat16",
    quantization="fp8",
    context_length=2048,
    mem_fraction_static=0.88,
    trust_remote_code=True,
    disable_cuda_graph=True,   # required for sm_120 Blackwell
    log_level="error",
)

def classify_disease_sglang(image: Image.Image, disease_group: str) -> str:
    prompt_text = (
        f"This skin lesion belongs to the group '{disease_group}'. "
        "Examine the lesion morphology (papules, plaques, macules), "
        "color (red, violet, white, brown), scale/crust, border sharpness, "
        "and distribution pattern. Based on these visual features, "
        "what is the specific skin disease?"
    )
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt_text},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    out = engine.generate(
        prompt=text,
        image_data=image,
        sampling_params={"max_new_tokens": 64, "temperature": 0.0},
    )
    raw = out["text"].strip() if isinstance(out, dict) else out[0]["text"].strip()
    # Strip dangling </think> token from Qwen3-VL thinking chain
    raw = raw.replace("</think>", "").strip()
    # Remove "This image shows" prefix if present
    import re
    raw = re.sub(r"(?i)^this image shows\s*", "", raw).strip()
    return raw
```

### Stage 3 — Caption Generation

```python
CAPTION_MODEL_PATH = "/path/to/skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_classification_merged"

# ⚠️  Load AFTER shutting down Stage 2 engine to reclaim VRAM
engine.shutdown()
import torch; torch.cuda.empty_cache()

processor_cap = AutoProcessor.from_pretrained(CAPTION_MODEL_PATH, trust_remote_code=True)
engine_cap = sgl.Engine(
    model_path=CAPTION_MODEL_PATH,
    dtype="bfloat16",
    quantization="fp8",
    context_length=2048,
    mem_fraction_static=0.88,
    trust_remote_code=True,
    disable_cuda_graph=True,
    log_level="error",
)

def generate_caption_sglang(image: Image.Image) -> str:
    prompt_text = (
        "Describe this skin lesion image in detail. Include information about its "
        "appearance, possible diagnosis, and recommended examinations."
    )
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt_text},
    ]}]
    text = processor_cap.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    out = engine_cap.generate(
        prompt=text,
        image_data=image,
        sampling_params={"max_new_tokens": 256, "temperature": 0.0},
    )
    return (out["text"] if isinstance(out, dict) else out[0]["text"]).strip()
```

### Batch Processing (SGLang — Maximum Throughput)

```python
from concurrent.futures import ThreadPoolExecutor

def batch_classify(images: list[Image.Image], groups: list[str],
                   batch_size: int = 4) -> list[str]:
    """
    Process images in batches.
    SGLang FP8 bs=4 → ~110 ms/img for Stage 2  (~9 imgs/sec)
    """
    results = []
    for i in range(0, len(images), batch_size):
        batch_imgs   = images[i : i + batch_size]
        batch_groups = groups[i : i + batch_size]
        batch_results = [
            classify_disease_sglang(img, grp)
            for img, grp in zip(batch_imgs, batch_groups)
        ]
        results.extend(batch_results)
    return results
```

---

## 📦 RAG Setup (Required for Both Engines)

```
  ┌─────────────────────────────────────────┐
  │           RAG Retrieval Setup           │
  │                                         │
  │   Training images (911)                 │
  │        │                                │
  │        ▼  CLIP ViT-B/32                 │
  │   ┌──────────────┐                      │
  │   │  RAG Index   │  ← rag_index_R0.npz  │
  │   │  img_embs    │    (pre-computed)     │
  │   │  labels      │                      │
  │   │  captions    │                      │
  │   └──────┬───────┘                      │
  │          │  cosine similarity            │
  │          ▼                               │
  │   Top-K references → Stage 2 prompt     │
  └─────────────────────────────────────────┘
```

```bash
# Build RAG index from training set (run once)
python build_rag_index.py --encoder R0 --split train --output rag_index_R0_train.npz
```

```python
# RAG retrieval at inference
from rag_retrieval import HybridRAGRetriever

rag = HybridRAGRetriever.load("rag_index_R0_train.npz")

def get_group_hint(image: Image.Image) -> str:
    """Retrieve top-1 reference to get disease group hint."""
    refs = rag.retrieve(image, k=1)
    # Map disease label → group (4 groups)
    return refs[0][1]   # returns disease label; map to group externally
```

---

## 📊 Speed Reference

```
  Stage 2  (max_new_tokens=64)
  ════════════════════════════════════════════════════
  Engine              bs=1        bs=4        Speedup
  ────────────────────────────────────────────────────
  Unsloth-4bit      1095 ms     500 ms         1.0×
  vLLM-BnB4          480 ms     179 ms    2.3× / 6.1×
  SGLang-FP8         331 ms     110 ms    3.3× / ~10×
  ════════════════════════════════════════════════════

  Stage 3  (max_new_tokens=256)
  ════════════════════════════════════════════════════
  Engine              bs=1        bs=4        Speedup
  ────────────────────────────────────────────────────
  Unsloth-4bit      6699 ms    3003 ms         1.0×
  vLLM-BnB4         2957 ms    1094 ms    2.3× / 6.1×
  SGLang-FP8        1695 ms     584 ms    3.9× / ~11.5×
  ════════════════════════════════════════════════════
```

---

## 🩺 Output Parsing

```python
import re
from fuzzywuzzy import fuzz

DISEASE_NAMES = [
    "psoriasis", "lupus erythematosus", "lichen planus", "scleroderma",
    "photodermatoses", "sarcoidosis", "melanocytic nevi",
    "squamous cell carcinoma in situ", "basal cell carcinoma", "acne vulgaris",
]

def match_disease(raw_output: str) -> str:
    """
    Parse raw model output to one of the 10 known disease labels.
    Returns 'Unknown' if no match found (threshold < 70).
    """
    text = raw_output.lower().strip()
    # Remove thinking tokens
    text = re.sub(r"</think>", "", text)
    text = re.sub(r"^this image shows\s*", "", text)
    # Remove trailing punctuation
    text = text.rstrip(".,;:").strip()

    best_score, best_match = 0, "Unknown"
    for disease in DISEASE_NAMES:
        # Word-overlap check first
        words_q   = set(text.split())
        words_ref = set(disease.split())
        if words_q & words_ref:
            score = fuzz.token_sort_ratio(text, disease)
            if score > best_score:
                best_score, best_match = score, disease

    return best_match if best_score >= 70 else "Unknown"
```

---

## ⚙️ Known Issues & Fixes

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                      Troubleshooting                           │
  └─────────────────────────────────────────────────────────────────┘
```

| Issue | Symptom | Fix |
|-------|---------|-----|
| **SGLang startup timeout** | Hangs for >2 min on WSL2/NTFS | Set `timeout=600` in engine init |
| **vLLM ABI conflict** | `undefined symbol _ZN3c104cuda29...` after SGLang install | `pip uninstall vllm -y && pip install vllm` |
| **LMDeploy OOM** | `bf16 16.78 GB > VRAM` | Not supported without pre-quantized weights |
| **TensorRT-LLM** | `CUDA driver ≥ 13.0 required` | Not supported on CUDA 12.x |
| **Image OOM** | VRAM crash on large inputs | Always use `img.thumbnail((672, 672), LANCZOS)` |
| **0% accuracy** | Model outputs disease not in training list | Verify `DISEASE_NAMES` matches training labels exactly |
| **SGLang `</think>` prefix** | `max_new_tokens=64` cuts thinking chain mid-way | Strip `</think>` from output before matching |

---

## 🗂️ File Structure

```
Model/
├── skincap_fuzzytopk_s1cascade_ragR2_a09_classification_merged/
│   ├── config.json
│   ├── model-00001-of-00004.safetensors
│   ├── model-00002-of-00004.safetensors
│   ├── model-00003-of-00004.safetensors
│   ├── model-00004-of-00004.safetensors
│   └── tokenizer_config.json
│
├── skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_classification_merged/
│   └── (same structure as above)
│
├── rag_index_R0_train.npz          ← CLIP image embeddings (911 train imgs)
├── split_info_3stage.json          ← Val split (99 samples, DO NOT change)
├── Deploy.md                       ← this file
└── inference_disease_classification.py
```

---

## 🎯 Quick Start

```bash
# ─── 1. Clone / navigate to project ───────────────────────────────
cd /path/to/HIKARI/Model

# ─── 2. Set CUDA path (WSL2) ──────────────────────────────────────
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=/usr/local/cuda-12.8

# ─── 3a. Run with vLLM (recommended) ─────────────────────────────
python inference_disease_classification.py \
    --method fuzzytopk \
    --rag R0 \
    --prompt P0 \
    --engine vllm \
    --batch_size 4

# ─── 3b. Run with SGLang FP8 (max throughput) ────────────────────
export SGLANG_DISABLE_DEEPGEMM=1
export FLASHINFER_DISABLE_VERSION_CHECK=1

python inference_disease_classification.py \
    --method fuzzytopk \
    --rag R0 \
    --prompt P0 \
    --engine sglang \
    --batch_size 4
```

---

```
  ╔══════════════════════════════════════════════════════════════╗
  ║                    Accuracy Summary                         ║
  ║                                                             ║
  ║   Best config  :  RAG-in-Training  ·  R0-P0               ║
  ║   Accuracy     :  85.86%  (85 / 99 samples)               ║
  ║   Engine       :  vLLM BnB-4bit  (accuracy-preserving)    ║
  ║   Speed (bs=4) :  ~179 ms/img Stage2 · ~1094 ms/img Stage3 ║
  ║                                                             ║
  ║   Max throughput :  SGLang FP8 bs=4                        ║
  ║   Speed (bs=4)   :  ~110 ms/img Stage2 · ~584 ms/img Stage3║
  ║   Accuracy       :  77.78%  (−5 pp vs vLLM)               ║
  ╚══════════════════════════════════════════════════════════════╝
```

---

*HIKARI · Skin Disease Classification · Qwen3-VL-8B-Thinking fine-tuned on SkinCAP*
*GPU: RTX 5070 Ti · Val set: 99 samples · 10 disease classes*
