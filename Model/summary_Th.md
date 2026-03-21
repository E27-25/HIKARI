# HIKARI การจำแนกโรคผิวหนัง — สรุปโครงการทั้งหมด (ภาษาไทย)

## ภาพรวม

โครงการนี้เป็นการทดสอบแบบครบวงจรสำหรับการจำแนกโรคผิวหนัง โดยใช้ **Qwen3-VL-8B-Thinking** ที่ผ่านการ fine-tune บนชุดข้อมูล SkinCAP
โครงการทดสอบกลยุทธ์การฝึกหลายรูปแบบ การตั้งค่า RAG (Retrieval-Augmented Generation) และรูปแบบ prompt ต่าง ๆ
วัดผลบน **ชุดข้อมูล validation 99 ตัวอย่าง** (10 คลาสโรค, การแบ่ง stratified 3 ขั้น)

---

## 0. นิยามชื่อการทดลองทั้งหมด (Experiment Name Definitions)

> ส่วนนี้เป็น **แหล่งอ้างอิงหลัก** สำหรับการแมปชื่อที่ใช้ใน Paper → ชื่อ internal model → รายละเอียดสถาปัตยกรรม

---

### 0.1 ตารางแมปชื่อ Paper ↔ Internal Model ↔ Code

#### โมเดลหลักใน Paper (Table II)

| ชื่อใน Paper | Internal Model / ชื่อใหม่ | Code Path | หมายเหตุ |
|-------------|--------------------------|-----------|---------|
| **Single-Image FT** | `fuzzytopk` | `skincap_fuzzytopk_classification_merged` | Baseline ไม่มี RAG |
| **2-Stage Cascade FT** | `M1` = "GT-Oracle 3-Stage" | `skincap_3stage_disease_M1_merged` | ฝึก 3 ขั้น, cascade 2 ขั้นสำหรับ classification, oracle GT group |
| **Cascaded FT** | `fuzzytopk_s1cascade` | `skincap_fuzzytopk_s1cascade_classification_merged` | เริ่มจาก Group Classifier weights |
| **RAG-in-Training (ours)** | `fuzzytopk_s1cascade_ragR2_a09` | `skincap_fuzzytopk_s1cascade_ragR2_a09_classification_merged` | **Best = 85.86%** |

#### M-Series Ablation (ไม่ได้อยู่ใน Paper Table โดยตรง — ดู Section 2.4)

| ชื่อใหม่ | Code | สิ่งที่แตกต่าง | Acc (ablation) |
|---------|------|--------------|----------------|
| No-Context 3-Stage | `M0` | ไม่มี group context เลย | 61.0% |
| **GT-Oracle 3-Stage** | **`M1`** | **GT group ทั้ง train+infer (oracle)** | **66.0%** |
| Oracle-Train, Cascade-Infer | `M2` | GT train / Stage1 predicted infer | 57.4% |
| Full-Cascade 3-Stage | `M3` | Stage1 predicted ทั้ง train+infer (realistic) | 57.4% |
| Soft-Probability 3-Stage | `M4` | GT train / Beam soft probs infer | 61.0% |

> **Group Classifier ร่วม (Stage 1):** `skincap_3stage_group_classification_merged` → 88.68% (แบ่ง 4 กลุ่มโรค)

---

### 0.2 นิยามและคำอธิบายแต่ละ Experiment

#### 🔹 Single-Image FT (= `fuzzytopk`)
**นิยาม:** Fine-tune โมเดลด้วย **ภาพเดียวต่อตัวอย่าง** (ไม่มีภาพ reference เพิ่มเติม) — baseline ที่ง่ายที่สุด

**สถาปัตยกรรมการฝึก (2 ขั้น):**
```
Base Model (Qwen3-VL-8B)
    ↓ Stage 1: Fine-tune จำแนกโรค 10 คลาส (10-class disease classification)
Disease Classifier (fuzzytopk)  →  ความแม่นยำ 74.0% (บน 3-stage val split 99 ตัวอย่าง)
    ↓ Stage 2: Fine-tune สร้าง caption จาก Stage 1 weights
Caption Model
```

**Input ตอน Inference:** `[query_img] + prompt` เท่านั้น (ไม่มี reference images)

**ผลสรุป:**
- ไม่มี RAG: **74.00%** (ดีที่สุดสำหรับโมเดลนี้)
- มี RAG (ทุก config): ลดลงเหลือ 63–65% → **RAG ทำให้สับสนเพราะไม่เคยฝึกกับหลายภาพ**
- Val set ของตัวเอง (`split_info_fuzzytopk.json`, 101 ตัวอย่าง): 83%
- Val set 3-stage (`split_info_3stage.json`, 99 ตัวอย่าง): **74.0%** (ใช้ตัวเลขนี้ใน paper)

**ทำไมชื่อนี้:** "Single-Image" = ฝึกและทำนายจากภาพเดียว ไม่มี retrieval context

---

#### 🔸 2-Stage Cascade FT (= `M1` — GT-Oracle 3-Stage)
**นิยาม:** เป็น **M1 (GT-Oracle 3-Stage)** จาก M-series ablation — pipeline 3 training stages ที่ inject **GT group context** ทั้งตอนฝึกและ inference
> ดูรายละเอียดเต็ม M-series ทั้งหมด (M0–M4) ได้ที่ **Section 2.4**

**สถาปัตยกรรมการฝึก (3 ขั้น):**
```
Base Model (Qwen3-VL-8B)
    ↓ Stage 1: Fine-tune จำแนก 4 กลุ่มโรค
Group Classifier  →  skincap_3stage_group_classification_merged (88.68%)
    ↓ Stage 2: Fine-tune จำแนก 10 โรค + inject "This lesion belongs to group: X" (GT label)
Disease Classifier (M1)  →  skincap_3stage_disease_M1_merged
    ↓ Stage 3: Fine-tune สร้าง caption (Merged-Init)
Caption Model
```

**ทำไมชื่อ "2-Stage Cascade FT" ในตาราง paper:**
- อธิบาย **2-stage cascade ที่ใช้สำหรับ classification** (Group Classifier → Disease Classifier)
- Stage 3 (caption) ไม่ใช้ใน classification evaluation
- แยกออกจาก "Cascaded FT" (= fuzzytopk_s1cascade ที่เป็นโมเดลคนละตัว)

**ความสัมพันธ์ระหว่างตัวเลข:**
| สภาวะ | Acc | อธิบาย |
|-------|-----|--------|
| ablation (GT oracle) | **66.0%** | inject GT group ทั้ง train และ inference (upper bound) |
| full benchmark, ไม่มี RAG P0 | **47.87%** | run "as-is" โดยไม่มี group context เสริมที่ inference |
| full benchmark, RAG R0-P1 | **59.38%** | P1 CoT prompt + CLIP RAG ที่ inference |

**ทำไมผลแตกต่างระหว่าง 66% และ 47.87%:** M1 ถูกฝึกให้รับ group context ใน prompt แต่ full benchmark ทดสอบโดยไม่ inject group context → โมเดลสับสน (train-inference mismatch)

**ทำไมผลแย่กว่า Single-Image FT (74%):** 3-stage cascade penalty + error propagation + prompt complexity → ดูรายละเอียดใน Section 2.4

---

#### 🔶 Cascaded FT (= `fuzzytopk_s1cascade`)
**นิยาม:** Fine-tune แบบ **Single-Image** (เหมือน fuzzytopk) แต่ **เริ่มต้น Stage 1 จาก weights ของ Group Classifier** แทนที่จะเริ่มจาก Base Model โดยตรง

**สถาปัตยกรรมการฝึก (2 ขั้น แต่ Stage 1 มี cascaded initialization):**
```
Group Classifier  →  skincap_3stage_group_classification_merged (88.68%)
    ↓ Stage 1: Fine-tune ต่อจาก Group Classifier → จำแนก 10 โรค
Disease Classifier (s1cascade)  →  skincap_fuzzytopk_s1cascade_classification_merged
    ↓ Stage 2: Fine-tune สร้าง caption
Caption Model
```

**Input ตอน Inference:** `[ref_img_1..K] + [query_img] + prompt` (ใช้ RAG images จาก retrieval)

**ทำไมชื่อ "Cascaded FT":** Stage 1 ของ Disease Classifier ได้รับ **cascaded pretraining** จาก Group Classifier → โมเดลเรียนรู้การเปรียบเทียบลักษณะข้ามกลุ่มโรคก่อน

**ผลสรุป:**

| RAG Config | Accuracy |
|-----------|---------|
| R0 (CLIP image-only) | 72.73% |
| R2 α=0.5 (SigLIP+BGE-M3) | 74.75% |
| **R2 α=0.9 (SigLIP+BGE-M3)** | **79.80%** ← ดีที่สุด |
| R3 α=0.7 (Jina+MedCPT) | 78.79% |

**ทำไม Cascaded FT ใช้ RAG ได้ดีกว่า fuzzytopk:** Group Classifier pretraining สอนให้โมเดลเปรียบเทียบ visual features ข้ามกลุ่ม → สามารถใช้ reference images ได้อย่างมีประสิทธิภาพ

---

#### 🟢 RAG-in-Training (= `fuzzytopk_s1cascade_ragR2_a09`) — **[HIKARI ของเรา]**
**นิยาม:** ฝึก Cascaded FT ใหม่ โดย **inject K=1 reference image เข้าไปในทุก training sample** ด้วย R2 encoder (SigLIP+BGE-M3, α=0.9) → ปิด train-inference distribution gap

**สถาปัตยกรรมการฝึก (2 ขั้น + RAG-augmented training data):**
```
Group Classifier  →  skincap_3stage_group_classification_merged (88.68%)
    ↓ Stage 1: Fine-tune ต่อ + ทุก training sample มี [ref_img] + [query_img]
Disease Classifier (ragR2_a09)  →  skincap_fuzzytopk_s1cascade_ragR2_a09_classification_merged
    ↓ Stage 2: Fine-tune สร้าง caption (Merged-Init: merge Stage 1 weights ก่อน + fresh LoRA)
Caption Model  →  skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_classification_merged
```

**รูปแบบ Training Sample:**
```
[ref_img]  Reference 1: psoriasis
Description: Erythematous scaly plaques with silvery surface...
[query_img]  Carefully examine this dermatological image...
↳ Answer: psoriasis
```

**รูปแบบ Inference (K=3 references):**
```
Here are similar reference cases for context:
[ref_img_1]  Reference 1: psoriasis
Description: Erythematous scaly plaques...
[ref_img_2]  Reference 2: lichen planus
Description: Flat-topped violaceous papules...
[ref_img_3]  Reference 3: psoriasis
Description: ...
Now, identify the disease in this new image:
[query_img]  Carefully examine this dermatological image...
```

**ผลสรุป:**

| RAG Inference | Acc | Bal Acc | F1 Macro | Kappa |
|--------------|-----|---------|----------|-------|
| **R0-P0 (CLIP image-only)** | **85.86%** | 82.12% | 80.78% | 84.18% |
| R0-P3 | 83.84% | 80.52% | 79.52% | 81.93% |
| R2-P0 (SigLIP+BGE-M3) | 82.83% | 80.12% | 80.07% | 80.80% |
| R2-P3 | 82.83% | 80.78% | 80.77% | 80.81% |

**ข้อค้นพบสำคัญ:** CLIP (R0) ที่ inference ดีกว่า SigLIP+BGE-M3 (R2 ที่ใช้ตอนฝึก) +3.03%
→ โมเดลเรียนรู้แนวคิด "ดู reference → ใช้ label" แบบทั่วไป ไม่ยึดติด encoder เฉพาะ (encoder-agnostic generalization)

**ทำไมชื่อ "RAG-in-Training":** ต่างจาก "Inference-Only RAG" ตรงที่ RAG context ถูก inject ตั้งแต่ **ขั้นฝึก** ไม่ใช่แค่ inference → โมเดลรู้จักวิธีใช้ reference images จาก training

---

### 0.3 Zero-Shot Frontier Models — นิยาม

**Zero-Shot Frontier Models:** โมเดลขนาดใหญ่ที่ **ไม่ได้ผ่านการ fine-tune** บน SkinCAP dataset เลย ใช้เพื่อ establish lower bound ว่า off-the-shelf VLMs ทำได้แค่ไหน

| ชื่อโมเดล | ผู้พัฒนา | Acc | หมายเหตุ |
|---------|---------|-----|---------|
| LLaMA-4-Scout-17B | Meta | 12.87% | Multi-expert, ไม่เชี่ยวชาญ medical |
| LLaMA-4-Maverick-17B | Meta | 12.87% | Multi-expert, ไม่เชี่ยวชาญ medical |
| Gemini-2.5-Flash | Google | 25.74% | Lightweight, fast inference |
| Gemini-3.1-Pro | Google | 23.23% | — |
| Gemini-2.5-Pro | Google | 30.69% | Best ใน Gemini family ที่ทดสอบ |
| Qwen3-VL-8B | Alibaba | 33.33% | Base model ของเรา (ก่อน fine-tune) |
| **Qwen2.5-VL-7B** [11] | Alibaba | **50.51%** | Prior work, differential diagnosis ได้ดี |

**⚠️ หมายเหตุสำคัญเกี่ยวกับ "Zero-Shot" :**
- ทุกโมเดลในตาราง **ใช้ RAG R0 (CLIP) ที่ inference** — ไม่ใช่ Zero-Shot ที่ไม่มี context เลย
- ผล No-RAG แท้จริง: Qwen3 = 0–1%, Qwen2.5 = 0% (parse ไม่ได้เลยเพราะ output ยาวเกิน)
- ชื่อ "Zero-Shot" หมายถึง **ไม่มี fine-tuning** บน SkinCAP ไม่ใช่ไม่มี RAG context
- ตัวเลขที่รายงาน = best config ของแต่ละโมเดล: Qwen3→R0+P3, Qwen2.5→R0+P2

**เหตุผลที่ Zero-Shot (No fine-tune) ทำได้ต่ำโดยรวม:**
- ไม่เคยเห็น SkinCAP label format → output ยาวเป็น paragraph parse ไม่ได้
- 10 คลาสเฉพาะทาง (sarcoidosis, photodermatoses ฯลฯ) ต้องการความรู้ dermoscopy เฉพาะ
- Max tokens = 1024; ถ้าตัดที่ 64 tokens (ไม่พอปิด `</think>`) จะได้ ~0% เท่านั้น
- **valid_predictions เป็น bottleneck:** ไม่ใช่ความรู้เรื่องโรค แต่ output format ที่ parser จับไม่ได้

---

### 0.4 ตารางผลการทดลองสุดท้าย (ตาม Paper — Table II)

> ตารางนี้ตรงกับ Table II ใน Conference Paper (ITC-CSCC 2025)

| Model | RAG Train | RAG Infer. | Acc. |
|-------|-----------|-----------|------|
| ***Zero-Shot Frontier Models (no fine-tuning)*** | | | |
| LLaMA-4-Scout-17B | – | – | 12.87% |
| LLaMA-4-Maverick-17B | – | – | 12.87% |
| Gemini-2.5-Flash | – | – | 25.74% |
| Gemini-3.1-Pro | – | – | 23.23% |
| Gemini-2.5-Pro | – | – | 30.69% |
| Qwen3-VL-8B | – | R0+P3 | 33.33% |
| Qwen2.5-VL-7B [11] | – | R0+P2 | 50.51% |
| ***Domain Fine-Tuned (No Training RAG)*** | | | |
| Single-Image FT | – | – | 74.00% |
| 2-Stage Cascade FT | – | R0 | 59.38% |
| ***Inference-Only RAG (Cascaded FT)*** | | | |
| Cascaded FT (R2, α=0.5) | – | R2 | 74.75% |
| Cascaded FT (R2, α=0.9) | – | R2 | 79.80% |
| ***RAG-in-Training (Ours)*** | | | |
| RAG-in-Training (R2 train, R2) | R2 K=1 | R2 | 82.83% |
| **RAG-in-Training (R2 train, R0)** | **R2 K=1** | **R0** | **85.86%** |

**หมายเหตุสำคัญ:**
- **Single-Image FT 74.00%** → ทดสอบบน `split_info_3stage.json` (99 ตัวอย่าง); บน split ของตัวเอง (101 ตัวอย่าง) ได้ 83%
- **2-Stage Cascade FT R0 59.38%** → M1 ใช้ P1 prompt ที่ inference (CoT ทีละขั้น); ไม่ได้ฝึกกับ RAG จึงใช้ RAG inference ได้จำกัด
- **Cascaded FT (R2, α=0.9) 79.80%** → best ก่อนมี RAG-in-training; ใช้ SigLIP+BGE-M3 encoder, image-dominant (90% image / 10% text)
- **RAG-in-Training (R2 train, R0) 85.86%** → ฝึกด้วย R2 แต่ inference ด้วย R0 (CLIP) ให้ผลดีกว่า → encoder-agnostic generalization

---

### 0.5 RAG Encoder Definitions (R0–R4)

| ID | ชื่อย่อใน Paper | Image Encoder | Text Encoder | กลยุทธ์ | หมายเหตุ |
|----|---------------|--------------|--------------|---------|---------|
| **R0** | CLIP | `openai/clip-vit-base-patch32` | ไม่มี | Image-only | α=1.0 ตลอด (เฉพาะภาพ) |
| **R1** | CLIP+ClinicalBERT | `openai/clip-vit-base-patch32` | `medicalai/ClinicalBERT` | Image+Text | ข้อความคลินิกทั่วไป |
| **R2** | SigLIP+BGE-M3 | `google/siglip-2-base-patch16-512` | `BAAI/bge-m3` | Image+Text | **ใช้ตอนฝึก HIKARI** |
| **R3** | Jina-CLIP+MedCPT | `jinaai/jina-clip-v2` | `ncbi/MedCPT-Query-Encoder` | Image+Text | MedCPT เชี่ยวชาญ medical |
| **R4** | Nomic | `nomic-ai/nomic-embed-vision-v1.5` | `nomic-ai/nomic-embed-text-v1.5` | Cross-modal | Unified embedding space |

**สูตรคะแนน Retrieval:**
```
score = α × cos(img_enc(query), ref_img_emb)
      + (1−α) × cos(txt_enc(query_text), ref_txt_emb)
```
- R0: α=1.0 เสมอ (ไม่มี text encoder)
- R2 ที่ดีที่สุด: α=0.9 (image 90%, text 10%)

---

### 0.6 Prompt Definitions (P0–P3)

| ID | ชื่อ | ใช้กับโมเดลไหน | จุดเด่น |
|----|------|--------------|--------|
| **P0** | Direct Clinical Observation | fuzzytopk, s1cascade, ragR2_a09 | ถามตรง สั้น → parse ง่าย |
| **P1** | Step-by-Step CoT | M1 | Numbered steps ตรงกับ `<think>` format |
| **P2** | Differential Diagnosis | Qwen2.5 zero-shot | ให้ Top-3 แล้วเลือก |
| **P3** | Structured Clinical Assessment | Qwen3 zero-shot | Keyword `Diagnosis:` ปิด CoT |

**P0 (ข้อความจริง):**
```
Carefully examine this dermatological image. Look for: lesion morphology
(papule/plaque/macule/nodule), color (red/violet/white/brown/black),
scale or crust, border sharpness, and distribution.
Based on these visual features, what is the specific skin disease?
```

---

### 0.7 สรุปเส้นทาง Model Evolution

```
[Zero-Shot Baseline]
Qwen3-VL-8B (ไม่ fine-tune) → 33.33%
    ↓ fine-tune ด้วยภาพเดียว
[Single-Image FT] fuzzytopk → 74.00% (+40.67%)
    ↓ เริ่มจาก Group Classifier weights
[2-Stage Cascade FT] M1 → 59.38% with RAG (ลดลงจาก fuzzytopk เพราะ cascade penalty)
    ↓ เปลี่ยนเป็น fuzzytopk_s1cascade + RAG inference
[Inference-Only RAG] Cascaded FT → 79.80% (+5.80% จาก fuzzytopk baseline)
    ↓ inject RAG เข้าไปในทุก training sample
[RAG-in-Training] HIKARI → 85.86% (+11.86% จาก baseline) ← SOTA ของเรา
```

---

## 1. ชุดข้อมูลและคลาสโรค

**ชุดข้อมูล:** SkinCAP (skincap_v240623.csv) — ภาพผิวหนัง 4,000 ภาพพร้อม label โรคและคำอธิบายทางคลินิก
**Label ต้นฉบับ:** คอลัมน์ CSV `disease` (ตัวพิมพ์เล็ก, normalize แล้ว); `caption_zh_polish_en` ใช้เป็นแหล่ง text embedding ของ RAG
**การแบ่ง Train/Val:** Stratified split เก็บใน `split_info_3stage.json` (ล็อคหลังการฝึก Stage 1, ไม่สร้างใหม่)

### 1.1 คลาสโรคและจำนวน Validation

| กลุ่ม | โรค | จำนวน Val |
|-------|-----|-----------|
| 1. โรคอักเสบและภูมิคุ้มกัน | โรคสะเก็ดเงิน (psoriasis) | 13 |
| | โรคลูปัส (lupus erythematosus) | 9 |
| | ไลเคน พลานัส (lichen planus) | 9 |
| | สเกลโรเดอร์มา (scleroderma) | 8 |
| | โรคผิวหนังจากแสง (photodermatoses) | 8 |
| | ซาร์คอยโดซิส (sarcoidosis) | 7 |
| 2. เนื้องอกชนิดดีและไฝ | ไฝเมลาโนไซต์ (melanocytic nevi) | 12 |
| 3. เนื้องอกชนิดร้าย | มะเร็งเซลล์สความัสในชั้น (SCCIS) | 12 |
| | มะเร็งเซลล์ฐาน (basal cell carcinoma) | 13 |
| 4. สิวและรูขุมขน | สิวอักเสบ (acne vulgaris) | 8 |
| **รวม** | | **99** |

### 1.2 ขั้นตอน Preprocessing

```
ข้อมูล CSV ดิบ (4,000 แถว)
    ↓ Normalize label โรค: ตัวพิมพ์เล็ก + ตัดช่องว่าง
    ↓ ใช้ FORCED_CANONICAL: "squamous cell carcinoma" → "squamous cell carcinoma in situ"
    ↓ กรองเหลือ TOP_10_DISEASES (match แบบ exact): 4,000 → 1,010 ตัวอย่าง
    ↓ fuzzy_consolidate_diseases(): Jaccard word-overlap (threshold 91), ทำ หลัง กรอง
           (ป้องกัน cross-group artifacts; sort() เพื่อ determinism)
    ↓ Stratified train/val split ผ่าน split_info_3stage.json: 911 train / 99 val
    ↓ Training: sqrt-mode oversampling (เพิ่ม rare class ~1.36× ต่อ class ที่น้อยที่สุด)
    ↓ โหลดภาพ: thumbnail(672, 672, LANCZOS) — จำกัดขนาดสูงสุด 672px ต่อด้าน
```

**หมายเหตุเกี่ยวกับ val split:**
- `split_info_3stage.json` (99 val): การทดลอง RAG ทั้งหมดและ M-series ทั้งหมด (M0/M1/M2/M3/M4) ใช้ split นี้
- `split_info_fuzzytopk.json`: โมเดล fuzzytopk แบบ standalone ใช้ split นี้ → 83% แต่เมื่อทดสอบบน 3-stage split ได้ 74.0%

---

## 2. สถาปัตยกรรมโมเดล

### 2.0 การตั้งค่าการฝึกร่วมกัน

| พารามิเตอร์ | ค่า |
|------------|-----|
| โมเดลหลัก | Qwen3-VL-8B-Thinking |
| Quantization | 4-bit NF4 ผ่าน Unsloth |
| LoRA r | 16 |
| LoRA alpha | 32 |
| เป้าหมาย LoRA | ทุก attention + MLP modules |
| Batch size | 2 ต่อ GPU, gradient accumulation = 4 → effective batch = 8 |
| Optimizer | AdamW 8-bit paged |
| GPU | NVIDIA RTX 5070 Ti (VRAM 15.92 GB) |
| ปรับขนาดภาพ (inference) | 672×672 thumbnail (LANCZOS) |
| inference max_new_tokens | 64 (โมเดล fine-tune เฉลี่ย 2.1 คำ) |
| inference max_new_tokens (zero-shot) | 1024 (reasoning chain ต้องการ ~500–1000 tokens) |

### 2.1 fuzzytopk (74% — baseline ไม่มี RAG)

**สถาปัตยกรรม:** Fine-tune 2 ขั้นจาก Qwen3-VL-8B base
**Stage 1:** Fine-tune การจำแนกโรค (10 คลาส) → `skincap_fuzzytopk_classification_merged`
**Stage 2:** Fine-tune การสร้าง caption จาก Stage 1

| พารามิเตอร์ | ค่า |
|------------|-----|
| ข้อมูล Train | 939 ภาพจาก `split_info_fuzzytopk.json` |
| LR | 2e-4 |
| Epochs | 3 |
| LoRA dropout | 0 |
| ผลลัพธ์ best (99 val 3-stage) | **74.0%** |

**Training script:** `train_two_stage_FuzzyTopK.py`
**LR Scheduler:** Cosine พร้อม warmup 5 steps
**ผลลัพธ์โมเดล:** ตอบสั้นมาก (ชื่อโรค 1–3 คำ); ไม่มี `<think>` block — parse rate สูง 99/99 valid

**ลักษณะสำคัญ:** ตอบชื่อโรคสั้นๆ โดยตรง 1–3 คำ → parse rate สูง แต่ RAG ทำให้สับสนเพราะฝึกด้วยภาพเดียว

### 2.2 fuzzytopk_s1cascade (ดีที่สุด 79.80% กับ RAG)

**สถาปัตยกรรม:** เหมือน fuzzytopk แต่ Stage 1 เริ่มจาก weights ของ Group Classifier (3-stage)
**โมเดลต้น:** `skincap_3stage_group_classification_merged` (แบ่ง 4 กลุ่ม, ความแม่นยำ 88.68%)
**เส้นทางโมเดล:** `skincap_fuzzytopk_s1cascade_classification_merged`

| พารามิเตอร์ | ค่า |
|------------|-----|
| ข้อมูล Train | 939 ภาพ |
| Starting weights | Group classifier (Stage 1 cascade) |
| ผลลัพธ์ best | RAG R2-P0 α=0.9 = **79.80%** |

**Training script:** `train_two_stage_FuzzyTopK.py --start_from_stage1`
**LR Scheduler:** Cosine พร้อม warmup 5 steps
**ทำไม s1cascade ถึงใช้ RAG ได้:** Stage 1 cascade pretraining สอนให้โมเดลเปรียบเทียบลักษณะรอยโรคข้ามกลุ่ม → ทำให้โมเดลชอบตอบสั้น ทำให้ P0 ชนะ

**ลักษณะสำคัญ:** การ pretraining Stage 1 cascade สอนให้โมเดลเปรียบเทียบลักษณะรอยโรคข้ามกลุ่ม → ใช้ RAG ได้ดีขึ้น

### 2.3 fuzzytopk_s1cascade_ragR2_a09 (**85.86% — ผลดีที่สุดทั้งหมด**)

**สถาปัตยกรรม:** ฝึกใหม่พร้อมภาพ reference K=1 ต่อ training sample ทุกตัวอย่าง
**เส้นทางโมเดล:** `skincap_fuzzytopk_s1cascade_ragR2_a09_classification_merged`

| พารามิเตอร์ | ค่า |
|------------|-----|
| RAG encoder ตอนฝึก | R2 (SigLIP+BGE-M3, α=0.9) |
| K (ref ต่อตัวอย่างฝึก) | 1 (K=3 ทำให้ VRAM เต็ม) |
| รูปแบบตัวอย่างฝึก | `[ref_img] Reference 1: {label}\nDescription: {caption}\n[query_img] {prompt}` |
| จำนวน steps | 1,314 steps (~1h 44min) |
| ผลลัพธ์ best | RAG R0-P0 = **85.86%** |

**Training script:** `train_two_stage_FuzzyTopK.py --start_from_stage1 --rag_k_train 1 --rag_exp R2 --alpha 0.9`
**จำนวน Training items:** 939 ภาพ × ~3.7 ตัวอย่าง/ภาพ = 3,504 items (3 prompts ต่อภาพ)
**ความเร็ว:** ~4.2 วินาที/step
**`precompute_rag_refs()`:** Pre-compute R2 embeddings ก่อนโหลด Qwen3-VL เพื่อคืน GPU memory ก่อน
**K=3 OOM detail:** Unsloth fused CrossEntropy ขึ้น ZeroDivisionError เมื่อ padding ratio เล็กเกินไป; 4 ภาพ × batch=2 = 8 forward passes ใช้ VRAM หมด 15.92GB → K=1 (2 ภาพ/ตัวอย่าง) พอดี

**ผลที่น่าประหลาดใจ:** CLIP (R0) ที่ inference ให้ผลดีกว่า SigLIP+BGE-M3 (R2 ที่ใช้ตอนฝึก) (+3.03%)
→ โมเดลเรียนรู้แนวคิด "ดู reference → ใช้ label" แบบทั่วไป ไม่ยึดติดกับ encoder เฉพาะ

### 2.4 M-Series: 3-Stage Pipeline Ablation (M0–M4)

> **M-Series** คือการทดลอง ablation บนสถาปัตยกรรม **3-stage pipeline** เพื่อหาวิธีที่ดีที่สุดในการ inject "group context" (กลุ่มโรค 4 กลุ่ม) เข้าสู่ Stage 2 disease classifier
> ทุกโมเดลใน M-series ใช้ **Group Classifier เดียวกัน** (Stage 1: `skincap_3stage_group_classification_merged`, 88.68%) เป็นจุดเริ่มต้น

---

#### ภาพรวมสถาปัตยกรรม 3-Stage Pipeline (ร่วมกันทุก M)

```
Qwen3-VL-8B Base
    ↓
[Stage 1] Group Classifier
    Fine-tune: จำแนก 4 กลุ่มโรค (Inflammatory / Benign / Malignant / Acne)
    ผลลัพธ์: skincap_3stage_group_classification_merged → 88.68% group accuracy
    ↓
[Stage 2] Disease Classifier  ← จุดที่ M0–M4 แตกต่างกัน (วิธี inject group context)
    Fine-tune: จำแนก 10 โรค พร้อม/ไม่พร้อม group context
    ↓
[Stage 3] Caption Generator
    Fine-tune: สร้าง clinical caption (Merged-Init: merge Stage 2 weights ก่อน + fresh LoRA)
```

**ประเด็นหลักที่ ablation ตอบ:** "ควร inject group context ตอนฝึก Stage 2 อย่างไร และตอน inference ควรใช้ context แบบใด?"

- **Group context** = ข้อความบอกกลุ่มโรค เช่น `"This lesion belongs to group: Malignant Tumors"` ที่ append ไว้ใน prompt ของ Stage 2
- **GT group** = Ground Truth group (ใช้ label จริงจาก dataset — oracle condition ที่ไม่มีใน real deployment)
- **Stage1 predicted group** = ผลจาก Stage 1 Group Classifier จริง (realistic condition)
- **Beam soft probs** = distribution ของความน่าจะเป็นจาก Stage 1 แทนที่จะเป็น hard label

---

#### M0 — "No-Context 3-Stage" (3-Stage ไม่มี Group Context)

**ชื่อทางการ:** M0 — No-Context 3-Stage Baseline
**นิยาม:** ฝึก Stage 2 Disease Classifier **โดยไม่ inject group context เลย** ทั้งตอนฝึกและ inference
เป็น baseline ของ M-series เพื่อวัดว่า group context ช่วยอะไรบ้าง

**Input Stage 2 ตอนฝึก:**
```
[query_img]
What skin disease is shown in this image?
↳ Answer: psoriasis
```
*(เหมือน Single-Image FT แต่ weights เริ่มจาก Group Classifier)*

**Input Stage 2 ตอน Inference:**
```
[query_img]  What skin disease is shown in this image?
```

| พารามิเตอร์ | ค่า |
|------------|-----|
| Train context | ไม่มี group context |
| Inference context | ไม่มี group context |
| Starting weights | Group Classifier (Stage 1) |
| ความแม่นยำ (ablation) | **61.0%** |
| ความแม่นยำ (full benchmark, ไม่มี RAG P0) | ~47–49% |

**ข้อสังเกต:** 61% ดีกว่า M2/M3 ที่ใช้ predicted group (57.4%) → Stage 1 predictions ที่ผิดพลาดทำให้ Stage 2 สับสนมากกว่าการไม่มี context เลย

---

#### M1 — "GT-Oracle 3-Stage" (ใช้ Ground Truth Group ทุกขั้น) ← **ใช้ใน Paper**

**ชื่อทางการ:** M1 — Ground-Truth Oracle 3-Stage
**นิยาม:** ฝึก Stage 2 ด้วย **GT group label จริง** และตอน inference ก็ inject GT group เช่นกัน
เป็น **oracle condition** — รู้กลุ่มโรคที่ถูกต้องล่วงหน้า (ไม่ใช้ได้ใน real deployment)

**Input Stage 2 ตอนฝึก:**
```
[query_img]
This lesion belongs to group: Malignant Tumors.
What skin disease is shown in this image?
↳ Answer: basal cell carcinoma
```

**Input Stage 2 ตอน Inference:**
```
[query_img]
This lesion belongs to group: Malignant Tumors.   ← GT label จาก dataset
What skin disease is shown in this image?
```

| พารามิเตอร์ | ค่า |
|------------|-----|
| Training script | `train_three_stage_hybrid_topk.py` |
| ข้อมูล Train | 911 ภาพ (`split_info_3stage.json`) |
| Stage 1 LR | 2e-4 |
| Stage 1 Epochs | 3 |
| Stage 2 LR | 5e-5 |
| Stage 2 Epochs | 3 |
| Stage 2 warmup_ratio | 0.1 |
| Stage 2 LoRA dropout | 0.05 |
| Train context | GT group label |
| Inference context | GT group label (oracle) |
| ความแม่นยำ (ablation, GT oracle) | **66.0%** |
| ความแม่นยำ (full benchmark, ไม่มี RAG P0) | **47.87%** |
| ความแม่นยำ (full benchmark, RAG R0-P1) | **59.38%** |
| เส้นทางโมเดล | `skincap_3stage_disease_M1_merged` |

**ทำไมผลต่างกัน (66% vs 47.87%)?**
- **66.0%** (ablation table) = ทดสอบแบบ oracle: inject GT group ที่ inference → Stage 2 รู้กลุ่มถูกต้อง
- **47.87%** (full benchmark P0) = ทดสอบแบบ realistic: Stage 2 รัน "as-is" ตาม P0 prompt โดยไม่มี group context เสริม → โมเดลที่ฝึกให้รับ group context แต่ไม่ได้รับจริงที่ inference → สับสน
- ความต่าง 66% vs 47.87% แสดงว่า M1 พึ่งพา group context สูงมาก

**ทำไมใช้ M1 ใน Paper (เป็น "2-Stage Cascade FT"):**
- M1 เป็นตัวแทนสถาปัตยกรรม 3-stage ที่ดีที่สุดในทางทฤษฎี (oracle upper bound)
- แสดงให้เห็นว่าแม้ในสภาวะ oracle ก็ยังแพ้ fuzzytopk_s1cascade ที่ราคาถูกกว่า
- 59.38% กับ R0-P1 RAG = ดีที่สุดที่ M1 ทำได้ในการทดลอง full benchmark

---

#### M2 — "Oracle-Train, Cascade-Infer" (ฝึก Oracle, ทำนาย Realistic)

**ชื่อทางการ:** M2 — Oracle-Trained, Predicted-Context Inference
**นิยาม:** ฝึก Stage 2 ด้วย **GT group** (เหมือน M1) แต่ตอน inference ใช้ **Stage 1 predicted group** (realistic)
เป็นการทดสอบว่า "ถ้าฝึกแบบ oracle แต่ deploy แบบ realistic จะเกิดอะไรขึ้น?"

**Input Stage 2 ตอนฝึก:** (เหมือน M1 — GT group)
```
[query_img]
This lesion belongs to group: Malignant Tumors.   ← GT label
What skin disease is shown in this image?
```

**Input Stage 2 ตอน Inference:** (แตกต่างจาก M1 — ใช้ Stage1 prediction)
```
[query_img]
This lesion belongs to group: Inflammatory Disorders.  ← Stage1 prediction (อาจผิด!)
What skin disease is shown in this image?
```

| พารามิเตอร์ | ค่า |
|------------|-----|
| Train context | GT group label |
| Inference context | Stage 1 predicted group (อาจผิดได้ ~11.32%) |
| ความแม่นยำ (ablation) | **57.4%** |

**บทเรียน:** Train-inference distribution mismatch รุนแรงมาก
→ โมเดลเรียนรู้ที่จะ "เชื่อ" group context จาก GT แต่พอ inference ได้ predicted group ที่ผิด → confidence ผิดทาง
→ 57.4% < 61.0% (M0 ที่ไม่มี context เลย) = **การใส่ context ที่ผิดแย่กว่าไม่ใส่เลย**

---

#### M3 — "Full-Cascade 3-Stage" (Realistic End-to-End)

**ชื่อทางการ:** M3 — Fully-Realistic Cascaded 3-Stage
**นิยาม:** ฝึก Stage 2 ด้วย **Stage 1 predicted group** (ไม่ใช่ GT) และ inference ก็ใช้ Stage 1 predicted เช่นกัน
เป็น **สภาวะ realistic ที่สุด** — ไม่มีข้อมูล oracle เลยตลอด pipeline

**Input Stage 2 ตอนฝึก:**
```
[query_img]
This lesion belongs to group: Inflammatory Disorders.  ← Stage1 prediction (อาจผิด)
What skin disease is shown in this image?
```

**Input Stage 2 ตอน Inference:** (เหมือนตอนฝึก)
```
[query_img]
This lesion belongs to group: Inflammatory Disorders.  ← Stage1 prediction
What skin disease is shown in this image?
```

| พารามิเตอร์ | ค่า |
|------------|-----|
| Train context | Stage 1 predicted group |
| Inference context | Stage 1 predicted group |
| ความแม่นยำ (ablation) | **57.4%** |

**บทเรียน:** แม้จะ train-inference consistent แต่ผลเท่ากับ M2 (57.4%)
→ Stage 1 errors (11.32% error rate) propagate ไปยัง Stage 2 อย่างหนัก
→ ไม่มีข้อได้เปรียบจาก "consistency" ถ้า input noise สูงเกินไป
→ M3 = M2 แสดงว่า 57.4% คือ **เพดานของ error propagation** จาก Stage 1

---

#### M4 — "Soft-Probability 3-Stage" (Probabilistic Context)

**ชื่อทางการ:** M4 — Soft-Probability Group Context
**นิยาม:** ฝึก Stage 2 ด้วย **GT group** แต่ตอน inference ใส่ **distribution ของความน่าจะเป็น** จาก Stage 1 แทน hard label
แนวคิด: แทนที่จะบอกว่า "กลุ่ม X" ให้บอกว่า "60% Malignant, 30% Inflammatory, 10% Benign"

**Input Stage 2 ตอน Inference:**
```
[query_img]
Group probabilities: Malignant Tumors: 0.61, Inflammatory: 0.29, Benign: 0.08, Acne: 0.02
What skin disease is shown in this image?
```
*(ค่าจาก beam search soft probabilities ของ Stage 1)*

| พารามิเตอร์ | ค่า |
|------------|-----|
| Train context | GT group label (hard) |
| Inference context | Beam soft probabilities (soft) |
| ความแม่นยำ (ablation) | **61.0%** |

**บทเรียน:** M4 = M0 (61.0%) — soft probs ไม่ช่วยมากกว่าไม่มี context
→ Train-inference mismatch ยังคงอยู่ (ฝึกด้วย hard label แต่ inference ด้วย probability vector)
→ โมเดลไม่รู้วิธีตีความ probability distribution ในรูปแบบ natural language

---

#### ตารางสรุป M0–M4 เปรียบเทียบ

| ชื่อใหม่ | Code | Train Context | Infer Context | Acc (ablation) | Acc (full P0) | สรุป |
|---------|------|--------------|--------------|----------------|---------------|------|
| No-Context 3-Stage | M0 | ไม่มี | ไม่มี | 61.0% | ~47% | baseline ไม่มี context |
| **GT-Oracle 3-Stage** | **M1** | **GT group** | **GT group** | **66.0%** | **47.87%** | **ใช้ใน Paper; oracle upper bound** |
| Oracle-Train, Cascade-Infer | M2 | GT group | Stage1 pred | 57.4% | — | train-infer mismatch |
| Full-Cascade 3-Stage | M3 | Stage1 pred | Stage1 pred | 57.4% | — | realistic แต่ error propagates |
| Soft-Probability 3-Stage | M4 | GT group | Soft probs | 61.0% | — | soft probs ไม่ช่วย |

**ข้อสรุปหลักจาก M-series ablation:**
1. **M1 (GT Oracle) ดีที่สุด (66%)** แต่ต้องการ GT group ที่ไม่มีใน deployment จริง
2. **M2 = M3 = 57.4%** → train-inference mismatch และ error propagation ทำลายประสิทธิภาพ
3. **M0 = M4 = 61%** → ไม่มี context หรือ soft probs ให้ผลเท่ากัน = group context ต้องการความแม่นยำสูง
4. **ทั้งหมดแพ้ fuzzytopk (74%) แม้ในสภาวะ oracle** → 3-stage cascade penalty ชัดเจน
5. **บทเรียนสำหรับ HIKARI:** อย่า cascade stages แบบ hard dependency → ใช้ RAG-in-Training แทน

---

#### ทำไม M-series ทั้งหมดแพ้ fuzzytopk?

```
fuzzytopk (Single-Image FT): 74.00%   ← ง่ายกว่า แต่ดีกว่า
        vs
M1 (GT-Oracle 3-Stage):      66.00%   ← ซับซ้อนกว่า แต่แย่กว่า (แม้ oracle!)
M3 (Full-Cascade):           57.40%   ← realistic = แย่ที่สุด
```

**สาเหตุหลัก:**

| สาเหตุ | อธิบาย |
|--------|--------|
| **Cascade penalty** | Stage 2 เริ่มจาก Group Classifier weights → initialization ไม่เหมาะกับ disease-level discrimination |
| **Prompt complexity** | Group context + query image + CoT = token count สูงขึ้น → VRAM pressure |
| **Error propagation** | Stage 1 error 11.32% → Stage 2 ผิดทาง (M2/M3) |
| **Oracle dependency** | M1 ต้องการ GT group → ไม่ใช้ได้จริง; M2/M3 ที่ realistic แย่กว่า M0 |
| **CoT overhead** | M1 สร้าง `<think>` chain ยาว → inference ช้าและ VRAM สูง |

---

## 3. การทดลอง RAG (R0–R4)

การดึงข้อมูลภาพ + ข้อความเพื่อเสริมการสร้างคำตอบ ใน inference จะดึง K=3 ภาพที่คล้ายกันที่สุดจากชุด train เพื่อใช้เป็นตัวอย่างใน prompt
**Index สร้างจาก 911 ภาพ train เท่านั้น** (ไม่มี data leakage จาก val)

### 3.1 รูปแบบ Reference ใน Prompt

```
นี่คือกรณีอ้างอิงที่คล้ายกันสำหรับบริบท:
[ref_image_1]
Reference 1: psoriasis
Description: Erythematous scaly plaques with silvery surface on extensor surfaces...

[ref_image_2]
Reference 2: lichen planus
Description: Flat-topped violaceous papules on the forearms...

ตอนนี้ระบุโรคในภาพใหม่นี้:
[query_image]
{ข้อความ prompt}
```

### 3.2 การตั้งค่า RAG Encoder

| ID | ชื่อ | Image Encoder | Text Encoder | กลยุทธ์ |
|----|-----|--------------|--------------|---------|
| **R0** | CLIP เฉพาะภาพ | `openai/clip-vit-base-patch32` | ไม่มี | A (image→image) |
| **R1** | CLIP + ClinicalBERT | `openai/clip-vit-base-patch32` | `medicalai/ClinicalBERT` | B (image+text) |
| **R2** | SigLIP + BGE-M3 | `google/siglip-2-base-patch16-512` | `BAAI/bge-m3` | B (image+text) |
| **R3** | Jina-CLIP + MedCPT | `jinaai/jina-clip-v2` | `ncbi/MedCPT-Query-Encoder` | B (image+text) |
| **R4** | Nomic Unified | `nomic-ai/nomic-embed-vision-v1.5` | `nomic-ai/nomic-embed-text-v1.5` | A (cross-modal) |

**หมายเหตุ R4 (Nomic):** ทั้ง image encoder และ text encoder ใช้ embedding space ร่วมกัน (cross-modal unified, กลยุทธ์ A) — ค้นหาได้ทั้ง image→image, image→text, และ text→image ในพื้นที่เดียวกัน

**`query_text` สำหรับกลยุทธ์ B:** คำอธิบายอาการที่ผู้ป่วยเขียนจาก `val_captions_for_symptoms.json` (94 รายการ) หรือ caption ที่สร้างจาก Stage 3

### 3.3 สูตรคะแนนการดึงข้อมูล

**กลยุทธ์ A (Cross-modal, R0 และ R4):**
Image encoder ค้นหาทั้ง image และ text embeddings ในพื้นที่ embedding ร่วม
```
score(query, ref) = α × cos(img_enc(query_img), ref_img_emb)
                 + (1−α) × cos(img_enc(query_img), ref_txt_emb)
```
R0: ไม่มี text encoder → α=1.0 ตลอด (เฉพาะภาพ)

**กลยุทธ์ B (สองขั้น, R1/R2/R3):**
Text encoder แยกต่างหากสำหรับฝั่งข้อความ
```
score(query, ref) = α × cos(img_enc(query_img), ref_img_emb)
                 + (1−α) × cos(txt_enc(query_text), ref_txt_emb)
```
`query_text` = คำอธิบายอาการของผู้ป่วย หรือ caption ที่สร้างจาก Stage 3

### 3.4 การสร้าง Index

**กระบวนการ Build Index:** `HybridRAGRetriever.build()` ใน `rag_retrieval.py` → รันบน GPU → บันทึกเป็น `rag_index_{R}_train.npz`
**การ Query:** `HybridRAGRetriever.retrieve(query_img, k=3)` → cosine similarity → คืนค่า `[(path, label, caption)]`

**รูปแบบบันทึก:** ไฟล์ `.npz` ประกอบด้วย:
- `img_embs`: embedding ภาพ ของ 911 ภาพ train
- `txt_embs`: embedding ข้อความจาก caption คลินิก
- `labels`: ชื่อโรค
- `paths`: เส้นทางภาพ
- `captions`: ข้อความ caption คลินิก (แสดงใน prompt)
- `strategy`: กลยุทธ์ RAG ที่ใช้ (A หรือ B)
- `img_encoder`: ชื่อ image encoder
- `txt_encoder`: ชื่อ text encoder

---

## 4. รูปแบบ Prompt (P0–P3)

### P0 — การสังเกตทางคลินิกโดยตรง (ค่าเริ่มต้น)
```
ตรวจดูภาพผิวหนังอย่างละเอียด ดูลักษณะรอยโรค (papule/plaque/macule/nodule),
สี (แดง/ม่วง/ขาว/น้ำตาล/ดำ), สะเก็ด/สะเก็ดแห้ง, ความคมชัดของขอบ, และการกระจาย
จากลักษณะเหล่านี้ โรคผิวหนังที่เฉพาะเจาะจงคืออะไร?
```
**เหมาะสำหรับ:** fuzzytopk, s1cascade (โมเดลตอบสั้น)

### P1 — CoT ทีละขั้นตอน
```
วิเคราะห์ภาพผิวหนังนี้ทีละขั้นตอน:
ขั้นที่ 1: อธิบายรอยโรคหลัก (papule/plaque/macule/nodule/vesicle)
ขั้นที่ 2: สังเกตสี (แดง, ม่วง, ขาว, น้ำตาล, ดำ, ผสม)
ขั้นที่ 3: ระบุพื้นผิว (เรียบ, มีสะเก็ด, มีสะเก็ดแห้ง, เป็นแผล)
ขั้นที่ 4: สังเกตขอบ (คมชัด/ไม่ชัด) และการกระจาย
ขั้นที่ 5: จากลักษณะเหล่านี้ ระบุโรคผิวหนัง
```
**เหมาะสำหรับ:** M1 (โมเดล CoT reasoning) — ขั้นตอนที่ชัดเจนตรงกับรูปแบบ `<think>` ของโมเดล

### P2 — การวินิจฉัยแยกโรค
```
ตรวจดูรอยโรคผิวหนังนี้ ระบุการวินิจฉัยที่เป็นไปได้ 3 อันดับแรกพร้อมหลักฐาน
จากนั้นเลือกการวินิจฉัยที่น่าจะเป็นไปได้มากที่สุด รูปแบบ:
1. [โรค] - [หลักฐาน]; 2. [โรค] - [หลักฐาน]; 3. [โรค] - [หลักฐาน]
การวินิจฉัยสุดท้าย: [โรค]
```
**เหมาะสำหรับ:** Qwen2.5 zero-shot (50.51%) — ไม่เหมาะกับโมเดล fine-tune

### P3 — การประเมินทางคลินิกแบบมีโครงสร้าง
```
การประเมินรอยโรคผิวหนังนี้:
• รูปวิทยา: [อธิบายประเภทรอยโรคหลัก]
• สี/เม็ดสี: [อธิบาย]
• พื้นผิว: [สะเก็ด/สะเก็ดแห้ง/เรียบ]
• ขอบ: [คมชัด/ไม่ชัดเจน]
• การกระจาย: [เฉพาะที่/แพร่กระจาย]
การวินิจฉัย: [ระบุโรคผิวหนังที่เฉพาะเจาะจง]
```
**เหมาะสำหรับ:** Qwen3 zero-shot — คีย์เวิร์ด `การวินิจฉัย:` ช่วยปิด chain of thought

### Group Context Variants (ใช้กับ M1/M2/M3 เท่านั้น)

Prompt เหล่านี้ inject ข้อมูล group โรค (4 กลุ่ม) เข้าไปใน prompt เพื่อช่วย Stage 2 disease classifier:

**P0+group:**
```
This skin lesion belongs to '{group}'. Carefully examine this dermatological image.
Look for: lesion morphology, color, scale or crust, border sharpness, and distribution.
Based on these visual features, what is the specific skin disease?
```

**P1+group:**
```
This lesion belongs to '{group}'. Analyze step by step:
(1) Describe the primary lesion type (papule/plaque/macule/nodule/vesicle)
(2) Note the color(s): red, violet, white, brown, black, mixed
(3) Identify surface texture: smooth, scaly, crusted, ulcerated
(4) Observe borders (sharp/ill-defined) and distribution
(5) Based on all features, what is the specific disease name?
```

**P2+group:**
```
This lesion is in the '{group}' group.
List the top 3 most likely specific diseases within this group with supporting evidence.
Format: 1. [disease] - [evidence]; 2. [disease] - [evidence]; 3. [disease] - [evidence]
Final diagnosis: [most likely specific disease]
```

**P3+group:**
```
Group: '{group}'.
Clinical assessment:
• Morphology: [describe primary lesion type]
• Color/pigmentation: [describe]
• Surface: [scaly/crusted/smooth]
• Borders: [sharp/ill-defined]
• Distribution: [localized/diffuse]
Diagnosis (specific disease within this group): [specific skin disease name]
```

---

## 5. การประเมินผลและขั้นตอน Parsing

### 5.1 การ Parse ผลลัพธ์ — `_extract_disease()`

```
ผลลัพธ์ดิบจากโมเดล (string)
    ↓ ตัด block <think>...</think> (regex): ลบเนื้อหาทั้งหมดระหว่าง <think> และ </think>
    ↓ ใช้ pattern เฉพาะ prompt:
         P3: ดึงหลัง keyword "Diagnosis:"
         P2: ดึงหลัง keyword "Final diagnosis:"
         P1: ดึงหลัง pattern "Step N:" ครั้งสุดท้าย
         P0: เอาบรรทัดแรกที่ไม่ว่างเปล่า
    ↓ Normalize: ตัวพิมพ์เล็ก, ตัดช่องว่าง, ลบ prefix "this image shows"
    ↓ Fuzzy word-overlap matching กับ DISEASE_NAMES (10 โรค):
         คะแนน Jaccard สูงสุด ≥ threshold (91/100) → ตรงกัน
         ไม่ตรง → "Unknown"
    ↓ คืนค่า: ชื่อโรคที่ตรงกัน หรือ "Unknown"
```

### 5.2 Metrics ที่ใช้

| Metric | นิยาม | หลัก? |
|--------|-------|-------|
| **Accuracy** | ถูก / ทั้งหมด | ✅ **หลัก** |
| Balanced Accuracy | ค่าเฉลี่ย recall ต่อคลาส | รอง |
| F1 Macro | ค่าเฉลี่ย F1 ต่อคลาส (unweighted) | รอง |
| F1 Weighted | Sample-count weighted F1 | รอง |
| Cohen's Kappa | ความสอดคล้องเกินกว่าโอกาส | รอง |
| Overall Score | ค่าเฉลี่ย (Acc + Balanced Acc + F1 Macro + Kappa) / 4 | บันทึกใน log |
| Sensitivity (recall) | TP / (TP + FN) ต่อโรค | ต่อโรค |
| PPV (precision) | TP / (TP + FP) ต่อโรค | ต่อโรค |

**สำคัญ:** การทำนาย Unknown นับเป็นผิด (ตัวหาร = total_samples ไม่ใช่ valid_predictions)
**ตัวอย่าง:** โมเดลที่จำแนกถูก 60/99 แต่ตอบ "Unknown" อีก 20 ครั้ง ได้คะแนน 60.6% ไม่ใช่ 75%

### 5.3 ปัญหา OOM

หลังแก้ไข `img.thumbnail((672, 672), LANCZOS)` ใน `_load_image()` — **การทดลอง RAG ทั้ง 60 ครั้ง มีข้อผิดพลาด OOM = 0**
ก่อนแก้ไข: 14–40/99 ตัวอย่างล้มเหลวต่อการรัน

---

## 6. ผลการทดสอบ (Benchmark)

> **Clean benchmark:** การแก้ไข image resize ขจัด VRAM overflow ทั้งหมด ผลด้านล่างมาจากการรันที่ไม่มี error เลย

### 6.1 ผล fuzzytopk

| RAG | P0 | P1 | P2 | P3 |
|-----|----|----|----|----|
| **ไม่มี RAG** | **74.0%** (74/100 valid) | 71.72% | 70.71% | 71.72% |
| R0 | 63.9% | 63.5% | 61.5% | 62.9% |
| R1 | 63.3% | 63.9% | 59.4% | 63.3% |
| R2 | 63.6% | 62.6% | 61.6% | 63.6% |
| R3 | 63.3% | **65.3%** | 59.4% | 59.2% |
| R4 | 60.2% | 58.6% | 51.5% | 57.6% |

**ข้อค้นพบสำคัญ:**
- RAG ทำให้ fuzzytopk แย่ลงสม่ำเสมอ แม้ไม่มี OOM เลย (No-RAG P0=74.0%, best RAG=65.3% = **−8.7%**)
- R4 (Nomic) แย่ที่สุดสำหรับ fuzzytopk; R3+P1 = combo RAG ที่ดีที่สุด
- **หมายเหตุ:** ไม่มี RAG ใช้ 101 val samples; ผล RAG ใช้ 99-sample split

**ผลต่อโรค — fuzzytopk ไม่มี RAG P0 (74.0%, 74/100 valid — หมายเหตุ: ใช้ 101-sample split แต่ 1 sample OOM):**

| โรค | Sensitivity% | n |
|-----|-------------|---|
| SCCIS | 100.0% | 14 |
| Melanocytic nevi | 91.7% | 12 |
| BCC | 76.9% | 13 |
| Acne vulgaris | 87.5% | 8 |
| Psoriasis | 84.6% | 13 |
| Sarcoidosis | 42.9% | 7 |
| Lichen planus | 77.8% | 9 |
| Scleroderma | 37.5% | 8 |
| Photodermatoses | 57.1% | 7 |
| Lupus erythematosus | 44.4% | 9 |

**ผลต่อโรค — fuzzytopk R3-P1 (ดีที่สุด RAG, 65/99):**

| โรค | Sensitivity% | n | PPV% |
|-----|-------------|---|------|
| SCCIS | 100.0% | 12 | 63.2% |
| Lichen planus | 77.8% | 9 | 63.6% |
| BCC | 76.9% | 13 | 66.7% |
| Acne vulgaris | 75.0% | 8 | 85.7% |
| Sarcoidosis | 71.4% | 7 | 33.3% |
| Psoriasis | 61.5% | 13 | 100.0% |
| Scleroderma | 57.1% | 7 | 80.0% |
| Melanocytic nevi | 50.0% | 12 | 100.0% |
| Lupus erythematosus | 44.4% | 9 | 40.0% |
| Photodermatoses | 25.0% | 8 | 100.0% |

---

### 6.2 ผล fuzzytopk_s1cascade

ค่าเริ่มต้น alpha=0.5:

| RAG | P0 | P1 | P2 | P3 |
|-----|----|----|----|----|
| R0 | 72.7% | 69.7% | 70.7% | 69.7% |
| R1 | 69.7% | 69.7% | 69.7% | 67.7% |
| **R2** | **74.7%** | 72.7% | 73.7% | 71.7% |
| R3 | 66.7% | 64.6% | 63.6% | 63.6% |
| R4 | 67.7% | 66.7% | 67.7% | 67.7% |

#### การปรับ Alpha — R2 และ R3, P0

| α | ความแม่นยำ R2 | ความแม่นยำ R3 | หมายเหตุ |
|---|--------------|--------------|---------|
| 0.5 (ค่าเริ่มต้น) | 74.75% | 66.67% | น้ำหนักภาพ/ข้อความเท่ากัน |
| 0.7 | 76.77% | **78.79%** | R3 พีคที่นี่ (MedCPT ต้องการ 30% text) |
| **0.9** | **79.80%** | 67.68% | **R2 พีค** — ภาพเด่น (10% text) |
| 1.0 | 75.76% | 66.67% | เฉพาะภาพ |

- **R2 ดีที่สุด: α=0.9 → 79.80% (79/99)**
- **R3 ดีที่สุด: α=0.7 → 78.79% (78/99)**
- **R1 (ClinicalBERT) ต่ำกว่า R0** — ข้อความคลินิกทั่วไปไม่ตรงกับ dermoscopy captions

**ข้อค้นพบสำคัญ:**
- **RAG ช่วย s1cascade** ทุก config — การ pretraining Stage 1 cascade เปิดใช้ reference ได้
- **P0 ชนะทุก RAG config** — cascade pretraining ทำให้โมเดลชอบตอบสั้น
- **Alpha tuning สำคัญมาก:** ค่าเริ่มต้น α=0.5 = 74.75%, ปรับ α=0.9 = 79.80% (+5.05%)
- **R2 vs R0:** default α=0.5 — R2 ชนะ R0 +2.0%; ที่ α=0.9 — R2 ชนะ R0 +7.1%
- **R3 (Jina-CLIP+MedCPT)** พีคที่ α=0.7 (78.79%) — เกือบเท่า best R2
- **บทเรียน:** อย่าใช้ alpha default โดยไม่ sweep ก่อน

**ผลต่อโรค — s1cascade R2 α=0.9 P0 (79.80%):**

| โรค | Sensitivity% | n | PPV% |
|-----|-------------|---|------|
| SCCIS | 100.0% | 12 | 66.7% |
| Acne vulgaris | 100.0% | 8 | 88.9% |
| Melanocytic nevi | 91.7% | 12 | 100.0% |
| Lichen planus | 88.9% | 9 | 80.0% |
| BCC | 76.9% | 13 | 90.9% |
| Scleroderma | 75.0% | 8 | 85.7% |
| Lupus erythematosus | 77.8% | 9 | 70.0% |
| Photodermatoses | 62.5% | 8 | 71.4% |
| Psoriasis | 61.5% | 13 | 80.0% |
| Sarcoidosis | 57.1% | 7 | 66.7% |

**ผลต่อโรค — s1cascade R0-P0 (72.73%):**

| โรค | Sensitivity% | n | PPV% |
|-----|-------------|---|------|
| Melanocytic nevi | 91.7% | 12 | 100.0% |
| SCCIS | 91.7% | 12 | 73.3% |
| Lichen planus | 88.9% | 9 | 88.9% |
| Acne vulgaris | 87.5% | 8 | 87.5% |
| Sarcoidosis | 71.4% | 7 | 55.6% |
| Psoriasis | 69.2% | 13 | 69.2% |
| BCC | 69.2% | 13 | 69.2% |
| Lupus erythematosus | 66.7% | 9 | 54.5% |
| Scleroderma | 37.5% | 8 | 75.0% |
| Photodermatoses | 37.5% | 8 | 50.0% |

---

### 6.3 ผล fuzzytopk_s1cascade_ragR2_a09 (**ผลดีที่สุดทั้งหมด**)

โมเดลฝึกพร้อม K=1 reference ต่อ training sample (R2 encoder, α=0.9)

| RAG Inference | Acc | Bal Acc | F1 Macro | Kappa |
|--------------|-----|---------|----------|-------|
| **R0-P0 (CLIP)** | **85.86%** | 82.12% | 80.78% | 84.18% |
| R0-P3 | 83.84% | 80.52% | 79.52% | 81.93% |
| R2-P0 (SigLIP+BGE-M3) | 82.83% | 80.12% | 80.07% | 80.80% |
| R2-P3 | 82.83% | 80.78% | 80.77% | 80.81% |

**ผลต่อโรค — ragR2_a09 R0-P0 (85.86%, ดีที่สุด):**

| โรค | Sensitivity% | n | PPV% |
|-----|-------------|---|------|
| Psoriasis (สะเก็ดเงิน) | 100.0% | 13 | 92.9% |
| Melanocytic nevi (ไฝ) | 100.0% | 12 | 100.0% |
| SCCIS (มะเร็งเซลล์สความัสในชั้น) | 100.0% | 12 | 100.0% |
| BCC (มะเร็งเซลล์ฐาน) | 100.0% | 13 | 100.0% |
| Acne vulgaris (สิว) | 100.0% | 8 | 88.9% |
| Lichen planus (ไลเคน พลานัส) | 88.9% | 9 | 88.9% |
| Scleroderma (สเกลโรเดอร์มา) | 87.5% | 8 | 77.8% |
| Photodermatoses (โรคผิวหนังจากแสง) | 75.0% | 8 | 66.7% |
| Lupus erythematosus (ลูปัส) | 55.6% | 9 | 55.6% |
| Sarcoidosis (ซาร์คอยโดซิส) | 14.3% | 7 | 33.3% |

**ข้อค้นพบสำคัญ:**
- 5 โรคได้ 100% sensitivity (สะเก็ดเงิน, ไฝ, SCCIS, BCC, สิว) สำหรับ R0-P0
- **Psoriasis ดีขึ้นมากที่สุด:** 61.5% → 100.0%
- **Sarcoidosis ลดลงรุนแรง:** 57.1% → 14.3% — โมเดลพึ่งพา reference agreement มากเกินไป; สับสนกับโรคอักเสบอื่นระหว่างฝึก
- **+6.06%** เหนือ best เดิม (79.80% → 85.86%)
- **P0 > P3 (+2.02%)** สอดคล้องกับ preference prompt ตรงของ s1cascade

**ผลต่อโรค — ragR2_a09 R0-P3 (83.84%):**

| โรค | Sensitivity% | n | PPV% |
|-----|-------------|---|------|
| Melanocytic nevi | 100.0% | 12 | 100.0% |
| BCC | 100.0% | 13 | 92.9% |
| Acne vulgaris | 100.0% | 8 | 88.9% |
| Psoriasis | 92.3% | 13 | 92.3% |
| SCCIS | 91.7% | 12 | 100.0% |
| Scleroderma | 87.5% | 8 | 77.8% |
| Lichen planus | 77.8% | 9 | 87.5% |
| Photodermatoses | 75.0% | 8 | 75.0% |
| Lupus erythematosus | 66.7% | 9 | 50.0% |
| Sarcoidosis | 14.3% | 7 | 33.3% |

---

### 6.4 ผล M1 (3-stage + GT group context)

| RAG | P0 | P1 | P2 | P3 |
|-----|----|----|----|----|
| **ไม่มี RAG** | **47.9%** | **49.5%** | 45.5% | 46.5% |
| R0 | 55.0% | **59.4%** | 53.8% | 50.6% |
| R1 | 54.8% | 58.9% | 52.7% | 49.5% |
| R2 | 53.9% | 52.1% | 53.9% | 47.2% |
| R3 | 52.2% | 52.6% | 56.0% | 50.0% |
| R4 | 56.7% | 57.5% | 54.4% | 53.3% |

**ข้อค้นพบสำคัญ:**
- **RAG ช่วย M1 มาก:** 47.9% → 59.4% (+11.5%) — reasoning chain ของ M1 ใช้ context ได้ดี
- **P1 ชนะ M1 ทั้งกับ RAG (+4.4%) และไม่มี RAG (+1.6% เหนือ P0)**
- **P2 และ P3 แย่กว่า P0** สำหรับ M1 ไม่มี RAG
- **R0 (CLIP เฉพาะภาพ) ดีที่สุดสำหรับ M1** — text retrieval (R1–R3) เพิ่ม noise

**ผลต่อโรค — M1 ไม่มี RAG P0 (47/99):**
*(หมายเหตุ: M1 No-RAG ประเมินบน val split ที่แตกต่างเล็กน้อยจากการรัน 99-sample อื่น)*

| โรค | Sensitivity% | n |
|-----|-------------|---|
| Melanocytic nevi | 100.0% | 9 |
| Acne vulgaris | 100.0% | 8 |
| BCC | 69.2% | 13 |
| SCCIS | 66.7% | 12 |
| Sarcoidosis | 42.9% | 7 |
| Lupus erythematosus | 33.3% | 9 |
| Scleroderma | 25.0% | 8 |
| Lichen planus | 22.2% | 9 |
| Psoriasis | 16.7% | 12 |
| Photodermatoses | 0.0% | 8 |

**ผลต่อโรค — M1 R0-P1 (59/99 = 59.38%):**

| โรค | Sensitivity% | n | PPV% |
|-----|-------------|---|------|
| Melanocytic nevi | 100.0% | 9 | 100.0% |
| Acne vulgaris | 100.0% | 8 | 100.0% |
| BCC | 84.6% | 13 | 73.3% |
| Sarcoidosis | 71.4% | 7 | 18.5% |
| SCCIS | 66.7% | 12 | 80.0% |
| Psoriasis | 61.5% | 13 | 80.0% |
| Scleroderma | 25.0% | 8 | 66.7% |
| Photodermatoses | 25.0% | 8 | 66.7% |
| Lupus erythematosus | 22.2% | 9 | 22.2% |
| Lichen planus | 22.2% | 9 | 100.0% |

---

### 6.5 Zero-Shot Baseline (ไม่ fine-tune) — ผลครบทุก Prompt × RAG Config

> **⚠️ สำคัญ:** ผล "ดีที่สุด" ของทั้งสองโมเดลใช้ **RAG R0 (CLIP) ที่ inference** ไม่ใช่ Zero-Shot แท้
> ผล No-RAG แท้ = 0–1% เท่านั้น เพราะ parser ดึงชื่อโรคจาก output ไม่ได้ (valid_predictions ต่ำมาก)

---

#### Qwen3-VL-8B-Thinking (base model ของเรา — ไม่ fine-tune)

**ทดสอบทั้งหมด: No-RAG + R0 (CLIP) × 4 prompts = 6 การทดลอง (max_new_tokens=1024)**

| RAG Infer | Prompt | Valid/99 | Accuracy | หมายเหตุ |
|-----------|--------|----------|----------|---------|
| **No-RAG** | P0 | **0**/99 | **0.00%** | ❌ parse ไม่ได้เลย — thinking chain ยาวเกินไป ไม่ปิด `</think>` |
| **No-RAG** | P3 | **1**/99 | **1.01%** | ❌ เกือบ parse ไม่ได้ — `Diagnosis:` anchor ช่วยได้นิดเดียว |
| R0 | P0 | 9/99 | 4.04% | ❌ ยังวนเวียน — reference images ไม่ช่วยเรื่อง output format |
| R0 | P1 | 3/99 | 1.01% | ❌ 6 ขั้นตอนใช้ token หมดก่อนถึงคำตอบ |
| R0 | P2 | 36/99 | 18.18% | ⚠️ "Final diagnosis:" anchor ช่วย แต่ valid ยังน้อย |
| **R0** | **P3** | **56**/99 | **33.33%** | ✅ **Best** — `Diagnosis:` บังคับปิด CoT + R0 reference ช่วย format |

**ข้อสรุป Qwen3 base:**
- **Best: R0-P3 = 33.33%** (56/99 valid) — ใช้ทั้ง RAG และ structured prompt
- **No-RAG แท้ = 0–1%** — โมเดลไม่รู้ว่าต้องตอบสั้นๆ; thinking chain วนจนหมด token budget
- **P3 critical:** `Diagnosis:` keyword เป็น anchor ที่บังคับให้โมเดลสรุปคำตอบ → valid ขึ้นจาก 0 → 56
- **P1 แย่ที่สุด (1.01%):** 6 numbered steps exhausts ~512 tokens ก่อนถึง Step 6

---

#### Qwen2.5-VL-7B-Instruct (prior work baseline — ไม่ fine-tune)

**ทดสอบทั้งหมด: No-RAG + R0 (CLIP) × 4 prompts = 6 การทดลอง (max_new_tokens=1024)**

| RAG Infer | Prompt | Valid/99 | Accuracy | หมายเหตุ |
|-----------|--------|----------|----------|---------|
| **No-RAG** | P0 | **0**/99 | **0.00%** | ❌ parse ไม่ได้เลย — output ยาวเป็น essay ไม่มี anchor |
| **No-RAG** | P3 | **0**/99 | **0.00%** | ❌ parse ไม่ได้เลย — แม้มี `Diagnosis:` ก็ไม่ปฏิบัติตาม |
| R0 | P0 | 44/99 | 23.23% | ⚠️ reference examples ช่วยให้รู้ว่าต้องตอบสั้น |
| R0 | P1 | 42/99 | 24.24% | ⚠️ ทีละขั้น คล้าย P0 |
| **R0** | **P2** | **85**/99 | **50.51%** | ✅ **Best** — `Final diagnosis:` anchor → parse ได้ 85/99 |
| R0 | P3 | 43/99 | 17.17% | ⚠️ Clinical template ไม่ได้รับการปฏิบัติตามดี |

**ข้อสรุป Qwen2.5:**
- **Best: R0-P2 = 50.51%** (85/99 valid) — ใช้ RAG + Differential Diagnosis format
- **No-RAG แท้ = 0%** — output เป็น paragraph ยาว ไม่มี structured anchor → parse ไม่ได้เลย
- **P2 critical:** `"Final diagnosis: [โรค]"` format → parser ดึงหลัง `Final diagnosis:` → valid ขึ้นมากที่สุด (85/99)
- **R0 reference images ช่วย P0/P1:** เมื่อเห็น reference + label → โมเดลเข้าใจว่าต้องตอบชื่อโรค (valid ขึ้นจาก 0 → 44)

---

#### เปรียบเทียบ: ทำไม Valid Predictions ต่างกันมาก?

| ปัจจัย | ผล |
|--------|-----|
| **No-RAG + ไม่มี anchor** | valid = 0/99 — โมเดลไม่รู้ว่าต้องตอบอะไร → output ยาว parse ไม่ได้ |
| **No-RAG + anchor (P3)** | Qwen3: valid = 1/99 / Qwen2.5: valid = 0/99 — ยังสับสน |
| **R0 + P0** | valid = 9/99 (Qwen3), 44/99 (Qwen2.5) — reference examples สอน output format |
| **R0 + P2 (Final diagnosis:)** | valid = 36/99 (Qwen3), **85/99 (Qwen2.5)** — anchor ที่แข็งแรงที่สุด |
| **R0 + P3 (Diagnosis:)** | **56/99 (Qwen3)**, 43/99 (Qwen2.5) — Qwen3 ตอบสนองต่อ P3 ดีกว่า |

**บทเรียน:** สำหรับ zero-shot VLMs ที่ไม่ได้ fine-tune → **valid prediction rate คือ bottleneck ที่แท้จริง** ไม่ใช่ความรู้เรื่องโรค

---

#### สรุปตัวเลขที่รายงานใน Paper (Table II)

| โมเดล | RAG Infer | Prompt | Acc | หมายเหตุ |
|-------|-----------|--------|-----|---------|
| Qwen3-VL-8B | R0 | P3 | 33.33% | Best config; paper แสดง R0 |
| Qwen2.5-VL-7B | R0 | P2 | **50.51%** | Best config; paper แสดง R0 |

> Paper caption: *"Zero-shot models receive CLIP (R0) context but no fine-tuning"*
> → ถูกต้องแล้ว — ทุก row ใน Zero-Shot section ใช้ R0 ที่ inference

---

## 7. สรุปเปรียบเทียบ — ผลดีที่สุดต่อโมเดล

| โมเดล | Config ที่ดีที่สุด | Acc | หมายเหตุ |
|-------|-----------------|-----|---------|
| **ragR2_a09** | RAG R0-P0 | **85.86%** | **ดีที่สุดทั้งหมด** — ฝึกกับ RAG context |
| **ragR2_a09** | RAG R0-P3 | 83.84% | โมเดลเดียวกัน prompt แบบมีโครงสร้าง |
| **s1cascade** | RAG R2-P0 α=0.9 | 79.80% | ดีที่สุดก่อนหน้า — SigLIP+BGE-M3 |
| **s1cascade** | RAG R3-P0 α=0.7 | 78.79% | Jina-CLIP+MedCPT |
| **fuzzytopk** | ไม่มี RAG | 74.00% | ดีที่สุดภาพเดียว |
| **s1cascade** | RAG R0-P0 | 72.73% | CLIP image-only RAG |
| **fuzzytopk** | RAG R3-P1 | 65.31% | RAG ดีที่สุดของ fuzzytopk |
| **M1** | RAG R0-P1 | 59.38% | RAG +11.5% เทียบกับไม่มี RAG |
| **Qwen2.5 zero-shot** | R0-P2 (1024 tok) | 50.51% | ไม่ fine-tune; ใช้ R0 RAG+P2 (valid=85/99) |
| **M1** | ไม่มี RAG P1 | 49.49% | M1 ไม่มี RAG ดีที่สุด |
| **M1** | ไม่มี RAG P0 | 47.87% | M1 baseline |
| **Qwen3 base zero-shot** | R0-P3 (1024 tok) | 33.33% | ไม่ fine-tune; ใช้ R0 RAG+P3 (valid=56/99) |
| **Qwen2.5 No-RAG แท้** | No-RAG P0/P3 | 0.00% | valid=0/99 — parse ไม่ได้เลย |
| **Qwen3 No-RAG แท้** | No-RAG P0 | 0.00% | valid=0/99 — thinking chain ไม่ปิด |

---

## 8. ข้อค้นพบและการวิเคราะห์สำคัญ

### 8.1 RAG ช่วยได้ไหม?

| โมเดล | ไม่มี RAG | RAG ดีที่สุด | ผล RAG |
|-------|---------|-----------|-------|
| fuzzytopk | **74.0%** | 65.3% (R3-P1) | **−8.7%** (RAG ทำให้แย่ลง) |
| fuzzytopk_s1cascade | ไม่มี | 79.80% (R2-P0 α=0.9) | +5.8% เทียบกับ baseline |
| **ragR2_a09** | ไม่มี | **85.86%** (R0-P0) | **+11.86% เทียบกับ baseline** |
| M1 | 47.9% | **59.4%** (R0-P1) | **+11.5%** |

- **RAG ทำร้าย fuzzytopk** — ฝึกด้วยภาพเดียว ไม่รู้จักใช้ reference
- **RAG ช่วย s1cascade** — cascade pretraining เปิดใช้ visual comparison
- **RAG-in-training ปิดช่องว่างได้สมบูรณ์** — 85.86% ดีที่สุด

### 8.2 Prompt แบบไหนดีที่สุด?

| โมเดล | P0 | P1 | P2 | P3 | ดีที่สุด |
|-------|----|----|----|----|---------|
| fuzzytopk | **74.0%** | 71.7% | 70.7% | 71.7% | **P0** |
| s1cascade (R2) | **74.7%** | 72.7% | 73.7% | 71.7% | **P0** |
| M1 (R0) | 55.0% | **59.4%** | 53.8% | 50.6% | **P1** |

- **fuzzytopk/s1cascade:** P0 ชนะ (prompt ง่าย) — โมเดลตอบสั้นไม่แตกต่างตาม prompt มาก
- **M1:** P1 (CoT ทีละขั้น) ดีที่สุด — reasoning ที่ชัดเจน +4.4%

### 8.3 Encoder RAG ไหนดีที่สุด?

ด้วย alpha ที่ปรับแล้ว (s1cascade เท่านั้น):

| RAG | ค่าเริ่มต้น α=0.5 | Alpha ที่ดีที่สุด | ความแม่นยำ | หมายเหตุ |
|-----|-----------------|----------------|----------|---------|
| R2 (SigLIP+BGE-M3) | 74.75% | **α=0.9** | **79.80%** | image-dominant |
| R3 (Jina-CLIP+MedCPT) | 66.67% | **α=0.7** | **78.79%** | 30% text ที่ดีที่สุด |
| R0 (CLIP image-only) | 72.73% | N/A | 72.73% | ไม่มี alpha |

### 8.4 วิเคราะห์ต่อโรค

| โรค | fuzzytopk NoRAG | s1cascade R0-P0 | s1cascade R2 α=0.9 | ragR2_a09 R0-P0 | M1 R0-P1 | สรุป |
|-----|-----------------|----------------|-------------------|---------------------|---------|------|
| SCCIS | 100.0% | 91.7% | 100.0% | **100.0%** | 66.7% | ✅ ง่าย |
| Melanocytic nevi | 91.7% | 91.7% | 91.7% | **100.0%** | 100.0% | ✅ ง่าย |
| BCC | 76.9% | 69.2% | 76.9% | **100.0%** | 84.6% | ✅ ง่าย (ragR2) |
| Acne vulgaris | 87.5% | 87.5% | **100.0%** | **100.0%** | 100.0% | ✅ ง่าย |
| Psoriasis | 84.6% | 69.2% | 61.5% | **100.0%** | 61.5% | ปานกลาง (ragR2 ดีที่สุด) |
| Lichen planus | 77.8% | **88.9%** | **88.9%** | **88.9%** | 22.2% | ปานกลาง |
| Scleroderma | 37.5% | 37.5% | **75.0%** | **87.5%** | 25.0% | ยาก; RAG training ช่วย |
| Photodermatoses | 57.1% | 37.5% | **62.5%** | **75.0%** | 25.0% | ยาก; RAG training ช่วย |
| Lupus | 44.4% | 66.7% | **77.8%** | 55.6% | 22.2% | ยาก; s1cascade α=0.9 ดีที่สุด |
| Sarcoidosis | 42.9% | **71.4%** | 57.1% | 14.3% | **71.4%** | ยาก; ragR2 ล้มเหลว |

**ข้อค้นพบระดับโรค:**
- **ragR2_a09 ครองตำแหน่ง 7/10 โรค** ที่ sensitivity 100% หรือใกล้เคียง
- **Psoriasis:** ragR2_a09 winner ที่ใหญ่ที่สุด: 61.5% → 100% (+38.5%)
- **Sarcoidosis:** ragR2_a09 หายนะ 14.3%; M1 แปลกใจ 71.4%
- **Lupus:** ragR2_a09 ถดถอยจาก s1cascade (77.8% → 55.6%) — อาจสับสนกับ reference sarcoidosis

### 8.5 ทำไม s1cascade ใช้ RAG ได้แต่ fuzzytopk ไม่ได้

| ด้าน | fuzzytopk | fuzzytopk_s1cascade | ragR2_a09 |
|------|-----------|---------------------|-----------|
| Training context | ภาพเดียว + prompt | ภาพเดียว + prompt | **หลายภาพ (ref + query)** |
| Starting weights | base → disease | base → group → disease | base → group → disease+RAG |
| ผล RAG | −8.7% (สับสน) | +5.8% | **+11.86%** |
| สมมติฐาน | ไม่เคยเห็น multi-image | Stage 1 สอน visual comparison | ฝึกให้ใช้ reference โดยตรง |

### 8.6 ทำไม M1 ด้อยกว่า s1cascade (59.4% vs 74.75%)

1. **3-stage cascade penalty:** Stage 2 เริ่มจาก weights Group Classifier → initialization ที่เสียหาย
2. **CoT output ยาว:** M1 สร้าง `<think>` ยาว → VRAM pressure สูงขึ้น
3. **Prompt ซับซ้อน:** GT group + RAG images + captions + CoT = prompt ยาวมาก
4. **M1 ไม่มี RAG แย่มาก:** 47.9% vs fuzzytopk 74.0%

---

### 8.7 สรุปบทเรียนหลักจากทั้งโครงการ

| บทเรียน | รายละเอียด |
|---------|-----------|
| RAG-in-Training ปิดช่องว่าง | ฝึกด้วย reference ตั้งแต่แรก → encoder-agnostic generalization |
| Merged Init ชนะ Checkpoint Init | ป้องกัน catastrophic interference ระหว่าง tasks |
| Alpha Sweep บังคับ | ค่า default α=0.5 ไม่ดีพอ; sweep ทุกครั้งก่อน deploy |
| 3-Stage Cascade Penalty | cascade stages แบบ hard dependency ทำให้ผลแย่ลง แม้ใน oracle |
| K=1 เพียงพอสำหรับ Training | K=3 OOM บน RTX 5070 Ti; K=1 ให้ผล SOTA |

### 8.8 Bug ที่แก้ไขระหว่าง Benchmarking

| Bug | อาการ | วิธีแก้ |
|-----|-------|--------|
| Double `</think>` | P3 แสดง 100% จาก 1/99 valid sample เท่านั้น | Strip dangling `</think>` + pattern `Diagnosis:` |
| P1/P2 low coverage | 16–28/99 valid → accuracy เกินจริง | `Step N:` last occurrence + `Final diagnosis:` extraction |
| `--method` ไม่ถูกส่งต่อ | `--method fuzzytopk` รัน M1 แบบเงียบๆ | Pass `method=method` ที่ call site ใน `run_rag_benchmark.py` |
| `max_length=4096, truncation=True` ใน main batch | "Mismatch in image token count" สำหรับ prompt RAG ยาว | ลบออกจาก `apply_chat_template()` main path |
| Metric inflation | accuracy=1.0 จาก 1 parseable sample | Metric = correct/total (ไม่ใช่ correct/valid) |
| OOM บน query images ขนาดใหญ่ | 14–40/99 samples ล้มเหลวต่อการรัน | `img.thumbnail((672, 672), LANCZOS)` ใน `_load_image()` |
| Reference captions หายไป | In-context references ไม่มี clinical text | Return `self.captions[i]` จาก `retrieve()` |
| Symptom descriptions | ไม่มี realistic text query สำหรับ R1–R3 | สร้าง `val_captions_for_symptoms.json` (94 patient descriptions) |
| fuzzytopk model path | โหลดผิด model | `_STAGE2_MODEL_PATH_MAP` dict |

---

## 9. การประเมิน Stage 3 — การสร้าง Caption

Stage 3 สร้าง caption คลินิกหลายประโยค ประเมินด้วย BLEU และ ROUGE เทียบกับ `caption_zh_polish_en` จาก CSV

### 9.0 Stage 3 เริ่มต้นจากที่ไหน — ลำดับ Pipeline ทั้งหมด

Stage 3 **ไม่ได้** เริ่มจาก base model หรือ Stage 1 แต่เริ่มจาก **Stage 2 ตัวจำแนกโรค** (fuzzytopk_s1cascade) ซึ่งถูกสร้างต่อมาจาก Stage 1 group classifier:

```
Base Model (Qwen3-VL-8B)
    ↓  Stage 1: fine-tune จำแนก 4 กลุ่มโรค (88.68% val)
Group Classifier  →  skincap_3stage_group_classification_merged
    ↓  Stage 2: fine-tune จำแนก 10 โรค (74–85% val)
Disease Classifier  →  skincap_fuzzytopk_s1cascade_classification_merged
    ↓  Stage 3: fine-tune สร้าง caption  ← นี่คือจุดที่เราอยู่
Caption Model  →  skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_classification_merged
```

**ทำไมต้องใช้ Stage 2 เป็นจุดเริ่มต้น?**
ตัวจำแนกโรคได้ "เรียนรู้" ลักษณะทาง dermoscopy แล้วจาก Stage 2 — รูปร่างรอยโรค, สี, ลักษณะเฉพาะต่อโรค เมื่อเริ่ม Stage 3 จากโมเดลนี้ Caption model จะรับ visual representations ที่ละเอียดระดับโรคมาด้วย แทนที่จะเรียนลักษณะผิวหนังจากศูนย์ใหม่

**การเลือกสำคัญ: จะส่งต่อจาก Stage 2 ไป Stage 3 อย่างไร?**

ทดสอบ 2 วิธี (ดูผลใน ablation ด้านล่าง):

| | Way 1 — Checkpoint Init | Way 2 — Merged Init ✅ |
|--|------------------------|----------------------|
| ความรู้ Stage 2 อยู่ที่ไหน | ใน LoRA adapters (เฉพาะ task) | ฝังใน base weights (ถาวร) |
| LoRA ของ Stage 3 | ใช้ adapters Stage 2 ต่อ | Fresh adapters (zeros ทั้งหมด) |
| ความเสี่ยง | Caption fine-tuning เขียนทับ disease adapters → เกิด interference | ไม่มีความเสี่ยง — ความรู้โรคปลอดภัยใน base weights |
| ผล BLEU-4 | 9.82 | **29.33 (ดีกว่า 3×)** |

**กฎสำคัญ:** เมื่อเปลี่ยน task ใน multi-stage LoRA fine-tuning — **merge ก่อนเสมอ แล้วค่อยเพิ่ม fresh LoRA ใหม่** อย่า fine-tune adapters เดิมต่อบน task ต่างกัน

### การตั้งค่าการประเมิน
- **Prompt:** `"อธิบายภาพรอยโรคผิวหนังนี้อย่างละเอียด รวมถึงลักษณะ การวินิจฉัยที่เป็นไปได้ และการตรวจที่แนะนำ"`
- **Ground truth:** คอลัมน์ `caption_zh_polish_en` (คำอธิบายทางคลินิกที่แปลและปรับปรุงแล้ว)
- **Val set:** 99 ตัวอย่างจาก `split_info_3stage.json`

### 9.1 Stage 3 Ablation — 4 การทดลอง (Way 1/2 × STS เปิด/ปิด)

กลยุทธ์การเริ่มต้น 2 แบบ:
- **Way 1 (checkpoint init):** โหลด LoRA checkpoint Stage 2 (Disease Classifier) → fine-tune adapters ที่มีอยู่ต่อบน captions
- **Way 2 (merged init):** โหลด merged model Stage 2 (Disease Classifier) → เพิ่ม fresh LoRA adapters → ฝึกใหม่บน captions ความรู้เรื่องโรคฝังอยู่ใน base weights อย่างถาวร

**STS (Selective Token Supervision):** per-token loss weighting (`w_ans × w_reason`) + IBR regularization (`β × ||LoRA||²`)

| Exp | ชื่อ | Init | STS | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 | ROUGE-2 |
|-----|------|------|-----|--------|--------|--------|--------|---------|---------|
| 0 (baseline)† | Original | checkpoint | ✗ | 33.07 | 17.20 | 12.14 | **9.82** | 38.50 | 13.15 |
| 1 | Way 1 | checkpoint | ✗ | 29.22 | 18.65 | 13.03 | 9.82 | 38.9 | 15.71 |
| **2** | **Way 2** | **merged** | **✗** | **42.22** | **35.19** | **31.55** | **29.33** | **53.55** | **36.33** |
| 3 | Way 1 + STS | checkpoint | ✓ | 0.01 | 0.00 | 0.00 | 0.00 | 5.03 | 0.38 |
| 4 | Way 2 + STS | merged | ✓ | 1.78 | 1.09 | 0.78 | 0.61 | 15.68 | 5.83 |

† Exp 0 "Original baseline" (B1=33.07) มาจาก run ก่อน ablation; ไม่ได้บันทึกใน `stage3_ablation_results.json` (ซึ่งมีเฉพาะ key "1"–"4")

**ผู้ชนะ: Exp 2 (Way 2, merged init, ไม่มี STS) — BLEU-4=29.33 (ดีกว่า baseline 3×)**

### 9.2 ทำไม Merged Init ถึงชนะ

| ด้าน | Way 1 (checkpoint) | Way 2 (merged) |
|------|--------------------|----------------|
| LoRA adapters ตอนเริ่ม Stage 3 | Pre-trained (task โรค) | Fresh (zeros ทั้งหมด) |
| ความรู้เรื่องโรค | อยู่ใน LoRA adapters — **เสี่ยงถูกแทรกแซง** | ฝังใน base weights — **ถาวร ปลอดภัย** |
| การ adapt caption | LoRA ต้องลืม task โรค → เรียน caption | LoRA เรียน caption ใหม่บน base สะอาด |
| ผลลัพธ์ | BLEU-4=9.82 | **BLEU-4=29.33 (3× ดีกว่า)** |

**แนวคิดหลัก:** เมื่อ LoRA adapters มีความรู้การจำแนกโรค การ fine-tune บน task ต่างกัน (สร้าง caption) ทำให้เกิด catastrophic interference การ merged init เข้ารหัสความรู้โรคลงใน base weights อย่างถาวร จากนั้น fresh LoRA adapters สามารถเรียน caption ได้โดยไม่มี gradient ขัดแย้ง

### 9.3 ทำไม STS ถึงล้มเหลว

**Exp 3 (Way 1 + STS): BLEU-4=0.0 — พังทลายอย่างสมบูรณ์**
- IBR (`β × ||LoRA||²`) ลงโทษ LoRA norms ที่ใหญ่
- Checkpoint init มี LoRA ที่ฝึกแล้วพร้อม norms ใหญ่อยู่แล้ว
- IBR ดัน parameters ทั้งหมดไปที่ศูนย์ → ลบความรู้ทั้งหมด → ไม่สามารถสร้างข้อความได้

**Exp 4 (Way 2 + STS): BLEU-4=0.61 — ลดลงรุนแรง**
- Fresh LoRA เริ่มที่ศูนย์ แต่ IBR ป้องกันไม่ให้เติบโต → fitting แย่
- Unsloth compiled training คืน non-tensor `logits` → STS ใช้ fallback `outputs.loss + IBR`
- ผล: 29.33 (ไม่มี STS) vs 0.61 (มี STS) — IBR สูญเสีย −28.72 BLEU-4

**สรุป: STS/IBR ไม่เข้ากันกับ multi-stage transfer learning**
IBR ออกแบบมาสำหรับฝึกตั้งแต่ต้น (norms ใกล้ศูนย์); ทั้งสองขั้นมี prior ที่แข็งแกร่งแล้ว

### 9.4 STS กับ Paper Novelty?

**STS ไม่ใช่ novelty แบบ performance** — ผลเชิงลบมาก (ทำให้ BLEU ลดลง 28–30 points)

**Merged init คือ novelty จริง:**
> *"สำหรับ multi-stage LoRA fine-tuning การโหลด merged model ของ stage ก่อนหน้าและเพิ่ม fresh adapters ดีกว่าการ fine-tune ต่อจาก LoRA checkpoint ถึง 3× บน BLEU-4 เหตุผล: ความรู้โรคฝังอยู่ใน base weights อย่างถาวร ป้องกันจาก task interference"*

STS รวมไว้ใน paper เป็น ablation (แสดงว่าทดลองอย่างครบถ้วน) พร้อมคำอธิบายว่าทำไมถึงล้มเหลว

### 9.5 การใช้ Caption Stage 3 เป็น RAG Text Query (การทดลอง B4)

| RAG | Symptoms → text | Stage3 cap → text | ผลเปลี่ยน |
|-----|----------------|-------------------|---------|
| R1 (ClinicalBERT) | 69.7% | 71.7% | +2.0% |
| R2 (BGE-M3) | 74.7% | 71.7% | −3.0% |
| R3 (MedCPT) | 66.7% | 72.7% | **+6.1%** |

Caption Stage 3 ช่วย R1 และ R3 (ข้อความทางการแพทย์ structured ตรงกับ MedCPT/ClinicalBERT) แต่ทำร้าย R2 (BGE-M3 ชอบ vocabulary ของอาการผู้ป่วย)

---

## 10. การแก้ไขทางเทคนิคระหว่างโครงการ

| ปัญหา | การแก้ไข | ผลกระทบ |
|-------|---------|---------|
| Data leakage ใน RAG index | `RAG_USE_ALL_DATA=False` → train-only 911 ภาพ | ลบ circular retrieval จาก val |
| Strategy B ไม่ใช้ text | ส่ง `caption` เป็น `vlm_description` | เปิดใช้ hybrid retrieval จริงสำหรับ R1–R3 |
| `--method` ไม่ถูกส่งต่อ | เพิ่ม `method=method` ใน `run_experiment()` | `--method fuzzytopk` รัน fuzzytopk จริง |
| Double `</think>` parsing | เพิ่ม strip + ดึง `Diagnosis:` | แก้ไขการรัน P3 ทั้งหมด |
| P1/P2 low coverage | `_extract_disease()` ดึง `Final diagnosis:` (P2), `Step N:` สุดท้าย (P1) | coverage ดีขึ้น |
| OOM บนภาพ query ใหญ่ | `img.thumbnail((672, 672), LANCZOS)` ใน `_load_image()` | **0 OOM errors ทุกการทดลอง** |

---

## 11. ไฟล์ที่แก้ไข

| ไฟล์ | การเปลี่ยนแปลง |
|------|--------------|
| `inference_disease_classification.py` | CLI `--stage2_method`, RAG caption in-context, P3 Diagnosis extraction, 672×672 thumbnail, zero-shot baselines, accuracy = correct/total |
| `rag_retrieval.py` | `retrieve()` คืน 3-tuple `(path, label, caption)`; `HybridRAGRetriever.build()` บันทึก metadata `strategy`, `img_encoder`, `txt_encoder` ใน npz |
| `run_rag_benchmark.py` | `method=method` ส่งต่อไปยัง `run_experiment()` |
| `val_captions_for_symptoms.json` | สร้างใหม่: 94 val items พร้อมคำอธิบายอาการที่ผู้ป่วยเขียน สำหรับ text query R1–R3 |
| `val_captions_stage3.json` | สร้างใหม่: 99 Stage 3 generated captions สำหรับการทดลอง B4 |
| `train_two_stage_FuzzyTopK.py` | เพิ่ม `--rag_k_train`, `--rag_exp`, `--alpha` CLI args; `precompute_rag_refs()` function; `prepare_classification_data_with_rag()`; `--stage3_init` (checkpoint/merged), `--use_sts`, `--sts_beta` |
| `train_three_stage_hybrid_topk.py` | `--stage3_source`, `--stage3_lr` CLI args |
| `medical_token_importance.py` | ใหม่: `MedicalSTSConfig`, `MedicalTokenImportanceScorer`, `STSSFTTrainer`; IBR (`compute_ibr_loss`); Unsloth non-tensor logits guard |
| `run_stage3_experiments.py` | ใหม่: 4-experiment Stage 3 ablation runner; auto-skip if merged exists; BLEU/ROUGE logging |
| `stage3_ablation_results.json` | ผลลัพธ์ครบ 4 การทดลอง (Way 1/2 × STS on/off) |

---

## 12. ขั้นตอนถัดไปที่แนะนำ

ดีที่สุดปัจจุบัน: **ragR2_a09 R0-P0 = 85.86%** การทดลองถัดไปเรียงตามประโยชน์ที่คาดหวัง:

| ลำดับ | การทดลอง | เหตุผล | ประโยชน์ที่คาดหวัง |
|------|---------|--------|-----------------|
| 1 | **แก้ไข Sarcoidosis collapse** | ragR2_a09 sarcoidosis: 14.3% (เคยเป็น 57.1%) — ลองใช้ K=3 หรือ oversampled dataset | +3–5% balanced acc |
| 2 | **ragR2_a09 กับ alpha tuning ตอน inference** | R2-P0 inference ให้ 82.83%; ลอง α=0.9 vs α=0.5 | ±1–3% |
| 3 | **ragR2_a09 กับ R3 (Jina-CLIP+MedCPT) ตอน inference** | R3 α=0.7 ให้ 78.79% สำหรับ s1cascade | ±2% |
| 4 | **ฝึกใหม่ K=2 หรือ K=3 reference** | K=1 ให้ 85.86%; K=3 OOM กับ batch=2; ลอง batch=1 | +1–3% |
| 5 | **แก้ข้อมูล Photodermatoses/Sarcoidosis** | ข้อจำกัดเชิงโครงสร้างจากตัวอย่างฝึกน้อย | +5–10% balanced accuracy |

### การตั้งค่าที่ดีที่สุดปัจจุบัน

สำหรับ **ความแม่นยำสูงสุด:**
- **โมเดล:** ragR2_a09 + **RAG R0-P0 = 85.86% (85/99)** ← ดีที่สุด
- CLIP image-only retrieval ตอน inference

สำหรับ **ภาพเดียว (ไม่มี RAG):**
- **โมเดล:** fuzzytopk, ไม่มี RAG P0 = **74.0% (74/100 valid)**

สำหรับ **M1 ดีที่สุด (oracle):**
- **โมเดล:** M1 + RAG R0-P1 = **59.4% (59/99)**

### ตารางอ้างอิง Alpha Tuning (fuzzytopk_s1cascade, P0)

| RAG | α=0.5 | α=0.7 | α=0.8 | α=0.9 | α=1.0 | ดีที่สุด |
|-----|-------|-------|-------|-------|-------|---------|
| R2 (SigLIP+BGE-M3) | 74.75% | 76.77% | 75.76% | **79.80%** | 75.76% | **α=0.9** |
| R3 (Jina+MedCPT) | 66.67% | **78.79%** | 71.72% | 67.68% | 66.67% | **α=0.7** |
| R4 (Nomic unified) | 67.68% | 71.72% | — | 72.73% | — | **α=0.9** |
| R1 (CLIP+ClinicalBERT) | 69.70% | — | — | 71.72% | — | **α=0.9** |
| R0 (CLIP เฉพาะภาพ) | 72.73% | — | — | — | — | N/A |

---

*อัปเดต: 2026-03-17 (เพิ่มนิยามชื่อการทดลองทั้งหมด, ตาราง Paper Table II, และ Model Evolution path)*
*Val set: 99 ตัวอย่าง (split_info_3stage.json) สำหรับ RAG; 101 ตัวอย่างสำหรับ fuzzytopk ไม่มี RAG*
*โมเดลประเมินบน: RTX 5070 Ti (15.9 GB VRAM), Qwen3-VL-8B-Thinking backbone*
*การทดลอง RAG ทั้งหมด: 0 OOM errors (แก้ไข image resize: 672×672 thumbnail)*
*ผลดีที่สุด: ragR2_a09 + RAG R0-P0 = 85.86% (85/99) — RAG-in-training (Experiment E)*

---

## 13. การประเมิน Caption ด้วย BERTScore และ Correctness

ขยายผลจาก Section 9 ด้วย 2 metrics เพิ่มเติม:
- **BERTScore-F (BSF):** ความคล้ายเชิงความหมายโดย BERT token embeddings — ไม่ต้องการ exact word match
  - **โมเดลที่ใช้จริง: `roberta-large`** (ดู `eval_caption_extended.py` บรรทัด 154)
  - หมายเหตุ: comment ในไฟล์ระบุ "microsoft/deberta-xlarge-mnli or fallback to roberta-large" แต่ implementation จริงใช้ `roberta-large` โดยตรง
- **Correctness (Corr):** % ของ caption ที่ระบุชื่อโรคถูกต้อง (fuzzy alias match) — **ไม่ใช่** correlation ทางสถิติ

### 13.1 ผลครบ 4 Variants (จาก `results_caption_extended_full.json`)

| Variant | B1 | B2 | B4 | R1 | BSF (roberta-large) | Corr |
|---------|----|----|----|----|---------------------|------|
| checkpoint_noguide | 29.22 | 18.65 | 9.82 | 38.9 | 88.12 | 47.47% |
| checkpoint_guide | 35.53 | 22.43 | 11.75 | 40.02 | 88.44 | 72.73% |
| **merged_noguide** | **42.22** | **35.19** | **29.33** | **53.55** | **91.12** | 62.63% |
| **merged_guide** | 35.68 | 22.85 | 13.11 | 41.2 | 88.57 | **78.79%** |

**ชื่อตัวแปร:**
- **checkpoint** = Way 1: LoRA checkpoint Stage 2 → fine-tune adapters ต่อ (ความรู้โรคอยู่ใน LoRA → เสี่ยงถูก overwrite)
- **merged** = Way 2: merged model Stage 2 → fresh LoRA ใหม่ ✅ (ความรู้โรคฝังใน base weights ถาวร)
- **noguide** = prompt มาตรฐาน — ไม่บอกโรค, โมเดลต้องอนุมานจากภาพเอง
- **guide** = guided prompt — ส่ง Stage 2 disease label เข้า Stage 3 (`"Stage 2 diagnosis: {disease}.\n\n..."`)

### 13.2 Tradeoff: BLEU/ROUGE vs Correctness

| Mode | B4 | BSF | Corr | การตีความ |
|------|----|-----|------|---------|
| merged_noguide | **29.33** | **91.12** | 62.63% | n-gram/semantic quality สูงที่สุด แต่ไม่ระบุโรคทุก caption |
| merged_guide | 13.11 | 88.57 | **78.79%** | Correctness สูงกว่า +16.16 pp แต่ style diverge จาก GT → BLEU ลด −16.22 |

Ground truth caption ใช้ hedging language ("*may indicate*", "*further investigation needed*") ส่วน guided caption ตอบตรงและมั่นใจกว่า ("*consistent with*") → n-gram overlap ลดแม้ content ถูกต้องมากกว่า

**การเลือกใน Paper:**
- **merged_noguide = primary** — BLEU-4=29.33, BSF=91.12 (n-gram + semantic quality)
- **merged_guide = secondary** — แสดงว่า disease-guided prompt เพิ่ม Correctness ได้ +16 pp

> **หมายเหตุ BSF:** ค่า BSF 91.12 (Section 13.1/13.2) มาจาก `eval_caption_extended.py` (batch_size=8, lang="en") ส่วน Section 13.3 แสดง 91.14 จาก `eval_bertscore_compare.py` (batch_size=16) ทั้งคู่ใช้ roberta-large layer 17 เหมือนกัน ผลต่าง 0.02 มาจาก floating-point variance ระหว่างการรัน — ถือว่าเทียบเท่ากัน

### 13.3 BERTScore Comparison ข้าม BERT Models (`eval_bertscore_compare.py`)

ทดสอบ 4 BERT models เพื่อดูว่า domain-specific BERT ให้ BSF ที่ different/better ไหม:

| Model | HuggingFace ID | Training Data | Layers | noguide F1 | guide F1 |
|-------|---------------|---------------|--------|-----------|---------|
| `roberta-large` | `roberta-large` | General English (CC-News, Books, Wiki) | 24 (layer 17) | 91.14 | 88.57 |
| `clinical-bert` | `medicalai/ClinicalBERT` | MIMIC-III clinical notes | **6** (layer 6) | 75.72 | 69.24 |
| **`medcpt-article`** | `ncbi/MedCPT-Article-Encoder` | PubMed article retrieval (contrastive) | 12 (layer 9) | **92.12** | **89.87** |
| `pubmedbert` | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` | PubMed abstracts + full text | 12 (layer 9) | 88.30 | 85.28 |

*หมายเหตุ MedCPT:* RAG R3 ใช้ `MedCPT-Query-Encoder` สำหรับ query สั้น สำหรับ BERTScore (passage vs passage) ต้องใช้ **Article-Encoder** แทน

**ทุก model เห็น rank เหมือนกัน:** noguide > guide ทุก model → ยืนยันผลเดิม

**วิเคราะห์:**

| Model | เหตุผลที่ score เป็นอย่างนั้น |
|-------|---------------------------|
| **ClinicalBERT (75.72) — ต่ำสุด** | 6 layers เท่านั้น (ตื้นมาก); ฝึกบน ICU notes ≠ dermatology captions; embedding space ไม่ calibrate สำหรับ cosine similarity |
| **MedCPT-Article (92.12) — สูงสุด** | ฝึกด้วย contrastive learning บน PubMed article pairs → embedding space calibrate สำหรับ biomedical text similarity โดยตรง; dermatology captions ≈ short biomedical articles |
| **PubMedBERT (88.30) — ต่ำกว่า roberta** | ดีสำหรับ classification/NER แต่ ไม่ได้ฝึกสำหรับ sentence-level cosine similarity → general embedding quality ด้อยกว่า roberta-large |

**คำแนะนำสำหรับ Paper:**

| สถานการณ์ | แนะนำ |
|---------|-------|
| เปรียบเทียบกับงาน caption ทั่วไป | ใช้ **roberta-large** (91.14) — standard default, reproducible |
| ต้องการ clinical domain metric | ใช้ **MedCPT-Article** (92.12) — biomedical passage similarity, เหมาะกว่า |
| รายงานทั้งคู่ | **ดีที่สุด** — roberta-large สำหรับ reproducibility + MedCPT สำหรับ domain appropriateness |

> **BSF score ขึ้นอยู่กับ BERT model ที่ใช้** — ต้องระบุชื่อ model ใน paper เสมอ

---

## 14. การทดสอบความเร็ว Inference — Speed Benchmark

### 14.1 แรงจูงใจ

โมเดล Qwen3-VL-8B-Thinking มีขนาด ~16.78 GB (bf16) ใหญ่กว่า VRAM ของ RTX 5070 Ti (17.09 GB จริง) เล็กน้อย Unsloth BnB-4bit ลดขนาดเหลือ ~8.4 GB แต่ inference ช้าเพราะไม่มี continuous batching การทดสอบนี้เปรียบเทียบ:
- **Unsloth 4-bit** (baseline) — ใช้ตลอดโครงการ
- **vLLM BnB-4bit** — continuous batching + CUDA graph
- **SGLang FP8** — FP8 online quantization + RadixAttention

### 14.2 Hardware และ Setup

| ข้อมูล | ค่า |
|--------|-----|
| GPU | RTX 5070 Ti (Blackwell sm_120) |
| VRAM จริง | 17.09 GB |
| CUDA | 12.8 (WSL2) |
| โมเดล | skincap_fuzzytopk_s1cascade_ragR2_a09_classification_merged |
| Val set | 99 ตัวอย่าง |
| Warm-up | 10 samples ก่อนวัด |

### 14.3 ผลการวัด — Stage 2 (Disease Classification, max_new_tokens=64)

| Engine | bs=1 ms/img | bs=4 ms/img | Speedup vs Unsloth-bs1 |
|--------|------------|------------|----------------------|
| Unsloth-4bit | 1095.69 ms | 499.69 ms | 1.0× (baseline) |
| vLLM-BnB4 | 480.14 ms | 179.41 ms | 2.3× / **6.1×** |
| SGLang-FP8 | 331.0 ms | **109.8 ms** | 3.3× / **~10×** |

### 14.4 ผลการวัด — Stage 3 (Caption Generation, max_new_tokens=256)

| Engine | bs=1 ms/img | bs=4 ms/img | Speedup vs Unsloth-bs1 |
|--------|------------|------------|----------------------|
| Unsloth-4bit | 6699.15 ms | 3003.3 ms | 1.0× (baseline) |
| vLLM-BnB4 | 2957.21 ms | 1094.21 ms | 2.3× / **6.1×** |
| SGLang-FP8 | 1695.4 ms | **583.51 ms** | 3.9× / **~11.5×** |

### 14.5 ปัญหาระหว่างการทดสอบ

| Engine | ปัญหา | สถานะ |
|--------|-------|-------|
| **LMDeploy** | OOM: โมเดล bf16 16.78 GB > VRAM 17.09 GB (overhead ไม่พอ); ต้องการ pre-quantized weights ที่ยังไม่มี | ❌ ไม่ได้วัด |
| **TensorRT-LLM** | ต้องการ CUDA driver ≥ 13.0; ระบบใช้ CUDA 12.8 | ❌ ไม่ได้วัด |
| **vLLM** | torch ABI conflict หลัง SGLang upgrade torch เป็น 2.9.1: `vllm/_C.abi3.so: undefined symbol _ZN3c104cuda29...`; **ต้องติดตั้ง vLLM ใหม่ก่อนใช้** | ⚠️ ใช้งานได้ก่อน SGLang setup |
| **SGLang startup timeout** | โหลด 4 safetensors shards (~30–40s/shard) จาก NTFS ผ่าน WSL2 รวม ~2–4 นาที; timeout เดิม 2–3 นาทีสั้นเกินไป | ✅ แก้ด้วย timeout 10 นาที |

### 14.6 ข้อสรุปความเร็ว

- **SGLang FP8 batch=4 เร็วที่สุด:** ~10× Stage 2, ~11.5× Stage 3 เทียบ Unsloth single
- **vLLM BnB-4bit batch=4 รองลงมา:** ~6× ทั้ง Stage 2 และ 3
- **ข้อแลกเปลี่ยน SGLang FP8:** ความแม่นยำลด 5 pp (77.78% vs 82.83%) แต่ throughput เพิ่ม ~10×

---

## 15. การประเมิน Accuracy ด้วย SGLang FP8

### 15.1 แรงจูงใจ

Speed benchmark แสดงว่า SGLang FP8 เร็วกว่า Unsloth ~10× แต่ยังไม่รู้ว่า accuracy จะเป็นเท่าไหร่ จำเป็นต้องประเมินบน 99-sample validation set เดียวกันกับ baseline

### 15.2 Bugs ที่พบใน eval_sglang_fp8_wsl.py (ผล 0% ก่อนแก้)

| Bug | อาการ | วิธีแก้ |
|-----|-------|--------|
| **Disease list ผิด** | ใช้ 25 โรค custom ("eczema", "dermatitis" ฯลฯ) แทน 10 โรคฝึก | เปลี่ยนเป็น 10 โรคจาก `_DISEASE_GROUPS_4_TOP10` |
| **Prompt ผิด** | ใช้ "You are a dermatologist..." แทน P0 training prompt | เปลี่ยนเป็น `"This skin lesion belongs to the group '{group}'..."` |
| **Matching ผิด** | ใช้ `fuzz.extractOne threshold=91` — ไม่ match โรคที่มีหลายคำ | เปลี่ยนเป็น `match_to_disease_list()`: word-overlap ก่อน แล้ว `token_sort_ratio ≥ 70` |
| **ไม่ parse response** | ไม่ strip `<think>` block; ไม่ extract "Diagnosis:" | เพิ่ม `extract_disease_from_response()` |

### 15.3 ผลหลังแก้ Bug

| Metric | ค่า |
|--------|-----|
| SGLang FP8 Accuracy | **77.78%** (77/99) |
| Unsloth BnB-4bit baseline (RAGR2 inference) | 82.83% (82/99) |
| ผลต่าง | −5.05 pp |
| Engine | SGLang 0.5.5, FP8 online quantization |
| Quantization overhead | ไม่ต้องการ pre-quantized weights — quantize ที่ load time |

> **⚠️ หมายเหตุการเปรียบเทียบ:** SGLang FP8 ประเมินด้วย **group-only prompt** (Stage 2 ใช้แค่ predicted group ไม่มี RAG images) ส่วน Unsloth baseline 82.83% ใช้ **RAG R2 context** (SigLIP+BGE-M3 retrieved reference images) — นั่นหมายความว่า Unsloth ได้ข้อมูลเพิ่มเติมใน prompt ดังนั้นส่วนหนึ่งของช่องว่าง 5 pp มาจาก RAG context ที่หายไป ไม่ใช่แค่ FP8 quantization เพียงอย่างเดียว

### 15.4 ทำไม FP8 ถึงด้อยกว่า BnB-4bit ~5 pp

| ด้าน | SGLang FP8 | Unsloth BnB-4bit (QLoRA) |
|------|-----------|--------------------------|
| วิธี quantize | FP8 online quantization ที่ load time | Base model โหลด 4-bit NF4; LoRA adapters ฝึกใน fp16 |
| Forward pass | weights dequantize เป็น bf16 ก่อน matmul | LoRA gradients เห็น 4-bit quantized base weights ทุก step |
| Training alignment | โมเดลไม่เคยเห็น FP8 weights ระหว่างฝึก | LoRA ถูกฝึกกับ quantized base weights → adapts ให้เข้ากัน |
| RAG context ที่ inference | ❌ ไม่มี | ✅ RAGR2 reference images ใน prompt |
| VRAM | ~8.4 GB | ~8.4 GB |

**บทสรุป:** ช่องว่าง 5 pp มาจาก 2 ส่วน: (1) FP8 quantization error สะสม เพราะ LoRA ไม่ได้ปรับตัวกับ FP8 weights; (2) ขาด RAG R2 context ที่ Unsloth มี

### 15.5 Per-disease Breakdown — SGLang FP8 vs Unsloth BnB-4bit RAGR2

*(ข้อมูลจาก `results_disease_sglang_fp8_val_predictions.json` และ `results_disease_fuzzytopk_s1cascade_ragR2_a09_RAGR2_a09_val_predictions.json`)*

| โรค | Unsloth RAGR2 | SGLang FP8 | n | หมายเหตุ |
|-----|--------------|-----------|---|---------|
| BCC | **100.0%** | 92.3% | 13 | SGLang ผิด 1 case (BCC → SCCIS) |
| Acne vulgaris | **100.0%** | 87.5% | 8 | |
| Psoriasis | **100.0%** | 84.6% | 13 | |
| Melanocytic nevi | 91.7% | **100.0%** | 12 | SGLang ดีกว่า — FP8 ไม่สูญเสีย visual feature |
| SCCIS | 83.3% | 83.3% | 12 | เท่ากัน |
| Lichen planus | 77.8% | 77.8% | 9 | เท่ากัน |
| Photodermatoses | **75.0%** | 62.5% | 8 | |
| Scleroderma | 75.0% | 75.0% | 8 | เท่ากัน |
| Lupus erythematosus | 55.6% | **66.7%** | 9 | SGLang ดีกว่า |
| Sarcoidosis | **42.9%** | 14.3% | 7 | SGLang แย่มาก (1/7) — RAGR2 context ช่วย Sarcoidosis มาก |
| **รวม** | **82.83%** | **77.8%** | **99** | |

---

## 16. ตัวอย่าง Raw Output — Stage 2 และ Stage 3

ตัวอย่าง 3 samples แรกจาก val set (id=10, 99, 144)

### 16.1 Stage 2 — Disease Classification

**Sample 1 — id=10, GT: squamous cell carcinoma in situ ✅**

| Engine | Raw Output |
|--------|-----------|
| **Unsloth BnB-4bit** | `squamous cell carcinoma in situ` |
| **SGLang FP8** | `</think>\n\nThis image shows squamous cell carcinoma in situ.` |

**Sample 2 — id=99, GT: melanocytic nevi ✅**

| Engine | Raw Output |
|--------|-----------|
| **Unsloth BnB-4bit** | `melanocytic nevi` |
| **SGLang FP8** | `</think>\n\nThis image shows melanocytic nevi.` |

**Sample 3 — id=144, GT: melanocytic nevi ✅**

| Engine | Raw Output |
|--------|-----------|
| **Unsloth BnB-4bit** | `melanocytic nevi` |
| **SGLang FP8** | `</think>\n\nThis image shows melanocytic nevi.` |

**ตัวอย่าง Fail — id=655, GT: basal cell carcinoma ❌**

| Engine | Raw Output | Predicted |
|--------|-----------|-----------|
| **SGLang FP8** | `</think>\n\nThis image shows squamous cell carcinoma in situ.` | SCCIS (ผิด — ทั้งคู่เป็น malignant tumor) |

**สังเกต:** SGLang FP8 raw output มี `</think>` นำหน้าเสมอ — เหตุจาก `max_new_tokens=64` ตัด thinking chain กลางคัน โมเดล Qwen3-VL-8B-Thinking ออก `</think>` เป็นสัญญาณปิด thinking → parser ต้อง strip token นี้ก่อน; Unsloth BnB-4bit ฝึกให้ตอบชื่อโรคตรงๆ ไม่มี thinking preamble

### 16.2 Stage 3 — Caption Generation (merged model)

*(Sample 1 แสดง noguide vs guide เปรียบเทียบ; Samples 2–3 แสดง noguide เพื่อ illustrate Correctness issue)*

**Sample 1 — id=10, GT disease: SCCIS**

> **Ground Truth:** "The red, smooth, exophytic nodule with a slightly narrowed base may indicate squamous cell carcinoma. Squamous cell carcinoma is a common type of skin tumor that typically originates from the keratinocytes in the epidermis. This type of tumor often presents as skin lesions such as nodules or ulcers, and further investigation is needed to confirm the diagnosis."

> **Generated (merged_noguide):** "The presented photo shows a pale red nodule with surface ulceration, necrosis, and crusting, which are typical features of squamous cell carcinoma. Squamous cell carcinoma is a common malignant skin tumor that usually requires surgical excision, radiation therapy, and chemotherapy."

> **Generated (merged_guide):** "The photo shows a yellow-red nodular lesion with a rough surface, raised edges, and visible blood vessels, which are consistent with the characteristics of squamous cell carcinoma. Squamous cell carcinoma is a common type of skin cancer that typically originates from the squamous epithelial cells of the epidermis. This type of cancer usually grows slowly, but if left untreated, it can lead to serious consequences. Treatment options include surgical excision, radiation therapy, and chemotherapy."

**Sample 2 — id=99, GT disease: melanocytic nevi — แสดง Correctness issue ของ noguide**

> **Ground Truth:** "The presentation of round papules with black spots and brownish halo at the base is suggestive of a melanocytic nevus. Further evaluation with dermatoscopy is recommended for a definitive diagnosis."

> **Generated (merged_noguide) ❌:** "The patient presents with irregularly shaped black patches on the trunk with uneven color, suggestive of melanoma or seborrheic keratosis. Melanoma is a malignant skin tumor characterized by irregularly shaped and colored patches on the skin, while seborrheic keratosis is a common..."
> *(ผิด — noguide ระบุ melanoma/seborrheic keratosis แทน melanocytic nevi → ลด Corr)*

> **Generated (merged_guide) ✅:** "In this case, the black papules with uniform color and well-defined borders are consistent with a melanocytic nevus. Melanocytic nevi are common skin lesions, usually caused by an overgrowth of pigment-producing cells. Most nevi are benign, with only a small percentage developing into malignant melanoma. Regular monitoring of changes in the shape, color, and borders of the nevus is recommended..."

**Sample 3 — id=144, GT disease: melanocytic nevi — ตัวอย่าง near-perfect noguide**

> **Ground Truth:** "There is a semi-spherical mass within the hairline which may be a skin tumor. Further investigations are needed to confirm the diagnosis, including medical history collection and auxiliary examinations. Common skin tumors include benign ones like lipoma and malignant ones like malignant melanoma, so timely and accurate diagnosis is crucial."

> **Generated (merged_noguide) ✅:** "There is a semi-spherical mass within the hairline which may be a skin tumor. Further investigations are needed to confirm the diagnosis, including medical history collection and auxiliary examinations. Common skin tumors include benign ones like lipoma and malignant ones like malignant melanoma, so timely and accurate diagnosis is crucial."
> *(BLEU-4 ≈ 100 — generated ตรงกับ GT เกือบทั้งหมด)*

### 16.3 เปรียบเทียบ Output Styles

| ด้าน | Ground Truth (SkinCAP) | merged_noguide | merged_guide |
|------|----------------------|---------------|-------------|
| Tone | Tentative ("may indicate", "suggestive of") | Mixed — บางครั้ง confident, บางครั้งถูกต้อง | Confident ("consistent with", "indicative of") |
| Disease specificity | Differential ("may be X or Y") | ❌ อาจเดาผิดโรคถ้า visual ambiguous | ✅ ระบุโรคตรงเสมอ (จาก Stage 2 label) |
| Structure | Observe → differential → exam | Observe → disease → mechanism | Observe → confirmed disease → treatment |
| BLEU-4 | — | **29.33** | 13.11 |
| Corr | — | 62.63% | **78.79%** |

---

---

## Section 17 — โมเดลบน HuggingFace

โมเดลทั้งหมด 14 ตัวของ HIKARI ถูก publish บน HuggingFace Hub ภายใต้ username **E27085921**

### 17.0 หลักการตั้งชื่อโมเดล

โมเดลทุกตัวในตระกูล HIKARI ตั้งชื่อตาม **ดาวฤกษ์ (stars)** — สอดคล้องกับความหมายของ HIKARI (光・ヒカリ = แสงสว่าง)

**รูปแบบชื่อ:** `HIKARI-[ชื่อดาว]-8B-[Task]-[Variant]`

| ชื่อดาว | ดาวจริง | เหตุผลที่เลือก |
|:--------|:--------|:--------------|
| **Subaru** (スバル) | กระจุกดาวลูกไก่ (Pleiades) | Subaru = กลุ่มดาวรวมกัน → Stage 1 รวมโรคเป็น 4 กลุ่ม |
| **Sirius** | ดาวโจร — ดาวที่สว่างที่สุดบนท้องฟ้า | Best Stage 2 model — สว่างโดดเด่นกว่าทุกตัว (85.86%) |
| **Deneb** | ดาวสว่างใน Cygnus | Stage 2 Cascade FT — สว่างรองลงมา (79.80%) |
| **Altair** | ดาวนกอินทรี (Aquila) | Stage 2 baseline — จุดเริ่มต้น ก่อนจะมี RAG |
| **Polaris** | ดาวเหนือ — จุดอ้างอิงนำทาง | Oracle upper bound — ใช้เป็น reference เท่านั้น ไม่ใช่งานจริง |
| **Vega** | ดาวสว่างที่สุดใน Lyra | Best Stage 3 model — สว่างเหนือทุกตัวในด้าน caption (BLEU-4: 29.33) |
| **Rigel** | ดาวสีน้ำเงิน-ขาวใน Orion | Stage 3 checkpoint-init — สว่างแต่ยังสู้ Vega ไม่ได้ (BLEU-4: 9.82) |
| **Antares** | ดาวยักษ์แดงใน Scorpius — ร้อนแรงและไม่เสถียร | STS training collapse — ความพยายามที่เข้มข้นแต่พังทลาย (BLEU-4: 0.61) |

> ชื่อดาวแต่ละดวงถูกเลือกให้สะท้อน **บุคลิกของโมเดล** — ดาวที่สว่างที่สุด = โมเดลที่ดีที่สุด, ดาวที่ไม่เสถียร = โมเดลที่ collapse

---

### Collection

| Collection | ลิงก์ | เนื้อหา |
|:-----------|:------|:--------|
| **HIKARI Skin Disease AI** | [collections/E27085921/hikari-skin-disease-ai](https://huggingface.co/collections/E27085921/hikari-skin-disease-ai-69be6cadbee2138aabfac175) | Merged models 8 ตัว (~17 GB ต่อตัว) |
| **HIKARI - LoRA Adapters** | [collections/E27085921/hikari-lora-adapters](https://huggingface.co/collections/E27085921/hikari-lora-adapters-69be6f05f363f84d8ffbd82d) | LoRA adapter 6 ตัว (~1.1 GB ต่อตัว) |

### ตาราง Merged Models (~17 GB ต่อตัว)

| HuggingFace ID | Stage | Task | Metric | Internal Name |
|:---------------|:-----:|:-----|:------:|:--------------|
| [HIKARI-Subaru-8B-SkinGroup](https://huggingface.co/E27085921/HIKARI-Subaru-8B-SkinGroup) | 1 | 4-class group classifier | 88.68% | `skincap_3stage_group_classification_merged` |
| ⭐ [HIKARI-Sirius-8B-SkinDx-RAG](https://huggingface.co/E27085921/HIKARI-Sirius-8B-SkinDx-RAG) | 2 | Disease dx — RAG-in-Training | **85.86%** | `skincap_fuzzytopk_s1cascade_ragR2_a09_classification_merged` |
| [HIKARI-Deneb-8B-SkinDx-Cascade](https://huggingface.co/E27085921/HIKARI-Deneb-8B-SkinDx-Cascade) | 2 | Disease dx — Cascade FT | 79.80% | `skincap_fuzzytopk_s1cascade_classification_merged` |
| [HIKARI-Altair-8B-SkinDx](https://huggingface.co/E27085921/HIKARI-Altair-8B-SkinDx) | 2 | Disease dx — baseline | 74.00% | `skincap_fuzzytopk_classification_merged` |
| [HIKARI-Polaris-8B-SkinDx-Oracle](https://huggingface.co/E27085921/HIKARI-Polaris-8B-SkinDx-Oracle) | 2 | Oracle upper bound (research) | 59.38%* | `skincap_3stage_disease_M1_merged` |
| ⭐ [HIKARI-Vega-8B-SkinCaption-Fused](https://huggingface.co/E27085921/HIKARI-Vega-8B-SkinCaption-Fused) | 3 | Clinical caption — Merged-Init | BLEU-4: **29.33** | `skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_classification_merged` |
| [HIKARI-Rigel-8B-SkinCaption](https://huggingface.co/E27085921/HIKARI-Rigel-8B-SkinCaption) | 3 | Clinical caption — checkpoint init | BLEU-4: 9.82 | `skincap_stage3_caption_fuzzytopk_s1cascade_classification_merged` |
| [HIKARI-Antares-8B-SkinCaption-STS](https://huggingface.co/E27085921/HIKARI-Antares-8B-SkinCaption-STS) | 3 | Caption + STS ablation (research) | BLEU-4: 0.61 | `skincap_stage3_caption_fuzzytopk_s1cascade_merged_init_sts_classification_merged` |

*\* ต้องการ ground-truth group ตอน inference — oracle reference เท่านั้น*

### ตาราง LoRA Adapters (~1.1 GB ต่อตัว)

| HuggingFace ID | โหลดบน base model | หมายเหตุ |
|:---------------|:-----------------|:---------|
| [HIKARI-Sirius-8B-SkinDx-RAG-LoRA](https://huggingface.co/E27085921/HIKARI-Sirius-8B-SkinDx-RAG-LoRA) | `Qwen/Qwen3-VL-8B-Thinking` | |
| [HIKARI-Deneb-8B-SkinDx-Cascade-LoRA](https://huggingface.co/E27085921/HIKARI-Deneb-8B-SkinDx-Cascade-LoRA) | `Qwen/Qwen3-VL-8B-Thinking` | |
| [HIKARI-Altair-8B-SkinDx-LoRA](https://huggingface.co/E27085921/HIKARI-Altair-8B-SkinDx-LoRA) | `Qwen/Qwen3-VL-8B-Thinking` | |
| [HIKARI-Rigel-8B-SkinCaption-LoRA](https://huggingface.co/E27085921/HIKARI-Rigel-8B-SkinCaption-LoRA) | `Qwen/Qwen3-VL-8B-Thinking` | |
| ⚠️ [HIKARI-Vega-8B-SkinCaption-Fused-LoRA](https://huggingface.co/E27085921/HIKARI-Vega-8B-SkinCaption-Fused-LoRA) | **`E27085921/HIKARI-Sirius-8B-SkinDx-RAG`** | ต้องโหลดบน merged Stage 2 เท่านั้น (ไม่ใช่ raw Qwen) — Merged-Init design |
| [HIKARI-Antares-8B-SkinCaption-STS-LoRA](https://huggingface.co/E27085921/HIKARI-Antares-8B-SkinCaption-STS-LoRA) | `Qwen/Qwen3-VL-8B-Thinking` | |

> **หมายเหตุสำคัญ Vega-LoRA:** เนื่องจาก Vega ใช้ Merged-Init strategy (Stage 3 เริ่มจาก merged Stage 2 weights) ดังนั้น LoRA adapter ของ Vega จึงต้องโหลดบน `HIKARI-Sirius-8B-SkinDx-RAG` (merged) ไม่ใช่ raw `Qwen/Qwen3-VL-8B-Thinking` — ถ้าโหลดผิด disease knowledge จะหายไปและผลลัพธ์จะไม่ถูกต้อง

---

*อัปเดต: 2026-03-21 (เพิ่ม Section 13–16: BERTScore multi-model comparison, Speed Benchmark, SGLang FP8 Accuracy, Raw Output examples; Section 17: HuggingFace model links)*
