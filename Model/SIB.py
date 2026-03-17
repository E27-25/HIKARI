import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import gc
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image

# ==========================================
# 1. SIB-TinyLoRA Adapter
# ==========================================
class SIBTinyLoRALinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, r=2, u=2, shared_v=None):
        super().__init__()
        self.r, self.u = r, u
        self.weight = nn.Parameter(original_layer.weight.data.clone(), requires_grad=False)
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone(), requires_grad=False)
        else:
            self.register_parameter('bias', None)
            
        with torch.no_grad():
            U_full, S_full, Vh_full = torch.linalg.svd(self.weight.float(), full_matrices=False)
            self.register_buffer('U', U_full[:, :r].to(self.weight.dtype))
            self.register_buffer('Sigma', torch.diag(S_full[:r]).to(self.weight.dtype))
            self.register_buffer('V', Vh_full[:r, :].T.to(self.weight.dtype)) 
            P = torch.randn(u, r, r, device=self.weight.device) / (r * math.sqrt(u))
            self.register_buffer('P', P.to(self.weight.dtype))
        
        if shared_v is not None:
            self.v = shared_v
        else:
            self.v = nn.Parameter(torch.zeros(u, dtype=self.weight.dtype, device=self.weight.device))

    def forward(self, x):
        base_output = F.linear(x, self.weight, self.bias)
        # คำนวณใน FP32 เพื่อป้องกัน Float16 Overflow (ค่าเกิน 65504 แล้วเกิด NaN)
        R_hat = torch.tensordot(self.v.float(), self.P.float(), dims=([0], [0]))
        adap_up = x.float() @ self.V.float() @ R_hat.t() @ self.Sigma.float() @ self.U.float().t()
        
        # บวกแบบ FP32 ก่อน เพื่อป้องกัน Float16 Overflow
        final_output = base_output.float() + adap_up
        
        # ป้องกันไม่ให้ค่าเกิน Limits ของ Float16 ที่จะทำให้เกิด infinity หรือ NaN
        if x.dtype == torch.float16:
            final_output = torch.clamp(final_output, min=-65000.0, max=65000.0)
            
        return final_output.to(x.dtype)

def inject_vlm_tiny_lora(model, target_modules=["q_proj", "v_proj"], r=2, u=2, k_tie=5):
    for param in model.parameters(): param.requires_grad = False
    replaced_count, tied_group_counter = 0, 0
    current_shared_v = None
    for name, module in dict(model.named_modules()).items():
        if any(name.endswith(target) for target in target_modules) and isinstance(module, nn.Linear):
            if tied_group_counter % k_tie == 0:
                current_shared_v = nn.Parameter(torch.zeros(u, device=module.weight.device, dtype=module.weight.dtype))
            tied_group_counter += 1
            tiny_lora_layer = SIBTinyLoRALinear(module, r, u, current_shared_v)
            parent_name = ".".join(name.split(".")[:-1])
            setattr(model.get_submodule(parent_name) if parent_name else model, name.split(".")[-1], tiny_lora_layer)
            replaced_count += 1
    return model

# ==========================================
# 2. Loss Functions (SFT vs STS) - [NaN Proof]
# ==========================================
def standard_sft_loss(adapter_logits, labels):
    """Loss พื้นฐาน (Cross-Entropy ธรรมดา ไม่มีการถ่วงน้ำหนักศัพท์แพทย์)"""
    shift_logits = adapter_logits[..., :-1, :].contiguous().to(torch.float32)
    shift_labels = labels[..., 1:].contiguous()
    
    B, T, V = shift_logits.shape
    loss = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1), ignore_index=-100)
    return loss

class MedicalSTSLossFast(nn.Module):
    def __init__(self, medical_vocab_ids, gamma=0.5, alpha=0.3, sigma=100.0):
        super().__init__()
        self.medical_vocab_ids = medical_vocab_ids
        self.gamma = gamma
        self.alpha = alpha
        self.sigma = sigma

    def forward(self, adapter_logits, labels, precomputed_surp_weights):
        # คำนวณใน FP32 เพื่อป้องกัน Underflow จนเกิด NaN
        adapter_logits = adapter_logits.to(torch.float32)
        precomputed_surp_weights = precomputed_surp_weights.to(torch.float32)
        
        shift_logits = adapter_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_surp = precomputed_surp_weights[..., 1:].contiguous()
        
        B, T, V = shift_logits.shape
        ce_loss_unreduced = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1), reduction='none', ignore_index=-100).view(B, T)
        ce_loss_unreduced = torch.clamp(ce_loss_unreduced, min=1e-5)
        
        valid_mask = (shift_labels != -100).float()

        w_reason = torch.full((B, T), self.gamma, device=shift_logits.device)
        w_ans = torch.ones((B, T), device=shift_logits.device)
        
        for b in range(B):
            valid_idx = valid_mask[b].nonzero(as_tuple=True)[0]
            if len(valid_idx) > 0:
                T_ans = valid_idx[-1].item()
            else:
                T_ans = T - 1

            for t in range(T):
                if shift_labels[b, t].item() in self.medical_vocab_ids:
                    w_reason[b, t] = 1.0
                
                dist_sq = (T_ans - t) ** 2
                w_ans[b, t] = self.alpha + (1.0 - self.alpha) * math.exp(-dist_sq / (2 * self.sigma ** 2))
        
        # ป้องกันค่า Surprise Weight เป็น 0 หรือเกิน 1
        shift_surp = torch.clamp(shift_surp, min=1e-5, max=1.0)
        w_combined = w_ans * w_reason * shift_surp * valid_mask
        
        # ป้องกันการหาร 0
        mean_weights = (w_combined.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1e-5)).unsqueeze(1)
        w_normalized = w_combined / mean_weights.clamp(min=1e-8)
        
        weighted_loss = ce_loss_unreduced * w_normalized
        
        # 🌟 Feature 1: STS-Max (Extreme Target Clamping)
        # Force the loss scalar on target (medical) words to strictly override grammar
        sts_max_mask = (w_normalized > 1.5).float() # Identify critical targets
        w_maxed = w_normalized + (sts_max_mask * 5.0) # Apply +5.0 scaler exclusively
        
        weighted_loss = ce_loss_unreduced * w_maxed
        final_loss = weighted_loss.sum() / valid_mask.sum().clamp(min=1)
        
        if torch.isnan(final_loss):
            return torch.tensor(0.0, device=final_loss.device, requires_grad=True)
        return final_loss

def get_model_ibr_loss(model):
    v_to_sigma0_sq = {}
    for module in model.modules():
        if isinstance(module, SIBTinyLoRALinear):
            if module.v not in v_to_sigma0_sq:
                v_to_sigma0_sq[module.v] = []
            sigmas = torch.diagonal(module.Sigma)
            if len(sigmas) > 1:
                sigma0_sq = torch.var(sigmas, unbiased=False) + 1e-8
            else:
                sigma0_sq = torch.tensor(1.0, device=module.Sigma.device)
            v_to_sigma0_sq[module.v].append(sigma0_sq)
            
    total_ibr = 0.0
    for v, vars_list in v_to_sigma0_sq.items():
        avg_var = sum(vars_list) / len(vars_list)
        total_ibr += (v ** 2).sum() / (2 * avg_var)
    return total_ibr

# ==========================================
# 3. Dataset & Evaluation Helper
# ==========================================
# ==========================================
# ASR Noise Label Templates
# Simulates real clinical speech-to-text output:
# doctor dictates diagnosis → ASR adds fillers, hesitations,
# self-corrections, and run-on phrases around the disease name.
# SFT must waste ALL TinyLoRA capacity on filler tokens.
# STS amplifies ONLY the high-surprisal disease token via Loss weighting.
# ==========================================
# ==========================================
# === Verbose Report Labels ===
# We train the model to output a full clinical report instead of just a 1-word disease name.
# This forces SFT to waste limited TinyLoRA parameters on grammar, while STS ignores grammar
# and focuses exclusively on the high-surprisal disease name token.

class FastSkinCAPDataset(Dataset):
    def __init__(self, data_list, processor):
        self.data = data_list
        self.processor = processor
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        image.thumbnail((336, 336), Image.Resampling.LANCZOS)
        
        disease_name = str(item['disease']).lower().strip()
        target_caption = str(item['description']).strip()
        
        if disease_name not in target_caption.lower():
            target_caption = f"{target_caption} Consistent with {disease_name}."
            
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the skin condition in this image and provide the exact diagnosis."}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": target_caption}]}
        ]
        return messages, image, item['p_base'], disease_name, target_caption

def fast_collate_fn(batch, processor):
    messages_list = [item[0] for item in batch]
    images_list = [item[1] for item in batch]
    p_base_list = [item[2] for item in batch]
    
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages_list]
    inputs = processor(text=texts, images=images_list, padding=True, return_tensors="pt")
    labels = inputs["input_ids"].clone()
    
    # ป้องกัน Pad Token หลุดเข้าไปใน Label
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
        
    surp_weights = torch.ones_like(labels, dtype=torch.float32)
    
    for i, msg in enumerate(messages_list):
        prompt_only = [msg[0]] 
        prompt_text = processor.apply_chat_template(prompt_only, tokenize=False, add_generation_prompt=True)
        prompt_inputs = processor(text=[prompt_text], images=[images_list[i]], padding=False, return_tensors="pt")
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[i, :prompt_len] = -100 
        
        valid_idx = (labels[i] != -100).nonzero(as_tuple=True)[0]
        for j, v_idx in enumerate(valid_idx):
            if j < len(p_base_list[i]):
                surp_weights[i, v_idx] = 1.0 - p_base_list[i][j]
            
    inputs["labels"] = labels
    inputs["precomputed_surp_weights"] = surp_weights
    return inputs

def evaluate_model(model, processor, test_dataset, device="cuda"):
    """Evaluate model accuracy with exact substring match."""
    model.eval()
    correct = 0
    total = len(test_dataset)
    
    print(f"\n[Eval] Evaluating on {total} Test Images...")
    for i in tqdm(range(total), desc="Testing"):
        item = test_dataset.data[i]
        expected_disease = str(item['disease']).lower().strip()
        
        image = Image.open(item['image_path']).convert('RGB')
        image.thumbnail((336, 336), Image.Resampling.LANCZOS)
        
        prompt_text = "Describe the skin condition in this image and provide the exact diagnosis."
        max_new_tokens = 60
        
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text}
        ]}]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            model.gradient_checkpointing_disable() 
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id
            )
            
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        generated_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].lower()
        
        if expected_disease in generated_text:
            correct += 1
            
        if i < 3:
            print(f"\n[EVAL {i}] Expected: {expected_disease}")
            print(f"[EVAL {i}] Generated: {generated_text}")
            
        del inputs, generated_ids
        torch.cuda.empty_cache()

    acc = (correct / total) * 100 if total > 0 else 0
    return acc

# ==========================================
# 4. Few-Shot Cap Helper
# ==========================================
def few_shot_limit_per_class(data, n_per_class=10):
    """Limit training data to at most n_per_class samples per disease category.
    This simulates an extreme data scarcity scenario to stress-test each training method."""
    from collections import defaultdict
    counts = defaultdict(int)
    result = []
    for item in data:
        label = str(item['disease']).lower().strip()
        if counts[label] < n_per_class:
            result.append(item)
            counts[label] += 1
    print(f"[Few-Shot] Capped training set: {len(result)} samples ({n_per_class} per class)")
    return result

# ==========================================
# 5. The A/B Testing Pipeline
# ==========================================
def run_ab_testing():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = "/content/SkinCAP" if os.path.exists("/content/SkinCAP") else "./SkinCAP"
    json_path = os.path.join(base_dir, "top10_stage3_precomputed.json")
    
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    
    # โหลด Data และแบ่ง Train 80% / Test 20%
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    print(f"[Experiment] Train: {len(train_data)} | Test: {len(test_data)}")
    unique_diseases_sorted = sorted(list(set([str(item['disease']).lower().strip() for item in data])))
    
    # 1. SETUP DATASETS
    train_dataset = FastSkinCAPDataset(train_data, processor)
    test_dataset  = FastSkinCAPDataset(test_data, processor)

    print(f"\n[Experiment] Train: {len(train_data)} | Test: {len(test_data)}")

    # 2. BASE MODEL (Zero-Shot)
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    base_model.eval()
    print("\n[Zero-Shot] Evaluating Base Model...")
    acc_zero = evaluate_model(base_model, processor, test_dataset, device)
    del base_model; gc.collect(); torch.cuda.empty_cache()

    # 3. STS LOSS INIT
    medical_vocab_ids = set()
    for disease in unique_diseases_sorted:
        medical_vocab_ids.update(processor.tokenizer.encode(disease, add_special_tokens=False))
    sts_loss_fn = MedicalSTSLossFast(medical_vocab_ids=medical_vocab_ids, gamma=0.5, alpha=0.3, sigma=100.0).to(device)

    # 4. TRAINING FUNCTION
    def train_adapter(mode="SFT", epochs=5):
        print(f"\n[{mode}] Initializing Base Model & LoRA...")
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
        model = inject_vlm_tiny_lora(model, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], r=2, u=2)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable_params, lr=2e-5, weight_decay=0.01)
        grad_accum = 8
        
        epoch_accs = []
        _loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda b: fast_collate_fn(b, processor))
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            for step, batch in enumerate(tqdm(_loader, desc=f"Ep {epoch+1} ({mode})")):
                inputs = {k: v.to(device) for k, v in batch.items()}
                labels = inputs.pop("labels")
                surp_weights = inputs.pop("precomputed_surp_weights")
                inputs['use_cache'] = False 
                
                adapter_logits = model(**inputs).logits
                
                if mode == "SFT":
                    loss = standard_sft_loss(adapter_logits, labels)
                else:
                    sts_loss = sts_loss_fn(adapter_logits, labels, surp_weights)
                    ibr_loss = get_model_ibr_loss(model)
                    loss = sts_loss + 0.01 * ibr_loss
                
                (loss / grad_accum).backward()
                
                if (step + 1) % grad_accum == 0 or (step + 1) == len(_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    
                del adapter_logits, loss
            
            # Aggressive Memory clear per epoch
            gc.collect()
            torch.cuda.empty_cache()
            
            # Evaluate after each epoch
            print(f"\n[{mode}] End of Epoch {epoch+1} -> Evaluating on Test Set...")
            acc = evaluate_model(model, processor, test_dataset, device)
            epoch_accs.append(acc)
            print(f"-> {mode} Epoch {epoch+1} Disease Accuracy: {acc:.2f}%")
            model.train() # Reset back to training
            
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        return epoch_accs

    # 5. RUN AB TESTING
    SEED = 42
    
    # SFT
    torch.manual_seed(SEED)
    sft_accs = train_adapter(mode="SFT", epochs=5)
    
    # STS
    torch.manual_seed(SEED)
    sts_accs = train_adapter(mode="STS", epochs=5)
    
    # 6. PRINT RESULTS
    print("\n" + "="*50)
    print(" STAGE 3 CAPTION GENERATOR RESULTS (Disease Acc)")
    print("="*50)
    print(f" Zero-Shot Acc: {acc_zero:.2f}%")
    print("-" * 50)
    for i in range(len(sft_accs)):
        print(f" [Epoch {i+1}] SFT: {sft_accs[i]:.2f}% | STS: {sts_accs[i]:.2f}% | Gap: {sts_accs[i] - sft_accs[i]:+.2f}%")
    print("="*50)

import os
import pandas as pd
import torch
import json
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import gc

def run_precompute_step():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = "/content/SkinCAP" if os.path.exists("/content/SkinCAP") else "./SkinCAP"
    csv_path = os.path.join(base_dir, "skincap_v240623.csv")
    img_dir = os.path.join(base_dir, "skincap")
    
    # Stage 3 Caption Precomputation requires the same prompt and target data as training
    import pandas as pd
        
    # 1. Load Data
    df = pd.read_csv(csv_path).dropna(subset=['caption_zh_polish_en', 'disease', 'skincap_file_path'])
    top10_diseases = df['disease'].value_counts().nlargest(10).index.tolist()
    df_top10 = df[df['disease'].isin(top10_diseases)].reset_index(drop=True)
    print(f"  Top 10 : {top10_diseases}")
    print(f" : {len(df_top10)} ")
    
    # 2. Load Base Model
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id)
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    
    precomputed_data = []
    
    print("\n  Base Probabilities ...")
    for idx, row in tqdm(df_top10.iterrows(), total=len(df_top10)):
        img_path = os.path.join(img_dir, str(row['skincap_file_path']))
        try:
            image = Image.open(img_path).convert('RGB')
            image.thumbnail((336, 336), Image.Resampling.LANCZOS)
        except Exception:
            continue
            
        disease_name = str(row['disease']).lower().strip()
        target_caption = str(row['caption_zh_polish_en']).strip()
        
        if disease_name not in target_caption.lower():
            target_caption = f"{target_caption} Consistent with {disease_name}."
        
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe the skin condition in this image and provide the exact diagnosis."}
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": target_caption}]}
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # Load to GPU
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)
        labels = inputs["input_ids"].clone()
        
        prompt_only = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the skin condition in this image and provide the exact diagnosis."}]}]
        prompt_text = processor.apply_chat_template(prompt_only, tokenize=False, add_generation_prompt=True)
        prompt_inputs = processor(text=[prompt_text], images=[image], padding=False, return_tensors="pt")
        prompt_len = prompt_inputs["input_ids"].shape[1]
        labels[0, :prompt_len] = -100
        
        # คำนวณ (ใช้ VRAM น้อยมากเพราะภาพถูกหั่นขนาดแล้ว)
        with torch.no_grad():
            logits = model(**inputs).logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # ใช้ float32 เฉพาะตอนทำ softmax เพื่อความแม่นยำ
            probs = torch.nn.functional.softmax(shift_logits.to(torch.float32), dim=-1)
            target_probs = torch.gather(probs.view(-1, probs.size(-1)), 1, shift_labels.view(-1, 1).clamp(min=0)).view(1, -1)
            
            valid_mask = shift_labels != -100
            token_p_base = target_probs[valid_mask].cpu().tolist()
            
        precomputed_data.append({
            "image_path": img_path,
            "description": target_caption,
            "disease": disease_name,
            "p_base": token_p_base
        })
        
        # 🚨 เคลียร์ขยะแบบดุดัน เพื่อคืน VRAM ทันที
        del inputs, logits, probs, target_probs, shift_logits, shift_labels, valid_mask
        gc.collect()
        torch.cuda.empty_cache()

    output_path = os.path.join(base_dir, "top10_stage3_precomputed.json")
    with open(output_path, 'w') as f:
        json.dump(precomputed_data, f)
    print(f"\n : {output_path}")

if __name__ == "__main__":
    base_dir = "/content/SkinCAP" if os.path.exists("/content/SkinCAP") else "./SkinCAP"
    if not os.path.exists(os.path.join(base_dir, "top10_stage3_precomputed.json")):
        run_precompute_step()
    run_ab_testing()
