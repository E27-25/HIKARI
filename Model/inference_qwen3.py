"""
Inference script for testing the trained Qwen3-VL model on SkinCAP dataset
Based on Qwen3-VL-8B-Thinking fine-tuned model
"""

import os
os.environ["UNSLOTH_DISABLE_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from unsloth import FastVisionModel
import json
from datetime import datetime


class SkinLesionInference:
    def __init__(self, model_path="./qwen3_vl_8b_skincap_final"):
        """Initialize inference with the trained Qwen3-VL model."""
        print("Loading trained Qwen3-VL model...")
        
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_path,
            load_in_4bit=True,
        )
        FastVisionModel.for_inference(self.model)
        self.model.eval()
        
        self.instruction = "Describe this skin lesion image in detail. Include information about its appearance, possible diagnosis, and recommended examinations."
        
        print("Qwen3-VL model loaded and ready for inference!")
    
    def predict(self, image_path, max_tokens=2048):
        """Generate prediction for a single image."""
        image = Image.open(image_path).convert("RGB")
        
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a medical imaging expert. Think step-by-step in <think> tags before providing your final answer. Analyze carefully and show your reasoning process."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.instruction + " Think through your analysis step by step."},
                ],
            },
        ]
        
        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            max_length=8192,  # Increase max_length to avoid truncation
            truncation=False,  # Disable truncation to prevent image token mismatch
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "assistant" in generated_text.lower():
            response = generated_text.split("assistant")[-1].strip()
        else:
            response = generated_text
        
        return response
    
    def predict_batch(self, image_paths, max_tokens=1024):
        """Generate predictions for multiple images."""
        results = []
        
        for i, img_path in enumerate(image_paths):
            print(f"Processing {i+1}/{len(image_paths)}: {img_path}")
            try:
                prediction = self.predict(img_path, max_tokens)
                results.append({
                    "image": str(img_path),
                    "prediction": prediction,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "image": str(img_path),
                    "prediction": None,
                    "status": f"error: {str(e)}"
                })
        
        return results
    
    def evaluate_on_dataset(self, csv_path, image_base_path, num_samples=999999, output_file="./evaluation_results_qwen3.json"):
        """Evaluate model on a subset of the dataset and compare with ground truth."""
        print(f"\nEvaluating Qwen3-VL on {num_samples} samples...")
        
        # Load dataset
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["skincap_file_path", "caption_zh_polish_en"])
        
        # Sample random rows
        if num_samples < len(df):
            df_sample = df.sample(n=num_samples, random_state=42)
        else:
            df_sample = df
        
        results = []
        
        for idx, row in df_sample.iterrows():
            img_path = Path(image_base_path) / row["skincap_file_path"]
            ground_truth = row["caption_zh_polish_en"]
            
            if img_path.exists():
                print(f"\nProcessing: {row['skincap_file_path']}")
                print(f"Disease: {row.get('disease', 'N/A')}")
                
                try:
                    prediction = self.predict(str(img_path))
                    
                    result = {
                        "id": int(row.get("id", idx)),
                        "image_path": str(row["skincap_file_path"]),
                        "disease": row.get("disease", "N/A"),
                        "ground_truth": ground_truth,
                        "prediction": prediction,
                        "status": "success"
                    }
                    
                    print(f"\n--- Ground Truth ---")
                    print(ground_truth[:200] + "..." if len(ground_truth) > 200 else ground_truth)
                    print(f"\n--- Prediction ---")
                    print(prediction[:200] + "..." if len(prediction) > 200 else prediction)
                    
                except Exception as e:
                    result = {
                        "id": int(row.get("id", idx)),
                        "image_path": str(row["skincap_file_path"]),
                        "disease": row.get("disease", "N/A"),
                        "ground_truth": ground_truth,
                        "prediction": None,
                        "status": f"error: {str(e)}"
                    }
                
                results.append(result)
        
        # Save results
        output = {
            "model": "qwen3_vl_8b_skincap",
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(results),
            "results": results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n\nResults saved to {output_file}")
        
        # Print summary
        successful = sum(1 for r in results if r["status"] == "success")
        print(f"\nSummary:")
        print(f"  Total samples: {len(results)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {len(results) - successful}")
        
        return results


def interactive_inference(model_path="./qwen3_vl_8b_skincap_final"):
    """Interactive mode for testing individual images."""
    print("=" * 60)
    print("Skin Lesion Analysis - Interactive Mode (Qwen3-VL)")
    print("=" * 60)
    
    inferencer = SkinLesionInference(model_path)
    
    while True:
        print("\n" + "-" * 40)
        image_path = input("Enter image path (or 'quit' to exit): ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
        
        if not Path(image_path).exists():
            print(f"Error: Image not found at {image_path}")
            continue
        
        print("\nAnalyzing image...")
        prediction = inferencer.predict(image_path)
        
        print("\n" + "=" * 40)
        print("ANALYSIS RESULT:")
        print("=" * 40)
        print(prediction)


def main():
    """Main function to run inference tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained Qwen3-VL skin lesion model')
    parser.add_argument('--mode', type=str, default='evaluate',
                        choices=['single', 'evaluate', 'interactive'],
                        help='Inference mode')
    parser.add_argument('--image', type=str, default='./SkinCAP/skincap/1.png',
                        help='Path to single image (for single mode)')
    parser.add_argument('--model', type=str, default='./qwen3_vl_8b_skincap_final',
                        help='Path to trained Qwen3-VL model')
    parser.add_argument('--csv', type=str, default='./SkinCAP/skincap_v240623.csv',
                        help='Path to dataset CSV')
    parser.add_argument('--image_dir', type=str, default='./SkinCAP/skincap',
                        help='Base directory for images')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples for evaluation')
    parser.add_argument('--output', type=str, default='./evaluation_results_qwen3.json',
                        help='Output file for evaluation results')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        # Single image inference
        print("Single Image Inference (Qwen3-VL)")
        print("=" * 50)
        
        inferencer = SkinLesionInference(args.model)
        
        if not Path(args.image).exists():
            print(f"Error: Image not found at {args.image}")
            return
        
        print(f"\nAnalyzing: {args.image}")
        prediction = inferencer.predict(args.image)
        
        print("\n" + "=" * 50)
        print("PREDICTION:")
        print("=" * 50)
        print(prediction)
        
    elif args.mode == 'evaluate':
        # Evaluate on dataset subset
        print("Dataset Evaluation (Qwen3-VL)")
        print("=" * 50)
        
        inferencer = SkinLesionInference(args.model)
        inferencer.evaluate_on_dataset(
            args.csv,
            args.image_dir,
            args.num_samples,
            args.output
        )
        
    elif args.mode == 'interactive':
        # Interactive mode
        interactive_inference(args.model)


if __name__ == "__main__":
    main()
