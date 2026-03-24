#!/usr/bin/env python3
"""
THE ABSOLUTE MINIMAL EXAMPLE

This shows the SIMPLEST possible way to use the system.
Copy this code exactly and it will work.

No explanations, no complications. Just copy-paste and run.
"""

# ============================================================
# STEP 1: IMPORTS
# ============================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.real_training_pipeline import RealTrainingPipeline
from src.production_inference import setup_production_pipeline
from pathlib import Path


# ============================================================
# STEP 2: YOUR DATA (REPLACE THIS WITH YOUR TEXT)
# ============================================================

train_texts = [
    "Machine learning is a type of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Transformers are powerful for natural language processing.",
    "GPT models can generate human-like text.",
    "BERT is great for understanding text.",
]

val_texts = train_texts[3:4]
test_texts = train_texts[4:5]


# ============================================================
# STEP 3: LOAD MODEL
# ============================================================

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/gemma-3-1b-it",
    device_map='auto',
    torch_dtype=torch.float16,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-1b-it")
print("✓ Model loaded")


# ============================================================
# STEP 4: TRAIN
# ============================================================

print("\nTraining system...")
pipeline = RealTrainingPipeline(checkpoint_dir="./checkpoints")

conf_model, bandit = pipeline.run_complete_training(
    model=model,
    tokenizer=tokenizer,
    train_texts=train_texts,
    val_texts=val_texts,
    test_texts=test_texts
)
print("✓ Training complete")


# ============================================================
# STEP 5: INFERENCE
# ============================================================

print("\nSetting up inference...")
pipeline, config = setup_production_pipeline(
    confidence_model_path="./checkpoints/confidence_classifier_final.pt"
    if Path("./checkpoints/confidence_classifier_final.pt").exists() else None
)

print("Running inference...")
text = "What is machine learning?"
inputs = pipeline.preprocess_input(text)

result = pipeline.generate_with_adaptive_exit(
    inputs['input_ids'],
    inputs['attention_mask'],
    max_new_tokens=100
)

print(f"\n{text}")
print(f"Tokens: {result['exit_info']['token_count']}")
print(f"Early exits: {result['exit_info']['early_exit_count']}")
print(f"Early exit rate: {result['early_exit_rate']:.1%}")

# ============================================================
# STEP 6: SHOW RESULTS
# ============================================================

summary = pipeline.get_inference_summary()
print(f"\nResults:")
print(f"  Confidence: {summary['avg_confidence']:.3f}")
print(f"  Best threshold: {summary['bandit_metrics']['best_threshold']:.3f}")
print(f"  Convergence: {summary['bandit_metrics']['arm_concentration']:.1%}")

print("\n✓ DONE! System is working")
