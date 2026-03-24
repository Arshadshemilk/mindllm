#!/usr/bin/env python3
"""
ABSOLUTE QUICKEST START - Copy & Run This

This is the MINIMUM code to:
1. Train the system
2. Do inference
3. Get results

Just run: python quick_test.py
"""

import torch
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("\n🚀 MindEdgeAI - Quick Start\n")
    
    # ========================
    # 1. LOAD MODEL
    # ========================
    logger.info("1️⃣  Loading model...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        "unsloth/gemma-3-1b-it",
        device_map='auto',
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-1b-it")
    logger.info("   ✓ Model loaded")
    
    # ========================
    # 2. CREATE SAMPLE DATA
    # ========================
    logger.info("\n2️⃣  Creating sample data...")
    
    train_texts = [
        "Machine learning is AI.",
        "Deep learning uses neural networks.",
        "Transformers are powerful models.",
        "NLP processes human language.",
        "Computer vision analyzes images.",
    ]
    
    val_texts = train_texts[2:3]
    test_texts = train_texts[3:4]
    
    logger.info(f"   ✓ {len(train_texts)} training examples")
    
    # ========================
    # 3. TRAIN SYSTEM
    # ========================
    logger.info("\n3️⃣  Training system (Phase 1 → 2 → 3)...")
    
    from src.real_training_pipeline import RealTrainingPipeline
    
    pipeline = RealTrainingPipeline(
        checkpoint_dir="./checkpoints",
        config={
            'batch_size': 8,
            'num_epochs_phase2': 3,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    )
    
    confidence_model, bandit = pipeline.run_complete_training(
        model=model,
        tokenizer=tokenizer,
        train_texts=train_texts,
        val_texts=val_texts,
        test_texts=test_texts
    )
    
    logger.info("   ✓ Training complete")
    
    # ========================
    # 4. DO INFERENCE
    # ========================
    logger.info("\n4️⃣  Running inference...")
    
    from src.production_inference import setup_production_pipeline
    from pathlib import Path
    
    pipeline, config = setup_production_pipeline(
        confidence_model_path="./checkpoints/confidence_classifier_final.pt"
        if Path("./checkpoints/confidence_classifier_final.pt").exists() else None
    )
    
    # Example inference
    text = "What is artificial intelligence?"
    inputs = pipeline.preprocess_input(text)
    result = pipeline.generate_with_adaptive_exit(
        inputs['input_ids'],
        inputs['attention_mask'],
        max_new_tokens=50
    )
    
    logger.info(f"   Input: {text}")
    logger.info(f"   Output tokens: {result['exit_info']['token_count']}")
    logger.info(f"   Early exits: {result['exit_info']['early_exit_count']}")
    logger.info(f"   Early exit rate: {result['early_exit_rate']:.1%}")
    
    # ========================
    # 5. SHOW RESULTS
    # ========================
    logger.info("\n5️⃣  Results Summary")
    logger.info("-" * 40)
    
    summary = pipeline.get_inference_summary()
    logger.info(f"   Early exit rate: {summary['early_exit_rate']:.1%}")
    logger.info(f"   Avg confidence: {summary['avg_confidence']:.3f}")
    logger.info(f"   Best threshold: {summary['bandit_metrics']['best_threshold']:.3f}")
    logger.info(f"   Convergence: {summary['bandit_metrics']['arm_concentration']:.1%}")
    
    logger.info("\n✅ SUCCESS! System is working\n")
    logger.info("Next: Read PRODUCTION_README.md or QUICKSTART_EXAMPLES.py\n")


if __name__ == "__main__":
    main()
