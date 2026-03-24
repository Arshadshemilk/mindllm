#!/usr/bin/env python3
"""
Train MindEdgeAI on your intents.json data

This script:
1. Loads patterns from src/intents.json
2. Trains Phase 1 → 2 → 3
3. Saves checkpoints
4. Shows inference results
"""

import torch
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_intents_as_training_data(intents_path='src/intents.json'):
    """Load patterns from intents.json as training texts"""
    
    with open(intents_path, 'r') as f:
        data = json.load(f)
    
    training_texts = []
    
    # Extract all patterns from intents
    for intent in data['intents']:
        patterns = intent.get('patterns', [])
        training_texts.extend(patterns)
    
    # Remove empty strings
    training_texts = [t.strip() for t in training_texts if t.strip()]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_texts = []
    for text in training_texts:
        if text not in seen:
            seen.add(text)
            unique_texts.append(text)
    
    logger.info(f"Loaded {len(unique_texts)} unique training patterns from intents.json")
    return unique_texts


def main():
    logger.info("\n" + "="*60)
    logger.info("TRAINING MINDEDGEAI ON INTENTS.JSON DATA")
    logger.info("="*60)
    
    # Step 1: Load data
    logger.info("\nStep 1: Loading training data from intents.json...")
    train_texts = load_intents_as_training_data()
    
    if len(train_texts) < 5:
        logger.error(f"Not enough training data ({len(train_texts)} texts). Need at least 5.")
        return
    
    # Split: 70% train, 15% val, 15% test
    n_train = int(0.7 * len(train_texts))
    n_val = int(0.15 * len(train_texts))
    
    train_texts_split = train_texts[:n_train]
    val_texts_split = train_texts[n_train:n_train + n_val]
    test_texts_split = train_texts[n_train + n_val:]
    
    logger.info(f"  ✓ Training texts: {len(train_texts_split)}")
    logger.info(f"  ✓ Validation texts: {len(val_texts_split)}")
    logger.info(f"  ✓ Test texts: {len(test_texts_split)}")
    
    # Step 2: Load model
    logger.info("\nStep 2: Loading model...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'unsloth/gemma-3-1b-it'
    
    logger.info(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32
    )
    model.eval()
    logger.info(f"  ✓ Model loaded on {device}")
    
    # Step 3: Train
    logger.info("\nStep 3: Training complete pipeline (Phase 1→2→3)...")
    
    from src.real_training_pipeline import RealTrainingPipeline
    
    config = {
        'device': device,
        'learning_rate': 1e-3,
        'batch_size': 8,
        'num_epochs': 5,
        'checkpoint_dir': 'checkpoints'
    }
    
    pipeline = RealTrainingPipeline(
        data_path="./data",
        checkpoint_dir="./checkpoints",
        config=config
    )
    
    try:
        confidence_model, bandit_controller = pipeline.run_complete_training(
            model=model,
            tokenizer=tokenizer,
            train_texts=train_texts_split,
            val_texts=val_texts_split,
            test_texts=test_texts_split
        )
        
        logger.info("\n" + "="*60)
        logger.info("✓ TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info("\nCheckpoints saved:")
        logger.info("  • ./checkpoints/confidence_classifier_final.pt")
        logger.info("  • ./checkpoints/bandit_checkpoint.json")
        logger.info("  • ./checkpoints/training_summary.json")
        
        # Step 4: Test inference
        logger.info("\n" + "="*60)
        logger.info("STEP 4: TESTING INFERENCE ON INTENTS DATA")
        logger.info("="*60)
        
        from src.production_inference import setup_production_pipeline
        
        test_pipeline, test_config = setup_production_pipeline(
            confidence_model_path='checkpoints/confidence_classifier_final.pt',
            device=device
        )
        
        # Test on a few intent patterns
        test_prompts = train_texts_split[:3]
        
        logger.info("\nSample inference:")
        for prompt in test_prompts:
            try:
                # Preprocess
                inputs = test_pipeline.preprocess_input(prompt)
                
                # Generate
                result = test_pipeline.generate_with_adaptive_exit(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=50
                )
                
                # Decode output
                generated_text = test_pipeline.tokenizer.decode(
                    result['generated_tokens'][0], 
                    skip_special_tokens=True
                )
                
                exit_info = result['exit_info']
                avg_exit = sum(exit_info['exit_layers']) / len(exit_info['exit_layers'])
                avg_conf = sum(exit_info['confidences']) / len(exit_info['confidences'])
                
                logger.info(f"\n  Input: {prompt}")
                logger.info(f"  Output: {generated_text[:100]}...")
                logger.info(f"  Avg Exit Layer: {avg_exit:.1f}")
                logger.info(f"  Avg Confidence: {avg_conf:.3f}")
            except Exception as e:
                logger.warning(f"  Inference failed for '{prompt}': {e}")
                import traceback
                traceback.print_exc()
        
        logger.info("\n" + "="*60)
        logger.info("✓ ALL DONE!")
        logger.info("="*60)
        logger.info("\nYou can now:")
        logger.info("  1. Use the model for inference on your intents")
        logger.info("  2. Deploy to production")
        logger.info("  3. Fine-tune on more data")
        
    except Exception as e:
        logger.error(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
