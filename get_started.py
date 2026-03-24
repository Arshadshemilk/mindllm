#!/usr/bin/env python3
"""
MindEdgeAI - START HERE SCRIPT

This is your quickest way to get the entire system working.
Just run: python get_started.py

It does everything:
1. Sets up the system
2. Trains all 3 phases
3. Does inference
4. Shows you the results

No config needed. Just run!
"""

import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def step_1_check_requirements():
    """Step 1: Verify everything is installed"""
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Checking requirements")
    logger.info("="*60)
    
    try:
        import torch
        logger.info(f"✓ PyTorch: {torch.__version__}")
    except ImportError:
        logger.error("✗ PyTorch not installed. Run: pip install torch")
        return False
    
    try:
        import transformers
        logger.info(f"✓ Transformers: {transformers.__version__}")
    except ImportError:
        logger.error("✗ Transformers not installed. Run: pip install transformers")
        return False
    
    try:
        import numpy
        logger.info(f"✓ NumPy: {numpy.__version__}")
    except ImportError:
        logger.error("✗ NumPy not installed. Run: pip install numpy")
        return False
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"✓ Device: {device}")
    
    return True


def step_2_prepare_data():
    """Step 2: Create sample training data"""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Creating sample training data")
    logger.info("="*60)
    
    # Simple sample texts for training
    training_texts = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
        "Neural networks are inspired by biological neural networks in animal brains.",
        "Deep learning uses multiple layers to progressively extract higher-level features from raw input.",
        "Natural language processing helps computers understand and generate human language.",
        "Transformers have revolutionized the field of NLP with attention mechanisms.",
        "BERT is a pre-trained language model that can be fine-tuned for many tasks.",
        "GPT models demonstrate impressive few-shot learning capabilities.",
        "Reinforcement learning trains agents to make sequences of decisions.",
        "Computer vision enables machines to interpret visual information from the world.",
        "Convolutional neural networks are effective for image classification tasks.",
        "Recurrent neural networks excel at processing sequential data.",
        "Attention mechanisms allow models to focus on relevant parts of input.",
        "Transfer learning enables reusing knowledge from one task for another.",
        "Data augmentation increases the effective size of training datasets.",
        "Batch normalization improves training speed and stability.",
    ]
    
    val_texts = training_texts[10:13]
    test_texts = training_texts[13:15]
    
    logger.info(f"Created {len(training_texts)} training texts")
    logger.info(f"Created {len(val_texts)} validation texts")
    logger.info(f"Created {len(test_texts)} test texts")
    
    return training_texts, val_texts, test_texts


def step_3_load_base_model():
    """Step 3: Load the Gemma 3 1B model"""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Loading base model (Gemma 3 1B)")
    logger.info("="*60)
    logger.info("Note: First time will download the model (~2.6GB)")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "unsloth/gemma-3-1b-it"
        logger.info(f"Loading {model_name}...")
        
        # Load model with auto device mapping
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        logger.info("✓ Model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        logger.error("Make sure you have internet and ~3GB disk space")
        return None, None


def step_4_train_system(model, tokenizer, train_texts, val_texts, test_texts):
    """Step 4: Train the complete system (Phase 1 → 2 → 3)"""
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Training complete system")
    logger.info("="*60)
    logger.info("This trains all 3 phases: multi-exit, confidence, bandit")
    
    try:
        from src.real_training_pipeline import RealTrainingPipeline
        
        # Create training pipeline
        pipeline = RealTrainingPipeline(
            checkpoint_dir="./checkpoints",
            config={
                'batch_size': 8,
                'num_epochs_phase2': 5,  # Fewer epochs for quick demo
                'learning_rate': 1e-3,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
        )
        
        # Run complete training
        logger.info("Starting Phase 1 → 2 → 3 training...")
        confidence_model, bandit_controller = pipeline.run_complete_training(
            model=model,
            tokenizer=tokenizer,
            train_texts=train_texts,
            val_texts=val_texts,
            test_texts=test_texts
        )
        
        logger.info("✓ Training complete!")
        logger.info("Checkpoints saved to ./checkpoints/")
        
        return confidence_model, bandit_controller
        
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def step_5_do_inference(model, tokenizer, confidence_model=None):
    """Step 5: Do inference with the trained system"""
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Running inference")
    logger.info("="*60)
    
    try:
        from src.production_inference import setup_production_pipeline
        
        # Setup pipeline
        logger.info("Setting up inference pipeline...")
        pipeline, config = setup_production_pipeline(
            model_path="unsloth/gemma-3-1b-it",
            confidence_model_path="./checkpoints/confidence_classifier_final.pt"
            if Path("./checkpoints/confidence_classifier_final.pt").exists()
            else None,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Example prompts
        prompts = [
            "What is machine learning?",
            "Explain deep learning",
            "How do transformers work?"
        ]
        
        logger.info("Running inference on sample prompts...")
        
        for prompt in prompts:
            logger.info(f"\nPrompt: {prompt}")
            
            # Preprocess
            inputs = pipeline.preprocess_input(prompt)
            
            # Generate with adaptive exit
            result = pipeline.generate_with_adaptive_exit(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=50  # Short for demo
            )
            
            # Show results
            exit_info = result['exit_info']
            logger.info(f"  Tokens generated: {exit_info['token_count']}")
            logger.info(f"  Early exits: {exit_info['early_exit_count']}")
            logger.info(f"  Early exit rate: {result['early_exit_rate']:.1%}")
            logger.info(f"  Avg exit layer: {sum(exit_info['exit_layers'])/len(exit_info['exit_layers']):.1f}")
        
        # Show final summary
        logger.info("\n" + "="*40)
        pipeline.log_inference_summary()
        
        return pipeline
        
    except Exception as e:
        logger.error(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def step_6_show_results(pipeline):
    """Step 6: Show final results and metrics"""
    logger.info("\n" + "="*60)
    logger.info("STEP 6: Final Results")
    logger.info("="*60)
    
    if pipeline is None:
        logger.error("Can't show results - inference failed")
        return
    
    summary = pipeline.get_inference_summary()
    
    logger.info(f"\nInference Statistics:")
    logger.info(f"  Total samples: {summary['total_samples']}")
    logger.info(f"  Total tokens: {summary['total_tokens']}")
    logger.info(f"  Avg tokens/sample: {summary['avg_tokens_per_sample']:.2f}")
    logger.info(f"  Early exit rate: {summary['early_exit_rate']:.2%}")
    logger.info(f"  Avg exit layer: {summary['avg_exit_layer']:.1f}")
    logger.info(f"  Avg confidence: {summary['avg_confidence']:.4f}")
    
    if 'bandit_metrics' in summary:
        bandit = summary['bandit_metrics']
        logger.info(f"\nBandit Learning:")
        logger.info(f"  Best threshold: {bandit.get('best_threshold', 'N/A')}")
        logger.info(f"  Best reward: {bandit.get('best_reward', 'N/A'):.4f}")
        logger.info(f"  Arm concentration: {bandit.get('arm_concentration', 'N/A'):.2%}")


def step_7_next_steps():
    """Step 7: Show what to do next"""
    logger.info("\n" + "="*60)
    logger.info("NEXT STEPS")
    logger.info("="*60)
    
    logger.info("\n1. Run inference on your own data:")
    logger.info("   from src.production_inference import setup_production_pipeline")
    logger.info("   pipeline, config = setup_production_pipeline()")
    logger.info("   result = pipeline.generate_with_adaptive_exit(...)")
    
    logger.info("\n2. Fine-tune for your task:")
    logger.info("   Modify real_training_pipeline.py with your data")
    logger.info("   Run: python scripts/train.py")
    
    logger.info("\n3. Deploy to production:")
    logger.info("   See PRODUCTION_README.md for deployment checklist")
    
    logger.info("\n4. Learn more:")
    logger.info("   Read PRODUCTION_README.md - complete API reference")
    logger.info("   See QUICKSTART_EXAMPLES.py - 10 more examples")
    
    logger.info("\n5. Understand each phase:")
    logger.info("   Phase 1: Multi-exit LLM → src/phase1_production.py")
    logger.info("   Phase 2: Confidence classifiers → src/phase2_production.py")
    logger.info("   Phase 3: Bandit controller → src/phase3_production.py")


def main():
    """Run all steps"""
    logger.info("\n")
    logger.info("╔════════════════════════════════════════════════════════╗")
    logger.info("║     MindEdgeAI - Energy-Efficient LLM Inference         ║")
    logger.info("║         Getting Started (Interactive Setup)             ║")
    logger.info("╚════════════════════════════════════════════════════════╝")
    
    # Step 1: Check requirements
    if not step_1_check_requirements():
        logger.error("\nPlease install missing requirements and try again")
        return
    
    # Step 2: Prepare data
    train_texts, val_texts, test_texts = step_2_prepare_data()
    
    # Step 3: Load model
    model, tokenizer = step_3_load_base_model()
    if model is None:
        logger.error("\nFailed to load model. Please check your setup and try again")
        return
    
    # Step 4: Train system
    confidence_model, bandit = step_4_train_system(
        model, tokenizer, train_texts, val_texts, test_texts
    )
    
    # Step 5: Do inference
    pipeline = step_5_do_inference(model, tokenizer, confidence_model)
    
    # Step 6: Show results
    if pipeline:
        step_6_show_results(pipeline)
    
    # Step 7: Next steps
    step_7_next_steps()
    
    logger.info("\n" + "="*60)
    logger.info("✓ COMPLETE! Your MindEdgeAI system is ready")
    logger.info("="*60 + "\n")


if __name__ == "__main__":
    main()
