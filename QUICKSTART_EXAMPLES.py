"""
Quick Start Guide for MindEdgeAI Production System

Copy-paste ready examples to get started immediately.
"""

# ============================================================================
# EXAMPLE 1: Train the Complete System (End-to-End)
# ============================================================================

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.real_training_pipeline import RealTrainingPipeline, create_synthetic_training_data
import logging

logging.basicConfig(level=logging.INFO)

# Step 1: Prepare data
print("Preparing training data...")
train_texts = create_synthetic_training_data(num_texts=50)
val_texts = create_synthetic_training_data(num_texts=20)
test_texts = create_synthetic_training_data(num_texts=30)

# Step 2: Load base model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/gemma-3-1b-it",
    device_map='auto',
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-1b-it")

# Step 3: Run complete training
print("Starting training pipeline...")
pipeline = RealTrainingPipeline(checkpoint_dir="./checkpoints")

confidence_model, bandit_controller = pipeline.run_complete_training(
    model=model,
    tokenizer=tokenizer,
    train_texts=train_texts,
    val_texts=val_texts,
    test_texts=test_texts
)

print("\n✓ Training complete!")
print("Checkpoints saved to ./checkpoints/")


# ============================================================================
# EXAMPLE 2: Do Inference With Pre-trained Models
# ============================================================================

from src.production_inference import setup_production_pipeline

# Setup pipeline with pre-trained checkpoints
pipeline, config = setup_production_pipeline(
    model_path="unsloth/gemma-3-1b-it",
    confidence_model_path="./checkpoints/confidence_classifier_final.pt",
    device="cuda"
)

# Single inference
text = "Explain quantum computing in one paragraph"
inputs = pipeline.preprocess_input(text)

result = pipeline.generate_with_adaptive_exit(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    max_new_tokens=100
)

print(f"Generated tokens: {result['exit_info']['token_count']}")
print(f"Early exit rate: {result['early_exit_rate']:.2%}")
print(f"Average exit layer: {sum(result['exit_info']['exit_layers']) / len(result['exit_info']['exit_layers']):.1f}")

# Get summary
pipeline.log_inference_summary()


# ============================================================================
# EXAMPLE 3: Batch Inference (Higher Throughput)
# ============================================================================

from src.production_inference import BatchInferencePipeline

texts = [
    "What is machine learning?",
    "Explain deep learning",
    "What is transformers architecture?",
    "How do neural networks work?",
    "What is supervised learning?",
    "Explain unsupervised learning",
    "What is reinforcement learning?",
    "How do LLMs work?"
]

# Create batch pipeline
batch_pipeline = BatchInferencePipeline(pipeline, batch_size=4)

# Process all texts
results = batch_pipeline.process_batch(texts, max_new_tokens=100)

# Print summary
print("\n=== BATCH INFERENCE RESULTS ===")
for i, result in enumerate(results):
    print(f"\nText {i+1}:")
    print(f"  Tokens generated: {result['exit_info']['token_count']}")
    print(f"  Early exits: {result['exit_info']['early_exit_count']}")
    print(f"  Exit rate: {result['early_exit_rate']:.2%}")

# Overall stats
total_tokens = sum(r['exit_info']['token_count'] for r in results)
total_early_exits = sum(r['exit_info']['early_exit_count'] for r in results)
print(f"\nTotal tokens: {total_tokens}")
print(f"Total early exits: {total_early_exits}")
print(f"Overall early exit rate: {total_early_exits / total_tokens:.2%}")


# ============================================================================
# EXAMPLE 4: Train Only Phase 2 (Confidence Classifiers)
# ============================================================================

from src.phase1_production import MultiExitLLMProduction
from src.phase2_production import ConfidenceClassifierEnsembleProduction, ConfidenceTrainerProduction
from torch.utils.data import DataLoader, TensorDataset

# Assume you already have hidden_states and targets from somewhere
# (In real scenario, collect via Phase 1)

# Initialize model and trainer
model = ConfidenceClassifierEnsembleProduction(
    exit_layer_indices=[6, 10, 14, 18, 22, 26],
    input_dim=1152,
    hidden_dim1=256,
    hidden_dim2=64,
    dropout_rate=0.2
)

trainer = ConfidenceTrainerProduction(
    model=model,
    device='cuda',
    learning_rate=1e-3,
    batch_size=32
)

# Training loop (assuming hidden_states_dict and targets_dict are prepared)
for epoch in range(20):
    train_metrics = trainer.train_epoch(hidden_states_dict, targets_dict)
    val_metrics = trainer.validate(val_hidden_dict, val_targets_dict)
    
    print(f"Epoch {epoch+1}: train_loss={train_metrics['loss']:.4f}, val_loss={val_metrics['loss']:.4f}")

# Apply calibration
trainer.apply_temperature_scaling(hidden_states_dict, targets_dict)

# Save
torch.save(model.state_dict(), "confidence_classifier.pt")
print("✓ Confidence classifier trained and saved")


# ============================================================================
# EXAMPLE 5: Train Only Phase 3 (Bandit Controller)
# ============================================================================

from src.phase3_production import UCBBanditControllerProduction
import numpy as np

# Initialize bandit
controller = UCBBanditControllerProduction(
    num_arms=20,
    threshold_range=(0.50, 0.99),
    warmup_steps=100,
    high_explore_steps=400
)

# Simulate inference with real confidence and exit layer data
# In production, this comes from actual inference
np.random.seed(42)
confidences = np.random.uniform(0.3, 0.95, 1000)
exit_layers = np.random.choice([6, 10, 14, 18, 22, 26], 1000)

# Run online learning
for confidence, exit_layer in zip(confidences, exit_layers):
    # Select arm using UCB
    arm_idx = controller.select_arm()
    
    # Pull arm with feedback
    result = controller.pull_arm(
        arm_idx=arm_idx,
        confidence=confidence,
        exit_layer=exit_layer
    )

# Log final statistics
controller.log_statistics()

# Get best arm
best_arm_idx, best_threshold = controller.get_best_arm()
print(f"\nBest arm: {best_arm_idx} (threshold={best_threshold:.4f})")


# ============================================================================
# EXAMPLE 6: Use Pre-trained Confidence Classifier in Inference
# ============================================================================

from src.production_inference import ProductionInferencePipeline, InferenceConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.phase2_production import ConfidenceClassifierEnsembleProduction
from src.phase3_production import AdaptiveThresholdManagerProduction

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "unsloth/gemma-3-1b-it",
    device_map='auto',
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-1b-it")

# Load trained confidence classifier
confidence_model = ConfidenceClassifierEnsembleProduction([6, 10, 14, 18, 22, 26])
confidence_model.load_state_dict(torch.load("./checkpoints/confidence_classifier_final.pt"))
confidence_model.to('cuda')

# Setup bandit
bandit_controller = AdaptiveThresholdManagerProduction(
    num_arms=20,
    threshold_range=(0.50, 0.99)
)

# Create inference pipeline
config = InferenceConfig(device='cuda', num_tokens_to_generate=100)
pipeline = ProductionInferencePipeline(
    model=model,
    confidence_classifier=confidence_model,
    bandit_controller=bandit_controller,
    config=config,
    tokenizer=tokenizer
)

# Do inference
text = "What are the benefits of early exiting in LLMs?"
inputs = pipeline.preprocess_input(text)
result = pipeline.generate_with_adaptive_exit(
    inputs['input_ids'],
    inputs['attention_mask'],
    max_new_tokens=100
)

# Print results
print(f"Early exit rate: {result['early_exit_rate']:.2%}")
print(f"Average confidence: {np.mean(result['exit_info']['confidences']):.4f}")
print(f"Bandit convergence: {bandit_controller.get_summary()['arm_concentration']:.2%}")


# ============================================================================
# EXAMPLE 7: Monitor Convergence in Production
# ============================================================================

from src.phase3_production import UCBBanditControllerProduction

controller = UCBBanditControllerProduction(num_arms=20)

# During production inference, periodically check convergence
convergence_metrics = controller.get_convergence_metrics()

print(f"Best arm: {convergence_metrics['best_arm']}")
print(f"Best threshold: {convergence_metrics['best_threshold']:.4f}")
print(f"Avg regret: {convergence_metrics['avg_regret']:.4f}")
print(f"Arm concentration: {convergence_metrics['arm_concentration']:.2%}")

# Check if converged (>70% on best arm)
if convergence_metrics['arm_concentration'] > 0.70:
    print("✓ Bandit has converged!")
else:
    print("⏳ Still exploring...")


# ============================================================================
# EXAMPLE 8: Evaluate Calibration of Confidence Classifiers
# ============================================================================

from src.phase2_production import ConfidenceTrainerProduction
import numpy as np

trainer = ConfidenceTrainerProduction(model)

# Get predictions
with torch.no_grad():
    predictions = model(hidden_states_dict, training=False)

# Compute ECE for each layer
for layer_idx in [6, 10, 14, 18, 22, 26]:
    if layer_idx in predictions:
        pred = predictions[layer_idx].cpu().numpy()
        target = targets_dict[layer_idx].cpu().numpy().astype(int).flatten()
        
        ece = trainer.compute_ece(pred.flatten(), target, n_bins=10)
        print(f"Layer {layer_idx} ECE: {ece:.4f}")

# Apply temperature scaling to improve calibration
trainer.apply_temperature_scaling(hidden_states_dict, targets_dict, target_ece=0.05)

print("✓ Temperature scaling applied")


# ============================================================================
# EXAMPLE 9: Custom Configuration for Different Scenarios
# ============================================================================

from src.production_inference import InferenceConfig, ProductionInferencePipeline

# Scenario 1: Low-latency (aggressive exiting)
config_low_latency = InferenceConfig(
    temperature=0.5,
    top_p=0.85,
    num_arms=10,  # Fewer arms for faster selection
    threshold_range=(0.70, 0.99)  # Higher thresholds for more exiting
)

# Scenario 2: High-quality (minimal exiting)
config_high_quality = InferenceConfig(
    temperature=0.3,
    top_p=0.95,
    num_arms=30,  # More arms for better coverage
    threshold_range=(0.30, 0.70)  # Lower thresholds for less exiting
)

# Scenario 3: Balanced
config_balanced = InferenceConfig(
    temperature=0.7,
    top_p=0.9,
    num_arms=20,
    threshold_range=(0.50, 0.99)
)

# Use appropriate config for your needs
pipeline = ProductionInferencePipeline(
    model=model,
    confidence_classifier=confidence_model,
    bandit_controller=bandit_controller,
    config=config_balanced,
    tokenizer=tokenizer
)


# ============================================================================
# EXAMPLE 10: Save and Load Complete System Checkpoint
# ============================================================================

import json

# After training, save everything
checkpoint = {
    'config': {
        'model_name': "unsloth/gemma-3-1b-it",
        'exit_layers': [6, 10, 14, 18, 22, 26],
        'num_arms': 20,
        'threshold_range': (0.50, 0.99)
    },
    'checkpoints': {
        'confidence_classifier': "./checkpoints/confidence_classifier_final.pt",
        'bandit_stats': "./checkpoints/bandit_checkpoint.json"
    }
}

with open("system_checkpoint.json", "w") as f:
    json.dump(checkpoint, f, indent=2)

print("✓ System checkpoint saved")

# Later, load and setup
with open("system_checkpoint.json") as f:
    checkpoint = json.load(f)

# Setup pipeline with loaded checkpoint
pipeline, config = setup_production_pipeline(
    model_path=checkpoint['config']['model_name'],
    confidence_model_path=checkpoint['checkpoints']['confidence_classifier']
)

print("✓ System loaded from checkpoint")


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

def monitor_production_metrics(pipeline):
    """Monitor key metrics during production"""
    
    summary = pipeline.get_inference_summary()
    
    metrics = {
        'early_exit_rate': summary['early_exit_rate'],
        'avg_confidence': summary['avg_confidence'],
        'avg_exit_layer': summary['avg_exit_layer'],
        'arm_concentration': summary['bandit_metrics']['arm_concentration'],
        'best_threshold': summary['bandit_metrics']['best_threshold']
    }
    
    # Check against targets
    issues = []
    
    if metrics['early_exit_rate'] < 0.3:
        issues.append("Early exit rate too low (target: >40%)")
    if metrics['early_exit_rate'] > 0.8:
        issues.append("Early exit rate too high (target: <60%)")
    if metrics['arm_concentration'] < 0.6:
        issues.append("Bandit not converged (target: >70%)")
    if metrics['avg_confidence'] < 0.8:
        issues.append("Confidence predictions too low")
    
    return metrics, issues

# Use it
metrics, issues = monitor_production_metrics(pipeline)

print("\n=== PRODUCTION METRICS ===")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

if issues:
    print("\n⚠️  Issues detected:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("\n✓ All metrics within target ranges")
