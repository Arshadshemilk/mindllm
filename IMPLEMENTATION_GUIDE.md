# MindEdgeAI Production System - Complete Implementation Guide

## What You Have

A complete, production-ready implementation of an energy-efficient LLM inference system with three integrated phases:

### The Software Stack

```
MindEdgeAI Production System
├── Phase 1: Multi-Exit Architecture
│   └── src/phase1_production.py (300 lines)
│       ├── ExitHead - Exit point logits
│       ├── MultiExitLLMProduction - Main LLM wrapper
│       ├── TextDataset - Training data loader
│       └── load_model() - Factory function
│
├── Phase 2: Confidence Classifiers
│   └── src/phase2_production.py (400 lines)
│       ├── ConfidenceHeadProduction - Single MLP
│       ├── ConfidenceClassifierEnsembleProduction - 6-head ensemble
│       ├── GeometricBCELossProduction - Custom loss function
│       ├── ConfidenceTrainerProduction - Training pipeline
│       └── train_confidence_classifiers() - Entry point
│
├── Phase 3: Bandit Controller
│   └── src/phase3_production.py (450 lines)
│       ├── ArmStatistics - Per-arm tracking
│       ├── UCBBanditArm - Single arm with UCB
│       ├── ExplorationSchedule - 3-phase schedule
│       ├── UCBBanditControllerProduction - Main controller
│       ├── AdaptiveThresholdManagerProduction - Integration
│       └── run_online_learning_episode() - Simulation
│
├── Phase 4: Production Inference
│   └── src/production_inference.py (400 lines)
│       ├── InferenceConfig - Configuration
│       ├── ProductionInferencePipeline - Main pipeline
│       ├── BatchInferencePipeline - Batch processor
│       └── setup_production_pipeline() - Factory function
│
└── Training: End-to-End Pipeline
    └── src/real_training_pipeline.py (500 lines)
        ├── RealTrainingPipeline - Orchestrator
        ├── phase1_data_collection() - Hidden state extraction
        ├── phase2_train_confidence_classifiers() - Supervised learning
        ├── phase3_bandit_learning() - Online learning
        ├── run_complete_training() - Phase 1→2→3
        └── create_synthetic_training_data() - Demo data
```

## Quick Reference: What Each File Does

### `src/phase1_production.py`
**What it does**: Loads the Gemma 3 1B model and extracts hidden states at 6 exit layers  
**When to use**: Always needed as the base for everything else  
**Key methods**:
- `model.get_hidden_states_at_layers()` - Get features for confidence training
- `model.generate_tokens()` - Generate text with early exiting
- `model.extract_training_data()` - Prepare data for Phase 2

**Example**:
```python
from src.phase1_production import MultiExitLLMProduction
model = MultiExitLLMProduction("unsloth/gemma-3-1b-it")
hidden_states = model.get_hidden_states_at_layers(input_ids, mask)
logits = model.generate_tokens(input_ids, max_length=100)
```

### `src/phase2_production.py`
**What it does**: Trains 6 confidence classifiers (one per exit layer) to predict "will this exit layer produce correct output?"  
**When to use**: After collecting training data from Phase 1  
**Key methods**:
- `ConfidenceTrainerProduction.train_epoch()` - Single epoch
- `trainer.validate()` - Validation
- `trainer.apply_temperature_scaling()` - Calibration

**Example**:
```python
from src.phase2_production import ConfidenceTrainerProduction
trainer = ConfidenceTrainerProduction(model)
for epoch in range(20):
    trainer.train_epoch(hidden_states_dict, targets_dict)
trainer.apply_temperature_scaling(hidden_states_dict, targets_dict)
```

### `src/phase3_production.py`
**What it does**: Uses the UCB bandit algorithm to learn which confidence threshold works best during inference  
**When to use**: During/after inference to adapt thresholds  
**Key methods**:
- `controller.select_arm()` - Pick a threshold
- `controller.pull_arm()` - Update with feedback
- `controller.get_convergence_metrics()` - Check if learned

**Example**:
```python
from src.phase3_production import UCBBanditControllerProduction
controller = UCBBanditControllerProduction(num_arms=20)
arm_idx = controller.select_arm()
result = controller.pull_arm(arm_idx, confidence, exit_layer)
metrics = controller.get_convergence_metrics()
```

### `src/production_inference.py`
**What it does**: Ties all three phases together for inference  
**When to use**: When you have trained Phase 1, 2, and want to inference with Phase 3 adapting  
**Key methods**:
- `setup_production_pipeline()` - Load everything
- `pipeline.generate_with_adaptive_exit()` - Generate tokens
- `BatchInferencePipeline()` - Process many texts

**Example**:
```python
from src.production_inference import setup_production_pipeline
pipeline, config = setup_production_pipeline()
inputs = pipeline.preprocess_input("Your text")
result = pipeline.generate_with_adaptive_exit(inputs['input_ids'], inputs['attention_mask'])
```

### `src/real_training_pipeline.py`
**What it does**: Orchestrates all three phases: Phase 1 (collect data) → Phase 2 (train) → Phase 3 (learn)  
**When to use**: When starting from scratch  
**Key methods**:
- `pipeline.phase1_data_collection()` - Collect hidden states
- `pipeline.phase2_train_confidence_classifiers()` - Train confidence
- `pipeline.phase3_bandit_learning()` - Learn thresholds
- `pipeline.run_complete_training()` - Do all three

**Example**:
```python
from src.real_training_pipeline import RealTrainingPipeline
pipeline = RealTrainingPipeline()
conf_model, bandit = pipeline.run_complete_training(model, tokenizer, train_texts, val_texts, test_texts)
```

## Usage Flowchart

```
START
  ↓
Do you have:
  - Trained confidence classifiers?
  - Trained bandit?
  
  NO → run_complete_training() [real_training_pipeline.py]
       ├─ Phase 1: Collect hidden states
       ├─ Phase 2: Train confidence heads
       ├─ Phase 3: Online bandit learning
       └─ Save checkpoints
       ↓
  YES → setup_production_pipeline() [production_inference.py]
        ├─ Load model
        ├─ Load confidence classifiers
        ├─ Setup bandit
        └─ Ready for inference
        ↓
Now run inference:
  pipeline.generate_with_adaptive_exit(input_ids, attention_mask)
  │
  ├─ Bandit selects threshold
  ├─ Model generates tokens
  ├─ Confidence classifier evaluates at each exit
  ├─ Early exit if confident
  └─ Update bandit with feedback
  ↓
END
```

## The Three Phases Explained Simply

```
┌─────────────────────────────────────────────────────────┐
│                    INFERENCE PROCESS                      │
└─────────────────────────────────────────────────────────┘

Input text: "What is machine learning?"

[PHASE 1: Multi-Exit LLM]
  ↓
  Base model processes: "What is..." → hidden states
  
  At layer 6:  logits → token prediction
  At layer 10: logits → token prediction
  At layer 14: logits → token prediction
  At layer 18: logits → token prediction
  At layer 22: logits → token prediction
  At layer 26: logits → token prediction (final)
  
[PHASE 2: Confidence Classifiers]
  ↓
  For each exit:
    hidden_state → MLP(2304→256→64→1) → confidence ∈ [0,1]
  
  Confidence score: "How likely is this exit's prediction correct?"
  
[PHASE 3: Bandit Controller]
  ↓
  Current best threshold: 0.73
  
  If confidence ≥ 0.73: EXIT and emit token
  If confidence < 0.73: CONTINUE to next layer's hidden state
  
  Update bandit: "This threshold worked/didn't work"
  Learn: "Try threshold 0.75 next time"
  
Output: token + metrics
```

## Common Scenarios

### Scenario 1: Train from Scratch
```python
from src.real_training_pipeline import RealTrainingPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("unsloth/gemma-3-1b-it", device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-1b-it")

pipeline = RealTrainingPipeline(checkpoint_dir="./checkpoints")
conf_model, bandit = pipeline.run_complete_training(
    model=model,
    tokenizer=tokenizer,
    train_texts=my_training_texts,
    val_texts=my_validation_texts,
    test_texts=my_test_texts
)

print("✓ Training complete!")
print("Checkpoints: ./checkpoints/confidence_classifier_final.pt")
```

### Scenario 2: Use Pre-trained Models
```python
from src.production_inference import setup_production_pipeline

pipeline, config = setup_production_pipeline(
    model_path="unsloth/gemma-3-1b-it",
    confidence_model_path="./checkpoints/confidence_classifier_final.pt"
)

# Single inference
inputs = pipeline.preprocess_input("Your question here")
result = pipeline.generate_with_adaptive_exit(
    inputs['input_ids'],
    inputs['attention_mask'],
    max_new_tokens=100
)

print(f"Early exit rate: {result['early_exit_rate']:.1%}")
print(f"Average exit layer: {np.mean(result['exit_info']['exit_layers']):.1f}")
```

### Scenario 3: Batch Processing
```python
from src.production_inference import BatchInferencePipeline

texts = ["Question 1?", "Question 2?", "Question 3?"]

batch_pipeline = BatchInferencePipeline(pipeline, batch_size=4)
results = batch_pipeline.process_batch(texts, max_new_tokens=100)

for i, result in enumerate(results):
    print(f"Text {i}: {result['early_exit_rate']:.1%} early exits")
```

### Scenario 4: Monitor Convergence
```python
from src.phase3_production import UCBBanditControllerProduction

metrics = controller.get_convergence_metrics()

print(f"Best threshold: {metrics['best_threshold']:.4f}")
print(f"Avg regret: {metrics['avg_regret']:.4f}")
print(f"Arm concentration: {metrics['arm_concentration']:.1%}")

if metrics['arm_concentration'] > 0.70:
    print("✓ Converged! Safe to lock in best arm")
else:
    print("⏳ Still learning... wait for more samples")
```

## Performance Expectations

| Metric | Range | Target |
|--------|-------|--------|
| Early exit rate | 30-70% | 40-60% |
| Avg tokens/sample | 20-50 | <40 |
| Token latency | <100ms | <50ms |
| Confidence avg | 0.7-0.95 | >0.85 |
| ECE (calibration) | 0.02-0.10 | <0.05 |
| Bandit concentration | 50-95% | >70% |
| Best arm reward | 0.3-0.6 | >0.45 |

## Files Reference

### Core Production Files
- `src/phase1_production.py` - Multi-exit LLM (300 lines)
- `src/phase2_production.py` - Confidence training (400 lines)
- `src/phase3_production.py` - Bandit controller (450 lines)
- `src/production_inference.py` - Production pipeline (400 lines)
- `src/real_training_pipeline.py` - Training orchestration (500 lines)

### Documentation
- `PRODUCTION_README.md` - Complete API reference
- `QUICKSTART_EXAMPLES.py` - 10 copy-paste examples
- `THIS FILE` - Implementation guide

### Checkpoints (Generated During Training)
- `checkpoints/confidence_classifier.pt` - Best model during training
- `checkpoints/confidence_classifier_final.pt` - Final confidence model
- `checkpoints/bandit_checkpoint.json` - Bandit arm statistics
- `checkpoints/training_summary.json` - Training metrics

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM during training | Reduce batch size, use gradient accumulation |
| Low confidence scores | More training data, longer training |
| Bandit not converging | Increase warmup steps, check data quality |
| High ECE | More temperature scaling, more data |
| Slow inference | Check device placement, use batch processing |

## Next Steps

1. **For immediate inference**: Use `setup_production_pipeline()` with pre-trained models
2. **For training**: Use `RealTrainingPipeline().run_complete_training()`
3. **For research**: Modify individual phases (Phase 1, 2, or 3)
4. **For deployment**: Follow PRODUCTION_README.md checklist

## Key Concepts

### Exit Head
A small neural network that predicts the next token at an intermediate layer.

### Confidence Classifier
Learns whether an exit's prediction will be correct, outputs [0, 1].

### Bandit Threshold
The confidence level needed to use an exit. Learned online.

### Early Exit
When confidence ≥ threshold, output token early and skip remaining layers.

### Regret
How much worse the current arm is vs. the best arm. Target: minimize.

### Concentration
What % of time we use the best arm. Target: >70%.

## Summary

You have a complete **production-grade** system for:
- ✅ Energy-efficient LLM inference
- ✅ Online learning during deployment
- ✅ Automated threshold optimization
- ✅ Real model loading and inference
- ✅ End-to-end training pipeline

All code is professional, well-documented, and ready to use.

**Start with**: `QUICKSTART_EXAMPLES.py` for copy-paste examples
**Then read**: `PRODUCTION_README.md` for complete API documentation
**Finally deploy**: Follow the checklist in PRODUCTION_README.md

Good luck! 🚀
