# MindEdgeAI: Energy-Aware Multi-Exit LLM System

> An advanced inference system combining **multi-exit architecture**, **confidence estimation**, and **online bandit learning** for energy-efficient large language model generation.

## Overview

MindEdgeAI implements a sophisticated three-phase system for reducing energy consumption during LLM inference while maintaining quality. Optimized for **Gemma 3 1B** inference!

### Phase 1: Multi-Exit Architecture with KV Cache Management
- Inject early exit heads at designated layers from Gemma 3 1B (6, 10, 14, 18, 22, 26)
- Maintain continuous context via intelligent KV cache propagation
- Enable dynamic early exiting based on confidence thresholds

### Phase 2: Meta Confidence Classification (Offline Training)
- Lightweight MLP heads for confidence estimation at each exit point
- Geometric-weighted loss function with discount factor (γ=0.9)
- Temperature scaling for calibration (target ECE <0.05)

### Phase 3: Bandit Controller (Online Learning)
- UCB (Upper Confidence Bound) algorithm for adaptive threshold selection
- Three-phase exploration strategy:
  - **Warmup** (0-100 tokens): Forced round-robin
  - **Exploration** (100-500 tokens): High exploration (c=3.0)
  - **Standard** (500+ tokens): Balanced exploration (c=2.0)
- Dynamic reward function balancing quality and energy cost

## PRODUCTION CODE 🚀

**Real, professional-grade code for production deployment** — not demos or examples!

### Production Modules

| Module | Purpose | Status |
|--------|---------|--------|
| `phase1_production.py` | Multi-exit LLM with real model loading | ✅ Production |
| `phase2_production.py` | Confidence classifier training pipeline | ✅ Production |
| `phase3_production.py` | Bandit controller with online learning | ✅ Production |
| `production_inference.py` | Complete inference integration | ✅ Production |
| `real_training_pipeline.py` | End-to-end training (Phase 1→2→3) | ✅ Production |

### Quick Start

```python
# Full system training (Phase 1 → Phase 2 → Phase 3)
from src.real_training_pipeline import RealTrainingPipeline
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("unsloth/gemma-3-1b-it", device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-1b-it")

pipeline = RealTrainingPipeline(checkpoint_dir="./checkpoints")
confidence_model, bandit = pipeline.run_complete_training(
    model=model,
    tokenizer=tokenizer,
    train_texts=my_data,
    val_texts=my_val,
    test_texts=my_test
)
```

### Production Inference

```python
# Setup pre-trained system
from src.production_inference import setup_production_pipeline

pipeline, config = setup_production_pipeline(
    model_path="unsloth/gemma-3-1b-it",
    confidence_model_path="./checkpoints/confidence_classifier_final.pt"
)

# Single inference with adaptive exiting
inputs = pipeline.preprocess_input("Your text here")
result = pipeline.generate_with_adaptive_exit(
    inputs['input_ids'],
    inputs['attention_mask'],
    max_new_tokens=100
)

# Batch processing
from src.production_inference import BatchInferencePipeline
batch_pipeline = BatchInferencePipeline(pipeline, batch_size=8)
results = batch_pipeline.process_batch(texts, max_new_tokens=100)
```

### Documentation

- **[PRODUCTION_README.md](PRODUCTION_README.md)** — Complete production documentation with all APIs and examples
- **[QUICKSTART_EXAMPLES.py](QUICKSTART_EXAMPLES.py)** — Copy-paste ready code samples for all scenarios

## Architecture

```
Input Tokens
    ↓
[Gemma 3 1B (Unsloth Optimized) - 26 Layers, 2304D]
    ↓
Layer 6  → Exit Head 1 → Confidence Head 1
Layer 10 → Exit Head 2 → Confidence Head 2
Layer 14 → Exit Head 3 → Confidence Head 3
Layer 18 → Exit Head 4 → Confidence Head 4
Layer 22 → Exit Head 5 → Confidence Head 5
Layer 26 → Exit Head 6 → Confidence Head 6
    ↓
[UCB Bandit Controller]
    ↓
Token + Cache + Metrics
```

## Project Structure

```
mindllm/
├── src/
│   ├── config.py                          # Configuration classes
│   ├── integration.py                     # Complete inference engine
│   ├── phase1_exit_architecture/
│   │   ├── __init__.py
│   │   └── multi_exit_llm.py              # Multi-exit LLM with cache management
│   ├── phase2_confidence_classifier/
│   │   ├── __init__.py
│   │   └── confidence_heads.py            # Confidence classifiers & training
│   ├── phase3_bandit_controller/
│   │   ├── __init__.py
│   │   └── ucb_controller.py              # UCB bandit & threshold adaptation
│   └── utils/
│       ├── __init__.py
│       └── utilities.py                   # Helper utilities
├── examples_integration.py                # Full integration demo
├── examples_phase2_training.py            # Confidence classifier training
├── examples_phase3_bandit.py              # Bandit controller demo
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone repository
git clone https://github.com/Arshadshemilk/mindllm.git
cd mindllm

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- NumPy, SciPy, scikit-learn

## Usage

### Phase 1: Multi-Exit Model Setup

```python
from src.phase1_exit_architecture.multi_exit_llm import MultiExitLLM
from src.utils.utilities import load_model_with_unsloth

# Load Gemma 3 1B directly from Hugging Face
base_model, tokenizer = load_model_with_unsloth(
    "unsloth/gemma-3-1b-it"
)

# Wrap with multi-exit architecture
multi_exit_llm = MultiExitLLM(
    base_model=base_model,
    hidden_dim=2304,
    exit_layer_indices=[6, 10, 14, 18, 22, 26],
    vocab_size=tokenizer.vocab_size
)

# Generate with early exiting
output = multi_exit_llm.generate_with_early_exit(
    input_ids=input_ids,
    exit_threshold=0.85,
    max_new_tokens=128
)
```

### Phase 2: Train Confidence Classifiers

```python
from src.phase2_confidence_classifier.confidence_heads import (
    ConfidenceClassifierEnsemble,
    ConfidenceTrainer
)

# Create ensemble
ensemble = ConfidenceClassifierEnsemble(
    exit_layer_indices=[4, 6, 8, 10, 12, 14, 16, 18]
)

# Train with geometric loss
trainer = ConfidenceTrainer(model=ensemble, gamma=0.9)
trainer.train_epoch(hidden_states_dict, targets_dict, batch_size=32)

# Calibrate
trainer.apply_temperature_scaling(
    hidden_states_dict, targets_dict, target_ece=0.05
)
```

### Phase 3: Adaptive Threshold Control

```python
from src.phase3_bandit_controller.ucb_controller import UCBBanditController

# Initialize bandit
bandit = UCBBanditController(
    min_threshold=0.50,
    max_threshold=0.99,
    num_arms=20,
    confidence_weight=0.7,
    energy_weight=0.3
)

# Select threshold for next token
threshold, arm_idx = bandit.select_threshold(state)

# Record outcome after generation
bandit.update_arm(arm_idx, exit_layer, confidence)
```

### Complete Integration

```python
from src.integration import MindEdgeAIEngine

# Create integrated engine
engine = MindEdgeAIEngine(
    multi_exit_model=multi_exit_llm,
    confidence_classifier=ensemble,
    bandit_controller=bandit,
    max_layers=18
)

# Generate sequence with full adaptation
result = engine.generate_sequence(
    input_ids=input_ids,
    max_new_tokens=128,
    total_energy_budget=1.0
)

print(f"Generated {result['num_tokens_generated']} tokens")
print(f"Energy used: {result['total_energy_used']:.4f}")
print(f"Efficiency: {result['energy_efficiency']:.2f} tokens/energy")
```

## Configuration

All system parameters are centralized in [src/config.py](src/config.py):

```python
# Model configuration (Gemma 3 1B)
ModelConfig(
    model_name="unsloth/gemma-3-1b-it",
    num_layers=26,
    hidden_dim=2304,
    exit_layer_indices=[6, 10, 14, 18, 22, 26]
)

# Confidence classifier configuration
ConfidenceClassifierConfig(
    input_dim=2304,           # Match Gemma 3 1B hidden dimension
    hidden_dim1=256,
    hidden_dim2=64,
    gamma=0.9,              # Discount factor
    target_ece=0.05         # Calibration target
)

# Bandit configuration
BanditConfig(
    min_threshold=0.50,
    max_threshold=0.99,
    num_arms=20,
    exploration_constant_warmup=1.0,
    exploration_constant_high=3.0,
    exploration_constant_standard=2.0,
    confidence_weight=0.7,
    energy_weight=0.3,
    warmup_tokens=100,
    extended_explore_tokens=500
)
```

## Key Components

### Exit Heads
Located at layers 4, 6, 8, 10, 12, 14, 16, 18, each with:
- Dense layer (2048 → 2048)
- LayerNorm
- Output projection (2048 → vocab_size)

### Confidence Heads
MLP architecture for each exit layer:
```
Input (2048) → Dense (256, ReLU) → Dropout
            → Dense (64, ReLU) → Dropout
            → Output (1, Sigmoid) → Confidence [0,1]
```

### Loss Functions
**Geometric BCE Loss:**
```
L_geom = Σ_l γ^(L-l) * BCE(pred_l, target_l)
```
where γ=0.9 heavily weights earlier layers for confident early predictions.

### Reward Function
```
R = 0.7 * confidence - 0.3 * (exit_layer / L)
```
Balances prediction quality with energy efficiency.

## Examples

Run the provided examples to understand the system:

```bash
# Phase 2: Training confidence classifiers
python examples_phase2_training.py

# Phase 3: Bandit controller learning
python examples_phase3_bandit.py

# Complete integration demo
python examples_integration.py
```

## Performance Metrics

The system tracks:
- **Confidence scores**: Model's self-reported confidence at each exit
- **Exit layers**: Which layer tokens exit at (lower = more energy efficient)
- **Rewards**: Quality-energy trade-off scores
- **UCB statistics**: Per-arm performance across all thresholds
- **Energy efficiency**: Tokens generated per energy unit

## Mathematical Formulation

### Multi-Exit Forward Pass
When token exits at layer L:
- Compute hidden state at L
- Extract KV projections: K_{L+1:L} = HiddenLayer[L] × W_K
- Cache for future tokens
- Output logits: y = LogitHead[L](HiddenLayer[L])

### Confidence Calibration
Temperature-scaled sigmoid:
```
p_calibrated = sigmoid(logits / T)
```

### UCB Selection
```
UCB_i = μ_i + c * sqrt(ln(t) / n_i)
```

## Deployment Considerations

1. **Initial Setup**: Load Gemma-1B and freeze weights
2. **Training**: Generate synthetic training data from forward passes
3. **Calibration**: Tune temperature scaling on validation data
4. **Online Learning**: Bandit adapts during first inference runs
5. **Monitoring**: Track energy budget vs. quality trade-offs

## Contributing

Contributions welcome! Focus areas:
- Support for additional base models
- Extended bandit algorithms (Thompson sampling, LinUCB)
- Advanced calibration techniques
- Performance optimizations

## References

- Multi-exit networks: Teerapittayanon et al., "BranchyNet"
- Confidence calibration: Guo et al., "On Calibration of Modern Neural Networks"
- Bandit algorithms: Lattimore & Szepesvári, "Bandit Algorithms"
- Energy-aware inference: Zhang et al., "Token-level Energy-Efficient LLM Inference"

## License

MIT License - See LICENSE file for details

## Contact

For questions and feedback, open an issue on GitHub.