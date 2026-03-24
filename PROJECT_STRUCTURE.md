# MindEdgeAI: Project Structure Overview

```
mindllm/
│
├── README.md                               # Comprehensive project documentation
├── GETTING_STARTED.md                      # Step-by-step deployment guide
├── requirements.txt                        # Python dependencies
│
├── src/
│   ├── __init__.py
│   ├── config.py                           # Centralized configuration classes
│   ├── integration.py                      # Main inference engine (MindEdgeAIEngine)
│   │
│   ├── phase1_exit_architecture/           # PHASE 1: Multi-Exit Architecture
│   │   ├── __init__.py
│   │   └── multi_exit_llm.py              # MultiExitLLM, ExitHead, ExitOutput
│   │                                       # - 8 exit points (layers 4,6,8,10,12,14,16,18)
│   │                                       # - KV cache propagation
│   │                                       # - Early exit support
│   │
│   ├── phase2_confidence_classifier/       # PHASE 2: Confidence Classification
│   │   ├── __init__.py
│   │   └── confidence_heads.py             # ConfidenceHead, ConfidenceClassifierEnsemble
│   │                                       # - GeometricBCELoss (γ=0.9)
│   │                                       # - ConfidenceTrainer
│   │                                       # - Temperature scaling calibration
│   │
│   ├── phase3_bandit_controller/           # PHASE 3: Bandit Controller
│   │   ├── __init__.py
│   │   └── ucb_controller.py               # UCBBanditController, BanditArm
│   │                                       # - 20 confidence threshold arms
│   │                                       # - 3-phase exploration strategy
│   │                                       # - Adaptive threshold selection
│   │
│   └── utils/                              # Utility modules
│       ├── __init__.py
│       └── utilities.py                    # KVCacheManager, EnergyEstimator
│                                           # MetricsTracker, HyperparameterScheduler
│
├── examples_integration.py                 # Full system demonstration
├── examples_phase2_training.py             # Confidence classifier training
├── examples_phase3_bandit.py               # Bandit controller learning demo
│
└── tests/
    └── test_components.py                  # Comprehensive unit tests
                                            # - Phase 1: ExitHead tests
                                            # - Phase 2: ConfidenceHead tests
                                            # - Phase 3: Bandit tests
                                            # - Config tests
                                            # - Utility tests
```

## Key Files and Their Purposes

### Core Implementation

| File | Purpose | Key Classes |
|------|---------|------------|
| `config.py` | Central configuration | ModelConfig, ConfidenceClassifierConfig, BanditConfig |
| `integration.py` | Main inference engine | MindEdgeAIEngine |
| `phase1_exit_architecture/multi_exit_llm.py` | Multi-exit model | MultiExitLLM, ExitHead, ExitOutput |
| `phase2_confidence_classifier/confidence_heads.py` | Confidence estimation | ConfidenceHead, ConfidenceClassifierEnsemble, ConfidenceTrainer |
| `phase3_bandit_controller/ucb_controller.py` | Bandit algorithm | UCBBanditController, BanditArm, BanditState, AdaptiveThresholdManager |
| `utils/utilities.py` | Helper utilities | KVCacheManager, EnergyEstimator, MetricsTracker |

### Examples & Tests

| File | Type | Purpose |
|------|------|---------|
| `examples_integration.py` | Example | Full system integration demo |
| `examples_phase2_training.py` | Example | Confidence classifier training |
| `examples_phase3_bandit.py` | Example | Bandit controller visualization |
| `tests/test_components.py` | Test Suite | Unit tests for all components |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | Complete system overview |
| `GETTING_STARTED.md` | Deployment & usage guide |

## Architecture Diagram

```
Input Tokens (Gemma-1B Tokenizer)
         │
         ▼
┌─────────────────────────┐
│  Phase 1: Base Model    │
│  (Gemma-1B, 18 layers)  │
└─────────────────────────┘
         │
    ┌────┼────┐
    │ Exit Points (8 layers)
    │
    ├─ Layer 4  → Exit Head 1 → Confidence Head 1 ─┐
    ├─ Layer 6  → Exit Head 2 → Confidence Head 2 ─┼─ Phase 2
    ├─ Layer 8  → Exit Head 3 → Confidence Head 3 ─┤ (Confidence Estimation)
    ├─ Layer 10 → Exit Head 4 → Confidence Head 4 ─┼
    ├─ Layer 12 → Exit Head 5 → Confidence Head 5 ─┤
    ├─ Layer 14 → Exit Head 6 → Confidence Head 6 ─┼
    ├─ Layer 16 → Exit Head 7 → Confidence Head 7 ─┤
    └─ Layer 18 → Exit Head 8 → Confidence Head 8 ─┘
         │
         ▼
    ┌─────────────────────────────────┐
    │ Phase 3: Bandit Controller      │
    │ (Adaptive Threshold Selection)   │
    │                                 │
    │ State: (position, energy, conf) │
    │ Arms: 20 thresholds [0.50-0.99] │
    │ UCB Selection + Update           │
    └─────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────┐
    │ KV Cache Manager                │
    │ (Propagate hidden states)       │
    └─────────────────────────────────┘
         │
         ▼
    Next Token + Metrics
```

## Data Flow

```
GENERATION PHASE:
─────────────────

For each token:
  1. System State (position, energy budget, confidence history)
         ↓
  2. Bandit selects threshold (UCB algorithm)
         ↓
  3. Forward pass through model with selected threshold
         ↓
  4. Model exits at appropriate layer (confidence-based)
         ↓
  5. Compute hidden state → K,V projections → cache
         ↓
  6. Compute reward = 0.7·confidence - 0.3·(layer/18)
         ↓
  7. Update bandit arm statistics
         ↓
  8. Repeat until energy budget exhausted or max tokens reached
```

## Component Interactions

```
MindEdgeAIEngine (Integration Layer)
    │
    ├─ Uses → MultiExitLLM (Phase 1)
    │         - Forward pass
    │         - Early exit decision
    │         - KV cache generation
    │
    ├─ Uses → ConfidenceClassifierEnsemble (Phase 2)
    │         - Confidence scoring
    │         - (Currently bypassed in demo)
    │
    ├─ Uses → UCBBanditController (Phase 3)
    │         - Threshold selection
    │         - Arm updates
    │         - Statistics
    │
    ├─ Uses → KVCacheManager
    │         - Cache tracking
    │
    ├─ Uses → EnergyEstimator
    │         - Energy tracking
    │
    └─ Uses → MetricsTracker
              - Metrics collection
```

## Configuration Hierarchy

```
config.py (Central)
    │
    ├─ ModelConfig
    │   └── Used by: MultiExitLLM initialization
    │
    ├─ ConfidenceClassifierConfig  
    │   └── Used by: ConfidenceHead, ConfidenceTrainer
    │
    └─ BanditConfig
        └── Used by: UCBBanditController initialization
```

## Testing Coverage

```
tests/test_components.py
    │
    ├─ TestExitHead
    │   ├─ test_exit_head_forward
    │   └─ test_exit_head_backward
    │
    ├─ TestConfidenceHead
    │   ├─ test_confidence_head_forward
    │   └─ test_confidence_classifier_ensemble
    │   └─ test_geometric_bce_loss
    │
    ├─ TestBanditController
    │   ├─ test_bandit_arm
    │   ├─ test_bandit_ucb_computation
    │   ├─ test_ucb_bandit_initialization
    │   ├─ test_bandit_threshold_selection
    │   ├─ test_bandit_reward_computation
    │   └─ test_bandit_state_computation
    │
    ├─ TestUtilities
    │   ├─ test_energy_estimator
    │   └─ test_metrics_tracker
    │
    └─ TestConfiguration
        ├─ test_model_config
        ├─ test_classifier_config
        └─ test_bandit_config
```

## Execution Flow Example

```
User Code
    │
    ├─ Load: config.py (ModelConfig, BanditConfig, etc.)
    ├─ Load: phase1_exit_architecture/multi_exit_llm.py
    ├─ Load: phase2_confidence_classifier/confidence_heads.py
    ├─ Load: phase3_bandit_controller/ucb_controller.py
    ├─ Load: utils/utilities.py
    │
    └─ Create: MindEdgeAIEngine (integration.py)
         │
         └─ Call: engine.generate_sequence()
              │
              ├─ For each token:
              │   ├─ bandit_controller.select_threshold()
              │   ├─ multi_exit_llm.forward()
              │   ├─ bandit_controller.update_arm()
              │   └─ metrics_tracker.record()
              │
              └─ Return: generation results + statistics
```
