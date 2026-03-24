# 🚀 START HERE - Exact Steps (Copy & Paste)

**Don't know where to start?** Follow these exact commands. Nothing more.

---

## Prerequisites Check

Your system needs:
- Python 3.8+
- GPU with 4GB+ RAM (or CPU, slower)
- ~3GB disk space

Verify:
```bash
python --version          # Should be 3.8 or higher
pip --version            # Should be installed
nvidia-smi              # Optional, checks GPU
```

---

## The Fastest Way (5 minutes)

### 1️⃣ Install Requirements
```bash
# Navigate to project
cd /workspaces/mindllm

# Install packages (already done if using this workspace)
pip install -r requirements.txt
```

### 2️⃣ Run the Quick Test
```bash
# This trains and does inference in one go
python quick_test.py
```

**That's it!** You'll see:
- Model loading ✓
- Training progress ✓
- Inference results ✓
- Summary metrics ✓

---

## The Complete Way (15 minutes)

If you want the detailed walkthrough:

```bash
# Runs all 7 steps with explanations
python get_started.py
```

This does the same as quick_test.py but with more detailed logging.

---

## The Manual Way (If you want control)

### Step 1: Load the model
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "unsloth/gemma-3-1b-it",
    device_map='auto',
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-1b-it")
```

### Step 2: Prepare your data
```python
train_texts = [
    "Machine learning is AI",
    "Deep learning uses neural networks",
    # ... add your texts here
]
val_texts = train_texts[:1]
test_texts = train_texts[1:2]
```

### Step 3: Train the complete system
```python
from src.real_training_pipeline import RealTrainingPipeline

pipeline = RealTrainingPipeline(checkpoint_dir="./checkpoints")

conf_model, bandit = pipeline.run_complete_training(
    model=model,
    tokenizer=tokenizer,
    train_texts=train_texts,
    val_texts=val_texts,
    test_texts=test_texts
)
```

### Step 4: Do inference
```python
from src.production_inference import setup_production_pipeline

pipeline, config = setup_production_pipeline(
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
```

### Step 5: Check results
```python
summary = pipeline.get_inference_summary()
print(f"Confidence: {summary['avg_confidence']:.3f}")
print(f"Convergence: {summary['bandit_metrics']['arm_concentration']:.1%}")
```

---

## Common Issues & Fixes

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Fix:** `pip install torch`

### Issue: "ModuleNotFoundError: No module named 'transformers'"
**Fix:** `pip install transformers`

### Issue: "CUDA out of memory"
**Fix:** Reduce batch_size to 4 or 8

### Issue: "Model fails to load"
**Fix:** Check internet, delete cache `rm -rf ~/.cache/huggingface/`

---

## After Getting Results

### See Training Code
Look at the training phases:
- Phase 1: `src/phase1_production.py`
- Phase 2: `src/phase2_production.py`
- Phase 3: `src/phase3_production.py`

### See More Examples
```python
# 10 different scenarios with code
open("QUICKSTART_EXAMPLES.py")
```

### Understand Everything
```python
# Complete API documentation
open("PRODUCTION_README.md")
```

---

## What Each Phase Does

```
Phase 1: Multi-Exit Architecture
├─ Loads Gemma 3 1B model
├─ Extracts hidden states at layers [6,10,14,18,22,26]
└─ Prepares data for Phase 2

Phase 2: Confidence Classifiers  
├─ Trains 6 MLP heads to predict "will this exit be correct?"
├─ Uses geometric loss for weighting
└─ Calibrates predictions (target ECE < 0.05)

Phase 3: Bandit Controller
├─ Learns optimal confidence thresholds
├─ Uses UCB algorithm with 20 arms
└─ Adapts during inference (online learning)
```

---

## What You Get

After running:
- ✅ Trained confidence classifiers
- ✅ Learned bandit thresholds
- ✅ Working inference pipeline
- ✅ Performance metrics
- ✅ Saved checkpoints

---

## Ready to Deploy?

Check the deployment checklist:
```python
open("PRODUCTION_README.md")  # See "Deployment Checklist"
```

---

## TL;DR - Just Run This Now

```bash
cd /workspaces/mindllm
python quick_test.py
```

Then read `PRODUCTION_README.md` for everything else.

**You're done.** 🎉
