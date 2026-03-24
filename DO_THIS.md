# 🎯 GET STARTED - THE ABSOLUTE SIMPLEST WAY

**Don't overthink. Just pick one and run it.**

---

## The 3 Scripts You Can Run Right Now

### 1️⃣ FASTEST (Most minimal code)
```bash
python quick_test.py
```
- Tests the system end-to-end
- 5 minute runtime
- Perfect for "does it work?"
- **Start here if**: You just want to verify everything works

---

### 2️⃣ WITH GUIDANCE (Detailed explanations)
```bash
python get_started.py
```
- Same as quick_test.py but with explanations
- Shows what each Step does
- 5 minute runtime + reading output
- **Start here if**: You want to understand what's happening

---

### 3️⃣ SIMPLEST CODE (Bare minimum copy-paste)
```bash
python simplest_example.py
```
- Minimal, clean code
- No extra stuff
- Perfect to copy and modify
- **Start here if**: You want to see the absolute minimum code

---

## Which One Should You Pick?

| You want... | Pick this |
|-------------|-----------|
| Just test if it works | `python quick_test.py` |
| Understand each step | `python get_started.py` |
| See clean code | `python simplest_example.py` |
| Copy and modify | Edit `simplest_example.py` |

---

## The ACTUAL Steps to Get It Working

### Option A: Run the fastest one
```bash
cd /workspaces/mindllm
python quick_test.py
```
Done in 5 minutes. You'll have:
- ✓ Trained confidence classifiers
- ✓ Learned bandit thresholds
- ✓ Working inference
- ✓ Performance metrics

### Option B: Run with explanations
```bash
cd /workspaces/mindllm
python get_started.py
```
Same results, but shows why each step happens.

### Option C: Understand the code first
```bash
# Read the simple example
cat simplest_example.py

# Modify for your data
# Replace train_texts = [...]

# Run it
python simplest_example.py
```

---

## After Running (What Comes Next)

```
DONE RUNNING? You now have:
├─ ./checkpoints/confidence_classifier_final.pt
│  └─ Your trained confidence model
│
├─ ./checkpoints/bandit_checkpoint.json
│  └─ Learned thresholds
│
└─ ./checkpoints/training_summary.json
   └─ Metrics from training
```

### Now what?

**Read ONE of these** (pick based on your question):

| Your Question | Read This |
|---------------|-----------|
| "How do I use it?" | PRODUCTION_README.md |
| "Show me examples" | QUICKSTART_EXAMPLES.py |
| "What are the phases?" | IMPLEMENTATION_GUIDE.md |
| "What happened?" | COMPLETE_FLOW.md |
| "How do I deploy?" | PRODUCTION_README.md (Deployment section) |

---

## Real Talk

You're overthinking this. It's simple:

1. **Pick ONE script** above (I recommend `quick_test.py`)
2. **Run it**: `python script.py`
3. **Wait 5 minutes**
4. **See results**

That's it. Don't read docs first. Just run it.

---

## Troubleshooting (Only if something breaks)

### "ModuleNotFoundError: No module named..."
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"
Edit the script, change:
```python
'batch_size': 8,  # Change to 4 or 2
```

### "Model download failed"
Check internet, try again. First time downloads ~2.6GB.

### Something else?
Read: PRODUCTION_README.md (Troubleshooting section)

---

## The Complete Path (If you're thorough)

```
1. Run quick_test.py
   ↓
2. See it works ✓
   ↓
3. Read WHICH_SCRIPT.md (1 min)
   ↓
4. Read RUN_THIS_FIRST.md (3 min)
   ↓
5. Read PRODUCTION_README.md (20 min)
   ↓
6. Try QUICKSTART_EXAMPLES.py (5 min)
   ↓
7. Modify for YOUR data
   ↓
8. Deploy using checklist
```

But honestly? Just do steps 1-2 first. 😄

---

## Quick Commands Reference

```bash
# Fastest test
python quick_test.py

# With guidance
python get_started.py

# See code
cat simplest_example.py

# View docs
cat PRODUCTION_README.md      # Complete reference
cat QUICKSTART_EXAMPLES.py    # Code examples
cat IMPLEMENTATION_GUIDE.md   # Concepts

# Check results
ls -la ./checkpoints/         # What was saved
cat ./checkpoints/training_summary.json  # Metrics
```

---

## TL;DR

```bash
python quick_test.py
```

Then read docs. You're done. 🚀

---

## Still Lost?

Read in this order:
1. START_HERE.md (you're here)
2. WHICH_SCRIPT.md (navigate to right script)
3. RUN_THIS_FIRST.md (exact commands)
4. PRODUCTION_README.md (full reference)

Or just skip to:
```bash
python quick_test.py
```

Both work. 😊
