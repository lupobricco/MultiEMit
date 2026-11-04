# MultiEMit Modular Python Benchmark (`final_codes_py`)

Pure Python `.py` implementations of the MultiEMit benchmarks, faithfully ported from [`code/bert/final_codes/`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/final_codes) with full modular CLI parameters and the **Emotion concatenation paradigm mirrored in Irony**.

---

## 1. Irony Task Architectural Improvement

In the original `code/bert/final_codes/train_textual_irony.ipynb`, training only used the textual dataset (`me_train`, 1,618 posts) without the multimodal training text (`df_mm_train`, 516 posts), unlike Emotion detection which concatenated both in Step 2.

In this folder, **`train_textual_irony.py` mirrors the exact, high-performing 2-stage Emotion pipeline**:

```
Step 1 (Without Correlations):
  IronITA (3,879 rows) + MultiEMit Textual Train (1,618 rows) = 5,497 rows
  Evaluated on MultiEMit Textual Test (me_test = 405 rows)
  -> Checkpoint saved to ./checkpoints/irony_textual_no_corr

Step 2 (With Correlations):
  Initialized from Step 1 checkpoint (./checkpoints/irony_textual_no_corr)
  Trained on MultiEMit Textual Train (1,618 rows) + MultiEMit Multimodal Train (516 rows) = 2,134 rows (with correlations)
  Evaluated on MultiEMit Textual Test (me_test = 405 rows, with correlations)
  -> Checkpoint saved to ./checkpoints/irony_textual_corr
```

---

## 2. Directory Structure

```
code/bert/final_codes_py/
├── run_experiments.py                   # Master CLI runner with full parameter controls
├── run_experiments.bat                  # One-click Windows batch runner
│
├── train_textual_emotion.py             # Task 1 Textual: FEEL-IT (EMit A + ME Textual -> ME Multimodal)
├── train_textual_irony.py               # Task 2 Textual: UmBERTo (IronITA + ME Textual -> ME Multimodal)
├── train_textual_relation.py            # Task 3 Textual: UmBERTo (MM Train -> MM Test)
│
├── multimodal_dual_encoder_emotion.py   # Task 1 Multimodal: FEEL-IT + ViT-224 Dual-Encoder
├── multimodal_dual_encoder_irony.py     # Task 2 Multimodal: UmBERTo + ViT-224 Dual-Encoder
├── multimodal_dual_encoder_relation.py  # Task 3 Multimodal: UmBERTo + ViT-224 Dual-Encoder
│
├── evaluate_relation_subsets.py         # Subsets evaluation helper
├── checkpoints/                         # Saved model checkpoints
└── results/                             # Generated CSVs, reports, and unified tables
```

---

## 3. Quickstart & Usage

### A. Using the Master Runner (`run_experiments.py`)

Run the master script to train and evaluate experiments in isolated subprocesses (guaranteeing 100% CUDA VRAM cleanup between models):

```bash
# 1. Preview execution plan without running (Dry Run):
python run_experiments.py --dry-run

# 2. Check available tasks and current checkpoint status:
python run_experiments.py --list

# 3. Run EVERYTHING (Emotion, Irony, Relation — Textual + Multimodal):
python run_experiments.py

# 4. Train ONLY Textual models:
python run_experiments.py --stages textual

# 5. Run ONLY Irony (Textual + Multimodal):
python run_experiments.py --tasks irony

# 6. Train Irony and Emotion with custom hyperparameters & force retraining:
python run_experiments.py --tasks emotion,irony --force-retrain --epochs 10 --batch-size 16 --lr 2e-5
```

### B. Running Individual Scripts Standalone

Every `.py` script can also be executed directly on its own:

```bash
# Textual Training:
python train_textual_emotion.py --epochs 10 --batch-size 16 --lr 2e-5 --force-retrain
python train_textual_irony.py --epochs 10 --batch-size 16 --lr 2e-5 --force-retrain
python train_textual_relation.py --epochs 10 --batch-size 16 --lr 2e-5 --force-retrain

# Multimodal Dual-Encoder Feature Extraction & Evaluation:
python multimodal_dual_encoder_emotion.py --batch-size 32
python multimodal_dual_encoder_irony.py --batch-size 32
python multimodal_dual_encoder_relation.py --batch-size 32
```

---

## 4. Output Results

All results are automatically aggregated in `./results/`:
- `all_experiments_summary.csv`: Combined metric table (Accuracy, Macro F1, Micro F1, Weighted F1).
- `all_experiments_summary.xlsx`: Excel spreadsheet ready for reporting.
- `all_experiments_full_report.md`: Markdown summary report with per-class classification matrices.

