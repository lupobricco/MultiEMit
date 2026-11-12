# MultiEMiT: Multi-Seed Sensitivity & Robustness Suite

> **Target Venue**: *IEEE Transactions on Affective Computing*  
> **Sensitivity Protocol**: 3 Training Seeds (`42, 123, 456`) with automatic **Mean ± Std** aggregation.  
> **Split Guarantee**: All evaluations are strictly locked to `SPLIT_SEED = 42` (`me_test` = 405 rows, `mm_test` = 129 rows). Zero data leakage.  
> **100% External Dataset Ingestion**: Step 1 trains on 100% of EMit-A (5,966 rows) and 100% of IronITA (4,849 rows), with zero discarded validation rows.

This directory is an isolated, cloud-ready suite designed to run all MultiEMiT tasks across 3 random seeds and compile publication-ready tables with variance estimates (`Mean ± Std`).

---

## 1. Quick Start on Google Jupyter Notebook

### Full Retraining Across 3 Seeds (Force Retrain All Checkpoints)

```bash
!python run_paper_experiments.py --seeds 42 123 456 --force-retrain --epochs 10 --batch-size 16 --eval-batch-size 32 --lr 2e-5 --weight-decay 0.01
```

### Linear Probe-Only Sensitivity (Evaluates Multimodal Probes Across 3 Seeds)

```bash
!python run_paper_experiments.py --eval-only --seeds 42 123 456
```

### Single Seed Run

```bash
!python run_paper_experiments.py --seed 42 --force-retrain
```

---

## 2. Directory Layout & Seed Isolation

```text
code/bert/clean_paper_codes_multiseed/
│
├── common_utils.py                    # Dual root detection, ClassifierChain(random_state=seed), multi-seed aggregator
│
├── train_textual_emotion.py           # 100% EMit-A (5,966) + me_train (1,618) = 7,584 rows. Shuffled with seed.
├── evaluate_multimodal_emotion.py     # Multimodal Case 1-6. Canonical order (no .sample()). Head fit with seed.
│
├── train_textual_irony.py             # 100% IronITA (4,849) + me_train (1,618) = 6,467 rows. Shuffled with seed.
├── evaluate_multimodal_irony.py       # Multimodal Case 1-6. Canonical order (no .sample()). Head fit with seed.
│
├── train_textual_relation.py          # mm_train (516) shuffled with seed. mm_test (129, split 42).
├── evaluate_multimodal_relation.py    # Multimodal Stage A & B. Canonical order (no .sample()). Head fit with seed.
│
├── run_paper_experiments.py          # Master Multi-Seed Runner & Aggregator
├── run_experiments.bat               # Windows batch launcher
├── requirements.txt                  # Pinned dependencies
│
├── checkpoints/
│   ├── seed_42/                       # Checkpoints for seed 42
│   ├── seed_123/                      # Checkpoints for seed 123
│   └── seed_456/                      # Checkpoints for seed 456
│
└── results/
    ├── seed_42/                       # Full reports, CSVs, CMs for seed 42
    ├── seed_123/                      # Full reports, CSVs, CMs for seed 123
    ├── seed_456/                      # Full reports, CSVs, CMs for seed 456
    └── MULTISEED_EXPERIMENTS_REPORT.md# Consolidated publication tables (Mean ± Std)
```

---

## 3. Command-Line Options Reference

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--seeds` | `int [...]` | `42 123 456` | List of random training seeds to execute in sequence. |
| `--force-retrain` | flag | `False` | Forces retraining all Transformer checkpoints from scratch. |
| `--eval-only` | flag | `False` | Skips Transformer fine-tuning and tests linear probe sensitivity on existing checkpoints. |
| `--report-only` | flag | `False` | Skips training and eval, recompiling reports from existing CSVs. |
| `--task` | choice | `all` | One of `all`, `emotion`, `irony`, `relation`. |
| `--epochs` | `int` | `10` | Fine-tuning epochs per stage. |
| `--batch-size` | `int` | `16` | Minibatch size for training. |
| `--eval-batch-size` | `int` | `32` | Minibatch size for inference / feature extraction. |
| `--lr` | `float` | `2e-5` | AdamW learning rate. |
| `--weight-decay` | `float` | `0.01` | AdamW weight decay. |
| `--checkpoints-dir` | path | `./checkpoints` | Root directory for checkpoints. |
| `--results-dir` | path | `./results` | Root directory for results. |
