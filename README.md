# MultiEMiT: Multilevel Emotion, Irony, and Semiotic Intent Benchmark in Social TV

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Venue](https://img.shields.io/badge/Target-IEEE%20Transactions%20on%20Affective%20Computing-purple.svg)]()

Official repository for the **MultiEMiT** benchmark, introduced in the manuscript:  
> **"MultiEMiT: Multilevel Emotion, Irony, and Semiotic Intent Benchmark in Italian Social TV Commentary"**

---

## 📌 Overview

**MultiEMiT** is a multi-layered multimodal benchmark of Italian social media commentary (Twitter/X) gathered during live television broadcasts (*Propaganda Live* on LA7 and *Pechino Express* on Rai2). Rather than restricting annotations to broad positive/negative sentiment, MultiEMiT establishes a granular, 5-level pragmatic schema evaluated independently by three annotators:

1. **Emotions (Multi-label):** 10 affective categories grounded in Robert Plutchik's psychoevolutionary model (*Joy, Sadness, Trust, Disgust, Fear, Anger, Surprise, Anticipation, Love, Neutral*).
2. **Irony and Sarcasm (3-class):** Distinguishing literal statements (*Unironic*), humorous figurative inversion (*Irony*), and hostile mockery (*Sarcasm*).
3. **Offensiveness (Binary):** Identifying abusive, derogatory, or harassment language.
4. **Target (Multi-label):** Differentiating commentary directed at the broadcast *Content* (political issues, guests, news events) versus the *Container* (studio host, network, production).
5. **Semiotic Text--Image Relation (3-class):** Categorizing multimodal intent into *Additive* (mutual enrichment), *Parallel* (redundant meaning), and *Divergent* (pragmatic or cognitive contrast generating meme subversion and irony).

---

## 📂 Repository Structure

```
MultiEMiT/
├── datasets/
│   ├── splits/                         # Official benchmark train/test partitions
│   │   ├── textual_train.csv           # Textual training set (N = 1,618)
│   │   ├── textual_test.csv            # Textual test set (N = 405)
│   │   ├── textual_full.csv            # Full consolidated textual dataset (N = 2,023)
│   │   ├── multimodal_train.csv        # Multimodal training set (N = 516)
│   │   ├── multimodal_test.csv         # Multimodal test set (N = 129)
│   │   └── multimodal_full.csv         # Full multimodal dataset (N = 645)
│   ├── annotations/
│   │   ├── raw/                        # Anonymized 3-rater records (Annotator_1, 2, 3)
│   │   ├── fused_textual_annotations_3raters.csv
│   │   ├── fused_multimodal_annotations_3raters.csv
│   │   └── fleiss_kappa_summary.csv    # Official Fleiss' kappa inter-annotator report
│   └── corpora/
│       ├── images/                     # Paired multimodal broadcast images & memes (494 files)
│       ├── correlations/               # Pearson cross-layer correlation matrices & prompts
│       ├── emit/                       # External EVALITA 2023 EMiT auxiliary training set
│       └── ironita/                    # External IronITA 2018 auxiliary training set
│
├── code/
│   ├── bert/
│   │   └── multiseed/                  # Official 3-seed BERT & Dual-Encoder pipeline
│   │       ├── common_utils.py         # Shared dataloaders, metrics, and path resolvers
│   │       ├── train_textual_*.py      # Step 1 & Step 2 text encoder fine-tuning
│   │       ├── evaluate_multimodal_*.py# Frozen Dual-Encoder linear probe evaluation
│   │       └── jup_results/            # Multi-seed experimental logs (Seeds 42, 123, 456)
│   ├── pixtral/
│   │   ├── finetune/                   # Pixtral 12B LoRA fine-tuning scripts
│   │   └── inference/                  # Zero-Shot, Few-Shot, and LoRA inference scripts
│   │       ├── inference_pixtral.py    # Main VLM evaluation harness
│   │       └── prompts_short_medium_long.py # Short, Medium, Long & v2 negative constraint prompts
│   ├── compute_fleiss_kappa_multiemit.py# Standalone Inter-Annotator Agreement pipeline
│   └── requirements.txt                # Python environment dependencies
│
├── online_paper/                       # Modular LaTeX manuscript source & vector figures
└── docs/                               # Master's thesis archive & guidelines
```

---

## 🚀 Quickstart & Installation

### 1. Environment Setup
We recommend Python 3.10 or 3.11 with a virtual environment or Conda:

```bash
# Clone the repository
git clone https://github.com/andrealuparia/MultiEMit.git
cd MultiEMit

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r code/requirements.txt
```

For GPU acceleration (PyTorch with CUDA 12.1+):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 📊 Reproducing Experiments

### 1. Inter-Annotator Agreement (Fleiss' $\kappa$)
To re-run the full 3-annotator reliability analysis and compute Fleiss' $\kappa$ across all pragmatic layers:
```bash
python code/compute_fleiss_kappa_multiemit.py
```
Outputs are saved directly to `datasets/annotations/fleiss_kappa_summary.csv`.

### 2. Multi-Seed BERT & Dual-Encoder Linear Probes
All discriminative experiments in the paper are averaged across three seeds (`42`, `123`, `456`) on the fixed `seed=42` benchmark split:

```bash
# Train textual encoders (Emotion, Irony, Relation)
python code/bert/multiseed/train_textual_emotion.py --seed 42
python code/bert/multiseed/train_textual_irony.py --seed 42
python code/bert/multiseed/train_textual_relation.py --seed 42

# Evaluate multimodal dual-encoders (BERT + ViT-224 linear probes)
python code/bert/multiseed/evaluate_multimodal_emotion.py --seed 42
python code/bert/multiseed/evaluate_multimodal_irony.py --seed 42
python code/bert/multiseed/evaluate_multimodal_relation.py --seed 42
```
Detailed logs and summary CSVs are preserved under `code/bert/multiseed/jup_results/`.

### 3. Generative Vision-Language Models (Pixtral 12B)
To run Zero-Shot, Three-Shot, or LoRA-fine-tuned evaluation on Pixtral 12B:
```bash
python code/pixtral/inference/inference_pixtral.py \
    --experiment_name pixtral_zero_shot \
    --dataset_path datasets/splits/multimodal_test.csv \
    --prompt_type short
```

---

## 🔒 Privacy & Ethical Compliance

All raw user handles, URLs, and annotator identities have been scrubbed or anonymized (`Annotator_1`, `Annotator_2`, `Annotator_3`) in compliance with European GDPR standards and IEEE computational ethics guidelines. Broadcast stills and social commentary are provided strictly for academic research and affective computing benchmarks.

---

## 📖 Citation

If you find this dataset or codebase useful in your research, please cite our paper:

```bibtex
@article{luparia2026multiemit,
  title={MultiEMiT: Multilevel Emotion, Irony, and Semiotic Intent Benchmark in Italian Social TV Commentary},
  author={Luparia, Andrea and others},
  journal={IEEE Transactions on Affective Computing},
  year={2026},
  publisher={IEEE}
}
```

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

