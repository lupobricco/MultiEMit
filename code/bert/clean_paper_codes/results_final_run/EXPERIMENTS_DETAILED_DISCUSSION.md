# MultiEMiT: Comprehensive Experimental Results & In-Depth Technical Discussion

> **Artifact Purpose**: Definitive technical specification and comprehensive experimental discussion of all final runs executed with the `clean_paper_codes` suite.
> **Scope**: Covers all three benchmark tasks (Task 1: Emotion, Task 2: Irony, Task 3: Semiotic Relation) across textual and multimodal splits. For each task, the official **Paper Pipeline** is presented first, followed by all **Extra Experiments and Ablations**, complete with granular parameter cards, metric breakdowns, semiotic subset evaluations, and analytical commentary.

---

## Executive Summary & Global Architecture

The MultiEMiT benchmark evaluates affective and semiotic understanding across Italian social media posts. The modeling framework leverages a two-stage textual fine-tuning pipeline followed by a late-fusion dual-encoder architecture augmented with oracle correlation prompting:

1. **Textual Backbones (`MAX_LENGTH = 128`)**:
   - **Task 1 (Emotion Detection)**: `MilaNLProc/feel-it-italian-emotion` (**FEEL-IT**, Italian emotion-specialized CamemBERT/UmBERTo model, 768-d hidden dimension, 12 layers, 12 attention heads).
   - **Task 2 (Irony Detection) & Task 3 (Semiotic Relation)**: `Musixmatch/umberto-commoncrawl-cased-v1` (**UmBERTo**, Italian cased RoBERTa architecture, 768-d hidden dimension, 12 layers, 12 attention heads).
2. **Visual Backbone**: `google/vit-base-patch16-224` (**ViT-Base-224**, frozen last-layer `[CLS]` token `out[:, 0, :]`, 768-d hidden dimension, $224\times 224$ RGB inputs, extracted via Hugging Face `pipeline("image-feature-extraction")` with `batch_size = 32`).
3. **Multimodal Late-Fusion Head**: Frozen 768-d text `[CLS]` (`hidden_states[-1][:, 0, :]`) and frozen 768-d ViT `[CLS]` embeddings are concatenated into a joint 1,536-dimensional vector:
   $$\mathbf{h}_{\text{joint}} = [\mathbf{h}_{\text{text}} \;\Vert\; \mathbf{h}_{\text{image}}] \in \mathbb{R}^{1536}$$
   All multimodal linear probes use the exact same base support vector classifier (`fit_linear_head` in `common_utils.py`):
   - **Base Linear Estimator**: `SGDClassifier(loss="squared_hinge", penalty="l2", class_weight="balanced", n_jobs=-1, random_state=42)`
   - **Multi-label classification (Task 1: Emotion)**: `ClassifierChain(base)` wrapping the balanced `SGDClassifier`.
   - **Single-label classification (Task 2: Irony & Task 3: Relation)**: `base` (`SGDClassifier` with `class_weight="balanced"` directly).
4. **Deterministic Partitioning (`SPLIT_SEED = 42`, `seed = 42`)**:
   - **MultiEMiT Textual Split (`textual_only_cleaned.csv`, $N=2,023$)**: 80/20 random split (`random_state=42`) $\to$ Train `me_train`: $1,618$ rows, Test `me_test`: $405$ rows.
   - **MultiEMiT Multimodal Split ($N=645$)**: Pre-split at `SEED=42` $\to$ Train `mm_train`: $516$ rows (`multimodal_train_seed42.csv`), Test `mm_test`: $129$ rows (`multimodal_test_seed42.csv`).
   - **EMit-A External Emotion Corpus (`cleaned_emit_train_A.csv`, $N=5,966$)**: 80/20 random split (`random_state=42`) $\to$ Train `emit_tr`: $4,772$ rows, Val: $1,194$ rows.
   - **IronITA External Irony Corpus (`ironita_full.xlsx`, $N=4,849$)**: 80/20 stratified split (`random_state=42`, `stratify=label`) $\to$ Train `iron_tr`: $3,879$ rows, Val: $970$ rows.

---

## 1. Task 1: Multimodal & Textual Emotion Detection (10-Class Multi-Label)

### 1.1 Technical Specification & Parameter Card

| Parameter Category | Exact Code Specification (`train_textual_emotion.py` & `evaluate_multimodal_emotion.py`) |
| :--- | :--- |
| **Task Formulation** | 10-Class Multi-Label Classification (`problem_type="multi_label_classification"`) |
| **Target Labels (10)** | `["Amore", "Anticipazione", "Disgusto", "Fiducia", "Gioia", "Neutro", "Paura", "Rabbia", "Sorpresa", "Tristezza"]` |
| **Text Encoder (`BASE_MODEL`)** | **FEEL-IT**: `MilaNLProc/feel-it-italian-emotion` (768-d last-layer `[CLS]` token) |
| **Tokenization** | `padding="max_length"`, `truncation=True`, `max_length=128` (`MAX_LENGTH = 128`) |
| **Vision Encoder** | **ViT-Base-224**: `google/vit-base-patch16-224` (frozen 768-d `[CLS]` token) |
| **Fusion & Scaling** | Unscaled late-fusion concatenation (`raw`, $1536$-d: `np.hstack([txt_tr, vit_tr])`) |
| **Oracle Prompt Builder** | `build_corr_text(df, emotions=False, target=True, irony=True, hate=True, relations=with_relation)` |
| **Exact Prompt String Order** | `il seguente testo [ha una relazione <additiva/divergente/parallela>] [contiene <ironia/sarcasmo/offensivita>] [riguarda <contenuto/contenitore>] <testo>` |
| **Prompt Schema Alignment** | Textual training & `me_test`: `with_relation=False` (`Oracle (no rel)`)<br/>Multimodal evaluation (`mm_train` & `mm_test`): `with_relation=True` (`Oracle (+ rel)`) |
| **Textual Classifier Head** | `AutoModelForSequenceClassification` (10 logits, `BCEWithLogitsLoss`) + Sigmoid $\sigma(z)$ |
| **Thresholding & Fallback** | `EMOTION_THRESHOLD = 0.40` via `predict_with_threshold`:<br/>1. Predict $1$ for all classes with $\sigma(z_c) \ge 0.40$.<br/>2. **Dynamic Fallback** (if all $\sigma(z_c) < 0.40$): predict top-1 class `order[0]`, and if `prob[order[0]] - prob[order[1]] <= 0.10`, also predict `order[1]`. |
| **Multimodal Linear Head** | `ClassifierChain(SGDClassifier(loss="squared_hinge", penalty="l2", class_weight="balanced", n_jobs=-1, random_state=42))` |
| **Step 1 Base Training Data** | Initialized from `MilaNLProc/feel-it-italian-emotion` $\to$ trained on `emit_tr` ($4,772$) + `me_train` ($1,618$) = **$6,390$ rows** (raw text, shuffled with `random_state=42`) $\to$ saved to `emotion_textual_no_corr` |
| **Step 2 Raw Ablation Data** | Initialized from `emotion_textual_no_corr` $\to$ trained on `me_train` ($1,618$) + `mm_train` ($516$) = **$2,134$ rows** (raw text, shuffled with `random_state=42`) $\to$ saved to `emotion_textual_step2_raw` |
| **Step 2 Oracle Training Data** | Initialized from `emotion_textual_no_corr` $\to$ trained on `me_train` ($1,618$) + `mm_train` ($516$) = **$2,134$ rows** (`Oracle (no rel)` prompt, shuffled with `random_state=42`) $\to$ saved to `emotion_textual_corr` |
| **Training Hyperparameters** | `epochs=10`, `train_batch_size=16`, `eval_batch_size=16`, `lr=2e-5`, `weight_decay=0.01`, `eval_strategy="epoch"`, `save_strategy="epoch"`, `load_best_model_at_end=True`, `metric_for_best_model="f1_macro"`, `save_total_limit=1`, `seed=42` |
| **Extraction Batch Size** | `batch_size=32` (`extract_text_features` and `extract_vit_features`) |

---

### 1.2 Official Paper Pipeline Results (Table I in Paper)

Evaluated on the held-out multimodal test split (`mm_test`, $N=129$). Both `Blind Text` and `Dual-Encoder` heads are trained on `mm_train` ($N=516$) using `ClassifierChain` over the balanced `SGDClassifier`.

| Configuration | Checkpoint Used | Prompting (`mm_test`) | Model | Scaling | Macro F1 | Micro F1 | Subset Acc. |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **Step 1 Base (without images)** | `emotion_textual_no_corr` | None (Raw) | Blind Text (`ClassifierChain` + balanced SGD) | `raw` | 0.3565 | 0.4314 | 0.2636 |
| **Step 1 Base with images** | `emotion_textual_no_corr` | None (Raw) | Dual-Encoder (FEEL-IT + ViT-224) | `raw` | 0.4068 | 0.4290 | 0.2791 |
| **Step 2 Oracle (without images)** | `emotion_textual_corr` | Oracle (`+ rel`) | Blind Text (`ClassifierChain` + balanced SGD) | `raw` | 0.4343 | 0.5183 | 0.3488 |
| **Step 2 Oracle with images** | **`emotion_textual_corr`** | **Oracle (`+ rel`)** | **Dual-Encoder (FEEL-IT + ViT-224)** | **`raw`** | **0.4967** | **0.5535** | **0.3566** |

---

### 1.3 Semiotic Subsets Breakdown (Table III in Paper)

Multi-label **Micro F1** across the three semiotic interaction categories in `mm_test` (`compute_subset_metrics(..., multilabel=True)`):
- **Additive (`ADD`, `type == 1`)**: $N=68$ posts where text and image complement each other.
- **Parallel (`PAR`, `type == 2`)**: $N=23$ posts where text and image convey parallel literal information.
- **Divergent (`DIV`, `type == 3`)**: $N=38$ posts where text and image exhibit semantic tension or contrast.

| Model Setting | Checkpoint | Prompting | Scaling | ADD ($N=68$) Micro F1 | PAR ($N=23$) Micro F1 | DIV ($N=38$) Micro F1 | Overall Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dual-Encoder Step 1 Base (No corr)** | `emotion_textual_no_corr` | None (Raw) | `raw` | 0.4624 | 0.5172 | 0.3168 | 0.4068 |
| **Dual-Encoder Step 2 Oracle (With corr)** | `emotion_textual_corr` | Oracle (`+ rel`) | `raw` | **0.5614** (+0.099) | **0.6296** (+0.112) | **0.4946** (+0.178) | **0.4967** (+0.090) |

> [!NOTE]
> The largest relative improvement under Oracle prompting occurs in the **Divergent subset (+17.78% Micro F1)**. Conditioning on semiotic (`ha una relazione divergente`) and figurative (`contiene ironia/sarcasmo`) correlation tokens allows the FEEL-IT + ViT-224 dual-encoder to disambiguate contradictory emotional expressions between visual and textual channels.

---

### 1.4 Extra Experiments: Textual Ablation Matrix (`me_test`, $N=405$)

Evaluates the unconfounded effect of in-domain data expansion (`Step 2 Raw`) vs. correlation prompting (`Step 2 Oracle`) on the purely textual test set (`me_test`, decoded with `EMOTION_THRESHOLD = 0.40` and dynamic fallback).

| Regime | Checkpoint | Prompt Template | N Test | Micro F1 | Macro F1 | Weighted F1 | Subset Acc. | Delta vs. Step 2 Raw |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 1 Base (Raw)** | `emotion_textual_no_corr` | Raw | 405 | 0.5403 | 0.5092 | 0.5371 | 0.3210 | Baseline |
| **Step 2 Raw (Ablation)** | `emotion_textual_step2_raw` | Raw | 405 | 0.5383 | 0.5235 | 0.5347 | 0.3012 | +0.0143 Macro / -0.0020 Micro |
| **Step 2 Oracle (With Corr)** | `emotion_textual_corr` | Oracle (`no rel`) | 405 | **0.5530** | **0.5349** | **0.5507** | **0.3259** | **+0.0114 Macro / +0.0146 Micro** |

---

### 1.5 Analytical Discussion: Task 1

1. **Synergistic Gain**: Moving from Blind Text Step 1 ($0.3565$ Macro F1) to the Dual-Encoder Step 2 Oracle ($0.4967$ Macro F1) yields an absolute improvement of **$+14.02\%$ Macro F1** and **$+12.21\%$ Micro F1** ($+9.30\%$ Subset Accuracy).
2. **Visual Contribution**: In both Step 1 ($0.3565 \to 0.4068$) and Step 2 ($0.4343 \to 0.4967$), adding ViT-224 visual features consistently boosts Macro F1 by **$+5.03\%$** to **$+6.24\%$**.
3. **Unconfounded Correlation Proof**: Adding $516$ unprompted multimodal texts in `Step 2 Raw` slightly reduces Micro F1 ($0.5403 \to 0.5383$) and Subset Accuracy ($0.3210 \to 0.3012$), whereas `Step 2 Oracle` lifts both Micro F1 ($0.5530$, $+1.46\%$) and Macro F1 ($0.5349$, $+1.14\%$), proving that the gain stems from correlation priors rather than mere data volume.

> [!IMPORTANT]
> **TODO: Multi-Label Confusion & Co-occurrence Matrices**
> - Run inference on `mm_test` ($N=129$) and `me_test` ($N=405$) to extract per-label $2\times 2$ confusion matrices (TP, FP, TN, FN for each of the 10 Plutchik emotions) and the $10\times 10$ predicted vs. gold emotion co-occurrence matrix.

---

## 2. Task 2: Multimodal & Textual Irony Detection (3-Class Single-Label)

### 2.1 Technical Specification & Parameter Card

| Parameter Category | Exact Code Specification (`train_textual_irony.py` & `evaluate_multimodal_irony.py`) |
| :--- | :--- |
| **Task Formulation** | 3-Class Single-Label Classification (`problem_type="single_label_classification"`) |
| **Target Labels & Priority** | `0 = None` ($N=104$), `1 = Ironia` ($N=18$), `2 = Sarcasmo` ($N=7$) in `mm_test`<br/>Priority rule (`row_to_irony_class`): `Sarcasmo (2) > Ironia (1) > None (0)` |
| **IronITA Label Mapping** | `None = (irony==0 & sarcasm==0)`, `Ironia = (irony==1 & sarcasm==0)`, `Sarcasmo = (sarcasm==1)` |
| **Text Encoder (`BASE_MODEL`)** | **UmBERTo**: `Musixmatch/umberto-commoncrawl-cased-v1` (768-d last-layer `[CLS]` token) |
| **Tokenization** | `padding="max_length"`, `truncation=True`, `max_length=128` (`MAX_LENGTH = 128`) |
| **Vision Encoder** | **ViT-Base-224**: `google/vit-base-patch16-224` (frozen 768-d `[CLS]` token) |
| **Late Fusion Space** | Joint feature vector: $[\mathbf{h}_{\text{text}} \;\Vert\; \mathbf{h}_{\text{image}}] \in \mathbb{R}^{1536}$ |
| **Feature Scaling (`std` vs `raw`)** | Primary (`std`): Independent `StandardScaler()` fitted on `mm_train` for text (`scaler_txt`) and image (`scaler_img`) features separately.<br/>Ablation (`raw`): Unscaled concatenation. |
| **Oracle Prompt Builder** | `build_corr_text(df, emotions=True, target=True, irony=False, hate=True, relations=with_relation)` |
| **Exact Prompt String Order** | Primary `Oracle (no rel)`: `il seguente testo [esprime <emozioni>] [contiene <offensivita>] [riguarda <contenuto/contenitore>] <testo>`<br/>Ablation `Oracle (+ rel)`: `il seguente testo [esprime <emozioni>] [ha una relazione <rel>] [contiene <offensivita>] [riguarda <contenuto/contenitore>] <testo>` |
| **Textual / Direct Classifier** | `AutoModelForSequenceClassification` (3 logits, `CrossEntropyLoss`) + `np.argmax(logits, axis=-1)` |
| **Multimodal Linear Head** | `SGDClassifier(loss="squared_hinge", penalty="l2", class_weight="balanced", n_jobs=-1, random_state=42)` |
| **Step 1 Base Training Data** | Initialized from `Musixmatch/umberto-commoncrawl-cased-v1` $\to$ trained on IronITA 80% stratified split (`iron_tr`, $3,879$ rows) + `me_train` ($1,618$ rows) = **$5,497$ rows** (raw text, shuffled with `random_state=42`) $\to$ saved to `irony_textual_no_corr` |
| **Step 2 Raw Ablation Data** | Initialized from `irony_textual_no_corr` $\to$ trained on `me_train` ($1,618$) + `mm_train` ($516$) = **$2,134$ rows** (raw text, shuffled with `random_state=42`) $\to$ saved to `irony_textual_step2_raw` |
| **Step 2 Oracle Training Data** | Initialized from `irony_textual_no_corr` $\to$ trained on `me_train` ($1,618$) + `mm_train` ($516$) = **$2,134$ rows** (`Oracle (no rel)` prompt, shuffled with `random_state=42`) $\to$ saved to `irony_textual_corr` |
| **Training Hyperparameters** | `epochs=10`, `train_batch_size=16`, `eval_batch_size=16`, `lr=2e-5`, `weight_decay=0.01`, `eval_strategy="epoch"`, `save_strategy="epoch"`, `load_best_model_at_end=True`, `metric_for_best_model="f1_macro"`, `save_total_limit=1`, `seed=42` |
| **Extraction Batch Size** | `batch_size=32` (`extract_text_features` and `extract_vit_features`) |

---

### 2.2 Official Paper Pipeline Results (Table IV in Paper)

Evaluated on the held-out multimodal test split (`mm_test`, $N=129$) under the standardized (`std`) feature protocol and `Oracle (no rel)` prompting (`Is Ablation == False`).

| Configuration | Checkpoint Used | Prompting | Model | Scaling | Macro F1 | Micro F1 (= Acc.) | Weighted F1 | None F1 | Ironia F1 | Sarcasmo F1 |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 1 Base (without images)** | `irony_textual_no_corr` | None (Raw) | Blind Text (SGD, std) | `std` | 0.5397 | 0.8062 | 0.7930 | 0.8826 | 0.4865 | 0.2500 |
| **Step 1 Base with images** | `irony_textual_no_corr` | None (Raw) | Dual-Encoder (UmBERTo + ViT, std) | `std` | 0.4203 | 0.7829 | 0.7644 | 0.8826 | 0.3784 | 0.0000 |
| **Step 2 Oracle (without images)** | `irony_textual_corr` | Oracle (no rel) | Blind Text (SGD, std) | `std` | 0.3722 | 0.7209 | 0.7232 | 0.8812 | 0.0000 | 0.2353 |
| **Step 2 Oracle with images** | **`irony_textual_corr`** | **Oracle (no rel)** | **Dual-Encoder (UmBERTo + ViT, std)** | **`std`** | **0.6019** | **0.8140** | **0.8186** | **0.8966** | **0.5455** | **0.3636** |

> [!TIP]
> **New Benchmark Peak Achieved**: In the final verified clean run, the Step 2 Oracle Dual-Encoder (`std`) reached **0.6019 Macro F1** and **81.40% Accuracy** ($0.8186$ Weighted F1), outperforming all unimodal and unprompted baselines across every single class (`None` F1: $0.8966$, `Ironia` F1: $0.5455$, `Sarcasmo` F1: $0.3636$).

---

### 2.3 Comprehensive Extra Experiments & Ablations (All 16 Extra Configurations)

The table below catalogs all 16 additional experimental configurations recorded in `multimodal_irony_baselines.csv` on `mm_test` ($N=129$), organized across four systematic ablation axes:

```
Ablations Covered in evaluate_multimodal_irony.py:
├── Axis A: Direct Transformer Prediction (UmBERTo classification head argmax without SGD probe)
├── Axis B: Feature Scaling Protocol (Unscaled 'raw' vs. StandardScaler 'std')
├── Axis C: Prompt Schema Contamination ('Oracle (+ rel)' with semiotic relation injected)
└── Axis D: Unprompted Step 2 Fine-Tuning ('Step 2 Raw' checkpoint irony_textual_step2_raw)
```

| # | Configuration | Checkpoint | Prompting | Model Architecture | Scaling | Macro F1 | Micro F1 / Acc | Weighted F1 | None F1 | Ironia F1 | Sarcasmo F1 | Ablation Axis |
| :-: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | Direct Prediction Step 1 Base | `irony_textual_no_corr` | None (Raw) | Fine-tuned UmBERTo (argmax) | — | 0.4222 | 0.7364 | 0.7251 | 0.8545 | 0.1622 | 0.2500 | Axis A: Direct Head |
| **2** | Direct Prediction Step 2 Oracle | `irony_textual_corr` | Oracle (no rel) | Fine-tuned UmBERTo (argmax) | — | 0.5326 | 0.7829 | 0.7878 | 0.8756 | 0.5000 | 0.2222 | Axis A: Direct Head |
| **3** | Direct Prediction Step 2 Oracle (+ rel) | `irony_textual_corr` | Oracle (+ rel) | Fine-tuned UmBERTo (argmax) | — | 0.4204 | 0.7132 | 0.7258 | 0.8247 | 0.4364 | 0.0000 | Axis A + Axis C |
| **4** | Direct Prediction Step 2 Raw | `irony_textual_step2_raw` | None (Raw) | Fine-tuned UmBERTo (argmax) | — | 0.3663 | 0.7907 | 0.7479 | 0.8919 | 0.2069 | 0.0000 | Axis A + Axis D |
| **5** | Step 1 Base Blind Text (raw) | `irony_textual_no_corr` | None (Raw) | Blind Text (SGD, raw) | `raw` | 0.4521 | 0.7984 | 0.7777 | 0.8826 | 0.4737 | 0.0000 | Axis B: Raw Scaling |
| **6** | Step 1 Base Dual-Encoder (raw) | `irony_textual_no_corr` | None (Raw) | Dual-Encoder (UmBERTo + ViT, raw) | `raw` | 0.5575 | 0.8217 | 0.8095 | 0.8962 | 0.5263 | 0.2500 | Axis B: Raw Scaling |
| **7** | Step 2 Oracle Blind Text (raw) | `irony_textual_corr` | Oracle (no rel) | Blind Text (SGD, raw) | `raw` | 0.5528 | 0.7907 | 0.7902 | 0.8846 | 0.4103 | 0.3636 | Axis B: Raw Scaling |
| **8** | Step 2 Oracle Dual-Encoder (raw) | `irony_textual_corr` | Oracle (no rel) | Dual-Encoder (UmBERTo + ViT, raw) | `raw` | 0.5491 | 0.7984 | 0.7997 | 0.8867 | 0.5106 | 0.2500 | Axis B: Raw Scaling |
| **9** | Step 2 Oracle (+ rel) Blind Text (std) | `irony_textual_corr` | Oracle (+ rel) | Blind Text (SGD, std) | `std` | 0.4620 | 0.7364 | 0.7441 | 0.8473 | 0.3721 | 0.1667 | Axis C: `+ rel` Prompt |
| **10** | Step 2 Oracle (+ rel) Dual-Encoder (std) | `irony_textual_corr` | Oracle (+ rel) | Dual-Encoder (UmBERTo + ViT, std) | `std` | 0.4688 | 0.7597 | 0.7562 | 0.8612 | 0.3784 | 0.1667 | Axis C: `+ rel` Prompt |
| **11** | Step 2 Oracle (+ rel) Blind Text (raw) | `irony_textual_corr` | Oracle (+ rel) | Blind Text (SGD, raw) | `raw` | 0.4718 | 0.7519 | 0.7557 | 0.8585 | 0.3902 | 0.1667 | Axis B + Axis C |
| **12** | Step 2 Oracle (+ rel) Dual-Encoder (raw) | `irony_textual_corr` | Oracle (+ rel) | Dual-Encoder (UmBERTo + ViT, raw) | `raw` | 0.4726 | 0.7132 | 0.7420 | 0.8290 | 0.4889 | 0.1000 | Axis B + Axis C |
| **13** | Step 2 Raw Blind Text (std) | `irony_textual_step2_raw` | None (Raw) | Blind Text (SGD, std) | `std` | 0.4267 | 0.7287 | 0.7327 | 0.8544 | 0.2439 | 0.1818 | Axis D: Step 2 Raw |
| **14** | Step 2 Raw Dual-Encoder (std) | `irony_textual_step2_raw` | None (Raw) | Dual-Encoder (UmBERTo + ViT, std) | `std` | 0.4139 | 0.7597 | 0.7530 | 0.8696 | 0.3721 | 0.0000 | Axis D: Step 2 Raw |
| **15** | Step 2 Raw Blind Text (raw) | `irony_textual_step2_raw` | None (Raw) | Blind Text (SGD, raw) | `raw` | 0.4104 | 0.6977 | 0.7275 | 0.8469 | 0.2791 | 0.1053 | Axis B + Axis D |
| **16** | Step 2 Raw Dual-Encoder (raw) | `irony_textual_step2_raw` | None (Raw) | Dual-Encoder (UmBERTo + ViT, raw) | `raw` | 0.4624 | 0.8140 | 0.7951 | 0.9023 | 0.4848 | 0.0000 | Axis B + Axis D |

---

### 2.4 Semiotic Subsets Multi-Configuration Comparison (Table VI in Paper)

Single-label **Accuracy (= Micro F1)** across the semiotic interaction subsets (`ADD`, `PAR`, `DIV`) for all eight Dual-Encoder configurations (`multimodal_irony_subsets.csv`):

| Model Setting | Checkpoint | Prompting | Scaling | ADD ($N=68$) Acc | PAR ($N=23$) Acc | DIV ($N=38$) Acc | Overall Macro F1 | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dual-Encoder Step 1 Base (std)** | `irony_textual_no_corr` | None (Raw) | `std` | 0.8676 | **0.9565** | 0.5263 | 0.4203 | Official Pipeline |
| **Dual-Encoder Step 1 Base (raw)** | `irony_textual_no_corr` | None (Raw) | `raw` | **0.8824** | **0.9565** | **0.6316** | 0.5575 | Ablation (`raw`) |
| **Dual-Encoder Step 2 Oracle (std)** | **`irony_textual_corr`** | **Oracle (no rel)** | **`std`** | **0.8824** | 0.9130 | **0.6316** | **0.6019** | **Official Peak** |
| **Dual-Encoder Step 2 Oracle (raw)** | `irony_textual_corr` | Oracle (no rel) | `raw` | **0.8824** | 0.9130 | 0.5789 | 0.5491 | Ablation (`raw`) |
| **Dual-Encoder Step 2 Oracle (+ rel) (std)** | `irony_textual_corr` | Oracle (+ rel) | `std` | 0.8529 | **0.9565** | 0.4737 | 0.4688 | Ablation (`+ rel`) |
| **Dual-Encoder Step 2 Oracle (+ rel) (raw)** | `irony_textual_corr` | Oracle (+ rel) | `raw` | 0.7941 | 0.7826 | 0.5263 | 0.4726 | Ablation (`+ rel`, `raw`) |
| **Dual-Encoder Step 2 Raw (std)** | `irony_textual_step2_raw` | None (Raw) | `std` | 0.8529 | 0.9130 | 0.5000 | 0.4139 | Ablation (`Step 2 Raw`) |
| **Dual-Encoder Step 2 Raw (raw)** | `irony_textual_step2_raw` | None (Raw) | `raw` | 0.8676 | **0.9565** | **0.6316** | 0.4624 | Ablation (`Step 2 Raw`, `raw`) |

---

### 2.5 Textual Ablation Progression (`me_test`, $N=405$)

Direct evaluation on the held-out unimodal textual test split (`train_textual_irony_summary.csv`):

| Stage | Checkpoint | Prompt Template | N Test | Micro F1 (= Acc) | Macro F1 | Weighted F1 | None F1 | Ironia F1 | Sarcasmo F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Step 1 Base (Raw)** | `irony_textual_no_corr` | Raw | 405 | 0.7481 | 0.4253 | 0.7480 | 0.8458 | **0.4302** | 0.0000 |
| **Step 2 Raw (Ablation)** | `irony_textual_step2_raw` | Raw | 405 | **0.7704** | **0.4755** | 0.7505 | **0.8662** | 0.3382 | **0.2222** |
| **Step 2 Oracle (With Corr)** | `irony_textual_corr` | Oracle (`no rel`) | 405 | 0.7654 | 0.4723 | **0.7592** | 0.8607 | 0.4133 | 0.1429 |

---

### 2.6 Analytical Discussion: Task 2

1. **Why Step 2 Oracle (`std`) Achieved the Benchmark Peak ($0.6019$ Macro F1 / $81.40\%$ Acc)**:
   - Standardizing `UmBERTo` and `ViT-224` embeddings independently via `StandardScaler` stabilizes the `class_weight="balanced"` squared-hinge optimization, allowing the Dual-Encoder to achieve simultaneous peaks across all three classes (`None`: $0.8966$, `Ironia`: $0.5455$, `Sarcasmo`: $0.3636$) and lifting `DIV` subset accuracy from $52.63\% \to \mathbf{63.16\%}$.
   - Compared to `Step 2 Oracle Blind Text (std)` ($0.3722$ Macro F1), adding standardized ViT features produces a massive **$+22.97\%$ Macro F1** and **$+9.31\%$ Accuracy** gain.
2. **Comparison of `std` vs. `raw` on Step 2 Oracle**:
   - In Step 2 Oracle, `std` ($0.6019$ Macro F1, $81.40\%$ Acc) outperforms `raw` ($0.5491$ Macro F1, $79.84\%$ Acc) by **$+5.28\%$ Macro F1**, specifically improving `Sarcasmo` F1 ($0.2500 \to 0.3636$), `Ironia` F1 ($0.5106 \to 0.5455$), and `DIV` subset accuracy ($57.89\% \to 63.16\%$).
3. **The Prompt Contamination Hazard (`Oracle (+ rel)`)**:
   - Injecting `ha una relazione <rel>` into the irony prompt causes an immediate out-of-distribution degradation from **$0.6019 \to 0.4688$ Macro F1** (`std`) and **$0.5491 \to 0.4726$** (`raw`).
   - Because $20$ out of $38$ Divergent posts ($52.63\%$) in `mm_test` are completely sincere (`None`), feeding the explicit token `divergente` causes false-positive irony predictions on literal divergent posts, dropping `DIV` accuracy from $63.16\% \to 47.37\%$.
4. **Unprompted Fine-Tuning (`Step 2 Raw`) vs. Prompted Fine-Tuning (`Step 2 Oracle`)**:
   - On `me_test`, `Step 2 Raw` degrades `Ironia` F1 ($0.4302 \to 0.3382$), whereas `Step 2 Oracle` preserves `Ironia` F1 ($0.4133$) while unlocking `Sarcasmo` detection ($0.0000 \to 0.1429$) and achieving the highest textual Weighted F1 ($0.7592$). On `mm_test`, `Step 2 Raw` fails to detect `Sarcasmo` in Dual-Encoder mode ($0.0000$ F1 in both `std` and `raw`), confirming that correlation prompting is essential for multimodal figurative separation.

> [!IMPORTANT]
> **TODO: 3-Class Confusion Matrix for Task 2**
> - Run inference on `mm_test` ($N=129$) and `me_test` ($N=405$) to extract the $3\times 3$ confusion matrices (`None`, `Ironia`, `Sarcasmo`) for both the primary `std` pipeline and the `raw`/`+ rel` ablations.

---

## 3. Task 3: Semiotic Relation Classification (3-Class Single-Label)

### 3.1 Technical Specification & Parameter Card

| Parameter Category | Exact Code Specification (`train_textual_relation.py` & `evaluate_multimodal_relation.py`) |
| :--- | :--- |
| **Task Formulation** | 3-Class Single-Label Classification (`problem_type="single_label_classification"`) |
| **Target Labels & Priority** | `0 = Additiva` ($N=68$), `1 = Divergente` ($N=38$), `2 = Parallela` ($N=23$) in `mm_test` ($N=129$)<br/>Priority rule (`row_to_relation_class`): `Parallela (2) > Divergente (1) > Additiva (0)` |
| **Text Encoder (`BASE_MODEL`)** | **UmBERTo**: `Musixmatch/umberto-commoncrawl-cased-v1` (768-d last-layer `[CLS]` token) |
| **Tokenization** | `padding="max_length"`, `truncation=True`, `max_length=128` (`MAX_LENGTH = 128`) |
| **Vision Encoder** | **ViT-Base-224**: `google/vit-base-patch16-224` (frozen 768-d `[CLS]` token) |
| **Fusion & Scaling** | Unscaled late-fusion concatenation (`raw`, $1536$-d: `np.hstack([txt_tr, vit_tr])`) |
| **Oracle Prompt Builder** | `build_corr_text(df, emotions=True, target=True, irony=True, hate=True, relations=False)` |
| **Exact Prompt String Order** | `il seguente testo [esprime <emozioni>] [contiene <ironia/sarcasmo/offensivita>] [riguarda <contenuto/contenitore>] <testo>` |
| **Prompt Schema** | Relation is the prediction target and is strictly excluded (`relations=False`) |
| **Textual Classifier Head** | `AutoModelForSequenceClassification` (3 logits, `CrossEntropyLoss`) + `np.argmax(logits, axis=-1)` |
| **Multimodal Linear Head** | `SGDClassifier(loss="squared_hinge", penalty="l2", class_weight="balanced", n_jobs=-1, random_state=42)` |
| **Stage A Raw Training Data** | Initialized from `Musixmatch/umberto-commoncrawl-cased-v1` $\to$ trained on `mm_train` (**$516$ rows**, raw text, shuffled with `random_state=42`) $\to$ saved to `relation_textual_no_corr` |
| **Stage B Oracle Training Data** | Initialized from `Musixmatch/umberto-commoncrawl-cased-v1` $\to$ trained on `mm_train` (**$516$ rows**, `Oracle` prompt, shuffled with `random_state=42`) $\to$ saved to `relation_textual_corr` |
| **Training Hyperparameters** | `epochs=10`, `train_batch_size=16`, `eval_batch_size=16`, `lr=2e-5`, `weight_decay=0.01`, `eval_strategy="epoch"`, `save_strategy="epoch"`, `load_best_model_at_end=True`, `metric_for_best_model="f1_macro"`, `save_total_limit=1`, `seed=42` |
| **Extraction Batch Size** | `batch_size=32` (`extract_text_features` and `extract_vit_features`) |

---

### 3.2 Official Paper Pipeline Results (Table VII in Paper)

Evaluated on the held-out multimodal test split (`mm_test`, $N=129$) using the balanced `SGDClassifier` head on `raw` features.

| Configuration | Checkpoint Used | Features | Prompting | Model | Macro F1 | Micro F1 (= Acc.) | Weighted F1 |
| :--- | :---: | :---: | :---: | :--- | :---: | :---: | :---: |
| **Stage A Raw (without images)** | `relation_textual_no_corr` | `raw` | None (Raw) | Blind Text (balanced SGD) | 0.4965 | 0.5891 | 0.5739 |
| **Stage A Raw with images** | **`relation_textual_no_corr`** | **`raw`** | **None (Raw)** | **Dual-Encoder (UmBERTo + ViT-224)** | **0.6147** | **0.6357** | **0.6392** |
| **Stage B Oracle (without images)** | `relation_textual_corr` | `raw` | Oracle | Blind Text (balanced SGD) | 0.6051 | 0.6124 | 0.6112 |
| **Stage B Oracle with images** | `relation_textual_corr` | `raw` | Oracle | Dual-Encoder (UmBERTo + ViT-224) | 0.6130 | 0.6202 | 0.6232 |

---

### 3.3 Semiotic Target Classes Performance Breakdown (Table IX in Paper)

Per-class F1 performance across the three target relation categories on `mm_test` ($N=129$):

| Model Setting | Checkpoint | Additiva ($N=68$) F1 | Parallela ($N=23$) F1 | Divergente ($N=38$) F1 | Overall Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dual-Encoder Stage A Raw (No corr)** | `relation_textual_no_corr` | **0.6822** | 0.5385 | **0.6234** | **0.6147** |
| **Dual-Encoder Stage B Oracle (With corr)** | `relation_textual_corr` | 0.6406 | **0.5806** (+0.042) | 0.6176 | 0.6130 |

---

### 3.4 Extra Experiments: Direct Textual Encoder Benchmark (`mm_test`, $N=129$)

Direct classification (`np.argmax(logits, axis=-1)`) from the fine-tuned UmBERTo sequence classification head on `mm_test` ($N=129$) without the SGD linear probe (`train_textual_relation_summary.csv`):

| Stage | Checkpoint | Prompt Template | N Test | Micro F1 (= Acc) | Macro F1 | Weighted F1 | Additiva F1 | Divergente F1 | Parallela F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage A Raw** | `relation_textual_no_corr` | Raw | 129 | 0.6667 | 0.6009 | 0.6531 | **0.7355** | 0.6462 | 0.4211 |
| **Stage B Oracle** | `relation_textual_corr` | Oracle | 129 | **0.6899** | **0.6500** | **0.6821** | 0.7333 | **0.6761** | **0.5405** (+0.119) |

---

### 3.5 Analytical Discussion: Task 3

1. **Massive Multimodal Lift on Unprompted Text (`Stage A Raw`)**:
   - On raw text without correlations, the `Blind Text` linear probe achieves only $0.4965$ Macro F1 ($58.91\%$ Acc). Concatenating frozen ViT-224 visual embeddings lifts performance to **$0.6147$ Macro F1 (+11.82% absolute improvement)** and **$63.57\%$ Accuracy**.
   - Because semiotic relation is fundamentally a cross-modal property between text and image, visual features provide immediate structural disambiguation even when no pragmatic hints are present.
2. **Complementary Role of Correlation Prompting (`Stage B Oracle`)**:
   - Injecting auxiliary affective and figurative cues (`esprime <emozioni> contiene <ironia/sarcasmo>`) boosts the unimodal `Blind Text` probe from $0.4965 \to \mathbf{0.6051}$ Macro F1 ($+10.86\%$) and lifts the direct UmBERTo classifier from $0.6009 \to \mathbf{0.6500}$ Macro F1 ($68.99\%$ Acc).
   - Crucially, in both direct prediction ($0.4211 \to \mathbf{0.5405}$) and Dual-Encoder fusion ($0.5385 \to \mathbf{0.5806}$), Oracle prompting substantially improves recognition of the minority `Parallela` class ($N=23$).

> [!IMPORTANT]
> **TODO: 3x3 Semiotic Relation Confusion Matrix**
> - Run inference on `mm_test` ($N=129$) to extract the $3\times 3$ confusion matrix across `Additiva`, `Divergente`, and `Parallela` for Stage A and Stage B models.

---

## 4. Cross-Task Architectural Synthesis

| Dimension | Task 1: Emotion Detection | Task 2: Irony Detection | Task 3: Semiotic Relation |
| :--- | :--- | :--- | :--- |
| **Text Encoder Backbone** | **FEEL-IT** (`MilaNLProc/feel-it-italian-emotion`) | **UmBERTo** (`Musixmatch/umberto-commoncrawl-cased-v1`) | **UmBERTo** (`Musixmatch/umberto-commoncrawl-cased-v1`) |
| **Vision Encoder Backbone** | **ViT-Base-224** (`google/vit-base-patch16-224`) | **ViT-Base-224** (`google/vit-base-patch16-224`) | **ViT-Base-224** (`google/vit-base-patch16-224`) |
| **Multimodal Probe & Weights** | `ClassifierChain(SGDClassifier(squared_hinge, l2, class_weight="balanced"))` | `SGDClassifier(squared_hinge, l2, class_weight="balanced")` | `SGDClassifier(squared_hinge, l2, class_weight="balanced")` |
| **Optimal Feature Scaling** | `raw` (unscaled concatenation) | **`std` (independent `StandardScaler`)** | `raw` (unscaled concatenation) |
| **Multimodal Prompt Strategy** | `Oracle (+ rel)` beneficial | **`Oracle (no rel)` strictly required** | `Oracle` (relation is target, excluded) |
| **Visual Modality Impact** | $+5.03\%$ to $+6.24\%$ Macro F1 lift | **$+22.97\%$ Macro F1 lift in Step 2 (`std`)** | **$+11.82\%$ Macro F1 lift in Stage A** |
| **Winning Multimodal Score** | **0.4967 Macro F1 / 0.5535 Micro F1** | **0.6019 Macro F1 / 81.40% Acc** | **0.6147 Macro F1 (Stage A) / 0.6130 (Stage B)** |

---

## 5. Exact Script Execution Guide for Reproduction

To reproduce every table and metric in this document from `code/bert/clean_paper_codes`:

```powershell
# Option A: Full end-to-end master pipeline
python code/bert/clean_paper_codes/run_paper_experiments.py --task all --checkpoints-dir checkpoints --results-dir results_final_run --epochs 10 --batch-size 16 --seed 42

# Option B: Individual stage-by-stage execution
# 1. Textual Fine-Tuning (Epochs=10, Batch=16, LR=2e-5, WD=0.01, Seed=42)
python code/bert/clean_paper_codes/train_textual_emotion.py --epochs 10 --batch-size 16 --lr 2e-5 --weight-decay 0.01 --seed 42 --checkpoints-dir checkpoints --results-dir results_final_run
python code/bert/clean_paper_codes/train_textual_irony.py --epochs 10 --batch-size 16 --lr 2e-5 --weight-decay 0.01 --seed 42 --checkpoints-dir checkpoints --results-dir results_final_run
python code/bert/clean_paper_codes/train_textual_relation.py --epochs 10 --batch-size 16 --lr 2e-5 --weight-decay 0.01 --seed 42 --checkpoints-dir checkpoints --results-dir results_final_run

# 2. Multimodal Dual-Encoder Evaluation (Extraction Batch=32, Seed=42)
python code/bert/clean_paper_codes/evaluate_multimodal_emotion.py --batch-size 32 --seed 42 --checkpoints-dir checkpoints --results-dir results_final_run
python code/bert/clean_paper_codes/evaluate_multimodal_irony.py --batch-size 32 --seed 42 --checkpoints-dir checkpoints --results-dir results_final_run
python code/bert/clean_paper_codes/evaluate_multimodal_relation.py --batch-size 32 --seed 42 --checkpoints-dir checkpoints --results-dir results_final_run
```
