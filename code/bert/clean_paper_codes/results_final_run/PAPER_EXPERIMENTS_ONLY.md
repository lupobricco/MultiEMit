# MultiEMiT: Official Paper Experiments & Specifications (`raw` Unified Protocol)

> **Scope**: This document contains **exclusively the official experiments selected for the paper** across all three MultiEMiT tasks (**Task 1: Emotion Detection**, **Task 2: Irony Detection**, **Task 3: Semiotic Relation Classification**), using a **unified unscaled (`raw`) feature fusion protocol** across every task.

---

## Global Paper Methodology & Unified Architecture

All three tasks follow a unified, deterministic pipeline (`SEED = 42`) with zero test-set leakage:

1. **Dataset Partitions (`SEED = 42`, 80/20 Train/Test Split)**:
   - **MultiEMiT Textual Corpus (`textual_only_cleaned.csv`, $N = 2,023$)**: `me_train` ($1,618$ rows) and `me_test` ($405$ rows).
   - **MultiEMiT Multimodal Corpus ($N = 645$)**: `mm_train` (`multimodal_train_seed42.csv`, $516$ rows) and `mm_test` (`multimodal_test_seed42.csv`, $129$ rows: $68$ `Additiva`, $38$ `Divergente`, $23$ `Parallela`).
   - **External Warm-Up Corpora (Step 1)**:
     - **EMit-A** (`cleaned_emit_train_A.csv`, $N = 5,966$): 80/20 split (`random_state=42`) $\to$ `emit_tr` ($4,772$ train) / `emit_val` ($1,194$ val).
     - **IronITA** (`ironita_full.xlsx`, $N = 4,849$): 80/20 stratified split (`random_state=42`, `stratify=label`) $\to$ `iron_tr` ($3,879$ train) / `iron_val` ($970$ val).
2. **Textual Encoders (`MAX_LENGTH = 128`)**:
   - **Task 1 (Emotion)**: **FEEL-IT** (`MilaNLProc/feel-it-italian-emotion`), 768-d last-layer `[CLS]` token (`hidden_states[-1][:, 0, :]`).
   - **Task 2 (Irony) & Task 3 (Semiotic Relation)**: **UmBERTo** (`Musixmatch/umberto-commoncrawl-cased-v1`), 768-d last-layer `[CLS]` token (`hidden_states[-1][:, 0, :]`).
   - **Fine-Tuning Hyperparameters**: `epochs = 10`, `train_batch_size = 16`, `eval_batch_size = 16`, `learning_rate = 2e-5`, `weight_decay = 0.01`, `eval_strategy = "epoch"`, `save_strategy = "epoch"`, `load_best_model_at_end = True`, `metric_for_best_model = "f1_macro"`, `seed = 42`.
3. **Visual Encoder (`batch_size = 32`)**:
   - **ViT-Base-224** (`google/vit-base-patch16-224`): Frozen 768-d last-layer `[CLS]` embedding (`out[:, 0, :]`) extracted from $224 \times 224$ RGB images.
4. **Unified Multimodal Late-Fusion & Linear Probe (`raw`)**:
   - **Feature Representation**: Direct unscaled (`raw`) concatenation of frozen 768-d text `[CLS]` and frozen 768-d ViT `[CLS]` features:
     $$\mathbf{h}_{\text{joint}} = [\mathbf{h}_{\text{text}} \;\Vert\; \mathbf{h}_{\text{image}}] \in \mathbb{R}^{1536}$$
   - **Base Linear Classifier (`fit_linear_head`)**:
     ```python
     SGDClassifier(loss="squared_hinge", penalty="l2", class_weight="balanced", n_jobs=-1, random_state=42)
     ```
   - Wrapped in `ClassifierChain(base)` for 10-class multi-label classification (Task 1) and applied directly as `base` for 3-class single-label classification (Tasks 2 & 3).

---

## 1. Task 1: Emotion Detection (10-Class Multi-Label)

### 1.1 Exact Paper Configuration Card

| Parameter | Specification (`train_textual_emotion.py` & `evaluate_multimodal_emotion.py`) |
| :--- | :--- |
| **Task Type & Labels (10)** | Multi-label classification: `Amore`, `Anticipazione`, `Disgusto`, `Fiducia`, `Gioia`, `Neutro`, `Paura`, `Rabbia`, `Sorpresa`, `Tristezza` |
| **Text Encoder** | **FEEL-IT** (`MilaNLProc/feel-it-italian-emotion`, 768-d) |
| **Vision Encoder** | **ViT-Base-224** (`google/vit-base-patch16-224`, frozen 768-d) |
| **Step 1 Base Training (`No corr`)** | `emit_tr` ($4,772$) + `me_train` ($1,618$) = **$6,390$ rows** (raw text) $\to$ checkpoint `emotion_textual_no_corr` |
| **Step 2 Oracle Training (`With corr`)** | Initialized from `emotion_textual_no_corr` $\to$ trained on `me_train` ($1,618$) + `mm_train` ($516$) = **$2,134$ rows** with `Oracle (no rel)` prompt $\to$ checkpoint `emotion_textual_corr` |
| **Oracle Prompt Template** | **Textual (`me_test`)**: `il seguente testo [contiene <ironia/sarcasmo/offensivita>] [riguarda <contenuto/contenitore>] <testo>` (`with_relation=False`)<br>**Multimodal (`mm_train` & `mm_test`)**: `il seguente testo [ha una relazione <additiva/divergente/parallela>] [contiene <ironia/sarcasmo/offensivita>] [riguarda <contenuto/contenitore>] <testo>` (`with_relation=True`) |
| **Textual Inference (`me_test`)** | Sequence classification head (`BCEWithLogitsLoss`) + Sigmoid threshold `EMOTION_THRESHOLD = 0.40` with dynamic fallback (if all $\sigma(z_c) < 0.40$, select `order[0]`, plus `order[1]` if `prob[order[0]] - prob[order[1]] <= 0.10`) |
| **Multimodal Probe (`mm_test`)** | Unscaled (`raw`) features $\to$ `ClassifierChain(SGDClassifier(loss="squared_hinge", penalty="l2", class_weight="balanced", random_state=42))` fitted on `mm_train` ($N=516$) |

### 1.2 Paper Table I: Emotion Detection Baselines (`me_test` & `mm_test`)

| Evaluation Partition | Configuration | Checkpoint | Prompting | Model | Features | Macro F1 | Micro F1 | Subset Acc. |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **Textual (`me_test`, $N=405$)** | Textual test set (No correlations) | `emotion_textual_no_corr` | None (Raw) | Fine-tuned FEEL-IT | — | 0.5092 *(0.509)* | 0.5403 *(0.540)* | 0.3210 |
| **Textual (`me_test`, $N=405$)** | **Textual test set (With correlations)** | **`emotion_textual_corr`** | **Oracle (no rel)** | **Fine-tuned FEEL-IT** | — | **0.5349 *(0.535)*** | **0.5530 *(0.553)*** | **0.3259** |
| **Multimodal (`mm_test`, $N=129$)** | Multimodal test set (without images, No corr.) | `emotion_textual_no_corr` | None (Raw) | Blind Text (`ClassifierChain` + balanced SGD) | `raw` | 0.3565 *(0.356)* | 0.4314 *(0.431)* | 0.2636 |
| **Multimodal (`mm_test`, $N=129$)** | Multimodal test set with images (Dual-Encoder, No corr.) | `emotion_textual_no_corr` | None (Raw) | Dual-Encoder (FEEL-IT + ViT-224) | `raw` | 0.4068 *(0.407)* | 0.4290 *(0.429)* | 0.2791 |
| **Multimodal (`mm_test`, $N=129$)** | Multimodal test set (without images, With corr.) | `emotion_textual_corr` | Oracle (+ rel) | Blind Text (`ClassifierChain` + balanced SGD) | `raw` | 0.4343 *(0.434)* | 0.5183 *(0.518)* | 0.3488 |
| **Multimodal (`mm_test`, $N=129$)** | **Multimodal test set with images and correlations (Dual-Encoder)** | **`emotion_textual_corr`** | **Oracle (+ rel)** | **Dual-Encoder (FEEL-IT + ViT-224)** | **`raw`** | **0.4967 *(0.497)*** | **0.5535 *(0.553)*** | **0.3566** |

### 1.3 Paper Table III: Emotion Micro F1 Across Semiotic Subsets (`mm_test`, $N=129$)

| Model Setting | Checkpoint | Prompting | Features | ADD ($N=68$) Micro F1 | PAR ($N=23$) Micro F1 | DIV ($N=38$) Micro F1 | Overall Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dual-Encoder (Step 1 Base, No corr.)** | `emotion_textual_no_corr` | None (Raw) | `raw` | 0.4624 *(0.46)* | 0.5172 *(0.52)* | 0.3168 *(0.32)* | 0.4068 *(0.407)* |
| **Dual-Encoder (Step 2 Oracle, With corr.)** | **`emotion_textual_corr`** | **Oracle (+ rel)** | **`raw`** | **0.5614 *(0.56)*** | **0.6296 *(0.63)*** | **0.4946 *(0.49)*** | **0.4967 *(0.497)*** |

### 1.4 Key Findings for Paper Discussion (Task 1)
- **Textual Correlation Gain (`me_test`)**: Oracle correlation prompting improves unimodal emotion recognition from $0.5092 \to \mathbf{0.5349}$ Macro F1 ($+2.57\%$) and $0.5403 \to \mathbf{0.5530}$ Micro F1 ($+1.27\%$).
- **Multimodal & Prompt Synergy (`mm_test`)**: Adding ViT-224 visual embeddings consistently boosts Macro F1 both without correlations ($0.3565 \to 0.4068$, $+5.03\%$) and with correlations ($0.4343 \to \mathbf{0.4967}$, $+6.24\%$). Combining both visual features and Oracle correlation prompts yields a total gain of **$+14.02\%$ Macro F1** and **$+12.21\%$ Micro F1** over the unprompted blind-text baseline.
- **Semiotic Subset Disambiguation**: While improvements are strong across Additive ($+9.90\%$ Micro F1) and Parallel ($+11.24\%$) posts, the largest gain occurs on **Divergent (`DIV`)** posts (**$+17.78\%$ Micro F1**, $0.3168 \to 0.4946$), where cross-layer pragmatic cues resolve semantic tension between image and text.

---

## 2. Task 2: Irony & Sarcasm Detection (3-Class Single-Label, `raw` Protocol)

### 2.1 Exact Paper Configuration Card

| Parameter | Specification (`train_textual_irony.py` & `evaluate_multimodal_irony.py`) |
| :--- | :--- |
| **Task Type & Labels (3)** | Single-label 3-class classification (`row_to_irony_class` priority: `Sarcasmo (2) > Ironia (1) > None (0)`) |
| **Text Encoder** | **UmBERTo** (`Musixmatch/umberto-commoncrawl-cased-v1`, 768-d) |
| **Vision Encoder** | **ViT-Base-224** (`google/vit-base-patch16-224`, frozen 768-d) |
| **Step 1 Base Training (`No corr`)** | `iron_tr` ($3,879$, stratified 80% of IronITA) + `me_train` ($1,618$) = **$5,497$ rows** (raw text) $\to$ checkpoint `irony_textual_no_corr` |
| **Step 2 Oracle Training (`With corr`)** | Initialized from `irony_textual_no_corr` $\to$ trained on `me_train` ($1,618$) + `mm_train` ($516$) = **$2,134$ rows** with `Oracle (no rel)` prompt $\to$ checkpoint `irony_textual_corr` |
| **Oracle Prompt Template** | **Textual (`me_test`) & Multimodal (`mm_train` & `mm_test`)**: `il seguente testo [esprime <emozioni>] [contiene <offensivita>] [riguarda <contenuto/contenitore>] <testo>` (`Oracle (no rel)`, `with_relation=False`) |
| **Textual Inference (`me_test`)** | Sequence classification head (`CrossEntropyLoss`, 3 logits) + `argmax` |
| **Multimodal Probe (`mm_test`)** | Unscaled (`raw`) features $\to$ `SGDClassifier(loss="squared_hinge", penalty="l2", class_weight="balanced", random_state=42)` fitted on `mm_train` ($N=516$) |

### 2.2 Paper Table IV: Irony & Sarcasm Detection Baselines (`me_test` & `mm_test`, `raw`)
*(Note: For single-label classification, Micro F1 is mathematically equal to Accuracy).*

| Evaluation Partition | Configuration | Checkpoint | Prompting | Model | Features | Macro F1 | Micro F1 (= Acc.) | Weighted F1 | `None` F1 | `Ironia` F1 | `Sarcasmo` F1 |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Textual (`me_test`, $N=405$)** | Textual test set (No correlations) | `irony_textual_no_corr` | None (Raw) | Fine-tuned UmBERTo | — | 0.4253 *(0.425)* | 0.7481 *(0.748)* | 0.7480 | 0.8458 | 0.4302 | 0.0000 |
| **Textual (`me_test`, $N=405$)** | **Textual test set (With correlations)** | **`irony_textual_corr`** | **Oracle (no rel)** | **Fine-tuned UmBERTo** | — | **0.4723 *(0.472)*** | **0.7654 *(0.765)*** | **0.7592** | **0.8607** | **0.4133** | **0.1429** |
| **Multimodal (`mm_test`, $N=129$)** | Multimodal test set (without images, No corr.) | `irony_textual_no_corr` | None (Raw) | Blind Text (balanced SGD) | `raw` | 0.4521 *(0.452)* | 0.7984 *(0.798)* | 0.7777 | 0.8826 | 0.4737 | 0.0000 |
| **Multimodal (`mm_test`, $N=129$)** | **Multimodal test set with images (Dual-Encoder, No corr.)** | **`irony_textual_no_corr`** | **None (Raw)** | **Dual-Encoder (UmBERTo + ViT-224)** | **`raw`** | **0.5575 *(0.558)*** | **0.8217 *(0.822)*** | **0.8095** | **0.8962** | **0.5263** | **0.2500** |
| **Multimodal (`mm_test`, $N=129$)** | Multimodal test set (without images, With corr.) | `irony_textual_corr` | Oracle (no rel) | Blind Text (balanced SGD) | `raw` | 0.5528 *(0.553)* | 0.7907 *(0.791)* | 0.7902 | 0.8846 | 0.4103 | **0.3636** |
| **Multimodal (`mm_test`, $N=129$)** | **Multimodal test set with images and correlations (Dual-Encoder)** | **`irony_textual_corr`** | **Oracle (no rel)** | **Dual-Encoder (UmBERTo + ViT-224)** | **`raw`** | **0.5491 *(0.549)*** | **0.7984 *(0.798)*** | **0.7997** | **0.8867** | **0.5106** | **0.2500** |

### 2.3 Paper Table VI: Irony Accuracy / Micro F1 Across Semiotic Subsets (`mm_test`, $N=129$, `raw`)

| Model Setting | Checkpoint | Prompting | Features | ADD ($N=68$) Acc | PAR ($N=23$) Acc | DIV ($N=38$) Acc | Overall Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dual-Encoder (Step 1 Base, No corr.)** | `irony_textual_no_corr` | None (Raw) | `raw` | **0.8824 *(0.88)*** | **0.9565 *(0.96)*** | **0.6316 *(0.63)*** | **0.5575 *(0.558)*** |
| **Dual-Encoder (Step 2 Oracle, With corr.)** | `irony_textual_corr` | Oracle (no rel) | `raw` | **0.8824 *(0.88)*** | 0.9130 *(0.91)* | 0.5789 *(0.58)* | 0.5491 *(0.549)* |

### 2.4 Key Findings for Paper Discussion (Task 2, `raw`)
- **Textual Correlation Gain (`me_test`)**: On the held-out textual split, Oracle correlation prompting boosts Macro F1 from $0.4253 \to \mathbf{0.4723}$ ($+4.70\%$) and Accuracy from $74.81\% \to \mathbf{76.54\%}$ ($+1.73\%$), unlocking detection of the extreme-minority `Sarcasmo` class ($0.0000 \to \mathbf{0.1429}$ F1).
- **Strong Visual Modality Impact in Step 1 (`mm_test`)**: Without correlation prompts, the text-only probe (`Blind Text, Step 1 Base`) completely fails to identify `Sarcasmo` ($0.0000$ F1, $0.4521$ Macro F1). Adding ViT-224 visual embeddings (`Dual-Encoder, Step 1 Base`) produces a **$+10.54\%$ Macro F1 jump** ($0.4521 \to \mathbf{0.5575}$) and reaches **$82.17\%$ Accuracy** ($0.8095$ Weighted F1), lifting `Ironia` F1 to $0.5263$ and `Sarcasmo` F1 to $0.2500$.
- **Complementary Effect of Correlations & Vision in Step 2 (`mm_test`)**:
  - Injecting Oracle correlation prompts into the text-only model (`Blind Text, Step 2 Oracle`) lifts Macro F1 by **$+10.07\%$** over the unprompted text baseline ($0.4521 \to \mathbf{0.5528}$), achieving the highest `Sarcasmo` F1 ($0.3636$) via affective cues (`Disgusto`, `Offensivita`), though at the expense of `Ironia` F1 ($0.4103$).
  - Combining visual features with correlation prompts (`Dual-Encoder, Step 2 Oracle`) restores balanced cross-class performance (`None` F1: $0.8867$, `Ironia` F1: $0.5106$ [+$10.03\%$ over Step 2 Blind Text], `Sarcasmo` F1: $0.2500$), improving both overall Accuracy ($79.07\% \to 79.84\%$) and Weighted F1 ($0.7902 \to 0.7997$).

---

## 3. Task 3: Semiotic Relation Classification (3-Class Single-Label)

### 3.1 Exact Paper Configuration Card

| Parameter | Specification (`train_textual_relation.py` & `evaluate_multimodal_relation.py`) |
| :--- | :--- |
| **Task Type & Labels (3)** | Single-label 3-class classification (`row_to_relation_class` priority: `Parallela (2) > Divergente (1) > Additiva (0)`) |
| **Text Encoder** | **UmBERTo** (`Musixmatch/umberto-commoncrawl-cased-v1`, 768-d) |
| **Vision Encoder** | **ViT-Base-224** (`google/vit-base-patch16-224`, frozen 768-d) |
| **Stage A Raw Training (`No corr`)** | Initialized from `UmBERTo` $\to$ trained on `mm_train` (**$516$ rows**, raw text) $\to$ checkpoint `relation_textual_no_corr` |
| **Stage B Oracle Training (`With corr`)** | Initialized from `UmBERTo` $\to$ trained on `mm_train` (**$516$ rows**, `Oracle` prompt) $\to$ checkpoint `relation_textual_corr` |
| **Oracle Prompt Template** | `il seguente testo [esprime <emozioni>] [contiene <ironia/sarcasmo/offensivita>] [riguarda <contenuto/contenitore>] <testo>` (`relations=False`, target excluded) |
| **Multimodal Probe (`mm_test`)** | Unscaled (`raw`) features $\to$ `SGDClassifier(loss="squared_hinge", penalty="l2", class_weight="balanced", random_state=42)` fitted on `mm_train` ($N=516$) |

### 3.2 Paper Table VII: Semiotic Relation Classification Baselines (`mm_test`, $N=129$)

| Evaluation Partition | Configuration | Checkpoint | Prompting | Model | Features | Macro F1 | Micro F1 (= Acc.) | Weighted F1 |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **Multimodal (`mm_test`, $N=129$)** | Multimodal test set (without images, Stage A Raw) | `relation_textual_no_corr` | None (Raw) | Blind Text (balanced SGD) | `raw` | 0.4965 *(0.496)* | 0.5891 *(0.589)* | 0.5739 *(0.574)* |
| **Multimodal (`mm_test`, $N=129$)** | **Multimodal test set with images (Dual-Encoder, Stage A Raw)** | **`relation_textual_no_corr`** | **None (Raw)** | **Dual-Encoder (UmBERTo + ViT-224)** | **`raw`** | **0.6147 *(0.615)*** | **0.6357 *(0.636)*** | **0.6392 *(0.639)*** |
| **Multimodal (`mm_test`, $N=129$)** | Multimodal test set (without images, Stage B Oracle) | `relation_textual_corr` | Oracle | Blind Text (balanced SGD) | `raw` | 0.6051 *(0.605)* | 0.6124 *(0.612)* | 0.6112 *(0.611)* |
| **Multimodal (`mm_test`, $N=129$)** | **Multimodal test set with images and correlations (Dual-Encoder)** | **`relation_textual_corr`** | **Oracle** | **Dual-Encoder (UmBERTo + ViT-224)** | **`raw`** | **0.6130 *(0.613)*** | **0.6202 *(0.620)*** | **0.6232 *(0.623)*** |

### 3.3 Paper Table IX: Semiotic Relation Per-Class F1 Breakdown (`mm_test`, $N=129$)

| Model Setting | Checkpoint | Prompting | Features | `Additiva` ($N=68$) F1 | `Parallela` ($N=23$) F1 | `Divergente` ($N=38$) F1 | Overall Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dual-Encoder (Stage A Raw, No corr.)** | `relation_textual_no_corr` | None (Raw) | `raw` | **0.6822 *(0.682)*** | 0.5385 *(0.538)* | **0.6234 *(0.623)*** | **0.6147 *(0.615)*** |
| **Dual-Encoder (Stage B Oracle, With corr.)** | `relation_textual_corr` | Oracle | `raw` | 0.6406 *(0.641)* | **0.5806 *(0.581)*** | 0.6176 *(0.618)* | 0.6130 *(0.613)* |

### 3.4 Key Findings for Paper Discussion (Task 3)
- **Visual Modality Impact (`Stage A Raw`)**: Because semiotic relation is inherently defined by the interaction between image and text, unimodal raw text (`Blind Text, Stage A Raw`) achieves only $0.4965$ Macro F1 ($58.91\%$ Acc). Concatenating frozen ViT-224 visual embeddings (`Dual-Encoder, Stage A Raw`) yields an **$+11.82\%$ Macro F1 leap** to **$0.6147$** ($63.57\%$ Acc, $0.6392$ Weighted F1).
- **Pragmatic Correlations as Cross-Modal Proxies (`Stage B Oracle`)**: When visual features are absent (`Blind Text, Stage B Oracle`), injecting affective and figurative correlation tokens (`esprime <emozioni> contiene <ironia/sarcasmo>`) lifts performance from $0.4965 \to \mathbf{0.6051}$ Macro F1 ($+10.86\%$), nearly matching the unprompted multimodal model.
- **Minority Class Enhancement (`Parallela`)**: In the Dual-Encoder setting, Stage B Oracle prompting boosts recognition of the minority `Parallela` class ($N=23$) by **$+4.21\%$ F1** ($0.5385 \to \mathbf{0.5806}$) while maintaining competitive overall Macro F1 ($0.6130$).

---

## 4. Consolidated Master Summary of All Official Paper Tables (`raw` Unified Protocol)

### 4.1 Main Benchmark Results (Tables I, IV, VII)

| Task | Partition | Configuration | Prompting | Features | Macro F1 (4-dec) | Micro F1 / Acc (4-dec) | Paper Macro F1 (3-dec) | Paper Micro F1 / Acc (3-dec) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task 1: Emotion** | `me_test` ($N=405$) | Textual test set (No correlations) | None (Raw) | — | 0.5092 | 0.5403 | 0.509 | 0.540 |
| **Task 1: Emotion** | `me_test` ($N=405$) | Textual test set (With correlations) | Oracle (no rel) | — | **0.5349** | **0.5530** | **0.535** | **0.553** |
| **Task 1: Emotion** | `mm_test` ($N=129$) | Multimodal test set (without images, No corr.) | None (Raw) | `raw` | 0.3565 | 0.4314 | 0.356 | 0.431 |
| **Task 1: Emotion** | `mm_test` ($N=129$) | Multimodal test set with images (Dual-Encoder, No corr.) | None (Raw) | `raw` | 0.4068 | 0.4290 | 0.407 | 0.429 |
| **Task 1: Emotion** | `mm_test` ($N=129$) | Multimodal test set (without images, With corr.) | Oracle (+ rel) | `raw` | 0.4343 | 0.5183 | 0.434 | 0.518 |
| **Task 1: Emotion** | `mm_test` ($N=129$) | **Multimodal test set with images and correlations (Dual-Encoder)** | **Oracle (+ rel)** | **`raw`** | **0.4967** | **0.5535** | **0.497** | **0.553** |
| **Task 2: Irony** | `me_test` ($N=405$) | Textual test set (No correlations) | None (Raw) | — | 0.4253 | 0.7481 | 0.425 | 0.748 |
| **Task 2: Irony** | `me_test` ($N=405$) | Textual test set (With correlations) | Oracle (no rel) | — | **0.4723** | **0.7654** | **0.472** | **0.765** |
| **Task 2: Irony** | `mm_test` ($N=129$) | Multimodal test set (without images, No corr.) | None (Raw) | `raw` | 0.4521 | 0.7984 | 0.452 | 0.798 |
| **Task 2: Irony** | `mm_test` ($N=129$) | **Multimodal test set with images (Dual-Encoder, No corr.)** | **None (Raw)** | **`raw`** | **0.5575** | **0.8217** | **0.558** | **0.822** |
| **Task 2: Irony** | `mm_test` ($N=129$) | Multimodal test set (without images, With corr.) | Oracle (no rel) | `raw` | 0.5528 | 0.7907 | 0.553 | 0.791 |
| **Task 2: Irony** | `mm_test` ($N=129$) | **Multimodal test set with images and correlations (Dual-Encoder)** | **Oracle (no rel)** | **`raw`** | **0.5491** | **0.7984** | **0.549** | **0.798** |
| **Task 3: Relation** | `mm_test` ($N=129$) | Multimodal test set (without images, Stage A Raw) | None (Raw) | `raw` | 0.4965 | 0.5891 | 0.496 | 0.589 |
| **Task 3: Relation** | `mm_test` ($N=129$) | **Multimodal test set with images (Dual-Encoder, Stage A Raw)** | **None (Raw)** | **`raw`** | **0.6147** | **0.6357** | **0.615** | **0.636** |
| **Task 3: Relation** | `mm_test` ($N=129$) | Multimodal test set (without images, Stage B Oracle) | Oracle | `raw` | 0.6051 | 0.6124 | 0.605 | 0.612 |
| **Task 3: Relation** | `mm_test` ($N=129$) | **Multimodal test set with images and correlations (Dual-Encoder)** | **Oracle** | **`raw`** | **0.6130** | **0.6202** | **0.613** | **0.620** |

### 4.2 Semiotic Subsets & Per-Class Breakdowns (Tables III, VI, IX on `mm_test`, $N=129$)

| Table | Task | Dual-Encoder Configuration | Features | `ADD` ($N=68$) | `PAR` ($N=23$) | `DIV` ($N=38$) | Overall Macro F1 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Table III** *(Subset Micro F1)* | **Task 1: Emotion** | Dual-Encoder (Step 1 Base, No corr.) | `raw` | 0.4624 *(0.46)* | 0.5172 *(0.52)* | 0.3168 *(0.32)* | 0.4068 *(0.407)* |
| **Table III** *(Subset Micro F1)* | **Task 1: Emotion** | **Dual-Encoder (Step 2 Oracle, With corr.)** | **`raw`** | **0.5614 *(0.56)*** | **0.6296 *(0.63)*** | **0.4946 *(0.49)*** | **0.4967 *(0.497)*** |
| **Table VI** *(Subset Acc / Micro F1)* | **Task 2: Irony** | **Dual-Encoder (Step 1 Base, No corr.)** | **`raw`** | **0.8824 *(0.88)*** | **0.9565 *(0.96)*** | **0.6316 *(0.63)*** | **0.5575 *(0.558)*** |
| **Table VI** *(Subset Acc / Micro F1)* | **Task 2: Irony** | **Dual-Encoder (Step 2 Oracle, With corr.)** | **`raw`** | **0.8824 *(0.88)*** | 0.9130 *(0.91)* | 0.5789 *(0.58)* | 0.5491 *(0.549)* |
| **Table IX** *(Per-Class F1)* | **Task 3: Relation** | **Dual-Encoder (Stage A Raw, No corr.)** | **`raw`** | **0.6822 *(0.682)*** | 0.5385 *(0.538)* | **0.6234 *(0.623)*** | **0.6147 *(0.615)*** |
| **Table IX** *(Per-Class F1)* | **Task 3: Relation** | **Dual-Encoder (Stage B Oracle, With corr.)** | **`raw`** | 0.6406 *(0.641)* | **0.5806 *(0.581)*** | 0.6176 *(0.618)* | 0.6130 *(0.613)* |

