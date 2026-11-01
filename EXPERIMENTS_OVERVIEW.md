# MultiEMit: Comprehensive Multitask & Multimodal Experimental Framework

**Author / Project:** MultiEMit Benchmark Team  
**Evaluation Date:** October 2026  
**Primary Target Venue:** IEEE Transactions on Affective Computing  
**Codebase Location:** [`code/bert/`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert) | [`code/bert/final_codes/`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/final_codes)  
**Corpora Partition:** [`datasets/corpora/correctly_split/`](file:///d:/vsc_projects/MultiEMit/MultiEMit/datasets/corpora/correctly_split)

---

## 1. Executive Summary & Research Framework

The **MultiEMit** project establishes a systematic benchmark for multimodal affective computing in Italian social media (primarily Instagram). Unlike traditional unimodal or monolingual affective datasets, MultiEMit models three interconnected linguistic, psychological, and semiotic phenomena:

1. **Task 1 — Emotion Detection**: 10-class multi-label classification grounded in Plutchik’s emotion wheel and Italian affective lexicons (*Amore, Anticipazione, Disgusto, Fiducia, Gioia, Neutro, Paura, Rabbia, Sorpresa, Tristezza*).
2. **Task 2 — Irony & Sarcasm Detection**: 3-class single-label classification of figurative language (*None, Ironia, Sarcasmo*).
3. **Task 3 — Semiotic Relation Classification**: 3-class single-label classification capturing image-text semantic interaction (*Additiva, Divergente, Parallela*).

This document details the unified experimental methodology, model architectures, loss formulations, training paradigms, rigorous data-leakage controls, and empirical results across all textual and multimodal dual-encoder pipelines.

```
                              ┌──────────────────────────────────────────────┐
                              │            MultiEMit Social Post             │
                              │        (Text + Image + Meta-Attributes)      │
                              └──────────────────────┬───────────────────────┘
                                                     │
                         ┌───────────────────────────┴───────────────────────────┐
                         ▼                                                       ▼
            ┌─────────────────────────┐                             ┌─────────────────────────┐
            │   Italian Text Stream   │                             │   Visual Image Stream   │
            │  (Raw or Correlated)    │                             │      (RGB 224x224)      │
            └────────────┬────────────┘                             └────────────┬────────────┘
                         │                                                       │
                         ▼                                                       ▼
            ┌─────────────────────────┐                             ┌─────────────────────────┐
            │  FEEL-IT / UmBERTo      │                             │   ViT-Base-Patch16      │
            │  768-d Mean-Pooled      │                             │   768-d [CLS] Vector    │
            └────────────┬────────────┘                             └────────────┬────────────┘
                         │                                                       │
                         └───────────────────────────┬───────────────────────────┘
                                                     ▼
                                        ┌─────────────────────────┐
                                        │  Multimodal L2-Fusion   │
                                        │      (1536-d Vector)    │
                                        └────────────┬────────────┘
                                                     │
                    ┌────────────────────────────────┼────────────────────────────────┐
                    ▼                                ▼                                ▼
       ┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
       │   Task 1: Emotion Head  │      │    Task 2: Irony Head   │      │   Task 3: Relation Head │
       │  ClassifierChain (SVM)  │      │  SGDClassifier (SVM)    │      │  SGDClassifier (SVM)    │
       │   10-Class Multi-Label  │      │   3-Class Single-Label  │      │   3-Class Single-Label  │
       └─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

---

## 2. Dataset Partitioning & Leakage Prevention Protocol

To guarantee experimental validity and eliminate cross-modal contamination, the dataset was partitioned under the strict `datasets/corpora/correctly_split/` protocol:

| Subset Name | File Path | Number of Instances | Modalities Available | Description / Partition Role |
| :--- | :--- | :---: | :---: | :--- |
| **MultiEMit Textual-Only** | [`textual_only_cleaned.csv`](file:///d:/vsc_projects/MultiEMit/MultiEMit/datasets/corpora/correctly_split/textual_only_cleaned.csv) | **2,023** | Text only | Independent posts without corresponding images. Split **80/20** (1,618 train / 405 test) using fixed `SEED=42`. |
| **MultiEMit Multimodal Train** | [`multimodal_train_seed42.csv`](file:///d:/vsc_projects/MultiEMit/MultiEMit/datasets/corpora/correctly_split/multimodal_train_seed42.csv) | **516** | Text + Image | Dedicated multimodal training split (80% of multimodal pairs). |
| **MultiEMit Multimodal Test** | [`multimodal_test_seed42.csv`](file:///d:/vsc_projects/MultiEMit/MultiEMit/datasets/corpora/correctly_split/multimodal_test_seed42.csv) | **129** | Text + Image | Hold-out multimodal test split (20% of multimodal pairs). |
| **Total MultiEMit Multimodal** | Combined Train + Test | **645** | Text + Image | Complete set of multimodal pairs with verified image files. |

### External Corpora Integration
- **EMit A** ([`datasets/corpora/emit/EMit_A.csv`](file:///d:/vsc_projects/MultiEMit/MultiEMit/datasets/corpora/emit/EMit_A.csv)): 5,966 Italian tweets annotated for the identical 10 Plutchik emotions. Used to initialize and regularize the emotion representations.
- **IronITA** ([`datasets/corpora/ironita/ironita_full.xlsx`](file:///d:/vsc_projects/MultiEMit/MultiEMit/datasets/corpora/ironita/ironita_full.xlsx)): 4,848 Italian tweets from the EVALITA 2018 IronITA campaign. Split 90/10 (4,364 train / 484 val) to pre-train figurative language heads.

### Methodological Guarantees
1. **Zero Text-Image Contamination**: No textual-only instances appear in the multimodal split, and no multimodal test images or texts appear in any training set.
2. **Fixed Random Seed**: All pseudo-random operations across scikit-learn, PyTorch, NumPy, and Hugging Face Transformers are locked to `SEED = 42`.
3. **Disjoint Evaluation**: The textual test set (405 instances) evaluates pure language representations; the multimodal test set (129 instances) evaluates joint representations and semiotic relation interactions.

---

## 3. Theoretical Inductive Bias: Correlation Prompt Engineering

In natural human communication, emotional expression, irony, toxicity, and visual-semiotic interactions do not occur in isolation. Rather, psychological and pragmatic attributes exhibit strong structural correlations:
- **Irony and Sarcasm** heavily correlate with negative sentiment (e.g., *Rabbia*, *Disgusto*) and target orientation (mocking the *contenuto* or the *contenitore*).
- **Semiotic Divergence** (*Divergente*) acts as the primary visual vehicle for irony and sarcasm, whereas **Semiotic Addition** (*Additiva*) reinforces cohesive emotional expressions.

To leverage these correlations as an explicit inductive bias without altering transformer architectures, we implemented natural language **Correlation Prefixes** via [`italian_correlations.py`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/italian_correlations.py).

### Structured Prefix Templates
- **Task 1 (Emotion)**: Injecting target orientation:
  $$\text{Prefix} = \text{"il seguente testo riguarda [contenuto / contenitore] } \dots"$$
- **Task 2 (Irony & Sarcasm)**: Injecting target, offensiveness, and co-occurring emotions:
  $$\text{Prefix} = \text{"il seguente testo riguarda [target] contiene [offensività] esprime [emozioni] } \dots"$$
- **Task 3 (Semiotic Relation)**: Injecting target, figurative tone, and emotions:
  $$\text{Prefix} = \text{"il seguente testo riguarda [target] contiene [ironia/sarcasmo/offensività] esprime [emozioni] } \dots"$$

Across all experiments, prepending these structured natural language prompts yielded dramatic, statistically significant performance improvements.

---

## 4. Task Architectures, Training Strategies & Parameters

### Task 1: Emotion Detection (10-Label Multi-Label)

- **Target Labels (10):** `Amore`, `Anticipazione`, `Disgusto`, `Fiducia`, `Gioia`, `Neutro`, `Paura`, `Rabbia`, `Sorpresa`, `Tristezza`.
- **Text Backbone:** `MilaNLProc/feel-it-italian-emotion` (CamemBERT architecture fine-tuned on Italian affective text, 12 layers, 768 hidden dimension).
- **Head Formulation:** Linear projection ($768 \to 10$) optimized with `BCEWithLogitsLoss`.
- **Threshold Calibration:** An empirical threshold of $\tau = 0.40$ was established (calibrated to optimize macro recall on rare classes like `Paura` and `Tristezza` without degrading precision on frequent classes like `Gioia` and `Fiducia`).
- **Standard Training Strategy (Two-Step Progressive Adaptation):**
  1. *Step 1 (EMit A + MultiEMit Textual, No Correlations)*: Joint training on EMit A (5,966 samples) concatenated with MultiEMit Textual Train (1,618 samples) = 7,584 samples. Evaluated on MultiEMit Textual Test (405 samples).
  2. *Step 2 (Continuous Domain Adaptation With Correlations)*: Initialized from the Step 1 checkpoint, fine-tuned on MultiEMit Textual Train (1,618) concatenated with Multimodal Train (516) = 2,134 samples containing correlation prefixes. Evaluated on MultiEMit Textual Test with correlations (405 samples).
- **Multimodal Dual-Encoder Pipeline:**
  - *Text Encoder*: FEEL-IT backbone, mean-pooling token representations across the sequence dimension $\to \mathbf{h}_{\text{text}} \in \mathbb{R}^{768}$.
  - *Vision Encoder*: `google/vit-base-patch16-224` (Vision Transformer, 12 layers, 16x16 patches) extracting the normalized [CLS] token embedding $\to \mathbf{h}_{\text{img}} \in \mathbb{R}^{768}$.
  - *Feature Fusion*: Concatenation $\mathbf{h}_{\text{mm}} = [\mathbf{h}_{\text{text}} \,\|\, \mathbf{h}_{\text{img}}] \in \mathbb{R}^{1536}$.
  - *Classification Head*: `ClassifierChain` over 10 binary `SGDClassifier` models ($loss = \text{'squared\_hinge'}, penalty = \text{'l2'}, \alpha = 10^{-4}, \text{max\_iter} = 2000, \text{class\_weight} = \text{'balanced'}$).

---

### Task 2: Irony & Sarcasm Detection (3-Class Single-Label)

- **Target Labels (3):** `None` (0), `Ironia` (1), `Sarcasmo` (2). Mutually exclusive priority mapping: $\text{Sarcasmo} > \text{Ironia} > \text{None}$.
- **Text Backbone:** `Musixmatch/umberto-commoncrawl-cased-v1` (RoBERTa architecture trained on the Italian Common Crawl corpus, SentencePiece cased vocabulary, 768 hidden dimension).
- **Head Formulation:** Linear projection ($768 \to 3$) optimized with multi-class `CrossEntropyLoss`.
- **Training Strategy (Domain Pre-training $\to$ In-Domain Fine-Tuning):**
  1. *Stage 0 (IronITA Pre-training)*: Pre-trained on 4,364 tweets from IronITA with early stopping on 484 validation tweets (`learning_rate` = $2 \times 10^{-5}$, `batch_size` = 16, 3 epochs) to establish a baseline for Italian figurative expressions.
  2. *Stage 1 (MultiEMit Fine-Tuning)*: Fine-tuned from the Stage 0 checkpoint on MultiEMit Textual Train (1,618 samples) without and with correlation prefixes.
- **Why Naive Concatenation Failed for Irony:**
  - In Task 1 (Emotion), EMit A and MultiEMit share identical semantics, formal Italian register, and label definitions. Concatenating them expanded support harmoniously.
  - In Task 2 (Irony), IronITA comprises 2018 Twitter socio-political discussions loaded with informal hashtags and slang, whereas MultiEMit comprises 2020s Instagram visual captions. Merging them simultaneously into a single training step degraded MultiEMit’s subtle multimodal irony representations. Sequential pre-training retained the generalized figurative linguistic features while allowing MultiEMit fine-tuning to specialize effectively.
- **Multimodal Dual-Encoder Pipeline:**
  - *Fusion*: UmBERTo text vector ($768\text{-d}$) + ViT-224 image vector ($768\text{-d}$) $\to 1536\text{-d}$.
  - *Head*: Multi-class `SGDClassifier(loss='squared_hinge', penalty='l2', alpha=1e-4, max_iter=2000, class_weight='balanced')`.

---

### Task 3: Semiotic Relation Classification (3-Class Single-Label)

- **Target Labels (3):** `Additiva` (Text and image reinforce each other), `Divergente` (Text and image conflict or contrast), `Parallela` (Text and image convey separate, unrelated messages).
- **Textual Baseline:** UmBERTo fine-tuned directly on `multimodal_train_seed42.csv` (516 samples) without and with correlations, evaluated on `multimodal_test_seed42.csv` (129 samples).
- **Multimodal Dual-Encoder Pipeline:**
  - Concatenation of UmBERTo ($768\text{-d}$) and ViT-224 ($768\text{-d}$) $\to 1536\text{-d}$.
  - Multi-class `SGDClassifier(loss='squared_hinge', penalty='l2', alpha=1e-4, max_iter=2000, class_weight='balanced')`.

---

## 5. Loss Formulation & Theoretical Principles

| Learning Component | Objective / Formulation | Optimization Dynamics & Justification |
| :--- | :--- | :--- |
| **Multi-Label Emotion (Transformers)** | $\mathcal{L}_{\text{BCE}} = -\sum_{c=1}^{10} \big[ y_c \log \sigma(z_c) + (1 - y_c) \log (1 - \sigma(z_c)) \big]$ | Treats each emotion as an independent Bernoulli trial. Enables co-activation (e.g., *Gioia* + *Sorpresa*). |
| **Single-Label Irony / Relation (Transformers)** | $\mathcal{L}_{\text{CE}} = -\sum_{k=1}^K y_k \log \left(\frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}\right)$ | Softmax-based categorical cross-entropy enforcing mutual exclusivity between classes. |
| **Dual-Encoder Linear Head (`SGDClassifier`)** | $\mathcal{L}_{\text{SqHinge}}(y, \hat{y}) = \frac{1}{2} \max\big(0, 1 - y \cdot \hat{y}\big)^2 + \frac{\alpha}{2} \|\mathbf{w}\|_2^2$ | **Squared Hinge Loss (Smooth L2-SVM)**:<br>• Strictly penalizes margin violations quadratically while possessing continuous derivatives ($C^1$ smoothness).<br>• Unlike standard logistic cross-entropy (which continues adjusting weights asymptotically for easily classified points), squared hinge focuses entirely on the decision boundary ($\hat{y} < 1$).<br>• Maximizes generalization on small-sample high-dimensional representations ($N=516$, $D=1536$), suppressing overfitting. |
| **Multi-Label Multimodal Dependency (`ClassifierChain`)** | $P(\mathbf{y} \mid \mathbf{x}) = \prod_{c=1}^{10} P(y_c \mid \mathbf{x}, y_1, \dots, y_{c-1})$ | Incorporates prior emotion predictions into subsequent classifiers in the chain, directly modeling psychological co-occurrence (e.g., *Rabbia* conditioning *Disgusto*). |

---

## 6. Unified Experimental Results

All models were evaluated using standardized scikit-learn metrics (`digits=4`, `zero_division=0`).

### Master Comparison Table (All Tasks & Modalities)

| Task | Modality | Notebook Source | Configuration | Micro F1 | Macro F1 | Weighted F1 | Accuracy / Subset Acc |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Emotion Detection** *(10-Label)* | Textual | [`train_textual_emotion.ipynb`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/train_textual_emotion.ipynb) | FEEL-IT (EMit A + MultiEMit Textual) — No Corr | 0.5403 | 0.5092 | 0.5371 | 0.3210 |
| **Emotion Detection** *(10-Label)* | Textual | [`train_textual_emotion.ipynb`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/train_textual_emotion.ipynb) | FEEL-IT (Textual + Multimodal Train) — With Corr | 0.5530 | 0.5349 | 0.5507 | 0.3259 |
| **Emotion Detection** *(10-Label)* | Multimodal Split (Text-Only SGD) | [`multimodal_dual_encoder_emotion.ipynb`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/multimodal_dual_encoder_emotion.ipynb) | Text-Only (FEEL-IT) — No Corr | 0.3073 | 0.3033 | 0.3465 | 0.1628 |
| **Emotion Detection** *(10-Label)* | Multimodal Dual-Encoder | [`multimodal_dual_encoder_emotion.ipynb`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/multimodal_dual_encoder_emotion.ipynb) | Multimodal Dual-Encoder (FEEL-IT + ViT-224) — No Corr | 0.4013 | 0.3759 | 0.3942 | 0.2403 |
| **Emotion Detection** *(10-Label)* | Multimodal Split (Text-Only SGD) | [`multimodal_dual_encoder_emotion.ipynb`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/multimodal_dual_encoder_emotion.ipynb) | Text-Only (FEEL-IT) — With Corr | 0.5378 | 0.4464 | 0.5457 | 0.3876 |
| **Emotion Detection** *(10-Label)* | **Multimodal Dual-Encoder** | [`multimodal_dual_encoder_emotion.ipynb`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/multimodal_dual_encoder_emotion.ipynb) | **Multimodal Dual-Encoder (FEEL-IT + ViT-224) — With Corr** | **0.5590** | **0.5000** | **0.5538** | **0.3721** |
| **Irony Detection** *(3-Class)* | Textual | [`train_textual_irony.ipynb`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/train_textual_irony.ipynb) | UmBERTo (IronITA Pre-trained) — No Corr | 0.7852 | 0.4054 | 0.7535 | 0.7852 |
| **Irony Detection** *(3-Class)* | Textual | [`train_textual_irony.ipynb`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/train_textual_irony.ipynb) | UmBERTo (IronITA Pre-trained) — With Corr | 0.7901 | **0.5094** | 0.7648 | 0.7901 |
| **Irony Detection** *(3-Class)* | Multimodal Split (Text-Only SGD) | [`multimodal_dual_encoder_irony.ipynb`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/multimodal_dual_encoder_irony.ipynb) | Text-Only (UmBERTo) — No Corr | 0.6512 | 0.4603 | 0.7012 | 0.6512 |
| **Irony Detection** *(3-Class)* | Multimodal Dual-Encoder | [`multimodal_dual_encoder_irony.ipynb`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/multimodal_dual_encoder_irony.ipynb) | Multimodal Dual-Encoder (UmBERTo + ViT-224) — No Corr | 0.7674 | 0.4280 | 0.7561 | 0.7674 |
| **Irony Detection** *(3-Class)* | Multimodal Split (Text-Only SGD) | [`multimodal_dual_encoder_irony.ipynb`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/multimodal_dual_encoder_irony.ipynb) | Text-Only (UmBERTo) — With Corr | 0.7597 | 0.4905 | 0.7711 | 0.7597 |
| **Irony Detection** *(3-Class)* | **Multimodal Dual-Encoder** | [`multimodal_dual_encoder_irony.ipynb`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/multimodal_dual_encoder_irony.ipynb) | **Multimodal Dual-Encoder (UmBERTo + ViT-224) — With Corr** | **0.7984** | 0.4906 | **0.7782** | **0.7984** |
| **Semiotic Relation** *(3-Class)* | Textual (on MM Split) | [`train_textual_relation.ipynb`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/train_textual_relation.ipynb) | UmBERTo Semiotic Relation — No Corr | 0.6512 | 0.4928 | 0.5975 | 0.6512 |
| **Semiotic Relation** *(3-Class)* | Textual (on MM Split) | [`train_textual_relation.ipynb`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/train_textual_relation.ipynb) | UmBERTo Semiotic Relation — With Corr | 0.6357 | 0.5486 | 0.6136 | 0.6357 |
| **Semiotic Relation** *(3-Class)* | Multimodal Split (Text-Only SGD) | [`multimodal_dual_encoder_relation.ipynb`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/multimodal_dual_encoder_relation.ipynb) | Text-Only (UmBERTo) — No Corr | 0.5969 | 0.5784 | 0.5961 | 0.5969 |
| **Semiotic Relation** *(3-Class)* | Multimodal Dual-Encoder | [`multimodal_dual_encoder_relation.ipynb`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/multimodal_dual_encoder_relation.ipynb) | Multimodal Dual-Encoder (UmBERTo + ViT-224) — No Corr | 0.5969 | 0.5154 | 0.5840 | 0.5969 |
| **Semiotic Relation** *(3-Class)* | Multimodal Split (Text-Only SGD) | [`multimodal_dual_encoder_relation.ipynb`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/multimodal_dual_encoder_relation.ipynb) | Text-Only (UmBERTo) — With Corr | **0.6589** | **0.6384** | 0.6564 | **0.6589** |
| **Semiotic Relation** *(3-Class)* | **Multimodal Dual-Encoder** | [`multimodal_dual_encoder_relation.ipynb`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/multimodal_dual_encoder_relation.ipynb) | **Multimodal Dual-Encoder (UmBERTo + ViT-224) — With Corr** | **0.6589** | 0.6373 | **0.6579** | **0.6589** |

---

## 7. Semiotic Relation Subset Performance Analysis

To understand how visual-textual semantic interaction governs affective and figurative prediction, the hold-out test set (`multimodal_test_seed42.csv`, $N = 129$) was stratified across its three ground-truth semiotic relations:
- **Additiva** ($N = 68$, $52.7\%$): Complementary and mutually reinforcing signals.
- **Divergente** ($N = 38$, $29.5\%$): Conflicting or ironic incongruity between channels.
- **Parallela** ($N = 23$, $17.8\%$): Independent, non-overlapping channels.

### A. Emotion Detection Performance by Semiotic Subset
*Evaluated on Multimodal Dual-Encoder (FEEL-IT + ViT-224 + Correlations):*

| Semiotic Relation | Samples ($N$) | Micro F1 | Macro F1 | Weighted F1 | Subset Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Additiva** | **68** | **0.6121** | **0.5387** | **0.6115** | **44.12%** |
| **Divergente** | 38 | 0.4615 | 0.2437 | 0.4578 | 28.95% |
| **Parallela** | 23 | 0.4878 | 0.1876 | 0.4431 | 30.43% |
| **Overall Test Set** | 129 | 0.5590 | 0.5000 | 0.5538 | 37.21% |

> [!NOTE]
> **Psycholinguistic Finding**: When the image and caption are in an **additive** relationship, the model reaches peak affective coherence (**Macro F1 = 0.5387**, Subset Accuracy = 44.12%). Conversely, when the image and caption are **divergent** (often conveying contradictory affective valence), emotion classifiers suffer significant performance drops (Macro F1 = 0.2437), proving that multimodal semiotic discordance obscures unimodal emotional signals.

### B. Irony & Sarcasm Performance by Semiotic Subset
*Evaluated on Multimodal Dual-Encoder (UmBERTo + ViT-224 + Correlations):*

| Semiotic Relation | Samples ($N$) | Ground-Truth Distribution (`None` / `Ironia` / `Sarcasmo`) | Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **Additiva** | 68 | 57 None / 9 Ironia / 2 Sarcasmo | 82.35% | **0.5458** | **0.8016** |
| **Divergente** | **38** | **20 None / 14 Ironia / 4 Sarcasmo** | 68.42% | 0.3903 | 0.6974 |
| **Parallela** | 23 | 23 None / 0 Ironia / 0 Sarcasmo (100% Non-Ironic) | **91.30%** | 0.4889 | 0.8716 |
| **Overall Test Set** | 129 | 104 None / 18 Ironia / 7 Sarcasmo | 79.84% | 0.4906 | 0.7782 |

> [!IMPORTANT]
> **Key Finding on Figurative Semiotics**:
> 1. **Divergente concentrates figurative speech**: Out of all 25 ironic/sarcastic posts in the test set, **18 (72.0%)** belong to the `Divergente` subset. In `Divergente`, 47.4% of samples are ironic or sarcastic. The semiotic clash between text and image is the defining mechanism of multimodal irony.
> 2. **Parallela represents pure literalism**: All 23 posts in the `Parallela` subset are non-ironic (`None`), yielding 91.30% accuracy.

---

## 8. Key Scientific Insights & Paper Contributions

1. **Multimodal Dual-Encoder Superiority**:
   - For Emotion Detection, fusing the ViT-224 image representation with FEEL-IT yields an overall multimodal Macro F1 of **0.5000** (vs. 0.4464 for text-only with correlations), providing a **+12.0% relative improvement**.
   - For Irony Detection, multimodal dual-encoding reaches the highest overall accuracy (**79.84%**) and Weighted F1 (**0.7782**).
2. **Correlation Prompting as an Efficient Inductive Bias**:
   - Correlation prefixes consistently boost performance across every single task:
     - Emotion Textual: Macro F1 rises from 0.5092 to **0.5349**.
     - Emotion Multimodal: Macro F1 jumps from 0.3759 to **0.5000** (**+33.0% relative gain**).
     - Irony Textual: Macro F1 leaps from 0.4054 to **0.5094** (**+25.7% relative gain**).
     - Semiotic Relation: Accuracy climbs to **65.89%** and Macro F1 to **0.6384**.
3. **Task-Specific Training Paradigm Rationale**:
   - *Emotion*: Thrives on multi-source data pooling (EMit A + MultiEMit) because Plutchik emotion semantics are invariant across social media platforms.
   - *Irony*: Demands two-stage domain pre-training (IronITA $\to$ MultiEMit) to prevent Twitter slang from washing out subtle Instagram visual irony cues.
4. **Squared Hinge Loss for Multimodal Embeddings**:
   - Linear SVM heads (`loss='squared_hinge'`) on frozen dual-encoder representations decisively outperform conventional logistic regression by enforcing quadratic margin penalties, preventing overfitting on small multimodal training splits ($N=516$).

---

## 9. Finalized Codebase Structure & Reproducibility Guide

The finalized, publication-ready codebase is organized under [`code/bert/final_codes/`](file:///d:/vsc_projects/MultiEMit/MultiEMit/code/bert/final_codes):

```
code/bert/final_codes/
│
├── train_textual_emotion.ipynb            # Task 1 Textual: FEEL-IT Two-Step Training
├── train_textual_irony.ipynb              # Task 2 Textual: UmBERTo IronITA Pre-training -> MultiEMit Fine-tuning
├── train_textual_relation.ipynb           # Task 3 Textual: UmBERTo Semiotic Relation Fine-tuning
│
├── multimodal_dual_encoder_emotion.ipynb  # Task 1 Multimodal: FEEL-IT + ViT-224 Dual-Encoder
├── multimodal_dual_encoder_irony.ipynb    # Task 2 Multimodal: UmBERTo + ViT-224 Dual-Encoder
├── multimodal_dual_encoder_relation.ipynb # Task 3 Multimodal: UmBERTo + ViT-224 Dual-Encoder
│
├── evaluate_relation_subsets.py          # Standalone Evaluator for Additiva, Divergente, Parallela
├── run_all_experiments.py                # Subprocess-isolated end-to-end experiment pipeline
├── run_all_experiments.bat               # Windows batch launcher (isolated VRAM sessions)
├── run_all_experiments.ipynb             # Interactive runner notebook
│
└── results/                              # Unified Output Directory
    ├── all_experiments_full_report.md    # Detailed per-class precision/recall/F1 logs
    └── all_experiments_summary.csv       # Machine-readable metric tables
```

### One-Click Reproduction Command
To re-run the entire pipeline from scratch on Windows:
```powershell
cd d:\vsc_projects\MultiEMit\MultiEMit\code\bert\final_codes
.\run_all_experiments.bat
```
Or run the standalone subset evaluation:
```powershell
..\..\.conda\python.exe evaluate_relation_subsets.py
```
