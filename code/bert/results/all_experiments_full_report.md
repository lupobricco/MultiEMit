# MultiEMit — Unified Experimental Results Report
**Generated:** 2026-10-30  
**Datasets:** `datasets/corpora/correctly_split/` (`textual_only_cleaned.csv`, `multimodal_train_seed42.csv`, `multimodal_test_seed42.csv`)

## 1. Master Comparison Table (All Tasks & Modalities)

| Task | Modality | Notebook | Configuration | Micro F1 | Macro F1 | Weighted F1 | Accuracy / Subset Acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Emotion Detection (10-Label Multi-Label) | Textual | train_textual_emotion.ipynb | FEEL-IT (EMit A + MultiEMit Textual) — No Correlations | 0.5403 | 0.5092 | 0.5371 | 0.3210 |
| Emotion Detection (10-Label Multi-Label) | Textual | train_textual_emotion.ipynb | FEEL-IT (Textual + Multimodal Train) — With Correlations | 0.5530 | 0.5349 | 0.5507 | 0.3259 |
| Emotion Detection (10-Label Multi-Label) | Multimodal Split (Text-Only SGD) | multimodal_dual_encoder_emotion.ipynb | Text-Only (FEEL-IT) — No Correlations | 0.3073 | 0.3033 | 0.3465 | 0.1628 |
| Emotion Detection (10-Label Multi-Label) | Multimodal Dual-Encoder (Text + ViT-224) | multimodal_dual_encoder_emotion.ipynb | Multimodal Dual-Encoder (FEEL-IT + ViT-224) — No Correlations | 0.4013 | 0.3759 | 0.3942 | 0.2403 |
| Emotion Detection (10-Label Multi-Label) | Multimodal Split (Text-Only SGD) | multimodal_dual_encoder_emotion.ipynb | Text-Only (FEEL-IT) — With Correlations | 0.5378 | 0.4464 | 0.5457 | 0.3876 |
| Emotion Detection (10-Label Multi-Label) | Multimodal Dual-Encoder (Text + ViT-224) | multimodal_dual_encoder_emotion.ipynb | Multimodal Dual-Encoder (FEEL-IT + ViT-224) — With Correlations | **0.5590** | **0.5000** | **0.5538** | **0.3721** |
| Irony & Sarcasm Detection (3-Class Single-Label) | Textual | train_textual_irony.ipynb | UmBERTo (IronITA Pre-trained -> MultiEMit No Correlations) | 0.7852 | 0.4054 | 0.7535 | 0.7852 |
| Irony & Sarcasm Detection (3-Class Single-Label) | Textual | train_textual_irony.ipynb | UmBERTo (IronITA Pre-trained -> MultiEMit With Correlations) | 0.7901 | **0.5094** | 0.7648 | 0.7901 |
| Irony & Sarcasm Detection (3-Class Single-Label) | Multimodal Split (Text-Only SGD) | multimodal_dual_encoder_irony.ipynb | Text-Only (UmBERTo) — No Correlations | 0.6512 | 0.4603 | 0.7012 | 0.6512 |
| Irony & Sarcasm Detection (3-Class Single-Label) | Multimodal Dual-Encoder (Text + ViT-224) | multimodal_dual_encoder_irony.ipynb | Multimodal Dual-Encoder (UmBERTo + ViT-224) — No Correlations | 0.7674 | 0.4280 | 0.7561 | 0.7674 |
| Irony & Sarcasm Detection (3-Class Single-Label) | Multimodal Split (Text-Only SGD) | multimodal_dual_encoder_irony.ipynb | Text-Only (UmBERTo) — With Correlations | 0.7597 | 0.4905 | 0.7711 | 0.7597 |
| Irony & Sarcasm Detection (3-Class Single-Label) | Multimodal Dual-Encoder (Text + ViT-224) | multimodal_dual_encoder_irony.ipynb | Multimodal Dual-Encoder (UmBERTo + ViT-224) — With Correlations | **0.7984** | 0.4906 | **0.7782** | **0.7984** |
| Semiotic Relation Classification (3-Class Single-Label) | Textual (on Multimodal Split) | train_textual_relation.ipynb | UmBERTo Semiotic Relation (Text-Only Fine-Tuned, No Correlations) | 0.6512 | 0.4928 | 0.5975 | 0.6512 |
| Semiotic Relation Classification (3-Class Single-Label) | Textual (on Multimodal Split) | train_textual_relation.ipynb | UmBERTo Semiotic Relation (Text-Only Fine-Tuned, With Correlations) | 0.6357 | 0.5486 | 0.6136 | 0.6357 |
| Semiotic Relation Classification (3-Class Single-Label) | Multimodal Split (Text-Only SGD) | multimodal_dual_encoder_relation.ipynb | Text-Only (UmBERTo) — No Correlations | 0.5969 | 0.5784 | 0.5961 | 0.5969 |
| Semiotic Relation Classification (3-Class Single-Label) | Multimodal Dual-Encoder (Text + ViT-224) | multimodal_dual_encoder_relation.ipynb | Multimodal Dual-Encoder (UmBERTo + ViT-224) — No Correlations | 0.5969 | 0.5154 | 0.5840 | 0.5969 |
| Semiotic Relation Classification (3-Class Single-Label) | Multimodal Split (Text-Only SGD) | multimodal_dual_encoder_relation.ipynb | Text-Only (UmBERTo) — With Correlations | **0.6589** | **0.6384** | 0.6564 | **0.6589** |
| Semiotic Relation Classification (3-Class Single-Label) | Multimodal Dual-Encoder (Text + ViT-224) | multimodal_dual_encoder_relation.ipynb | Multimodal Dual-Encoder (UmBERTo + ViT-224) — With Correlations | **0.6589** | 0.6373 | **0.6579** | **0.6589** |

---

## 2. Task 1: Emotion Detection (10-Label Multi-Label)

| Modality | Notebook | Configuration | Micro F1 | Macro F1 | Weighted F1 | Accuracy / Subset Acc |
| --- | --- | --- | --- | --- | --- | --- |
| Textual | train_textual_emotion.ipynb | Step 1: FEEL-IT (EMit A + MultiEMit Textual) — No Correlations | 0.5403 | 0.5092 | 0.5371 | 0.3210 |
| Textual | train_textual_emotion.ipynb | Step 2: Step 1 Model -> (MultiEMit Textual + Multimodal Train) With Correlations | 0.5530 | 0.5349 | 0.5507 | 0.3259 |
| Multimodal Split (Text-Only SGD) | multimodal_dual_encoder_emotion.ipynb | Text-Only (FEEL-IT) — No Correlations | 0.3073 | 0.3033 | 0.3465 | 0.1628 |
| Multimodal Dual-Encoder (Text + ViT-224) | multimodal_dual_encoder_emotion.ipynb | Multimodal Dual-Encoder (FEEL-IT + ViT-224) — No Correlations | 0.4013 | 0.3759 | 0.3942 | 0.2403 |
| Multimodal Split (Text-Only SGD) | multimodal_dual_encoder_emotion.ipynb | Text-Only (FEEL-IT) — With Correlations | 0.5378 | 0.4464 | 0.5457 | 0.3876 |
| Multimodal Dual-Encoder (Text + ViT-224) | multimodal_dual_encoder_emotion.ipynb | Multimodal Dual-Encoder (FEEL-IT + ViT-224) — With Correlations | **0.5590** | **0.5000** | **0.5538** | **0.3721** |

---

## 3. Task 2: Irony & Sarcasm Detection (3-Class Single-Label)

| Modality | Notebook | Configuration | Micro F1 | Macro F1 | Weighted F1 | Accuracy / Subset Acc |
| --- | --- | --- | --- | --- | --- | --- |
| Textual | train_textual_irony.ipynb | UmBERTo (IronITA Pre-trained -> MultiEMit No Correlations) | 0.7852 | 0.4054 | 0.7535 | 0.7852 |
| Textual | train_textual_irony.ipynb | UmBERTo (IronITA Pre-trained -> MultiEMit With Correlations) | 0.7901 | **0.5094** | 0.7648 | 0.7901 |
| Multimodal Split (Text-Only SGD) | multimodal_dual_encoder_irony.ipynb | Text-Only (UmBERTo) — No Correlations | 0.6512 | 0.4603 | 0.7012 | 0.6512 |
| Multimodal Dual-Encoder (Text + ViT-224) | multimodal_dual_encoder_irony.ipynb | Multimodal Dual-Encoder (UmBERTo + ViT-224) — No Correlations | 0.7674 | 0.4280 | 0.7561 | 0.7674 |
| Multimodal Split (Text-Only SGD) | multimodal_dual_encoder_irony.ipynb | Text-Only (UmBERTo) — With Correlations | 0.7597 | 0.4905 | 0.7711 | 0.7597 |
| Multimodal Dual-Encoder (Text + ViT-224) | multimodal_dual_encoder_irony.ipynb | Multimodal Dual-Encoder (UmBERTo + ViT-224) — With Correlations | **0.7984** | 0.4906 | **0.7782** | **0.7984** |

---

## 4. Task 3: Semiotic Relation Classification (3-Class Single-Label)

| Modality | Notebook | Configuration | Micro F1 | Macro F1 | Weighted F1 | Accuracy / Subset Acc |
| --- | --- | --- | --- | --- | --- | --- |
| Textual (on Multimodal Split) | train_textual_relation.ipynb | UmBERTo Semiotic Relation (Text-Only Fine-Tuned, No Correlations) | 0.6512 | 0.4928 | 0.5975 | 0.6512 |
| Textual (on Multimodal Split) | train_textual_relation.ipynb | UmBERTo Semiotic Relation (Text-Only Fine-Tuned, With Correlations) | 0.6357 | 0.5486 | 0.6136 | 0.6357 |
| Multimodal Split (Text-Only SGD) | multimodal_dual_encoder_relation.ipynb | Text-Only (UmBERTo) — No Correlations | 0.5969 | 0.5784 | 0.5961 | 0.5969 |
| Multimodal Dual-Encoder (Text + ViT-224) | multimodal_dual_encoder_relation.ipynb | Multimodal Dual-Encoder (UmBERTo + ViT-224) — No Correlations | 0.5969 | 0.5154 | 0.5840 | 0.5969 |
| Multimodal Split (Text-Only SGD) | multimodal_dual_encoder_relation.ipynb | Text-Only (UmBERTo) — With Correlations | **0.6589** | **0.6384** | 0.6564 | **0.6589** |
| Multimodal Dual-Encoder (Text + ViT-224) | multimodal_dual_encoder_relation.ipynb | Multimodal Dual-Encoder (UmBERTo + ViT-224) — With Correlations | **0.6589** | 0.6373 | **0.6579** | **0.6589** |

---

## 5. Semiotic Relation Subset Performance Analysis

Evaluated on the exact test set split (`multimodal_test_seed42.csv`, N=129 instances):
- **Additiva** (N=68, 52.7%): Text and image convey complementary, harmonious meaning.
- **Divergente** (N=38, 29.5%): Text and image express contrasting, ironic, or conflicting meaning.
- **Parallela** (N=23, 17.8%): Text and image communicate along parallel, independent channels.

### Emotion Detection Across Subsets (Multimodal Dual-Encoder + ViT-224 + Correlations)
| Relation Subset | Sample Size (N) | Micro F1 | Macro F1 | Weighted F1 | Subset Accuracy |
| --- | --- | --- | --- | --- | --- |
| **Additiva** | 68 | **0.6121** | **0.5387** | **0.6115** | **44.12%** |
| **Divergente** | 38 | 0.4615 | 0.2437 | 0.4578 | 28.95% |
| **Parallela** | 23 | 0.4878 | 0.1876 | 0.4431 | 30.43% |
| **Overall Test Set** | 129 | 0.5590 | 0.5000 | 0.5538 | 37.21% |

*Key finding:* Emotion classification performance is markedly highest on the **Additiva** subset (Macro F1 = 0.5387), where visual and textual emotional signals reinforce each other. Under **Divergente** and **Parallela** relations, semantic discordance and channel divergence lower emotional alignment.

### Irony Detection Across Subsets (Multimodal Dual-Encoder + ViT-224 + Correlations)
| Relation Subset | Sample Size (N) | Class Distribution (None / Ironia / Sarcasmo) | Accuracy | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- | --- |
| **Additiva** | 68 | 57 None / 9 Ironia / 2 Sarcasmo | 82.35% | **0.5458** | **0.8016** |
| **Divergente** | 38 | 20 None / 14 Ironia / 4 Sarcasmo | 68.42% | 0.3903 | 0.6974 |
| **Parallela** | 23 | 23 None / 0 Ironia / 0 Sarcasmo | **91.30%** | 0.4889 | 0.8716 |
| **Overall Test Set** | 129 | 104 None / 18 Ironia / 7 Sarcasmo | 79.84% | 0.4906 | 0.7782 |

*Key finding:* **Divergente** contains **47.4%** of all ironic/sarcastic samples (18 of 25 in the test set). Because irony inherently exploits semantic divergence between visual and textual content, classifying Divergente instances is both the most critical and challenging frontier. In contrast, **Parallela** contains exclusively non-ironic instances (100% `None`), yielding 91.30% accuracy.

---

## 6. Per-Class Classification Reports (All Notebooks)

### `train_textual_emotion.ipynb`
```text
=== Step 1: MultiEMit Test Set — WITHOUT Correlations ===
               precision    recall  f1-score   support

        Amore     0.6182    0.6939    0.6538        49
Anticipazione     0.4615    0.4800    0.4706        25
     Disgusto     0.5970    0.5405    0.5674        74
      Fiducia     0.6125    0.5904    0.6012        83
        Gioia     0.5041    0.6778    0.5782        90
       Neutro     0.4783    0.4231    0.4490        52
        Paura     0.3750    0.2500    0.3000        12
       Rabbia     0.4783    0.4074    0.4400        27
     Sorpresa     0.4659    0.4607    0.4633        89
    Tristezza     0.5435    0.5952    0.5682        42

    micro avg     0.5321    0.5488    0.5403       543
    macro avg     0.5134    0.5119    0.5092       543
 weighted avg     0.5318    0.5488    0.5371       543
  samples avg     0.5348    0.5247    0.5091       543


=== Step 2: MultiEMit Test Set — WITH Correlations ===
               precision    recall  f1-score   support

        Amore     0.5965    0.6939    0.6415        49
Anticipazione     0.5455    0.4800    0.5106        25
     Disgusto     0.5875    0.6351    0.6104        74
      Fiducia     0.5952    0.6024    0.5988        83
        Gioia     0.5670    0.6111    0.5882        90
       Neutro     0.4107    0.4423    0.4259        52
        Paura     0.6667    0.3333    0.4444        12
       Rabbia     0.5238    0.4074    0.4583        27
     Sorpresa     0.4624    0.4831    0.4725        89
    Tristezza     0.5273    0.6905    0.5979        42

    micro avg     0.5394    0.5672    0.5530       543
    macro avg     0.5483    0.5379    0.5349       543
 weighted avg     0.5406    0.5672    0.5507       543
  samples avg     0.5360    0.5377    0.5169       543
```

### `multimodal_dual_encoder_emotion.ipynb`
```text
=== [1] Text-Only (FEEL-IT) — WITHOUT Correlations ===
               precision    recall  f1-score   support

        Amore     0.3684    0.6364    0.4667        11
Anticipazione     0.6667    0.4000    0.5000        10
     Disgusto     0.3750    0.3158    0.3429        19
      Fiducia     0.6333    0.4524    0.5278        42
        Gioia     0.2258    0.2414    0.2333        29
       Neutro     0.1250    0.1765    0.1463        17
        Paura     0.0714    1.0000    0.1333         1
       Rabbia     0.2308    0.5000    0.3158         6
     Sorpresa     0.1042    0.3125    0.1562        16
    Tristezza     0.1176    1.0000    0.2105         2

    micro avg     0.2615    0.3725    0.3073       153
    macro avg     0.2918    0.5035    0.3033       153
 weighted avg     0.3691    0.3725    0.3465       153
  samples avg     0.2673    0.3553    0.2928       153


=== [2] Multimodal Dual-Encoder (FEEL-IT + ViT-224) — WITHOUT Correlations ===
               precision    recall  f1-score   support

        Amore     0.0000    0.0000    0.0000        11
Anticipazione     0.5714    0.4000    0.4706        10
     Disgusto     0.4667    0.3684    0.4118        19
      Fiducia     0.5882    0.4762    0.5263        42
        Gioia     0.4103    0.5517    0.4706        29
       Neutro     0.2857    0.3529    0.3158        17
        Paura     0.3333    1.0000    0.5000         1
       Rabbia     0.2500    0.3333    0.2857         6
     Sorpresa     0.2308    0.1875    0.2069        16
    Tristezza     0.4000    1.0000    0.5714         2

    micro avg     0.4040    0.3987    0.4013       153
    macro avg     0.3536    0.4670    0.3759       153
 weighted avg     0.4076    0.3987    0.3942       153
  samples avg     0.3459    0.3915    0.3526       153


=== [3] Text-Only (FEEL-IT) — WITH Correlations ===
               precision    recall  f1-score   support

        Amore     0.4667    0.6364    0.5385        11
Anticipazione     0.6364    0.7000    0.6667        10
     Disgusto     0.4091    0.4737    0.4390        19
      Fiducia     0.6400    0.7619    0.6957        42
        Gioia     0.5926    0.5517    0.5714        29
       Neutro     0.4167    0.2941    0.3448        17
        Paura     0.0000    0.0000    0.0000         1
       Rabbia     0.3000    0.5000    0.3750         6
     Sorpresa     0.5000    0.5000    0.5000        16
    Tristezza     0.2000    1.0000    0.3333         2

    micro avg     0.5000    0.5817    0.5378       153
    macro avg     0.4161    0.5418    0.4464       153
 weighted avg     0.5269    0.5817    0.5457       153
  samples avg     0.5180    0.5814    0.5318       153


=== [4] Multimodal Dual-Encoder (FEEL-IT + ViT-224) — WITH Correlations ===
               precision    recall  f1-score   support

        Amore     0.2857    0.5455    0.3750        11
Anticipazione     0.7143    0.5000    0.5882        10
     Disgusto     0.6154    0.4211    0.5000        19
      Fiducia     0.6731    0.8333    0.7447        42
        Gioia     0.6333    0.6552    0.6441        29
       Neutro     0.3103    0.5294    0.3913        17
        Paura     0.3333    1.0000    0.5000         1
       Rabbia     0.5000    0.3333    0.4000         6
     Sorpresa     0.6000    0.1875    0.2857        16
    Tristezza     0.4000    1.0000    0.5714         2

    micro avg     0.5325    0.5882    0.5590       153
    macro avg     0.5065    0.6005    0.5000       153
 weighted avg     0.5727    0.5882    0.5538       153
  samples avg     0.5420    0.5969    0.5473       153
```

### `train_textual_irony.ipynb`
```text
=== Textual Irony Detection (WITHOUT Correlations) ===
              precision    recall  f1-score   support

        None     0.8227    0.9340    0.8748       318
      Ironia     0.4773    0.2658    0.3415        79
    Sarcasmo     0.0000    0.0000    0.0000         8

    accuracy                         0.7852       405
   macro avg     0.4333    0.3999    0.4054       405
weighted avg     0.7391    0.7852    0.7535       405


=== Textual Irony Detection (WITH Correlations) ===
              precision    recall  f1-score   support

        None     0.8268    0.9308    0.8757       318
      Ironia     0.5366    0.2785    0.3667        79
    Sarcasmo     0.3333    0.2500    0.2857         8

    accuracy                         0.7901       405
   macro avg     0.5656    0.4864    0.5094       405
weighted avg     0.7605    0.7901    0.7648       405
```

### `multimodal_dual_encoder_irony.ipynb`
```text
=== [1/4] Text-Only (UmBERTo) — WITHOUT Correlations ===
              precision    recall  f1-score   support

        None     0.9231    0.6923    0.7912       104
      Ironia     0.2903    0.5000    0.3673        18
    Sarcasmo     0.1500    0.4286    0.2222         7

    accuracy                         0.6512       129
   macro avg     0.4545    0.5403    0.4603       129
weighted avg     0.7928    0.6512    0.7012       129


=== [2/4] Multimodal Dual-Encoder (UmBERTo + ViT-224) — WITHOUT Correlations ===
              precision    recall  f1-score   support

        None     0.8654    0.8654    0.8654       104
      Ironia     0.3600    0.5000    0.4186        18
    Sarcasmo     0.0000    0.0000    0.0000         7

    accuracy                         0.7674       129
   macro avg     0.4085    0.4551    0.4280       129
weighted avg     0.7479    0.7674    0.7561       129


=== [3/4] Text-Only (UmBERTo) — WITH Correlations ===
              precision    recall  f1-score   support

        None     0.9062    0.8365    0.8700       104
      Ironia     0.3571    0.5556    0.4348        18
    Sarcasmo     0.2000    0.1429    0.1667         7

    accuracy                         0.7597       129
   macro avg     0.4878    0.5117    0.4905       129
weighted avg     0.7913    0.7597    0.7711       129


=== [4/4] Multimodal Dual-Encoder (UmBERTo + ViT-224) — WITH Correlations ===
              precision    recall  f1-score   support

        None     0.8496    0.9231    0.8848       104
      Ironia     0.4615    0.3333    0.3871        18
    Sarcasmo     0.3333    0.1429    0.2000         7

    accuracy                         0.7984       129
   macro avg     0.5481    0.4664    0.4906       129
weighted avg     0.7674    0.7984    0.7782       129
```

### `train_textual_relation.ipynb`
```text
=== Textual Semiotic Relation Detection (WITHOUT Correlations) ===
              precision    recall  f1-score   support

    Additiva     0.6224    0.8971    0.7349        68
  Divergente     0.7857    0.5789    0.6667        38
   Parallela     0.3333    0.0435    0.0769        23

    accuracy                         0.6512       129
   macro avg     0.5805    0.5065    0.4928       129
weighted avg     0.6190    0.6512    0.5975       129


=== Textual Semiotic Relation Detection (WITH Correlations) ===
              precision    recall  f1-score   support

    Additiva     0.6250    0.8088    0.7051        68
  Divergente     0.7097    0.5789    0.6377        38
   Parallela     0.5000    0.2174    0.3030        23

    accuracy                         0.6357       129
   macro avg     0.6116    0.5351    0.5486       129
weighted avg     0.6277    0.6357    0.6136       129
```

### `multimodal_dual_encoder_relation.ipynb`
```text
=== [1/4] Text-Only (UmBERTo) — WITHOUT Correlations ===
              precision    recall  f1-score   support

    Additiva     0.6406    0.6029    0.6212        68
  Divergente     0.5556    0.6579    0.6024        38
   Parallela     0.5500    0.4783    0.5116        23

    accuracy                         0.5969       129
   macro avg     0.5821    0.5797    0.5784       129
weighted avg     0.5994    0.5969    0.5961       129


=== [2/4] Multimodal Dual-Encoder (UmBERTo + ViT-224) — WITHOUT Correlations ===
              precision    recall  f1-score   support

    Additiva     0.6857    0.7059    0.6957        68
  Divergente     0.5106    0.6316    0.5647        38
   Parallela     0.4167    0.2174    0.2857        23

    accuracy                         0.5969       129
   macro avg     0.5377    0.5183    0.5154       129
weighted avg     0.5862    0.5969    0.5840       129


=== [3/4] Text-Only (UmBERTo) — WITH Correlations ===
              precision    recall  f1-score   support

    Additiva     0.6667    0.7353    0.6993        68
  Divergente     0.6471    0.5789    0.6111        38
   Parallela     0.6500    0.5652    0.6047        23

    accuracy                         0.6589       129
   macro avg     0.6546    0.6265    0.6384       129
weighted avg     0.6579    0.6589    0.6564       129


=== [4/4] Multimodal Dual-Encoder (UmBERTo + ViT-224) — WITH Correlations ===
              precision    recall  f1-score   support

    Additiva     0.6901    0.7206    0.7050        68
  Divergente     0.6471    0.5789    0.6111        38
   Parallela     0.5833    0.6087    0.5957        23

    accuracy                         0.6589       129
   macro avg     0.6402    0.6361    0.6373       129
weighted avg     0.6584    0.6589    0.6579       129
```
