# MultiEMiT: Streamlined Publication Tables Summary

> For the comprehensive technical specification, pipelines, and prompt mechanics, see `PAPER_TABLES.md`.

---

## 1. Baseline Discriminative Tables (Tables I, IV, VII)

| Task              | Section                              | Configuration                                                        |   Macro F1 |   Micro F1 / Acc |
|:------------------|:-------------------------------------|:---------------------------------------------------------------------|-----------:|-----------------:|
| Emotion Detection | Textual Test Set (me_test, N=405)    | Textual test set (No correlations)                                   |      0.506 |            0.532 |
| Emotion Detection | Textual Test Set (me_test, N=405)    | Textual test set (With correlations)                                 |      0.546 |            0.562 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, No corr.)                       |      0.344 |            0.478 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, No corr.)             |      0.427 |            0.453 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, With corr.)                     |      0.456 |            0.561 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images and correlations (Dual-Encoder)      |      0.532 |            0.571 |
| Irony Detection   | Textual Test Set (me_test, N=405)    | Textual test set (No correlations)                                   |      0.459 |            0.756 |
| Irony Detection   | Textual Test Set (me_test, N=405)    | Textual test set (With correlations)                                 |      0.552 |            0.77  |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, No corr., std)                  |      0.443 |            0.752 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, No corr., std)        |      0.514 |            0.806 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, With corr., std)                |      0.373 |            0.69  |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images and correlations (Dual-Encoder, std) |      0.428 |            0.698 |
| Semiotic Relation | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, No corr.)                       |      0.506 |            0.597 |
| Semiotic Relation | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, No corr.)             |      0.573 |            0.667 |
| Semiotic Relation | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, With corr.)                     |      0.541 |            0.643 |
| Semiotic Relation | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images and correlations (Dual-Encoder)      |      0.624 |            0.651 |

---

## 2. Semiotic Subsets Micro F1 / Accuracy (Tables III & VI)

| Task              | Model Setting                                 |   ADD |   PAR |   DIV |
|:------------------|:----------------------------------------------|------:|------:|------:|
| Emotion Detection | Dual-Encoder (Step 1 Base (No corr))          |  0.46 |  0.58 |  0.36 |
| Emotion Detection | Dual-Encoder (Step 2 Oracle (With corr))      |  0.6  |  0.55 |  0.53 |
| Irony Detection   | Dual-Encoder (Step 1 Base (No corr, std))     |  0.88 |  1    |  0.55 |
| Irony Detection   | Dual-Encoder (Step 2 Oracle (With corr, std)) |  0.74 |  0.96 |  0.47 |

---

## 3. The Unconfounded Ablation Matrix (Reviewer Defense)

| Task              | Step 1 Base (Raw)           | Step 2 Raw Ablation         | Step 2 Oracle (With Corr)   | Delta (Oracle vs Step 2 Raw)   |
|:------------------|:----------------------------|:----------------------------|:----------------------------|:-------------------------------|
| Emotion Detection | 0.5320 Micro / 0.5061 Macro | 0.5257 Micro / 0.5116 Macro | 0.5616 Micro / 0.5456 Macro | +0.0359 Micro / +0.0340 Macro  |
| Irony Detection   | 0.7556 Acc / 0.4590 Macro   | 0.7481 Acc / 0.4979 Macro   | 0.7704 Acc / 0.5516 Macro   | +0.0222 Acc / +0.0537 Macro    |

---

## 4. Semiotic Relation Target Classes Breakdown (Table IX)

| Model Setting                             |   ADD (n=68) |   PAR (n=23) |   DIV (n=38) |   Macro F1 |
|:------------------------------------------|-------------:|-------------:|-------------:|-----------:|
| Dual-Encoder (Stage A Raw (No corr))      |        0.739 |        0.323 |        0.657 |      0.573 |
| Dual-Encoder (Stage B Oracle (With corr)) |        0.701 |        0.542 |        0.63  |      0.624 |
