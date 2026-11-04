# MultiEMiT: Streamlined Publication Tables Summary

> For the comprehensive technical specification, pipelines, and prompt mechanics, see `PAPER_TABLES.md`.

---

## 1. Baseline Discriminative Tables (Tables I, IV, VII)

| Task              | Section                              | Configuration                                                        |   Macro F1 |   Micro F1 / Acc |
|:------------------|:-------------------------------------|:---------------------------------------------------------------------|-----------:|-----------------:|
| Emotion Detection | Textual Test Set (me_test, N=405)    | Textual test set (No correlations)                                   |      0.509 |            0.54  |
| Emotion Detection | Textual Test Set (me_test, N=405)    | Textual test set (With correlations)                                 |      0.535 |            0.553 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, No corr.)                       |      0.356 |            0.431 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, No corr.)             |      0.407 |            0.429 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, With corr.)                     |      0.434 |            0.518 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images and correlations (Dual-Encoder)      |      0.497 |            0.553 |
| Irony Detection   | Textual Test Set (me_test, N=405)    | Textual test set (No correlations)                                   |      0.425 |            0.748 |
| Irony Detection   | Textual Test Set (me_test, N=405)    | Textual test set (With correlations)                                 |      0.472 |            0.765 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, No corr., std)                  |      0.54  |            0.806 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, No corr., std)        |      0.42  |            0.783 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, With corr., std)                |      0.372 |            0.721 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images and correlations (Dual-Encoder, std) |      0.602 |            0.814 |
| Semiotic Relation | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, No corr.)                       |      0.496 |            0.589 |
| Semiotic Relation | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, No corr.)             |      0.615 |            0.636 |
| Semiotic Relation | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, With corr.)                     |      0.605 |            0.612 |
| Semiotic Relation | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images and correlations (Dual-Encoder)      |      0.613 |            0.62  |

---

## 2. Semiotic Subsets Micro F1 / Accuracy (Tables III & VI)

| Task              | Model Setting                                 |   ADD |   PAR |   DIV |
|:------------------|:----------------------------------------------|------:|------:|------:|
| Emotion Detection | Dual-Encoder (Step 1 Base (No corr))          |  0.46 |  0.52 |  0.32 |
| Emotion Detection | Dual-Encoder (Step 2 Oracle (With corr))      |  0.56 |  0.63 |  0.49 |
| Irony Detection   | Dual-Encoder (Step 1 Base (No corr, std))     |  0.87 |  0.96 |  0.53 |
| Irony Detection   | Dual-Encoder (Step 1 Base (No corr, raw))     |  0.88 |  0.96 |  0.63 |
| Irony Detection   | Dual-Encoder (Step 2 Oracle (With corr, std)) |  0.88 |  0.91 |  0.63 |
| Irony Detection   | Dual-Encoder (Step 2 Oracle (no rel, raw))    |  0.88 |  0.91 |  0.58 |
| Irony Detection   | Dual-Encoder (Step 2 Oracle (+ rel) (std))    |  0.85 |  0.96 |  0.47 |
| Irony Detection   | Dual-Encoder (Step 2 Oracle (+ rel) (raw))    |  0.79 |  0.78 |  0.53 |
| Irony Detection   | Dual-Encoder (Step 2 Raw (std))               |  0.85 |  0.91 |  0.5  |
| Irony Detection   | Dual-Encoder (Step 2 Raw (raw))               |  0.87 |  0.96 |  0.63 |

---

## 3. The Unconfounded Ablation Matrix (Reviewer Defense)

| Task              | Step 1 Base (Raw)           | Step 2 Raw Ablation         | Step 2 Oracle (With Corr)   | Delta (Oracle vs Step 2 Raw)   |
|:------------------|:----------------------------|:----------------------------|:----------------------------|:-------------------------------|
| Emotion Detection | 0.5403 Micro / 0.5092 Macro | 0.5383 Micro / 0.5235 Macro | 0.5530 Micro / 0.5349 Macro | +0.0146 Micro / +0.0114 Macro  |
| Irony Detection   | 0.7481 Acc / 0.4253 Macro   | 0.7704 Acc / 0.4755 Macro   | 0.7654 Acc / 0.4723 Macro   | +-0.0049 Acc / +-0.0033 Macro  |

---

## 4. Semiotic Relation Target Classes Breakdown (Table IX)

| Model Setting                             |   ADD (n=68) |   PAR (n=23) |   DIV (n=38) |   Macro F1 |
|:------------------------------------------|-------------:|-------------:|-------------:|-----------:|
| Dual-Encoder (Stage A Raw (No corr))      |        0.682 |        0.538 |        0.623 |      0.615 |
| Dual-Encoder (Stage B Oracle (With corr)) |        0.641 |        0.581 |        0.618 |      0.613 |
