# MultiEMiT: Streamlined Publication Tables Summary (`raw` Unified Protocol)

> For the exhaustive case-by-case breakdown including `std` vs `raw`, `+rel` vs `no rel`, per-label `classification_report`, and confusion matrices, see `ALL_EXPERIMENTS_COMPLETE_REPORT.md`.

---

## 1. Baseline Discriminative Tables (`raw` Feature Protocol — Tables I, IV, VII)

| Task              | Section                              | Configuration                                                                                    | Prompting                            | Scaling   |   Macro F1 |   Micro F1 / Acc |
|:------------------|:-------------------------------------|:-------------------------------------------------------------------------------------------------|:-------------------------------------|:----------|-----------:|-----------------:|
| Emotion Detection | Textual Test Set (me_test, N=405)    | Textual test set (Step 1 Base (Raw))                                                             | Raw                                  | —         |      0.512 |            0.533 |
| Emotion Detection | Textual Test Set (me_test, N=405)    | Textual test set (Step 2 Raw (Ablation))                                                         | Raw                                  | —         |      0.510 |            0.535 |
| Emotion Detection | Textual Test Set (me_test, N=405)    | Textual test set (Step 2 Oracle (Train: no rel))                                                 | Oracle (Train: no rel, Eval: no rel) | —         |      0.530 |            0.557 |
| Emotion Detection | Textual Test Set (me_test, N=405)    | Textual test set (Step 2 Oracle (Train: +rel))                                                   | Oracle (Train: +rel, Eval: no rel)   | —         |      0.559 |            0.573 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 1 Base (No corr), raw)                                 | None (Raw)                           | raw       |      0.194 |            0.274 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 1 Base (No corr), raw)                       | None (Raw)                           | raw       |      0.442 |            0.435 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Raw (No corr), raw)                                  | None (Raw)                           | raw       |      0.378 |            0.449 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Raw (No corr), raw)                        | None (Raw)                           | raw       |      0.476 |            0.494 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: no rel, Eval: no rel), raw)           | Oracle (no rel)                      | raw       |      0.434 |            0.524 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: no rel, Eval: no rel), raw) | Oracle (no rel)                      | raw       |      0.522 |            0.522 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: no rel, Eval: +rel), raw)             | Oracle (+ rel)                       | raw       |      0.432 |            0.513 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: no rel, Eval: +rel), raw)   | Oracle (+ rel)                       | raw       |      0.393 |            0.492 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: +rel, Eval: no rel), raw)             | Oracle (no rel)                      | raw       |      0.446 |            0.536 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: +rel, Eval: no rel), raw)   | Oracle (no rel)                      | raw       |      0.543 |            0.552 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: +rel, Eval: +rel), raw)               | Oracle (+ rel)                       | raw       |      0.425 |            0.517 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: +rel, Eval: +rel), raw)     | Oracle (+ rel)                       | raw       |      0.474 |            0.487 |
| Irony Detection   | Textual Test Set (me_test, N=405)    | Textual test set (Step 1 Base (Raw))                                                             | Raw                                  | —         |      0.397 |            0.731 |
| Irony Detection   | Textual Test Set (me_test, N=405)    | Textual test set (Step 2 Raw (Ablation))                                                         | Raw                                  | —         |      0.449 |            0.751 |
| Irony Detection   | Textual Test Set (me_test, N=405)    | Textual test set (Step 2 Oracle (Train: no rel))                                                 | Oracle (Train: no rel, Eval: no rel) | —         |      0.542 |            0.753 |
| Irony Detection   | Textual Test Set (me_test, N=405)    | Textual test set (Step 2 Oracle (Train: +rel))                                                   | Oracle (Train: +rel, Eval: no rel)   | —         |      0.496 |            0.726 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 1 Base (No corr), raw)                                 | None (Raw)                           | raw       |      0.425 |            0.744 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 1 Base (No corr), raw)                       | None (Raw)                           | raw       |      0.505 |            0.791 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Raw (No corr), raw)                                  | None (Raw)                           | raw       |      0.478 |            0.760 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Raw (No corr), raw)                        | None (Raw)                           | raw       |      0.508 |            0.783 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: no rel, Eval: no rel), raw)           | Oracle (no rel)                      | raw       |      0.463 |            0.736 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: no rel, Eval: no rel), raw) | Oracle (no rel)                      | raw       |      0.432 |            0.752 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: no rel, Eval: +rel), raw)             | Oracle (+ rel)                       | raw       |      0.469 |            0.744 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: no rel, Eval: +rel), raw)   | Oracle (+ rel)                       | raw       |      0.419 |            0.736 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: +rel, Eval: no rel), raw)             | Oracle (no rel)                      | raw       |      0.428 |            0.798 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: +rel, Eval: no rel), raw)   | Oracle (no rel)                      | raw       |      0.444 |            0.791 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: +rel, Eval: +rel), raw)               | Oracle (+ rel)                       | raw       |      0.438 |            0.760 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: +rel, Eval: +rel), raw)     | Oracle (+ rel)                       | raw       |      0.446 |            0.791 |
| Semiotic Relation | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Stage A Raw (No corr), raw)                                 | None (Raw)                           | raw       |      0.596 |            0.636 |
| Semiotic Relation | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Stage A Raw (No corr), raw)                       | None (Raw)                           | raw       |      0.560 |            0.574 |
| Semiotic Relation | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Stage B Oracle (With corr), raw)                            | Oracle                               | raw       |      0.543 |            0.535 |
| Semiotic Relation | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Stage B Oracle (With corr), raw)                  | Oracle                               | raw       |      0.605 |            0.651 |

---

## 2. Semiotic Subsets Micro F1 / Accuracy (`raw` Dual-Encoder — Tables III & VI)

| Task                                  | Case   | Model Setting                                                                       | Prompting       | Scaling   |   ADD (N=68) |   PAR (N=23) |   DIV (N=38) |   Overall Macro F1 |
|:--------------------------------------|:-------|:------------------------------------------------------------------------------------|:----------------|:----------|-------------:|-------------:|-------------:|-------------------:|
| Emotion Detection (Micro F1)          | Case 1 | Dual-Encoder (FEEL-IT + ViT-224) (Step 1 Base (No corr), raw)                       | None (Raw)      | raw       |        0.483 |        0.393 |        0.377 |              0.442 |
| Emotion Detection (Micro F1)          | Case 2 | Dual-Encoder (FEEL-IT + ViT-224) (Step 2 Raw (No corr), raw)                        | None (Raw)      | raw       |        0.586 |        0.588 |        0.274 |              0.476 |
| Emotion Detection (Micro F1)          | Case 3 | Dual-Encoder (FEEL-IT + ViT-224) (Step 2 Oracle (Train: no rel, Eval: no rel), raw) | Oracle (no rel) | raw       |        0.565 |        0.600 |        0.400 |              0.522 |
| Emotion Detection (Micro F1)          | Case 4 | Dual-Encoder (FEEL-IT + ViT-224) (Step 2 Oracle (Train: no rel, Eval: +rel), raw)   | Oracle (+ rel)  | raw       |        0.548 |        0.588 |        0.340 |              0.393 |
| Emotion Detection (Micro F1)          | Case 5 | Dual-Encoder (FEEL-IT + ViT-224) (Step 2 Oracle (Train: +rel, Eval: no rel), raw)   | Oracle (no rel) | raw       |        0.582 |        0.618 |        0.452 |              0.543 |
| Emotion Detection (Micro F1)          | Case 6 | Dual-Encoder (FEEL-IT + ViT-224) (Step 2 Oracle (Train: +rel, Eval: +rel), raw)     | Oracle (+ rel)  | raw       |        0.506 |        0.490 |        0.449 |              0.474 |
| Irony Detection (Accuracy / Micro F1) | Case 1 | Dual-Encoder (UmBERTo + ViT-224, raw) (Step 1 Base (No corr))                       | None (Raw)      | raw       |        0.838 |        0.957 |        0.605 |              0.505 |
| Irony Detection (Accuracy / Micro F1) | Case 2 | Dual-Encoder (UmBERTo + ViT-224, raw) (Step 2 Raw (No corr))                        | None (Raw)      | raw       |        0.897 |        0.826 |        0.553 |              0.508 |
| Irony Detection (Accuracy / Micro F1) | Case 3 | Dual-Encoder (UmBERTo + ViT-224, raw) (Step 2 Oracle (Train: no rel, Eval: no rel)) | Oracle (no rel) | raw       |        0.838 |        0.913 |        0.500 |              0.432 |
| Irony Detection (Accuracy / Micro F1) | Case 4 | Dual-Encoder (UmBERTo + ViT-224, raw) (Step 2 Oracle (Train: no rel, Eval: +rel))   | Oracle (+ rel)  | raw       |        0.794 |        0.870 |        0.553 |              0.419 |
| Irony Detection (Accuracy / Micro F1) | Case 5 | Dual-Encoder (UmBERTo + ViT-224, raw) (Step 2 Oracle (Train: +rel, Eval: no rel))   | Oracle (no rel) | raw       |        0.853 |        0.913 |        0.605 |              0.444 |
| Irony Detection (Accuracy / Micro F1) | Case 6 | Dual-Encoder (UmBERTo + ViT-224, raw) (Step 2 Oracle (Train: +rel, Eval: +rel))     | Oracle (+ rel)  | raw       |        0.882 |        0.913 |        0.553 |              0.446 |

---

## 3. The Unconfounded Textual Ablation Matrix (`me_test`, $N=405$)

| Task                               | Step 1 Base (Raw)           | Step 2 Raw Ablation         | Step 2 Oracle (Train: no rel)   | Step 2 Oracle (Train: +rel)   | Delta (Oracle no-rel vs Step 2 Raw)   |
|:-----------------------------------|:----------------------------|:----------------------------|:--------------------------------|:------------------------------|:--------------------------------------|
| Emotion Detection (me_test, N=405) | 0.5326 Micro / 0.5123 Macro | 0.5352 Micro / 0.5104 Macro | 0.5572 Micro / 0.5304 Macro     | 0.5732 Micro / 0.5592 Macro   | +0.0220 Micro / +0.0200 Macro         |
| Irony Detection (me_test, N=405)   | 0.7309 Acc / 0.3970 Macro   | 0.7506 Acc / 0.4485 Macro   | 0.7531 Acc / 0.5419 Macro       | 0.7259 Acc / 0.4962 Macro     | +0.0025 Acc / +0.0934 Macro           |

---

## 4. Semiotic Relation Target Classes Breakdown (`raw` — Table IX)

|   Seed | Case   | Regime                     | Checkpoint               | Prompting   | Model                                 | Scaling   | Model Setting                                                      |   ADD (n=68) |   PAR (n=23) |   DIV (n=38) |   Macro F1 |   Micro F1 (= Acc.) |
|-------:|:-------|:---------------------------|:-------------------------|:------------|:--------------------------------------|:----------|:-------------------------------------------------------------------|-------------:|-------------:|-------------:|-----------:|--------------------:|
|     42 | Case 1 | Stage A Raw (No corr)      | relation_textual_no_corr | None (Raw)  | Blind Text (SGD, raw)                 | raw       | Blind Text (SGD, raw) (Stage A Raw (No corr))                      |        0.718 |        0.536 |        0.533 |      0.596 |               0.636 |
|     42 | Case 1 | Stage A Raw (No corr)      | relation_textual_no_corr | None (Raw)  | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       | Dual-Encoder (UmBERTo + ViT-224, raw) (Stage A Raw (No corr))      |        0.623 |        0.500 |        0.556 |      0.560 |               0.574 |
|     42 | Case 2 | Stage B Oracle (With corr) | relation_textual_corr    | Oracle      | Blind Text (SGD, raw)                 | raw       | Blind Text (SGD, raw) (Stage B Oracle (With corr))                 |        0.584 |        0.667 |        0.380 |      0.543 |               0.535 |
|     42 | Case 2 | Stage B Oracle (With corr) | relation_textual_corr    | Oracle      | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       | Dual-Encoder (UmBERTo + ViT-224, raw) (Stage B Oracle (With corr)) |        0.719 |        0.524 |        0.571 |      0.605 |               0.651 |
