# MultiEMiT: Streamlined Publication Tables Summary (`raw` Unified Protocol)

> For the exhaustive case-by-case breakdown including `std` vs `raw`, `+rel` vs `no rel`, per-label `classification_report`, and confusion matrices, see `ALL_EXPERIMENTS_COMPLETE_REPORT.md`.

---

## 1. Baseline Discriminative Tables (`raw` Feature Protocol — Tables I, IV, VII)

| Task              | Section                              | Configuration                                                                                    | Prompting                            | Scaling   |   Macro F1 |   Micro F1 / Acc |
|:------------------|:-------------------------------------|:-------------------------------------------------------------------------------------------------|:-------------------------------------|:----------|-----------:|-----------------:|
| Emotion Detection | Textual Test Set (me_test, N=405)    | Textual test set (Step 1 Base (Raw))                                                             | Raw                                  | —         |      0.506 |            0.532 |
| Emotion Detection | Textual Test Set (me_test, N=405)    | Textual test set (Step 2 Raw (Ablation))                                                         | Raw                                  | —         |      0.530 |            0.526 |
| Emotion Detection | Textual Test Set (me_test, N=405)    | Textual test set (Step 2 Oracle (Train: no rel))                                                 | Oracle (Train: no rel, Eval: no rel) | —         |      0.540 |            0.562 |
| Emotion Detection | Textual Test Set (me_test, N=405)    | Textual test set (Step 2 Oracle (Train: +rel))                                                   | Oracle (Train: +rel, Eval: no rel)   | —         |      0.545 |            0.554 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 1 Base (No corr), raw)                                 | None (Raw)                           | raw       |      0.310 |            0.354 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 1 Base (No corr), raw)                       | None (Raw)                           | raw       |      0.396 |            0.369 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Raw (No corr), raw)                                  | None (Raw)                           | raw       |      0.453 |            0.503 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Raw (No corr), raw)                        | None (Raw)                           | raw       |      0.430 |            0.491 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: no rel, Eval: no rel), raw)           | Oracle (no rel)                      | raw       |      0.423 |            0.459 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: no rel, Eval: no rel), raw) | Oracle (no rel)                      | raw       |      0.509 |            0.535 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: no rel, Eval: +rel), raw)             | Oracle (+ rel)                       | raw       |      0.364 |            0.414 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: no rel, Eval: +rel), raw)   | Oracle (+ rel)                       | raw       |      0.507 |            0.522 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: +rel, Eval: no rel), raw)             | Oracle (no rel)                      | raw       |      0.428 |            0.484 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: +rel, Eval: no rel), raw)   | Oracle (no rel)                      | raw       |      0.482 |            0.506 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: +rel, Eval: +rel), raw)               | Oracle (+ rel)                       | raw       |      0.470 |            0.525 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: +rel, Eval: +rel), raw)     | Oracle (+ rel)                       | raw       |      0.469 |            0.540 |
| Irony Detection   | Textual Test Set (me_test, N=405)    | Textual test set (Step 1 Base (Raw))                                                             | Raw                                  | —         |      0.469 |            0.743 |
| Irony Detection   | Textual Test Set (me_test, N=405)    | Textual test set (Step 2 Raw (Ablation))                                                         | Raw                                  | —         |      0.529 |            0.785 |
| Irony Detection   | Textual Test Set (me_test, N=405)    | Textual test set (Step 2 Oracle (Train: no rel))                                                 | Oracle (Train: no rel, Eval: no rel) | —         |      0.515 |            0.751 |
| Irony Detection   | Textual Test Set (me_test, N=405)    | Textual test set (Step 2 Oracle (Train: +rel))                                                   | Oracle (Train: +rel, Eval: no rel)   | —         |      0.501 |            0.763 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 1 Base (No corr), raw)                                 | None (Raw)                           | raw       |      0.438 |            0.775 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 1 Base (No corr), raw)                       | None (Raw)                           | raw       |      0.533 |            0.814 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Raw (No corr), raw)                                  | None (Raw)                           | raw       |      0.479 |            0.775 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Raw (No corr), raw)                        | None (Raw)                           | raw       |      0.520 |            0.806 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: no rel, Eval: no rel), raw)           | Oracle (no rel)                      | raw       |      0.464 |            0.721 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: no rel, Eval: no rel), raw) | Oracle (no rel)                      | raw       |      0.527 |            0.767 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: no rel, Eval: +rel), raw)             | Oracle (+ rel)                       | raw       |      0.505 |            0.713 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: no rel, Eval: +rel), raw)   | Oracle (+ rel)                       | raw       |      0.537 |            0.806 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: +rel, Eval: no rel), raw)             | Oracle (no rel)                      | raw       |      0.385 |            0.744 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: +rel, Eval: no rel), raw)   | Oracle (no rel)                      | raw       |      0.363 |            0.721 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: +rel, Eval: +rel), raw)               | Oracle (+ rel)                       | raw       |      0.442 |            0.767 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: +rel, Eval: +rel), raw)     | Oracle (+ rel)                       | raw       |      0.419 |            0.775 |
| Semiotic Relation | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Stage A Raw (No corr), raw)                                 | None (Raw)                           | raw       |      0.582 |            0.605 |
| Semiotic Relation | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Stage A Raw (No corr), raw)                       | None (Raw)                           | raw       |      0.519 |            0.589 |
| Semiotic Relation | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Stage B Oracle (With corr), raw)                            | Oracle                               | raw       |      0.632 |            0.682 |
| Semiotic Relation | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Stage B Oracle (With corr), raw)                  | Oracle                               | raw       |      0.542 |            0.589 |

---

## 2. Semiotic Subsets Micro F1 / Accuracy (`raw` Dual-Encoder — Tables III & VI)

| Task                                  | Case   | Model Setting                                                                       | Prompting       | Scaling   |   ADD (N=68) |   PAR (N=23) |   DIV (N=38) |   Overall Macro F1 |
|:--------------------------------------|:-------|:------------------------------------------------------------------------------------|:----------------|:----------|-------------:|-------------:|-------------:|-------------------:|
| Emotion Detection (Micro F1)          | Case 1 | Dual-Encoder (FEEL-IT + ViT-224) (Step 1 Base (No corr), raw)                       | None (Raw)      | raw       |        0.441 |        0.344 |        0.250 |              0.396 |
| Emotion Detection (Micro F1)          | Case 2 | Dual-Encoder (FEEL-IT + ViT-224) (Step 2 Raw (No corr), raw)                        | None (Raw)      | raw       |        0.578 |        0.508 |        0.327 |              0.430 |
| Emotion Detection (Micro F1)          | Case 3 | Dual-Encoder (FEEL-IT + ViT-224) (Step 2 Oracle (Train: no rel, Eval: no rel), raw) | Oracle (no rel) | raw       |        0.601 |        0.545 |        0.412 |              0.509 |
| Emotion Detection (Micro F1)          | Case 4 | Dual-Encoder (FEEL-IT + ViT-224) (Step 2 Oracle (Train: no rel, Eval: +rel), raw)   | Oracle (+ rel)  | raw       |        0.542 |        0.594 |        0.442 |              0.507 |
| Emotion Detection (Micro F1)          | Case 5 | Dual-Encoder (FEEL-IT + ViT-224) (Step 2 Oracle (Train: +rel, Eval: no rel), raw)   | Oracle (no rel) | raw       |        0.580 |        0.491 |        0.392 |              0.482 |
| Emotion Detection (Micro F1)          | Case 6 | Dual-Encoder (FEEL-IT + ViT-224) (Step 2 Oracle (Train: +rel, Eval: +rel), raw)     | Oracle (+ rel)  | raw       |        0.588 |        0.491 |        0.486 |              0.469 |
| Irony Detection (Accuracy / Micro F1) | Case 1 | Dual-Encoder (UmBERTo + ViT-224, raw) (Step 1 Base (No corr))                       | None (Raw)      | raw       |        0.882 |        0.957 |        0.605 |              0.533 |
| Irony Detection (Accuracy / Micro F1) | Case 2 | Dual-Encoder (UmBERTo + ViT-224, raw) (Step 2 Raw (No corr))                        | None (Raw)      | raw       |        0.853 |        0.913 |        0.658 |              0.520 |
| Irony Detection (Accuracy / Micro F1) | Case 3 | Dual-Encoder (UmBERTo + ViT-224, raw) (Step 2 Oracle (Train: no rel, Eval: no rel)) | Oracle (no rel) | raw       |        0.838 |        0.870 |        0.579 |              0.527 |
| Irony Detection (Accuracy / Micro F1) | Case 4 | Dual-Encoder (UmBERTo + ViT-224, raw) (Step 2 Oracle (Train: no rel, Eval: +rel))   | Oracle (+ rel)  | raw       |        0.853 |        0.957 |        0.632 |              0.537 |
| Irony Detection (Accuracy / Micro F1) | Case 5 | Dual-Encoder (UmBERTo + ViT-224, raw) (Step 2 Oracle (Train: +rel, Eval: no rel))   | Oracle (no rel) | raw       |        0.809 |        0.913 |        0.447 |              0.363 |
| Irony Detection (Accuracy / Micro F1) | Case 6 | Dual-Encoder (UmBERTo + ViT-224, raw) (Step 2 Oracle (Train: +rel, Eval: +rel))     | Oracle (+ rel)  | raw       |        0.868 |        0.913 |        0.526 |              0.419 |

---

## 3. The Unconfounded Textual Ablation Matrix (`me_test`, $N=405$)

| Task                               | Step 1 Base (Raw)           | Step 2 Raw Ablation         | Step 2 Oracle (Train: no rel)   | Step 2 Oracle (Train: +rel)   | Delta (Oracle no-rel vs Step 2 Raw)   |
|:-----------------------------------|:----------------------------|:----------------------------|:--------------------------------|:------------------------------|:--------------------------------------|
| Emotion Detection (me_test, N=405) | 0.5324 Micro / 0.5063 Macro | 0.5258 Micro / 0.5296 Macro | 0.5620 Micro / 0.5399 Macro     | 0.5541 Micro / 0.5449 Macro   | +0.0362 Micro / +0.0103 Macro         |
| Irony Detection (me_test, N=405)   | 0.7432 Acc / 0.4692 Macro   | 0.7852 Acc / 0.5292 Macro   | 0.7506 Acc / 0.5146 Macro       | 0.7630 Acc / 0.5011 Macro     | -0.0346 Acc / -0.0146 Macro           |

---

## 4. Semiotic Relation Target Classes Breakdown (`raw` — Table IX)

|   Seed | Case   | Regime                     | Checkpoint               | Prompting   | Model                                 | Scaling   | Model Setting                                                      |   ADD (n=68) |   PAR (n=23) |   DIV (n=38) |   Macro F1 |   Micro F1 (= Acc.) |
|-------:|:-------|:---------------------------|:-------------------------|:------------|:--------------------------------------|:----------|:-------------------------------------------------------------------|-------------:|-------------:|-------------:|-----------:|--------------------:|
|    123 | Case 1 | Stage A Raw (No corr)      | relation_textual_no_corr | None (Raw)  | Blind Text (SGD, raw)                 | raw       | Blind Text (SGD, raw) (Stage A Raw (No corr))                      |        0.652 |        0.510 |        0.583 |      0.582 |               0.605 |
|    123 | Case 1 | Stage A Raw (No corr)      | relation_textual_no_corr | None (Raw)  | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       | Dual-Encoder (UmBERTo + ViT-224, raw) (Stage A Raw (No corr))      |        0.685 |        0.318 |        0.554 |      0.519 |               0.589 |
|    123 | Case 2 | Stage B Oracle (With corr) | relation_textual_corr    | Oracle      | Blind Text (SGD, raw)                 | raw       | Blind Text (SGD, raw) (Stage B Oracle (With corr))                 |        0.735 |        0.514 |        0.647 |      0.632 |               0.682 |
|    123 | Case 2 | Stage B Oracle (With corr) | relation_textual_corr    | Oracle      | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       | Dual-Encoder (UmBERTo + ViT-224, raw) (Stage B Oracle (With corr)) |        0.676 |        0.465 |        0.486 |      0.542 |               0.589 |
