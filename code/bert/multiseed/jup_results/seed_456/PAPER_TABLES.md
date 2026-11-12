# MultiEMiT: Streamlined Publication Tables Summary (`raw` Unified Protocol)

> For the exhaustive case-by-case breakdown including `std` vs `raw`, `+rel` vs `no rel`, per-label `classification_report`, and confusion matrices, see `ALL_EXPERIMENTS_COMPLETE_REPORT.md`.

---

## 1. Baseline Discriminative Tables (`raw` Feature Protocol — Tables I, IV, VII)

| Task              | Section                              | Configuration                                                                                    | Prompting                            | Scaling   |   Macro F1 |   Micro F1 / Acc |
|:------------------|:-------------------------------------|:-------------------------------------------------------------------------------------------------|:-------------------------------------|:----------|-----------:|-----------------:|
| Emotion Detection | Textual Test Set (me_test, N=405)    | Textual test set (Step 1 Base (Raw))                                                             | Raw                                  | —         |      0.499 |            0.522 |
| Emotion Detection | Textual Test Set (me_test, N=405)    | Textual test set (Step 2 Raw (Ablation))                                                         | Raw                                  | —         |      0.505 |            0.523 |
| Emotion Detection | Textual Test Set (me_test, N=405)    | Textual test set (Step 2 Oracle (Train: no rel))                                                 | Oracle (Train: no rel, Eval: no rel) | —         |      0.521 |            0.546 |
| Emotion Detection | Textual Test Set (me_test, N=405)    | Textual test set (Step 2 Oracle (Train: +rel))                                                   | Oracle (Train: +rel, Eval: no rel)   | —         |      0.528 |            0.544 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 1 Base (No corr), raw)                                 | None (Raw)                           | raw       |      0.326 |            0.387 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 1 Base (No corr), raw)                       | None (Raw)                           | raw       |      0.408 |            0.424 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Raw (No corr), raw)                                  | None (Raw)                           | raw       |      0.441 |            0.523 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Raw (No corr), raw)                        | None (Raw)                           | raw       |      0.391 |            0.440 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: no rel, Eval: no rel), raw)           | Oracle (no rel)                      | raw       |      0.422 |            0.512 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: no rel, Eval: no rel), raw) | Oracle (no rel)                      | raw       |      0.484 |            0.529 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: no rel, Eval: +rel), raw)             | Oracle (+ rel)                       | raw       |      0.409 |            0.479 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: no rel, Eval: +rel), raw)   | Oracle (+ rel)                       | raw       |      0.444 |            0.538 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: +rel, Eval: no rel), raw)             | Oracle (no rel)                      | raw       |      0.444 |            0.550 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: +rel, Eval: no rel), raw)   | Oracle (no rel)                      | raw       |      0.399 |            0.500 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: +rel, Eval: +rel), raw)               | Oracle (+ rel)                       | raw       |      0.421 |            0.503 |
| Emotion Detection | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: +rel, Eval: +rel), raw)     | Oracle (+ rel)                       | raw       |      0.431 |            0.533 |
| Irony Detection   | Textual Test Set (me_test, N=405)    | Textual test set (Step 1 Base (Raw))                                                             | Raw                                  | —         |      0.481 |            0.738 |
| Irony Detection   | Textual Test Set (me_test, N=405)    | Textual test set (Step 2 Raw (Ablation))                                                         | Raw                                  | —         |      0.540 |            0.743 |
| Irony Detection   | Textual Test Set (me_test, N=405)    | Textual test set (Step 2 Oracle (Train: no rel))                                                 | Oracle (Train: no rel, Eval: no rel) | —         |      0.542 |            0.790 |
| Irony Detection   | Textual Test Set (me_test, N=405)    | Textual test set (Step 2 Oracle (Train: +rel))                                                   | Oracle (Train: +rel, Eval: no rel)   | —         |      0.533 |            0.790 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 1 Base (No corr), raw)                                 | None (Raw)                           | raw       |      0.400 |            0.721 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 1 Base (No corr), raw)                       | None (Raw)                           | raw       |      0.467 |            0.814 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Raw (No corr), raw)                                  | None (Raw)                           | raw       |      0.402 |            0.721 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Raw (No corr), raw)                        | None (Raw)                           | raw       |      0.556 |            0.837 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: no rel, Eval: no rel), raw)           | Oracle (no rel)                      | raw       |      0.395 |            0.729 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: no rel, Eval: no rel), raw) | Oracle (no rel)                      | raw       |      0.407 |            0.760 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: no rel, Eval: +rel), raw)             | Oracle (+ rel)                       | raw       |      0.390 |            0.721 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: no rel, Eval: +rel), raw)   | Oracle (+ rel)                       | raw       |      0.441 |            0.775 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: +rel, Eval: no rel), raw)             | Oracle (no rel)                      | raw       |      0.449 |            0.744 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: +rel, Eval: no rel), raw)   | Oracle (no rel)                      | raw       |      0.416 |            0.767 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Step 2 Oracle (Train: +rel, Eval: +rel), raw)               | Oracle (+ rel)                       | raw       |      0.394 |            0.752 |
| Irony Detection   | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Step 2 Oracle (Train: +rel, Eval: +rel), raw)     | Oracle (+ rel)                       | raw       |      0.432 |            0.791 |
| Semiotic Relation | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Stage A Raw (No corr), raw)                                 | None (Raw)                           | raw       |      0.537 |            0.620 |
| Semiotic Relation | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Stage A Raw (No corr), raw)                       | None (Raw)                           | raw       |      0.578 |            0.620 |
| Semiotic Relation | Multimodal Test Set (mm_test, N=129) | Multimodal test set (without images, Stage B Oracle (With corr), raw)                            | Oracle                               | raw       |      0.634 |            0.667 |
| Semiotic Relation | Multimodal Test Set (mm_test, N=129) | Multimodal test set with images (Dual-Encoder, Stage B Oracle (With corr), raw)                  | Oracle                               | raw       |      0.609 |            0.659 |

---

## 2. Semiotic Subsets Micro F1 / Accuracy (`raw` Dual-Encoder — Tables III & VI)

| Task                                  | Case   | Model Setting                                                                       | Prompting       | Scaling   |   ADD (N=68) |   PAR (N=23) |   DIV (N=38) |   Overall Macro F1 |
|:--------------------------------------|:-------|:------------------------------------------------------------------------------------|:----------------|:----------|-------------:|-------------:|-------------:|-------------------:|
| Emotion Detection (Micro F1)          | Case 1 | Dual-Encoder (FEEL-IT + ViT-224) (Step 1 Base (No corr), raw)                       | None (Raw)      | raw       |        0.444 |        0.582 |        0.303 |              0.408 |
| Emotion Detection (Micro F1)          | Case 2 | Dual-Encoder (FEEL-IT + ViT-224) (Step 2 Raw (No corr), raw)                        | None (Raw)      | raw       |        0.518 |        0.456 |        0.295 |              0.391 |
| Emotion Detection (Micro F1)          | Case 3 | Dual-Encoder (FEEL-IT + ViT-224) (Step 2 Oracle (Train: no rel, Eval: no rel), raw) | Oracle (no rel) | raw       |        0.573 |        0.517 |        0.454 |              0.484 |
| Emotion Detection (Micro F1)          | Case 4 | Dual-Encoder (FEEL-IT + ViT-224) (Step 2 Oracle (Train: no rel, Eval: +rel), raw)   | Oracle (+ rel)  | raw       |        0.570 |        0.615 |        0.442 |              0.444 |
| Emotion Detection (Micro F1)          | Case 5 | Dual-Encoder (FEEL-IT + ViT-224) (Step 2 Oracle (Train: +rel, Eval: no rel), raw)   | Oracle (no rel) | raw       |        0.549 |        0.528 |        0.404 |              0.399 |
| Emotion Detection (Micro F1)          | Case 6 | Dual-Encoder (FEEL-IT + ViT-224) (Step 2 Oracle (Train: +rel, Eval: +rel), raw)     | Oracle (+ rel)  | raw       |        0.567 |        0.542 |        0.472 |              0.431 |
| Irony Detection (Accuracy / Micro F1) | Case 1 | Dual-Encoder (UmBERTo + ViT-224, raw) (Step 1 Base (No corr))                       | None (Raw)      | raw       |        0.868 |        0.957 |        0.632 |              0.467 |
| Irony Detection (Accuracy / Micro F1) | Case 2 | Dual-Encoder (UmBERTo + ViT-224, raw) (Step 2 Raw (No corr))                        | None (Raw)      | raw       |        0.882 |        0.957 |        0.684 |              0.556 |
| Irony Detection (Accuracy / Micro F1) | Case 3 | Dual-Encoder (UmBERTo + ViT-224, raw) (Step 2 Oracle (Train: no rel, Eval: no rel)) | Oracle (no rel) | raw       |        0.853 |        0.913 |        0.500 |              0.407 |
| Irony Detection (Accuracy / Micro F1) | Case 4 | Dual-Encoder (UmBERTo + ViT-224, raw) (Step 2 Oracle (Train: no rel, Eval: +rel))   | Oracle (+ rel)  | raw       |        0.824 |        0.913 |        0.605 |              0.441 |
| Irony Detection (Accuracy / Micro F1) | Case 5 | Dual-Encoder (UmBERTo + ViT-224, raw) (Step 2 Oracle (Train: +rel, Eval: no rel))   | Oracle (no rel) | raw       |        0.853 |        0.913 |        0.526 |              0.416 |
| Irony Detection (Accuracy / Micro F1) | Case 6 | Dual-Encoder (UmBERTo + ViT-224, raw) (Step 2 Oracle (Train: +rel, Eval: +rel))     | Oracle (+ rel)  | raw       |        0.882 |        0.957 |        0.526 |              0.432 |

---

## 3. The Unconfounded Textual Ablation Matrix (`me_test`, $N=405$)

| Task                               | Step 1 Base (Raw)           | Step 2 Raw Ablation         | Step 2 Oracle (Train: no rel)   | Step 2 Oracle (Train: +rel)   | Delta (Oracle no-rel vs Step 2 Raw)   |
|:-----------------------------------|:----------------------------|:----------------------------|:--------------------------------|:------------------------------|:--------------------------------------|
| Emotion Detection (me_test, N=405) | 0.5221 Micro / 0.4987 Macro | 0.5229 Micro / 0.5047 Macro | 0.5464 Micro / 0.5205 Macro     | 0.5435 Micro / 0.5284 Macro   | +0.0235 Micro / +0.0158 Macro         |
| Irony Detection (me_test, N=405)   | 0.7383 Acc / 0.4805 Macro   | 0.7432 Acc / 0.5404 Macro   | 0.7901 Acc / 0.5418 Macro       | 0.7901 Acc / 0.5329 Macro     | +0.0469 Acc / +0.0014 Macro           |

---

## 4. Semiotic Relation Target Classes Breakdown (`raw` — Table IX)

|   Seed | Case   | Regime                     | Checkpoint               | Prompting   | Model                                 | Scaling   | Model Setting                                                      |   ADD (n=68) |   PAR (n=23) |   DIV (n=38) |   Macro F1 |   Micro F1 (= Acc.) |
|-------:|:-------|:---------------------------|:-------------------------|:------------|:--------------------------------------|:----------|:-------------------------------------------------------------------|-------------:|-------------:|-------------:|-----------:|--------------------:|
|    456 | Case 1 | Stage A Raw (No corr)      | relation_textual_no_corr | None (Raw)  | Blind Text (SGD, raw)                 | raw       | Blind Text (SGD, raw) (Stage A Raw (No corr))                      |        0.704 |        0.343 |        0.562 |      0.537 |               0.620 |
|    456 | Case 1 | Stage A Raw (No corr)      | relation_textual_no_corr | None (Raw)  | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       | Dual-Encoder (UmBERTo + ViT-224, raw) (Stage A Raw (No corr))      |        0.700 |        0.478 |        0.556 |      0.578 |               0.620 |
|    456 | Case 2 | Stage B Oracle (With corr) | relation_textual_corr    | Oracle      | Blind Text (SGD, raw)                 | raw       | Blind Text (SGD, raw) (Stage B Oracle (With corr))                 |        0.712 |        0.533 |        0.657 |      0.634 |               0.667 |
|    456 | Case 2 | Stage B Oracle (With corr) | relation_textual_corr    | Oracle      | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       | Dual-Encoder (UmBERTo + ViT-224, raw) (Stage B Oracle (With corr)) |        0.725 |        0.465 |        0.636 |      0.609 |               0.659 |
