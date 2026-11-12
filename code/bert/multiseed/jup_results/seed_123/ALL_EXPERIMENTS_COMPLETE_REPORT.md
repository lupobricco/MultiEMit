# MultiEMiT: Complete Experimental Report (All Runs, Classification Reports & Confusion Matrices)

> **Auto-Generated Exhaustive Report**: Covers every textual and multimodal experiment across **Task 1 (Emotion Detection)**, **Task 2 (Irony & Sarcasm Detection)**, and **Task 3 (Semiotic Relation Classification)**.
> **Dimensions Evaluated**: `Step 1 Base` vs. `Step 2 Raw` vs. `Step 2 Oracle (Train: no rel)` vs. `Step 2 Oracle (Train: +rel)`; Multimodal `Eval: no rel` vs. `Eval: +rel`; Feature Scaling `raw` (no std) vs. `std` (`StandardScaler`); `Blind Text` vs. `Dual-Encoder`.

---

## 1. Task 1: Emotion Detection (10-Class Multi-Label, `FEEL-IT` + `ViT-224`)

- **Text Encoder**: `MilaNLProc/feel-it-italian-emotion` (`FEEL-IT`, 768-d `[CLS]`)
- **Vision Encoder**: `google/vit-base-patch16-224` (`ViT-Base-224`, frozen 768-d `[CLS]`)
- **Textual Head (`me_test`, $N=405$)**: Sequence classification head + `EMOTION_THRESHOLD = 0.40` with dynamic top-2 fallback (`prob[0] - prob[1] <= 0.10`)
- **Multimodal Head (`mm_test`, $N=129$)**: `ClassifierChain(SGDClassifier(loss='squared_hinge', penalty='l2', class_weight='balanced', random_state=42))`

### 1.1 Textual Emotion Benchmark (`me_test`, $N=405$)

#### Overall Metrics Summary
| Regime                        | Checkpoint                    | Prompt                               |   N Test |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:------------------------------|:------------------------------|:-------------------------------------|---------:|-----------:|-----------:|--------------:|--------------:|
| Step 1 Base (Raw)             | emotion_textual_no_corr       | Raw                                  |      405 |     0.5063 |     0.5324 |        0.5283 |        0.2914 |
| Step 2 Raw (Ablation)         | emotion_textual_step2_raw     | Raw                                  |      405 |     0.5296 |     0.5258 |        0.5254 |        0.3012 |
| Step 2 Oracle (Train: no rel) | emotion_textual_corr          | Oracle (Train: no rel, Eval: no rel) |      405 |     0.5399 |     0.5620 |        0.5589 |        0.3185 |
| Step 2 Oracle (Train: +rel)   | emotion_textual_corr_with_rel | Oracle (Train: +rel, Eval: no rel)   |      405 |     0.5449 |     0.5541 |        0.5529 |        0.3407 |

#### Per-Emotion F1 Breakdown (`me_test`)
| Regime                        |   Amore F1 |   Anticipazione F1 |   Disgusto F1 |   Fiducia F1 |   Gioia F1 |   Neutro F1 |   Paura F1 |   Rabbia F1 |   Sorpresa F1 |   Tristezza F1 |
|:------------------------------|-----------:|-------------------:|--------------:|-------------:|-----------:|------------:|-----------:|------------:|--------------:|---------------:|
| Step 1 Base (Raw)             |     0.6604 |             0.5306 |        0.5789 |       0.5696 |     0.5545 |      0.4356 |     0.2353 |      0.4615 |        0.4199 |         0.6170 |
| Step 2 Raw (Ablation)         |     0.6337 |             0.5714 |        0.5854 |       0.5521 |     0.5444 |      0.3964 |     0.5882 |      0.4400 |        0.4444 |         0.5400 |
| Step 2 Oracle (Train: no rel) |     0.6126 |             0.4898 |        0.6275 |       0.6415 |     0.6030 |      0.4950 |     0.4706 |      0.4231 |        0.4457 |         0.5905 |
| Step 2 Oracle (Train: +rel)   |     0.6226 |             0.5238 |        0.6275 |       0.6163 |     0.5814 |      0.4746 |     0.5556 |      0.4364 |        0.4497 |         0.5614 |

#### Textual Case: `Step 1 Base (Raw)` (`emotion_textual_no_corr`)
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.6140 |   0.7143 |     0.6604 |        49 |
| Anticipazione |      0.5417 |   0.5200 |     0.5306 |        25 |
| Disgusto      |      0.5641 |   0.5946 |     0.5789 |        74 |
| Fiducia       |      0.6000 |   0.5422 |     0.5696 |        83 |
| Gioia         |      0.5000 |   0.6222 |     0.5545 |        90 |
| Neutro        |      0.4490 |   0.4231 |     0.4356 |        52 |
| Paura         |      0.4000 |   0.1667 |     0.2353 |        12 |
| Rabbia        |      0.4800 |   0.4444 |     0.4615 |        27 |
| Sorpresa      |      0.4130 |   0.4270 |     0.4199 |        89 |
| Tristezza     |      0.5577 |   0.6905 |     0.6170 |        42 |
| micro avg     |      0.5202 |   0.5451 |     0.5324 |       543 |
| macro avg     |      0.5120 |   0.5145 |     0.5063 |       543 |
| weighted avg  |      0.5183 |   0.5451 |     0.5283 |       543 |
| samples avg   |      0.5259 |   0.5294 |     0.5039 |       543 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  334 |   22 |   14 |   35 |      0.6140 |   0.7143 |     0.6604 |        49 |
| Anticipazione |  369 |   11 |   12 |   13 |      0.5417 |   0.5200 |     0.5306 |        25 |
| Disgusto      |  297 |   34 |   30 |   44 |      0.5641 |   0.5946 |     0.5789 |        74 |
| Fiducia       |  292 |   30 |   38 |   45 |      0.6000 |   0.5422 |     0.5696 |        83 |
| Gioia         |  259 |   56 |   34 |   56 |      0.5000 |   0.6222 |     0.5545 |        90 |
| Neutro        |  326 |   27 |   30 |   22 |      0.4490 |   0.4231 |     0.4356 |        52 |
| Paura         |  390 |    3 |   10 |    2 |      0.4000 |   0.1667 |     0.2353 |        12 |
| Rabbia        |  365 |   13 |   15 |   12 |      0.4800 |   0.4444 |     0.4615 |        27 |
| Sorpresa      |  262 |   54 |   51 |   38 |      0.4130 |   0.4270 |     0.4199 |        89 |
| Tristezza     |  340 |   23 |   13 |   29 |      0.5577 |   0.6905 |     0.6170 |        42 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             49 |           35 |                    1 |               2 |             16 |           18 |             1 |            0 |             1 |               1 |                2 |           0 |
| Anticipazione  |             25 |            0 |                   13 |               3 |              5 |            5 |             1 |            0 |             1 |               3 |                2 |           0 |
| Disgusto       |             74 |            1 |                    3 |              44 |              4 |            4 |             6 |            2 |            18 |              17 |               21 |           0 |
| Fiducia        |             83 |           21 |                    5 |               6 |             45 |           18 |             9 |            2 |             2 |               9 |                8 |           0 |
| Gioia          |             90 |           14 |                    3 |               1 |             20 |           56 |             4 |            0 |             0 |              18 |                5 |           0 |
| Neutro         |             52 |            3 |                    2 |               6 |              5 |            9 |            22 |            0 |             0 |              12 |                1 |           0 |
| Paura          |             12 |            0 |                    1 |               3 |              1 |            1 |             2 |            2 |             0 |               7 |                5 |           0 |
| Rabbia         |             27 |            0 |                    1 |              20 |              1 |            0 |             1 |            0 |            12 |               5 |                6 |           0 |
| Sorpresa       |             89 |            3 |                    1 |              25 |              8 |           22 |             8 |            5 |            12 |              38 |               25 |           0 |
| Tristezza      |             42 |            0 |                    2 |              16 |              4 |            2 |             0 |            2 |             7 |              14 |               29 |           0 |

#### Textual Case: `Step 2 Raw (Ablation)` (`emotion_textual_step2_raw`)
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.6154 |   0.6531 |     0.6337 |        49 |
| Anticipazione |      0.5161 |   0.6400 |     0.5714 |        25 |
| Disgusto      |      0.5333 |   0.6486 |     0.5854 |        74 |
| Fiducia       |      0.5625 |   0.5422 |     0.5521 |        83 |
| Gioia         |      0.5444 |   0.5444 |     0.5444 |        90 |
| Neutro        |      0.3729 |   0.4231 |     0.3964 |        52 |
| Paura         |      1.0000 |   0.4167 |     0.5882 |        12 |
| Rabbia        |      0.4783 |   0.4074 |     0.4400 |        27 |
| Sorpresa      |      0.4396 |   0.4494 |     0.4444 |        89 |
| Tristezza     |      0.4655 |   0.6429 |     0.5400 |        42 |
| micro avg     |      0.5095 |   0.5433 |     0.5258 |       543 |
| macro avg     |      0.5528 |   0.5368 |     0.5296 |       543 |
| weighted avg  |      0.5178 |   0.5433 |     0.5254 |       543 |
| samples avg   |      0.5107 |   0.5214 |     0.4957 |       543 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  336 |   20 |   17 |   32 |      0.6154 |   0.6531 |     0.6337 |        49 |
| Anticipazione |  365 |   15 |    9 |   16 |      0.5161 |   0.6400 |     0.5714 |        25 |
| Disgusto      |  289 |   42 |   26 |   48 |      0.5333 |   0.6486 |     0.5854 |        74 |
| Fiducia       |  287 |   35 |   38 |   45 |      0.5625 |   0.5422 |     0.5521 |        83 |
| Gioia         |  274 |   41 |   41 |   49 |      0.5444 |   0.5444 |     0.5444 |        90 |
| Neutro        |  316 |   37 |   30 |   22 |      0.3729 |   0.4231 |     0.3964 |        52 |
| Paura         |  393 |    0 |    7 |    5 |      1.0000 |   0.4167 |     0.5882 |        12 |
| Rabbia        |  366 |   12 |   16 |   11 |      0.4783 |   0.4074 |     0.4400 |        27 |
| Sorpresa      |  265 |   51 |   49 |   40 |      0.4396 |   0.4494 |     0.4444 |        89 |
| Tristezza     |  332 |   31 |   15 |   27 |      0.4655 |   0.6429 |     0.5400 |        42 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             49 |           32 |                    0 |               2 |             15 |           13 |             2 |            0 |             1 |               5 |                0 |           0 |
| Anticipazione  |             25 |            0 |                   16 |               3 |              6 |            1 |             1 |            1 |             1 |               1 |                3 |           0 |
| Disgusto       |             74 |            1 |                    2 |              48 |              7 |            3 |             8 |            0 |            19 |              15 |               23 |           0 |
| Fiducia        |             83 |           20 |                    5 |               6 |             45 |           15 |             9 |            1 |             2 |              12 |                8 |           0 |
| Gioia          |             90 |           13 |                    5 |               5 |             19 |           49 |             9 |            0 |             0 |              22 |                3 |           0 |
| Neutro         |             52 |            2 |                    2 |               9 |              4 |           11 |            22 |            0 |             0 |              14 |                1 |           0 |
| Paura          |             12 |            0 |                    1 |               1 |              1 |            0 |             2 |            5 |             0 |               4 |                7 |           0 |
| Rabbia         |             27 |            0 |                    1 |              25 |              0 |            0 |             0 |            0 |            11 |               5 |                7 |           0 |
| Sorpresa       |             89 |            3 |                    2 |              31 |              7 |           20 |            13 |            3 |            11 |              40 |               27 |           0 |
| Tristezza      |             42 |            0 |                    4 |              18 |              4 |            0 |             1 |            2 |             7 |              10 |               27 |           0 |

#### Textual Case: `Step 2 Oracle (Train: no rel)` (`emotion_textual_corr`)
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.5484 |   0.6939 |     0.6126 |        49 |
| Anticipazione |      0.5000 |   0.4800 |     0.4898 |        25 |
| Disgusto      |      0.6076 |   0.6486 |     0.6275 |        74 |
| Fiducia       |      0.6711 |   0.6145 |     0.6415 |        83 |
| Gioia         |      0.5505 |   0.6667 |     0.6030 |        90 |
| Neutro        |      0.5102 |   0.4808 |     0.4950 |        52 |
| Paura         |      0.8000 |   0.3333 |     0.4706 |        12 |
| Rabbia        |      0.4400 |   0.4074 |     0.4231 |        27 |
| Sorpresa      |      0.4535 |   0.4382 |     0.4457 |        89 |
| Tristezza     |      0.4921 |   0.7381 |     0.5905 |        42 |
| micro avg     |      0.5450 |   0.5801 |     0.5620 |       543 |
| macro avg     |      0.5573 |   0.5501 |     0.5399 |       543 |
| weighted avg  |      0.5499 |   0.5801 |     0.5589 |       543 |
| samples avg   |      0.5479 |   0.5543 |     0.5302 |       543 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  328 |   28 |   15 |   34 |      0.5484 |   0.6939 |     0.6126 |        49 |
| Anticipazione |  368 |   12 |   13 |   12 |      0.5000 |   0.4800 |     0.4898 |        25 |
| Disgusto      |  300 |   31 |   26 |   48 |      0.6076 |   0.6486 |     0.6275 |        74 |
| Fiducia       |  297 |   25 |   32 |   51 |      0.6711 |   0.6145 |     0.6415 |        83 |
| Gioia         |  266 |   49 |   30 |   60 |      0.5505 |   0.6667 |     0.6030 |        90 |
| Neutro        |  329 |   24 |   27 |   25 |      0.5102 |   0.4808 |     0.4950 |        52 |
| Paura         |  392 |    1 |    8 |    4 |      0.8000 |   0.3333 |     0.4706 |        12 |
| Rabbia        |  364 |   14 |   16 |   11 |      0.4400 |   0.4074 |     0.4231 |        27 |
| Sorpresa      |  269 |   47 |   50 |   39 |      0.4535 |   0.4382 |     0.4457 |        89 |
| Tristezza     |  331 |   32 |   11 |   31 |      0.4921 |   0.7381 |     0.5905 |        42 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             49 |           34 |                    0 |               2 |             14 |           19 |             1 |            0 |             1 |               2 |                1 |           0 |
| Anticipazione  |             25 |            0 |                   12 |               3 |              4 |            2 |             0 |            1 |             1 |               2 |                6 |           0 |
| Disgusto       |             74 |            1 |                    2 |              48 |              6 |            3 |             7 |            0 |            19 |              20 |               24 |           0 |
| Fiducia        |             83 |           26 |                    5 |               5 |             51 |           18 |             7 |            1 |             2 |               8 |                8 |           0 |
| Gioia          |             90 |           16 |                    2 |               2 |             15 |           60 |             5 |            0 |             0 |              19 |                2 |           0 |
| Neutro         |             52 |            4 |                    2 |               5 |              2 |            9 |            25 |            0 |             0 |               9 |                3 |           0 |
| Paura          |             12 |            0 |                    1 |               2 |              1 |            1 |             0 |            4 |             0 |               8 |                7 |           0 |
| Rabbia         |             27 |            0 |                    0 |              22 |              1 |            0 |             0 |            0 |            11 |               8 |               11 |           0 |
| Sorpresa       |             89 |            3 |                    3 |              27 |              6 |           24 |             9 |            3 |            11 |              39 |               28 |           0 |
| Tristezza      |             42 |            0 |                    2 |              18 |              5 |            0 |             1 |            3 |             6 |              14 |               31 |           0 |

#### Textual Case: `Step 2 Oracle (Train: +rel)` (`emotion_textual_corr_with_rel`)
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.5789 |   0.6735 |     0.6226 |        49 |
| Anticipazione |      0.6471 |   0.4400 |     0.5238 |        25 |
| Disgusto      |      0.6076 |   0.6486 |     0.6275 |        74 |
| Fiducia       |      0.5955 |   0.6386 |     0.6163 |        83 |
| Gioia         |      0.6098 |   0.5556 |     0.5814 |        90 |
| Neutro        |      0.4242 |   0.5385 |     0.4746 |        52 |
| Paura         |      0.8333 |   0.4167 |     0.5556 |        12 |
| Rabbia        |      0.4286 |   0.4444 |     0.4364 |        27 |
| Sorpresa      |      0.4750 |   0.4270 |     0.4497 |        89 |
| Tristezza     |      0.4444 |   0.7619 |     0.5614 |        42 |
| micro avg     |      0.5382 |   0.5709 |     0.5541 |       543 |
| macro avg     |      0.5644 |   0.5545 |     0.5449 |       543 |
| weighted avg  |      0.5495 |   0.5709 |     0.5529 |       543 |
| samples avg   |      0.5480 |   0.5510 |     0.5288 |       543 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  332 |   24 |   16 |   33 |      0.5789 |   0.6735 |     0.6226 |        49 |
| Anticipazione |  374 |    6 |   14 |   11 |      0.6471 |   0.4400 |     0.5238 |        25 |
| Disgusto      |  300 |   31 |   26 |   48 |      0.6076 |   0.6486 |     0.6275 |        74 |
| Fiducia       |  286 |   36 |   30 |   53 |      0.5955 |   0.6386 |     0.6163 |        83 |
| Gioia         |  283 |   32 |   40 |   50 |      0.6098 |   0.5556 |     0.5814 |        90 |
| Neutro        |  315 |   38 |   24 |   28 |      0.4242 |   0.5385 |     0.4746 |        52 |
| Paura         |  392 |    1 |    7 |    5 |      0.8333 |   0.4167 |     0.5556 |        12 |
| Rabbia        |  362 |   16 |   15 |   12 |      0.4286 |   0.4444 |     0.4364 |        27 |
| Sorpresa      |  274 |   42 |   51 |   38 |      0.4750 |   0.4270 |     0.4497 |        89 |
| Tristezza     |  323 |   40 |   10 |   32 |      0.4444 |   0.7619 |     0.5614 |        42 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             49 |           33 |                    0 |               2 |             20 |            8 |             1 |            0 |             1 |               3 |                1 |           0 |
| Anticipazione  |             25 |            0 |                   11 |               4 |              4 |            2 |             1 |            1 |             1 |               1 |                6 |           0 |
| Disgusto       |             74 |            1 |                    1 |              48 |              7 |            1 |             8 |            1 |            20 |              13 |               23 |           0 |
| Fiducia        |             83 |           23 |                    3 |               5 |             53 |            8 |             9 |            1 |             3 |               7 |               12 |           0 |
| Gioia          |             90 |           18 |                    0 |               2 |             15 |           50 |            10 |            0 |             1 |              24 |                8 |           0 |
| Neutro         |             52 |            1 |                    2 |               2 |              4 |           12 |            28 |            0 |             0 |               8 |                2 |           0 |
| Paura          |             12 |            0 |                    1 |               3 |              2 |            0 |             2 |            5 |             0 |               5 |                6 |           0 |
| Rabbia         |             27 |            0 |                    0 |              22 |              1 |            1 |             0 |            0 |            12 |               4 |               10 |           0 |
| Sorpresa       |             89 |            5 |                    0 |              25 |              7 |           18 |            14 |            4 |            13 |              38 |               32 |           0 |
| Tristezza      |             42 |            0 |                    1 |              16 |              6 |            0 |             2 |            3 |             7 |              11 |               32 |           0 |

---

### 1.2 Multimodal Emotion Benchmark (`mm_test`, $N=129$ — All 6 Cases $\times$ `raw` vs `std`)

#### Master Comparison Table (All Multimodal Emotion Runs)
| Case   | Regime                                      | Train Relation   | Eval Relation   | Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:-------|:--------------------------------------------|:-----------------|:----------------|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Blind Text (ClassifierChain)     | raw       |     0.3104 |     0.3536 |        0.3636 |        0.2093 |
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.3957 |     0.3691 |        0.3894 |        0.1473 |
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Blind Text (ClassifierChain)     | std       |     0.3532 |     0.3774 |        0.4462 |        0.0853 |
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.3137 |     0.3636 |        0.4217 |        0.1318 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Blind Text (ClassifierChain)     | raw       |     0.4530 |     0.5029 |        0.5099 |        0.2713 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4300 |     0.4909 |        0.4944 |        0.3566 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Blind Text (ClassifierChain)     | std       |     0.4283 |     0.4718 |        0.4982 |        0.2326 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4301 |     0.4747 |        0.5011 |        0.2558 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Blind Text (ClassifierChain)     | raw       |     0.4233 |     0.4593 |        0.5291 |        0.1395 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.5090 |     0.5354 |        0.5342 |        0.4031 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Blind Text (ClassifierChain)     | std       |     0.4208 |     0.4878 |        0.5420 |        0.2791 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4405 |     0.4821 |        0.5540 |        0.2481 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Blind Text (ClassifierChain)     | raw       |     0.3645 |     0.4136 |        0.4237 |        0.2403 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.5072 |     0.5222 |        0.5184 |        0.3023 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Blind Text (ClassifierChain)     | std       |     0.4098 |     0.4397 |        0.5203 |        0.1705 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4263 |     0.4875 |        0.5122 |        0.2713 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Blind Text (ClassifierChain)     | raw       |     0.4285 |     0.4840 |        0.4956 |        0.3023 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4821 |     0.5062 |        0.5206 |        0.3566 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Blind Text (ClassifierChain)     | std       |     0.4390 |     0.4886 |        0.5178 |        0.3488 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4424 |     0.4945 |        0.5440 |        0.2868 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Blind Text (ClassifierChain)     | raw       |     0.4700 |     0.5252 |        0.5539 |        0.2558 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4688 |     0.5401 |        0.5568 |        0.3798 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Blind Text (ClassifierChain)     | std       |     0.4572 |     0.5000 |        0.5603 |        0.2403 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4553 |     0.5100 |        0.5467 |        0.2946 |

#### Per-Emotion F1 Breakdown (All Multimodal Emotion Runs)
| Case   | Regime                                      | Model                            | Scaling   |   Amore F1 |   Anticipazione F1 |   Disgusto F1 |   Fiducia F1 |   Gioia F1 |   Neutro F1 |   Paura F1 |   Rabbia F1 |   Sorpresa F1 |   Tristezza F1 |
|:-------|:--------------------------------------------|:---------------------------------|:----------|-----------:|-------------------:|--------------:|-------------:|-----------:|------------:|-----------:|------------:|--------------:|---------------:|
| Case 1 | Step 1 Base (No corr)                       | Blind Text (ClassifierChain)     | raw       |     0.4667 |             0.5217 |        0.3830 |       0.4884 |     0.2745 |      0.3077 |     0.0000 |      0.1622 |        0.1667 |         0.3333 |
| Case 1 | Step 1 Base (No corr)                       | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.2128 |             0.4444 |        0.4211 |       0.5714 |     0.3284 |      0.2791 |     0.6667 |      0.2667 |        0.1951 |         0.5714 |
| Case 1 | Step 1 Base (No corr)                       | Blind Text (ClassifierChain)     | std       |     0.4000 |             0.4828 |        0.4255 |       0.5714 |     0.4533 |      0.4324 |     0.0385 |      0.3077 |        0.2667 |         0.1538 |
| Case 1 | Step 1 Base (No corr)                       | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.2286 |             0.1538 |        0.3500 |       0.7089 |     0.3607 |      0.3182 |     0.0444 |      0.1818 |        0.3902 |         0.4000 |
| Case 2 | Step 2 Raw (No corr)                        | Blind Text (ClassifierChain)     | raw       |     0.6316 |             0.4286 |        0.4571 |       0.6667 |     0.4928 |      0.3529 |     0.0000 |      0.3000 |        0.4000 |         0.8000 |
| Case 2 | Step 2 Raw (No corr)                        | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4444 |             0.5517 |        0.4516 |       0.6410 |     0.4918 |      0.3810 |     0.0000 |      0.3077 |        0.3636 |         0.6667 |
| Case 2 | Step 2 Raw (No corr)                        | Blind Text (ClassifierChain)     | std       |     0.4865 |             0.3810 |        0.5000 |       0.6744 |     0.4545 |      0.4231 |     0.1250 |      0.3077 |        0.3590 |         0.5714 |
| Case 2 | Step 2 Raw (No corr)                        | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4091 |             0.5600 |        0.3684 |       0.6977 |     0.5217 |      0.3333 |     0.1818 |      0.5714 |        0.3243 |         0.3333 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (ClassifierChain)     | raw       |     0.5517 |             0.2195 |        0.6667 |       0.6585 |     0.6176 |      0.1875 |     0.0769 |      0.4211 |        0.5000 |         0.3333 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.3000 |             0.6316 |        0.5455 |       0.7191 |     0.5588 |      0.4000 |     0.6667 |      0.3077 |        0.2941 |         0.6667 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (ClassifierChain)     | std       |     0.5333 |             0.4828 |        0.5882 |       0.6818 |     0.5714 |      0.3256 |     0.0667 |      0.3333 |        0.5000 |         0.1250 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.5385 |             0.6087 |        0.6500 |       0.6500 |     0.5970 |      0.4211 |     0.0476 |      0.2581 |        0.4118 |         0.2222 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Blind Text (ClassifierChain)     | raw       |     0.2769 |             0.5000 |        0.5424 |       0.5429 |     0.5000 |      0.0952 |     0.2000 |      0.4706 |        0.2500 |         0.2667 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.3571 |             0.5882 |        0.5405 |       0.6239 |     0.5600 |      0.4000 |     0.5000 |      0.3333 |        0.3684 |         0.8000 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Blind Text (ClassifierChain)     | std       |     0.2545 |             0.4000 |        0.6829 |       0.6207 |     0.5862 |      0.3000 |     0.0426 |      0.3750 |        0.5500 |         0.2857 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.2632 |             0.5600 |        0.5238 |       0.6437 |     0.6071 |      0.3333 |     0.2500 |      0.2857 |        0.4324 |         0.3636 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Blind Text (ClassifierChain)     | raw       |     0.5333 |             0.5385 |        0.5333 |       0.6818 |     0.3333 |      0.3571 |     0.1000 |      0.3750 |        0.4324 |         0.4000 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.3478 |             0.5333 |        0.5161 |       0.6585 |     0.6102 |      0.3500 |     0.5000 |      0.4000 |        0.3333 |         0.5714 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Blind Text (ClassifierChain)     | std       |     0.4500 |             0.5000 |        0.6667 |       0.6076 |     0.4151 |      0.4286 |     0.1176 |      0.3636 |        0.5556 |         0.2857 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4848 |             0.4762 |        0.6111 |       0.6824 |     0.6000 |      0.3182 |     0.0667 |      0.5455 |        0.3889 |         0.2500 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Blind Text (ClassifierChain)     | raw       |     0.4103 |             0.4000 |        0.7895 |       0.6667 |     0.5429 |      0.3571 |     0.1818 |      0.2400 |        0.5405 |         0.5714 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.2703 |             0.5000 |        0.6667 |       0.7143 |     0.6087 |      0.4000 |     0.5000 |      0.2857 |        0.4571 |         0.2857 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Blind Text (ClassifierChain)     | std       |     0.4138 |             0.5714 |        0.7143 |       0.6458 |     0.5763 |      0.4286 |     0.0526 |      0.2400 |        0.5294 |         0.4000 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4103 |             0.5000 |        0.5455 |       0.6593 |     0.6286 |      0.4500 |     0.0833 |      0.6000 |        0.3684 |         0.3077 |

#### Semiotic Subsets Breakdown (`ADD`, `PAR`, `DIV` on `mm_test`, $N=129$)
| Case   | Regime                                      | Model                            | Scaling   |   ADD (N=68) |   PAR (N=23) |   DIV (N=38) |   ADD Macro F1 |   PAR Macro F1 |   DIV Macro F1 |   Macro F1 |   Micro F1 |
|:-------|:--------------------------------------------|:---------------------------------|:----------|-------------:|-------------:|-------------:|---------------:|---------------:|---------------:|-----------:|-----------:|
| Case 1 | Step 1 Base (No corr)                       | Blind Text (ClassifierChain)     | raw       |       0.4000 |       0.4000 |       0.2545 |         0.3575 |         0.2283 |         0.2183 |     0.3104 |     0.3536 |
| Case 1 | Step 1 Base (No corr)                       | Dual-Encoder (FEEL-IT + ViT-224) | raw       |       0.4410 |       0.3438 |       0.2500 |         0.4499 |         0.1933 |         0.1634 |     0.3957 |     0.3691 |
| Case 1 | Step 1 Base (No corr)                       | Blind Text (ClassifierChain)     | std       |       0.4490 |       0.2857 |       0.3014 |         0.4274 |         0.2125 |         0.2657 |     0.3532 |     0.3774 |
| Case 1 | Step 1 Base (No corr)                       | Dual-Encoder (FEEL-IT + ViT-224) | std       |       0.4149 |       0.4444 |       0.2537 |         0.3619 |         0.2156 |         0.2299 |     0.3137 |     0.3636 |
| Case 2 | Step 2 Raw (No corr)                        | Blind Text (ClassifierChain)     | raw       |       0.5587 |       0.5357 |       0.3925 |         0.5185 |         0.3322 |         0.3127 |     0.4530 |     0.5029 |
| Case 2 | Step 2 Raw (No corr)                        | Dual-Encoder (FEEL-IT + ViT-224) | raw       |       0.5780 |       0.5085 |       0.3265 |         0.4988 |         0.3157 |         0.2373 |     0.4300 |     0.4909 |
| Case 2 | Step 2 Raw (No corr)                        | Blind Text (ClassifierChain)     | std       |       0.5524 |       0.4412 |       0.3393 |         0.5069 |         0.2924 |         0.2829 |     0.4283 |     0.4718 |
| Case 2 | Step 2 Raw (No corr)                        | Dual-Encoder (FEEL-IT + ViT-224) | std       |       0.5366 |       0.5231 |       0.3492 |         0.5159 |         0.3191 |         0.2674 |     0.4301 |     0.4747 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (ClassifierChain)     | raw       |       0.5278 |       0.4242 |       0.3676 |         0.5142 |         0.4569 |         0.3506 |     0.4233 |     0.4593 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (FEEL-IT + ViT-224) | raw       |       0.6012 |       0.5455 |       0.4124 |         0.5714 |         0.2167 |         0.2871 |     0.5090 |     0.5354 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (ClassifierChain)     | std       |       0.5524 |       0.5075 |       0.3759 |         0.5077 |         0.3904 |         0.3281 |     0.4208 |     0.4878 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (FEEL-IT + ViT-224) | std       |       0.5481 |       0.5660 |       0.3411 |         0.5286 |         0.3286 |         0.2913 |     0.4405 |     0.4821 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Blind Text (ClassifierChain)     | raw       |       0.5054 |       0.3571 |       0.2883 |         0.4519 |         0.1962 |         0.1918 |     0.3645 |     0.4136 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Dual-Encoder (FEEL-IT + ViT-224) | raw       |       0.5417 |       0.5938 |       0.4423 |         0.5318 |         0.3451 |         0.2529 |     0.5072 |     0.5222 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Blind Text (ClassifierChain)     | std       |       0.5048 |       0.4638 |       0.3333 |         0.4994 |         0.4420 |         0.3005 |     0.4098 |     0.4397 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Dual-Encoder (FEEL-IT + ViT-224) | std       |       0.5864 |       0.5085 |       0.3063 |         0.5326 |         0.2611 |         0.1871 |     0.4263 |     0.4875 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Blind Text (ClassifierChain)     | raw       |       0.5638 |       0.5000 |       0.3301 |         0.5230 |         0.2758 |         0.2744 |     0.4285 |     0.4840 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Dual-Encoder (FEEL-IT + ViT-224) | raw       |       0.5799 |       0.4906 |       0.3922 |         0.5626 |         0.2346 |         0.2731 |     0.4821 |     0.5062 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Blind Text (ClassifierChain)     | std       |       0.5532 |       0.4828 |       0.3774 |         0.5106 |         0.3206 |         0.3053 |     0.4390 |     0.4886 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Dual-Encoder (FEEL-IT + ViT-224) | std       |       0.5521 |       0.5161 |       0.3818 |         0.5182 |         0.3444 |         0.2980 |     0.4424 |     0.4945 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Blind Text (ClassifierChain)     | raw       |       0.5784 |       0.5246 |       0.4286 |         0.5403 |         0.3214 |         0.3148 |     0.4700 |     0.5252 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Dual-Encoder (FEEL-IT + ViT-224) | raw       |       0.5876 |       0.4906 |       0.4860 |         0.5190 |         0.2442 |         0.2966 |     0.4688 |     0.5401 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Blind Text (ClassifierChain)     | std       |       0.5592 |       0.5333 |       0.3840 |         0.5353 |         0.3422 |         0.2894 |     0.4572 |     0.5000 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Dual-Encoder (FEEL-IT + ViT-224) | std       |       0.5631 |       0.5312 |       0.4154 |         0.5314 |         0.3341 |         0.2953 |     0.4553 |     0.5100 |

### 1.3 Multimodal Emotion Detailed Reports by Case (`Case 1` – `Case 6`)

#### Case 1: Step 1 Base (No corr) (Checkpoint: `emotion_textual_no_corr`, Eval Prompt: `None (Raw)`)

| Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Blind Text (ClassifierChain)     | raw       |     0.3104 |     0.3536 |        0.3636 |        0.2093 |
| Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.3957 |     0.3691 |        0.3894 |        0.1473 |
| Blind Text (ClassifierChain)     | std       |     0.3532 |     0.3774 |        0.4462 |        0.0853 |
| Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.3137 |     0.3636 |        0.4217 |        0.1318 |

##### Run: `Case 1` | `Blind Text (ClassifierChain)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3684 |   0.6364 |     0.4667 |        11 |
| Anticipazione |      0.4615 |   0.6000 |     0.5217 |        10 |
| Disgusto      |      0.3214 |   0.4737 |     0.3830 |        19 |
| Fiducia       |      0.4773 |   0.5000 |     0.4884 |        42 |
| Gioia         |      0.3182 |   0.2414 |     0.2745 |        29 |
| Neutro        |      0.4444 |   0.2353 |     0.3077 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.0968 |   0.5000 |     0.1622 |         6 |
| Sorpresa      |      0.1500 |   0.1875 |     0.1667 |        16 |
| Tristezza     |      0.2500 |   0.5000 |     0.3333 |         2 |
| micro avg     |      0.3177 |   0.3987 |     0.3536 |       153 |
| macro avg     |      0.2888 |   0.3874 |     0.3104 |       153 |
| weighted avg  |      0.3600 |   0.3987 |     0.3636 |       153 |
| samples avg   |      0.3289 |   0.3966 |     0.3444 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  106 |   12 |    4 |    7 |      0.3684 |   0.6364 |     0.4667 |        11 |
| Anticipazione |  112 |    7 |    4 |    6 |      0.4615 |   0.6000 |     0.5217 |        10 |
| Disgusto      |   91 |   19 |   10 |    9 |      0.3214 |   0.4737 |     0.3830 |        19 |
| Fiducia       |   64 |   23 |   21 |   21 |      0.4773 |   0.5000 |     0.4884 |        42 |
| Gioia         |   85 |   15 |   22 |    7 |      0.3182 |   0.2414 |     0.2745 |        29 |
| Neutro        |  107 |    5 |   13 |    4 |      0.4444 |   0.2353 |     0.3077 |        17 |
| Paura         |  126 |    2 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |   95 |   28 |    3 |    3 |      0.0968 |   0.5000 |     0.1622 |         6 |
| Sorpresa      |   96 |   17 |   13 |    3 |      0.1500 |   0.1875 |     0.1667 |        16 |
| Tristezza     |  124 |    3 |    1 |    1 |      0.2500 |   0.5000 |     0.3333 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            7 |                    0 |               0 |              8 |            1 |             0 |            0 |             0 |               0 |                1 |           0 |
| Anticipazione  |             10 |            0 |                    6 |               0 |              1 |            2 |             1 |            0 |             1 |               0 |                0 |           1 |
| Disgusto       |             19 |            1 |                    1 |               9 |              1 |            2 |             2 |            0 |            14 |               8 |                1 |           0 |
| Fiducia        |             42 |           10 |                    2 |               4 |             21 |           10 |             0 |            0 |             5 |               6 |                1 |           0 |
| Gioia          |             29 |            6 |                    1 |               7 |             14 |            7 |             0 |            0 |             2 |               1 |                0 |           1 |
| Neutro         |             17 |            3 |                    3 |               2 |              4 |            1 |             4 |            0 |             3 |               1 |                0 |           0 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            1 |             0 |            0 |             1 |               1 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               2 |              1 |            4 |             0 |            1 |             3 |               3 |                1 |           0 |
| Sorpresa       |             16 |            0 |                    0 |               6 |              5 |            3 |             2 |            2 |             6 |               3 |                2 |           1 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             1 |            1 |             2 |               1 |                1 |           0 |

##### Run: `Case 1` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.1389 |   0.4545 |     0.2128 |        11 |
| Anticipazione |      0.5000 |   0.4000 |     0.4444 |        10 |
| Disgusto      |      0.4211 |   0.4211 |     0.4211 |        19 |
| Fiducia       |      0.5714 |   0.5714 |     0.5714 |        42 |
| Gioia         |      0.2895 |   0.3793 |     0.3284 |        29 |
| Neutro        |      0.2308 |   0.3529 |     0.2791 |        17 |
| Paura         |      0.5000 |   1.0000 |     0.6667 |         1 |
| Rabbia        |      0.2222 |   0.3333 |     0.2667 |         6 |
| Sorpresa      |      0.1600 |   0.2500 |     0.1951 |        16 |
| Tristezza     |      0.4000 |   1.0000 |     0.5714 |         2 |
| micro avg     |      0.3190 |   0.4379 |     0.3691 |       153 |
| macro avg     |      0.3434 |   0.5163 |     0.3957 |       153 |
| weighted avg  |      0.3663 |   0.4379 |     0.3894 |       153 |
| samples avg   |      0.3258 |   0.4444 |     0.3490 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |   87 |   31 |    6 |    5 |      0.1389 |   0.4545 |     0.2128 |        11 |
| Anticipazione |  115 |    4 |    6 |    4 |      0.5000 |   0.4000 |     0.4444 |        10 |
| Disgusto      |   99 |   11 |   11 |    8 |      0.4211 |   0.4211 |     0.4211 |        19 |
| Fiducia       |   69 |   18 |   18 |   24 |      0.5714 |   0.5714 |     0.5714 |        42 |
| Gioia         |   73 |   27 |   18 |   11 |      0.2895 |   0.3793 |     0.3284 |        29 |
| Neutro        |   92 |   20 |   11 |    6 |      0.2308 |   0.3529 |     0.2791 |        17 |
| Paura         |  127 |    1 |    0 |    1 |      0.5000 |   1.0000 |     0.6667 |         1 |
| Rabbia        |  116 |    7 |    4 |    2 |      0.2222 |   0.3333 |     0.2667 |         6 |
| Sorpresa      |   92 |   21 |   12 |    4 |      0.1600 |   0.2500 |     0.1951 |        16 |
| Tristezza     |  124 |    3 |    0 |    2 |      0.4000 |   1.0000 |     0.5714 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            5 |                    0 |               0 |              8 |            2 |             0 |            0 |             1 |               1 |                0 |           0 |
| Anticipazione  |             10 |            3 |                    4 |               0 |              2 |            3 |             1 |            0 |             0 |               0 |                0 |           1 |
| Disgusto       |             19 |            3 |                    0 |               8 |              1 |            6 |             5 |            0 |             2 |               4 |                1 |           0 |
| Fiducia        |             42 |           15 |                    2 |               1 |             24 |           13 |             6 |            0 |             1 |               7 |                0 |           5 |
| Gioia          |             29 |            2 |                    1 |               2 |              9 |           11 |             6 |            0 |             1 |               7 |                0 |           3 |
| Neutro         |             17 |            5 |                    0 |               2 |              6 |            4 |             6 |            0 |             2 |               4 |                0 |           0 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               0 |                1 |           0 |
| Rabbia         |              6 |            2 |                    0 |               2 |              1 |            1 |             1 |            0 |             2 |               0 |                2 |           0 |
| Sorpresa       |             16 |            3 |                    1 |               4 |              0 |            7 |             4 |            1 |             2 |               4 |                3 |           1 |
| Tristezza      |              2 |            1 |                    1 |               0 |              0 |            0 |             0 |            1 |             1 |               0 |                2 |           0 |

##### Run: `Case 1` | `Blind Text (ClassifierChain)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2759 |   0.7273 |     0.4000 |        11 |
| Anticipazione |      0.3684 |   0.7000 |     0.4828 |        10 |
| Disgusto      |      0.3571 |   0.5263 |     0.4255 |        19 |
| Fiducia       |      0.5714 |   0.5714 |     0.5714 |        42 |
| Gioia         |      0.3696 |   0.5862 |     0.4533 |        29 |
| Neutro        |      0.4000 |   0.4706 |     0.4324 |        17 |
| Paura         |      0.0196 |   1.0000 |     0.0385 |         1 |
| Rabbia        |      0.2000 |   0.6667 |     0.3077 |         6 |
| Sorpresa      |      0.2069 |   0.3750 |     0.2667 |        16 |
| Tristezza     |      0.0833 |   1.0000 |     0.1538 |         2 |
| micro avg     |      0.2825 |   0.5686 |     0.3774 |       153 |
| macro avg     |      0.2852 |   0.6623 |     0.3532 |       153 |
| weighted avg  |      0.3903 |   0.5686 |     0.4462 |       153 |
| samples avg   |      0.3094 |   0.5736 |     0.3822 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |   97 |   21 |    3 |    8 |      0.2759 |   0.7273 |     0.4000 |        11 |
| Anticipazione |  107 |   12 |    3 |    7 |      0.3684 |   0.7000 |     0.4828 |        10 |
| Disgusto      |   92 |   18 |    9 |   10 |      0.3571 |   0.5263 |     0.4255 |        19 |
| Fiducia       |   69 |   18 |   18 |   24 |      0.5714 |   0.5714 |     0.5714 |        42 |
| Gioia         |   71 |   29 |   12 |   17 |      0.3696 |   0.5862 |     0.4533 |        29 |
| Neutro        |  100 |   12 |    9 |    8 |      0.4000 |   0.4706 |     0.4324 |        17 |
| Paura         |   78 |   50 |    0 |    1 |      0.0196 |   1.0000 |     0.0385 |         1 |
| Rabbia        |  107 |   16 |    2 |    4 |      0.2000 |   0.6667 |     0.3077 |         6 |
| Sorpresa      |   90 |   23 |   10 |    6 |      0.2069 |   0.3750 |     0.2667 |        16 |
| Tristezza     |  105 |   22 |    0 |    2 |      0.0833 |   1.0000 |     0.1538 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            8 |                    1 |               1 |              6 |            7 |             1 |            1 |             1 |               1 |                1 |           0 |
| Anticipazione  |             10 |            0 |                    7 |               0 |              5 |            2 |             2 |            4 |             0 |               0 |                1 |           0 |
| Disgusto       |             19 |            2 |                    1 |              10 |              3 |            3 |             1 |           11 |            10 |               6 |                6 |           1 |
| Fiducia        |             42 |            9 |                    5 |               5 |             24 |           15 |             6 |            9 |             4 |               9 |                6 |           0 |
| Gioia          |             29 |            9 |                    2 |               4 |              9 |           17 |             3 |           11 |             1 |               3 |                4 |           1 |
| Neutro         |             17 |            4 |                    3 |               2 |              1 |            6 |             8 |            9 |             0 |               2 |                4 |           0 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             1 |               1 |                1 |           0 |
| Rabbia         |              6 |            0 |                    1 |               3 |              1 |            1 |             0 |            4 |             4 |               3 |                4 |           0 |
| Sorpresa       |             16 |            4 |                    1 |               7 |              1 |            7 |             2 |            9 |             3 |               6 |                5 |           0 |
| Tristezza      |              2 |            0 |                    0 |               2 |              0 |            0 |             0 |            2 |             2 |               2 |                2 |           0 |

##### Run: `Case 1` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.1667 |   0.3636 |     0.2286 |        11 |
| Anticipazione |      0.3333 |   0.1000 |     0.1538 |        10 |
| Disgusto      |      0.3333 |   0.3684 |     0.3500 |        19 |
| Fiducia       |      0.7568 |   0.6667 |     0.7089 |        42 |
| Gioia         |      0.3438 |   0.3793 |     0.3607 |        29 |
| Neutro        |      0.2593 |   0.4118 |     0.3182 |        17 |
| Paura         |      0.0227 |   1.0000 |     0.0444 |         1 |
| Rabbia        |      0.1250 |   0.3333 |     0.1818 |         6 |
| Sorpresa      |      0.3200 |   0.5000 |     0.3902 |        16 |
| Tristezza     |      0.3333 |   0.5000 |     0.4000 |         2 |
| micro avg     |      0.3017 |   0.4575 |     0.3636 |       153 |
| macro avg     |      0.2994 |   0.4623 |     0.3137 |       153 |
| weighted avg  |      0.4197 |   0.4575 |     0.4217 |       153 |
| samples avg   |      0.3140 |   0.4496 |     0.3506 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |   98 |   20 |    7 |    4 |      0.1667 |   0.3636 |     0.2286 |        11 |
| Anticipazione |  117 |    2 |    9 |    1 |      0.3333 |   0.1000 |     0.1538 |        10 |
| Disgusto      |   96 |   14 |   12 |    7 |      0.3333 |   0.3684 |     0.3500 |        19 |
| Fiducia       |   78 |    9 |   14 |   28 |      0.7568 |   0.6667 |     0.7089 |        42 |
| Gioia         |   79 |   21 |   18 |   11 |      0.3438 |   0.3793 |     0.3607 |        29 |
| Neutro        |   92 |   20 |   10 |    7 |      0.2593 |   0.4118 |     0.3182 |        17 |
| Paura         |   85 |   43 |    0 |    1 |      0.0227 |   1.0000 |     0.0444 |         1 |
| Rabbia        |  109 |   14 |    4 |    2 |      0.1250 |   0.3333 |     0.1818 |         6 |
| Sorpresa      |   96 |   17 |    8 |    8 |      0.3200 |   0.5000 |     0.3902 |        16 |
| Tristezza     |  125 |    2 |    1 |    1 |      0.3333 |   0.5000 |     0.4000 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            4 |                    0 |               0 |              9 |            3 |             0 |            1 |             1 |               0 |                0 |           1 |
| Anticipazione  |             10 |            1 |                    1 |               0 |              0 |            1 |             1 |            2 |             0 |               0 |                0 |           5 |
| Disgusto       |             19 |            1 |                    0 |               7 |              0 |            5 |             5 |           11 |             2 |               5 |                0 |           2 |
| Fiducia        |             42 |           11 |                    1 |               1 |             28 |           14 |             5 |            8 |             3 |               3 |                1 |           1 |
| Gioia          |             29 |            7 |                    0 |               6 |              8 |           11 |             7 |           11 |             7 |               6 |                1 |           2 |
| Neutro         |             17 |            3 |                    1 |               3 |              2 |            3 |             7 |            5 |             2 |               4 |                0 |           3 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             0 |               1 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               2 |              0 |            2 |             1 |            4 |             2 |               3 |                0 |           0 |
| Sorpresa       |             16 |            0 |                    0 |               4 |              1 |            4 |             4 |            9 |             3 |               8 |                1 |           0 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            2 |             1 |               2 |                1 |           0 |

#### Case 2: Step 2 Raw (No corr) (Checkpoint: `emotion_textual_step2_raw`, Eval Prompt: `None (Raw)`)

| Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Blind Text (ClassifierChain)     | raw       |     0.4530 |     0.5029 |        0.5099 |        0.2713 |
| Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4300 |     0.4909 |        0.4944 |        0.3566 |
| Blind Text (ClassifierChain)     | std       |     0.4283 |     0.4718 |        0.4982 |        0.2326 |
| Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4301 |     0.4747 |        0.5011 |        0.2558 |

##### Run: `Case 2` | `Blind Text (ClassifierChain)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.7500 |   0.5455 |     0.6316 |        11 |
| Anticipazione |      0.3333 |   0.6000 |     0.4286 |        10 |
| Disgusto      |      0.5000 |   0.4211 |     0.4571 |        19 |
| Fiducia       |      0.6078 |   0.7381 |     0.6667 |        42 |
| Gioia         |      0.4250 |   0.5862 |     0.4928 |        29 |
| Neutro        |      0.3529 |   0.3529 |     0.3529 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.2143 |   0.5000 |     0.3000 |         6 |
| Sorpresa      |      0.3684 |   0.4375 |     0.4000 |        16 |
| Tristezza     |      0.6667 |   1.0000 |     0.8000 |         2 |
| micro avg     |      0.4550 |   0.5621 |     0.5029 |       153 |
| macro avg     |      0.4218 |   0.5181 |     0.4530 |       153 |
| weighted avg  |      0.4801 |   0.5621 |     0.5099 |       153 |
| samples avg   |      0.4521 |   0.5646 |     0.4801 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  116 |    2 |    5 |    6 |      0.7500 |   0.5455 |     0.6316 |        11 |
| Anticipazione |  107 |   12 |    4 |    6 |      0.3333 |   0.6000 |     0.4286 |        10 |
| Disgusto      |  102 |    8 |   11 |    8 |      0.5000 |   0.4211 |     0.4571 |        19 |
| Fiducia       |   67 |   20 |   11 |   31 |      0.6078 |   0.7381 |     0.6667 |        42 |
| Gioia         |   77 |   23 |   12 |   17 |      0.4250 |   0.5862 |     0.4928 |        29 |
| Neutro        |  101 |   11 |   11 |    6 |      0.3529 |   0.3529 |     0.3529 |        17 |
| Paura         |  125 |    3 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  112 |   11 |    3 |    3 |      0.2143 |   0.5000 |     0.3000 |         6 |
| Sorpresa      |  101 |   12 |    9 |    7 |      0.3684 |   0.4375 |     0.4000 |        16 |
| Tristezza     |  126 |    1 |    0 |    2 |      0.6667 |   1.0000 |     0.8000 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            6 |                    2 |               0 |              7 |            6 |             0 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            0 |                    6 |               0 |              3 |            4 |             0 |            0 |             0 |               0 |                0 |           2 |
| Disgusto       |             19 |            0 |                    1 |               8 |              4 |            2 |             2 |            0 |             9 |               6 |                0 |           0 |
| Fiducia        |             42 |            5 |                    5 |               0 |             31 |           14 |             0 |            0 |             0 |               1 |                0 |           5 |
| Gioia          |             29 |            3 |                    5 |               1 |              4 |           17 |             6 |            0 |             0 |               5 |                0 |           2 |
| Neutro         |             17 |            0 |                    1 |               1 |              4 |            4 |             6 |            0 |             1 |               2 |                0 |           3 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            0 |             1 |               1 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               1 |              2 |            1 |             1 |            2 |             3 |               2 |                1 |           0 |
| Sorpresa       |             16 |            0 |                    1 |               4 |              2 |            2 |             4 |            2 |             2 |               7 |                3 |           1 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             2 |               2 |                2 |           0 |

##### Run: `Case 2` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3200 |   0.7273 |     0.4444 |        11 |
| Anticipazione |      0.4211 |   0.8000 |     0.5517 |        10 |
| Disgusto      |      0.5833 |   0.3684 |     0.4516 |        19 |
| Fiducia       |      0.6944 |   0.5952 |     0.6410 |        42 |
| Gioia         |      0.4688 |   0.5172 |     0.4918 |        29 |
| Neutro        |      0.3200 |   0.4706 |     0.3810 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.2857 |   0.3333 |     0.3077 |         6 |
| Sorpresa      |      0.3529 |   0.3750 |     0.3636 |        16 |
| Tristezza     |      0.5000 |   1.0000 |     0.6667 |         2 |
| micro avg     |      0.4576 |   0.5294 |     0.4909 |       153 |
| macro avg     |      0.3946 |   0.5187 |     0.4300 |       153 |
| weighted avg  |      0.4927 |   0.5294 |     0.4944 |       153 |
| samples avg   |      0.4851 |   0.5413 |     0.4946 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  101 |   17 |    3 |    8 |      0.3200 |   0.7273 |     0.4444 |        11 |
| Anticipazione |  108 |   11 |    2 |    8 |      0.4211 |   0.8000 |     0.5517 |        10 |
| Disgusto      |  105 |    5 |   12 |    7 |      0.5833 |   0.3684 |     0.4516 |        19 |
| Fiducia       |   76 |   11 |   17 |   25 |      0.6944 |   0.5952 |     0.6410 |        42 |
| Gioia         |   83 |   17 |   14 |   15 |      0.4688 |   0.5172 |     0.4918 |        29 |
| Neutro        |   95 |   17 |    9 |    8 |      0.3200 |   0.4706 |     0.3810 |        17 |
| Paura         |  128 |    0 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  118 |    5 |    4 |    2 |      0.2857 |   0.3333 |     0.3077 |         6 |
| Sorpresa      |  102 |   11 |   10 |    6 |      0.3529 |   0.3750 |     0.3636 |        16 |
| Tristezza     |  125 |    2 |    0 |    2 |      0.5000 |   1.0000 |     0.6667 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            8 |                    1 |               0 |              4 |            4 |             2 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    8 |               0 |              1 |            3 |             2 |            0 |             0 |               1 |                0 |           0 |
| Disgusto       |             19 |            2 |                    1 |               7 |              3 |            2 |             1 |            0 |             5 |               5 |                0 |           0 |
| Fiducia        |             42 |            8 |                    6 |               0 |             25 |           10 |             5 |            0 |             0 |               2 |                0 |           3 |
| Gioia          |             29 |            5 |                    3 |               0 |              4 |           15 |             6 |            0 |             0 |               5 |                0 |           2 |
| Neutro         |             17 |            4 |                    2 |               1 |              2 |            4 |             8 |            0 |             1 |               1 |                0 |           1 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            0 |             1 |               1 |                1 |           0 |
| Rabbia         |              6 |            1 |                    0 |               1 |              1 |            1 |             0 |            0 |             2 |               2 |                1 |           1 |
| Sorpresa       |             16 |            2 |                    0 |               2 |              1 |            2 |             3 |            0 |             0 |               6 |                3 |           2 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            0 |             0 |               1 |                2 |           0 |

##### Run: `Case 2` | `Blind Text (ClassifierChain)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3462 |   0.8182 |     0.4865 |        11 |
| Anticipazione |      0.2500 |   0.8000 |     0.3810 |        10 |
| Disgusto      |      0.6154 |   0.4211 |     0.5000 |        19 |
| Fiducia       |      0.6591 |   0.6905 |     0.6744 |        42 |
| Gioia         |      0.4054 |   0.5172 |     0.4545 |        29 |
| Neutro        |      0.3143 |   0.6471 |     0.4231 |        17 |
| Paura         |      0.0667 |   1.0000 |     0.1250 |         1 |
| Rabbia        |      0.2857 |   0.3333 |     0.3077 |         6 |
| Sorpresa      |      0.3043 |   0.4375 |     0.3590 |        16 |
| Tristezza     |      0.4000 |   1.0000 |     0.5714 |         2 |
| micro avg     |      0.3882 |   0.6013 |     0.4718 |       153 |
| macro avg     |      0.3647 |   0.6665 |     0.4283 |       153 |
| weighted avg  |      0.4590 |   0.6013 |     0.4982 |       153 |
| samples avg   |      0.4283 |   0.6111 |     0.4765 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  101 |   17 |    2 |    9 |      0.3462 |   0.8182 |     0.4865 |        11 |
| Anticipazione |   95 |   24 |    2 |    8 |      0.2500 |   0.8000 |     0.3810 |        10 |
| Disgusto      |  105 |    5 |   11 |    8 |      0.6154 |   0.4211 |     0.5000 |        19 |
| Fiducia       |   72 |   15 |   13 |   29 |      0.6591 |   0.6905 |     0.6744 |        42 |
| Gioia         |   78 |   22 |   14 |   15 |      0.4054 |   0.5172 |     0.4545 |        29 |
| Neutro        |   88 |   24 |    6 |   11 |      0.3143 |   0.6471 |     0.4231 |        17 |
| Paura         |  114 |   14 |    0 |    1 |      0.0667 |   1.0000 |     0.1250 |         1 |
| Rabbia        |  118 |    5 |    4 |    2 |      0.2857 |   0.3333 |     0.3077 |         6 |
| Sorpresa      |   97 |   16 |    9 |    7 |      0.3043 |   0.4375 |     0.3590 |        16 |
| Tristezza     |  124 |    3 |    0 |    2 |      0.4000 |   1.0000 |     0.5714 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            9 |                    2 |               0 |              6 |            5 |             3 |            1 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    8 |               0 |              3 |            2 |             0 |            2 |             0 |               1 |                1 |           0 |
| Disgusto       |             19 |            1 |                    2 |               8 |              4 |            2 |             2 |            2 |             4 |               5 |                0 |           0 |
| Fiducia        |             42 |           10 |                   11 |               1 |             29 |           13 |             8 |            3 |             0 |               4 |                1 |           1 |
| Gioia          |             29 |            6 |                    7 |               1 |              4 |           15 |             6 |            3 |             0 |               7 |                0 |           1 |
| Neutro         |             17 |            4 |                    5 |               1 |              2 |            4 |            11 |            1 |             0 |               3 |                1 |           0 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               1 |                0 |           0 |
| Rabbia         |              6 |            1 |                    0 |               2 |              2 |            1 |             1 |            2 |             2 |               2 |                1 |           0 |
| Sorpresa       |             16 |            1 |                    0 |               0 |              1 |            3 |             6 |            4 |             1 |               7 |                3 |           1 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            2 |             1 |               2 |                2 |           0 |

##### Run: `Case 2` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2727 |   0.8182 |     0.4091 |        11 |
| Anticipazione |      0.4667 |   0.7000 |     0.5600 |        10 |
| Disgusto      |      0.3684 |   0.3684 |     0.3684 |        19 |
| Fiducia       |      0.6818 |   0.7143 |     0.6977 |        42 |
| Gioia         |      0.4500 |   0.6207 |     0.5217 |        29 |
| Neutro        |      0.2326 |   0.5882 |     0.3333 |        17 |
| Paura         |      0.1000 |   1.0000 |     0.1818 |         1 |
| Rabbia        |      0.5000 |   0.6667 |     0.5714 |         6 |
| Sorpresa      |      0.2857 |   0.3750 |     0.3243 |        16 |
| Tristezza     |      0.2000 |   1.0000 |     0.3333 |         2 |
| micro avg     |      0.3868 |   0.6144 |     0.4747 |       153 |
| macro avg     |      0.3558 |   0.6851 |     0.4301 |       153 |
| weighted avg  |      0.4469 |   0.6144 |     0.5011 |       153 |
| samples avg   |      0.4566 |   0.6382 |     0.5043 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |   94 |   24 |    2 |    9 |      0.2727 |   0.8182 |     0.4091 |        11 |
| Anticipazione |  111 |    8 |    3 |    7 |      0.4667 |   0.7000 |     0.5600 |        10 |
| Disgusto      |   98 |   12 |   12 |    7 |      0.3684 |   0.3684 |     0.3684 |        19 |
| Fiducia       |   73 |   14 |   12 |   30 |      0.6818 |   0.7143 |     0.6977 |        42 |
| Gioia         |   78 |   22 |   11 |   18 |      0.4500 |   0.6207 |     0.5217 |        29 |
| Neutro        |   79 |   33 |    7 |   10 |      0.2326 |   0.5882 |     0.3333 |        17 |
| Paura         |  119 |    9 |    0 |    1 |      0.1000 |   1.0000 |     0.1818 |         1 |
| Rabbia        |  119 |    4 |    2 |    4 |      0.5000 |   0.6667 |     0.5714 |         6 |
| Sorpresa      |   98 |   15 |   10 |    6 |      0.2857 |   0.3750 |     0.3243 |        16 |
| Tristezza     |  119 |    8 |    0 |    2 |      0.2000 |   1.0000 |     0.3333 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            9 |                    0 |               0 |              5 |            3 |             7 |            1 |             0 |               0 |                1 |           0 |
| Anticipazione  |             10 |            1 |                    7 |               0 |              2 |            3 |             2 |            3 |             0 |               1 |                0 |           0 |
| Disgusto       |             19 |            2 |                    2 |               7 |              4 |            2 |             3 |            0 |             3 |               6 |                2 |           0 |
| Fiducia        |             42 |           14 |                    2 |               1 |             30 |           13 |            11 |            3 |             0 |               6 |                1 |           1 |
| Gioia          |             29 |            7 |                    3 |               3 |              4 |           18 |            11 |            1 |             1 |               5 |                1 |           1 |
| Neutro         |             17 |            3 |                    1 |               2 |              3 |            5 |            10 |            1 |             1 |               1 |                1 |           0 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               1 |                1 |           0 |
| Rabbia         |              6 |            2 |                    0 |               2 |              1 |            1 |             1 |            1 |             4 |               2 |                1 |           0 |
| Sorpresa       |             16 |            2 |                    0 |               3 |              1 |            4 |             6 |            2 |             1 |               6 |                4 |           1 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             0 |               1 |                2 |           0 |

#### Case 3: Step 2 Oracle (Train: no rel, Eval: no rel) (Checkpoint: `emotion_textual_corr`, Eval Prompt: `Oracle (no rel)`)

| Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Blind Text (ClassifierChain)     | raw       |     0.4233 |     0.4593 |        0.5291 |        0.1395 |
| Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.5090 |     0.5354 |        0.5342 |        0.4031 |
| Blind Text (ClassifierChain)     | std       |     0.4208 |     0.4878 |        0.5420 |        0.2791 |
| Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4405 |     0.4821 |        0.5540 |        0.2481 |

##### Run: `Case 3` | `Blind Text (ClassifierChain)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.4444 |   0.7273 |     0.5517 |        11 |
| Anticipazione |      0.1250 |   0.9000 |     0.2195 |        10 |
| Disgusto      |      0.6087 |   0.7368 |     0.6667 |        19 |
| Fiducia       |      0.6750 |   0.6429 |     0.6585 |        42 |
| Gioia         |      0.5385 |   0.7241 |     0.6176 |        29 |
| Neutro        |      0.2000 |   0.1765 |     0.1875 |        17 |
| Paura         |      0.0400 |   1.0000 |     0.0769 |         1 |
| Rabbia        |      0.3077 |   0.6667 |     0.4211 |         6 |
| Sorpresa      |      0.5000 |   0.5000 |     0.5000 |        16 |
| Tristezza     |      0.2500 |   0.5000 |     0.3333 |         2 |
| micro avg     |      0.3623 |   0.6275 |     0.4593 |       153 |
| macro avg     |      0.3689 |   0.6574 |     0.4233 |       153 |
| weighted avg  |      0.4932 |   0.6275 |     0.5291 |       153 |
| samples avg   |      0.3837 |   0.6059 |     0.4516 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  108 |   10 |    3 |    8 |      0.4444 |   0.7273 |     0.5517 |        11 |
| Anticipazione |   56 |   63 |    1 |    9 |      0.1250 |   0.9000 |     0.2195 |        10 |
| Disgusto      |  101 |    9 |    5 |   14 |      0.6087 |   0.7368 |     0.6667 |        19 |
| Fiducia       |   74 |   13 |   15 |   27 |      0.6750 |   0.6429 |     0.6585 |        42 |
| Gioia         |   82 |   18 |    8 |   21 |      0.5385 |   0.7241 |     0.6176 |        29 |
| Neutro        |  100 |   12 |   14 |    3 |      0.2000 |   0.1765 |     0.1875 |        17 |
| Paura         |  104 |   24 |    0 |    1 |      0.0400 |   1.0000 |     0.0769 |         1 |
| Rabbia        |  114 |    9 |    2 |    4 |      0.3077 |   0.6667 |     0.4211 |         6 |
| Sorpresa      |  105 |    8 |    8 |    8 |      0.5000 |   0.5000 |     0.5000 |        16 |
| Tristezza     |  124 |    3 |    1 |    1 |      0.2500 |   0.5000 |     0.3333 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            8 |                    7 |               0 |              7 |            5 |             0 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    9 |               0 |              4 |            3 |             0 |            2 |             0 |               1 |                0 |           0 |
| Disgusto       |             19 |            0 |                    2 |              14 |              1 |            1 |             1 |           12 |             6 |               2 |                1 |           0 |
| Fiducia        |             42 |            8 |                   25 |               2 |             27 |           14 |             1 |            2 |             0 |               3 |                0 |           0 |
| Gioia          |             29 |            5 |                   24 |               3 |              6 |           21 |             4 |            2 |             0 |               6 |                1 |           0 |
| Neutro         |             17 |            1 |                   12 |               1 |              2 |            4 |             3 |            1 |             2 |               1 |                0 |           1 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             1 |               1 |                0 |           0 |
| Rabbia         |              6 |            0 |                    1 |               3 |              1 |            0 |             1 |            4 |             4 |               1 |                1 |           0 |
| Sorpresa       |             16 |            0 |                    6 |               2 |              2 |            7 |             5 |            6 |             3 |               8 |                2 |           0 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             1 |            2 |             2 |               1 |                1 |           0 |

##### Run: `Case 3` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3333 |   0.2727 |     0.3000 |        11 |
| Anticipazione |      0.6667 |   0.6000 |     0.6316 |        10 |
| Disgusto      |      0.6429 |   0.4737 |     0.5455 |        19 |
| Fiducia       |      0.6809 |   0.7619 |     0.7191 |        42 |
| Gioia         |      0.4872 |   0.6552 |     0.5588 |        29 |
| Neutro        |      0.3478 |   0.4706 |     0.4000 |        17 |
| Paura         |      0.5000 |   1.0000 |     0.6667 |         1 |
| Rabbia        |      0.2857 |   0.3333 |     0.3077 |         6 |
| Sorpresa      |      0.2778 |   0.3125 |     0.2941 |        16 |
| Tristezza     |      0.5000 |   1.0000 |     0.6667 |         2 |
| micro avg     |      0.5058 |   0.5686 |     0.5354 |       153 |
| macro avg     |      0.4722 |   0.5880 |     0.5090 |       153 |
| weighted avg  |      0.5153 |   0.5686 |     0.5342 |       153 |
| samples avg   |      0.5255 |   0.5736 |     0.5341 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  112 |    6 |    8 |    3 |      0.3333 |   0.2727 |     0.3000 |        11 |
| Anticipazione |  116 |    3 |    4 |    6 |      0.6667 |   0.6000 |     0.6316 |        10 |
| Disgusto      |  105 |    5 |   10 |    9 |      0.6429 |   0.4737 |     0.5455 |        19 |
| Fiducia       |   72 |   15 |   10 |   32 |      0.6809 |   0.7619 |     0.7191 |        42 |
| Gioia         |   80 |   20 |   10 |   19 |      0.4872 |   0.6552 |     0.5588 |        29 |
| Neutro        |   97 |   15 |    9 |    8 |      0.3478 |   0.4706 |     0.4000 |        17 |
| Paura         |  127 |    1 |    0 |    1 |      0.5000 |   1.0000 |     0.6667 |         1 |
| Rabbia        |  118 |    5 |    4 |    2 |      0.2857 |   0.3333 |     0.3077 |         6 |
| Sorpresa      |  100 |   13 |   11 |    5 |      0.2778 |   0.3125 |     0.2941 |        16 |
| Tristezza     |  125 |    2 |    0 |    2 |      0.5000 |   1.0000 |     0.6667 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            3 |                    0 |               0 |              9 |            5 |             3 |            0 |             0 |               2 |                0 |           1 |
| Anticipazione  |             10 |            0 |                    6 |               0 |              1 |            5 |             0 |            0 |             0 |               0 |                0 |           1 |
| Disgusto       |             19 |            1 |                    1 |               9 |              1 |            2 |             2 |            0 |             4 |               4 |                1 |           1 |
| Fiducia        |             42 |            3 |                    2 |               0 |             32 |           12 |             5 |            0 |             0 |               3 |                0 |           3 |
| Gioia          |             29 |            3 |                    0 |               1 |              7 |           19 |             4 |            0 |             1 |               5 |                0 |           3 |
| Neutro         |             17 |            1 |                    0 |               1 |              4 |            5 |             8 |            0 |             0 |               2 |                0 |           0 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               1 |                1 |           0 |
| Rabbia         |              6 |            1 |                    0 |               2 |              1 |            1 |             1 |            0 |             2 |               1 |                1 |           1 |
| Sorpresa       |             16 |            0 |                    0 |               0 |              2 |            6 |             1 |            1 |             1 |               5 |                2 |           5 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             1 |               1 |                2 |           0 |

##### Run: `Case 3` | `Blind Text (ClassifierChain)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.4211 |   0.7273 |     0.5333 |        11 |
| Anticipazione |      0.3684 |   0.7000 |     0.4828 |        10 |
| Disgusto      |      0.4688 |   0.7895 |     0.5882 |        19 |
| Fiducia       |      0.6522 |   0.7143 |     0.6818 |        42 |
| Gioia         |      0.5294 |   0.6207 |     0.5714 |        29 |
| Neutro        |      0.2692 |   0.4118 |     0.3256 |        17 |
| Paura         |      0.0345 |   1.0000 |     0.0667 |         1 |
| Rabbia        |      0.2222 |   0.6667 |     0.3333 |         6 |
| Sorpresa      |      0.4500 |   0.5625 |     0.5000 |        16 |
| Tristezza     |      0.0714 |   0.5000 |     0.1250 |         2 |
| micro avg     |      0.3891 |   0.6536 |     0.4878 |       153 |
| macro avg     |      0.3487 |   0.6693 |     0.4208 |       153 |
| weighted avg  |      0.4788 |   0.6536 |     0.5420 |       153 |
| samples avg   |      0.4642 |   0.6447 |     0.5133 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  107 |   11 |    3 |    8 |      0.4211 |   0.7273 |     0.5333 |        11 |
| Anticipazione |  107 |   12 |    3 |    7 |      0.3684 |   0.7000 |     0.4828 |        10 |
| Disgusto      |   93 |   17 |    4 |   15 |      0.4688 |   0.7895 |     0.5882 |        19 |
| Fiducia       |   71 |   16 |   12 |   30 |      0.6522 |   0.7143 |     0.6818 |        42 |
| Gioia         |   84 |   16 |   11 |   18 |      0.5294 |   0.6207 |     0.5714 |        29 |
| Neutro        |   93 |   19 |   10 |    7 |      0.2692 |   0.4118 |     0.3256 |        17 |
| Paura         |  100 |   28 |    0 |    1 |      0.0345 |   1.0000 |     0.0667 |         1 |
| Rabbia        |  109 |   14 |    2 |    4 |      0.2222 |   0.6667 |     0.3333 |         6 |
| Sorpresa      |  102 |   11 |    7 |    9 |      0.4500 |   0.5625 |     0.5000 |        16 |
| Tristezza     |  114 |   13 |    1 |    1 |      0.0714 |   0.5000 |     0.1250 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            8 |                    0 |               0 |              8 |            4 |             4 |            0 |             1 |               1 |                1 |           0 |
| Anticipazione  |             10 |            1 |                    7 |               1 |              2 |            3 |             1 |            3 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            0 |                    1 |              15 |              2 |            1 |             2 |            9 |             8 |               2 |                2 |           0 |
| Fiducia        |             42 |            7 |                    4 |               4 |             30 |           13 |             3 |            4 |             1 |               4 |                3 |           0 |
| Gioia          |             29 |            6 |                    3 |               3 |              9 |           18 |             6 |            3 |             2 |               9 |                4 |           0 |
| Neutro         |             17 |            3 |                    4 |               1 |              2 |            3 |             7 |            3 |             2 |               1 |                2 |           0 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               0 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               4 |              1 |            0 |             1 |            3 |             4 |               1 |                2 |           0 |
| Sorpresa       |             16 |            0 |                    2 |               7 |              3 |            5 |             3 |            8 |             3 |               9 |                5 |           0 |
| Tristezza      |              2 |            0 |                    0 |               2 |              0 |            0 |             0 |            2 |             2 |               1 |                1 |           0 |

##### Run: `Case 3` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.4667 |   0.6364 |     0.5385 |        11 |
| Anticipazione |      0.5385 |   0.7000 |     0.6087 |        10 |
| Disgusto      |      0.6190 |   0.6842 |     0.6500 |        19 |
| Fiducia       |      0.6842 |   0.6190 |     0.6500 |        42 |
| Gioia         |      0.5263 |   0.6897 |     0.5970 |        29 |
| Neutro        |      0.3810 |   0.4706 |     0.4211 |        17 |
| Paura         |      0.0244 |   1.0000 |     0.0476 |         1 |
| Rabbia        |      0.1600 |   0.6667 |     0.2581 |         6 |
| Sorpresa      |      0.3889 |   0.4375 |     0.4118 |        16 |
| Tristezza     |      0.1429 |   0.5000 |     0.2222 |         2 |
| micro avg     |      0.3966 |   0.6144 |     0.4821 |       153 |
| macro avg     |      0.3932 |   0.6404 |     0.4405 |       153 |
| weighted avg  |      0.5245 |   0.6144 |     0.5540 |       153 |
| samples avg   |      0.4516 |   0.6266 |     0.4979 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  110 |    8 |    4 |    7 |      0.4667 |   0.6364 |     0.5385 |        11 |
| Anticipazione |  113 |    6 |    3 |    7 |      0.5385 |   0.7000 |     0.6087 |        10 |
| Disgusto      |  102 |    8 |    6 |   13 |      0.6190 |   0.6842 |     0.6500 |        19 |
| Fiducia       |   75 |   12 |   16 |   26 |      0.6842 |   0.6190 |     0.6500 |        42 |
| Gioia         |   82 |   18 |    9 |   20 |      0.5263 |   0.6897 |     0.5970 |        29 |
| Neutro        |   99 |   13 |    9 |    8 |      0.3810 |   0.4706 |     0.4211 |        17 |
| Paura         |   88 |   40 |    0 |    1 |      0.0244 |   1.0000 |     0.0476 |         1 |
| Rabbia        |  102 |   21 |    2 |    4 |      0.1600 |   0.6667 |     0.2581 |         6 |
| Sorpresa      |  102 |   11 |    9 |    7 |      0.3889 |   0.4375 |     0.4118 |        16 |
| Tristezza     |  121 |    6 |    1 |    1 |      0.1429 |   0.5000 |     0.2222 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            7 |                    0 |               0 |              1 |            5 |             5 |            0 |             0 |               1 |                1 |           0 |
| Anticipazione  |             10 |            0 |                    7 |               0 |              1 |            4 |             1 |            4 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            1 |                    1 |              13 |              1 |            2 |             1 |           17 |            14 |               3 |                1 |           0 |
| Fiducia        |             42 |            3 |                    3 |               1 |             26 |           12 |             4 |            6 |             1 |               5 |                1 |           1 |
| Gioia          |             29 |            6 |                    1 |               4 |              5 |           20 |             3 |            5 |             4 |               5 |                1 |           1 |
| Neutro         |             17 |            3 |                    2 |               0 |              4 |            4 |             8 |            2 |             1 |               1 |                0 |           0 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             1 |               0 |                1 |           0 |
| Rabbia         |              6 |            0 |                    0 |               3 |              1 |            1 |             1 |            4 |             4 |               0 |                1 |           0 |
| Sorpresa       |             16 |            0 |                    0 |               2 |              3 |            7 |             1 |            8 |             5 |               7 |                2 |           0 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            2 |             2 |               1 |                1 |           0 |

#### Case 4: Step 2 Oracle (Train: no rel, Eval: +rel) (Checkpoint: `emotion_textual_corr`, Eval Prompt: `Oracle (+ rel)`)

| Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Blind Text (ClassifierChain)     | raw       |     0.3645 |     0.4136 |        0.4237 |        0.2403 |
| Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.5072 |     0.5222 |        0.5184 |        0.3023 |
| Blind Text (ClassifierChain)     | std       |     0.4098 |     0.4397 |        0.5203 |        0.1705 |
| Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4263 |     0.4875 |        0.5122 |        0.2713 |

##### Run: `Case 4` | `Blind Text (ClassifierChain)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.1667 |   0.8182 |     0.2769 |        11 |
| Anticipazione |      0.4286 |   0.6000 |     0.5000 |        10 |
| Disgusto      |      0.4000 |   0.8421 |     0.5424 |        19 |
| Fiducia       |      0.6786 |   0.4524 |     0.5429 |        42 |
| Gioia         |      0.6316 |   0.4138 |     0.5000 |        29 |
| Neutro        |      0.2500 |   0.0588 |     0.0952 |        17 |
| Paura         |      0.1111 |   1.0000 |     0.2000 |         1 |
| Rabbia        |      0.3636 |   0.6667 |     0.4706 |         6 |
| Sorpresa      |      0.3750 |   0.1875 |     0.2500 |        16 |
| Tristezza     |      0.1538 |   1.0000 |     0.2667 |         2 |
| micro avg     |      0.3650 |   0.4771 |     0.4136 |       153 |
| macro avg     |      0.3559 |   0.6039 |     0.3645 |       153 |
| weighted avg  |      0.4796 |   0.4771 |     0.4237 |       153 |
| samples avg   |      0.3859 |   0.4729 |     0.4075 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |   73 |   45 |    2 |    9 |      0.1667 |   0.8182 |     0.2769 |        11 |
| Anticipazione |  111 |    8 |    4 |    6 |      0.4286 |   0.6000 |     0.5000 |        10 |
| Disgusto      |   86 |   24 |    3 |   16 |      0.4000 |   0.8421 |     0.5424 |        19 |
| Fiducia       |   78 |    9 |   23 |   19 |      0.6786 |   0.4524 |     0.5429 |        42 |
| Gioia         |   93 |    7 |   17 |   12 |      0.6316 |   0.4138 |     0.5000 |        29 |
| Neutro        |  109 |    3 |   16 |    1 |      0.2500 |   0.0588 |     0.0952 |        17 |
| Paura         |  120 |    8 |    0 |    1 |      0.1111 |   1.0000 |     0.2000 |         1 |
| Rabbia        |  116 |    7 |    2 |    4 |      0.3636 |   0.6667 |     0.4706 |         6 |
| Sorpresa      |  108 |    5 |   13 |    3 |      0.3750 |   0.1875 |     0.2500 |        16 |
| Tristezza     |  116 |   11 |    0 |    2 |      0.1538 |   1.0000 |     0.2667 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            9 |                    0 |               0 |              5 |            4 |             0 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            2 |                    6 |               1 |              2 |            1 |             0 |            1 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            1 |                    1 |              16 |              1 |            0 |             1 |            3 |             5 |               1 |                1 |           0 |
| Fiducia        |             42 |           21 |                    4 |               5 |             19 |            9 |             1 |            1 |             0 |               0 |                1 |           1 |
| Gioia          |             29 |           18 |                    2 |               8 |              2 |           12 |             0 |            0 |             0 |               4 |                4 |           0 |
| Neutro         |             17 |            7 |                    3 |               3 |              2 |            1 |             1 |            0 |             1 |               1 |                1 |           0 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               0 |                1 |           0 |
| Rabbia         |              6 |            0 |                    0 |               3 |              0 |            0 |             1 |            1 |             4 |               1 |                2 |           0 |
| Sorpresa       |             16 |            6 |                    1 |              10 |              1 |            1 |             0 |            4 |             3 |               3 |                4 |           0 |
| Tristezza      |              2 |            0 |                    0 |               2 |              0 |            0 |             0 |            2 |             2 |               0 |                2 |           0 |

##### Run: `Case 4` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2941 |   0.4545 |     0.3571 |        11 |
| Anticipazione |      0.7143 |   0.5000 |     0.5882 |        10 |
| Disgusto      |      0.5556 |   0.5263 |     0.5405 |        19 |
| Fiducia       |      0.5075 |   0.8095 |     0.6239 |        42 |
| Gioia         |      0.4565 |   0.7241 |     0.5600 |        29 |
| Neutro        |      0.3889 |   0.4118 |     0.4000 |        17 |
| Paura         |      0.3333 |   1.0000 |     0.5000 |         1 |
| Rabbia        |      0.3333 |   0.3333 |     0.3333 |         6 |
| Sorpresa      |      0.3182 |   0.4375 |     0.3684 |        16 |
| Tristezza     |      0.6667 |   1.0000 |     0.8000 |         2 |
| micro avg     |      0.4541 |   0.6144 |     0.5222 |       153 |
| macro avg     |      0.4568 |   0.6197 |     0.5072 |       153 |
| weighted avg  |      0.4631 |   0.6144 |     0.5184 |       153 |
| samples avg   |      0.4860 |   0.6176 |     0.5196 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  106 |   12 |    6 |    5 |      0.2941 |   0.4545 |     0.3571 |        11 |
| Anticipazione |  117 |    2 |    5 |    5 |      0.7143 |   0.5000 |     0.5882 |        10 |
| Disgusto      |  102 |    8 |    9 |   10 |      0.5556 |   0.5263 |     0.5405 |        19 |
| Fiducia       |   54 |   33 |    8 |   34 |      0.5075 |   0.8095 |     0.6239 |        42 |
| Gioia         |   75 |   25 |    8 |   21 |      0.4565 |   0.7241 |     0.5600 |        29 |
| Neutro        |  101 |   11 |   10 |    7 |      0.3889 |   0.4118 |     0.4000 |        17 |
| Paura         |  126 |    2 |    0 |    1 |      0.3333 |   1.0000 |     0.5000 |         1 |
| Rabbia        |  119 |    4 |    4 |    2 |      0.3333 |   0.3333 |     0.3333 |         6 |
| Sorpresa      |   98 |   15 |    9 |    7 |      0.3182 |   0.4375 |     0.3684 |        16 |
| Tristezza     |  126 |    1 |    0 |    2 |      0.6667 |   1.0000 |     0.8000 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            5 |                    0 |               0 |             10 |            5 |             0 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    5 |               0 |              4 |            7 |             0 |            0 |             0 |               1 |                0 |           0 |
| Disgusto       |             19 |            1 |                    1 |              10 |              4 |            0 |             3 |            0 |             3 |               6 |                0 |           1 |
| Fiducia        |             42 |            5 |                    1 |               0 |             34 |           14 |             3 |            0 |             0 |               4 |                0 |           2 |
| Gioia          |             29 |            2 |                    0 |               2 |             12 |           21 |             3 |            0 |             1 |               7 |                0 |           1 |
| Neutro         |             17 |            3 |                    0 |               1 |             10 |            6 |             7 |            0 |             0 |               1 |                0 |           0 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               1 |                1 |           0 |
| Rabbia         |              6 |            1 |                    0 |               3 |              1 |            0 |             1 |            1 |             2 |               1 |                0 |           0 |
| Sorpresa       |             16 |            1 |                    0 |               3 |              4 |            6 |             1 |            2 |             1 |               7 |                2 |           2 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               1 |                2 |           0 |

##### Run: `Case 4` | `Blind Text (ClassifierChain)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.1591 |   0.6364 |     0.2545 |        11 |
| Anticipazione |      0.3333 |   0.5000 |     0.4000 |        10 |
| Disgusto      |      0.6364 |   0.7368 |     0.6829 |        19 |
| Fiducia       |      0.6000 |   0.6429 |     0.6207 |        42 |
| Gioia         |      0.5862 |   0.5862 |     0.5862 |        29 |
| Neutro        |      0.2609 |   0.3529 |     0.3000 |        17 |
| Paura         |      0.0217 |   1.0000 |     0.0426 |         1 |
| Rabbia        |      0.3000 |   0.5000 |     0.3750 |         6 |
| Sorpresa      |      0.4583 |   0.6875 |     0.5500 |        16 |
| Tristezza     |      0.1667 |   1.0000 |     0.2857 |         2 |
| micro avg     |      0.3444 |   0.6078 |     0.4397 |       153 |
| macro avg     |      0.3523 |   0.6643 |     0.4098 |       153 |
| weighted avg  |      0.4791 |   0.6078 |     0.5203 |       153 |
| samples avg   |      0.3810 |   0.6047 |     0.4488 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |   81 |   37 |    4 |    7 |      0.1591 |   0.6364 |     0.2545 |        11 |
| Anticipazione |  109 |   10 |    5 |    5 |      0.3333 |   0.5000 |     0.4000 |        10 |
| Disgusto      |  102 |    8 |    5 |   14 |      0.6364 |   0.7368 |     0.6829 |        19 |
| Fiducia       |   69 |   18 |   15 |   27 |      0.6000 |   0.6429 |     0.6207 |        42 |
| Gioia         |   88 |   12 |   12 |   17 |      0.5862 |   0.5862 |     0.5862 |        29 |
| Neutro        |   95 |   17 |   11 |    6 |      0.2609 |   0.3529 |     0.3000 |        17 |
| Paura         |   83 |   45 |    0 |    1 |      0.0217 |   1.0000 |     0.0426 |         1 |
| Rabbia        |  116 |    7 |    3 |    3 |      0.3000 |   0.5000 |     0.3750 |         6 |
| Sorpresa      |  100 |   13 |    5 |   11 |      0.4583 |   0.6875 |     0.5500 |        16 |
| Tristezza     |  117 |   10 |    0 |    2 |      0.1667 |   1.0000 |     0.2857 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            7 |                    0 |               0 |              8 |            4 |             2 |            2 |             0 |               0 |                2 |           0 |
| Anticipazione  |             10 |            1 |                    5 |               0 |              3 |            2 |             1 |            2 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            3 |                    1 |              14 |              1 |            1 |             2 |           15 |             6 |               3 |                1 |           0 |
| Fiducia        |             42 |           12 |                    2 |               0 |             27 |           11 |             6 |            7 |             0 |               2 |                3 |           2 |
| Gioia          |             29 |           15 |                    3 |               2 |              6 |           17 |             3 |           10 |             1 |              11 |                6 |           1 |
| Neutro         |             17 |            9 |                    3 |               1 |              4 |            1 |             6 |            5 |             0 |               2 |                1 |           0 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             1 |               1 |                0 |           0 |
| Rabbia         |              6 |            1 |                    0 |               4 |              0 |            0 |             2 |            4 |             3 |               1 |                2 |           0 |
| Sorpresa       |             16 |            4 |                    2 |               4 |              1 |            5 |             2 |           10 |             2 |              11 |                5 |           1 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            2 |             1 |               2 |                2 |           0 |

##### Run: `Case 4` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.1852 |   0.4545 |     0.2632 |        11 |
| Anticipazione |      0.4667 |   0.7000 |     0.5600 |        10 |
| Disgusto      |      0.4783 |   0.5789 |     0.5238 |        19 |
| Fiducia       |      0.6222 |   0.6667 |     0.6437 |        42 |
| Gioia         |      0.6296 |   0.5862 |     0.6071 |        29 |
| Neutro        |      0.3158 |   0.3529 |     0.3333 |        17 |
| Paura         |      0.1429 |   1.0000 |     0.2500 |         1 |
| Rabbia        |      0.2000 |   0.5000 |     0.2857 |         6 |
| Sorpresa      |      0.3810 |   0.5000 |     0.4324 |        16 |
| Tristezza     |      0.2222 |   1.0000 |     0.3636 |         2 |
| micro avg     |      0.4231 |   0.5752 |     0.4875 |       153 |
| macro avg     |      0.3644 |   0.6339 |     0.4263 |       153 |
| weighted avg  |      0.4800 |   0.5752 |     0.5122 |       153 |
| samples avg   |      0.4748 |   0.5840 |     0.4969 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |   96 |   22 |    6 |    5 |      0.1852 |   0.4545 |     0.2632 |        11 |
| Anticipazione |  111 |    8 |    3 |    7 |      0.4667 |   0.7000 |     0.5600 |        10 |
| Disgusto      |   98 |   12 |    8 |   11 |      0.4783 |   0.5789 |     0.5238 |        19 |
| Fiducia       |   70 |   17 |   14 |   28 |      0.6222 |   0.6667 |     0.6437 |        42 |
| Gioia         |   90 |   10 |   12 |   17 |      0.6296 |   0.5862 |     0.6071 |        29 |
| Neutro        |   99 |   13 |   11 |    6 |      0.3158 |   0.3529 |     0.3333 |        17 |
| Paura         |  122 |    6 |    0 |    1 |      0.1429 |   1.0000 |     0.2500 |         1 |
| Rabbia        |  111 |   12 |    3 |    3 |      0.2000 |   0.5000 |     0.2857 |         6 |
| Sorpresa      |  100 |   13 |    8 |    8 |      0.3810 |   0.5000 |     0.4324 |        16 |
| Tristezza     |  120 |    7 |    0 |    2 |      0.2222 |   1.0000 |     0.3636 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            5 |                    0 |               0 |              6 |            3 |             1 |            2 |             0 |               2 |                0 |           0 |
| Anticipazione  |             10 |            2 |                    7 |               0 |              1 |            3 |             0 |            0 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            2 |                    1 |              11 |              1 |            0 |             1 |            1 |             5 |               3 |                1 |           1 |
| Fiducia        |             42 |            5 |                    4 |               1 |             28 |           11 |             7 |            0 |             0 |               5 |                0 |           2 |
| Gioia          |             29 |            8 |                    2 |               5 |              5 |           17 |             4 |            1 |             3 |               8 |                3 |           2 |
| Neutro         |             17 |            5 |                    2 |               1 |              6 |            0 |             6 |            0 |             4 |               1 |                0 |           2 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             1 |               1 |                1 |           0 |
| Rabbia         |              6 |            0 |                    0 |               3 |              0 |            0 |             0 |            1 |             3 |               1 |                2 |           0 |
| Sorpresa       |             16 |            3 |                    0 |               4 |              4 |            4 |             1 |            2 |             2 |               8 |                3 |           1 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             1 |               2 |                2 |           0 |

#### Case 5: Step 2 Oracle (Train: +rel, Eval: no rel) (Checkpoint: `emotion_textual_corr_with_rel`, Eval Prompt: `Oracle (no rel)`)

| Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Blind Text (ClassifierChain)     | raw       |     0.4285 |     0.4840 |        0.4956 |        0.3023 |
| Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4821 |     0.5062 |        0.5206 |        0.3566 |
| Blind Text (ClassifierChain)     | std       |     0.4390 |     0.4886 |        0.5178 |        0.3488 |
| Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4424 |     0.4945 |        0.5440 |        0.2868 |

##### Run: `Case 5` | `Blind Text (ClassifierChain)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.4211 |   0.7273 |     0.5333 |        11 |
| Anticipazione |      0.4375 |   0.7000 |     0.5385 |        10 |
| Disgusto      |      0.4615 |   0.6316 |     0.5333 |        19 |
| Fiducia       |      0.6522 |   0.7143 |     0.6818 |        42 |
| Gioia         |      0.4211 |   0.2759 |     0.3333 |        29 |
| Neutro        |      0.4545 |   0.2941 |     0.3571 |        17 |
| Paura         |      0.0526 |   1.0000 |     0.1000 |         1 |
| Rabbia        |      0.3000 |   0.5000 |     0.3750 |         6 |
| Sorpresa      |      0.3810 |   0.5000 |     0.4324 |        16 |
| Tristezza     |      0.3333 |   0.5000 |     0.4000 |         2 |
| micro avg     |      0.4368 |   0.5425 |     0.4840 |       153 |
| macro avg     |      0.3915 |   0.5843 |     0.4285 |       153 |
| weighted avg  |      0.4818 |   0.5425 |     0.4956 |       153 |
| samples avg   |      0.4496 |   0.5465 |     0.4765 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  107 |   11 |    3 |    8 |      0.4211 |   0.7273 |     0.5333 |        11 |
| Anticipazione |  110 |    9 |    3 |    7 |      0.4375 |   0.7000 |     0.5385 |        10 |
| Disgusto      |   96 |   14 |    7 |   12 |      0.4615 |   0.6316 |     0.5333 |        19 |
| Fiducia       |   71 |   16 |   12 |   30 |      0.6522 |   0.7143 |     0.6818 |        42 |
| Gioia         |   89 |   11 |   21 |    8 |      0.4211 |   0.2759 |     0.3333 |        29 |
| Neutro        |  106 |    6 |   12 |    5 |      0.4545 |   0.2941 |     0.3571 |        17 |
| Paura         |  110 |   18 |    0 |    1 |      0.0526 |   1.0000 |     0.1000 |         1 |
| Rabbia        |  116 |    7 |    3 |    3 |      0.3000 |   0.5000 |     0.3750 |         6 |
| Sorpresa      |  100 |   13 |    8 |    8 |      0.3810 |   0.5000 |     0.4324 |        16 |
| Tristezza     |  125 |    2 |    1 |    1 |      0.3333 |   0.5000 |     0.4000 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            8 |                    0 |               0 |              7 |            1 |             0 |            1 |             0 |               1 |                0 |           0 |
| Anticipazione  |             10 |            2 |                    7 |               1 |              1 |            3 |             0 |            2 |             0 |               2 |                0 |           0 |
| Disgusto       |             19 |            0 |                    1 |              12 |              2 |            1 |             1 |           10 |             6 |               3 |                0 |           1 |
| Fiducia        |             42 |            7 |                    2 |               1 |             30 |            7 |             0 |            2 |             0 |               5 |                0 |           4 |
| Gioia          |             29 |            5 |                    5 |               5 |              6 |            8 |             2 |            1 |             0 |               9 |                1 |           2 |
| Neutro         |             17 |            2 |                    1 |               4 |              3 |            2 |             5 |            0 |             1 |               1 |                0 |           1 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             1 |               0 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               3 |              2 |            1 |             1 |            4 |             3 |               1 |                1 |           0 |
| Sorpresa       |             16 |            0 |                    1 |               3 |              4 |            3 |             2 |            3 |             1 |               8 |                2 |           1 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            2 |             1 |               2 |                1 |           0 |

##### Run: `Case 5` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3333 |   0.3636 |     0.3478 |        11 |
| Anticipazione |      0.8000 |   0.4000 |     0.5333 |        10 |
| Disgusto      |      0.6667 |   0.4211 |     0.5161 |        19 |
| Fiducia       |      0.6750 |   0.6429 |     0.6585 |        42 |
| Gioia         |      0.6000 |   0.6207 |     0.6102 |        29 |
| Neutro        |      0.3043 |   0.4118 |     0.3500 |        17 |
| Paura         |      0.3333 |   1.0000 |     0.5000 |         1 |
| Rabbia        |      0.3333 |   0.5000 |     0.4000 |         6 |
| Sorpresa      |      0.2500 |   0.5000 |     0.3333 |        16 |
| Tristezza     |      0.4000 |   1.0000 |     0.5714 |         2 |
| micro avg     |      0.4795 |   0.5359 |     0.5062 |       153 |
| macro avg     |      0.4696 |   0.5860 |     0.4821 |       153 |
| weighted avg  |      0.5385 |   0.5359 |     0.5206 |       153 |
| samples avg   |      0.4886 |   0.5336 |     0.4943 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  110 |    8 |    7 |    4 |      0.3333 |   0.3636 |     0.3478 |        11 |
| Anticipazione |  118 |    1 |    6 |    4 |      0.8000 |   0.4000 |     0.5333 |        10 |
| Disgusto      |  106 |    4 |   11 |    8 |      0.6667 |   0.4211 |     0.5161 |        19 |
| Fiducia       |   74 |   13 |   15 |   27 |      0.6750 |   0.6429 |     0.6585 |        42 |
| Gioia         |   88 |   12 |   11 |   18 |      0.6000 |   0.6207 |     0.6102 |        29 |
| Neutro        |   96 |   16 |   10 |    7 |      0.3043 |   0.4118 |     0.3500 |        17 |
| Paura         |  126 |    2 |    0 |    1 |      0.3333 |   1.0000 |     0.5000 |         1 |
| Rabbia        |  117 |    6 |    3 |    3 |      0.3333 |   0.5000 |     0.4000 |         6 |
| Sorpresa      |   89 |   24 |    8 |    8 |      0.2500 |   0.5000 |     0.3333 |        16 |
| Tristezza     |  124 |    3 |    0 |    2 |      0.4000 |   1.0000 |     0.5714 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            4 |                    0 |               0 |              6 |            2 |             3 |            0 |             0 |               2 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    4 |               0 |              1 |            5 |             1 |            0 |             0 |               2 |                0 |           2 |
| Disgusto       |             19 |            0 |                    1 |               8 |              3 |            0 |             2 |            0 |             3 |               6 |                0 |           1 |
| Fiducia        |             42 |            3 |                    0 |               0 |             27 |            9 |             2 |            0 |             0 |               8 |                0 |           6 |
| Gioia          |             29 |            7 |                    0 |               0 |              4 |           18 |             6 |            0 |             3 |              11 |                1 |           1 |
| Neutro         |             17 |            1 |                    0 |               1 |              3 |            4 |             7 |            0 |             0 |               3 |                0 |           2 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               1 |                1 |           0 |
| Rabbia         |              6 |            0 |                    0 |               3 |              0 |            0 |             0 |            1 |             3 |               0 |                1 |           0 |
| Sorpresa       |             16 |            1 |                    0 |               0 |              1 |            5 |             5 |            2 |             2 |               8 |                3 |           1 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             1 |               2 |                2 |           0 |

##### Run: `Case 5` | `Blind Text (ClassifierChain)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3103 |   0.8182 |     0.4500 |        11 |
| Anticipazione |      0.5000 |   0.5000 |     0.5000 |        10 |
| Disgusto      |      0.7059 |   0.6316 |     0.6667 |        19 |
| Fiducia       |      0.6486 |   0.5714 |     0.6076 |        42 |
| Gioia         |      0.4583 |   0.3793 |     0.4151 |        29 |
| Neutro        |      0.3600 |   0.5294 |     0.4286 |        17 |
| Paura         |      0.0625 |   1.0000 |     0.1176 |         1 |
| Rabbia        |      0.2500 |   0.6667 |     0.3636 |         6 |
| Sorpresa      |      0.5000 |   0.6250 |     0.5556 |        16 |
| Tristezza     |      0.2000 |   0.5000 |     0.2857 |         2 |
| micro avg     |      0.4322 |   0.5621 |     0.4886 |       153 |
| macro avg     |      0.3996 |   0.6222 |     0.4390 |       153 |
| weighted avg  |      0.5127 |   0.5621 |     0.5178 |       153 |
| samples avg   |      0.4942 |   0.5775 |     0.5114 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |   98 |   20 |    2 |    9 |      0.3103 |   0.8182 |     0.4500 |        11 |
| Anticipazione |  114 |    5 |    5 |    5 |      0.5000 |   0.5000 |     0.5000 |        10 |
| Disgusto      |  105 |    5 |    7 |   12 |      0.7059 |   0.6316 |     0.6667 |        19 |
| Fiducia       |   74 |   13 |   18 |   24 |      0.6486 |   0.5714 |     0.6076 |        42 |
| Gioia         |   87 |   13 |   18 |   11 |      0.4583 |   0.3793 |     0.4151 |        29 |
| Neutro        |   96 |   16 |    8 |    9 |      0.3600 |   0.5294 |     0.4286 |        17 |
| Paura         |  113 |   15 |    0 |    1 |      0.0625 |   1.0000 |     0.1176 |         1 |
| Rabbia        |  111 |   12 |    2 |    4 |      0.2500 |   0.6667 |     0.3636 |         6 |
| Sorpresa      |  103 |   10 |    6 |   10 |      0.5000 |   0.6250 |     0.5556 |        16 |
| Tristezza     |  123 |    4 |    1 |    1 |      0.2000 |   0.5000 |     0.2857 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            9 |                    0 |               0 |              1 |            3 |             0 |            1 |             1 |               0 |                1 |           0 |
| Anticipazione  |             10 |            1 |                    5 |               0 |              2 |            2 |             1 |            1 |             2 |               1 |                0 |           0 |
| Disgusto       |             19 |            0 |                    1 |              12 |              2 |            1 |             2 |            2 |             6 |               4 |                0 |           1 |
| Fiducia        |             42 |           12 |                    2 |               0 |             24 |           10 |             3 |            3 |             2 |               3 |                1 |           3 |
| Gioia          |             29 |            6 |                    1 |               3 |              6 |           11 |             6 |            2 |             2 |               5 |                1 |           1 |
| Neutro         |             17 |            5 |                    1 |               0 |              3 |            2 |             9 |            3 |             0 |               1 |                0 |           0 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             1 |               0 |                1 |           0 |
| Rabbia         |              6 |            0 |                    0 |               3 |              0 |            1 |             1 |            2 |             4 |               1 |                1 |           0 |
| Sorpresa       |             16 |            1 |                    0 |               2 |              3 |            3 |             5 |            3 |             4 |              10 |                2 |           1 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            2 |             2 |               2 |                1 |           0 |

##### Run: `Case 5` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3636 |   0.7273 |     0.4848 |        11 |
| Anticipazione |      0.4545 |   0.5000 |     0.4762 |        10 |
| Disgusto      |      0.6471 |   0.5789 |     0.6111 |        19 |
| Fiducia       |      0.6744 |   0.6905 |     0.6824 |        42 |
| Gioia         |      0.5806 |   0.6207 |     0.6000 |        29 |
| Neutro        |      0.2593 |   0.4118 |     0.3182 |        17 |
| Paura         |      0.0345 |   1.0000 |     0.0667 |         1 |
| Rabbia        |      0.6000 |   0.5000 |     0.5455 |         6 |
| Sorpresa      |      0.3500 |   0.4375 |     0.3889 |        16 |
| Tristezza     |      0.1667 |   0.5000 |     0.2500 |         2 |
| micro avg     |      0.4265 |   0.5882 |     0.4945 |       153 |
| macro avg     |      0.4131 |   0.5967 |     0.4424 |       153 |
| weighted avg  |      0.5227 |   0.5882 |     0.5440 |       153 |
| samples avg   |      0.4673 |   0.5982 |     0.4997 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  104 |   14 |    3 |    8 |      0.3636 |   0.7273 |     0.4848 |        11 |
| Anticipazione |  113 |    6 |    5 |    5 |      0.4545 |   0.5000 |     0.4762 |        10 |
| Disgusto      |  104 |    6 |    8 |   11 |      0.6471 |   0.5789 |     0.6111 |        19 |
| Fiducia       |   73 |   14 |   13 |   29 |      0.6744 |   0.6905 |     0.6824 |        42 |
| Gioia         |   87 |   13 |   11 |   18 |      0.5806 |   0.6207 |     0.6000 |        29 |
| Neutro        |   92 |   20 |   10 |    7 |      0.2593 |   0.4118 |     0.3182 |        17 |
| Paura         |  100 |   28 |    0 |    1 |      0.0345 |   1.0000 |     0.0667 |         1 |
| Rabbia        |  121 |    2 |    3 |    3 |      0.6000 |   0.5000 |     0.5455 |         6 |
| Sorpresa      |  100 |   13 |    9 |    7 |      0.3500 |   0.4375 |     0.3889 |        16 |
| Tristezza     |  122 |    5 |    1 |    1 |      0.1667 |   0.5000 |     0.2500 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            8 |                    0 |               0 |              8 |            3 |             0 |            3 |             0 |               1 |                1 |           0 |
| Anticipazione  |             10 |            1 |                    5 |               0 |              1 |            3 |             0 |            2 |             0 |               2 |                0 |           1 |
| Disgusto       |             19 |            1 |                    1 |              11 |              3 |            0 |             3 |            1 |             3 |               3 |                0 |           0 |
| Fiducia        |             42 |            7 |                    2 |               0 |             29 |           10 |             3 |            8 |             0 |               6 |                1 |           2 |
| Gioia          |             29 |            6 |                    2 |               2 |              5 |           18 |             8 |            6 |             0 |               6 |                1 |           1 |
| Neutro         |             17 |            3 |                    1 |               0 |              2 |            3 |             7 |            5 |             0 |               1 |                0 |           1 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               1 |                1 |           0 |
| Rabbia         |              6 |            0 |                    0 |               3 |              0 |            1 |             1 |            2 |             3 |               1 |                2 |           0 |
| Sorpresa       |             16 |            1 |                    1 |               2 |              1 |            4 |             7 |            4 |             0 |               7 |                2 |           1 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             0 |               1 |                1 |           1 |

#### Case 6: Step 2 Oracle (Train: +rel, Eval: +rel) (Checkpoint: `emotion_textual_corr_with_rel`, Eval Prompt: `Oracle (+ rel)`)

| Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Blind Text (ClassifierChain)     | raw       |     0.4700 |     0.5252 |        0.5539 |        0.2558 |
| Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4688 |     0.5401 |        0.5568 |        0.3798 |
| Blind Text (ClassifierChain)     | std       |     0.4572 |     0.5000 |        0.5603 |        0.2403 |
| Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4553 |     0.5100 |        0.5467 |        0.2946 |

##### Run: `Case 6` | `Blind Text (ClassifierChain)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2857 |   0.7273 |     0.4103 |        11 |
| Anticipazione |      0.2800 |   0.7000 |     0.4000 |        10 |
| Disgusto      |      0.7895 |   0.7895 |     0.7895 |        19 |
| Fiducia       |      0.6444 |   0.6905 |     0.6667 |        42 |
| Gioia         |      0.4634 |   0.6552 |     0.5429 |        29 |
| Neutro        |      0.4545 |   0.2941 |     0.3571 |        17 |
| Paura         |      0.1000 |   1.0000 |     0.1818 |         1 |
| Rabbia        |      0.1579 |   0.5000 |     0.2400 |         6 |
| Sorpresa      |      0.4762 |   0.6250 |     0.5405 |        16 |
| Tristezza     |      0.4000 |   1.0000 |     0.5714 |         2 |
| micro avg     |      0.4420 |   0.6471 |     0.5252 |       153 |
| macro avg     |      0.4052 |   0.6982 |     0.4700 |       153 |
| weighted avg  |      0.5140 |   0.6471 |     0.5539 |       153 |
| samples avg   |      0.4729 |   0.6550 |     0.5253 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |   98 |   20 |    3 |    8 |      0.2857 |   0.7273 |     0.4103 |        11 |
| Anticipazione |  101 |   18 |    3 |    7 |      0.2800 |   0.7000 |     0.4000 |        10 |
| Disgusto      |  106 |    4 |    4 |   15 |      0.7895 |   0.7895 |     0.7895 |        19 |
| Fiducia       |   71 |   16 |   13 |   29 |      0.6444 |   0.6905 |     0.6667 |        42 |
| Gioia         |   78 |   22 |   10 |   19 |      0.4634 |   0.6552 |     0.5429 |        29 |
| Neutro        |  106 |    6 |   12 |    5 |      0.4545 |   0.2941 |     0.3571 |        17 |
| Paura         |  119 |    9 |    0 |    1 |      0.1000 |   1.0000 |     0.1818 |         1 |
| Rabbia        |  107 |   16 |    3 |    3 |      0.1579 |   0.5000 |     0.2400 |         6 |
| Sorpresa      |  102 |   11 |    6 |   10 |      0.4762 |   0.6250 |     0.5405 |        16 |
| Tristezza     |  124 |    3 |    0 |    2 |      0.4000 |   1.0000 |     0.5714 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            8 |                    0 |               0 |              5 |            7 |             0 |            1 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            2 |                    7 |               0 |              2 |            5 |             0 |            2 |             0 |               2 |                0 |           0 |
| Disgusto       |             19 |            1 |                    2 |              15 |              1 |            0 |             2 |            3 |            14 |               2 |                0 |           0 |
| Fiducia        |             42 |           10 |                    5 |               0 |             29 |           13 |             0 |            2 |             0 |               4 |                0 |           2 |
| Gioia          |             29 |            7 |                    8 |               1 |              7 |           19 |             2 |            0 |             0 |               8 |                1 |           2 |
| Neutro         |             17 |            4 |                    4 |               0 |              4 |            4 |             5 |            0 |             1 |               1 |                0 |           2 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               1 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               3 |              1 |            0 |             1 |            1 |             3 |               1 |                2 |           0 |
| Sorpresa       |             16 |            1 |                    3 |               1 |              3 |            4 |             3 |            3 |             2 |              10 |                3 |           0 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            2 |             2 |               2 |                2 |           0 |

##### Run: `Case 6` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.1923 |   0.4545 |     0.2703 |        11 |
| Anticipazione |      0.6667 |   0.4000 |     0.5000 |        10 |
| Disgusto      |      0.7059 |   0.6316 |     0.6667 |        19 |
| Fiducia       |      0.7143 |   0.7143 |     0.7143 |        42 |
| Gioia         |      0.5250 |   0.7241 |     0.6087 |        29 |
| Neutro        |      0.3889 |   0.4118 |     0.4000 |        17 |
| Paura         |      0.3333 |   1.0000 |     0.5000 |         1 |
| Rabbia        |      0.2500 |   0.3333 |     0.2857 |         6 |
| Sorpresa      |      0.4211 |   0.5000 |     0.4571 |        16 |
| Tristezza     |      0.2000 |   0.5000 |     0.2857 |         2 |
| micro avg     |      0.4946 |   0.5948 |     0.5401 |       153 |
| macro avg     |      0.4397 |   0.5670 |     0.4688 |       153 |
| weighted avg  |      0.5425 |   0.5948 |     0.5568 |       153 |
| samples avg   |      0.5403 |   0.6059 |     0.5530 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |   97 |   21 |    6 |    5 |      0.1923 |   0.4545 |     0.2703 |        11 |
| Anticipazione |  117 |    2 |    6 |    4 |      0.6667 |   0.4000 |     0.5000 |        10 |
| Disgusto      |  105 |    5 |    7 |   12 |      0.7059 |   0.6316 |     0.6667 |        19 |
| Fiducia       |   75 |   12 |   12 |   30 |      0.7143 |   0.7143 |     0.7143 |        42 |
| Gioia         |   81 |   19 |    8 |   21 |      0.5250 |   0.7241 |     0.6087 |        29 |
| Neutro        |  101 |   11 |   10 |    7 |      0.3889 |   0.4118 |     0.4000 |        17 |
| Paura         |  126 |    2 |    0 |    1 |      0.3333 |   1.0000 |     0.5000 |         1 |
| Rabbia        |  117 |    6 |    4 |    2 |      0.2500 |   0.3333 |     0.2857 |         6 |
| Sorpresa      |  102 |   11 |    8 |    8 |      0.4211 |   0.5000 |     0.4571 |        16 |
| Tristezza     |  123 |    4 |    1 |    1 |      0.2000 |   0.5000 |     0.2857 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            5 |                    0 |               0 |              4 |            5 |             2 |            0 |             1 |               0 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    4 |               0 |              1 |            5 |             0 |            0 |             0 |               2 |                1 |           2 |
| Disgusto       |             19 |            1 |                    1 |              12 |              1 |            0 |             2 |            0 |             5 |               2 |                0 |           0 |
| Fiducia        |             42 |            7 |                    1 |               0 |             30 |           13 |             3 |            0 |             1 |               2 |                1 |           2 |
| Gioia          |             29 |            6 |                    0 |               1 |              8 |           21 |             3 |            0 |             0 |               7 |                1 |           1 |
| Neutro         |             17 |            6 |                    0 |               1 |              3 |            4 |             7 |            0 |             0 |               3 |                0 |           0 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               1 |                1 |           0 |
| Rabbia         |              6 |            2 |                    0 |               3 |              1 |            0 |             1 |            1 |             2 |               0 |                1 |           0 |
| Sorpresa       |             16 |            1 |                    0 |               0 |              2 |            5 |             3 |            2 |             1 |               8 |                2 |           0 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             1 |               1 |                1 |           0 |

##### Run: `Case 6` | `Blind Text (ClassifierChain)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3333 |   0.5455 |     0.4138 |        11 |
| Anticipazione |      0.5455 |   0.6000 |     0.5714 |        10 |
| Disgusto      |      0.6522 |   0.7895 |     0.7143 |        19 |
| Fiducia       |      0.5741 |   0.7381 |     0.6458 |        42 |
| Gioia         |      0.5667 |   0.5862 |     0.5763 |        29 |
| Neutro        |      0.3600 |   0.5294 |     0.4286 |        17 |
| Paura         |      0.0270 |   1.0000 |     0.0526 |         1 |
| Rabbia        |      0.1579 |   0.5000 |     0.2400 |         6 |
| Sorpresa      |      0.5000 |   0.5625 |     0.5294 |        16 |
| Tristezza     |      0.2500 |   1.0000 |     0.4000 |         2 |
| micro avg     |      0.4074 |   0.6471 |     0.5000 |       153 |
| macro avg     |      0.3967 |   0.6851 |     0.4572 |       153 |
| weighted avg  |      0.5075 |   0.6471 |     0.5603 |       153 |
| samples avg   |      0.4435 |   0.6537 |     0.5011 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  106 |   12 |    5 |    6 |      0.3333 |   0.5455 |     0.4138 |        11 |
| Anticipazione |  114 |    5 |    4 |    6 |      0.5455 |   0.6000 |     0.5714 |        10 |
| Disgusto      |  102 |    8 |    4 |   15 |      0.6522 |   0.7895 |     0.7143 |        19 |
| Fiducia       |   64 |   23 |   11 |   31 |      0.5741 |   0.7381 |     0.6458 |        42 |
| Gioia         |   87 |   13 |   12 |   17 |      0.5667 |   0.5862 |     0.5763 |        29 |
| Neutro        |   96 |   16 |    8 |    9 |      0.3600 |   0.5294 |     0.4286 |        17 |
| Paura         |   92 |   36 |    0 |    1 |      0.0270 |   1.0000 |     0.0526 |         1 |
| Rabbia        |  107 |   16 |    3 |    3 |      0.1579 |   0.5000 |     0.2400 |         6 |
| Sorpresa      |  104 |    9 |    7 |    9 |      0.5000 |   0.5625 |     0.5294 |        16 |
| Tristezza     |  121 |    6 |    0 |    2 |      0.2500 |   1.0000 |     0.4000 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            6 |                    0 |               0 |             10 |            3 |             4 |            1 |             0 |               1 |                1 |           0 |
| Anticipazione  |             10 |            2 |                    6 |               0 |              2 |            3 |             1 |            2 |             0 |               2 |                0 |           0 |
| Disgusto       |             19 |            0 |                    1 |              15 |              1 |            0 |             2 |           16 |            15 |               2 |                0 |           0 |
| Fiducia        |             42 |            5 |                    3 |               1 |             31 |           10 |             5 |            3 |             0 |               3 |                1 |           3 |
| Gioia          |             29 |            3 |                    1 |               1 |             12 |           17 |             3 |            5 |             0 |               6 |                2 |           3 |
| Neutro         |             17 |            4 |                    1 |               0 |              5 |            2 |             9 |            3 |             0 |               1 |                1 |           1 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               1 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               4 |              1 |            0 |             1 |            4 |             3 |               1 |                2 |           0 |
| Sorpresa       |             16 |            1 |                    0 |               4 |              3 |            4 |             3 |            8 |             2 |               9 |                3 |           1 |
| Tristezza      |              2 |            0 |                    0 |               2 |              0 |            0 |             0 |            2 |             2 |               2 |                2 |           0 |

##### Run: `Case 6` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2857 |   0.7273 |     0.4103 |        11 |
| Anticipazione |      0.5000 |   0.5000 |     0.5000 |        10 |
| Disgusto      |      0.4167 |   0.7895 |     0.5455 |        19 |
| Fiducia       |      0.6122 |   0.7143 |     0.6593 |        42 |
| Gioia         |      0.5366 |   0.7586 |     0.6286 |        29 |
| Neutro        |      0.3913 |   0.5294 |     0.4500 |        17 |
| Paura         |      0.0435 |   1.0000 |     0.0833 |         1 |
| Rabbia        |      0.7500 |   0.5000 |     0.6000 |         6 |
| Sorpresa      |      0.3182 |   0.4375 |     0.3684 |        16 |
| Tristezza     |      0.1818 |   1.0000 |     0.3077 |         2 |
| micro avg     |      0.4130 |   0.6667 |     0.5100 |       153 |
| macro avg     |      0.4036 |   0.6957 |     0.4553 |       153 |
| weighted avg  |      0.4836 |   0.6667 |     0.5467 |       153 |
| samples avg   |      0.4951 |   0.6873 |     0.5475 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |   98 |   20 |    3 |    8 |      0.2857 |   0.7273 |     0.4103 |        11 |
| Anticipazione |  114 |    5 |    5 |    5 |      0.5000 |   0.5000 |     0.5000 |        10 |
| Disgusto      |   89 |   21 |    4 |   15 |      0.4167 |   0.7895 |     0.5455 |        19 |
| Fiducia       |   68 |   19 |   12 |   30 |      0.6122 |   0.7143 |     0.6593 |        42 |
| Gioia         |   81 |   19 |    7 |   22 |      0.5366 |   0.7586 |     0.6286 |        29 |
| Neutro        |   98 |   14 |    8 |    9 |      0.3913 |   0.5294 |     0.4500 |        17 |
| Paura         |  106 |   22 |    0 |    1 |      0.0435 |   1.0000 |     0.0833 |         1 |
| Rabbia        |  122 |    1 |    3 |    3 |      0.7500 |   0.5000 |     0.6000 |         6 |
| Sorpresa      |   98 |   15 |    9 |    7 |      0.3182 |   0.4375 |     0.3684 |        16 |
| Tristezza     |  118 |    9 |    0 |    2 |      0.1818 |   1.0000 |     0.3077 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            8 |                    0 |               0 |              7 |            4 |             4 |            2 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            2 |                    5 |               0 |              2 |            6 |             2 |            3 |             0 |               1 |                0 |           0 |
| Disgusto       |             19 |            0 |                    1 |              15 |              1 |            0 |             2 |            3 |             3 |               3 |                1 |           0 |
| Fiducia        |             42 |            8 |                    2 |               1 |             30 |           14 |             3 |            5 |             0 |               5 |                1 |           1 |
| Gioia          |             29 |            8 |                    1 |               9 |              7 |           22 |             3 |            5 |             0 |               7 |                1 |           1 |
| Neutro         |             17 |            5 |                    1 |               3 |              5 |            3 |             9 |            1 |             0 |               2 |                1 |           0 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             0 |               1 |                1 |           0 |
| Rabbia         |              6 |            1 |                    0 |               4 |              1 |            0 |             1 |            4 |             3 |               0 |                2 |           0 |
| Sorpresa       |             16 |            2 |                    0 |               9 |              3 |            5 |             2 |            6 |             0 |               7 |                5 |           0 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            2 |             0 |               1 |                2 |           0 |

---

## 2. Task 2: Irony & Sarcasm Detection (3-Class Single-Label, `UmBERTo` + `ViT-224`)

- **Text Encoder**: `Musixmatch/umberto-commoncrawl-cased-v1` (`UmBERTo`, 768-d `[CLS]`)
- **Vision Encoder**: `google/vit-base-patch16-224` (`ViT-Base-224`, frozen 768-d `[CLS]`)
- **Multimodal Head (`mm_test`, $N=129$)**: `SGDClassifier(loss='squared_hinge', penalty='l2', class_weight='balanced', random_state=42)`

### 2.1 Textual Irony Benchmark (`me_test`, $N=405$)

| Regime                        | Checkpoint                  | Prompt                               |   N Test |   Macro F1 |   Micro F1 (= Acc) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:------------------------------|:----------------------------|:-------------------------------------|---------:|-----------:|-------------------:|--------------:|----------:|------------:|--------------:|
| Step 1 Base (Raw)             | irony_textual_no_corr       | Raw                                  |      405 |     0.4692 |             0.7432 |        0.7342 |    0.8449 |      0.3404 |        0.2222 |
| Step 2 Raw (Ablation)         | irony_textual_step2_raw     | Raw                                  |      405 |     0.5292 |             0.7852 |        0.7662 |    0.8726 |      0.3817 |        0.3333 |
| Step 2 Oracle (Train: no rel) | irony_textual_corr          | Oracle (Train: no rel, Eval: no rel) |      405 |     0.5146 |             0.7506 |        0.7551 |    0.8485 |      0.4286 |        0.2667 |
| Step 2 Oracle (Train: +rel)   | irony_textual_corr_with_rel | Oracle (Train: +rel, Eval: no rel)   |      405 |     0.5011 |             0.7630 |        0.7533 |    0.8589 |      0.3776 |        0.2667 |

#### Textual Case: `Step 1 Base (Raw)` (`irony_textual_no_corr`)
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8258 |   0.8648 |     0.8449 |       318 |
| Ironia       |      0.3871 |   0.3038 |     0.3404 |        79 |
| Sarcasmo     |      0.2000 |   0.2500 |     0.2222 |         8 |
| macro avg    |      0.4710 |   0.4729 |     0.4692 |       405 |
| weighted avg |      0.7279 |   0.7432 |     0.7342 |       405 |
| accuracy     |      0.7432 |   0.7432 |     0.7432 |       405 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |         275 |            37 |               6 |          318 |
| Ironia       |          53 |            24 |               2 |           79 |
| Sarcasmo     |           5 |             1 |               2 |            8 |

#### Textual Case: `Step 2 Raw (Ablation)` (`irony_textual_step2_raw`)
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8338 |   0.9151 |     0.8726 |       318 |
| Ironia       |      0.4808 |   0.3165 |     0.3817 |        79 |
| Sarcasmo     |      0.5000 |   0.2500 |     0.3333 |         8 |
| macro avg    |      0.6049 |   0.4939 |     0.5292 |       405 |
| weighted avg |      0.7584 |   0.7852 |     0.7662 |       405 |
| accuracy     |      0.7852 |   0.7852 |     0.7852 |       405 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |         291 |            25 |               2 |          318 |
| Ironia       |          54 |            25 |               0 |           79 |
| Sarcasmo     |           4 |             2 |               2 |            8 |

#### Textual Case: `Step 2 Oracle (Train: no rel)` (`irony_textual_corr`)
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8608 |   0.8365 |     0.8485 |       318 |
| Ironia       |      0.4045 |   0.4557 |     0.4286 |        79 |
| Sarcasmo     |      0.2857 |   0.2500 |     0.2667 |         8 |
| macro avg    |      0.5170 |   0.5141 |     0.5146 |       405 |
| weighted avg |      0.7605 |   0.7506 |     0.7551 |       405 |
| accuracy     |      0.7506 |   0.7506 |     0.7506 |       405 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |         266 |            49 |               3 |          318 |
| Ironia       |          41 |            36 |               2 |           79 |
| Sarcasmo     |           2 |             4 |               2 |            8 |

#### Textual Case: `Step 2 Oracle (Train: +rel)` (`irony_textual_corr_with_rel`)
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8383 |   0.8805 |     0.8589 |       318 |
| Ironia       |      0.4219 |   0.3418 |     0.3776 |        79 |
| Sarcasmo     |      0.2857 |   0.2500 |     0.2667 |         8 |
| macro avg    |      0.5153 |   0.4908 |     0.5011 |       405 |
| weighted avg |      0.7462 |   0.7630 |     0.7533 |       405 |
| accuracy     |      0.7630 |   0.7630 |     0.7630 |       405 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |         280 |            35 |               3 |          318 |
| Ironia       |          50 |            27 |               2 |           79 |
| Sarcasmo     |           4 |             2 |               2 |            8 |

---

### 2.2 Multimodal Irony Benchmark (`mm_test`, $N=129$ — All 6 Cases $\times$ `raw` vs `std`)

#### Master Comparison Table (All Multimodal Irony Runs)
| Case   | Regime                                      | Train Relation   | Eval Relation   | Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:-------|:--------------------------------------------|:-----------------|:----------------|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Blind Text (SGD, raw)                 | raw       |     0.4377 |              0.7752 |        0.7554 |    0.8796 |      0.2667 |        0.1667 |
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5334 |              0.8140 |        0.7973 |    0.8930 |      0.4571 |        0.2500 |
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Blind Text (SGD, std)                 | std       |     0.5112 |              0.7132 |        0.7280 |    0.8259 |      0.2791 |        0.4286 |
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.4318 |              0.7907 |        0.7699 |    0.8837 |      0.4118 |        0.0000 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Blind Text (SGD, raw)                 | raw       |     0.4788 |              0.7752 |        0.7690 |    0.8762 |      0.3784 |        0.1818 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5196 |              0.8062 |        0.7932 |    0.8920 |      0.4444 |        0.2222 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Blind Text (SGD, std)                 | std       |     0.4603 |              0.7519 |        0.7618 |    0.8780 |      0.3125 |        0.1905 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.5029 |              0.7829 |        0.7757 |    0.8762 |      0.4103 |        0.2222 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Blind Text (SGD, raw)                 | raw       |     0.4642 |              0.7209 |        0.7432 |    0.8586 |      0.2581 |        0.2759 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5267 |              0.7674 |        0.7783 |    0.8687 |      0.4615 |        0.2500 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Blind Text (SGD, std)                 | std       |     0.4983 |              0.7674 |        0.7713 |    0.8725 |      0.4000 |        0.2222 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.4722 |              0.7519 |        0.7573 |    0.8627 |      0.3721 |        0.1818 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Blind Text (SGD, raw)                 | raw       |     0.5052 |              0.7132 |        0.7421 |    0.8280 |      0.4375 |        0.2500 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5374 |              0.8062 |        0.7993 |    0.8900 |      0.5000 |        0.2222 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Blind Text (SGD, std)                 | std       |     0.5013 |              0.7597 |        0.7637 |    0.8627 |      0.3913 |        0.2500 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.5179 |              0.8140 |        0.7939 |    0.8940 |      0.4375 |        0.2222 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Blind Text (SGD, raw)                 | raw       |     0.3846 |              0.7442 |        0.7352 |    0.8612 |      0.2927 |        0.0000 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.3634 |              0.7209 |        0.7162 |    0.8462 |      0.2439 |        0.0000 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Blind Text (SGD, std)                 | std       |     0.3617 |              0.5039 |        0.5809 |    0.6506 |      0.3846 |        0.0500 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.3805 |              0.7209 |        0.7251 |    0.8488 |      0.2927 |        0.0000 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Blind Text (SGD, raw)                 | raw       |     0.4424 |              0.7674 |        0.7669 |    0.8725 |      0.4545 |        0.0000 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.4186 |              0.7752 |        0.7601 |    0.8774 |      0.3784 |        0.0000 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Blind Text (SGD, std)                 | std       |     0.4314 |              0.7132 |        0.7206 |    0.8333 |      0.2791 |        0.1818 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.4304 |              0.7519 |        0.7573 |    0.8657 |      0.4255 |        0.0000 |

#### Semiotic Subsets Breakdown (`ADD`, `PAR`, `DIV` Accuracy / Micro F1 on `mm_test`, $N=129$)
| Case   | Regime                                      | Model                                 | Scaling   |   ADD (N=68) |   PAR (N=23) |   DIV (N=38) |   Macro F1 |   Micro F1 (= Acc.) |
|:-------|:--------------------------------------------|:--------------------------------------|:----------|-------------:|-------------:|-------------:|-----------:|--------------------:|
| Case 1 | Step 1 Base (No corr)                       | Blind Text (SGD, raw)                 | raw       |       0.8676 |       1.0000 |       0.4737 |     0.4377 |              0.7752 |
| Case 1 | Step 1 Base (No corr)                       | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.8824 |       0.9565 |       0.6053 |     0.5334 |              0.8140 |
| Case 1 | Step 1 Base (No corr)                       | Blind Text (SGD, std)                 | std       |       0.7500 |       0.9565 |       0.5000 |     0.5112 |              0.7132 |
| Case 1 | Step 1 Base (No corr)                       | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.8676 |       0.9565 |       0.5526 |     0.4318 |              0.7907 |
| Case 2 | Step 2 Raw (No corr)                        | Blind Text (SGD, raw)                 | raw       |       0.8529 |       0.9130 |       0.5526 |     0.4788 |              0.7752 |
| Case 2 | Step 2 Raw (No corr)                        | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.8529 |       0.9130 |       0.6579 |     0.5196 |              0.8062 |
| Case 2 | Step 2 Raw (No corr)                        | Blind Text (SGD, std)                 | std       |       0.8382 |       0.9130 |       0.5000 |     0.4603 |              0.7519 |
| Case 2 | Step 2 Raw (No corr)                        | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.8382 |       0.9130 |       0.6053 |     0.5029 |              0.7829 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (SGD, raw)                 | raw       |       0.8088 |       0.9130 |       0.4474 |     0.4642 |              0.7209 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.8382 |       0.8696 |       0.5789 |     0.5267 |              0.7674 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (SGD, std)                 | std       |       0.8382 |       0.9565 |       0.5263 |     0.4983 |              0.7674 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.8529 |       0.9130 |       0.4737 |     0.4722 |              0.7519 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Blind Text (SGD, raw)                 | raw       |       0.7500 |       0.9565 |       0.5000 |     0.5052 |              0.7132 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.8529 |       0.9565 |       0.6316 |     0.5374 |              0.8062 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Blind Text (SGD, std)                 | std       |       0.8235 |       1.0000 |       0.5000 |     0.5013 |              0.7597 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.8676 |       1.0000 |       0.6053 |     0.5179 |              0.8140 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Blind Text (SGD, raw)                 | raw       |       0.8088 |       0.9130 |       0.5263 |     0.3846 |              0.7442 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.8088 |       0.9130 |       0.4474 |     0.3634 |              0.7209 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Blind Text (SGD, std)                 | std       |       0.5441 |       0.5652 |       0.3947 |     0.3617 |              0.5039 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.7794 |       0.9130 |       0.5000 |     0.3805 |              0.7209 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Blind Text (SGD, raw)                 | raw       |       0.8382 |       0.9130 |       0.5526 |     0.4424 |              0.7674 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.8676 |       0.9130 |       0.5263 |     0.4186 |              0.7752 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Blind Text (SGD, std)                 | std       |       0.7941 |       0.7826 |       0.5263 |     0.4314 |              0.7132 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.8088 |       0.9130 |       0.5526 |     0.4304 |              0.7519 |

### 2.3 Multimodal Irony Detailed Reports by Case (`Case 1` – `Case 6`)

#### Case 1: Step 1 Base (No corr) (Checkpoint: `irony_textual_no_corr`, Eval Prompt: `None (Raw)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Blind Text (SGD, raw)                 | raw       |     0.4377 |              0.7752 |        0.7554 |    0.8796 |      0.2667 |        0.1667 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5334 |              0.8140 |        0.7973 |    0.8930 |      0.4571 |        0.2500 |
| Blind Text (SGD, std)                 | std       |     0.5112 |              0.7132 |        0.7280 |    0.8259 |      0.2791 |        0.4286 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.4318 |              0.7907 |        0.7699 |    0.8837 |      0.4118 |        0.0000 |

##### Run: `Case 1` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8482 |   0.9135 |     0.8796 |       104 |
| Ironia       |      0.3333 |   0.2222 |     0.2667 |        18 |
| Sarcasmo     |      0.2000 |   0.1429 |     0.1667 |         7 |
| macro avg    |      0.4605 |   0.4262 |     0.4377 |       129 |
| weighted avg |      0.7412 |   0.7752 |     0.7554 |       129 |
| accuracy     |      0.7752 |   0.7752 |     0.7752 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          95 |             6 |               3 |          104 |
| Ironia       |          13 |             4 |               1 |           18 |
| Sarcasmo     |           4 |             2 |               1 |            7 |

##### Run: `Case 1` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8649 |   0.9231 |     0.8930 |       104 |
| Ironia       |      0.4706 |   0.4444 |     0.4571 |        18 |
| Sarcasmo     |      1.0000 |   0.1429 |     0.2500 |         7 |
| macro avg    |      0.7785 |   0.5035 |     0.5334 |       129 |
| weighted avg |      0.8172 |   0.8140 |     0.7973 |       129 |
| accuracy     |      0.8140 |   0.8140 |     0.8140 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          96 |             8 |               0 |          104 |
| Ironia       |          10 |             8 |               0 |           18 |
| Sarcasmo     |           5 |             1 |               1 |            7 |

##### Run: `Case 1` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8557 |   0.7981 |     0.8259 |       104 |
| Ironia       |      0.2400 |   0.3333 |     0.2791 |        18 |
| Sarcasmo     |      0.4286 |   0.4286 |     0.4286 |         7 |
| macro avg    |      0.5081 |   0.5200 |     0.5112 |       129 |
| weighted avg |      0.7466 |   0.7132 |     0.7280 |       129 |
| accuracy     |      0.7132 |   0.7132 |     0.7132 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          83 |            17 |               4 |          104 |
| Ironia       |          12 |             6 |               0 |           18 |
| Sarcasmo     |           2 |             2 |               3 |            7 |

##### Run: `Case 1` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8559 |   0.9135 |     0.8837 |       104 |
| Ironia       |      0.4375 |   0.3889 |     0.4118 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.4311 |   0.4341 |     0.4318 |       129 |
| weighted avg |      0.7510 |   0.7907 |     0.7699 |       129 |
| accuracy     |      0.7907 |   0.7907 |     0.7907 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          95 |             7 |               2 |          104 |
| Ironia       |          11 |             7 |               0 |           18 |
| Sarcasmo     |           5 |             2 |               0 |            7 |

#### Case 2: Step 2 Raw (No corr) (Checkpoint: `irony_textual_step2_raw`, Eval Prompt: `None (Raw)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Blind Text (SGD, raw)                 | raw       |     0.4788 |              0.7752 |        0.7690 |    0.8762 |      0.3784 |        0.1818 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5196 |              0.8062 |        0.7932 |    0.8920 |      0.4444 |        0.2222 |
| Blind Text (SGD, std)                 | std       |     0.4603 |              0.7519 |        0.7618 |    0.8780 |      0.3125 |        0.1905 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.5029 |              0.7829 |        0.7757 |    0.8762 |      0.4103 |        0.2222 |

##### Run: `Case 2` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8679 |   0.8846 |     0.8762 |       104 |
| Ironia       |      0.3684 |   0.3889 |     0.3784 |        18 |
| Sarcasmo     |      0.2500 |   0.1429 |     0.1818 |         7 |
| macro avg    |      0.4954 |   0.4721 |     0.4788 |       129 |
| weighted avg |      0.7647 |   0.7752 |     0.7690 |       129 |
| accuracy     |      0.7752 |   0.7752 |     0.7752 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          92 |            10 |               2 |          104 |
| Ironia       |          10 |             7 |               1 |           18 |
| Sarcasmo     |           4 |             2 |               1 |            7 |

##### Run: `Case 2` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8716 |   0.9135 |     0.8920 |       104 |
| Ironia       |      0.4444 |   0.4444 |     0.4444 |        18 |
| Sarcasmo     |      0.5000 |   0.1429 |     0.2222 |         7 |
| macro avg    |      0.6053 |   0.5003 |     0.5196 |       129 |
| weighted avg |      0.7918 |   0.8062 |     0.7932 |       129 |
| accuracy     |      0.8062 |   0.8062 |     0.8062 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          95 |             8 |               1 |          104 |
| Ironia       |          10 |             8 |               0 |           18 |
| Sarcasmo     |           4 |             2 |               1 |            7 |

##### Run: `Case 2` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8911 |   0.8654 |     0.8780 |       104 |
| Ironia       |      0.3571 |   0.2778 |     0.3125 |        18 |
| Sarcasmo     |      0.1429 |   0.2857 |     0.1905 |         7 |
| macro avg    |      0.4637 |   0.4763 |     0.4603 |       129 |
| weighted avg |      0.7760 |   0.7519 |     0.7618 |       129 |
| accuracy     |      0.7519 |   0.7519 |     0.7519 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          90 |             8 |               6 |          104 |
| Ironia       |           7 |             5 |               6 |           18 |
| Sarcasmo     |           4 |             1 |               2 |            7 |

##### Run: `Case 2` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8679 |   0.8846 |     0.8762 |       104 |
| Ironia       |      0.3810 |   0.4444 |     0.4103 |        18 |
| Sarcasmo     |      0.5000 |   0.1429 |     0.2222 |         7 |
| macro avg    |      0.5830 |   0.4906 |     0.5029 |       129 |
| weighted avg |      0.7800 |   0.7829 |     0.7757 |       129 |
| accuracy     |      0.7829 |   0.7829 |     0.7829 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          92 |            11 |               1 |          104 |
| Ironia       |          10 |             8 |               0 |           18 |
| Sarcasmo     |           4 |             2 |               1 |            7 |

#### Case 3: Step 2 Oracle (Train: no rel, Eval: no rel) (Checkpoint: `irony_textual_corr`, Eval Prompt: `Oracle (no rel)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Blind Text (SGD, raw)                 | raw       |     0.4642 |              0.7209 |        0.7432 |    0.8586 |      0.2581 |        0.2759 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5267 |              0.7674 |        0.7783 |    0.8687 |      0.4615 |        0.2500 |
| Blind Text (SGD, std)                 | std       |     0.4983 |              0.7674 |        0.7713 |    0.8725 |      0.4000 |        0.2222 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.4722 |              0.7519 |        0.7573 |    0.8627 |      0.3721 |        0.1818 |

##### Run: `Case 3` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.9043 |   0.8173 |     0.8586 |       104 |
| Ironia       |      0.3077 |   0.2222 |     0.2581 |        18 |
| Sarcasmo     |      0.1818 |   0.5714 |     0.2759 |         7 |
| macro avg    |      0.4646 |   0.5370 |     0.4642 |       129 |
| weighted avg |      0.7818 |   0.7209 |     0.7432 |       129 |
| accuracy     |      0.7209 |   0.7209 |     0.7209 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          85 |             9 |              10 |          104 |
| Ironia       |           6 |             4 |               8 |           18 |
| Sarcasmo     |           3 |             0 |               4 |            7 |

##### Run: `Case 3` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.9149 |   0.8269 |     0.8687 |       104 |
| Ironia       |      0.3529 |   0.6667 |     0.4615 |        18 |
| Sarcasmo     |      1.0000 |   0.1429 |     0.2500 |         7 |
| macro avg    |      0.7559 |   0.5455 |     0.5267 |       129 |
| weighted avg |      0.8411 |   0.7674 |     0.7783 |       129 |
| accuracy     |      0.7674 |   0.7674 |     0.7674 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          86 |            18 |               0 |          104 |
| Ironia       |           6 |            12 |               0 |           18 |
| Sarcasmo     |           2 |             4 |               1 |            7 |

##### Run: `Case 3` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8900 |   0.8558 |     0.8725 |       104 |
| Ironia       |      0.3333 |   0.5000 |     0.4000 |        18 |
| Sarcasmo     |      0.5000 |   0.1429 |     0.2222 |         7 |
| macro avg    |      0.5744 |   0.4995 |     0.4983 |       129 |
| weighted avg |      0.7912 |   0.7674 |     0.7713 |       129 |
| accuracy     |      0.7674 |   0.7674 |     0.7674 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          89 |            15 |               0 |          104 |
| Ironia       |           8 |             9 |               1 |           18 |
| Sarcasmo     |           3 |             3 |               1 |            7 |

##### Run: `Case 3` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8800 |   0.8462 |     0.8627 |       104 |
| Ironia       |      0.3200 |   0.4444 |     0.3721 |        18 |
| Sarcasmo     |      0.2500 |   0.1429 |     0.1818 |         7 |
| macro avg    |      0.4833 |   0.4778 |     0.4722 |       129 |
| weighted avg |      0.7677 |   0.7519 |     0.7573 |       129 |
| accuracy     |      0.7519 |   0.7519 |     0.7519 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          88 |            14 |               2 |          104 |
| Ironia       |           9 |             8 |               1 |           18 |
| Sarcasmo     |           3 |             3 |               1 |            7 |

#### Case 4: Step 2 Oracle (Train: no rel, Eval: +rel) (Checkpoint: `irony_textual_corr`, Eval Prompt: `Oracle (+ rel)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Blind Text (SGD, raw)                 | raw       |     0.5052 |              0.7132 |        0.7421 |    0.8280 |      0.4375 |        0.2500 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5374 |              0.8062 |        0.7993 |    0.8900 |      0.5000 |        0.2222 |
| Blind Text (SGD, std)                 | std       |     0.5013 |              0.7597 |        0.7637 |    0.8627 |      0.3913 |        0.2500 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.5179 |              0.8140 |        0.7939 |    0.8940 |      0.4375 |        0.2222 |

##### Run: `Case 4` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.9390 |   0.7404 |     0.8280 |       104 |
| Ironia       |      0.3043 |   0.7778 |     0.4375 |        18 |
| Sarcasmo     |      1.0000 |   0.1429 |     0.2500 |         7 |
| macro avg    |      0.7478 |   0.5537 |     0.5052 |       129 |
| weighted avg |      0.8538 |   0.7132 |     0.7421 |       129 |
| accuracy     |      0.7132 |   0.7132 |     0.7132 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          77 |            27 |               0 |          104 |
| Ironia       |           4 |            14 |               0 |           18 |
| Sarcasmo     |           1 |             5 |               1 |            7 |

##### Run: `Case 4` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8857 |   0.8942 |     0.8900 |       104 |
| Ironia       |      0.4545 |   0.5556 |     0.5000 |        18 |
| Sarcasmo     |      0.5000 |   0.1429 |     0.2222 |         7 |
| macro avg    |      0.6134 |   0.5309 |     0.5374 |       129 |
| weighted avg |      0.8046 |   0.8062 |     0.7993 |       129 |
| accuracy     |      0.8062 |   0.8062 |     0.8062 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          93 |            10 |               1 |          104 |
| Ironia       |           8 |            10 |               0 |           18 |
| Sarcasmo     |           4 |             2 |               1 |            7 |

##### Run: `Case 4` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8800 |   0.8462 |     0.8627 |       104 |
| Ironia       |      0.3214 |   0.5000 |     0.3913 |        18 |
| Sarcasmo     |      1.0000 |   0.1429 |     0.2500 |         7 |
| macro avg    |      0.7338 |   0.4963 |     0.5013 |       129 |
| weighted avg |      0.8086 |   0.7597 |     0.7637 |       129 |
| accuracy     |      0.7597 |   0.7597 |     0.7597 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          88 |            16 |               0 |          104 |
| Ironia       |           9 |             9 |               0 |           18 |
| Sarcasmo     |           3 |             3 |               1 |            7 |

##### Run: `Case 4` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8584 |   0.9327 |     0.8940 |       104 |
| Ironia       |      0.5000 |   0.3889 |     0.4375 |        18 |
| Sarcasmo     |      0.5000 |   0.1429 |     0.2222 |         7 |
| macro avg    |      0.6195 |   0.4881 |     0.5179 |       129 |
| weighted avg |      0.7889 |   0.8140 |     0.7939 |       129 |
| accuracy     |      0.8140 |   0.8140 |     0.8140 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          97 |             6 |               1 |          104 |
| Ironia       |          11 |             7 |               0 |           18 |
| Sarcasmo     |           5 |             1 |               1 |            7 |

#### Case 5: Step 2 Oracle (Train: +rel, Eval: no rel) (Checkpoint: `irony_textual_corr_with_rel`, Eval Prompt: `Oracle (no rel)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Blind Text (SGD, raw)                 | raw       |     0.3846 |              0.7442 |        0.7352 |    0.8612 |      0.2927 |        0.0000 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.3634 |              0.7209 |        0.7162 |    0.8462 |      0.2439 |        0.0000 |
| Blind Text (SGD, std)                 | std       |     0.3617 |              0.5039 |        0.5809 |    0.6506 |      0.3846 |        0.0500 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.3805 |              0.7209 |        0.7251 |    0.8488 |      0.2927 |        0.0000 |

##### Run: `Case 5` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8571 |   0.8654 |     0.8612 |       104 |
| Ironia       |      0.2609 |   0.3333 |     0.2927 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.3727 |   0.3996 |     0.3846 |       129 |
| weighted avg |      0.7274 |   0.7442 |     0.7352 |       129 |
| accuracy     |      0.7442 |   0.7442 |     0.7442 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          90 |            14 |               0 |          104 |
| Ironia       |          11 |             6 |               1 |           18 |
| Sarcasmo     |           4 |             3 |               0 |            7 |

##### Run: `Case 5` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8462 |   0.8462 |     0.8462 |       104 |
| Ironia       |      0.2174 |   0.2778 |     0.2439 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.3545 |   0.3746 |     0.3634 |       129 |
| weighted avg |      0.7125 |   0.7209 |     0.7162 |       129 |
| accuracy     |      0.7209 |   0.7209 |     0.7209 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          88 |            15 |               1 |          104 |
| Ironia       |          12 |             5 |               1 |           18 |
| Sarcasmo     |           4 |             3 |               0 |            7 |

##### Run: `Case 5` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8710 |   0.5192 |     0.6506 |       104 |
| Ironia       |      0.2941 |   0.5556 |     0.3846 |        18 |
| Sarcasmo     |      0.0303 |   0.1429 |     0.0500 |         7 |
| macro avg    |      0.3985 |   0.4059 |     0.3617 |       129 |
| weighted avg |      0.7449 |   0.5039 |     0.5809 |       129 |
| accuracy     |      0.5039 |   0.5039 |     0.5039 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          54 |            21 |              29 |          104 |
| Ironia       |           5 |            10 |               3 |           18 |
| Sarcasmo     |           3 |             3 |               1 |            7 |

##### Run: `Case 5` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8614 |   0.8365 |     0.8488 |       104 |
| Ironia       |      0.2609 |   0.3333 |     0.2927 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.3741 |   0.3900 |     0.3805 |       129 |
| weighted avg |      0.7309 |   0.7209 |     0.7251 |       129 |
| accuracy     |      0.7209 |   0.7209 |     0.7209 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          87 |            14 |               3 |          104 |
| Ironia       |          10 |             6 |               2 |           18 |
| Sarcasmo     |           4 |             3 |               0 |            7 |

#### Case 6: Step 2 Oracle (Train: +rel, Eval: +rel) (Checkpoint: `irony_textual_corr_with_rel`, Eval Prompt: `Oracle (+ rel)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Blind Text (SGD, raw)                 | raw       |     0.4424 |              0.7674 |        0.7669 |    0.8725 |      0.4545 |        0.0000 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.4186 |              0.7752 |        0.7601 |    0.8774 |      0.3784 |        0.0000 |
| Blind Text (SGD, std)                 | std       |     0.4314 |              0.7132 |        0.7206 |    0.8333 |      0.2791 |        0.1818 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.4304 |              0.7519 |        0.7573 |    0.8657 |      0.4255 |        0.0000 |

##### Run: `Case 6` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8900 |   0.8558 |     0.8725 |       104 |
| Ironia       |      0.3846 |   0.5556 |     0.4545 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.4249 |   0.4704 |     0.4424 |       129 |
| weighted avg |      0.7712 |   0.7674 |     0.7669 |       129 |
| accuracy     |      0.7674 |   0.7674 |     0.7674 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          89 |            13 |               2 |          104 |
| Ironia       |           7 |            10 |               1 |           18 |
| Sarcasmo     |           4 |             3 |               0 |            7 |

##### Run: `Case 6` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8611 |   0.8942 |     0.8774 |       104 |
| Ironia       |      0.3684 |   0.3889 |     0.3784 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.4098 |   0.4277 |     0.4186 |       129 |
| weighted avg |      0.7456 |   0.7752 |     0.7601 |       129 |
| accuracy     |      0.7752 |   0.7752 |     0.7752 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          93 |             9 |               2 |          104 |
| Ironia       |          11 |             7 |               0 |           18 |
| Sarcasmo     |           4 |             3 |               0 |            7 |

##### Run: `Case 6` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8500 |   0.8173 |     0.8333 |       104 |
| Ironia       |      0.2400 |   0.3333 |     0.2791 |        18 |
| Sarcasmo     |      0.2500 |   0.1429 |     0.1818 |         7 |
| macro avg    |      0.4467 |   0.4312 |     0.4314 |       129 |
| weighted avg |      0.7323 |   0.7132 |     0.7206 |       129 |
| accuracy     |      0.7132 |   0.7132 |     0.7132 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          85 |            17 |               2 |          104 |
| Ironia       |          11 |             6 |               1 |           18 |
| Sarcasmo     |           4 |             2 |               1 |            7 |

##### Run: `Case 6` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8969 |   0.8365 |     0.8657 |       104 |
| Ironia       |      0.3448 |   0.5556 |     0.4255 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.4139 |   0.4640 |     0.4304 |       129 |
| weighted avg |      0.7712 |   0.7519 |     0.7573 |       129 |
| accuracy     |      0.7519 |   0.7519 |     0.7519 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          87 |            15 |               2 |          104 |
| Ironia       |           7 |            10 |               1 |           18 |
| Sarcasmo     |           3 |             4 |               0 |            7 |

---

## 3. Task 3: Semiotic Relation Classification (3-Class Single-Label, `UmBERTo` + `ViT-224`)

- **Text Encoder**: `Musixmatch/umberto-commoncrawl-cased-v1` (`UmBERTo`, 768-d `[CLS]`)
- **Vision Encoder**: `google/vit-base-patch16-224` (`ViT-Base-224`, frozen 768-d `[CLS]`)
- **Multimodal Head (`mm_test`, $N=129$)**: `SGDClassifier(loss='squared_hinge', penalty='l2', class_weight='balanced', random_state=42)`

### 3.1 Textual Encoder Training Benchmark (`mm_test`, $N=129$)

| Regime         | Checkpoint               | Prompt   |   N Test |   Macro F1 |   Micro F1 (= Acc) |   Weighted F1 |   Additiva F1 |   Divergente F1 |   Parallela F1 |
|:---------------|:-------------------------|:---------|---------:|-----------:|-------------------:|--------------:|--------------:|----------------:|---------------:|
| Stage A Raw    | relation_textual_no_corr | Raw      |      129 |     0.4649 |             0.6512 |        0.5825 |        0.7381 |          0.6567 |         0.0000 |
| Stage B Oracle | relation_textual_corr    | Oracle   |      129 |     0.4668 |             0.6512 |        0.5832 |        0.7337 |          0.6667 |         0.0000 |

#### Textual Encoder Stage: `Stage A Raw` (`relation_textual_no_corr`)
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6200 |   0.9118 |     0.7381 |        68 |
| Divergente   |      0.7586 |   0.5789 |     0.6567 |        38 |
| Parallela    |      0.0000 |   0.0000 |     0.0000 |        23 |
| macro avg    |      0.4595 |   0.4969 |     0.4649 |       129 |
| weighted avg |      0.5503 |   0.6512 |     0.5825 |       129 |
| accuracy     |      0.6512 |   0.6512 |     0.6512 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              62 |                 6 |                0 |           68 |
| Divergente   |              16 |                22 |                0 |           38 |
| Parallela    |              22 |                 1 |                0 |           23 |

#### Textual Encoder Stage: `Stage B Oracle` (`relation_textual_corr`)
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6139 |   0.9118 |     0.7337 |        68 |
| Divergente   |      0.7857 |   0.5789 |     0.6667 |        38 |
| Parallela    |      0.0000 |   0.0000 |     0.0000 |        23 |
| macro avg    |      0.4665 |   0.4969 |     0.4668 |       129 |
| weighted avg |      0.5550 |   0.6512 |     0.5832 |       129 |
| accuracy     |      0.6512 |   0.6512 |     0.6512 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              62 |                 6 |                0 |           68 |
| Divergente   |              16 |                22 |                0 |           38 |
| Parallela    |              23 |                 0 |                0 |           23 |

---

### 3.2 Multimodal Semiotic Relation Benchmark (`mm_test`, $N=129$ — `Stage A Raw` vs `Stage B Oracle` $\times$ `raw` vs `std`)

#### Master Comparison Table (All Multimodal Relation Runs)
| Case   | Regime                     | Prompting   | Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   Additiva F1 |   Divergente F1 |   Parallela F1 |
|:-------|:---------------------------|:------------|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|--------------:|----------------:|---------------:|
| Case 1 | Stage A Raw (No corr)      | None (Raw)  | Blind Text (SGD, raw)                 | raw       |     0.5817 |              0.6047 |        0.6063 |        0.6519 |          0.5833 |         0.5098 |
| Case 1 | Stage A Raw (No corr)      | None (Raw)  | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5189 |              0.5891 |        0.5807 |        0.6846 |          0.5538 |         0.3182 |
| Case 1 | Stage A Raw (No corr)      | None (Raw)  | Blind Text (SGD, std)                 | std       |     0.5591 |              0.6124 |        0.6059 |        0.6759 |          0.6111 |         0.3902 |
| Case 1 | Stage A Raw (No corr)      | None (Raw)  | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.6048 |              0.6434 |        0.6401 |        0.6986 |          0.6269 |         0.4889 |
| Case 2 | Stage B Oracle (With corr) | Oracle      | Blind Text (SGD, raw)                 | raw       |     0.6323 |              0.6822 |        0.6700 |        0.7355 |          0.6471 |         0.5143 |
| Case 2 | Stage B Oracle (With corr) | Oracle      | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5422 |              0.5891 |        0.5823 |        0.6759 |          0.4857 |         0.4651 |
| Case 2 | Stage B Oracle (With corr) | Oracle      | Blind Text (SGD, std)                 | std       |     0.5711 |              0.6512 |        0.6317 |        0.7226 |          0.6377 |         0.3529 |
| Case 2 | Stage B Oracle (With corr) | Oracle      | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.6180 |              0.6822 |        0.6691 |        0.7451 |          0.6765 |         0.4324 |

#### Target Classes Per-Class F1 Breakdown (`ADD`, `PAR`, `DIV` on `mm_test`, $N=129$)
| Case   | Regime                     | Model                                 | Scaling   |   ADD (n=68) |   PAR (n=23) |   DIV (n=38) |   Macro F1 |   Micro F1 (= Acc.) |
|:-------|:---------------------------|:--------------------------------------|:----------|-------------:|-------------:|-------------:|-----------:|--------------------:|
| Case 1 | Stage A Raw (No corr)      | Blind Text (SGD, raw)                 | raw       |       0.6519 |       0.5098 |       0.5833 |     0.5817 |              0.6047 |
| Case 1 | Stage A Raw (No corr)      | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.6846 |       0.3182 |       0.5538 |     0.5189 |              0.5891 |
| Case 1 | Stage A Raw (No corr)      | Blind Text (SGD, std)                 | std       |       0.6759 |       0.3902 |       0.6111 |     0.5591 |              0.6124 |
| Case 1 | Stage A Raw (No corr)      | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.6986 |       0.4889 |       0.6269 |     0.6048 |              0.6434 |
| Case 2 | Stage B Oracle (With corr) | Blind Text (SGD, raw)                 | raw       |       0.7355 |       0.5143 |       0.6471 |     0.6323 |              0.6822 |
| Case 2 | Stage B Oracle (With corr) | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.6759 |       0.4651 |       0.4857 |     0.5422 |              0.5891 |
| Case 2 | Stage B Oracle (With corr) | Blind Text (SGD, std)                 | std       |       0.7226 |       0.3529 |       0.6377 |     0.5711 |              0.6512 |
| Case 2 | Stage B Oracle (With corr) | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.7451 |       0.4324 |       0.6765 |     0.6180 |              0.6822 |

### 3.3 Multimodal Semiotic Relation Detailed Reports by Case (`Case 1` & `Case 2`)

#### Case 1: Stage A Raw (No corr) (Checkpoint: `relation_textual_no_corr`, Eval Prompt: `None (Raw)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   Additiva F1 |   Divergente F1 |   Parallela F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|--------------:|----------------:|---------------:|
| Blind Text (SGD, raw)                 | raw       |     0.5817 |              0.6047 |        0.6063 |        0.6519 |          0.5833 |         0.5098 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5189 |              0.5891 |        0.5807 |        0.6846 |          0.5538 |         0.3182 |
| Blind Text (SGD, std)                 | std       |     0.5591 |              0.6124 |        0.6059 |        0.6759 |          0.6111 |         0.3902 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.6048 |              0.6434 |        0.6401 |        0.6986 |          0.6269 |         0.4889 |

##### Run: `Case 1` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6567 |   0.6471 |     0.6519 |        68 |
| Divergente   |      0.6176 |   0.5526 |     0.5833 |        38 |
| Parallela    |      0.4643 |   0.5652 |     0.5098 |        23 |
| macro avg    |      0.5795 |   0.5883 |     0.5817 |       129 |
| weighted avg |      0.6109 |   0.6047 |     0.6063 |       129 |
| accuracy     |      0.6047 |   0.6047 |     0.6047 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              44 |                11 |               13 |           68 |
| Divergente   |              15 |                21 |                2 |           38 |
| Parallela    |               8 |                 2 |               13 |           23 |

##### Run: `Case 1` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6296 |   0.7500 |     0.6846 |        68 |
| Divergente   |      0.6667 |   0.4737 |     0.5538 |        38 |
| Parallela    |      0.3333 |   0.3043 |     0.3182 |        23 |
| macro avg    |      0.5432 |   0.5093 |     0.5189 |       129 |
| weighted avg |      0.5877 |   0.5891 |     0.5807 |       129 |
| accuracy     |      0.5891 |   0.5891 |     0.5891 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              51 |                 6 |               11 |           68 |
| Divergente   |              17 |                18 |                3 |           38 |
| Parallela    |              13 |                 3 |                7 |           23 |

##### Run: `Case 1` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6364 |   0.7206 |     0.6759 |        68 |
| Divergente   |      0.6471 |   0.5789 |     0.6111 |        38 |
| Parallela    |      0.4444 |   0.3478 |     0.3902 |        23 |
| macro avg    |      0.5760 |   0.5491 |     0.5591 |       129 |
| weighted avg |      0.6053 |   0.6124 |     0.6059 |       129 |
| accuracy     |      0.6124 |   0.6124 |     0.6124 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              49 |                10 |                9 |           68 |
| Divergente   |              15 |                22 |                1 |           38 |
| Parallela    |              13 |                 2 |                8 |           23 |

##### Run: `Case 1` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6538 |   0.7500 |     0.6986 |        68 |
| Divergente   |      0.7241 |   0.5526 |     0.6269 |        38 |
| Parallela    |      0.5000 |   0.4783 |     0.4889 |        23 |
| macro avg    |      0.6260 |   0.5936 |     0.6048 |       129 |
| weighted avg |      0.6471 |   0.6434 |     0.6401 |       129 |
| accuracy     |      0.6434 |   0.6434 |     0.6434 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              51 |                 7 |               10 |           68 |
| Divergente   |              16 |                21 |                1 |           38 |
| Parallela    |              11 |                 1 |               11 |           23 |

#### Case 2: Stage B Oracle (With corr) (Checkpoint: `relation_textual_corr`, Eval Prompt: `Oracle`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   Additiva F1 |   Divergente F1 |   Parallela F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|--------------:|----------------:|---------------:|
| Blind Text (SGD, raw)                 | raw       |     0.6323 |              0.6822 |        0.6700 |        0.7355 |          0.6471 |         0.5143 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5422 |              0.5891 |        0.5823 |        0.6759 |          0.4857 |         0.4651 |
| Blind Text (SGD, std)                 | std       |     0.5711 |              0.6512 |        0.6317 |        0.7226 |          0.6377 |         0.3529 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.6180 |              0.6822 |        0.6691 |        0.7451 |          0.6765 |         0.4324 |

##### Run: `Case 2` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6552 |   0.8382 |     0.7355 |        68 |
| Divergente   |      0.7333 |   0.5789 |     0.6471 |        38 |
| Parallela    |      0.7500 |   0.3913 |     0.5143 |        23 |
| macro avg    |      0.7128 |   0.6028 |     0.6323 |       129 |
| weighted avg |      0.6951 |   0.6822 |     0.6700 |       129 |
| accuracy     |      0.6822 |   0.6822 |     0.6822 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              57 |                 8 |                3 |           68 |
| Divergente   |              16 |                22 |                0 |           38 |
| Parallela    |              14 |                 0 |                9 |           23 |

##### Run: `Case 2` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6364 |   0.7206 |     0.6759 |        68 |
| Divergente   |      0.5312 |   0.4474 |     0.4857 |        38 |
| Parallela    |      0.5000 |   0.4348 |     0.4651 |        23 |
| macro avg    |      0.5559 |   0.5342 |     0.5422 |       129 |
| weighted avg |      0.5811 |   0.5891 |     0.5823 |       129 |
| accuracy     |      0.5891 |   0.5891 |     0.5891 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              49 |                11 |                8 |           68 |
| Divergente   |              19 |                17 |                2 |           38 |
| Parallela    |               9 |                 4 |               10 |           23 |

##### Run: `Case 2` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6437 |   0.8235 |     0.7226 |        68 |
| Divergente   |      0.7097 |   0.5789 |     0.6377 |        38 |
| Parallela    |      0.5455 |   0.2609 |     0.3529 |        23 |
| macro avg    |      0.6329 |   0.5544 |     0.5711 |       129 |
| weighted avg |      0.6456 |   0.6512 |     0.6317 |       129 |
| accuracy     |      0.6512 |   0.6512 |     0.6512 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              56 |                 9 |                3 |           68 |
| Divergente   |              14 |                22 |                2 |           38 |
| Parallela    |              17 |                 0 |                6 |           23 |

##### Run: `Case 2` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6706 |   0.8382 |     0.7451 |        68 |
| Divergente   |      0.7667 |   0.6053 |     0.6765 |        38 |
| Parallela    |      0.5714 |   0.3478 |     0.4324 |        23 |
| macro avg    |      0.6696 |   0.5971 |     0.6180 |       129 |
| weighted avg |      0.6812 |   0.6822 |     0.6691 |       129 |
| accuracy     |      0.6822 |   0.6822 |     0.6822 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              57 |                 6 |                5 |           68 |
| Divergente   |              14 |                23 |                1 |           38 |
| Parallela    |              14 |                 1 |                8 |           23 |
