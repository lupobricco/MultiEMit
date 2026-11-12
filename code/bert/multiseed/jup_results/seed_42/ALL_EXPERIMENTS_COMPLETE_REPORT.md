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
| Step 1 Base (Raw)             | emotion_textual_no_corr       | Raw                                  |      405 |     0.5123 |     0.5326 |        0.5297 |        0.3333 |
| Step 2 Raw (Ablation)         | emotion_textual_step2_raw     | Raw                                  |      405 |     0.5104 |     0.5352 |        0.5341 |        0.3210 |
| Step 2 Oracle (Train: no rel) | emotion_textual_corr          | Oracle (Train: no rel, Eval: no rel) |      405 |     0.5304 |     0.5572 |        0.5556 |        0.3235 |
| Step 2 Oracle (Train: +rel)   | emotion_textual_corr_with_rel | Oracle (Train: +rel, Eval: no rel)   |      405 |     0.5592 |     0.5732 |        0.5733 |        0.3333 |

#### Per-Emotion F1 Breakdown (`me_test`)
| Regime                        |   Amore F1 |   Anticipazione F1 |   Disgusto F1 |   Fiducia F1 |   Gioia F1 |   Neutro F1 |   Paura F1 |   Rabbia F1 |   Sorpresa F1 |   Tristezza F1 |
|:------------------------------|-----------:|-------------------:|--------------:|-------------:|-----------:|------------:|-----------:|------------:|--------------:|---------------:|
| Step 1 Base (Raw)             |     0.6538 |             0.5000 |        0.5897 |       0.5644 |     0.5714 |      0.4762 |     0.3750 |      0.4151 |        0.4181 |         0.5591 |
| Step 2 Raw (Ablation)         |     0.6392 |             0.4490 |        0.5658 |       0.5549 |     0.5970 |      0.5051 |     0.3333 |      0.4906 |        0.4592 |         0.5102 |
| Step 2 Oracle (Train: no rel) |     0.6374 |             0.4815 |        0.6027 |       0.6286 |     0.6154 |      0.4536 |     0.4000 |      0.4444 |        0.4717 |         0.5686 |
| Step 2 Oracle (Train: +rel)   |     0.6882 |             0.5455 |        0.5976 |       0.6199 |     0.6091 |      0.5347 |     0.4706 |      0.4615 |        0.4848 |         0.5806 |

#### Textual Case: `Step 1 Base (Raw)` (`emotion_textual_no_corr`)
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.6182 |   0.6939 |     0.6538 |        49 |
| Anticipazione |      0.5217 |   0.4800 |     0.5000 |        25 |
| Disgusto      |      0.5610 |   0.6216 |     0.5897 |        74 |
| Fiducia       |      0.5750 |   0.5542 |     0.5644 |        83 |
| Gioia         |      0.5455 |   0.6000 |     0.5714 |        90 |
| Neutro        |      0.4717 |   0.4808 |     0.4762 |        52 |
| Paura         |      0.7500 |   0.2500 |     0.3750 |        12 |
| Rabbia        |      0.4231 |   0.4074 |     0.4151 |        27 |
| Sorpresa      |      0.4205 |   0.4157 |     0.4181 |        89 |
| Tristezza     |      0.5098 |   0.6190 |     0.5591 |        42 |
| micro avg     |      0.5241 |   0.5414 |     0.5326 |       543 |
| macro avg     |      0.5396 |   0.5123 |     0.5123 |       543 |
| weighted avg  |      0.5257 |   0.5414 |     0.5297 |       543 |
| samples avg   |      0.5290 |   0.5173 |     0.5045 |       543 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  335 |   21 |   15 |   34 |      0.6182 |   0.6939 |     0.6538 |        49 |
| Anticipazione |  369 |   11 |   13 |   12 |      0.5217 |   0.4800 |     0.5000 |        25 |
| Disgusto      |  295 |   36 |   28 |   46 |      0.5610 |   0.6216 |     0.5897 |        74 |
| Fiducia       |  288 |   34 |   37 |   46 |      0.5750 |   0.5542 |     0.5644 |        83 |
| Gioia         |  270 |   45 |   36 |   54 |      0.5455 |   0.6000 |     0.5714 |        90 |
| Neutro        |  325 |   28 |   27 |   25 |      0.4717 |   0.4808 |     0.4762 |        52 |
| Paura         |  392 |    1 |    9 |    3 |      0.7500 |   0.2500 |     0.3750 |        12 |
| Rabbia        |  363 |   15 |   16 |   11 |      0.4231 |   0.4074 |     0.4151 |        27 |
| Sorpresa      |  265 |   51 |   52 |   37 |      0.4205 |   0.4157 |     0.4181 |        89 |
| Tristezza     |  338 |   25 |   16 |   26 |      0.5098 |   0.6190 |     0.5591 |        42 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             49 |           34 |                    1 |               2 |             21 |           11 |             1 |            0 |             1 |               1 |                1 |           0 |
| Anticipazione  |             25 |            0 |                   12 |               4 |              4 |            3 |             1 |            1 |             1 |               4 |                3 |           0 |
| Disgusto       |             74 |            0 |                    3 |              46 |              7 |            3 |             5 |            0 |            18 |              22 |               20 |           0 |
| Fiducia        |             83 |           19 |                    4 |               7 |             46 |           16 |             8 |            1 |             3 |               8 |                8 |           0 |
| Gioia          |             90 |           15 |                    2 |               3 |             17 |           54 |             5 |            0 |             1 |              19 |                2 |           0 |
| Neutro         |             52 |            3 |                    5 |               5 |              5 |            8 |            25 |            0 |             0 |              10 |                0 |           0 |
| Paura          |             12 |            0 |                    1 |               3 |              0 |            2 |             2 |            3 |             1 |               5 |                7 |           0 |
| Rabbia         |             27 |            0 |                    1 |              24 |              0 |            1 |             2 |            1 |            11 |               4 |                5 |           0 |
| Sorpresa       |             89 |            4 |                    1 |              28 |              9 |           19 |             9 |            3 |            13 |              37 |               27 |           0 |
| Tristezza      |             42 |            0 |                    0 |              21 |              5 |            2 |             1 |            0 |             8 |              11 |               26 |           0 |

#### Textual Case: `Step 2 Raw (Ablation)` (`emotion_textual_step2_raw`)
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.6458 |   0.6327 |     0.6392 |        49 |
| Anticipazione |      0.4583 |   0.4400 |     0.4490 |        25 |
| Disgusto      |      0.5513 |   0.5811 |     0.5658 |        74 |
| Fiducia       |      0.5333 |   0.5783 |     0.5549 |        83 |
| Gioia         |      0.5405 |   0.6667 |     0.5970 |        90 |
| Neutro        |      0.5319 |   0.4808 |     0.5051 |        52 |
| Paura         |      0.5000 |   0.2500 |     0.3333 |        12 |
| Rabbia        |      0.5000 |   0.4815 |     0.4906 |        27 |
| Sorpresa      |      0.4206 |   0.5056 |     0.4592 |        89 |
| Tristezza     |      0.4464 |   0.5952 |     0.5102 |        42 |
| micro avg     |      0.5126 |   0.5599 |     0.5352 |       543 |
| macro avg     |      0.5128 |   0.5212 |     0.5104 |       543 |
| weighted avg  |      0.5159 |   0.5599 |     0.5341 |       543 |
| samples avg   |      0.5216 |   0.5405 |     0.5105 |       543 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  339 |   17 |   18 |   31 |      0.6458 |   0.6327 |     0.6392 |        49 |
| Anticipazione |  367 |   13 |   14 |   11 |      0.4583 |   0.4400 |     0.4490 |        25 |
| Disgusto      |  296 |   35 |   31 |   43 |      0.5513 |   0.5811 |     0.5658 |        74 |
| Fiducia       |  280 |   42 |   35 |   48 |      0.5333 |   0.5783 |     0.5549 |        83 |
| Gioia         |  264 |   51 |   30 |   60 |      0.5405 |   0.6667 |     0.5970 |        90 |
| Neutro        |  331 |   22 |   27 |   25 |      0.5319 |   0.4808 |     0.5051 |        52 |
| Paura         |  390 |    3 |    9 |    3 |      0.5000 |   0.2500 |     0.3333 |        12 |
| Rabbia        |  365 |   13 |   14 |   13 |      0.5000 |   0.4815 |     0.4906 |        27 |
| Sorpresa      |  254 |   62 |   44 |   45 |      0.4206 |   0.5056 |     0.4592 |        89 |
| Tristezza     |  332 |   31 |   17 |   25 |      0.4464 |   0.5952 |     0.5102 |        42 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             49 |           31 |                    0 |               3 |             19 |           15 |             1 |            1 |             1 |               6 |                2 |           0 |
| Anticipazione  |             25 |            0 |                   11 |               4 |              5 |            2 |             1 |            1 |             1 |               2 |                5 |           0 |
| Disgusto       |             74 |            0 |                    4 |              43 |              8 |            4 |             4 |            1 |            18 |              22 |               22 |           0 |
| Fiducia        |             83 |           16 |                    5 |               5 |             48 |           20 |             6 |            3 |             3 |              10 |                8 |           0 |
| Gioia          |             90 |           14 |                    2 |               3 |             22 |           60 |             4 |            0 |             0 |              29 |                1 |           0 |
| Neutro         |             52 |            2 |                    2 |               5 |              4 |           10 |            25 |            0 |             0 |              15 |                1 |           0 |
| Paura          |             12 |            0 |                    1 |               2 |              1 |            1 |             2 |            3 |             0 |               7 |                8 |           0 |
| Rabbia         |             27 |            0 |                    0 |              23 |              0 |            1 |             0 |            0 |            13 |               7 |                8 |           0 |
| Sorpresa       |             89 |            4 |                    1 |              22 |             10 |           23 |             8 |            4 |            15 |              45 |               27 |           0 |
| Tristezza      |             42 |            1 |                    3 |              17 |              6 |            1 |             1 |            1 |             9 |              11 |               25 |           0 |

#### Textual Case: `Step 2 Oracle (Train: no rel)` (`emotion_textual_corr`)
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.6905 |   0.5918 |     0.6374 |        49 |
| Anticipazione |      0.4483 |   0.5200 |     0.4815 |        25 |
| Disgusto      |      0.6111 |   0.5946 |     0.6027 |        74 |
| Fiducia       |      0.5978 |   0.6627 |     0.6286 |        83 |
| Gioia         |      0.5424 |   0.7111 |     0.6154 |        90 |
| Neutro        |      0.4889 |   0.4231 |     0.4536 |        52 |
| Paura         |      1.0000 |   0.2500 |     0.4000 |        12 |
| Rabbia        |      0.5556 |   0.3704 |     0.4444 |        27 |
| Sorpresa      |      0.4065 |   0.5618 |     0.4717 |        89 |
| Tristezza     |      0.4833 |   0.6905 |     0.5686 |        42 |
| micro avg     |      0.5299 |   0.5875 |     0.5572 |       543 |
| macro avg     |      0.5824 |   0.5376 |     0.5304 |       543 |
| weighted avg  |      0.5481 |   0.5875 |     0.5556 |       543 |
| samples avg   |      0.5401 |   0.5621 |     0.5308 |       543 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  343 |   13 |   20 |   29 |      0.6905 |   0.5918 |     0.6374 |        49 |
| Anticipazione |  364 |   16 |   12 |   13 |      0.4483 |   0.5200 |     0.4815 |        25 |
| Disgusto      |  303 |   28 |   30 |   44 |      0.6111 |   0.5946 |     0.6027 |        74 |
| Fiducia       |  285 |   37 |   28 |   55 |      0.5978 |   0.6627 |     0.6286 |        83 |
| Gioia         |  261 |   54 |   26 |   64 |      0.5424 |   0.7111 |     0.6154 |        90 |
| Neutro        |  330 |   23 |   30 |   22 |      0.4889 |   0.4231 |     0.4536 |        52 |
| Paura         |  393 |    0 |    9 |    3 |      1.0000 |   0.2500 |     0.4000 |        12 |
| Rabbia        |  370 |    8 |   17 |   10 |      0.5556 |   0.3704 |     0.4444 |        27 |
| Sorpresa      |  243 |   73 |   39 |   50 |      0.4065 |   0.5618 |     0.4717 |        89 |
| Tristezza     |  332 |   31 |   13 |   29 |      0.4833 |   0.6905 |     0.5686 |        42 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             49 |           29 |                    0 |               2 |             23 |           16 |             0 |            0 |             1 |               5 |                3 |           0 |
| Anticipazione  |             25 |            0 |                   13 |               4 |              5 |            4 |             0 |            1 |             1 |               5 |                6 |           0 |
| Disgusto       |             74 |            0 |                    3 |              44 |              6 |            4 |             5 |            0 |            13 |              26 |               19 |           0 |
| Fiducia        |             83 |           16 |                    5 |               5 |             55 |           20 |             5 |            1 |             2 |              13 |               12 |           0 |
| Gioia          |             90 |           12 |                    1 |               2 |             18 |           64 |             5 |            0 |             0 |              35 |                3 |           0 |
| Neutro         |             52 |            2 |                    5 |               5 |              3 |           10 |            22 |            0 |             0 |              16 |                2 |           0 |
| Paura          |             12 |            0 |                    1 |               1 |              0 |            1 |             3 |            3 |             0 |               7 |                7 |           0 |
| Rabbia         |             27 |            0 |                    1 |              19 |              1 |            2 |             0 |            0 |            10 |               6 |                8 |           0 |
| Sorpresa       |             89 |            4 |                    3 |              21 |              5 |           26 |             8 |            2 |             9 |              50 |               26 |           0 |
| Tristezza      |             42 |            1 |                    3 |              15 |              6 |            1 |             3 |            0 |             5 |              16 |               29 |           0 |

#### Textual Case: `Step 2 Oracle (Train: +rel)` (`emotion_textual_corr_with_rel`)
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.7273 |   0.6531 |     0.6882 |        49 |
| Anticipazione |      0.5000 |   0.6000 |     0.5455 |        25 |
| Disgusto      |      0.5444 |   0.6622 |     0.5976 |        74 |
| Fiducia       |      0.6023 |   0.6386 |     0.6199 |        83 |
| Gioia         |      0.5607 |   0.6667 |     0.6091 |        90 |
| Neutro        |      0.5510 |   0.5192 |     0.5347 |        52 |
| Paura         |      0.8000 |   0.3333 |     0.4706 |        12 |
| Rabbia        |      0.4800 |   0.4444 |     0.4615 |        27 |
| Sorpresa      |      0.4404 |   0.5393 |     0.4848 |        89 |
| Tristezza     |      0.5294 |   0.6429 |     0.5806 |        42 |
| micro avg     |      0.5468 |   0.6022 |     0.5732 |       543 |
| macro avg     |      0.5736 |   0.5700 |     0.5592 |       543 |
| weighted avg  |      0.5553 |   0.6022 |     0.5733 |       543 |
| samples avg   |      0.5529 |   0.5776 |     0.5444 |       543 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  344 |   12 |   17 |   32 |      0.7273 |   0.6531 |     0.6882 |        49 |
| Anticipazione |  365 |   15 |   10 |   15 |      0.5000 |   0.6000 |     0.5455 |        25 |
| Disgusto      |  290 |   41 |   25 |   49 |      0.5444 |   0.6622 |     0.5976 |        74 |
| Fiducia       |  287 |   35 |   30 |   53 |      0.6023 |   0.6386 |     0.6199 |        83 |
| Gioia         |  268 |   47 |   30 |   60 |      0.5607 |   0.6667 |     0.6091 |        90 |
| Neutro        |  331 |   22 |   25 |   27 |      0.5510 |   0.5192 |     0.5347 |        52 |
| Paura         |  392 |    1 |    8 |    4 |      0.8000 |   0.3333 |     0.4706 |        12 |
| Rabbia        |  365 |   13 |   15 |   12 |      0.4800 |   0.4444 |     0.4615 |        27 |
| Sorpresa      |  255 |   61 |   41 |   48 |      0.4404 |   0.5393 |     0.4848 |        89 |
| Tristezza     |  339 |   24 |   15 |   27 |      0.5294 |   0.6429 |     0.5806 |        42 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             49 |           32 |                    1 |               2 |             24 |           15 |             0 |            0 |             1 |               1 |                3 |           0 |
| Anticipazione  |             25 |            0 |                   15 |               5 |              3 |            4 |             1 |            2 |             1 |               3 |                5 |           0 |
| Disgusto       |             74 |            0 |                    3 |              49 |              6 |            2 |             4 |            0 |            18 |              21 |               16 |           0 |
| Fiducia        |             83 |           17 |                    8 |               6 |             53 |           19 |             5 |            1 |             2 |              13 |                9 |           0 |
| Gioia          |             90 |           12 |                    3 |               2 |             16 |           60 |             6 |            0 |             0 |              35 |                3 |           0 |
| Neutro         |             52 |            1 |                    4 |               8 |              2 |           10 |            27 |            0 |             0 |              11 |                0 |           0 |
| Paura          |             12 |            0 |                    1 |               2 |              0 |            1 |             2 |            4 |             0 |               8 |                7 |           0 |
| Rabbia         |             27 |            0 |                    1 |              23 |              0 |            1 |             0 |            0 |            12 |               6 |                6 |           0 |
| Sorpresa       |             89 |            4 |                    3 |              28 |              6 |           20 |             9 |            2 |            11 |              48 |               25 |           0 |
| Tristezza      |             42 |            0 |                    3 |              19 |              6 |            1 |             1 |            1 |             6 |              13 |               27 |           0 |

---

### 1.2 Multimodal Emotion Benchmark (`mm_test`, $N=129$ — All 6 Cases $\times$ `raw` vs `std`)

#### Master Comparison Table (All Multimodal Emotion Runs)
| Case   | Regime                                      | Train Relation   | Eval Relation   | Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:-------|:--------------------------------------------|:-----------------|:----------------|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Blind Text (ClassifierChain)     | raw       |     0.1942 |     0.2737 |        0.2307 |        0.1085 |
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4423 |     0.4353 |        0.4519 |        0.2248 |
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Blind Text (ClassifierChain)     | std       |     0.3757 |     0.4404 |        0.4644 |        0.2016 |
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.3396 |     0.4286 |        0.4569 |        0.1860 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Blind Text (ClassifierChain)     | raw       |     0.3784 |     0.4492 |        0.4539 |        0.3488 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4761 |     0.4938 |        0.4867 |        0.3178 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Blind Text (ClassifierChain)     | std       |     0.3659 |     0.4167 |        0.4724 |        0.2093 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4092 |     0.4619 |        0.4887 |        0.2481 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Blind Text (ClassifierChain)     | raw       |     0.4339 |     0.5242 |        0.5262 |        0.3178 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.5222 |     0.5223 |        0.5275 |        0.3023 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Blind Text (ClassifierChain)     | std       |     0.4642 |     0.5254 |        0.5584 |        0.3333 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.3965 |     0.4787 |        0.5054 |        0.2403 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Blind Text (ClassifierChain)     | raw       |     0.4322 |     0.5132 |        0.5083 |        0.4031 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.3931 |     0.4920 |        0.4771 |        0.3411 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Blind Text (ClassifierChain)     | std       |     0.4518 |     0.5127 |        0.5750 |        0.2636 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4345 |     0.4929 |        0.5193 |        0.2791 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Blind Text (ClassifierChain)     | raw       |     0.4461 |     0.5360 |        0.5465 |        0.4109 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.5431 |     0.5515 |        0.5573 |        0.3178 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Blind Text (ClassifierChain)     | std       |     0.4344 |     0.5013 |        0.5472 |        0.3178 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4460 |     0.5282 |        0.5501 |        0.3643 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Blind Text (ClassifierChain)     | raw       |     0.4250 |     0.5174 |        0.5368 |        0.3101 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4744 |     0.4867 |        0.4958 |        0.3256 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Blind Text (ClassifierChain)     | std       |     0.4155 |     0.4757 |        0.5160 |        0.3256 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4153 |     0.4922 |        0.5191 |        0.1938 |

#### Per-Emotion F1 Breakdown (All Multimodal Emotion Runs)
| Case   | Regime                                      | Model                            | Scaling   |   Amore F1 |   Anticipazione F1 |   Disgusto F1 |   Fiducia F1 |   Gioia F1 |   Neutro F1 |   Paura F1 |   Rabbia F1 |   Sorpresa F1 |   Tristezza F1 |
|:-------|:--------------------------------------------|:---------------------------------|:----------|-----------:|-------------------:|--------------:|-------------:|-----------:|------------:|-----------:|------------:|--------------:|---------------:|
| Case 1 | Step 1 Base (No corr)                       | Blind Text (ClassifierChain)     | raw       |     0.3226 |             0.4667 |        0.2857 |       0.4127 |     0.0000 |      0.0000 |     0.0000 |      0.2941 |        0.1600 |         0.0000 |
| Case 1 | Step 1 Base (No corr)                       | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.3200 |             0.5882 |        0.4545 |       0.6133 |     0.3571 |      0.4737 |     0.6667 |      0.3333 |        0.2162 |         0.4000 |
| Case 1 | Step 1 Base (No corr)                       | Blind Text (ClassifierChain)     | std       |     0.4000 |             0.4800 |        0.5517 |       0.6353 |     0.4068 |      0.3030 |     0.0000 |      0.4444 |        0.2857 |         0.2500 |
| Case 1 | Step 1 Base (No corr)                       | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4118 |             0.4211 |        0.3333 |       0.7209 |     0.4561 |      0.2609 |     0.0000 |      0.2667 |        0.3030 |         0.2222 |
| Case 2 | Step 2 Raw (No corr)                        | Blind Text (ClassifierChain)     | raw       |     0.3750 |             0.5600 |        0.4500 |       0.6133 |     0.5000 |      0.2286 |     0.0000 |      0.4286 |        0.2286 |         0.4000 |
| Case 2 | Step 2 Raw (No corr)                        | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.3200 |             0.6364 |        0.4242 |       0.6506 |     0.5915 |      0.2759 |     0.5000 |      0.3750 |        0.1875 |         0.8000 |
| Case 2 | Step 2 Raw (No corr)                        | Blind Text (ClassifierChain)     | std       |     0.5600 |             0.5185 |        0.2222 |       0.6582 |     0.5902 |      0.3684 |     0.0833 |      0.3333 |        0.2105 |         0.1143 |
| Case 2 | Step 2 Raw (No corr)                        | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4324 |             0.4828 |        0.4375 |       0.6364 |     0.5758 |      0.3600 |     0.1250 |      0.5455 |        0.2105 |         0.2857 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (ClassifierChain)     | raw       |     0.4103 |             0.5263 |        0.6667 |       0.6105 |     0.5750 |      0.3333 |     0.0000 |      0.4286 |        0.4242 |         0.3636 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4706 |             0.5000 |        0.6500 |       0.6593 |     0.5714 |      0.2222 |     1.0000 |      0.3636 |        0.3846 |         0.4000 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (ClassifierChain)     | std       |     0.4828 |             0.5455 |        0.6829 |       0.5974 |     0.5882 |      0.4737 |     0.0952 |      0.4615 |        0.5143 |         0.2000 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4762 |             0.4800 |        0.5946 |       0.6186 |     0.5625 |      0.3529 |     0.2222 |      0.1053 |        0.4103 |         0.1429 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Blind Text (ClassifierChain)     | raw       |     0.3478 |             0.6667 |        0.6522 |       0.5750 |     0.5600 |      0.3333 |     0.0000 |      0.6154 |        0.2857 |         0.2857 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.2353 |             0.5263 |        0.3871 |       0.6122 |     0.5846 |      0.3889 |     0.0000 |      0.3636 |        0.3333 |         0.5000 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Blind Text (ClassifierChain)     | std       |     0.4286 |             0.6400 |        0.7222 |       0.6522 |     0.6061 |      0.4348 |     0.0000 |      0.4286 |        0.5000 |         0.1053 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.3871 |             0.4762 |        0.5128 |       0.6512 |     0.4918 |      0.4500 |     0.1053 |      0.5000 |        0.4848 |         0.2857 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Blind Text (ClassifierChain)     | raw       |     0.4615 |             0.5556 |        0.6829 |       0.5895 |     0.5806 |      0.4583 |     0.0000 |      0.4000 |        0.4828 |         0.2500 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4242 |             0.6000 |        0.5789 |       0.7059 |     0.5538 |      0.4737 |     1.0000 |      0.3077 |        0.3871 |         0.4000 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Blind Text (ClassifierChain)     | std       |     0.4706 |             0.6000 |        0.6977 |       0.6667 |     0.4918 |      0.4211 |     0.0000 |      0.3529 |        0.4615 |         0.1818 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.3158 |             0.6000 |        0.6667 |       0.7045 |     0.5763 |      0.3636 |     0.2222 |      0.3750 |        0.4138 |         0.2222 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Blind Text (ClassifierChain)     | raw       |     0.3333 |             0.5714 |        0.6286 |       0.6420 |     0.5672 |      0.4000 |     0.0000 |      0.4000 |        0.4848 |         0.2222 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.1905 |             0.5263 |        0.6667 |       0.6133 |     0.5517 |      0.2791 |     0.6667 |      0.3333 |        0.3448 |         0.5714 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Blind Text (ClassifierChain)     | std       |     0.3182 |             0.5263 |        0.6275 |       0.5882 |     0.5484 |      0.4091 |     0.0000 |      0.4000 |        0.4878 |         0.2500 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4000 |             0.4545 |        0.6809 |       0.6154 |     0.5970 |      0.3810 |     0.1250 |      0.2609 |        0.3529 |         0.2857 |

#### Semiotic Subsets Breakdown (`ADD`, `PAR`, `DIV` on `mm_test`, $N=129$)
| Case   | Regime                                      | Model                            | Scaling   |   ADD (N=68) |   PAR (N=23) |   DIV (N=38) |   ADD Macro F1 |   PAR Macro F1 |   DIV Macro F1 |   Macro F1 |   Micro F1 |
|:-------|:--------------------------------------------|:---------------------------------|:----------|-------------:|-------------:|-------------:|---------------:|---------------:|---------------:|-----------:|-----------:|
| Case 1 | Step 1 Base (No corr)                       | Blind Text (ClassifierChain)     | raw       |       0.3618 |       0.1429 |       0.1748 |         0.2288 |         0.1082 |         0.1054 |     0.1942 |     0.2737 |
| Case 1 | Step 1 Base (No corr)                       | Dual-Encoder (FEEL-IT + ViT-224) | raw       |       0.4831 |       0.3929 |       0.3774 |         0.4687 |         0.1935 |         0.3206 |     0.4423 |     0.4353 |
| Case 1 | Step 1 Base (No corr)                       | Blind Text (ClassifierChain)     | std       |       0.4766 |       0.3729 |       0.4071 |         0.4072 |         0.2187 |         0.3222 |     0.3757 |     0.4404 |
| Case 1 | Step 1 Base (No corr)                       | Dual-Encoder (FEEL-IT + ViT-224) | std       |       0.5027 |       0.4923 |       0.2679 |         0.4209 |         0.2849 |         0.2187 |     0.3396 |     0.4286 |
| Case 2 | Step 2 Raw (No corr)                        | Blind Text (ClassifierChain)     | raw       |       0.5371 |       0.4082 |       0.3168 |         0.4541 |         0.2452 |         0.2467 |     0.3784 |     0.4492 |
| Case 2 | Step 2 Raw (No corr)                        | Dual-Encoder (FEEL-IT + ViT-224) | raw       |       0.5862 |       0.5882 |       0.2737 |         0.5615 |         0.2959 |         0.1643 |     0.4761 |     0.4938 |
| Case 2 | Step 2 Raw (No corr)                        | Blind Text (ClassifierChain)     | std       |       0.4785 |       0.4262 |       0.2982 |         0.4285 |         0.3561 |         0.2665 |     0.3659 |     0.4167 |
| Case 2 | Step 2 Raw (No corr)                        | Dual-Encoder (FEEL-IT + ViT-224) | std       |       0.5446 |       0.5246 |       0.2881 |         0.4942 |         0.3306 |         0.2213 |     0.4092 |     0.4619 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (ClassifierChain)     | raw       |       0.5625 |       0.5714 |       0.4272 |         0.4962 |         0.2857 |         0.2762 |     0.4339 |     0.5242 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (FEEL-IT + ViT-224) | raw       |       0.5650 |       0.6000 |       0.4000 |         0.5539 |         0.3463 |         0.2814 |     0.5222 |     0.5223 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (ClassifierChain)     | std       |       0.5608 |       0.5424 |       0.4528 |         0.5194 |         0.5778 |         0.3511 |     0.4642 |     0.5254 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (FEEL-IT + ViT-224) | std       |       0.5128 |       0.4928 |       0.4107 |         0.4298 |         0.3400 |         0.3129 |     0.3965 |     0.4787 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Blind Text (ClassifierChain)     | raw       |       0.5610 |       0.4681 |       0.4516 |         0.4640 |         0.2382 |         0.3374 |     0.4322 |     0.5132 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Dual-Encoder (FEEL-IT + ViT-224) | raw       |       0.5476 |       0.5882 |       0.3404 |         0.4472 |         0.2401 |         0.2060 |     0.3931 |     0.4920 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Blind Text (ClassifierChain)     | std       |       0.5604 |       0.5079 |       0.4355 |         0.5122 |         0.3120 |         0.3937 |     0.4518 |     0.5127 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Dual-Encoder (FEEL-IT + ViT-224) | std       |       0.5269 |       0.4561 |       0.4545 |         0.4763 |         0.2558 |         0.3540 |     0.4345 |     0.4929 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Blind Text (ClassifierChain)     | raw       |       0.5405 |       0.6939 |       0.4602 |         0.4521 |         0.5681 |         0.3355 |     0.4461 |     0.5360 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Dual-Encoder (FEEL-IT + ViT-224) | raw       |       0.5824 |       0.6182 |       0.4516 |         0.5723 |         0.3219 |         0.2853 |     0.5431 |     0.5515 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Blind Text (ClassifierChain)     | std       |       0.5408 |       0.5312 |       0.4174 |         0.4752 |         0.4550 |         0.3258 |     0.4344 |     0.5013 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Dual-Encoder (FEEL-IT + ViT-224) | std       |       0.5934 |       0.5556 |       0.3960 |         0.5062 |         0.2567 |         0.2868 |     0.4460 |     0.5282 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Blind Text (ClassifierChain)     | raw       |       0.5109 |       0.5000 |       0.5370 |         0.4368 |         0.2546 |         0.4522 |     0.4250 |     0.5174 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Dual-Encoder (FEEL-IT + ViT-224) | raw       |       0.5062 |       0.4898 |       0.4494 |         0.4649 |         0.2639 |         0.2502 |     0.4744 |     0.4867 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Blind Text (ClassifierChain)     | std       |       0.5075 |       0.4194 |       0.4531 |         0.4628 |         0.2668 |         0.3507 |     0.4155 |     0.4757 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Dual-Encoder (FEEL-IT + ViT-224) | std       |       0.5392 |       0.5538 |       0.3761 |         0.4820 |         0.3354 |         0.2333 |     0.4153 |     0.4922 |

### 1.3 Multimodal Emotion Detailed Reports by Case (`Case 1` – `Case 6`)

#### Case 1: Step 1 Base (No corr) (Checkpoint: `emotion_textual_no_corr`, Eval Prompt: `None (Raw)`)

| Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Blind Text (ClassifierChain)     | raw       |     0.1942 |     0.2737 |        0.2307 |        0.1085 |
| Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4423 |     0.4353 |        0.4519 |        0.2248 |
| Blind Text (ClassifierChain)     | std       |     0.3757 |     0.4404 |        0.4644 |        0.2016 |
| Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.3396 |     0.4286 |        0.4569 |        0.1860 |

##### Run: `Case 1` | `Blind Text (ClassifierChain)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2500 |   0.4545 |     0.3226 |        11 |
| Anticipazione |      0.3500 |   0.7000 |     0.4667 |        10 |
| Disgusto      |      0.1700 |   0.8947 |     0.2857 |        19 |
| Fiducia       |      0.6190 |   0.3095 |     0.4127 |        42 |
| Gioia         |      0.0000 |   0.0000 |     0.0000 |        29 |
| Neutro        |      0.0000 |   0.0000 |     0.0000 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.1786 |   0.8333 |     0.2941 |         6 |
| Sorpresa      |      0.2222 |   0.1250 |     0.1600 |        16 |
| Tristezza     |      0.0000 |   0.0000 |     0.0000 |         2 |
| micro avg     |      0.2390 |   0.3203 |     0.2737 |       153 |
| macro avg     |      0.1790 |   0.3317 |     0.1942 |       153 |
| weighted avg  |      0.2621 |   0.3203 |     0.2307 |       153 |
| samples avg   |      0.2280 |   0.3282 |     0.2602 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  103 |   15 |    6 |    5 |      0.2500 |   0.4545 |     0.3226 |        11 |
| Anticipazione |  106 |   13 |    3 |    7 |      0.3500 |   0.7000 |     0.4667 |        10 |
| Disgusto      |   27 |   83 |    2 |   17 |      0.1700 |   0.8947 |     0.2857 |        19 |
| Fiducia       |   79 |    8 |   29 |   13 |      0.6190 |   0.3095 |     0.4127 |        42 |
| Gioia         |   96 |    4 |   29 |    0 |      0.0000 |   0.0000 |     0.0000 |        29 |
| Neutro        |  112 |    0 |   17 |    0 |      0.0000 |   0.0000 |     0.0000 |        17 |
| Paura         |  125 |    3 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  100 |   23 |    1 |    5 |      0.1786 |   0.8333 |     0.2941 |         6 |
| Sorpresa      |  106 |    7 |   14 |    2 |      0.2222 |   0.1250 |     0.1600 |        16 |
| Tristezza     |  127 |    0 |    2 |    0 |      0.0000 |   0.0000 |     0.0000 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            5 |                    2 |               6 |              5 |            0 |             0 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            0 |                    7 |               2 |              3 |            3 |             0 |            0 |             1 |               0 |                0 |           0 |
| Disgusto       |             19 |            2 |                    1 |              17 |              1 |            0 |             0 |            1 |            12 |               2 |                0 |           0 |
| Fiducia        |             42 |            8 |                    4 |              31 |             13 |            1 |             0 |            0 |             4 |               1 |                0 |           0 |
| Gioia          |             29 |            9 |                    2 |              24 |              4 |            0 |             0 |            0 |             3 |               1 |                0 |           0 |
| Neutro         |             17 |            2 |                    3 |              14 |              0 |            1 |             0 |            0 |             3 |               3 |                0 |           0 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            0 |             1 |               0 |                0 |           0 |
| Rabbia         |              6 |            0 |                    1 |               6 |              0 |            0 |             0 |            1 |             5 |               0 |                0 |           0 |
| Sorpresa       |             16 |            2 |                    2 |              16 |              0 |            0 |             0 |            2 |             4 |               2 |                0 |           0 |
| Tristezza      |              2 |            0 |                    1 |               2 |              0 |            0 |             0 |            1 |             2 |               0 |                0 |           0 |

##### Run: `Case 1` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2857 |   0.3636 |     0.3200 |        11 |
| Anticipazione |      0.7143 |   0.5000 |     0.5882 |        10 |
| Disgusto      |      0.4000 |   0.5263 |     0.4545 |        19 |
| Fiducia       |      0.6970 |   0.5476 |     0.6133 |        42 |
| Gioia         |      0.2727 |   0.5172 |     0.3571 |        29 |
| Neutro        |      0.4286 |   0.5294 |     0.4737 |        17 |
| Paura         |      0.5000 |   1.0000 |     0.6667 |         1 |
| Rabbia        |      0.3333 |   0.3333 |     0.3333 |         6 |
| Sorpresa      |      0.1905 |   0.2500 |     0.2162 |        16 |
| Tristezza     |      0.3333 |   0.5000 |     0.4000 |         2 |
| micro avg     |      0.3957 |   0.4837 |     0.4353 |       153 |
| macro avg     |      0.4155 |   0.5068 |     0.4423 |       153 |
| weighted avg  |      0.4482 |   0.4837 |     0.4519 |       153 |
| samples avg   |      0.3862 |   0.4832 |     0.4076 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  108 |   10 |    7 |    4 |      0.2857 |   0.3636 |     0.3200 |        11 |
| Anticipazione |  117 |    2 |    5 |    5 |      0.7143 |   0.5000 |     0.5882 |        10 |
| Disgusto      |   95 |   15 |    9 |   10 |      0.4000 |   0.5263 |     0.4545 |        19 |
| Fiducia       |   77 |   10 |   19 |   23 |      0.6970 |   0.5476 |     0.6133 |        42 |
| Gioia         |   60 |   40 |   14 |   15 |      0.2727 |   0.5172 |     0.3571 |        29 |
| Neutro        |  100 |   12 |    8 |    9 |      0.4286 |   0.5294 |     0.4737 |        17 |
| Paura         |  127 |    1 |    0 |    1 |      0.5000 |   1.0000 |     0.6667 |         1 |
| Rabbia        |  119 |    4 |    4 |    2 |      0.3333 |   0.3333 |     0.3333 |         6 |
| Sorpresa      |   96 |   17 |   12 |    4 |      0.1905 |   0.2500 |     0.2162 |        16 |
| Tristezza     |  125 |    2 |    1 |    1 |      0.3333 |   0.5000 |     0.4000 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            4 |                    0 |               0 |              8 |            4 |             1 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            2 |                    5 |               0 |              0 |            5 |             1 |            0 |             0 |               1 |                0 |           0 |
| Disgusto       |             19 |            1 |                    0 |              10 |              1 |            6 |             6 |            0 |             2 |               6 |                1 |           0 |
| Fiducia        |             42 |            7 |                    1 |               0 |             23 |           21 |             1 |            0 |             0 |               6 |                0 |           3 |
| Gioia          |             29 |            3 |                    1 |               6 |              5 |           15 |             3 |            0 |             1 |               6 |                0 |           3 |
| Neutro         |             17 |            0 |                    0 |               3 |              3 |            7 |             9 |            0 |             1 |               2 |                0 |           0 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            1 |             0 |            1 |             1 |               0 |                1 |           0 |
| Rabbia         |              6 |            0 |                    0 |               3 |              1 |            2 |             1 |            0 |             2 |               1 |                1 |           1 |
| Sorpresa       |             16 |            2 |                    0 |               5 |              0 |            8 |             1 |            1 |             1 |               4 |                1 |           1 |
| Tristezza      |              2 |            1 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               0 |                1 |           0 |

##### Run: `Case 1` | `Blind Text (ClassifierChain)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2759 |   0.7273 |     0.4000 |        11 |
| Anticipazione |      0.4000 |   0.6000 |     0.4800 |        10 |
| Disgusto      |      0.4103 |   0.8421 |     0.5517 |        19 |
| Fiducia       |      0.6279 |   0.6429 |     0.6353 |        42 |
| Gioia         |      0.4000 |   0.4138 |     0.4068 |        29 |
| Neutro        |      0.3125 |   0.2941 |     0.3030 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.3333 |   0.6667 |     0.4444 |         6 |
| Sorpresa      |      0.2632 |   0.3125 |     0.2857 |        16 |
| Tristezza     |      0.1429 |   1.0000 |     0.2500 |         2 |
| micro avg     |      0.3648 |   0.5556 |     0.4404 |       153 |
| macro avg     |      0.3166 |   0.5499 |     0.3757 |       153 |
| weighted avg  |      0.4223 |   0.5556 |     0.4644 |       153 |
| samples avg   |      0.3876 |   0.5491 |     0.4320 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |   97 |   21 |    3 |    8 |      0.2759 |   0.7273 |     0.4000 |        11 |
| Anticipazione |  110 |    9 |    4 |    6 |      0.4000 |   0.6000 |     0.4800 |        10 |
| Disgusto      |   87 |   23 |    3 |   16 |      0.4103 |   0.8421 |     0.5517 |        19 |
| Fiducia       |   71 |   16 |   15 |   27 |      0.6279 |   0.6429 |     0.6353 |        42 |
| Gioia         |   82 |   18 |   17 |   12 |      0.4000 |   0.4138 |     0.4068 |        29 |
| Neutro        |  101 |   11 |   12 |    5 |      0.3125 |   0.2941 |     0.3030 |        17 |
| Paura         |  112 |   16 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  115 |    8 |    2 |    4 |      0.3333 |   0.6667 |     0.4444 |         6 |
| Sorpresa      |   99 |   14 |   11 |    5 |      0.2632 |   0.3125 |     0.2857 |        16 |
| Tristezza     |  115 |   12 |    0 |    2 |      0.1429 |   1.0000 |     0.2500 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            8 |                    0 |               0 |             10 |            4 |             0 |            0 |             1 |               1 |                0 |           0 |
| Anticipazione  |             10 |            2 |                    6 |               1 |              3 |            2 |             1 |            1 |             0 |               0 |                0 |           1 |
| Disgusto       |             19 |            0 |                    1 |              16 |              0 |            2 |             2 |            1 |             4 |               1 |                5 |           1 |
| Fiducia        |             42 |           14 |                    1 |               1 |             27 |            8 |             4 |            4 |             2 |               4 |                1 |           3 |
| Gioia          |             29 |           11 |                    3 |              10 |             11 |           12 |             4 |            7 |             0 |               5 |                0 |           0 |
| Neutro         |             17 |            2 |                    3 |               4 |              2 |            7 |             5 |            0 |             1 |               3 |                3 |           0 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            0 |             1 |               0 |                1 |           0 |
| Rabbia         |              6 |            0 |                    0 |               4 |              1 |            1 |             1 |            2 |             4 |               1 |                4 |           0 |
| Sorpresa       |             16 |            1 |                    1 |               5 |              4 |            4 |             3 |            4 |             3 |               5 |                3 |           1 |
| Tristezza      |              2 |            0 |                    1 |               0 |              0 |            0 |             0 |            1 |             2 |               0 |                2 |           0 |

##### Run: `Case 1` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3043 |   0.6364 |     0.4118 |        11 |
| Anticipazione |      0.4444 |   0.4000 |     0.4211 |        10 |
| Disgusto      |      0.3043 |   0.3684 |     0.3333 |        19 |
| Fiducia       |      0.7045 |   0.7381 |     0.7209 |        42 |
| Gioia         |      0.4643 |   0.4483 |     0.4561 |        29 |
| Neutro        |      0.2069 |   0.3529 |     0.2609 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.1667 |   0.6667 |     0.2667 |         6 |
| Sorpresa      |      0.2941 |   0.3125 |     0.3030 |        16 |
| Tristezza     |      0.1429 |   0.5000 |     0.2222 |         2 |
| micro avg     |      0.3697 |   0.5098 |     0.4286 |       153 |
| macro avg     |      0.3033 |   0.4423 |     0.3396 |       153 |
| weighted avg  |      0.4323 |   0.5098 |     0.4569 |       153 |
| samples avg   |      0.3663 |   0.5026 |     0.4057 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  102 |   16 |    4 |    7 |      0.3043 |   0.6364 |     0.4118 |        11 |
| Anticipazione |  114 |    5 |    6 |    4 |      0.4444 |   0.4000 |     0.4211 |        10 |
| Disgusto      |   94 |   16 |   12 |    7 |      0.3043 |   0.3684 |     0.3333 |        19 |
| Fiducia       |   74 |   13 |   11 |   31 |      0.7045 |   0.7381 |     0.7209 |        42 |
| Gioia         |   85 |   15 |   16 |   13 |      0.4643 |   0.4483 |     0.4561 |        29 |
| Neutro        |   89 |   23 |   11 |    6 |      0.2069 |   0.3529 |     0.2609 |        17 |
| Paura         |  121 |    7 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  103 |   20 |    2 |    4 |      0.1667 |   0.6667 |     0.2667 |         6 |
| Sorpresa      |  101 |   12 |   11 |    5 |      0.2941 |   0.3125 |     0.3030 |        16 |
| Tristezza     |  121 |    6 |    1 |    1 |      0.1429 |   0.5000 |     0.2222 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            7 |                    0 |               0 |              7 |            2 |             4 |            1 |             0 |               0 |                0 |           1 |
| Anticipazione  |             10 |            1 |                    4 |               0 |              2 |            3 |             0 |            0 |             0 |               0 |                0 |           1 |
| Disgusto       |             19 |            0 |                    0 |               7 |              2 |            2 |             4 |            1 |             9 |               5 |                0 |           2 |
| Fiducia        |             42 |           12 |                    1 |               4 |             31 |            8 |            10 |            0 |             2 |               4 |                1 |           1 |
| Gioia          |             29 |            6 |                    2 |               6 |              7 |           13 |             5 |            1 |             3 |               3 |                2 |           3 |
| Neutro         |             17 |            3 |                    2 |               1 |              3 |            4 |             6 |            0 |             2 |               1 |                1 |           2 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            0 |             1 |               0 |                1 |           0 |
| Rabbia         |              6 |            0 |                    0 |               4 |              1 |            0 |             0 |            2 |             4 |               2 |                1 |           1 |
| Sorpresa       |             16 |            2 |                    2 |               4 |              2 |            5 |             3 |            3 |             6 |               5 |                2 |           1 |
| Tristezza      |              2 |            1 |                    0 |               1 |              0 |            0 |             0 |            1 |             2 |               1 |                1 |           0 |

#### Case 2: Step 2 Raw (No corr) (Checkpoint: `emotion_textual_step2_raw`, Eval Prompt: `None (Raw)`)

| Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Blind Text (ClassifierChain)     | raw       |     0.3784 |     0.4492 |        0.4539 |        0.3488 |
| Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4761 |     0.4938 |        0.4867 |        0.3178 |
| Blind Text (ClassifierChain)     | std       |     0.3659 |     0.4167 |        0.4724 |        0.2093 |
| Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4092 |     0.4619 |        0.4887 |        0.2481 |

##### Run: `Case 2` | `Blind Text (ClassifierChain)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.6000 |   0.2727 |     0.3750 |        11 |
| Anticipazione |      0.4667 |   0.7000 |     0.5600 |        10 |
| Disgusto      |      0.4286 |   0.4737 |     0.4500 |        19 |
| Fiducia       |      0.6970 |   0.5476 |     0.6133 |        42 |
| Gioia         |      0.4186 |   0.6207 |     0.5000 |        29 |
| Neutro        |      0.2222 |   0.2353 |     0.2286 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.3750 |   0.5000 |     0.4286 |         6 |
| Sorpresa      |      0.2105 |   0.2500 |     0.2286 |        16 |
| Tristezza     |      0.2500 |   1.0000 |     0.4000 |         2 |
| micro avg     |      0.4244 |   0.4771 |     0.4492 |       153 |
| macro avg     |      0.3669 |   0.4600 |     0.3784 |       153 |
| weighted avg  |      0.4622 |   0.4771 |     0.4539 |       153 |
| samples avg   |      0.4322 |   0.4535 |     0.4318 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  116 |    2 |    8 |    3 |      0.6000 |   0.2727 |     0.3750 |        11 |
| Anticipazione |  111 |    8 |    3 |    7 |      0.4667 |   0.7000 |     0.5600 |        10 |
| Disgusto      |   98 |   12 |   10 |    9 |      0.4286 |   0.4737 |     0.4500 |        19 |
| Fiducia       |   77 |   10 |   19 |   23 |      0.6970 |   0.5476 |     0.6133 |        42 |
| Gioia         |   75 |   25 |   11 |   18 |      0.4186 |   0.6207 |     0.5000 |        29 |
| Neutro        |   98 |   14 |   13 |    4 |      0.2222 |   0.2353 |     0.2286 |        17 |
| Paura         |  126 |    2 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  118 |    5 |    3 |    3 |      0.3750 |   0.5000 |     0.4286 |         6 |
| Sorpresa      |   98 |   15 |   12 |    4 |      0.2105 |   0.2500 |     0.2286 |        16 |
| Tristezza     |  121 |    6 |    0 |    2 |      0.2500 |   1.0000 |     0.4000 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            3 |                    0 |               0 |              7 |            5 |             1 |            0 |             0 |               2 |                0 |           0 |
| Anticipazione  |             10 |            0 |                    7 |               0 |              0 |            4 |             0 |            0 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            0 |                    1 |               9 |              0 |            3 |             4 |            0 |             6 |               5 |                1 |           0 |
| Fiducia        |             42 |            4 |                    1 |               1 |             23 |           17 |             3 |            0 |             0 |               3 |                0 |           1 |
| Gioia          |             29 |            3 |                    3 |               2 |              4 |           18 |             5 |            0 |             0 |               2 |                1 |           2 |
| Neutro         |             17 |            0 |                    3 |               2 |              3 |            5 |             4 |            0 |             0 |               5 |                2 |           1 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            0 |             1 |               0 |                1 |           0 |
| Rabbia         |              6 |            0 |                    0 |               5 |              0 |            1 |             0 |            1 |             3 |               0 |                1 |           0 |
| Sorpresa       |             16 |            0 |                    0 |               4 |              3 |            5 |             3 |            2 |             0 |               4 |                3 |           0 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             0 |               1 |                2 |           0 |

##### Run: `Case 2` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2857 |   0.3636 |     0.3200 |        11 |
| Anticipazione |      0.5833 |   0.7000 |     0.6364 |        10 |
| Disgusto      |      0.5000 |   0.3684 |     0.4242 |        19 |
| Fiducia       |      0.6585 |   0.6429 |     0.6506 |        42 |
| Gioia         |      0.5000 |   0.7241 |     0.5915 |        29 |
| Neutro        |      0.3333 |   0.2353 |     0.2759 |        17 |
| Paura         |      0.3333 |   1.0000 |     0.5000 |         1 |
| Rabbia        |      0.3000 |   0.5000 |     0.3750 |         6 |
| Sorpresa      |      0.1875 |   0.1875 |     0.1875 |        16 |
| Tristezza     |      0.6667 |   1.0000 |     0.8000 |         2 |
| micro avg     |      0.4731 |   0.5163 |     0.4938 |       153 |
| macro avg     |      0.4348 |   0.5722 |     0.4761 |       153 |
| weighted avg  |      0.4756 |   0.5163 |     0.4867 |       153 |
| samples avg   |      0.4744 |   0.5284 |     0.4793 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  108 |   10 |    7 |    4 |      0.2857 |   0.3636 |     0.3200 |        11 |
| Anticipazione |  114 |    5 |    3 |    7 |      0.5833 |   0.7000 |     0.6364 |        10 |
| Disgusto      |  103 |    7 |   12 |    7 |      0.5000 |   0.3684 |     0.4242 |        19 |
| Fiducia       |   73 |   14 |   15 |   27 |      0.6585 |   0.6429 |     0.6506 |        42 |
| Gioia         |   79 |   21 |    8 |   21 |      0.5000 |   0.7241 |     0.5915 |        29 |
| Neutro        |  104 |    8 |   13 |    4 |      0.3333 |   0.2353 |     0.2759 |        17 |
| Paura         |  126 |    2 |    0 |    1 |      0.3333 |   1.0000 |     0.5000 |         1 |
| Rabbia        |  116 |    7 |    3 |    3 |      0.3000 |   0.5000 |     0.3750 |         6 |
| Sorpresa      |  100 |   13 |   13 |    3 |      0.1875 |   0.1875 |     0.1875 |        16 |
| Tristezza     |  126 |    1 |    0 |    2 |      0.6667 |   1.0000 |     0.8000 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            4 |                    0 |               0 |              7 |            3 |             0 |            0 |             0 |               0 |                0 |           1 |
| Anticipazione  |             10 |            1 |                    7 |               0 |              0 |            4 |             0 |            0 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            1 |                    1 |               7 |              0 |            4 |             3 |            0 |             5 |               6 |                0 |           2 |
| Fiducia        |             42 |            3 |                    2 |               0 |             27 |           13 |             1 |            0 |             0 |               0 |                0 |           6 |
| Gioia          |             29 |            5 |                    1 |               0 |              5 |           21 |             4 |            0 |             0 |               4 |                0 |           1 |
| Neutro         |             17 |            1 |                    1 |               2 |              4 |            4 |             4 |            0 |             1 |               3 |                0 |           3 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            1 |             0 |            1 |             1 |               1 |                0 |           0 |
| Rabbia         |              6 |            1 |                    0 |               2 |              0 |            2 |             0 |            1 |             3 |               1 |                1 |           0 |
| Sorpresa       |             16 |            2 |                    0 |               2 |              3 |            5 |             1 |            2 |             2 |               3 |                3 |           0 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             0 |               1 |                2 |           0 |

##### Run: `Case 2` | `Blind Text (ClassifierChain)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.5000 |   0.6364 |     0.5600 |        11 |
| Anticipazione |      0.4118 |   0.7000 |     0.5185 |        10 |
| Disgusto      |      0.3750 |   0.1579 |     0.2222 |        19 |
| Fiducia       |      0.7027 |   0.6190 |     0.6582 |        42 |
| Gioia         |      0.5625 |   0.6207 |     0.5902 |        29 |
| Neutro        |      0.3333 |   0.4118 |     0.3684 |        17 |
| Paura         |      0.0435 |   1.0000 |     0.0833 |         1 |
| Rabbia        |      0.2083 |   0.8333 |     0.3333 |         6 |
| Sorpresa      |      0.1818 |   0.2500 |     0.2105 |        16 |
| Tristezza     |      0.0606 |   1.0000 |     0.1143 |         2 |
| micro avg     |      0.3463 |   0.5229 |     0.4167 |       153 |
| macro avg     |      0.3380 |   0.6229 |     0.3659 |       153 |
| weighted avg  |      0.4742 |   0.5229 |     0.4724 |       153 |
| samples avg   |      0.3850 |   0.5129 |     0.4191 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  111 |    7 |    4 |    7 |      0.5000 |   0.6364 |     0.5600 |        11 |
| Anticipazione |  109 |   10 |    3 |    7 |      0.4118 |   0.7000 |     0.5185 |        10 |
| Disgusto      |  105 |    5 |   16 |    3 |      0.3750 |   0.1579 |     0.2222 |        19 |
| Fiducia       |   76 |   11 |   16 |   26 |      0.7027 |   0.6190 |     0.6582 |        42 |
| Gioia         |   86 |   14 |   11 |   18 |      0.5625 |   0.6207 |     0.5902 |        29 |
| Neutro        |   98 |   14 |   10 |    7 |      0.3333 |   0.4118 |     0.3684 |        17 |
| Paura         |  106 |   22 |    0 |    1 |      0.0435 |   1.0000 |     0.0833 |         1 |
| Rabbia        |  104 |   19 |    1 |    5 |      0.2083 |   0.8333 |     0.3333 |         6 |
| Sorpresa      |   95 |   18 |   12 |    4 |      0.1818 |   0.2500 |     0.2105 |        16 |
| Tristezza     |   96 |   31 |    0 |    2 |      0.0606 |   1.0000 |     0.1143 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            7 |                    1 |               0 |              8 |            3 |             1 |            3 |             0 |               2 |                2 |           0 |
| Anticipazione  |             10 |            0 |                    7 |               0 |              0 |            3 |             1 |            1 |             0 |               0 |                2 |           0 |
| Disgusto       |             19 |            0 |                    1 |               3 |              1 |            3 |             5 |            4 |             8 |               5 |                3 |           1 |
| Fiducia        |             42 |            7 |                    3 |               0 |             26 |           12 |             3 |            7 |             1 |               5 |                7 |           0 |
| Gioia          |             29 |            3 |                    3 |               1 |              6 |           18 |             4 |            1 |             3 |               3 |                9 |           1 |
| Neutro         |             17 |            2 |                    3 |               0 |              1 |            1 |             7 |            4 |             4 |               5 |                7 |           2 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             1 |               0 |                1 |           0 |
| Rabbia         |              6 |            0 |                    0 |               2 |              1 |            1 |             0 |            3 |             5 |               0 |                2 |           0 |
| Sorpresa       |             16 |            0 |                    0 |               2 |              3 |            4 |             2 |            4 |             6 |               4 |                4 |           1 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            2 |             2 |               1 |                2 |           0 |

##### Run: `Case 2` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3077 |   0.7273 |     0.4324 |        11 |
| Anticipazione |      0.3684 |   0.7000 |     0.4828 |        10 |
| Disgusto      |      0.5385 |   0.3684 |     0.4375 |        19 |
| Fiducia       |      0.6087 |   0.6667 |     0.6364 |        42 |
| Gioia         |      0.5135 |   0.6552 |     0.5758 |        29 |
| Neutro        |      0.2727 |   0.5294 |     0.3600 |        17 |
| Paura         |      0.0667 |   1.0000 |     0.1250 |         1 |
| Rabbia        |      0.6000 |   0.5000 |     0.5455 |         6 |
| Sorpresa      |      0.1818 |   0.2500 |     0.2105 |        16 |
| Tristezza     |      0.1667 |   1.0000 |     0.2857 |         2 |
| micro avg     |      0.3860 |   0.5752 |     0.4619 |       153 |
| macro avg     |      0.3625 |   0.6397 |     0.4092 |       153 |
| weighted avg  |      0.4530 |   0.5752 |     0.4887 |       153 |
| samples avg   |      0.4393 |   0.5775 |     0.4721 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  100 |   18 |    3 |    8 |      0.3077 |   0.7273 |     0.4324 |        11 |
| Anticipazione |  107 |   12 |    3 |    7 |      0.3684 |   0.7000 |     0.4828 |        10 |
| Disgusto      |  104 |    6 |   12 |    7 |      0.5385 |   0.3684 |     0.4375 |        19 |
| Fiducia       |   69 |   18 |   14 |   28 |      0.6087 |   0.6667 |     0.6364 |        42 |
| Gioia         |   82 |   18 |   10 |   19 |      0.5135 |   0.6552 |     0.5758 |        29 |
| Neutro        |   88 |   24 |    8 |    9 |      0.2727 |   0.5294 |     0.3600 |        17 |
| Paura         |  114 |   14 |    0 |    1 |      0.0667 |   1.0000 |     0.1250 |         1 |
| Rabbia        |  121 |    2 |    3 |    3 |      0.6000 |   0.5000 |     0.5455 |         6 |
| Sorpresa      |   95 |   18 |   12 |    4 |      0.1818 |   0.2500 |     0.2105 |        16 |
| Tristezza     |  117 |   10 |    0 |    2 |      0.1667 |   1.0000 |     0.2857 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            8 |                    0 |               0 |              8 |            2 |             3 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            2 |                    7 |               0 |              0 |            4 |             1 |            0 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            1 |                    1 |               7 |              0 |            4 |             5 |            3 |             3 |               8 |                2 |           0 |
| Fiducia        |             42 |            9 |                    5 |               1 |             28 |           12 |             7 |            1 |             0 |               2 |                0 |           1 |
| Gioia          |             29 |            9 |                    2 |               1 |              9 |           19 |             6 |            3 |             0 |               3 |                1 |           2 |
| Neutro         |             17 |            2 |                    4 |               2 |              5 |            4 |             9 |            3 |             0 |               5 |                4 |           0 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             1 |               1 |                1 |           0 |
| Rabbia         |              6 |            0 |                    0 |               3 |              0 |            2 |             0 |            2 |             3 |               1 |                2 |           0 |
| Sorpresa       |             16 |            2 |                    0 |               0 |              4 |            5 |             5 |            4 |             0 |               4 |                3 |           0 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            2 |             0 |               1 |                2 |           0 |

#### Case 3: Step 2 Oracle (Train: no rel, Eval: no rel) (Checkpoint: `emotion_textual_corr`, Eval Prompt: `Oracle (no rel)`)

| Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Blind Text (ClassifierChain)     | raw       |     0.4339 |     0.5242 |        0.5262 |        0.3178 |
| Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.5222 |     0.5223 |        0.5275 |        0.3023 |
| Blind Text (ClassifierChain)     | std       |     0.4642 |     0.5254 |        0.5584 |        0.3333 |
| Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.3965 |     0.4787 |        0.5054 |        0.2403 |

##### Run: `Case 3` | `Blind Text (ClassifierChain)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2857 |   0.7273 |     0.4103 |        11 |
| Anticipazione |      0.5556 |   0.5000 |     0.5263 |        10 |
| Disgusto      |      0.7857 |   0.5789 |     0.6667 |        19 |
| Fiducia       |      0.5472 |   0.6905 |     0.6105 |        42 |
| Gioia         |      0.4510 |   0.7931 |     0.5750 |        29 |
| Neutro        |      0.5714 |   0.2353 |     0.3333 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.3750 |   0.5000 |     0.4286 |         6 |
| Sorpresa      |      0.4118 |   0.4375 |     0.4242 |        16 |
| Tristezza     |      0.2222 |   1.0000 |     0.3636 |         2 |
| micro avg     |      0.4646 |   0.6013 |     0.5242 |       153 |
| macro avg     |      0.4206 |   0.5463 |     0.4339 |       153 |
| weighted avg  |      0.5143 |   0.6013 |     0.5262 |       153 |
| samples avg   |      0.4948 |   0.5943 |     0.5194 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |   98 |   20 |    3 |    8 |      0.2857 |   0.7273 |     0.4103 |        11 |
| Anticipazione |  115 |    4 |    5 |    5 |      0.5556 |   0.5000 |     0.5263 |        10 |
| Disgusto      |  107 |    3 |    8 |   11 |      0.7857 |   0.5789 |     0.6667 |        19 |
| Fiducia       |   63 |   24 |   13 |   29 |      0.5472 |   0.6905 |     0.6105 |        42 |
| Gioia         |   72 |   28 |    6 |   23 |      0.4510 |   0.7931 |     0.5750 |        29 |
| Neutro        |  109 |    3 |   13 |    4 |      0.5714 |   0.2353 |     0.3333 |        17 |
| Paura         |  126 |    2 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  118 |    5 |    3 |    3 |      0.3750 |   0.5000 |     0.4286 |         6 |
| Sorpresa      |  103 |   10 |    9 |    7 |      0.4118 |   0.4375 |     0.4242 |        16 |
| Tristezza     |  120 |    7 |    0 |    2 |      0.2222 |   1.0000 |     0.3636 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            8 |                    1 |               0 |              8 |            5 |             0 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    5 |               0 |              1 |            6 |             0 |            0 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            1 |                    1 |              11 |              2 |            2 |             0 |            1 |             2 |               3 |                1 |           2 |
| Fiducia        |             42 |            8 |                    2 |               0 |             29 |           18 |             0 |            0 |             0 |               3 |                0 |           1 |
| Gioia          |             29 |           10 |                    0 |               1 |              7 |           23 |             3 |            0 |             0 |               5 |                1 |           0 |
| Neutro         |             17 |            2 |                    1 |               0 |              9 |            6 |             4 |            0 |             1 |               3 |                3 |           0 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            0 |             1 |               0 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               2 |              0 |            1 |             0 |            2 |             3 |               0 |                2 |           1 |
| Sorpresa       |             16 |            4 |                    0 |               0 |              4 |            7 |             0 |            0 |             3 |               7 |                3 |           0 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            0 |             2 |               2 |                2 |           0 |

##### Run: `Case 3` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3478 |   0.7273 |     0.4706 |        11 |
| Anticipazione |      0.5000 |   0.5000 |     0.5000 |        10 |
| Disgusto      |      0.6190 |   0.6842 |     0.6500 |        19 |
| Fiducia       |      0.6122 |   0.7143 |     0.6593 |        42 |
| Gioia         |      0.5294 |   0.6207 |     0.5714 |        29 |
| Neutro        |      0.1786 |   0.2941 |     0.2222 |        17 |
| Paura         |      1.0000 |   1.0000 |     1.0000 |         1 |
| Rabbia        |      0.4000 |   0.3333 |     0.3636 |         6 |
| Sorpresa      |      0.5000 |   0.3125 |     0.3846 |        16 |
| Tristezza     |      0.3333 |   0.5000 |     0.4000 |         2 |
| micro avg     |      0.4783 |   0.5752 |     0.5223 |       153 |
| macro avg     |      0.5020 |   0.5686 |     0.5222 |       153 |
| weighted avg  |      0.5017 |   0.5752 |     0.5275 |       153 |
| samples avg   |      0.4877 |   0.5866 |     0.5098 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  103 |   15 |    3 |    8 |      0.3478 |   0.7273 |     0.4706 |        11 |
| Anticipazione |  114 |    5 |    5 |    5 |      0.5000 |   0.5000 |     0.5000 |        10 |
| Disgusto      |  102 |    8 |    6 |   13 |      0.6190 |   0.6842 |     0.6500 |        19 |
| Fiducia       |   68 |   19 |   12 |   30 |      0.6122 |   0.7143 |     0.6593 |        42 |
| Gioia         |   84 |   16 |   11 |   18 |      0.5294 |   0.6207 |     0.5714 |        29 |
| Neutro        |   89 |   23 |   12 |    5 |      0.1786 |   0.2941 |     0.2222 |        17 |
| Paura         |  128 |    0 |    0 |    1 |      1.0000 |   1.0000 |     1.0000 |         1 |
| Rabbia        |  120 |    3 |    4 |    2 |      0.4000 |   0.3333 |     0.3636 |         6 |
| Sorpresa      |  108 |    5 |   11 |    5 |      0.5000 |   0.3125 |     0.3846 |        16 |
| Tristezza     |  125 |    2 |    1 |    1 |      0.3333 |   0.5000 |     0.4000 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            8 |                    0 |               0 |              8 |            2 |             5 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    5 |               0 |              0 |            3 |             0 |            0 |             0 |               0 |                0 |           1 |
| Disgusto       |             19 |            1 |                    1 |              13 |              1 |            1 |             3 |            0 |             3 |               1 |                1 |           0 |
| Fiducia        |             42 |           10 |                    2 |               0 |             30 |           13 |             7 |            0 |             0 |               1 |                0 |           3 |
| Gioia          |             29 |            4 |                    0 |               3 |              7 |           18 |             7 |            0 |             0 |               4 |                0 |           1 |
| Neutro         |             17 |            2 |                    2 |               0 |              4 |            4 |             5 |            0 |             0 |               1 |                0 |           4 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               0 |                1 |           0 |
| Rabbia         |              6 |            1 |                    0 |               3 |              1 |            1 |             1 |            0 |             2 |               0 |                1 |           1 |
| Sorpresa       |             16 |            1 |                    0 |               4 |              4 |            5 |             3 |            0 |             1 |               5 |                1 |           1 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            0 |             0 |               1 |                1 |           0 |

##### Run: `Case 3` | `Blind Text (ClassifierChain)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3889 |   0.6364 |     0.4828 |        11 |
| Anticipazione |      0.5000 |   0.6000 |     0.5455 |        10 |
| Disgusto      |      0.6364 |   0.7368 |     0.6829 |        19 |
| Fiducia       |      0.6571 |   0.5476 |     0.5974 |        42 |
| Gioia         |      0.5128 |   0.6897 |     0.5882 |        29 |
| Neutro        |      0.4286 |   0.5294 |     0.4737 |        17 |
| Paura         |      0.0500 |   1.0000 |     0.0952 |         1 |
| Rabbia        |      0.4286 |   0.5000 |     0.4615 |         6 |
| Sorpresa      |      0.4737 |   0.5625 |     0.5143 |        16 |
| Tristezza     |      0.1250 |   0.5000 |     0.2000 |         2 |
| micro avg     |      0.4627 |   0.6078 |     0.5254 |       153 |
| macro avg     |      0.4201 |   0.6302 |     0.4642 |       153 |
| weighted avg  |      0.5332 |   0.6078 |     0.5584 |       153 |
| samples avg   |      0.4884 |   0.5943 |     0.5189 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  107 |   11 |    4 |    7 |      0.3889 |   0.6364 |     0.4828 |        11 |
| Anticipazione |  113 |    6 |    4 |    6 |      0.5000 |   0.6000 |     0.5455 |        10 |
| Disgusto      |  102 |    8 |    5 |   14 |      0.6364 |   0.7368 |     0.6829 |        19 |
| Fiducia       |   75 |   12 |   19 |   23 |      0.6571 |   0.5476 |     0.5974 |        42 |
| Gioia         |   81 |   19 |    9 |   20 |      0.5128 |   0.6897 |     0.5882 |        29 |
| Neutro        |  100 |   12 |    8 |    9 |      0.4286 |   0.5294 |     0.4737 |        17 |
| Paura         |  109 |   19 |    0 |    1 |      0.0500 |   1.0000 |     0.0952 |         1 |
| Rabbia        |  119 |    4 |    3 |    3 |      0.4286 |   0.5000 |     0.4615 |         6 |
| Sorpresa      |  103 |   10 |    7 |    9 |      0.4737 |   0.5625 |     0.5143 |        16 |
| Tristezza     |  120 |    7 |    1 |    1 |      0.1250 |   0.5000 |     0.2000 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            7 |                    0 |               0 |              7 |            5 |             2 |            3 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    6 |               0 |              0 |            4 |             0 |            0 |             2 |               0 |                0 |           0 |
| Disgusto       |             19 |            0 |                    1 |              14 |              1 |            1 |             1 |            4 |             2 |               1 |                1 |           0 |
| Fiducia        |             42 |            9 |                    1 |               1 |             23 |           17 |             4 |            4 |             1 |               3 |                1 |           3 |
| Gioia          |             29 |            8 |                    2 |               2 |              5 |           20 |             3 |            4 |             0 |               8 |                1 |           1 |
| Neutro         |             17 |            1 |                    1 |               1 |              3 |            2 |             9 |            4 |             0 |               3 |                2 |           2 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               0 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               4 |              0 |            0 |             0 |            3 |             3 |               1 |                2 |           0 |
| Sorpresa       |             16 |            0 |                    1 |               1 |              1 |            6 |             3 |            4 |             0 |               9 |                2 |           1 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            2 |             0 |               2 |                1 |           0 |

##### Run: `Case 3` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.5000 |   0.4545 |     0.4762 |        11 |
| Anticipazione |      0.4000 |   0.6000 |     0.4800 |        10 |
| Disgusto      |      0.6111 |   0.5789 |     0.5946 |        19 |
| Fiducia       |      0.5455 |   0.7143 |     0.6186 |        42 |
| Gioia         |      0.5143 |   0.6207 |     0.5625 |        29 |
| Neutro        |      0.2647 |   0.5294 |     0.3529 |        17 |
| Paura         |      0.1250 |   1.0000 |     0.2222 |         1 |
| Rabbia        |      0.0769 |   0.1667 |     0.1053 |         6 |
| Sorpresa      |      0.3478 |   0.5000 |     0.4103 |        16 |
| Tristezza     |      0.0833 |   0.5000 |     0.1429 |         2 |
| micro avg     |      0.4036 |   0.5882 |     0.4787 |       153 |
| macro avg     |      0.3469 |   0.5665 |     0.3965 |       153 |
| weighted avg  |      0.4559 |   0.5882 |     0.5054 |       153 |
| samples avg   |      0.4501 |   0.5995 |     0.4868 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  113 |    5 |    6 |    5 |      0.5000 |   0.4545 |     0.4762 |        11 |
| Anticipazione |  110 |    9 |    4 |    6 |      0.4000 |   0.6000 |     0.4800 |        10 |
| Disgusto      |  103 |    7 |    8 |   11 |      0.6111 |   0.5789 |     0.5946 |        19 |
| Fiducia       |   62 |   25 |   12 |   30 |      0.5455 |   0.7143 |     0.6186 |        42 |
| Gioia         |   83 |   17 |   11 |   18 |      0.5143 |   0.6207 |     0.5625 |        29 |
| Neutro        |   87 |   25 |    8 |    9 |      0.2647 |   0.5294 |     0.3529 |        17 |
| Paura         |  121 |    7 |    0 |    1 |      0.1250 |   1.0000 |     0.2222 |         1 |
| Rabbia        |  111 |   12 |    5 |    1 |      0.0769 |   0.1667 |     0.1053 |         6 |
| Sorpresa      |   98 |   15 |    8 |    8 |      0.3478 |   0.5000 |     0.4103 |        16 |
| Tristezza     |  116 |   11 |    1 |    1 |      0.0833 |   0.5000 |     0.1429 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            5 |                    0 |               0 |              7 |            1 |             6 |            1 |             1 |               1 |                0 |           0 |
| Anticipazione  |             10 |            0 |                    6 |               0 |              1 |            3 |             1 |            0 |             0 |               1 |                0 |           1 |
| Disgusto       |             19 |            0 |                    2 |              11 |              1 |            1 |             4 |            1 |             1 |               1 |                2 |           2 |
| Fiducia        |             42 |            4 |                    2 |               0 |             30 |           14 |            11 |            0 |             3 |               6 |                0 |           1 |
| Gioia          |             29 |            3 |                    1 |               5 |             12 |           18 |             4 |            1 |             3 |               7 |                3 |           1 |
| Neutro         |             17 |            1 |                    3 |               0 |              5 |            3 |             9 |            2 |             5 |               4 |                3 |           1 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               0 |                1 |           0 |
| Rabbia         |              6 |            1 |                    0 |               2 |              1 |            1 |             0 |            1 |             1 |               0 |                2 |           0 |
| Sorpresa       |             16 |            1 |                    1 |               2 |              5 |            6 |             3 |            2 |             1 |               8 |                2 |           0 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             0 |               1 |                1 |           0 |

#### Case 4: Step 2 Oracle (Train: no rel, Eval: +rel) (Checkpoint: `emotion_textual_corr`, Eval Prompt: `Oracle (+ rel)`)

| Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Blind Text (ClassifierChain)     | raw       |     0.4322 |     0.5132 |        0.5083 |        0.4031 |
| Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.3931 |     0.4920 |        0.4771 |        0.3411 |
| Blind Text (ClassifierChain)     | std       |     0.4518 |     0.5127 |        0.5750 |        0.2636 |
| Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4345 |     0.4929 |        0.5193 |        0.2791 |

##### Run: `Case 4` | `Blind Text (ClassifierChain)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3333 |   0.3636 |     0.3478 |        11 |
| Anticipazione |      0.5714 |   0.8000 |     0.6667 |        10 |
| Disgusto      |      0.5556 |   0.7895 |     0.6522 |        19 |
| Fiducia       |      0.6053 |   0.5476 |     0.5750 |        42 |
| Gioia         |      0.6667 |   0.4828 |     0.5600 |        29 |
| Neutro        |      0.3158 |   0.3529 |     0.3333 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.5714 |   0.6667 |     0.6154 |         6 |
| Sorpresa      |      0.6000 |   0.1875 |     0.2857 |        16 |
| Tristezza     |      0.2000 |   0.5000 |     0.2857 |         2 |
| micro avg     |      0.5166 |   0.5098 |     0.5132 |       153 |
| macro avg     |      0.4419 |   0.4691 |     0.4322 |       153 |
| weighted avg  |      0.5457 |   0.5098 |     0.5083 |       153 |
| samples avg   |      0.5032 |   0.5000 |     0.4889 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  110 |    8 |    7 |    4 |      0.3333 |   0.3636 |     0.3478 |        11 |
| Anticipazione |  113 |    6 |    2 |    8 |      0.5714 |   0.8000 |     0.6667 |        10 |
| Disgusto      |   98 |   12 |    4 |   15 |      0.5556 |   0.7895 |     0.6522 |        19 |
| Fiducia       |   72 |   15 |   19 |   23 |      0.6053 |   0.5476 |     0.5750 |        42 |
| Gioia         |   93 |    7 |   15 |   14 |      0.6667 |   0.4828 |     0.5600 |        29 |
| Neutro        |   99 |   13 |   11 |    6 |      0.3158 |   0.3529 |     0.3333 |        17 |
| Paura         |  125 |    3 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  120 |    3 |    2 |    4 |      0.5714 |   0.6667 |     0.6154 |         6 |
| Sorpresa      |  111 |    2 |   13 |    3 |      0.6000 |   0.1875 |     0.2857 |        16 |
| Tristezza     |  123 |    4 |    1 |    1 |      0.2000 |   0.5000 |     0.2857 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            4 |                    0 |               0 |              7 |            2 |             2 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            0 |                    8 |               0 |              1 |            1 |             1 |            0 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            0 |                    1 |              15 |              1 |            1 |             0 |            1 |             3 |               1 |                0 |           1 |
| Fiducia        |             42 |            7 |                    4 |               3 |             23 |            9 |             2 |            0 |             0 |               0 |                0 |           2 |
| Gioia          |             29 |            4 |                    1 |               1 |              5 |           14 |             5 |            0 |             0 |               4 |                2 |           2 |
| Neutro         |             17 |            0 |                    2 |               3 |              2 |            0 |             6 |            0 |             1 |               0 |                1 |           3 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            0 |             1 |            0 |             0 |               0 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               3 |              1 |            0 |             1 |            1 |             4 |               0 |                1 |           1 |
| Sorpresa       |             16 |            1 |                    0 |               5 |              3 |            5 |             2 |            2 |             2 |               3 |                3 |           0 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               0 |                1 |           0 |

##### Run: `Case 4` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3333 |   0.1818 |     0.2353 |        11 |
| Anticipazione |      0.5556 |   0.5000 |     0.5263 |        10 |
| Disgusto      |      0.5000 |   0.3158 |     0.3871 |        19 |
| Fiducia       |      0.5357 |   0.7143 |     0.6122 |        42 |
| Gioia         |      0.5278 |   0.6552 |     0.5846 |        29 |
| Neutro        |      0.3684 |   0.4118 |     0.3889 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.4000 |   0.3333 |     0.3636 |         6 |
| Sorpresa      |      0.3571 |   0.3125 |     0.3333 |        16 |
| Tristezza     |      0.5000 |   0.5000 |     0.5000 |         2 |
| micro avg     |      0.4813 |   0.5033 |     0.4920 |       153 |
| macro avg     |      0.4078 |   0.3925 |     0.3931 |       153 |
| weighted avg  |      0.4700 |   0.5033 |     0.4771 |       153 |
| samples avg   |      0.4690 |   0.5116 |     0.4734 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  114 |    4 |    9 |    2 |      0.3333 |   0.1818 |     0.2353 |        11 |
| Anticipazione |  115 |    4 |    5 |    5 |      0.5556 |   0.5000 |     0.5263 |        10 |
| Disgusto      |  104 |    6 |   13 |    6 |      0.5000 |   0.3158 |     0.3871 |        19 |
| Fiducia       |   61 |   26 |   12 |   30 |      0.5357 |   0.7143 |     0.6122 |        42 |
| Gioia         |   83 |   17 |   10 |   19 |      0.5278 |   0.6552 |     0.5846 |        29 |
| Neutro        |  100 |   12 |   10 |    7 |      0.3684 |   0.4118 |     0.3889 |        17 |
| Paura         |  127 |    1 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  120 |    3 |    4 |    2 |      0.4000 |   0.3333 |     0.3636 |         6 |
| Sorpresa      |  104 |    9 |   11 |    5 |      0.3571 |   0.3125 |     0.3333 |        16 |
| Tristezza     |  126 |    1 |    1 |    1 |      0.5000 |   0.5000 |     0.5000 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            2 |                    0 |               0 |              7 |            2 |             0 |            0 |             0 |               0 |                0 |           1 |
| Anticipazione  |             10 |            0 |                    5 |               0 |              2 |            5 |             0 |            0 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            0 |                    1 |               6 |              1 |            1 |             3 |            0 |             3 |               1 |                1 |           5 |
| Fiducia        |             42 |            1 |                    1 |               0 |             30 |           12 |             1 |            0 |             0 |               2 |                0 |           4 |
| Gioia          |             29 |            2 |                    0 |               1 |              8 |           19 |             6 |            0 |             0 |               6 |                0 |           0 |
| Neutro         |             17 |            0 |                    1 |               2 |              6 |            2 |             7 |            0 |             1 |               3 |                0 |           2 |
| Paura          |              1 |            0 |                    0 |               0 |              1 |            1 |             0 |            0 |             0 |               0 |                0 |           0 |
| Rabbia         |              6 |            1 |                    0 |               3 |              1 |            2 |             0 |            0 |             2 |               0 |                1 |           1 |
| Sorpresa       |             16 |            1 |                    1 |               2 |              6 |            6 |             4 |            1 |             1 |               5 |                1 |           1 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               1 |                1 |           0 |

##### Run: `Case 4` | `Blind Text (ClassifierChain)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3529 |   0.5455 |     0.4286 |        11 |
| Anticipazione |      0.5333 |   0.8000 |     0.6400 |        10 |
| Disgusto      |      0.7647 |   0.6842 |     0.7222 |        19 |
| Fiducia       |      0.6000 |   0.7143 |     0.6522 |        42 |
| Gioia         |      0.5405 |   0.6897 |     0.6061 |        29 |
| Neutro        |      0.3448 |   0.5882 |     0.4348 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.3750 |   0.5000 |     0.4286 |         6 |
| Sorpresa      |      0.4500 |   0.5625 |     0.5000 |        16 |
| Tristezza     |      0.0556 |   1.0000 |     0.1053 |         2 |
| micro avg     |      0.4191 |   0.6601 |     0.5127 |       153 |
| macro avg     |      0.4017 |   0.6084 |     0.4518 |       153 |
| weighted avg  |      0.5232 |   0.6601 |     0.5750 |       153 |
| samples avg   |      0.4625 |   0.6408 |     0.5178 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  107 |   11 |    5 |    6 |      0.3529 |   0.5455 |     0.4286 |        11 |
| Anticipazione |  112 |    7 |    2 |    8 |      0.5333 |   0.8000 |     0.6400 |        10 |
| Disgusto      |  106 |    4 |    6 |   13 |      0.7647 |   0.6842 |     0.7222 |        19 |
| Fiducia       |   67 |   20 |   12 |   30 |      0.6000 |   0.7143 |     0.6522 |        42 |
| Gioia         |   83 |   17 |    9 |   20 |      0.5405 |   0.6897 |     0.6061 |        29 |
| Neutro        |   93 |   19 |    7 |   10 |      0.3448 |   0.5882 |     0.4348 |        17 |
| Paura         |  116 |   12 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  118 |    5 |    3 |    3 |      0.3750 |   0.5000 |     0.4286 |         6 |
| Sorpresa      |  102 |   11 |    7 |    9 |      0.4500 |   0.5625 |     0.5000 |        16 |
| Tristezza     |   93 |   34 |    0 |    2 |      0.0556 |   1.0000 |     0.1053 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            6 |                    0 |               0 |              8 |            3 |             1 |            1 |             0 |               1 |                2 |           0 |
| Anticipazione  |             10 |            1 |                    8 |               0 |              2 |            3 |             1 |            0 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            1 |                    1 |              13 |              2 |            2 |             2 |            2 |             3 |               2 |               13 |           0 |
| Fiducia        |             42 |            8 |                    5 |               2 |             30 |           15 |             4 |            2 |             1 |               3 |                3 |           0 |
| Gioia          |             29 |            3 |                    1 |               1 |              9 |           20 |             7 |            4 |             0 |               9 |               12 |           0 |
| Neutro         |             17 |            2 |                    2 |               0 |              4 |            2 |            10 |            2 |             1 |               3 |                3 |           1 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            0 |             1 |            0 |             0 |               0 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               3 |              1 |            0 |             1 |            2 |             3 |               1 |                5 |           0 |
| Sorpresa       |             16 |            0 |                    0 |               0 |              4 |            6 |             4 |            2 |             3 |               9 |                8 |           0 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             2 |               2 |                2 |           0 |

##### Run: `Case 4` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3000 |   0.5455 |     0.3871 |        11 |
| Anticipazione |      0.4545 |   0.5000 |     0.4762 |        10 |
| Disgusto      |      0.5000 |   0.5263 |     0.5128 |        19 |
| Fiducia       |      0.6364 |   0.6667 |     0.6512 |        42 |
| Gioia         |      0.4688 |   0.5172 |     0.4918 |        29 |
| Neutro        |      0.3913 |   0.5294 |     0.4500 |        17 |
| Paura         |      0.0556 |   1.0000 |     0.1053 |         1 |
| Rabbia        |      0.4000 |   0.6667 |     0.5000 |         6 |
| Sorpresa      |      0.4706 |   0.5000 |     0.4848 |        16 |
| Tristezza     |      0.2000 |   0.5000 |     0.2857 |         2 |
| micro avg     |      0.4350 |   0.5686 |     0.4929 |       153 |
| macro avg     |      0.3877 |   0.5952 |     0.4345 |       153 |
| weighted avg  |      0.4883 |   0.5686 |     0.5193 |       153 |
| samples avg   |      0.4561 |   0.5711 |     0.4866 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  104 |   14 |    5 |    6 |      0.3000 |   0.5455 |     0.3871 |        11 |
| Anticipazione |  113 |    6 |    5 |    5 |      0.4545 |   0.5000 |     0.4762 |        10 |
| Disgusto      |  100 |   10 |    9 |   10 |      0.5000 |   0.5263 |     0.5128 |        19 |
| Fiducia       |   71 |   16 |   14 |   28 |      0.6364 |   0.6667 |     0.6512 |        42 |
| Gioia         |   83 |   17 |   14 |   15 |      0.4688 |   0.5172 |     0.4918 |        29 |
| Neutro        |   98 |   14 |    8 |    9 |      0.3913 |   0.5294 |     0.4500 |        17 |
| Paura         |  111 |   17 |    0 |    1 |      0.0556 |   1.0000 |     0.1053 |         1 |
| Rabbia        |  117 |    6 |    2 |    4 |      0.4000 |   0.6667 |     0.5000 |         6 |
| Sorpresa      |  104 |    9 |    8 |    8 |      0.4706 |   0.5000 |     0.4848 |        16 |
| Tristezza     |  123 |    4 |    1 |    1 |      0.2000 |   0.5000 |     0.2857 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            6 |                    0 |               0 |              6 |            2 |             1 |            2 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    5 |               0 |              1 |            7 |             0 |            0 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            0 |                    1 |              10 |              1 |            0 |             2 |            1 |             3 |               3 |                0 |           2 |
| Fiducia        |             42 |            7 |                    2 |               3 |             28 |           13 |             2 |            3 |             1 |               1 |                0 |           1 |
| Gioia          |             29 |            8 |                    0 |               6 |              8 |           15 |             5 |            4 |             1 |               7 |                2 |           0 |
| Neutro         |             17 |            2 |                    2 |               0 |              4 |            3 |             9 |            3 |             1 |               2 |                1 |           0 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            0 |             1 |            1 |             1 |               0 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               2 |              0 |            1 |             1 |            2 |             4 |               1 |                1 |           0 |
| Sorpresa       |             16 |            1 |                    1 |               3 |              2 |            4 |             3 |            3 |             4 |               8 |                2 |           0 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             2 |               1 |                1 |           0 |

#### Case 5: Step 2 Oracle (Train: +rel, Eval: no rel) (Checkpoint: `emotion_textual_corr_with_rel`, Eval Prompt: `Oracle (no rel)`)

| Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Blind Text (ClassifierChain)     | raw       |     0.4461 |     0.5360 |        0.5465 |        0.4109 |
| Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.5431 |     0.5515 |        0.5573 |        0.3178 |
| Blind Text (ClassifierChain)     | std       |     0.4344 |     0.5013 |        0.5472 |        0.3178 |
| Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4460 |     0.5282 |        0.5501 |        0.3643 |

##### Run: `Case 5` | `Blind Text (ClassifierChain)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.4000 |   0.5455 |     0.4615 |        11 |
| Anticipazione |      0.6250 |   0.5000 |     0.5556 |        10 |
| Disgusto      |      0.6364 |   0.7368 |     0.6829 |        19 |
| Fiducia       |      0.5283 |   0.6667 |     0.5895 |        42 |
| Gioia         |      0.5455 |   0.6207 |     0.5806 |        29 |
| Neutro        |      0.3548 |   0.6471 |     0.4583 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.3333 |   0.5000 |     0.4000 |         6 |
| Sorpresa      |      0.5385 |   0.4375 |     0.4828 |        16 |
| Tristezza     |      0.1667 |   0.5000 |     0.2500 |         2 |
| micro avg     |      0.4794 |   0.6078 |     0.5360 |       153 |
| macro avg     |      0.4128 |   0.5154 |     0.4461 |       153 |
| weighted avg  |      0.5080 |   0.6078 |     0.5465 |       153 |
| samples avg   |      0.5306 |   0.6021 |     0.5475 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  109 |    9 |    5 |    6 |      0.4000 |   0.5455 |     0.4615 |        11 |
| Anticipazione |  116 |    3 |    5 |    5 |      0.6250 |   0.5000 |     0.5556 |        10 |
| Disgusto      |  102 |    8 |    5 |   14 |      0.6364 |   0.7368 |     0.6829 |        19 |
| Fiducia       |   62 |   25 |   14 |   28 |      0.5283 |   0.6667 |     0.5895 |        42 |
| Gioia         |   85 |   15 |   11 |   18 |      0.5455 |   0.6207 |     0.5806 |        29 |
| Neutro        |   92 |   20 |    6 |   11 |      0.3548 |   0.6471 |     0.4583 |        17 |
| Paura         |  124 |    4 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  117 |    6 |    3 |    3 |      0.3333 |   0.5000 |     0.4000 |         6 |
| Sorpresa      |  107 |    6 |    9 |    7 |      0.5385 |   0.4375 |     0.4828 |        16 |
| Tristezza     |  122 |    5 |    1 |    1 |      0.1667 |   0.5000 |     0.2500 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            6 |                    0 |               0 |              9 |            2 |             0 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    5 |               0 |              0 |            4 |             2 |            0 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            0 |                    1 |              14 |              2 |            1 |             5 |            2 |             4 |               2 |                1 |           0 |
| Fiducia        |             42 |            8 |                    1 |               3 |             28 |           13 |             3 |            0 |             0 |               1 |                0 |           0 |
| Gioia          |             29 |            6 |                    0 |               1 |              8 |           18 |             6 |            0 |             0 |               5 |                1 |           0 |
| Neutro         |             17 |            0 |                    1 |               1 |              6 |            3 |            11 |            0 |             1 |               2 |                2 |           0 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            0 |             1 |               0 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               4 |              2 |            1 |             1 |            1 |             3 |               0 |                1 |           0 |
| Sorpresa       |             16 |            1 |                    0 |               0 |              6 |            6 |             4 |            2 |             3 |               7 |                2 |           0 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             2 |               1 |                1 |           0 |

##### Run: `Case 5` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3182 |   0.6364 |     0.4242 |        11 |
| Anticipazione |      0.6000 |   0.6000 |     0.6000 |        10 |
| Disgusto      |      0.5789 |   0.5789 |     0.5789 |        19 |
| Fiducia       |      0.6977 |   0.7143 |     0.7059 |        42 |
| Gioia         |      0.5000 |   0.6207 |     0.5538 |        29 |
| Neutro        |      0.4286 |   0.5294 |     0.4737 |        17 |
| Paura         |      1.0000 |   1.0000 |     1.0000 |         1 |
| Rabbia        |      0.2857 |   0.3333 |     0.3077 |         6 |
| Sorpresa      |      0.4000 |   0.3750 |     0.3871 |        16 |
| Tristezza     |      0.3333 |   0.5000 |     0.4000 |         2 |
| micro avg     |      0.5141 |   0.5948 |     0.5515 |       153 |
| macro avg     |      0.5142 |   0.5888 |     0.5431 |       153 |
| weighted avg  |      0.5318 |   0.5948 |     0.5573 |       153 |
| samples avg   |      0.4822 |   0.5969 |     0.5127 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  103 |   15 |    4 |    7 |      0.3182 |   0.6364 |     0.4242 |        11 |
| Anticipazione |  115 |    4 |    4 |    6 |      0.6000 |   0.6000 |     0.6000 |        10 |
| Disgusto      |  102 |    8 |    8 |   11 |      0.5789 |   0.5789 |     0.5789 |        19 |
| Fiducia       |   74 |   13 |   12 |   30 |      0.6977 |   0.7143 |     0.7059 |        42 |
| Gioia         |   82 |   18 |   11 |   18 |      0.5000 |   0.6207 |     0.5538 |        29 |
| Neutro        |  100 |   12 |    8 |    9 |      0.4286 |   0.5294 |     0.4737 |        17 |
| Paura         |  128 |    0 |    0 |    1 |      1.0000 |   1.0000 |     1.0000 |         1 |
| Rabbia        |  118 |    5 |    4 |    2 |      0.2857 |   0.3333 |     0.3077 |         6 |
| Sorpresa      |  104 |    9 |   10 |    6 |      0.4000 |   0.3750 |     0.3871 |        16 |
| Tristezza     |  125 |    2 |    1 |    1 |      0.3333 |   0.5000 |     0.4000 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            7 |                    0 |               0 |              9 |            2 |             1 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    6 |               0 |              0 |            6 |             0 |            0 |             0 |               0 |                0 |           1 |
| Disgusto       |             19 |            0 |                    1 |              11 |              1 |            0 |             2 |            0 |             5 |               5 |                1 |           0 |
| Fiducia        |             42 |           10 |                    1 |               0 |             30 |           13 |             1 |            0 |             0 |               1 |                0 |           5 |
| Gioia          |             29 |            2 |                    0 |               2 |              9 |           18 |             7 |            0 |             0 |               3 |                0 |           4 |
| Neutro         |             17 |            3 |                    2 |               2 |              3 |            4 |             9 |            0 |             1 |               2 |                0 |           0 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               1 |                1 |           0 |
| Rabbia         |              6 |            1 |                    0 |               3 |              0 |            0 |             0 |            0 |             2 |               1 |                1 |           2 |
| Sorpresa       |             16 |            1 |                    0 |               2 |              2 |            6 |             3 |            0 |             0 |               6 |                1 |           2 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            1 |             0 |            0 |             0 |               1 |                1 |           0 |

##### Run: `Case 5` | `Blind Text (ClassifierChain)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3478 |   0.7273 |     0.4706 |        11 |
| Anticipazione |      0.6000 |   0.6000 |     0.6000 |        10 |
| Disgusto      |      0.6250 |   0.7895 |     0.6977 |        19 |
| Fiducia       |      0.6444 |   0.6905 |     0.6667 |        42 |
| Gioia         |      0.4688 |   0.5172 |     0.4918 |        29 |
| Neutro        |      0.3810 |   0.4706 |     0.4211 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.2727 |   0.5000 |     0.3529 |         6 |
| Sorpresa      |      0.3913 |   0.5625 |     0.4615 |        16 |
| Tristezza     |      0.1111 |   0.5000 |     0.1818 |         2 |
| micro avg     |      0.4234 |   0.6144 |     0.5013 |       153 |
| macro avg     |      0.3842 |   0.5358 |     0.4344 |       153 |
| weighted avg  |      0.5030 |   0.6144 |     0.5472 |       153 |
| samples avg   |      0.4673 |   0.6124 |     0.5075 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  103 |   15 |    3 |    8 |      0.3478 |   0.7273 |     0.4706 |        11 |
| Anticipazione |  115 |    4 |    4 |    6 |      0.6000 |   0.6000 |     0.6000 |        10 |
| Disgusto      |  101 |    9 |    4 |   15 |      0.6250 |   0.7895 |     0.6977 |        19 |
| Fiducia       |   71 |   16 |   13 |   29 |      0.6444 |   0.6905 |     0.6667 |        42 |
| Gioia         |   83 |   17 |   14 |   15 |      0.4688 |   0.5172 |     0.4918 |        29 |
| Neutro        |   99 |   13 |    9 |    8 |      0.3810 |   0.4706 |     0.4211 |        17 |
| Paura         |  104 |   24 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  115 |    8 |    3 |    3 |      0.2727 |   0.5000 |     0.3529 |         6 |
| Sorpresa      |   99 |   14 |    7 |    9 |      0.3913 |   0.5625 |     0.4615 |        16 |
| Tristezza     |  119 |    8 |    1 |    1 |      0.1111 |   0.5000 |     0.1818 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            8 |                    0 |               0 |              9 |            2 |             3 |            3 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    6 |               0 |              0 |            7 |             1 |            0 |             0 |               0 |                0 |           1 |
| Disgusto       |             19 |            0 |                    1 |              15 |              1 |            0 |             1 |            4 |             7 |               2 |                2 |           0 |
| Fiducia        |             42 |           11 |                    1 |               2 |             29 |           11 |             2 |            4 |             0 |               2 |                1 |           0 |
| Gioia          |             29 |            9 |                    0 |               1 |              8 |           15 |             3 |           12 |             0 |              10 |                0 |           1 |
| Neutro         |             17 |            1 |                    2 |               1 |              2 |            3 |             8 |            3 |             0 |               5 |                3 |           1 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            0 |             1 |               0 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               5 |              0 |            0 |             1 |            2 |             3 |               0 |                2 |           0 |
| Sorpresa       |             16 |            0 |                    0 |               2 |              3 |            5 |             3 |            4 |             2 |               9 |                2 |           0 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             2 |               2 |                1 |           0 |

##### Run: `Case 5` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2222 |   0.5455 |     0.3158 |        11 |
| Anticipazione |      0.6000 |   0.6000 |     0.6000 |        10 |
| Disgusto      |      0.7059 |   0.6316 |     0.6667 |        19 |
| Fiducia       |      0.6739 |   0.7381 |     0.7045 |        42 |
| Gioia         |      0.5667 |   0.5862 |     0.5763 |        29 |
| Neutro        |      0.3750 |   0.3529 |     0.3636 |        17 |
| Paura         |      0.1250 |   1.0000 |     0.2222 |         1 |
| Rabbia        |      0.3000 |   0.5000 |     0.3750 |         6 |
| Sorpresa      |      0.4615 |   0.3750 |     0.4138 |        16 |
| Tristezza     |      0.1429 |   0.5000 |     0.2222 |         2 |
| micro avg     |      0.4837 |   0.5817 |     0.5282 |       153 |
| macro avg     |      0.4173 |   0.5829 |     0.4460 |       153 |
| weighted avg  |      0.5396 |   0.5817 |     0.5501 |       153 |
| samples avg   |      0.4906 |   0.5827 |     0.5150 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |   97 |   21 |    5 |    6 |      0.2222 |   0.5455 |     0.3158 |        11 |
| Anticipazione |  115 |    4 |    4 |    6 |      0.6000 |   0.6000 |     0.6000 |        10 |
| Disgusto      |  105 |    5 |    7 |   12 |      0.7059 |   0.6316 |     0.6667 |        19 |
| Fiducia       |   72 |   15 |   11 |   31 |      0.6739 |   0.7381 |     0.7045 |        42 |
| Gioia         |   87 |   13 |   12 |   17 |      0.5667 |   0.5862 |     0.5763 |        29 |
| Neutro        |  102 |   10 |   11 |    6 |      0.3750 |   0.3529 |     0.3636 |        17 |
| Paura         |  121 |    7 |    0 |    1 |      0.1250 |   1.0000 |     0.2222 |         1 |
| Rabbia        |  116 |    7 |    3 |    3 |      0.3000 |   0.5000 |     0.3750 |         6 |
| Sorpresa      |  106 |    7 |   10 |    6 |      0.4615 |   0.3750 |     0.4138 |        16 |
| Tristezza     |  121 |    6 |    1 |    1 |      0.1429 |   0.5000 |     0.2222 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            6 |                    0 |               0 |              8 |            2 |             1 |            0 |             0 |               0 |                1 |           0 |
| Anticipazione  |             10 |            1 |                    6 |               0 |              1 |            4 |             0 |            0 |             0 |               0 |                0 |           1 |
| Disgusto       |             19 |            0 |                    1 |              12 |              1 |            0 |             2 |            1 |             3 |               2 |                1 |           2 |
| Fiducia        |             42 |            8 |                    1 |               0 |             31 |           12 |             0 |            1 |             0 |               2 |                0 |           3 |
| Gioia          |             29 |           10 |                    0 |               1 |              7 |           17 |             6 |            1 |             1 |               4 |                0 |           1 |
| Neutro         |             17 |            4 |                    2 |               1 |              4 |            3 |             6 |            2 |             2 |               2 |                2 |           1 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               0 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               4 |              0 |            0 |             0 |            1 |             3 |               0 |                2 |           0 |
| Sorpresa       |             16 |            4 |                    0 |               0 |              3 |            4 |             2 |            2 |             3 |               6 |                2 |           2 |
| Tristezza      |              2 |            1 |                    0 |               0 |              0 |            0 |             0 |            1 |             2 |               1 |                1 |           0 |

#### Case 6: Step 2 Oracle (Train: +rel, Eval: +rel) (Checkpoint: `emotion_textual_corr_with_rel`, Eval Prompt: `Oracle (+ rel)`)

| Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Blind Text (ClassifierChain)     | raw       |     0.4250 |     0.5174 |        0.5368 |        0.3101 |
| Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4744 |     0.4867 |        0.4958 |        0.3256 |
| Blind Text (ClassifierChain)     | std       |     0.4155 |     0.4757 |        0.5160 |        0.3256 |
| Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4153 |     0.4922 |        0.5191 |        0.1938 |

##### Run: `Case 6` | `Blind Text (ClassifierChain)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3077 |   0.3636 |     0.3333 |        11 |
| Anticipazione |      0.5455 |   0.6000 |     0.5714 |        10 |
| Disgusto      |      0.6875 |   0.5789 |     0.6286 |        19 |
| Fiducia       |      0.6667 |   0.6190 |     0.6420 |        42 |
| Gioia         |      0.5000 |   0.6552 |     0.5672 |        29 |
| Neutro        |      0.3214 |   0.5294 |     0.4000 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.2632 |   0.8333 |     0.4000 |         6 |
| Sorpresa      |      0.4706 |   0.5000 |     0.4848 |        16 |
| Tristezza     |      0.1429 |   0.5000 |     0.2222 |         2 |
| micro avg     |      0.4660 |   0.5817 |     0.5174 |       153 |
| macro avg     |      0.3905 |   0.5180 |     0.4250 |       153 |
| weighted avg  |      0.5180 |   0.5817 |     0.5368 |       153 |
| samples avg   |      0.4767 |   0.5711 |     0.5021 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  109 |    9 |    7 |    4 |      0.3077 |   0.3636 |     0.3333 |        11 |
| Anticipazione |  114 |    5 |    4 |    6 |      0.5455 |   0.6000 |     0.5714 |        10 |
| Disgusto      |  105 |    5 |    8 |   11 |      0.6875 |   0.5789 |     0.6286 |        19 |
| Fiducia       |   74 |   13 |   16 |   26 |      0.6667 |   0.6190 |     0.6420 |        42 |
| Gioia         |   81 |   19 |   10 |   19 |      0.5000 |   0.6552 |     0.5672 |        29 |
| Neutro        |   93 |   19 |    8 |    9 |      0.3214 |   0.5294 |     0.4000 |        17 |
| Paura         |  125 |    3 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  109 |   14 |    1 |    5 |      0.2632 |   0.8333 |     0.4000 |         6 |
| Sorpresa      |  104 |    9 |    8 |    8 |      0.4706 |   0.5000 |     0.4848 |        16 |
| Tristezza     |  121 |    6 |    1 |    1 |      0.1429 |   0.5000 |     0.2222 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            4 |                    0 |               0 |              9 |            5 |             3 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            0 |                    6 |               0 |              2 |            5 |             0 |            0 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            0 |                    1 |              11 |              2 |            1 |             3 |            1 |            14 |               1 |                1 |           1 |
| Fiducia        |             42 |            9 |                    2 |               0 |             26 |           15 |             4 |            0 |             0 |               1 |                0 |           3 |
| Gioia          |             29 |            7 |                    0 |               1 |              4 |           19 |             6 |            0 |             0 |               9 |                2 |           0 |
| Neutro         |             17 |            0 |                    2 |               1 |              1 |            3 |             9 |            0 |             0 |               2 |                1 |           1 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            0 |             1 |               0 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               3 |              1 |            0 |             0 |            1 |             5 |               0 |                2 |           0 |
| Sorpresa       |             16 |            0 |                    0 |               0 |              2 |            5 |             5 |            2 |             2 |               8 |                2 |           0 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             1 |               2 |                1 |           0 |

##### Run: `Case 6` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2000 |   0.1818 |     0.1905 |        11 |
| Anticipazione |      0.5556 |   0.5000 |     0.5263 |        10 |
| Disgusto      |      0.7857 |   0.5789 |     0.6667 |        19 |
| Fiducia       |      0.6970 |   0.5476 |     0.6133 |        42 |
| Gioia         |      0.5517 |   0.5517 |     0.5517 |        29 |
| Neutro        |      0.2308 |   0.3529 |     0.2791 |        17 |
| Paura         |      0.5000 |   1.0000 |     0.6667 |         1 |
| Rabbia        |      0.3333 |   0.3333 |     0.3333 |         6 |
| Sorpresa      |      0.3846 |   0.3125 |     0.3448 |        16 |
| Tristezza     |      0.4000 |   1.0000 |     0.5714 |         2 |
| micro avg     |      0.4966 |   0.4771 |     0.4867 |       153 |
| macro avg     |      0.4639 |   0.5359 |     0.4744 |       153 |
| weighted avg  |      0.5316 |   0.4771 |     0.4958 |       153 |
| samples avg   |      0.4419 |   0.4884 |     0.4472 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  110 |    8 |    9 |    2 |      0.2000 |   0.1818 |     0.1905 |        11 |
| Anticipazione |  115 |    4 |    5 |    5 |      0.5556 |   0.5000 |     0.5263 |        10 |
| Disgusto      |  107 |    3 |    8 |   11 |      0.7857 |   0.5789 |     0.6667 |        19 |
| Fiducia       |   77 |   10 |   19 |   23 |      0.6970 |   0.5476 |     0.6133 |        42 |
| Gioia         |   87 |   13 |   13 |   16 |      0.5517 |   0.5517 |     0.5517 |        29 |
| Neutro        |   92 |   20 |   11 |    6 |      0.2308 |   0.3529 |     0.2791 |        17 |
| Paura         |  127 |    1 |    0 |    1 |      0.5000 |   1.0000 |     0.6667 |         1 |
| Rabbia        |  119 |    4 |    4 |    2 |      0.3333 |   0.3333 |     0.3333 |         6 |
| Sorpresa      |  105 |    8 |   11 |    5 |      0.3846 |   0.3125 |     0.3448 |        16 |
| Tristezza     |  124 |    3 |    0 |    2 |      0.4000 |   1.0000 |     0.5714 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            2 |                    0 |               0 |              4 |            2 |             3 |            0 |             0 |               0 |                0 |           2 |
| Anticipazione  |             10 |            0 |                    5 |               0 |              0 |            4 |             1 |            0 |             0 |               0 |                0 |           1 |
| Disgusto       |             19 |            0 |                    1 |              11 |              1 |            0 |             3 |            0 |             4 |               2 |                1 |           2 |
| Fiducia        |             42 |            1 |                    1 |               0 |             23 |           11 |            10 |            0 |             0 |               0 |                0 |           7 |
| Gioia          |             29 |            4 |                    0 |               0 |              3 |           16 |             3 |            0 |             0 |               5 |                0 |           5 |
| Neutro         |             17 |            3 |                    2 |               0 |              3 |            2 |             6 |            0 |             0 |               2 |                0 |           4 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            1 |             0 |            1 |             1 |               1 |                1 |           0 |
| Rabbia         |              6 |            0 |                    0 |               3 |              1 |            0 |             0 |            0 |             2 |               1 |                2 |           0 |
| Sorpresa       |             16 |            1 |                    0 |               0 |              1 |            6 |             3 |            1 |             1 |               5 |                3 |           2 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             1 |               0 |                2 |           0 |

##### Run: `Case 6` | `Blind Text (ClassifierChain)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2121 |   0.6364 |     0.3182 |        11 |
| Anticipazione |      0.5556 |   0.5000 |     0.5263 |        10 |
| Disgusto      |      0.5000 |   0.8421 |     0.6275 |        19 |
| Fiducia       |      0.5814 |   0.5952 |     0.5882 |        42 |
| Gioia         |      0.5152 |   0.5862 |     0.5484 |        29 |
| Neutro        |      0.3333 |   0.5294 |     0.4091 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.3333 |   0.5000 |     0.4000 |         6 |
| Sorpresa      |      0.4000 |   0.6250 |     0.4878 |        16 |
| Tristezza     |      0.1667 |   0.5000 |     0.2500 |         2 |
| micro avg     |      0.3908 |   0.6078 |     0.4757 |       153 |
| macro avg     |      0.3598 |   0.5314 |     0.4155 |       153 |
| weighted avg  |      0.4650 |   0.6078 |     0.5160 |       153 |
| samples avg   |      0.4634 |   0.5943 |     0.5025 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |   92 |   26 |    4 |    7 |      0.2121 |   0.6364 |     0.3182 |        11 |
| Anticipazione |  115 |    4 |    5 |    5 |      0.5556 |   0.5000 |     0.5263 |        10 |
| Disgusto      |   94 |   16 |    3 |   16 |      0.5000 |   0.8421 |     0.6275 |        19 |
| Fiducia       |   69 |   18 |   17 |   25 |      0.5814 |   0.5952 |     0.5882 |        42 |
| Gioia         |   84 |   16 |   12 |   17 |      0.5152 |   0.5862 |     0.5484 |        29 |
| Neutro        |   94 |   18 |    8 |    9 |      0.3333 |   0.5294 |     0.4091 |        17 |
| Paura         |  107 |   21 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  117 |    6 |    3 |    3 |      0.3333 |   0.5000 |     0.4000 |         6 |
| Sorpresa      |   98 |   15 |    6 |   10 |      0.4000 |   0.6250 |     0.4878 |        16 |
| Tristezza     |  122 |    5 |    1 |    1 |      0.1667 |   0.5000 |     0.2500 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            7 |                    0 |               0 |              7 |            4 |             1 |            1 |             0 |               1 |                1 |           0 |
| Anticipazione  |             10 |            1 |                    5 |               0 |              2 |            4 |             1 |            0 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            1 |                    1 |              16 |              2 |            1 |             1 |            3 |             6 |               1 |                0 |           0 |
| Fiducia        |             42 |           13 |                    2 |               3 |             25 |           15 |             3 |            2 |             1 |               5 |                1 |           0 |
| Gioia          |             29 |           17 |                    0 |               3 |              4 |           17 |             9 |           10 |             0 |              12 |                0 |           0 |
| Neutro         |             17 |            3 |                    1 |               2 |              3 |            2 |             9 |            4 |             0 |               3 |                2 |           1 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            0 |             1 |               0 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               5 |              2 |            0 |             0 |            3 |             3 |               1 |                1 |           0 |
| Sorpresa       |             16 |            4 |                    0 |               5 |              6 |            5 |             4 |            4 |             0 |              10 |                2 |           0 |
| Tristezza      |              2 |            0 |                    0 |               1 |              1 |            0 |             0 |            1 |             0 |               2 |                1 |           0 |

##### Run: `Case 6` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3158 |   0.5455 |     0.4000 |        11 |
| Anticipazione |      0.4167 |   0.5000 |     0.4545 |        10 |
| Disgusto      |      0.5714 |   0.8421 |     0.6809 |        19 |
| Fiducia       |      0.5714 |   0.6667 |     0.6154 |        42 |
| Gioia         |      0.5263 |   0.6897 |     0.5970 |        29 |
| Neutro        |      0.3200 |   0.4706 |     0.3810 |        17 |
| Paura         |      0.0667 |   1.0000 |     0.1250 |         1 |
| Rabbia        |      0.1765 |   0.5000 |     0.2609 |         6 |
| Sorpresa      |      0.3333 |   0.3750 |     0.3529 |        16 |
| Tristezza     |      0.1667 |   1.0000 |     0.2857 |         2 |
| micro avg     |      0.4077 |   0.6209 |     0.4922 |       153 |
| macro avg     |      0.3465 |   0.6589 |     0.4153 |       153 |
| weighted avg  |      0.4575 |   0.6209 |     0.5191 |       153 |
| samples avg   |      0.4291 |   0.6357 |     0.4863 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  105 |   13 |    5 |    6 |      0.3158 |   0.5455 |     0.4000 |        11 |
| Anticipazione |  112 |    7 |    5 |    5 |      0.4167 |   0.5000 |     0.4545 |        10 |
| Disgusto      |   98 |   12 |    3 |   16 |      0.5714 |   0.8421 |     0.6809 |        19 |
| Fiducia       |   66 |   21 |   14 |   28 |      0.5714 |   0.6667 |     0.6154 |        42 |
| Gioia         |   82 |   18 |    9 |   20 |      0.5263 |   0.6897 |     0.5970 |        29 |
| Neutro        |   95 |   17 |    9 |    8 |      0.3200 |   0.4706 |     0.3810 |        17 |
| Paura         |  114 |   14 |    0 |    1 |      0.0667 |   1.0000 |     0.1250 |         1 |
| Rabbia        |  109 |   14 |    3 |    3 |      0.1765 |   0.5000 |     0.2609 |         6 |
| Sorpresa      |  101 |   12 |   10 |    6 |      0.3333 |   0.3750 |     0.3529 |        16 |
| Tristezza     |  117 |   10 |    0 |    2 |      0.1667 |   1.0000 |     0.2857 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            6 |                    0 |               0 |              7 |            4 |             2 |            1 |             1 |               0 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    5 |               0 |              0 |            6 |             0 |            0 |             1 |               1 |                0 |           1 |
| Disgusto       |             19 |            0 |                    1 |              16 |              2 |            0 |             4 |            2 |            10 |               2 |                2 |           0 |
| Fiducia        |             42 |            5 |                    2 |               2 |             28 |           15 |             7 |            2 |             2 |               4 |                1 |           1 |
| Gioia          |             29 |            7 |                    0 |               4 |              5 |           20 |             4 |            3 |             2 |               7 |                2 |           2 |
| Neutro         |             17 |            3 |                    3 |               1 |              8 |            3 |             8 |            3 |             0 |               2 |                3 |           0 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               0 |                1 |           0 |
| Rabbia         |              6 |            0 |                    0 |               4 |              0 |            0 |             0 |            2 |             3 |               1 |                2 |           1 |
| Sorpresa       |             16 |            1 |                    1 |               3 |              4 |            5 |             3 |            3 |             1 |               6 |                3 |           0 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               1 |                2 |           0 |

---

## 2. Task 2: Irony & Sarcasm Detection (3-Class Single-Label, `UmBERTo` + `ViT-224`)

- **Text Encoder**: `Musixmatch/umberto-commoncrawl-cased-v1` (`UmBERTo`, 768-d `[CLS]`)
- **Vision Encoder**: `google/vit-base-patch16-224` (`ViT-Base-224`, frozen 768-d `[CLS]`)
- **Multimodal Head (`mm_test`, $N=129$)**: `SGDClassifier(loss='squared_hinge', penalty='l2', class_weight='balanced', random_state=42)`

### 2.1 Textual Irony Benchmark (`me_test`, $N=405$)

| Regime                        | Checkpoint                  | Prompt                               |   N Test |   Macro F1 |   Micro F1 (= Acc) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:------------------------------|:----------------------------|:-------------------------------------|---------:|-----------:|-------------------:|--------------:|----------:|------------:|--------------:|
| Step 1 Base (Raw)             | irony_textual_no_corr       | Raw                                  |      405 |     0.3970 |             0.7309 |        0.7268 |    0.8380 |      0.3529 |        0.0000 |
| Step 2 Raw (Ablation)         | irony_textual_step2_raw     | Raw                                  |      405 |     0.4485 |             0.7506 |        0.7440 |    0.8532 |      0.3673 |        0.1250 |
| Step 2 Oracle (Train: no rel) | irony_textual_corr          | Oracle (Train: no rel, Eval: no rel) |      405 |     0.5419 |             0.7531 |        0.7583 |    0.8466 |      0.4458 |        0.3333 |
| Step 2 Oracle (Train: +rel)   | irony_textual_corr_with_rel | Oracle (Train: +rel, Eval: no rel)   |      405 |     0.4962 |             0.7259 |        0.7412 |    0.8264 |      0.4516 |        0.2105 |

#### Textual Case: `Step 1 Base (Raw)` (`irony_textual_no_corr`)
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8302 |   0.8459 |     0.8380 |       318 |
| Ironia       |      0.3649 |   0.3418 |     0.3529 |        79 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         8 |
| macro avg    |      0.3984 |   0.3959 |     0.3970 |       405 |
| weighted avg |      0.7231 |   0.7309 |     0.7268 |       405 |
| accuracy     |      0.7309 |   0.7309 |     0.7309 |       405 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |         269 |            44 |               5 |          318 |
| Ironia       |          50 |            27 |               2 |           79 |
| Sarcasmo     |           5 |             3 |               0 |            8 |

#### Textual Case: `Step 2 Raw (Ablation)` (`irony_textual_step2_raw`)
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8389 |   0.8679 |     0.8532 |       318 |
| Ironia       |      0.3971 |   0.3418 |     0.3673 |        79 |
| Sarcasmo     |      0.1250 |   0.1250 |     0.1250 |         8 |
| macro avg    |      0.4537 |   0.4449 |     0.4485 |       405 |
| weighted avg |      0.7386 |   0.7506 |     0.7440 |       405 |
| accuracy     |      0.7506 |   0.7506 |     0.7506 |       405 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |         276 |            38 |               4 |          318 |
| Ironia       |          49 |            27 |               3 |           79 |
| Sarcasmo     |           4 |             3 |               1 |            8 |

#### Textual Case: `Step 2 Oracle (Train: no rel)` (`irony_textual_corr`)
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8604 |   0.8333 |     0.8466 |       318 |
| Ironia       |      0.4253 |   0.4684 |     0.4458 |        79 |
| Sarcasmo     |      0.3000 |   0.3750 |     0.3333 |         8 |
| macro avg    |      0.5286 |   0.5589 |     0.5419 |       405 |
| weighted avg |      0.7644 |   0.7531 |     0.7583 |       405 |
| accuracy     |      0.7531 |   0.7531 |     0.7531 |       405 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |         265 |            47 |               6 |          318 |
| Ironia       |          41 |            37 |               1 |           79 |
| Sarcasmo     |           2 |             3 |               3 |            8 |

#### Textual Case: `Step 2 Oracle (Train: +rel)` (`irony_textual_corr_with_rel`)
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8711 |   0.7862 |     0.8264 |       318 |
| Ironia       |      0.3925 |   0.5316 |     0.4516 |        79 |
| Sarcasmo     |      0.1818 |   0.2500 |     0.2105 |         8 |
| macro avg    |      0.4818 |   0.5226 |     0.4962 |       405 |
| weighted avg |      0.7641 |   0.7259 |     0.7412 |       405 |
| accuracy     |      0.7259 |   0.7259 |     0.7259 |       405 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |         250 |            62 |               6 |          318 |
| Ironia       |          34 |            42 |               3 |           79 |
| Sarcasmo     |           3 |             3 |               2 |            8 |

---

### 2.2 Multimodal Irony Benchmark (`mm_test`, $N=129$ — All 6 Cases $\times$ `raw` vs `std`)

#### Master Comparison Table (All Multimodal Irony Runs)
| Case   | Regime                                      | Train Relation   | Eval Relation   | Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:-------|:--------------------------------------------|:-----------------|:----------------|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Blind Text (SGD, raw)                 | raw       |     0.4250 |              0.7442 |        0.7442 |    0.8750 |      0.2000 |        0.2000 |
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5046 |              0.7907 |        0.7819 |    0.8815 |      0.4324 |        0.2000 |
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Blind Text (SGD, std)                 | std       |     0.4299 |              0.7209 |        0.7395 |    0.8515 |      0.3429 |        0.0952 |
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.5257 |              0.8217 |        0.8074 |    0.9065 |      0.4706 |        0.2000 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Blind Text (SGD, raw)                 | raw       |     0.4784 |              0.7597 |        0.7715 |    0.8824 |      0.3529 |        0.2000 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5083 |              0.7829 |        0.7791 |    0.8750 |      0.4500 |        0.2000 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Blind Text (SGD, std)                 | std       |     0.4452 |              0.7674 |        0.7606 |    0.8774 |      0.3333 |        0.1250 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.5347 |              0.7984 |        0.7951 |    0.8889 |      0.4651 |        0.2500 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Blind Text (SGD, raw)                 | raw       |     0.4632 |              0.7364 |        0.7548 |    0.8657 |      0.3333 |        0.1905 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.4321 |              0.7519 |        0.7551 |    0.8614 |      0.4348 |        0.0000 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Blind Text (SGD, std)                 | std       |     0.4868 |              0.7287 |        0.7465 |    0.8442 |      0.3810 |        0.2353 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.4855 |              0.7519 |        0.7652 |    0.8643 |      0.4255 |        0.1667 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Blind Text (SGD, raw)                 | raw       |     0.4690 |              0.7442 |        0.7494 |    0.8529 |      0.3721 |        0.1818 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.4194 |              0.7364 |        0.7422 |    0.8500 |      0.4082 |        0.0000 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Blind Text (SGD, std)                 | std       |     0.5346 |              0.7597 |        0.7696 |    0.8614 |      0.4091 |        0.3333 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.4730 |              0.7364 |        0.7463 |    0.8458 |      0.3913 |        0.1818 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Blind Text (SGD, raw)                 | raw       |     0.4284 |              0.7984 |        0.7781 |    0.8981 |      0.3871 |        0.0000 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.4438 |              0.7907 |        0.7735 |    0.8815 |      0.4500 |        0.0000 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Blind Text (SGD, std)                 | std       |     0.4604 |              0.7829 |        0.7802 |    0.8812 |      0.5000 |        0.0000 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.4357 |              0.7597 |        0.7576 |    0.8627 |      0.4444 |        0.0000 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Blind Text (SGD, raw)                 | raw       |     0.4382 |              0.7597 |        0.7606 |    0.8657 |      0.4490 |        0.0000 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.4463 |              0.7907 |        0.7717 |    0.8774 |      0.4615 |        0.0000 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Blind Text (SGD, std)                 | std       |     0.3979 |              0.7442 |        0.7519 |    0.8780 |      0.3158 |        0.0000 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.5063 |              0.7752 |        0.7809 |    0.8835 |      0.4000 |        0.2353 |

#### Semiotic Subsets Breakdown (`ADD`, `PAR`, `DIV` Accuracy / Micro F1 on `mm_test`, $N=129$)
| Case   | Regime                                      | Model                                 | Scaling   |   ADD (N=68) |   PAR (N=23) |   DIV (N=38) |   Macro F1 |   Micro F1 (= Acc.) |
|:-------|:--------------------------------------------|:--------------------------------------|:----------|-------------:|-------------:|-------------:|-----------:|--------------------:|
| Case 1 | Step 1 Base (No corr)                       | Blind Text (SGD, raw)                 | raw       |       0.8235 |       1.0000 |       0.4474 |     0.4250 |              0.7442 |
| Case 1 | Step 1 Base (No corr)                       | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.8382 |       0.9565 |       0.6053 |     0.5046 |              0.7907 |
| Case 1 | Step 1 Base (No corr)                       | Blind Text (SGD, std)                 | std       |       0.8088 |       0.9565 |       0.4211 |     0.4299 |              0.7209 |
| Case 1 | Step 1 Base (No corr)                       | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.9118 |       1.0000 |       0.5526 |     0.5257 |              0.8217 |
| Case 2 | Step 2 Raw (No corr)                        | Blind Text (SGD, raw)                 | raw       |       0.8235 |       0.9565 |       0.5263 |     0.4784 |              0.7597 |
| Case 2 | Step 2 Raw (No corr)                        | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.8971 |       0.8261 |       0.5526 |     0.5083 |              0.7829 |
| Case 2 | Step 2 Raw (No corr)                        | Blind Text (SGD, std)                 | std       |       0.8529 |       0.9565 |       0.5000 |     0.4452 |              0.7674 |
| Case 2 | Step 2 Raw (No corr)                        | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.8824 |       0.8696 |       0.6053 |     0.5347 |              0.7984 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (SGD, raw)                 | raw       |       0.8529 |       0.9130 |       0.4211 |     0.4632 |              0.7364 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.8382 |       0.9130 |       0.5000 |     0.4321 |              0.7519 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (SGD, std)                 | std       |       0.8235 |       0.9130 |       0.4474 |     0.4868 |              0.7287 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.8529 |       0.9130 |       0.4737 |     0.4855 |              0.7519 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Blind Text (SGD, raw)                 | raw       |       0.7941 |       0.9130 |       0.5526 |     0.4690 |              0.7442 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.7941 |       0.8696 |       0.5526 |     0.4194 |              0.7364 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Blind Text (SGD, std)                 | std       |       0.8088 |       0.9130 |       0.5789 |     0.5346 |              0.7597 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.8088 |       0.9130 |       0.5000 |     0.4730 |              0.7364 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Blind Text (SGD, raw)                 | raw       |       0.8676 |       1.0000 |       0.5526 |     0.4284 |              0.7984 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.8529 |       0.9130 |       0.6053 |     0.4438 |              0.7907 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Blind Text (SGD, std)                 | std       |       0.8676 |       0.9565 |       0.5263 |     0.4604 |              0.7829 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.8382 |       0.8696 |       0.5526 |     0.4357 |              0.7597 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Blind Text (SGD, raw)                 | raw       |       0.8382 |       0.9565 |       0.5000 |     0.4382 |              0.7597 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.8824 |       0.9130 |       0.5526 |     0.4463 |              0.7907 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Blind Text (SGD, std)                 | std       |       0.8382 |       0.9565 |       0.4474 |     0.3979 |              0.7442 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.8529 |       0.9565 |       0.5263 |     0.5063 |              0.7752 |

### 2.3 Multimodal Irony Detailed Reports by Case (`Case 1` – `Case 6`)

#### Case 1: Step 1 Base (No corr) (Checkpoint: `irony_textual_no_corr`, Eval Prompt: `None (Raw)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Blind Text (SGD, raw)                 | raw       |     0.4250 |              0.7442 |        0.7442 |    0.8750 |      0.2000 |        0.2000 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5046 |              0.7907 |        0.7819 |    0.8815 |      0.4324 |        0.2000 |
| Blind Text (SGD, std)                 | std       |     0.4299 |              0.7209 |        0.7395 |    0.8515 |      0.3429 |        0.0952 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.5257 |              0.8217 |        0.8074 |    0.9065 |      0.4706 |        0.2000 |

##### Run: `Case 1` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8750 |   0.8750 |     0.8750 |       104 |
| Ironia       |      0.2500 |   0.1667 |     0.2000 |        18 |
| Sarcasmo     |      0.1538 |   0.2857 |     0.2000 |         7 |
| macro avg    |      0.4263 |   0.4425 |     0.4250 |       129 |
| weighted avg |      0.7487 |   0.7442 |     0.7442 |       129 |
| accuracy     |      0.7442 |   0.7442 |     0.7442 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          91 |             7 |               6 |          104 |
| Ironia       |          10 |             3 |               5 |           18 |
| Sarcasmo     |           3 |             2 |               2 |            7 |

##### Run: `Case 1` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8692 |   0.8942 |     0.8815 |       104 |
| Ironia       |      0.4211 |   0.4444 |     0.4324 |        18 |
| Sarcasmo     |      0.3333 |   0.1429 |     0.2000 |         7 |
| macro avg    |      0.5412 |   0.4938 |     0.5046 |       129 |
| weighted avg |      0.7776 |   0.7907 |     0.7819 |       129 |
| accuracy     |      0.7907 |   0.7907 |     0.7907 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          93 |            10 |               1 |          104 |
| Ironia       |           9 |             8 |               1 |           18 |
| Sarcasmo     |           5 |             1 |               1 |            7 |

##### Run: `Case 1` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8776 |   0.8269 |     0.8515 |       104 |
| Ironia       |      0.3529 |   0.3333 |     0.3429 |        18 |
| Sarcasmo     |      0.0714 |   0.1429 |     0.0952 |         7 |
| macro avg    |      0.4340 |   0.4344 |     0.4299 |       129 |
| weighted avg |      0.7606 |   0.7209 |     0.7395 |       129 |
| accuracy     |      0.7209 |   0.7209 |     0.7209 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          86 |             8 |              10 |          104 |
| Ironia       |           9 |             6 |               3 |           18 |
| Sarcasmo     |           3 |             3 |               1 |            7 |

##### Run: `Case 1` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8818 |   0.9327 |     0.9065 |       104 |
| Ironia       |      0.5000 |   0.4444 |     0.4706 |        18 |
| Sarcasmo     |      0.3333 |   0.1429 |     0.2000 |         7 |
| macro avg    |      0.5717 |   0.5067 |     0.5257 |       129 |
| weighted avg |      0.7988 |   0.8217 |     0.8074 |       129 |
| accuracy     |      0.8217 |   0.8217 |     0.8217 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          97 |             6 |               1 |          104 |
| Ironia       |           9 |             8 |               1 |           18 |
| Sarcasmo     |           4 |             2 |               1 |            7 |

#### Case 2: Step 2 Raw (No corr) (Checkpoint: `irony_textual_step2_raw`, Eval Prompt: `None (Raw)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Blind Text (SGD, raw)                 | raw       |     0.4784 |              0.7597 |        0.7715 |    0.8824 |      0.3529 |        0.2000 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5083 |              0.7829 |        0.7791 |    0.8750 |      0.4500 |        0.2000 |
| Blind Text (SGD, std)                 | std       |     0.4452 |              0.7674 |        0.7606 |    0.8774 |      0.3333 |        0.1250 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.5347 |              0.7984 |        0.7951 |    0.8889 |      0.4651 |        0.2500 |

##### Run: `Case 2` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.9000 |   0.8654 |     0.8824 |       104 |
| Ironia       |      0.3750 |   0.3333 |     0.3529 |        18 |
| Sarcasmo     |      0.1538 |   0.2857 |     0.2000 |         7 |
| macro avg    |      0.4763 |   0.4948 |     0.4784 |       129 |
| weighted avg |      0.7863 |   0.7597 |     0.7715 |       129 |
| accuracy     |      0.7597 |   0.7597 |     0.7597 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          90 |             7 |               7 |          104 |
| Ironia       |           8 |             6 |               4 |           18 |
| Sarcasmo     |           2 |             3 |               2 |            7 |

##### Run: `Case 2` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8750 |   0.8750 |     0.8750 |       104 |
| Ironia       |      0.4091 |   0.5000 |     0.4500 |        18 |
| Sarcasmo     |      0.3333 |   0.1429 |     0.2000 |         7 |
| macro avg    |      0.5391 |   0.5060 |     0.5083 |       129 |
| weighted avg |      0.7806 |   0.7829 |     0.7791 |       129 |
| accuracy     |      0.7829 |   0.7829 |     0.7829 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          91 |            11 |               2 |          104 |
| Ironia       |           9 |             9 |               0 |           18 |
| Sarcasmo     |           4 |             2 |               1 |            7 |

##### Run: `Case 2` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8611 |   0.8942 |     0.8774 |       104 |
| Ironia       |      0.4167 |   0.2778 |     0.3333 |        18 |
| Sarcasmo     |      0.1111 |   0.1429 |     0.1250 |         7 |
| macro avg    |      0.4630 |   0.4383 |     0.4452 |       129 |
| weighted avg |      0.7584 |   0.7674 |     0.7606 |       129 |
| accuracy     |      0.7674 |   0.7674 |     0.7674 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          93 |             6 |               5 |          104 |
| Ironia       |          10 |             5 |               3 |           18 |
| Sarcasmo     |           5 |             1 |               1 |            7 |

##### Run: `Case 2` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8932 |   0.8846 |     0.8889 |       104 |
| Ironia       |      0.4000 |   0.5556 |     0.4651 |        18 |
| Sarcasmo     |      1.0000 |   0.1429 |     0.2500 |         7 |
| macro avg    |      0.7644 |   0.5277 |     0.5347 |       129 |
| weighted avg |      0.8302 |   0.7984 |     0.7951 |       129 |
| accuracy     |      0.7984 |   0.7984 |     0.7984 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          92 |            12 |               0 |          104 |
| Ironia       |           8 |            10 |               0 |           18 |
| Sarcasmo     |           3 |             3 |               1 |            7 |

#### Case 3: Step 2 Oracle (Train: no rel, Eval: no rel) (Checkpoint: `irony_textual_corr`, Eval Prompt: `Oracle (no rel)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Blind Text (SGD, raw)                 | raw       |     0.4632 |              0.7364 |        0.7548 |    0.8657 |      0.3333 |        0.1905 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.4321 |              0.7519 |        0.7551 |    0.8614 |      0.4348 |        0.0000 |
| Blind Text (SGD, std)                 | std       |     0.4868 |              0.7287 |        0.7465 |    0.8442 |      0.3810 |        0.2353 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.4855 |              0.7519 |        0.7652 |    0.8643 |      0.4255 |        0.1667 |

##### Run: `Case 3` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8969 |   0.8365 |     0.8657 |       104 |
| Ironia       |      0.3333 |   0.3333 |     0.3333 |        18 |
| Sarcasmo     |      0.1429 |   0.2857 |     0.1905 |         7 |
| macro avg    |      0.4577 |   0.4852 |     0.4632 |       129 |
| weighted avg |      0.7774 |   0.7364 |     0.7548 |       129 |
| accuracy     |      0.7364 |   0.7364 |     0.7364 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          87 |            10 |               7 |          104 |
| Ironia       |           7 |             6 |               5 |           18 |
| Sarcasmo     |           3 |             2 |               2 |            7 |

##### Run: `Case 3` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8878 |   0.8365 |     0.8614 |       104 |
| Ironia       |      0.3571 |   0.5556 |     0.4348 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.4150 |   0.4640 |     0.4321 |       129 |
| weighted avg |      0.7655 |   0.7519 |     0.7551 |       129 |
| accuracy     |      0.7519 |   0.7519 |     0.7519 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          87 |            15 |               2 |          104 |
| Ironia       |           7 |            10 |               1 |           18 |
| Sarcasmo     |           4 |             3 |               0 |            7 |

##### Run: `Case 3` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8842 |   0.8077 |     0.8442 |       104 |
| Ironia       |      0.3333 |   0.4444 |     0.3810 |        18 |
| Sarcasmo     |      0.2000 |   0.2857 |     0.2353 |         7 |
| macro avg    |      0.4725 |   0.5126 |     0.4868 |       129 |
| weighted avg |      0.7702 |   0.7287 |     0.7465 |       129 |
| accuracy     |      0.7287 |   0.7287 |     0.7287 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          84 |            14 |               6 |          104 |
| Ironia       |           8 |             8 |               2 |           18 |
| Sarcasmo     |           3 |             2 |               2 |            7 |

##### Run: `Case 3` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.9053 |   0.8269 |     0.8643 |       104 |
| Ironia       |      0.3448 |   0.5556 |     0.4255 |        18 |
| Sarcasmo     |      0.2000 |   0.1429 |     0.1667 |         7 |
| macro avg    |      0.4834 |   0.5084 |     0.4855 |       129 |
| weighted avg |      0.7888 |   0.7519 |     0.7652 |       129 |
| accuracy     |      0.7519 |   0.7519 |     0.7519 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          86 |            16 |               2 |          104 |
| Ironia       |           6 |            10 |               2 |           18 |
| Sarcasmo     |           3 |             3 |               1 |            7 |

#### Case 4: Step 2 Oracle (Train: no rel, Eval: +rel) (Checkpoint: `irony_textual_corr`, Eval Prompt: `Oracle (+ rel)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Blind Text (SGD, raw)                 | raw       |     0.4690 |              0.7442 |        0.7494 |    0.8529 |      0.3721 |        0.1818 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.4194 |              0.7364 |        0.7422 |    0.8500 |      0.4082 |        0.0000 |
| Blind Text (SGD, std)                 | std       |     0.5346 |              0.7597 |        0.7696 |    0.8614 |      0.4091 |        0.3333 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.4730 |              0.7364 |        0.7463 |    0.8458 |      0.3913 |        0.1818 |

##### Run: `Case 4` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8700 |   0.8365 |     0.8529 |       104 |
| Ironia       |      0.3200 |   0.4444 |     0.3721 |        18 |
| Sarcasmo     |      0.2500 |   0.1429 |     0.1818 |         7 |
| macro avg    |      0.4800 |   0.4746 |     0.4690 |       129 |
| weighted avg |      0.7596 |   0.7442 |     0.7494 |       129 |
| accuracy     |      0.7442 |   0.7442 |     0.7442 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          87 |            15 |               2 |          104 |
| Ironia       |           9 |             8 |               1 |           18 |
| Sarcasmo     |           4 |             2 |               1 |            7 |

##### Run: `Case 4` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8854 |   0.8173 |     0.8500 |       104 |
| Ironia       |      0.3226 |   0.5556 |     0.4082 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.4027 |   0.4576 |     0.4194 |       129 |
| weighted avg |      0.7588 |   0.7364 |     0.7422 |       129 |
| accuracy     |      0.7364 |   0.7364 |     0.7364 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          85 |            18 |               1 |          104 |
| Ironia       |           7 |            10 |               1 |           18 |
| Sarcasmo     |           4 |             3 |               0 |            7 |

##### Run: `Case 4` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8878 |   0.8365 |     0.8614 |       104 |
| Ironia       |      0.3462 |   0.5000 |     0.4091 |        18 |
| Sarcasmo     |      0.4000 |   0.2857 |     0.3333 |         7 |
| macro avg    |      0.5446 |   0.5408 |     0.5346 |       129 |
| weighted avg |      0.7857 |   0.7597 |     0.7696 |       129 |
| accuracy     |      0.7597 |   0.7597 |     0.7597 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          87 |            15 |               2 |          104 |
| Ironia       |           8 |             9 |               1 |           18 |
| Sarcasmo     |           3 |             2 |               2 |            7 |

##### Run: `Case 4` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8763 |   0.8173 |     0.8458 |       104 |
| Ironia       |      0.3214 |   0.5000 |     0.3913 |        18 |
| Sarcasmo     |      0.2500 |   0.1429 |     0.1818 |         7 |
| macro avg    |      0.4826 |   0.4867 |     0.4730 |       129 |
| weighted avg |      0.7649 |   0.7364 |     0.7463 |       129 |
| accuracy     |      0.7364 |   0.7364 |     0.7364 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          85 |            17 |               2 |          104 |
| Ironia       |           8 |             9 |               1 |           18 |
| Sarcasmo     |           4 |             2 |               1 |            7 |

#### Case 5: Step 2 Oracle (Train: +rel, Eval: no rel) (Checkpoint: `irony_textual_corr_with_rel`, Eval Prompt: `Oracle (no rel)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Blind Text (SGD, raw)                 | raw       |     0.4284 |              0.7984 |        0.7781 |    0.8981 |      0.3871 |        0.0000 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.4438 |              0.7907 |        0.7735 |    0.8815 |      0.4500 |        0.0000 |
| Blind Text (SGD, std)                 | std       |     0.4604 |              0.7829 |        0.7802 |    0.8812 |      0.5000 |        0.0000 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.4357 |              0.7597 |        0.7576 |    0.8627 |      0.4444 |        0.0000 |

##### Run: `Case 5` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8661 |   0.9327 |     0.8981 |       104 |
| Ironia       |      0.4615 |   0.3333 |     0.3871 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.4425 |   0.4220 |     0.4284 |       129 |
| weighted avg |      0.7626 |   0.7984 |     0.7781 |       129 |
| accuracy     |      0.7984 |   0.7984 |     0.7984 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          97 |             3 |               4 |          104 |
| Ironia       |          12 |             6 |               0 |           18 |
| Sarcasmo     |           3 |             4 |               0 |            7 |

##### Run: `Case 5` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8692 |   0.8942 |     0.8815 |       104 |
| Ironia       |      0.4091 |   0.5000 |     0.4500 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.4261 |   0.4647 |     0.4438 |       129 |
| weighted avg |      0.7578 |   0.7907 |     0.7735 |       129 |
| accuracy     |      0.7907 |   0.7907 |     0.7907 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          93 |            11 |               0 |          104 |
| Ironia       |           9 |             9 |               0 |           18 |
| Sarcasmo     |           5 |             2 |               0 |            7 |

##### Run: `Case 5` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.9082 |   0.8558 |     0.8812 |       104 |
| Ironia       |      0.4000 |   0.6667 |     0.5000 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.4361 |   0.5075 |     0.4604 |       129 |
| weighted avg |      0.7880 |   0.7829 |     0.7802 |       129 |
| accuracy     |      0.7829 |   0.7829 |     0.7829 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          89 |            14 |               1 |          104 |
| Ironia       |           6 |            12 |               0 |           18 |
| Sarcasmo     |           3 |             4 |               0 |            7 |

##### Run: `Case 5` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8800 |   0.8462 |     0.8627 |       104 |
| Ironia       |      0.3704 |   0.5556 |     0.4444 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.4168 |   0.4672 |     0.4357 |       129 |
| weighted avg |      0.7611 |   0.7597 |     0.7576 |       129 |
| accuracy     |      0.7597 |   0.7597 |     0.7597 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          88 |            14 |               2 |          104 |
| Ironia       |           8 |            10 |               0 |           18 |
| Sarcasmo     |           4 |             3 |               0 |            7 |

#### Case 6: Step 2 Oracle (Train: +rel, Eval: +rel) (Checkpoint: `irony_textual_corr_with_rel`, Eval Prompt: `Oracle (+ rel)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Blind Text (SGD, raw)                 | raw       |     0.4382 |              0.7597 |        0.7606 |    0.8657 |      0.4490 |        0.0000 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.4463 |              0.7907 |        0.7717 |    0.8774 |      0.4615 |        0.0000 |
| Blind Text (SGD, std)                 | std       |     0.3979 |              0.7442 |        0.7519 |    0.8780 |      0.3158 |        0.0000 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.5063 |              0.7752 |        0.7809 |    0.8835 |      0.4000 |        0.2353 |

##### Run: `Case 6` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8969 |   0.8365 |     0.8657 |       104 |
| Ironia       |      0.3548 |   0.6111 |     0.4490 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.4172 |   0.4825 |     0.4382 |       129 |
| weighted avg |      0.7726 |   0.7597 |     0.7606 |       129 |
| accuracy     |      0.7597 |   0.7597 |     0.7597 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          87 |            16 |               1 |          104 |
| Ironia       |           7 |            11 |               0 |           18 |
| Sarcasmo     |           3 |             4 |               0 |            7 |

##### Run: `Case 6` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8611 |   0.8942 |     0.8774 |       104 |
| Ironia       |      0.4286 |   0.5000 |     0.4615 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.4299 |   0.4647 |     0.4463 |       129 |
| weighted avg |      0.7540 |   0.7907 |     0.7717 |       129 |
| accuracy     |      0.7907 |   0.7907 |     0.7907 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          93 |            11 |               0 |          104 |
| Ironia       |           9 |             9 |               0 |           18 |
| Sarcasmo     |           6 |             1 |               0 |            7 |

##### Run: `Case 6` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8911 |   0.8654 |     0.8780 |       104 |
| Ironia       |      0.3000 |   0.3333 |     0.3158 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.3970 |   0.3996 |     0.3979 |       129 |
| weighted avg |      0.7603 |   0.7442 |     0.7519 |       129 |
| accuracy     |      0.7442 |   0.7442 |     0.7442 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          90 |            10 |               4 |          104 |
| Ironia       |           8 |             6 |               4 |           18 |
| Sarcasmo     |           3 |             4 |               0 |            7 |

##### Run: `Case 6` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8922 |   0.8750 |     0.8835 |       104 |
| Ironia       |      0.4118 |   0.3889 |     0.4000 |        18 |
| Sarcasmo     |      0.2000 |   0.2857 |     0.2353 |         7 |
| macro avg    |      0.5013 |   0.5165 |     0.5063 |       129 |
| weighted avg |      0.7876 |   0.7752 |     0.7809 |       129 |
| accuracy     |      0.7752 |   0.7752 |     0.7752 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          91 |             7 |               6 |          104 |
| Ironia       |           9 |             7 |               2 |           18 |
| Sarcasmo     |           2 |             3 |               2 |            7 |

---

## 3. Task 3: Semiotic Relation Classification (3-Class Single-Label, `UmBERTo` + `ViT-224`)

- **Text Encoder**: `Musixmatch/umberto-commoncrawl-cased-v1` (`UmBERTo`, 768-d `[CLS]`)
- **Vision Encoder**: `google/vit-base-patch16-224` (`ViT-Base-224`, frozen 768-d `[CLS]`)
- **Multimodal Head (`mm_test`, $N=129$)**: `SGDClassifier(loss='squared_hinge', penalty='l2', class_weight='balanced', random_state=42)`

### 3.1 Textual Encoder Training Benchmark (`mm_test`, $N=129$)

| Regime         | Checkpoint               | Prompt   |   N Test |   Macro F1 |   Micro F1 (= Acc) |   Weighted F1 |   Additiva F1 |   Divergente F1 |   Parallela F1 |
|:---------------|:-------------------------|:---------|---------:|-----------:|-------------------:|--------------:|--------------:|----------------:|---------------:|
| Stage A Raw    | relation_textual_no_corr | Raw      |      129 |     0.5680 |             0.6667 |        0.6390 |        0.7453 |          0.6462 |         0.3125 |
| Stage B Oracle | relation_textual_corr    | Oracle   |      129 |     0.6541 |             0.6977 |        0.6844 |        0.7547 |          0.6129 |         0.5946 |

#### Textual Encoder Stage: `Stage A Raw` (`relation_textual_no_corr`)
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6452 |   0.8824 |     0.7453 |        68 |
| Divergente   |      0.7778 |   0.5526 |     0.6462 |        38 |
| Parallela    |      0.5556 |   0.2174 |     0.3125 |        23 |
| macro avg    |      0.6595 |   0.5508 |     0.5680 |       129 |
| weighted avg |      0.6683 |   0.6667 |     0.6390 |       129 |
| accuracy     |      0.6667 |   0.6667 |     0.6667 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              60 |                 6 |                2 |           68 |
| Divergente   |              15 |                21 |                2 |           38 |
| Parallela    |              18 |                 0 |                5 |           23 |

#### Textual Encoder Stage: `Stage B Oracle` (`relation_textual_corr`)
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6593 |   0.8824 |     0.7547 |        68 |
| Divergente   |      0.7917 |   0.5000 |     0.6129 |        38 |
| Parallela    |      0.7857 |   0.4783 |     0.5946 |        23 |
| macro avg    |      0.7456 |   0.6202 |     0.6541 |       129 |
| weighted avg |      0.7209 |   0.6977 |     0.6844 |       129 |
| accuracy     |      0.6977 |   0.6977 |     0.6977 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              60 |                 5 |                3 |           68 |
| Divergente   |              19 |                19 |                0 |           38 |
| Parallela    |              12 |                 0 |               11 |           23 |

---

### 3.2 Multimodal Semiotic Relation Benchmark (`mm_test`, $N=129$ — `Stage A Raw` vs `Stage B Oracle` $\times$ `raw` vs `std`)

#### Master Comparison Table (All Multimodal Relation Runs)
| Case   | Regime                     | Prompting   | Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   Additiva F1 |   Divergente F1 |   Parallela F1 |
|:-------|:---------------------------|:------------|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|--------------:|----------------:|---------------:|
| Case 1 | Stage A Raw (No corr)      | None (Raw)  | Blind Text (SGD, raw)                 | raw       |     0.5958 |              0.6357 |        0.6313 |        0.7183 |          0.5333 |         0.5357 |
| Case 1 | Stage A Raw (No corr)      | None (Raw)  | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5595 |              0.5736 |        0.5812 |        0.6230 |          0.5556 |         0.5000 |
| Case 1 | Stage A Raw (No corr)      | None (Raw)  | Blind Text (SGD, std)                 | std       |     0.5699 |              0.6124 |        0.5941 |        0.6887 |          0.4211 |         0.6000 |
| Case 1 | Stage A Raw (No corr)      | None (Raw)  | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.5612 |              0.6279 |        0.6175 |        0.6974 |          0.6364 |         0.3500 |
| Case 2 | Stage B Oracle (With corr) | Oracle      | Blind Text (SGD, raw)                 | raw       |     0.5435 |              0.5349 |        0.5385 |        0.5839 |          0.3797 |         0.6667 |
| Case 2 | Stage B Oracle (With corr) | Oracle      | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.6047 |              0.6512 |        0.6407 |        0.7190 |          0.5714 |         0.5238 |
| Case 2 | Stage B Oracle (With corr) | Oracle      | Blind Text (SGD, std)                 | std       |     0.6087 |              0.6512 |        0.6401 |        0.7097 |          0.5758 |         0.5405 |
| Case 2 | Stage B Oracle (With corr) | Oracle      | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.5341 |              0.6124 |        0.5931 |        0.6923 |          0.5672 |         0.3429 |

#### Target Classes Per-Class F1 Breakdown (`ADD`, `PAR`, `DIV` on `mm_test`, $N=129$)
| Case   | Regime                     | Model                                 | Scaling   |   ADD (n=68) |   PAR (n=23) |   DIV (n=38) |   Macro F1 |   Micro F1 (= Acc.) |
|:-------|:---------------------------|:--------------------------------------|:----------|-------------:|-------------:|-------------:|-----------:|--------------------:|
| Case 1 | Stage A Raw (No corr)      | Blind Text (SGD, raw)                 | raw       |       0.7183 |       0.5357 |       0.5333 |     0.5958 |              0.6357 |
| Case 1 | Stage A Raw (No corr)      | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.6230 |       0.5000 |       0.5556 |     0.5595 |              0.5736 |
| Case 1 | Stage A Raw (No corr)      | Blind Text (SGD, std)                 | std       |       0.6887 |       0.6000 |       0.4211 |     0.5699 |              0.6124 |
| Case 1 | Stage A Raw (No corr)      | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.6974 |       0.3500 |       0.6364 |     0.5612 |              0.6279 |
| Case 2 | Stage B Oracle (With corr) | Blind Text (SGD, raw)                 | raw       |       0.5839 |       0.6667 |       0.3797 |     0.5435 |              0.5349 |
| Case 2 | Stage B Oracle (With corr) | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.7190 |       0.5238 |       0.5714 |     0.6047 |              0.6512 |
| Case 2 | Stage B Oracle (With corr) | Blind Text (SGD, std)                 | std       |       0.7097 |       0.5405 |       0.5758 |     0.6087 |              0.6512 |
| Case 2 | Stage B Oracle (With corr) | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.6923 |       0.3429 |       0.5672 |     0.5341 |              0.6124 |

### 3.3 Multimodal Semiotic Relation Detailed Reports by Case (`Case 1` & `Case 2`)

#### Case 1: Stage A Raw (No corr) (Checkpoint: `relation_textual_no_corr`, Eval Prompt: `None (Raw)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   Additiva F1 |   Divergente F1 |   Parallela F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|--------------:|----------------:|---------------:|
| Blind Text (SGD, raw)                 | raw       |     0.5958 |              0.6357 |        0.6313 |        0.7183 |          0.5333 |         0.5357 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5595 |              0.5736 |        0.5812 |        0.6230 |          0.5556 |         0.5000 |
| Blind Text (SGD, std)                 | std       |     0.5699 |              0.6124 |        0.5941 |        0.6887 |          0.4211 |         0.6000 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.5612 |              0.6279 |        0.6175 |        0.6974 |          0.6364 |         0.3500 |

##### Run: `Case 1` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6892 |   0.7500 |     0.7183 |        68 |
| Divergente   |      0.7273 |   0.4211 |     0.5333 |        38 |
| Parallela    |      0.4545 |   0.6522 |     0.5357 |        23 |
| macro avg    |      0.6237 |   0.6077 |     0.5958 |       129 |
| weighted avg |      0.6586 |   0.6357 |     0.6313 |       129 |
| accuracy     |      0.6357 |   0.6357 |     0.6357 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              51 |                 6 |               11 |           68 |
| Divergente   |              15 |                16 |                7 |           38 |
| Parallela    |               8 |                 0 |               15 |           23 |

##### Run: `Case 1` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.7037 |   0.5588 |     0.6230 |        68 |
| Divergente   |      0.5882 |   0.5263 |     0.5556 |        38 |
| Parallela    |      0.3902 |   0.6957 |     0.5000 |        23 |
| macro avg    |      0.5607 |   0.5936 |     0.5595 |       129 |
| weighted avg |      0.6138 |   0.5736 |     0.5812 |       129 |
| accuracy     |      0.5736 |   0.5736 |     0.5736 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              38 |                11 |               19 |           68 |
| Divergente   |              12 |                20 |                6 |           38 |
| Parallela    |               4 |                 3 |               16 |           23 |

##### Run: `Case 1` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6265 |   0.7647 |     0.6887 |        68 |
| Divergente   |      0.6316 |   0.3158 |     0.4211 |        38 |
| Parallela    |      0.5556 |   0.6522 |     0.6000 |        23 |
| macro avg    |      0.6045 |   0.5776 |     0.5699 |       129 |
| weighted avg |      0.6154 |   0.6124 |     0.5941 |       129 |
| accuracy     |      0.6124 |   0.6124 |     0.6124 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              52 |                 6 |               10 |           68 |
| Divergente   |              24 |                12 |                2 |           38 |
| Parallela    |               7 |                 1 |               15 |           23 |

##### Run: `Case 1` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6310 |   0.7794 |     0.6974 |        68 |
| Divergente   |      0.7500 |   0.5526 |     0.6364 |        38 |
| Parallela    |      0.4118 |   0.3043 |     0.3500 |        23 |
| macro avg    |      0.5976 |   0.5455 |     0.5612 |       129 |
| weighted avg |      0.6269 |   0.6279 |     0.6175 |       129 |
| accuracy     |      0.6279 |   0.6279 |     0.6279 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              53 |                 6 |                9 |           68 |
| Divergente   |              16 |                21 |                1 |           38 |
| Parallela    |              15 |                 1 |                7 |           23 |

#### Case 2: Stage B Oracle (With corr) (Checkpoint: `relation_textual_corr`, Eval Prompt: `Oracle`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   Additiva F1 |   Divergente F1 |   Parallela F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|--------------:|----------------:|---------------:|
| Blind Text (SGD, raw)                 | raw       |     0.5435 |              0.5349 |        0.5385 |        0.5839 |          0.3797 |         0.6667 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.6047 |              0.6512 |        0.6407 |        0.7190 |          0.5714 |         0.5238 |
| Blind Text (SGD, std)                 | std       |     0.6087 |              0.6512 |        0.6401 |        0.7097 |          0.5758 |         0.5405 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.5341 |              0.6124 |        0.5931 |        0.6923 |          0.5672 |         0.3429 |

##### Run: `Case 2` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.5797 |   0.5882 |     0.5839 |        68 |
| Divergente   |      0.3659 |   0.3947 |     0.3797 |        38 |
| Parallela    |      0.7368 |   0.6087 |     0.6667 |        23 |
| macro avg    |      0.5608 |   0.5306 |     0.5435 |       129 |
| weighted avg |      0.5447 |   0.5349 |     0.5385 |       129 |
| accuracy     |      0.5349 |   0.5349 |     0.5349 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              40 |                23 |                5 |           68 |
| Divergente   |              23 |                15 |                0 |           38 |
| Parallela    |               6 |                 3 |               14 |           23 |

##### Run: `Case 2` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6471 |   0.8088 |     0.7190 |        68 |
| Divergente   |      0.7200 |   0.4737 |     0.5714 |        38 |
| Parallela    |      0.5789 |   0.4783 |     0.5238 |        23 |
| macro avg    |      0.6487 |   0.5869 |     0.6047 |       129 |
| weighted avg |      0.6564 |   0.6512 |     0.6407 |       129 |
| accuracy     |      0.6512 |   0.6512 |     0.6512 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              55 |                 6 |                7 |           68 |
| Divergente   |              19 |                18 |                1 |           38 |
| Parallela    |              11 |                 1 |               11 |           23 |

##### Run: `Case 2` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6322 |   0.8088 |     0.7097 |        68 |
| Divergente   |      0.6786 |   0.5000 |     0.5758 |        38 |
| Parallela    |      0.7143 |   0.4348 |     0.5405 |        23 |
| macro avg    |      0.6750 |   0.5812 |     0.6087 |       129 |
| weighted avg |      0.6605 |   0.6512 |     0.6401 |       129 |
| accuracy     |      0.6512 |   0.6512 |     0.6512 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              55 |                 9 |                4 |           68 |
| Divergente   |              19 |                19 |                0 |           38 |
| Parallela    |              13 |                 0 |               10 |           23 |

##### Run: `Case 2` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6136 |   0.7941 |     0.6923 |        68 |
| Divergente   |      0.6552 |   0.5000 |     0.5672 |        38 |
| Parallela    |      0.5000 |   0.2609 |     0.3429 |        23 |
| macro avg    |      0.5896 |   0.5183 |     0.5341 |       129 |
| weighted avg |      0.6056 |   0.6124 |     0.5931 |       129 |
| accuracy     |      0.6124 |   0.6124 |     0.6124 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              54 |                 9 |                5 |           68 |
| Divergente   |              18 |                19 |                1 |           38 |
| Parallela    |              16 |                 1 |                6 |           23 |
