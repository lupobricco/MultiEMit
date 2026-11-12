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
| Step 1 Base (Raw)             | emotion_textual_no_corr       | Raw                                  |      405 |     0.4987 |     0.5221 |        0.5182 |        0.3160 |
| Step 2 Raw (Ablation)         | emotion_textual_step2_raw     | Raw                                  |      405 |     0.5047 |     0.5229 |        0.5221 |        0.3235 |
| Step 2 Oracle (Train: no rel) | emotion_textual_corr          | Oracle (Train: no rel, Eval: no rel) |      405 |     0.5205 |     0.5464 |        0.5420 |        0.3432 |
| Step 2 Oracle (Train: +rel)   | emotion_textual_corr_with_rel | Oracle (Train: +rel, Eval: no rel)   |      405 |     0.5284 |     0.5435 |        0.5434 |        0.3333 |

#### Per-Emotion F1 Breakdown (`me_test`)
| Regime                        |   Amore F1 |   Anticipazione F1 |   Disgusto F1 |   Fiducia F1 |   Gioia F1 |   Neutro F1 |   Paura F1 |   Rabbia F1 |   Sorpresa F1 |   Tristezza F1 |
|:------------------------------|-----------:|-------------------:|--------------:|-------------:|-----------:|------------:|-----------:|------------:|--------------:|---------------:|
| Step 1 Base (Raw)             |     0.6042 |             0.5000 |        0.5714 |       0.5783 |     0.5842 |      0.4490 |     0.3529 |      0.4074 |        0.3905 |         0.5495 |
| Step 2 Raw (Ablation)         |     0.6364 |             0.5106 |        0.5191 |       0.5811 |     0.5500 |      0.4646 |     0.3750 |      0.3902 |        0.4550 |         0.5652 |
| Step 2 Oracle (Train: no rel) |     0.6186 |             0.4545 |        0.5556 |       0.6042 |     0.6327 |      0.4536 |     0.3750 |      0.5000 |        0.4339 |         0.5773 |
| Step 2 Oracle (Train: +rel)   |     0.6275 |             0.4643 |        0.5912 |       0.6272 |     0.5820 |      0.4381 |     0.5556 |      0.3704 |        0.4656 |         0.5625 |

#### Textual Case: `Step 1 Base (Raw)` (`emotion_textual_no_corr`)
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.6170 |   0.5918 |     0.6042 |        49 |
| Anticipazione |      0.5217 |   0.4800 |     0.5000 |        25 |
| Disgusto      |      0.5753 |   0.5676 |     0.5714 |        74 |
| Fiducia       |      0.5783 |   0.5783 |     0.5783 |        83 |
| Gioia         |      0.5268 |   0.6556 |     0.5842 |        90 |
| Neutro        |      0.4783 |   0.4231 |     0.4490 |        52 |
| Paura         |      0.6000 |   0.2500 |     0.3529 |        12 |
| Rabbia        |      0.4074 |   0.4074 |     0.4074 |        27 |
| Sorpresa      |      0.4125 |   0.3708 |     0.3905 |        89 |
| Tristezza     |      0.5102 |   0.5952 |     0.5495 |        42 |
| micro avg     |      0.5211 |   0.5230 |     0.5221 |       543 |
| macro avg     |      0.5228 |   0.4920 |     0.4987 |       543 |
| weighted avg  |      0.5202 |   0.5230 |     0.5182 |       543 |
| samples avg   |      0.5179 |   0.5076 |     0.4929 |       543 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  338 |   18 |   20 |   29 |      0.6170 |   0.5918 |     0.6042 |        49 |
| Anticipazione |  369 |   11 |   13 |   12 |      0.5217 |   0.4800 |     0.5000 |        25 |
| Disgusto      |  300 |   31 |   32 |   42 |      0.5753 |   0.5676 |     0.5714 |        74 |
| Fiducia       |  287 |   35 |   35 |   48 |      0.5783 |   0.5783 |     0.5783 |        83 |
| Gioia         |  262 |   53 |   31 |   59 |      0.5268 |   0.6556 |     0.5842 |        90 |
| Neutro        |  329 |   24 |   30 |   22 |      0.4783 |   0.4231 |     0.4490 |        52 |
| Paura         |  391 |    2 |    9 |    3 |      0.6000 |   0.2500 |     0.3529 |        12 |
| Rabbia        |  362 |   16 |   16 |   11 |      0.4074 |   0.4074 |     0.4074 |        27 |
| Sorpresa      |  269 |   47 |   56 |   33 |      0.4125 |   0.3708 |     0.3905 |        89 |
| Tristezza     |  339 |   24 |   17 |   25 |      0.5102 |   0.5952 |     0.5495 |        42 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             49 |           29 |                    1 |               2 |             19 |           17 |             1 |            0 |             1 |               0 |                1 |           0 |
| Anticipazione  |             25 |            0 |                   12 |               4 |              4 |            3 |             0 |            1 |             1 |               3 |                2 |           0 |
| Disgusto       |             74 |            2 |                    3 |              42 |              5 |            5 |             6 |            1 |            19 |              18 |               23 |           0 |
| Fiducia        |             83 |           18 |                    6 |               6 |             48 |           18 |             5 |            2 |             2 |               6 |                9 |           0 |
| Gioia          |             90 |           11 |                    2 |               2 |             17 |           59 |             4 |            0 |             0 |              21 |                1 |           0 |
| Neutro         |             52 |            1 |                    3 |               4 |              7 |           10 |            22 |            0 |             1 |               7 |                0 |           0 |
| Paura          |             12 |            0 |                    1 |               1 |              0 |            2 |             2 |            3 |             1 |               3 |                5 |           0 |
| Rabbia         |             27 |            0 |                    1 |              21 |              0 |            2 |             0 |            0 |            11 |               5 |                6 |           0 |
| Sorpresa       |             89 |            3 |                    1 |              26 |              6 |           24 |             9 |            4 |            11 |              33 |               24 |           0 |
| Tristezza      |             42 |            0 |                    3 |              16 |              5 |            2 |             1 |            2 |             8 |              13 |               25 |           0 |

#### Textual Case: `Step 2 Raw (Ablation)` (`emotion_textual_step2_raw`)
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.7179 |   0.5714 |     0.6364 |        49 |
| Anticipazione |      0.5455 |   0.4800 |     0.5106 |        25 |
| Disgusto      |      0.5965 |   0.4595 |     0.5191 |        74 |
| Fiducia       |      0.6615 |   0.5181 |     0.5811 |        83 |
| Gioia         |      0.4400 |   0.7333 |     0.5500 |        90 |
| Neutro        |      0.4894 |   0.4423 |     0.4646 |        52 |
| Paura         |      0.7500 |   0.2500 |     0.3750 |        12 |
| Rabbia        |      0.5714 |   0.2963 |     0.3902 |        27 |
| Sorpresa      |      0.3934 |   0.5393 |     0.4550 |        89 |
| Tristezza     |      0.5200 |   0.6190 |     0.5652 |        42 |
| micro avg     |      0.5105 |   0.5359 |     0.5229 |       543 |
| macro avg     |      0.5686 |   0.4909 |     0.5047 |       543 |
| weighted avg  |      0.5418 |   0.5359 |     0.5221 |       543 |
| samples avg   |      0.5272 |   0.5156 |     0.5020 |       543 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  345 |   11 |   21 |   28 |      0.7179 |   0.5714 |     0.6364 |        49 |
| Anticipazione |  370 |   10 |   13 |   12 |      0.5455 |   0.4800 |     0.5106 |        25 |
| Disgusto      |  308 |   23 |   40 |   34 |      0.5965 |   0.4595 |     0.5191 |        74 |
| Fiducia       |  300 |   22 |   40 |   43 |      0.6615 |   0.5181 |     0.5811 |        83 |
| Gioia         |  231 |   84 |   24 |   66 |      0.4400 |   0.7333 |     0.5500 |        90 |
| Neutro        |  329 |   24 |   29 |   23 |      0.4894 |   0.4423 |     0.4646 |        52 |
| Paura         |  392 |    1 |    9 |    3 |      0.7500 |   0.2500 |     0.3750 |        12 |
| Rabbia        |  372 |    6 |   19 |    8 |      0.5714 |   0.2963 |     0.3902 |        27 |
| Sorpresa      |  242 |   74 |   41 |   48 |      0.3934 |   0.5393 |     0.4550 |        89 |
| Tristezza     |  339 |   24 |   16 |   26 |      0.5200 |   0.6190 |     0.5652 |        42 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             49 |           28 |                    0 |               2 |             15 |           15 |             0 |            0 |             1 |               4 |                1 |           0 |
| Anticipazione  |             25 |            0 |                   12 |               2 |              3 |            5 |             0 |            1 |             1 |               5 |                3 |           0 |
| Disgusto       |             74 |            0 |                    3 |              34 |              6 |            9 |             6 |            1 |            10 |              32 |               25 |           0 |
| Fiducia        |             83 |           14 |                    3 |               4 |             43 |           26 |             5 |            1 |             1 |              13 |                7 |           0 |
| Gioia          |             90 |            9 |                    2 |               1 |             11 |           66 |             4 |            0 |             0 |              25 |                1 |           0 |
| Neutro         |             52 |            1 |                    3 |               4 |              3 |           16 |            23 |            0 |             0 |              13 |                0 |           0 |
| Paura          |             12 |            0 |                    1 |               1 |              0 |            3 |             2 |            3 |             0 |               5 |                4 |           0 |
| Rabbia         |             27 |            0 |                    1 |              18 |              0 |            2 |             0 |            0 |             8 |              10 |                8 |           0 |
| Sorpresa       |             89 |            3 |                    1 |              20 |              6 |           32 |            12 |            3 |             8 |              48 |               23 |           0 |
| Tristezza      |             42 |            0 |                    1 |              14 |              6 |            3 |             2 |            0 |             4 |              19 |               26 |           0 |

#### Textual Case: `Step 2 Oracle (Train: no rel)` (`emotion_textual_corr`)
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.6250 |   0.6122 |     0.6186 |        49 |
| Anticipazione |      0.5263 |   0.4000 |     0.4545 |        25 |
| Disgusto      |      0.5714 |   0.5405 |     0.5556 |        74 |
| Fiducia       |      0.5321 |   0.6988 |     0.6042 |        83 |
| Gioia         |      0.5849 |   0.6889 |     0.6327 |        90 |
| Neutro        |      0.4889 |   0.4231 |     0.4536 |        52 |
| Paura         |      0.7500 |   0.2500 |     0.3750 |        12 |
| Rabbia        |      0.5714 |   0.4444 |     0.5000 |        27 |
| Sorpresa      |      0.4100 |   0.4607 |     0.4339 |        89 |
| Tristezza     |      0.5091 |   0.6667 |     0.5773 |        42 |
| micro avg     |      0.5303 |   0.5635 |     0.5464 |       543 |
| macro avg     |      0.5569 |   0.5185 |     0.5205 |       543 |
| weighted avg  |      0.5352 |   0.5635 |     0.5420 |       543 |
| samples avg   |      0.5453 |   0.5492 |     0.5277 |       543 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  338 |   18 |   19 |   30 |      0.6250 |   0.6122 |     0.6186 |        49 |
| Anticipazione |  371 |    9 |   15 |   10 |      0.5263 |   0.4000 |     0.4545 |        25 |
| Disgusto      |  301 |   30 |   34 |   40 |      0.5714 |   0.5405 |     0.5556 |        74 |
| Fiducia       |  271 |   51 |   25 |   58 |      0.5321 |   0.6988 |     0.6042 |        83 |
| Gioia         |  271 |   44 |   28 |   62 |      0.5849 |   0.6889 |     0.6327 |        90 |
| Neutro        |  330 |   23 |   30 |   22 |      0.4889 |   0.4231 |     0.4536 |        52 |
| Paura         |  392 |    1 |    9 |    3 |      0.7500 |   0.2500 |     0.3750 |        12 |
| Rabbia        |  369 |    9 |   15 |   12 |      0.5714 |   0.4444 |     0.5000 |        27 |
| Sorpresa      |  257 |   59 |   48 |   41 |      0.4100 |   0.4607 |     0.4339 |        89 |
| Tristezza     |  336 |   27 |   14 |   28 |      0.5091 |   0.6667 |     0.5773 |        42 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             49 |           30 |                    0 |               1 |             24 |           14 |             0 |            0 |             1 |               2 |                1 |           0 |
| Anticipazione  |             25 |            0 |                   10 |               4 |              5 |            3 |             0 |            2 |             1 |               3 |                5 |           0 |
| Disgusto       |             74 |            0 |                    2 |              40 |              7 |            2 |             7 |            0 |            16 |              24 |               22 |           0 |
| Fiducia        |             83 |           20 |                    4 |               5 |             58 |           14 |             3 |            1 |             1 |              13 |               10 |           0 |
| Gioia          |             90 |           13 |                    2 |               2 |             16 |           62 |             4 |            0 |             0 |              26 |                2 |           0 |
| Neutro         |             52 |            2 |                    1 |               6 |             11 |           14 |            22 |            0 |             0 |              12 |                0 |           0 |
| Paura          |             12 |            0 |                    1 |               1 |              3 |            1 |             3 |            3 |             1 |               5 |                4 |           0 |
| Rabbia         |             27 |            0 |                    0 |              20 |              0 |            1 |             0 |            0 |            12 |               7 |                9 |           0 |
| Sorpresa       |             89 |            3 |                    3 |              20 |             13 |           20 |             8 |            2 |             8 |              41 |               23 |           0 |
| Tristezza      |             42 |            0 |                    2 |              13 |              7 |            0 |             2 |            1 |             5 |              16 |               28 |           0 |

#### Textual Case: `Step 2 Oracle (Train: +rel)` (`emotion_textual_corr_with_rel`)
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.6038 |   0.6531 |     0.6275 |        49 |
| Anticipazione |      0.4194 |   0.5200 |     0.4643 |        25 |
| Disgusto      |      0.5529 |   0.6351 |     0.5912 |        74 |
| Fiducia       |      0.6163 |   0.6386 |     0.6272 |        83 |
| Gioia         |      0.5556 |   0.6111 |     0.5820 |        90 |
| Neutro        |      0.4340 |   0.4423 |     0.4381 |        52 |
| Paura         |      0.8333 |   0.4167 |     0.5556 |        12 |
| Rabbia        |      0.3704 |   0.3704 |     0.3704 |        27 |
| Sorpresa      |      0.4400 |   0.4944 |     0.4656 |        89 |
| Tristezza     |      0.5000 |   0.6429 |     0.5625 |        42 |
| micro avg     |      0.5202 |   0.5691 |     0.5435 |       543 |
| macro avg     |      0.5326 |   0.5424 |     0.5284 |       543 |
| weighted avg  |      0.5246 |   0.5691 |     0.5434 |       543 |
| samples avg   |      0.5342 |   0.5521 |     0.5234 |       543 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  335 |   21 |   17 |   32 |      0.6038 |   0.6531 |     0.6275 |        49 |
| Anticipazione |  362 |   18 |   12 |   13 |      0.4194 |   0.5200 |     0.4643 |        25 |
| Disgusto      |  293 |   38 |   27 |   47 |      0.5529 |   0.6351 |     0.5912 |        74 |
| Fiducia       |  289 |   33 |   30 |   53 |      0.6163 |   0.6386 |     0.6272 |        83 |
| Gioia         |  271 |   44 |   35 |   55 |      0.5556 |   0.6111 |     0.5820 |        90 |
| Neutro        |  323 |   30 |   29 |   23 |      0.4340 |   0.4423 |     0.4381 |        52 |
| Paura         |  392 |    1 |    7 |    5 |      0.8333 |   0.4167 |     0.5556 |        12 |
| Rabbia        |  361 |   17 |   17 |   10 |      0.3704 |   0.3704 |     0.3704 |        27 |
| Sorpresa      |  260 |   56 |   45 |   44 |      0.4400 |   0.4944 |     0.4656 |        89 |
| Tristezza     |  336 |   27 |   15 |   27 |      0.5000 |   0.6429 |     0.5625 |        42 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             49 |           32 |                    0 |               2 |             18 |           12 |             0 |            0 |             1 |               3 |                1 |           0 |
| Anticipazione  |             25 |            0 |                   13 |               5 |              6 |            2 |             1 |            2 |             1 |               1 |                4 |           0 |
| Disgusto       |             74 |            0 |                    2 |              47 |              8 |            4 |             7 |            0 |            19 |              18 |               20 |           0 |
| Fiducia        |             83 |           21 |                    5 |               5 |             53 |           15 |             6 |            1 |             2 |              12 |                9 |           0 |
| Gioia          |             90 |           14 |                    4 |               3 |             12 |           55 |             3 |            0 |             0 |              37 |                2 |           0 |
| Neutro         |             52 |            2 |                    5 |               6 |              4 |           12 |            23 |            0 |             1 |              10 |                0 |           0 |
| Paura          |             12 |            0 |                    1 |               3 |              2 |            1 |             3 |            5 |             1 |               4 |                7 |           0 |
| Rabbia         |             27 |            0 |                    1 |              21 |              0 |            2 |             0 |            0 |            10 |               8 |                6 |           0 |
| Sorpresa       |             89 |            3 |                    3 |              27 |              9 |           20 |            14 |            3 |            12 |              44 |               24 |           0 |
| Tristezza      |             42 |            0 |                    4 |              21 |              8 |            0 |             3 |            2 |            10 |              11 |               27 |           0 |

---

### 1.2 Multimodal Emotion Benchmark (`mm_test`, $N=129$ — All 6 Cases $\times$ `raw` vs `std`)

#### Master Comparison Table (All Multimodal Emotion Runs)
| Case   | Regime                                      | Train Relation   | Eval Relation   | Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:-------|:--------------------------------------------|:-----------------|:----------------|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Blind Text (ClassifierChain)     | raw       |     0.3263 |     0.3873 |        0.3828 |        0.2093 |
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4079 |     0.4241 |        0.4121 |        0.2171 |
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Blind Text (ClassifierChain)     | std       |     0.3381 |     0.4020 |        0.4312 |        0.1550 |
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.3276 |     0.4149 |        0.4647 |        0.1783 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Blind Text (ClassifierChain)     | raw       |     0.4408 |     0.5232 |        0.5237 |        0.2868 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.3912 |     0.4403 |        0.4461 |        0.2636 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Blind Text (ClassifierChain)     | std       |     0.3839 |     0.4375 |        0.5065 |        0.1318 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.3809 |     0.4538 |        0.4737 |        0.2016 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Blind Text (ClassifierChain)     | raw       |     0.4219 |     0.5123 |        0.5386 |        0.2636 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4837 |     0.5285 |        0.5395 |        0.2791 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Blind Text (ClassifierChain)     | std       |     0.4352 |     0.4802 |        0.5539 |        0.2636 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4525 |     0.5260 |        0.5575 |        0.3333 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Blind Text (ClassifierChain)     | raw       |     0.4093 |     0.4786 |        0.5040 |        0.2713 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4436 |     0.5385 |        0.5385 |        0.4109 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Blind Text (ClassifierChain)     | std       |     0.4100 |     0.5027 |        0.5286 |        0.2558 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4441 |     0.5215 |        0.5553 |        0.3101 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Blind Text (ClassifierChain)     | raw       |     0.4438 |     0.5500 |        0.5631 |        0.4186 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.3986 |     0.5000 |        0.5271 |        0.3333 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Blind Text (ClassifierChain)     | std       |     0.4247 |     0.5041 |        0.5483 |        0.3023 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4548 |     0.5333 |        0.5502 |        0.3411 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Blind Text (ClassifierChain)     | raw       |     0.4207 |     0.5029 |        0.5196 |        0.3333 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4306 |     0.5333 |        0.5389 |        0.3333 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Blind Text (ClassifierChain)     | std       |     0.4504 |     0.5066 |        0.5515 |        0.3178 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4337 |     0.5000 |        0.5336 |        0.2868 |

#### Per-Emotion F1 Breakdown (All Multimodal Emotion Runs)
| Case   | Regime                                      | Model                            | Scaling   |   Amore F1 |   Anticipazione F1 |   Disgusto F1 |   Fiducia F1 |   Gioia F1 |   Neutro F1 |   Paura F1 |   Rabbia F1 |   Sorpresa F1 |   Tristezza F1 |
|:-------|:--------------------------------------------|:---------------------------------|:----------|-----------:|-------------------:|--------------:|-------------:|-----------:|------------:|-----------:|------------:|--------------:|---------------:|
| Case 1 | Step 1 Base (No corr)                       | Blind Text (ClassifierChain)     | raw       |     0.3000 |             0.4000 |        0.4615 |       0.5625 |     0.2162 |      0.3448 |     0.0769 |      0.3750 |        0.2400 |         0.2857 |
| Case 1 | Step 1 Base (No corr)                       | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.3158 |             0.3158 |        0.2941 |       0.7158 |     0.3462 |      0.1379 |     0.6667 |      0.3529 |        0.2667 |         0.6667 |
| Case 1 | Step 1 Base (No corr)                       | Blind Text (ClassifierChain)     | std       |     0.2581 |             0.3871 |        0.6383 |       0.5684 |     0.3881 |      0.3590 |     0.0000 |      0.2609 |        0.2353 |         0.2857 |
| Case 1 | Step 1 Base (No corr)                       | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.2400 |             0.4000 |        0.3390 |       0.7222 |     0.5246 |      0.2791 |     0.0000 |      0.2857 |        0.3600 |         0.1250 |
| Case 2 | Step 2 Raw (No corr)                        | Blind Text (ClassifierChain)     | raw       |     0.5714 |             0.3636 |        0.5854 |       0.6552 |     0.4800 |      0.4571 |     0.0000 |      0.3846 |        0.4103 |         0.5000 |
| Case 2 | Step 2 Raw (No corr)                        | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.3590 |             0.4211 |        0.3784 |       0.6154 |     0.5455 |      0.2857 |     0.0000 |      0.3333 |        0.1739 |         0.8000 |
| Case 2 | Step 2 Raw (No corr)                        | Blind Text (ClassifierChain)     | std       |     0.5556 |             0.4138 |        0.5652 |       0.6824 |     0.4935 |      0.2326 |     0.0000 |      0.3200 |        0.4651 |         0.1111 |
| Case 2 | Step 2 Raw (No corr)                        | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.5238 |             0.4706 |        0.3030 |       0.6250 |     0.5231 |      0.3415 |     0.0000 |      0.4211 |        0.3784 |         0.2222 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (ClassifierChain)     | raw       |     0.3077 |             0.3846 |        0.7000 |       0.6667 |     0.6000 |      0.4138 |     0.1176 |      0.3636 |        0.4151 |         0.2500 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.2759 |             0.5263 |        0.5714 |       0.7111 |     0.5614 |      0.3810 |     0.6667 |      0.3529 |        0.4571 |         0.3333 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (ClassifierChain)     | std       |     0.3889 |             0.5600 |        0.7429 |       0.6301 |     0.6429 |      0.3913 |     0.0556 |      0.3000 |        0.4186 |         0.2222 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4000 |             0.5714 |        0.6000 |       0.7045 |     0.5965 |      0.4762 |     0.1818 |      0.4000 |        0.3590 |         0.2353 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Blind Text (ClassifierChain)     | raw       |     0.3810 |             0.6087 |        0.6522 |       0.5789 |     0.5397 |      0.4000 |     0.0000 |      0.3750 |        0.3077 |         0.2500 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.3636 |             0.5882 |        0.5455 |       0.6957 |     0.5862 |      0.5000 |     0.0000 |      0.3077 |        0.2778 |         0.5714 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Blind Text (ClassifierChain)     | std       |     0.3871 |             0.4118 |        0.6667 |       0.6437 |     0.6462 |      0.4000 |     0.1250 |      0.1667 |        0.3448 |         0.3077 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4615 |             0.5000 |        0.5946 |       0.6882 |     0.6552 |      0.4400 |     0.1538 |      0.4444 |        0.3125 |         0.1905 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Blind Text (ClassifierChain)     | raw       |     0.4615 |             0.4167 |        0.6667 |       0.7000 |     0.6333 |      0.4706 |     0.0000 |      0.5455 |        0.2941 |         0.2500 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.2667 |             0.4762 |        0.6500 |       0.6753 |     0.6429 |      0.3590 |     0.0000 |      0.4615 |        0.2727 |         0.1818 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Blind Text (ClassifierChain)     | std       |     0.4706 |             0.4615 |        0.6047 |       0.7089 |     0.6275 |      0.3922 |     0.0000 |      0.4615 |        0.2979 |         0.2222 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4242 |             0.4000 |        0.6190 |       0.6739 |     0.6452 |      0.4571 |     0.2000 |      0.4615 |        0.3333 |         0.3333 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Blind Text (ClassifierChain)     | raw       |     0.3333 |             0.5455 |        0.5833 |       0.6585 |     0.5862 |      0.3871 |     0.0000 |      0.3529 |        0.3158 |         0.4444 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.2222 |             0.5000 |        0.7317 |       0.6526 |     0.6316 |      0.3333 |     0.0000 |      0.4444 |        0.3902 |         0.4000 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Blind Text (ClassifierChain)     | std       |     0.3030 |             0.4615 |        0.7179 |       0.6292 |     0.6452 |      0.5116 |     0.0000 |      0.5714 |        0.3000 |         0.3636 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.2581 |             0.4615 |        0.6842 |       0.6452 |     0.6154 |      0.4390 |     0.0909 |      0.4000 |        0.3429 |         0.4000 |

#### Semiotic Subsets Breakdown (`ADD`, `PAR`, `DIV` on `mm_test`, $N=129$)
| Case   | Regime                                      | Model                            | Scaling   |   ADD (N=68) |   PAR (N=23) |   DIV (N=38) |   ADD Macro F1 |   PAR Macro F1 |   DIV Macro F1 |   Macro F1 |   Micro F1 |
|:-------|:--------------------------------------------|:---------------------------------|:----------|-------------:|-------------:|-------------:|---------------:|---------------:|---------------:|-----------:|-----------:|
| Case 1 | Step 1 Base (No corr)                       | Blind Text (ClassifierChain)     | raw       |       0.4842 |       0.2979 |       0.2569 |         0.4300 |         0.0864 |         0.1741 |     0.3263 |     0.3873 |
| Case 1 | Step 1 Base (No corr)                       | Dual-Encoder (FEEL-IT + ViT-224) | raw       |       0.4444 |       0.5818 |       0.3030 |         0.4229 |         0.2883 |         0.1816 |     0.4079 |     0.4241 |
| Case 1 | Step 1 Base (No corr)                       | Blind Text (ClassifierChain)     | std       |       0.4498 |       0.4058 |       0.3200 |         0.4043 |         0.2672 |         0.2516 |     0.3381 |     0.4020 |
| Case 1 | Step 1 Base (No corr)                       | Dual-Encoder (FEEL-IT + ViT-224) | std       |       0.4876 |       0.3934 |       0.2982 |         0.3980 |         0.2730 |         0.1972 |     0.3276 |     0.4149 |
| Case 2 | Step 2 Raw (No corr)                        | Blind Text (ClassifierChain)     | raw       |       0.5584 |       0.5862 |       0.4286 |         0.4884 |         0.4716 |         0.3788 |     0.4408 |     0.5232 |
| Case 2 | Step 2 Raw (No corr)                        | Dual-Encoder (FEEL-IT + ViT-224) | raw       |       0.5181 |       0.4561 |       0.2947 |         0.4485 |         0.2458 |         0.2048 |     0.3912 |     0.4403 |
| Case 2 | Step 2 Raw (No corr)                        | Blind Text (ClassifierChain)     | std       |       0.4603 |       0.3448 |       0.4590 |         0.4320 |         0.3219 |         0.3967 |     0.3839 |     0.4375 |
| Case 2 | Step 2 Raw (No corr)                        | Dual-Encoder (FEEL-IT + ViT-224) | std       |       0.5149 |       0.5161 |       0.3130 |         0.4650 |         0.2732 |         0.2579 |     0.3809 |     0.4538 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (ClassifierChain)     | raw       |       0.5274 |       0.5556 |       0.4643 |         0.4686 |         0.3457 |         0.2940 |     0.4219 |     0.5123 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (FEEL-IT + ViT-224) | raw       |       0.5730 |       0.5172 |       0.4536 |         0.5248 |         0.2746 |         0.2743 |     0.4837 |     0.5285 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (ClassifierChain)     | std       |       0.5400 |       0.4746 |       0.3833 |         0.4957 |         0.4081 |         0.3116 |     0.4352 |     0.4802 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (FEEL-IT + ViT-224) | std       |       0.5579 |       0.5818 |       0.4500 |         0.4973 |         0.3733 |         0.3402 |     0.4525 |     0.5260 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Blind Text (ClassifierChain)     | raw       |       0.5131 |       0.3929 |       0.4615 |         0.4684 |         0.2443 |         0.3196 |     0.4093 |     0.4786 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Dual-Encoder (FEEL-IT + ViT-224) | raw       |       0.5697 |       0.6154 |       0.4421 |         0.4791 |         0.3296 |         0.2831 |     0.4436 |     0.5385 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Blind Text (ClassifierChain)     | std       |       0.5300 |       0.5000 |       0.4545 |         0.4693 |         0.3590 |         0.3455 |     0.4100 |     0.5027 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Dual-Encoder (FEEL-IT + ViT-224) | std       |       0.5949 |       0.5538 |       0.3750 |         0.5305 |         0.3513 |         0.2681 |     0.4441 |     0.5215 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Blind Text (ClassifierChain)     | raw       |       0.5795 |       0.5833 |       0.4792 |         0.4884 |         0.3667 |         0.3288 |     0.4438 |     0.5500 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Dual-Encoder (FEEL-IT + ViT-224) | raw       |       0.5486 |       0.5283 |       0.4038 |         0.4532 |         0.2886 |         0.2481 |     0.3986 |     0.5000 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Blind Text (ClassifierChain)     | std       |       0.5185 |       0.5667 |       0.4483 |         0.4609 |         0.4281 |         0.3306 |     0.4247 |     0.5041 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Dual-Encoder (FEEL-IT + ViT-224) | std       |       0.5773 |       0.5091 |       0.4685 |         0.5137 |         0.3421 |         0.3287 |     0.4548 |     0.5333 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Blind Text (ClassifierChain)     | raw       |       0.5435 |       0.5600 |       0.4107 |         0.4708 |         0.3556 |         0.2339 |     0.4207 |     0.5029 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Dual-Encoder (FEEL-IT + ViT-224) | raw       |       0.5667 |       0.5424 |       0.4717 |         0.4753 |         0.3095 |         0.2763 |     0.4306 |     0.5333 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Blind Text (ClassifierChain)     | std       |       0.5314 |       0.5246 |       0.4505 |         0.4884 |         0.3303 |         0.2920 |     0.4504 |     0.5066 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Dual-Encoder (FEEL-IT + ViT-224) | std       |       0.5288 |       0.5424 |       0.4220 |         0.4705 |         0.3365 |         0.2827 |     0.4337 |     0.5000 |

### 1.3 Multimodal Emotion Detailed Reports by Case (`Case 1` – `Case 6`)

#### Case 1: Step 1 Base (No corr) (Checkpoint: `emotion_textual_no_corr`, Eval Prompt: `None (Raw)`)

| Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Blind Text (ClassifierChain)     | raw       |     0.3263 |     0.3873 |        0.3828 |        0.2093 |
| Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4079 |     0.4241 |        0.4121 |        0.2171 |
| Blind Text (ClassifierChain)     | std       |     0.3381 |     0.4020 |        0.4312 |        0.1550 |
| Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.3276 |     0.4149 |        0.4647 |        0.1783 |

##### Run: `Case 1` | `Blind Text (ClassifierChain)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3333 |   0.2727 |     0.3000 |        11 |
| Anticipazione |      0.3333 |   0.5000 |     0.4000 |        10 |
| Disgusto      |      0.3261 |   0.7895 |     0.4615 |        19 |
| Fiducia       |      0.5000 |   0.6429 |     0.5625 |        42 |
| Gioia         |      0.5000 |   0.1379 |     0.2162 |        29 |
| Neutro        |      0.4167 |   0.2941 |     0.3448 |        17 |
| Paura         |      0.0400 |   1.0000 |     0.0769 |         1 |
| Rabbia        |      0.3000 |   0.5000 |     0.3750 |         6 |
| Sorpresa      |      0.3333 |   0.1875 |     0.2400 |        16 |
| Tristezza     |      0.2000 |   0.5000 |     0.2857 |         2 |
| micro avg     |      0.3472 |   0.4379 |     0.3873 |       153 |
| macro avg     |      0.3283 |   0.4825 |     0.3263 |       153 |
| weighted avg  |      0.4141 |   0.4379 |     0.3828 |       153 |
| samples avg   |      0.3598 |   0.4328 |     0.3742 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  112 |    6 |    8 |    3 |      0.3333 |   0.2727 |     0.3000 |        11 |
| Anticipazione |  109 |   10 |    5 |    5 |      0.3333 |   0.5000 |     0.4000 |        10 |
| Disgusto      |   79 |   31 |    4 |   15 |      0.3261 |   0.7895 |     0.4615 |        19 |
| Fiducia       |   60 |   27 |   15 |   27 |      0.5000 |   0.6429 |     0.5625 |        42 |
| Gioia         |   96 |    4 |   25 |    4 |      0.5000 |   0.1379 |     0.2162 |        29 |
| Neutro        |  105 |    7 |   12 |    5 |      0.4167 |   0.2941 |     0.3448 |        17 |
| Paura         |  104 |   24 |    0 |    1 |      0.0400 |   1.0000 |     0.0769 |         1 |
| Rabbia        |  116 |    7 |    3 |    3 |      0.3000 |   0.5000 |     0.3750 |         6 |
| Sorpresa      |  107 |    6 |   13 |    3 |      0.3333 |   0.1875 |     0.2400 |        16 |
| Tristezza     |  123 |    4 |    1 |    1 |      0.2000 |   0.5000 |     0.2857 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            3 |                    0 |               0 |             11 |            3 |             0 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            0 |                    5 |               0 |              6 |            1 |             0 |            1 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            1 |                    0 |              15 |              2 |            1 |             1 |           13 |             4 |               2 |                2 |           0 |
| Fiducia        |             42 |            3 |                    3 |               8 |             27 |            4 |             2 |            2 |             1 |               1 |                1 |           0 |
| Gioia          |             29 |            3 |                    4 |               8 |             13 |            4 |             2 |            1 |             1 |               3 |                0 |           1 |
| Neutro         |             17 |            1 |                    2 |               6 |              4 |            0 |             5 |            2 |             1 |               0 |                0 |           0 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               0 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               4 |              2 |            0 |             0 |            4 |             3 |               1 |                2 |           0 |
| Sorpresa       |             16 |            1 |                    2 |               8 |              3 |            1 |             2 |            7 |             2 |               3 |                2 |           1 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             1 |            2 |             1 |               0 |                1 |           0 |

##### Run: `Case 1` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3750 |   0.2727 |     0.3158 |        11 |
| Anticipazione |      0.3333 |   0.3000 |     0.3158 |        10 |
| Disgusto      |      0.3333 |   0.2632 |     0.2941 |        19 |
| Fiducia       |      0.6415 |   0.8095 |     0.7158 |        42 |
| Gioia         |      0.3913 |   0.3103 |     0.3462 |        29 |
| Neutro        |      0.1667 |   0.1176 |     0.1379 |        17 |
| Paura         |      0.5000 |   1.0000 |     0.6667 |         1 |
| Rabbia        |      0.2727 |   0.5000 |     0.3529 |         6 |
| Sorpresa      |      0.2069 |   0.3750 |     0.2667 |        16 |
| Tristezza     |      1.0000 |   0.5000 |     0.6667 |         2 |
| micro avg     |      0.4110 |   0.4379 |     0.4241 |       153 |
| macro avg     |      0.4221 |   0.4448 |     0.4079 |       153 |
| weighted avg  |      0.4076 |   0.4379 |     0.4121 |       153 |
| samples avg   |      0.3708 |   0.4393 |     0.3835 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  113 |    5 |    8 |    3 |      0.3750 |   0.2727 |     0.3158 |        11 |
| Anticipazione |  113 |    6 |    7 |    3 |      0.3333 |   0.3000 |     0.3158 |        10 |
| Disgusto      |  100 |   10 |   14 |    5 |      0.3333 |   0.2632 |     0.2941 |        19 |
| Fiducia       |   68 |   19 |    8 |   34 |      0.6415 |   0.8095 |     0.7158 |        42 |
| Gioia         |   86 |   14 |   20 |    9 |      0.3913 |   0.3103 |     0.3462 |        29 |
| Neutro        |  102 |   10 |   15 |    2 |      0.1667 |   0.1176 |     0.1379 |        17 |
| Paura         |  127 |    1 |    0 |    1 |      0.5000 |   1.0000 |     0.6667 |         1 |
| Rabbia        |  115 |    8 |    3 |    3 |      0.2727 |   0.5000 |     0.3529 |         6 |
| Sorpresa      |   90 |   23 |   10 |    6 |      0.2069 |   0.3750 |     0.2667 |        16 |
| Tristezza     |  127 |    0 |    1 |    1 |      1.0000 |   0.5000 |     0.6667 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            3 |                    0 |               0 |              9 |            1 |             0 |            0 |             1 |               1 |                0 |           0 |
| Anticipazione  |             10 |            0 |                    3 |               0 |              3 |            3 |             1 |            0 |             0 |               0 |                0 |           3 |
| Disgusto       |             19 |            0 |                    1 |               5 |              2 |            4 |             2 |            0 |             3 |               9 |                0 |           2 |
| Fiducia        |             42 |            2 |                    1 |               0 |             34 |            5 |             2 |            0 |             1 |               6 |                0 |           6 |
| Gioia          |             29 |            0 |                    1 |               3 |              9 |            9 |             5 |            0 |             1 |               5 |                0 |           6 |
| Neutro         |             17 |            2 |                    1 |               1 |              5 |            4 |             2 |            0 |             2 |               3 |                0 |           3 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               0 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               2 |              1 |            1 |             1 |            0 |             3 |               1 |                0 |           0 |
| Sorpresa       |             16 |            1 |                    2 |               5 |              2 |            4 |             2 |            1 |             3 |               6 |                1 |           3 |
| Tristezza      |              2 |            1 |                    1 |               1 |              0 |            0 |             0 |            1 |             2 |               1 |                1 |           0 |

##### Run: `Case 1` | `Blind Text (ClassifierChain)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2000 |   0.3636 |     0.2581 |        11 |
| Anticipazione |      0.2857 |   0.6000 |     0.3871 |        10 |
| Disgusto      |      0.5357 |   0.7895 |     0.6383 |        19 |
| Fiducia       |      0.5094 |   0.6429 |     0.5684 |        42 |
| Gioia         |      0.3421 |   0.4483 |     0.3881 |        29 |
| Neutro        |      0.3182 |   0.4118 |     0.3590 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.1765 |   0.5000 |     0.2609 |         6 |
| Sorpresa      |      0.2222 |   0.2500 |     0.2353 |        16 |
| Tristezza     |      0.1667 |   1.0000 |     0.2857 |         2 |
| micro avg     |      0.3240 |   0.5294 |     0.4020 |       153 |
| macro avg     |      0.2757 |   0.5006 |     0.3381 |       153 |
| weighted avg  |      0.3720 |   0.5294 |     0.4312 |       153 |
| samples avg   |      0.3491 |   0.5323 |     0.3915 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  102 |   16 |    7 |    4 |      0.2000 |   0.3636 |     0.2581 |        11 |
| Anticipazione |  104 |   15 |    4 |    6 |      0.2857 |   0.6000 |     0.3871 |        10 |
| Disgusto      |   97 |   13 |    4 |   15 |      0.5357 |   0.7895 |     0.6383 |        19 |
| Fiducia       |   61 |   26 |   15 |   27 |      0.5094 |   0.6429 |     0.5684 |        42 |
| Gioia         |   75 |   25 |   16 |   13 |      0.3421 |   0.4483 |     0.3881 |        29 |
| Neutro        |   97 |   15 |   10 |    7 |      0.3182 |   0.4118 |     0.3590 |        17 |
| Paura         |  107 |   21 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  109 |   14 |    3 |    3 |      0.1765 |   0.5000 |     0.2609 |         6 |
| Sorpresa      |   99 |   14 |   12 |    4 |      0.2222 |   0.2500 |     0.2353 |        16 |
| Tristezza     |  117 |   10 |    0 |    2 |      0.1667 |   1.0000 |     0.2857 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            4 |                    0 |               0 |             11 |            4 |             0 |            1 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            0 |                    6 |               0 |              4 |            3 |             2 |            2 |             1 |               1 |                0 |           0 |
| Disgusto       |             19 |            4 |                    0 |              15 |              4 |            4 |             3 |            4 |             6 |               3 |                2 |           1 |
| Fiducia        |             42 |            4 |                    3 |               3 |             27 |           14 |             4 |            2 |             3 |               6 |                2 |           4 |
| Gioia          |             29 |            5 |                    5 |               3 |             11 |           13 |             4 |            8 |             3 |               3 |                2 |           0 |
| Neutro         |             17 |            4 |                    6 |               2 |              4 |            4 |             7 |            2 |             1 |               3 |                2 |           1 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            1 |             0 |            0 |             1 |               0 |                1 |           0 |
| Rabbia         |              6 |            0 |                    0 |               4 |              3 |            0 |             1 |            1 |             3 |               0 |                2 |           0 |
| Sorpresa       |             16 |            3 |                    2 |               4 |              4 |            4 |             3 |            5 |             3 |               4 |                4 |           0 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             1 |            1 |             2 |               1 |                2 |           0 |

##### Run: `Case 1` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2143 |   0.2727 |     0.2400 |        11 |
| Anticipazione |      0.4000 |   0.4000 |     0.4000 |        10 |
| Disgusto      |      0.2500 |   0.5263 |     0.3390 |        19 |
| Fiducia       |      0.8667 |   0.6190 |     0.7222 |        42 |
| Gioia         |      0.5000 |   0.5517 |     0.5246 |        29 |
| Neutro        |      0.2308 |   0.3529 |     0.2791 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.2000 |   0.5000 |     0.2857 |         6 |
| Sorpresa      |      0.2647 |   0.5625 |     0.3600 |        16 |
| Tristezza     |      0.0714 |   0.5000 |     0.1250 |         2 |
| micro avg     |      0.3498 |   0.5098 |     0.4149 |       153 |
| macro avg     |      0.2998 |   0.4285 |     0.3276 |       153 |
| weighted avg  |      0.4674 |   0.5098 |     0.4647 |       153 |
| samples avg   |      0.3592 |   0.5039 |     0.3990 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  107 |   11 |    8 |    3 |      0.2143 |   0.2727 |     0.2400 |        11 |
| Anticipazione |  113 |    6 |    6 |    4 |      0.4000 |   0.4000 |     0.4000 |        10 |
| Disgusto      |   80 |   30 |    9 |   10 |      0.2500 |   0.5263 |     0.3390 |        19 |
| Fiducia       |   83 |    4 |   16 |   26 |      0.8667 |   0.6190 |     0.7222 |        42 |
| Gioia         |   84 |   16 |   13 |   16 |      0.5000 |   0.5517 |     0.5246 |        29 |
| Neutro        |   92 |   20 |   11 |    6 |      0.2308 |   0.3529 |     0.2791 |        17 |
| Paura         |  120 |    8 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  111 |   12 |    3 |    3 |      0.2000 |   0.5000 |     0.2857 |         6 |
| Sorpresa      |   88 |   25 |    7 |    9 |      0.2647 |   0.5625 |     0.3600 |        16 |
| Tristezza     |  114 |   13 |    1 |    1 |      0.0714 |   0.5000 |     0.1250 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            3 |                    0 |               1 |              5 |            2 |             1 |            1 |             0 |               2 |                1 |           1 |
| Anticipazione  |             10 |            1 |                    4 |               0 |              0 |            4 |             1 |            0 |             0 |               0 |                1 |           2 |
| Disgusto       |             19 |            1 |                    0 |              10 |              0 |            3 |             6 |            1 |             3 |               8 |                2 |           2 |
| Fiducia        |             42 |            3 |                    1 |               8 |             26 |           10 |             7 |            0 |             4 |               9 |                5 |           2 |
| Gioia          |             29 |            3 |                    3 |              11 |              6 |           16 |             4 |            2 |             4 |               7 |                2 |           2 |
| Neutro         |             17 |            3 |                    1 |               4 |              1 |            4 |             6 |            0 |             1 |               3 |                0 |           3 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            0 |             1 |               0 |                1 |           0 |
| Rabbia         |              6 |            0 |                    0 |               3 |              0 |            0 |             1 |            2 |             3 |               2 |                2 |           0 |
| Sorpresa       |             16 |            0 |                    1 |               7 |              1 |            5 |             3 |            3 |             4 |               9 |                2 |           0 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             1 |            1 |             2 |               1 |                1 |           0 |

#### Case 2: Step 2 Raw (No corr) (Checkpoint: `emotion_textual_step2_raw`, Eval Prompt: `None (Raw)`)

| Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Blind Text (ClassifierChain)     | raw       |     0.4408 |     0.5232 |        0.5237 |        0.2868 |
| Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.3912 |     0.4403 |        0.4461 |        0.2636 |
| Blind Text (ClassifierChain)     | std       |     0.3839 |     0.4375 |        0.5065 |        0.1318 |
| Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.3809 |     0.4538 |        0.4737 |        0.2016 |

##### Run: `Case 2` | `Blind Text (ClassifierChain)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.4706 |   0.7273 |     0.5714 |        11 |
| Anticipazione |      0.3333 |   0.4000 |     0.3636 |        10 |
| Disgusto      |      0.5455 |   0.6316 |     0.5854 |        19 |
| Fiducia       |      0.5135 |   0.9048 |     0.6552 |        42 |
| Gioia         |      0.5714 |   0.4138 |     0.4800 |        29 |
| Neutro        |      0.4444 |   0.4706 |     0.4571 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.2500 |   0.8333 |     0.3846 |         6 |
| Sorpresa      |      0.3478 |   0.5000 |     0.4103 |        16 |
| Tristezza     |      0.5000 |   0.5000 |     0.5000 |         2 |
| micro avg     |      0.4486 |   0.6275 |     0.5232 |       153 |
| macro avg     |      0.3977 |   0.5381 |     0.4408 |       153 |
| weighted avg  |      0.4747 |   0.6275 |     0.5237 |       153 |
| samples avg   |      0.4663 |   0.6072 |     0.5080 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  109 |    9 |    3 |    8 |      0.4706 |   0.7273 |     0.5714 |        11 |
| Anticipazione |  111 |    8 |    6 |    4 |      0.3333 |   0.4000 |     0.3636 |        10 |
| Disgusto      |  100 |   10 |    7 |   12 |      0.5455 |   0.6316 |     0.5854 |        19 |
| Fiducia       |   51 |   36 |    4 |   38 |      0.5135 |   0.9048 |     0.6552 |        42 |
| Gioia         |   91 |    9 |   17 |   12 |      0.5714 |   0.4138 |     0.4800 |        29 |
| Neutro        |  102 |   10 |    9 |    8 |      0.4444 |   0.4706 |     0.4571 |        17 |
| Paura         |  123 |    5 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  108 |   15 |    1 |    5 |      0.2500 |   0.8333 |     0.3846 |         6 |
| Sorpresa      |   98 |   15 |    8 |    8 |      0.3478 |   0.5000 |     0.4103 |        16 |
| Tristezza     |  126 |    1 |    1 |    1 |      0.5000 |   0.5000 |     0.5000 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            8 |                    0 |               0 |             11 |            5 |             0 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    4 |               0 |              4 |            1 |             1 |            0 |             2 |               0 |                0 |           0 |
| Disgusto       |             19 |            0 |                    1 |              12 |              6 |            1 |             1 |            1 |             6 |               9 |                0 |           0 |
| Fiducia        |             42 |            8 |                    4 |               1 |             38 |            8 |             1 |            0 |             4 |               1 |                0 |           0 |
| Gioia          |             29 |            6 |                    0 |               3 |             17 |           12 |             4 |            0 |             2 |               5 |                0 |           0 |
| Neutro         |             17 |            2 |                    2 |               1 |              7 |            0 |             8 |            1 |             2 |               3 |                0 |           0 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            1 |             0 |            0 |             1 |               1 |                0 |           0 |
| Rabbia         |              6 |            0 |                    1 |               4 |              1 |            0 |             0 |            2 |             5 |               1 |                1 |           0 |
| Sorpresa       |             16 |            1 |                    1 |               3 |              4 |            4 |             3 |            2 |             4 |               8 |                2 |           0 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             1 |            1 |             2 |               1 |                1 |           0 |

##### Run: `Case 2` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2500 |   0.6364 |     0.3590 |        11 |
| Anticipazione |      0.4444 |   0.4000 |     0.4211 |        10 |
| Disgusto      |      0.3889 |   0.3684 |     0.3784 |        19 |
| Fiducia       |      0.6667 |   0.5714 |     0.6154 |        42 |
| Gioia         |      0.5769 |   0.5172 |     0.5455 |        29 |
| Neutro        |      0.2400 |   0.3529 |     0.2857 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.2500 |   0.5000 |     0.3333 |         6 |
| Sorpresa      |      0.2857 |   0.1250 |     0.1739 |        16 |
| Tristezza     |      0.6667 |   1.0000 |     0.8000 |         2 |
| micro avg     |      0.4242 |   0.4575 |     0.4403 |       153 |
| macro avg     |      0.3769 |   0.4471 |     0.3912 |       153 |
| weighted avg  |      0.4627 |   0.4575 |     0.4461 |       153 |
| samples avg   |      0.4124 |   0.4638 |     0.4164 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |   97 |   21 |    4 |    7 |      0.2500 |   0.6364 |     0.3590 |        11 |
| Anticipazione |  114 |    5 |    6 |    4 |      0.4444 |   0.4000 |     0.4211 |        10 |
| Disgusto      |   99 |   11 |   12 |    7 |      0.3889 |   0.3684 |     0.3784 |        19 |
| Fiducia       |   75 |   12 |   18 |   24 |      0.6667 |   0.5714 |     0.6154 |        42 |
| Gioia         |   89 |   11 |   14 |   15 |      0.5769 |   0.5172 |     0.5455 |        29 |
| Neutro        |   93 |   19 |   11 |    6 |      0.2400 |   0.3529 |     0.2857 |        17 |
| Paura         |  127 |    1 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  114 |    9 |    3 |    3 |      0.2500 |   0.5000 |     0.3333 |         6 |
| Sorpresa      |  108 |    5 |   14 |    2 |      0.2857 |   0.1250 |     0.1739 |        16 |
| Tristezza     |  126 |    1 |    0 |    2 |      0.6667 |   1.0000 |     0.8000 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            7 |                    0 |               0 |              6 |            0 |             2 |            0 |             1 |               0 |                0 |           1 |
| Anticipazione  |             10 |            3 |                    4 |               0 |              0 |            2 |             1 |            0 |             0 |               0 |                0 |           2 |
| Disgusto       |             19 |            3 |                    0 |               7 |              0 |            2 |             6 |            0 |             3 |               3 |                0 |           2 |
| Fiducia        |             42 |            9 |                    0 |               0 |             24 |            6 |             5 |            0 |             1 |               0 |                0 |           9 |
| Gioia          |             29 |            5 |                    2 |               3 |              4 |           15 |             5 |            0 |             2 |               2 |                0 |           1 |
| Neutro         |             17 |            3 |                    2 |               2 |              6 |            4 |             6 |            0 |             2 |               1 |                0 |           2 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            0 |             0 |               0 |                0 |           0 |
| Rabbia         |              6 |            1 |                    0 |               3 |              0 |            1 |             1 |            0 |             3 |               0 |                1 |           0 |
| Sorpresa       |             16 |            3 |                    1 |               5 |              1 |            5 |             2 |            1 |             3 |               2 |                3 |           1 |
| Tristezza      |              2 |            1 |                    0 |               1 |              0 |            0 |             0 |            1 |             2 |               0 |                2 |           0 |

##### Run: `Case 2` | `Blind Text (ClassifierChain)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.4000 |   0.9091 |     0.5556 |        11 |
| Anticipazione |      0.3158 |   0.6000 |     0.4138 |        10 |
| Disgusto      |      0.4815 |   0.6842 |     0.5652 |        19 |
| Fiducia       |      0.6744 |   0.6905 |     0.6824 |        42 |
| Gioia         |      0.3958 |   0.6552 |     0.4935 |        29 |
| Neutro        |      0.1923 |   0.2941 |     0.2326 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.2105 |   0.6667 |     0.3200 |         6 |
| Sorpresa      |      0.3704 |   0.6250 |     0.4651 |        16 |
| Tristezza     |      0.0588 |   1.0000 |     0.1111 |         2 |
| micro avg     |      0.3322 |   0.6405 |     0.4375 |       153 |
| macro avg     |      0.3100 |   0.6125 |     0.3839 |       153 |
| weighted avg  |      0.4385 |   0.6405 |     0.5065 |       153 |
| samples avg   |      0.3478 |   0.6034 |     0.4193 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  103 |   15 |    1 |   10 |      0.4000 |   0.9091 |     0.5556 |        11 |
| Anticipazione |  106 |   13 |    4 |    6 |      0.3158 |   0.6000 |     0.4138 |        10 |
| Disgusto      |   96 |   14 |    6 |   13 |      0.4815 |   0.6842 |     0.5652 |        19 |
| Fiducia       |   73 |   14 |   13 |   29 |      0.6744 |   0.6905 |     0.6824 |        42 |
| Gioia         |   71 |   29 |   10 |   19 |      0.3958 |   0.6552 |     0.4935 |        29 |
| Neutro        |   91 |   21 |   12 |    5 |      0.1923 |   0.2941 |     0.2326 |        17 |
| Paura         |  101 |   27 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  108 |   15 |    2 |    4 |      0.2105 |   0.6667 |     0.3200 |         6 |
| Sorpresa      |   96 |   17 |    6 |   10 |      0.3704 |   0.6250 |     0.4651 |        16 |
| Tristezza     |   95 |   32 |    0 |    2 |      0.0588 |   1.0000 |     0.1111 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |           10 |                    1 |               0 |              8 |            8 |             2 |            1 |             1 |               0 |                2 |           0 |
| Anticipazione  |             10 |            2 |                    6 |               0 |              3 |            3 |             1 |            3 |             0 |               0 |                3 |           0 |
| Disgusto       |             19 |            0 |                    1 |              13 |              2 |            4 |             2 |            6 |             4 |               4 |                3 |           0 |
| Fiducia        |             42 |           11 |                    6 |               1 |             29 |           19 |             7 |            7 |             5 |               5 |               18 |           0 |
| Gioia          |             29 |            8 |                    4 |               3 |              8 |           19 |             7 |            4 |             3 |               9 |                3 |           1 |
| Neutro         |             17 |            2 |                    1 |               3 |              2 |            4 |             5 |            4 |             2 |               4 |                3 |           2 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            1 |             1 |            0 |             1 |               0 |                0 |           0 |
| Rabbia         |              6 |            0 |                    1 |               4 |              1 |            2 |             2 |            2 |             4 |               1 |                1 |           0 |
| Sorpresa       |             16 |            1 |                    1 |               6 |              2 |            7 |             3 |            3 |             3 |              10 |                3 |           2 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            1 |             0 |            1 |             2 |               2 |                2 |           0 |

##### Run: `Case 2` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3548 |   1.0000 |     0.5238 |        11 |
| Anticipazione |      0.3333 |   0.8000 |     0.4706 |        10 |
| Disgusto      |      0.3571 |   0.2632 |     0.3030 |        19 |
| Fiducia       |      0.6579 |   0.5952 |     0.6250 |        42 |
| Gioia         |      0.4722 |   0.5862 |     0.5231 |        29 |
| Neutro        |      0.2917 |   0.4118 |     0.3415 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.3077 |   0.6667 |     0.4211 |         6 |
| Sorpresa      |      0.3333 |   0.4375 |     0.3784 |        16 |
| Tristezza     |      0.1250 |   1.0000 |     0.2222 |         2 |
| micro avg     |      0.3805 |   0.5621 |     0.4538 |       153 |
| macro avg     |      0.3233 |   0.5761 |     0.3809 |       153 |
| weighted avg  |      0.4427 |   0.5621 |     0.4737 |       153 |
| samples avg   |      0.3806 |   0.5491 |     0.4277 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |   98 |   20 |    0 |   11 |      0.3548 |   1.0000 |     0.5238 |        11 |
| Anticipazione |  103 |   16 |    2 |    8 |      0.3333 |   0.8000 |     0.4706 |        10 |
| Disgusto      |  101 |    9 |   14 |    5 |      0.3571 |   0.2632 |     0.3030 |        19 |
| Fiducia       |   74 |   13 |   17 |   25 |      0.6579 |   0.5952 |     0.6250 |        42 |
| Gioia         |   81 |   19 |   12 |   17 |      0.4722 |   0.5862 |     0.5231 |        29 |
| Neutro        |   95 |   17 |   10 |    7 |      0.2917 |   0.4118 |     0.3415 |        17 |
| Paura         |  119 |    9 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  114 |    9 |    2 |    4 |      0.3077 |   0.6667 |     0.4211 |         6 |
| Sorpresa      |   99 |   14 |    9 |    7 |      0.3333 |   0.4375 |     0.3784 |        16 |
| Tristezza     |  113 |   14 |    0 |    2 |      0.1250 |   1.0000 |     0.2222 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |           11 |                    0 |               0 |              7 |            5 |             1 |            1 |             1 |               0 |                0 |           0 |
| Anticipazione  |             10 |            3 |                    8 |               0 |              1 |            7 |             1 |            0 |             1 |               0 |                1 |           0 |
| Disgusto       |             19 |            1 |                    1 |               5 |              0 |            2 |             5 |            1 |             3 |               6 |                5 |           2 |
| Fiducia        |             42 |           13 |                    5 |               1 |             25 |           11 |             1 |            1 |             2 |               2 |                3 |           4 |
| Gioia          |             29 |            7 |                    4 |               3 |              7 |           17 |             6 |            2 |             2 |               5 |                2 |           1 |
| Neutro         |             17 |            3 |                    6 |               0 |              3 |            3 |             7 |            1 |             1 |               4 |                2 |           1 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            0 |             0 |            0 |             1 |               0 |                0 |           0 |
| Rabbia         |              6 |            1 |                    0 |               2 |              0 |            1 |             1 |            2 |             4 |               1 |                3 |           1 |
| Sorpresa       |             16 |            1 |                    1 |               5 |              1 |            5 |             4 |            2 |             2 |               7 |                3 |           0 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               2 |                2 |           0 |

#### Case 3: Step 2 Oracle (Train: no rel, Eval: no rel) (Checkpoint: `emotion_textual_corr`, Eval Prompt: `Oracle (no rel)`)

| Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Blind Text (ClassifierChain)     | raw       |     0.4219 |     0.5123 |        0.5386 |        0.2636 |
| Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4837 |     0.5285 |        0.5395 |        0.2791 |
| Blind Text (ClassifierChain)     | std       |     0.4352 |     0.4802 |        0.5539 |        0.2636 |
| Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4525 |     0.5260 |        0.5575 |        0.3333 |

##### Run: `Case 3` | `Blind Text (ClassifierChain)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2667 |   0.3636 |     0.3077 |        11 |
| Anticipazione |      0.3125 |   0.5000 |     0.3846 |        10 |
| Disgusto      |      0.6667 |   0.7368 |     0.7000 |        19 |
| Fiducia       |      0.6444 |   0.6905 |     0.6667 |        42 |
| Gioia         |      0.5122 |   0.7241 |     0.6000 |        29 |
| Neutro        |      0.5000 |   0.3529 |     0.4138 |        17 |
| Paura         |      0.0625 |   1.0000 |     0.1176 |         1 |
| Rabbia        |      0.4000 |   0.3333 |     0.3636 |         6 |
| Sorpresa      |      0.2973 |   0.6875 |     0.4151 |        16 |
| Tristezza     |      0.1667 |   0.5000 |     0.2500 |         2 |
| micro avg     |      0.4393 |   0.6144 |     0.5123 |       153 |
| macro avg     |      0.3829 |   0.5889 |     0.4219 |       153 |
| weighted avg  |      0.5013 |   0.6144 |     0.5386 |       153 |
| samples avg   |      0.4598 |   0.6059 |     0.5011 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  107 |   11 |    7 |    4 |      0.2667 |   0.3636 |     0.3077 |        11 |
| Anticipazione |  108 |   11 |    5 |    5 |      0.3125 |   0.5000 |     0.3846 |        10 |
| Disgusto      |  103 |    7 |    5 |   14 |      0.6667 |   0.7368 |     0.7000 |        19 |
| Fiducia       |   71 |   16 |   13 |   29 |      0.6444 |   0.6905 |     0.6667 |        42 |
| Gioia         |   80 |   20 |    8 |   21 |      0.5122 |   0.7241 |     0.6000 |        29 |
| Neutro        |  106 |    6 |   11 |    6 |      0.5000 |   0.3529 |     0.4138 |        17 |
| Paura         |  113 |   15 |    0 |    1 |      0.0625 |   1.0000 |     0.1176 |         1 |
| Rabbia        |  120 |    3 |    4 |    2 |      0.4000 |   0.3333 |     0.3636 |         6 |
| Sorpresa      |   87 |   26 |    5 |   11 |      0.2973 |   0.6875 |     0.4151 |        16 |
| Tristezza     |  122 |    5 |    1 |    1 |      0.1667 |   0.5000 |     0.2500 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            4 |                    1 |               0 |              9 |            4 |             0 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            4 |                    5 |               0 |              2 |            4 |             1 |            0 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            1 |                    1 |              14 |              1 |            0 |             2 |           11 |             2 |               3 |                0 |           0 |
| Fiducia        |             42 |            5 |                    3 |               0 |             29 |           18 |             0 |            0 |             0 |               7 |                2 |           1 |
| Gioia          |             29 |            2 |                    3 |               1 |              5 |           21 |             2 |            0 |             0 |              16 |                2 |           1 |
| Neutro         |             17 |            0 |                    3 |               1 |              3 |            4 |             6 |            1 |             0 |               6 |                1 |           1 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               1 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               4 |              2 |            0 |             0 |            4 |             2 |               1 |                1 |           0 |
| Sorpresa       |             16 |            1 |                    1 |               2 |              3 |            6 |             2 |            2 |             1 |              11 |                2 |           0 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               2 |                1 |           0 |

##### Run: `Case 3` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2222 |   0.3636 |     0.2759 |        11 |
| Anticipazione |      0.5556 |   0.5000 |     0.5263 |        10 |
| Disgusto      |      0.6250 |   0.5263 |     0.5714 |        19 |
| Fiducia       |      0.6667 |   0.7619 |     0.7111 |        42 |
| Gioia         |      0.5714 |   0.5517 |     0.5614 |        29 |
| Neutro        |      0.3200 |   0.4706 |     0.3810 |        17 |
| Paura         |      0.5000 |   1.0000 |     0.6667 |         1 |
| Rabbia        |      0.2727 |   0.5000 |     0.3529 |         6 |
| Sorpresa      |      0.4211 |   0.5000 |     0.4571 |        16 |
| Tristezza     |      0.2500 |   0.5000 |     0.3333 |         2 |
| micro avg     |      0.4889 |   0.5752 |     0.5285 |       153 |
| macro avg     |      0.4405 |   0.5674 |     0.4837 |       153 |
| weighted avg  |      0.5180 |   0.5752 |     0.5395 |       153 |
| samples avg   |      0.4757 |   0.5788 |     0.4997 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  104 |   14 |    7 |    4 |      0.2222 |   0.3636 |     0.2759 |        11 |
| Anticipazione |  115 |    4 |    5 |    5 |      0.5556 |   0.5000 |     0.5263 |        10 |
| Disgusto      |  104 |    6 |    9 |   10 |      0.6250 |   0.5263 |     0.5714 |        19 |
| Fiducia       |   71 |   16 |   10 |   32 |      0.6667 |   0.7619 |     0.7111 |        42 |
| Gioia         |   88 |   12 |   13 |   16 |      0.5714 |   0.5517 |     0.5614 |        29 |
| Neutro        |   95 |   17 |    9 |    8 |      0.3200 |   0.4706 |     0.3810 |        17 |
| Paura         |  127 |    1 |    0 |    1 |      0.5000 |   1.0000 |     0.6667 |         1 |
| Rabbia        |  115 |    8 |    3 |    3 |      0.2727 |   0.5000 |     0.3529 |         6 |
| Sorpresa      |  102 |   11 |    8 |    8 |      0.4211 |   0.5000 |     0.4571 |        16 |
| Tristezza     |  124 |    3 |    1 |    1 |      0.2500 |   0.5000 |     0.3333 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            4 |                    0 |               0 |              7 |            3 |             3 |            0 |             1 |               0 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    5 |               0 |              3 |            4 |             2 |            0 |             0 |               0 |                0 |           1 |
| Disgusto       |             19 |            0 |                    1 |              10 |              0 |            0 |             3 |            0 |             5 |               5 |                1 |           2 |
| Fiducia        |             42 |            8 |                    1 |               0 |             32 |            8 |             6 |            0 |             1 |               1 |                0 |           5 |
| Gioia          |             29 |            5 |                    0 |               0 |              7 |           16 |             3 |            0 |             0 |               5 |                0 |           4 |
| Neutro         |             17 |            1 |                    1 |               1 |              4 |            3 |             8 |            0 |             2 |               3 |                0 |           2 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               1 |                1 |           0 |
| Rabbia         |              6 |            1 |                    0 |               3 |              1 |            0 |             0 |            0 |             3 |               1 |                2 |           0 |
| Sorpresa       |             16 |            1 |                    1 |               2 |              2 |            6 |             2 |            1 |             1 |               8 |                2 |           2 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               1 |                1 |           0 |

##### Run: `Case 3` | `Blind Text (ClassifierChain)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2800 |   0.6364 |     0.3889 |        11 |
| Anticipazione |      0.4667 |   0.7000 |     0.5600 |        10 |
| Disgusto      |      0.8125 |   0.6842 |     0.7429 |        19 |
| Fiducia       |      0.7419 |   0.5476 |     0.6301 |        42 |
| Gioia         |      0.6667 |   0.6207 |     0.6429 |        29 |
| Neutro        |      0.3103 |   0.5294 |     0.3913 |        17 |
| Paura         |      0.0286 |   1.0000 |     0.0556 |         1 |
| Rabbia        |      0.2143 |   0.5000 |     0.3000 |         6 |
| Sorpresa      |      0.3333 |   0.5625 |     0.4186 |        16 |
| Tristezza     |      0.1429 |   0.5000 |     0.2222 |         2 |
| micro avg     |      0.4027 |   0.5948 |     0.4802 |       153 |
| macro avg     |      0.3997 |   0.6281 |     0.4352 |       153 |
| weighted avg  |      0.5614 |   0.5948 |     0.5539 |       153 |
| samples avg   |      0.4438 |   0.5866 |     0.4832 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  100 |   18 |    4 |    7 |      0.2800 |   0.6364 |     0.3889 |        11 |
| Anticipazione |  111 |    8 |    3 |    7 |      0.4667 |   0.7000 |     0.5600 |        10 |
| Disgusto      |  107 |    3 |    6 |   13 |      0.8125 |   0.6842 |     0.7429 |        19 |
| Fiducia       |   79 |    8 |   19 |   23 |      0.7419 |   0.5476 |     0.6301 |        42 |
| Gioia         |   91 |    9 |   11 |   18 |      0.6667 |   0.6207 |     0.6429 |        29 |
| Neutro        |   92 |   20 |    8 |    9 |      0.3103 |   0.5294 |     0.3913 |        17 |
| Paura         |   94 |   34 |    0 |    1 |      0.0286 |   1.0000 |     0.0556 |         1 |
| Rabbia        |  112 |   11 |    3 |    3 |      0.2143 |   0.5000 |     0.3000 |         6 |
| Sorpresa      |   95 |   18 |    7 |    9 |      0.3333 |   0.5625 |     0.4186 |        16 |
| Tristezza     |  121 |    6 |    1 |    1 |      0.1429 |   0.5000 |     0.2222 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            7 |                    1 |               0 |              5 |            1 |             5 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            3 |                    7 |               0 |              2 |            2 |             1 |            2 |             0 |               0 |                0 |           1 |
| Disgusto       |             19 |            4 |                    1 |              13 |              1 |            0 |             2 |           16 |             6 |               2 |                1 |           0 |
| Fiducia        |             42 |            7 |                    5 |               0 |             23 |            8 |             3 |            4 |             2 |               5 |                1 |           4 |
| Gioia          |             29 |            7 |                    2 |               0 |              5 |           18 |             5 |            3 |             1 |              10 |                2 |           0 |
| Neutro         |             17 |            1 |                    1 |               1 |              0 |            3 |             9 |            3 |             2 |               5 |                1 |           1 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             1 |               1 |                0 |           0 |
| Rabbia         |              6 |            1 |                    0 |               4 |              2 |            0 |             0 |            5 |             3 |               1 |                1 |           0 |
| Sorpresa       |             16 |            1 |                    0 |               0 |              2 |            7 |             6 |            5 |             1 |               9 |                2 |           0 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            2 |             1 |               2 |                1 |           0 |

##### Run: `Case 3` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2917 |   0.6364 |     0.4000 |        11 |
| Anticipazione |      0.5455 |   0.6000 |     0.5714 |        10 |
| Disgusto      |      0.5714 |   0.6316 |     0.6000 |        19 |
| Fiducia       |      0.6739 |   0.7381 |     0.7045 |        42 |
| Gioia         |      0.6071 |   0.5862 |     0.5965 |        29 |
| Neutro        |      0.4000 |   0.5882 |     0.4762 |        17 |
| Paura         |      0.1000 |   1.0000 |     0.1818 |         1 |
| Rabbia        |      0.3333 |   0.5000 |     0.4000 |         6 |
| Sorpresa      |      0.3043 |   0.4375 |     0.3590 |        16 |
| Tristezza     |      0.1333 |   1.0000 |     0.2353 |         2 |
| micro avg     |      0.4528 |   0.6275 |     0.5260 |       153 |
| macro avg     |      0.3961 |   0.6718 |     0.4525 |       153 |
| weighted avg  |      0.5194 |   0.6275 |     0.5575 |       153 |
| samples avg   |      0.5028 |   0.6111 |     0.5339 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  101 |   17 |    4 |    7 |      0.2917 |   0.6364 |     0.4000 |        11 |
| Anticipazione |  114 |    5 |    4 |    6 |      0.5455 |   0.6000 |     0.5714 |        10 |
| Disgusto      |  101 |    9 |    7 |   12 |      0.5714 |   0.6316 |     0.6000 |        19 |
| Fiducia       |   72 |   15 |   11 |   31 |      0.6739 |   0.7381 |     0.7045 |        42 |
| Gioia         |   89 |   11 |   12 |   17 |      0.6071 |   0.5862 |     0.5965 |        29 |
| Neutro        |   97 |   15 |    7 |   10 |      0.4000 |   0.5882 |     0.4762 |        17 |
| Paura         |  119 |    9 |    0 |    1 |      0.1000 |   1.0000 |     0.1818 |         1 |
| Rabbia        |  117 |    6 |    3 |    3 |      0.3333 |   0.5000 |     0.4000 |         6 |
| Sorpresa      |   97 |   16 |    9 |    7 |      0.3043 |   0.4375 |     0.3590 |        16 |
| Tristezza     |  114 |   13 |    0 |    2 |      0.1333 |   1.0000 |     0.2353 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            7 |                    0 |               0 |              8 |            3 |             0 |            0 |             1 |               0 |                0 |           0 |
| Anticipazione  |             10 |            2 |                    6 |               0 |              1 |            2 |             0 |            0 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            1 |                    1 |              12 |              0 |            0 |             4 |            1 |             4 |               4 |                2 |           1 |
| Fiducia        |             42 |            8 |                    2 |               0 |             31 |           10 |             3 |            3 |             1 |               4 |                2 |           1 |
| Gioia          |             29 |           10 |                    1 |               4 |             10 |           17 |             4 |            2 |             0 |               7 |                4 |           1 |
| Neutro         |             17 |            2 |                    2 |               2 |              4 |            3 |            10 |            1 |             2 |               5 |                3 |           0 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               1 |                1 |           0 |
| Rabbia         |              6 |            1 |                    0 |               4 |              1 |            0 |             1 |            1 |             3 |               0 |                2 |           0 |
| Sorpresa       |             16 |            2 |                    0 |               1 |              3 |            7 |             4 |            2 |             0 |               7 |                4 |           0 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            1 |             0 |            1 |             0 |               1 |                2 |           0 |

#### Case 4: Step 2 Oracle (Train: no rel, Eval: +rel) (Checkpoint: `emotion_textual_corr`, Eval Prompt: `Oracle (+ rel)`)

| Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Blind Text (ClassifierChain)     | raw       |     0.4093 |     0.4786 |        0.5040 |        0.2713 |
| Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4436 |     0.5385 |        0.5385 |        0.4109 |
| Blind Text (ClassifierChain)     | std       |     0.4100 |     0.5027 |        0.5286 |        0.2558 |
| Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4441 |     0.5215 |        0.5553 |        0.3101 |

##### Run: `Case 4` | `Blind Text (ClassifierChain)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.4000 |   0.3636 |     0.3810 |        11 |
| Anticipazione |      0.5385 |   0.7000 |     0.6087 |        10 |
| Disgusto      |      0.5556 |   0.7895 |     0.6522 |        19 |
| Fiducia       |      0.6471 |   0.5238 |     0.5789 |        42 |
| Gioia         |      0.5000 |   0.5862 |     0.5397 |        29 |
| Neutro        |      0.2895 |   0.6471 |     0.4000 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.3000 |   0.5000 |     0.3750 |         6 |
| Sorpresa      |      0.4000 |   0.2500 |     0.3077 |        16 |
| Tristezza     |      0.1667 |   0.5000 |     0.2500 |         2 |
| micro avg     |      0.4242 |   0.5490 |     0.4786 |       153 |
| macro avg     |      0.3797 |   0.4860 |     0.4093 |       153 |
| weighted avg  |      0.4933 |   0.5490 |     0.5040 |       153 |
| samples avg   |      0.4496 |   0.5491 |     0.4736 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  112 |    6 |    7 |    4 |      0.4000 |   0.3636 |     0.3810 |        11 |
| Anticipazione |  113 |    6 |    3 |    7 |      0.5385 |   0.7000 |     0.6087 |        10 |
| Disgusto      |   98 |   12 |    4 |   15 |      0.5556 |   0.7895 |     0.6522 |        19 |
| Fiducia       |   75 |   12 |   20 |   22 |      0.6471 |   0.5238 |     0.5789 |        42 |
| Gioia         |   83 |   17 |   12 |   17 |      0.5000 |   0.5862 |     0.5397 |        29 |
| Neutro        |   85 |   27 |    6 |   11 |      0.2895 |   0.6471 |     0.4000 |        17 |
| Paura         |  112 |   16 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  116 |    7 |    3 |    3 |      0.3000 |   0.5000 |     0.3750 |         6 |
| Sorpresa      |  107 |    6 |   12 |    4 |      0.4000 |   0.2500 |     0.3077 |        16 |
| Tristezza     |  122 |    5 |    1 |    1 |      0.1667 |   0.5000 |     0.2500 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            4 |                    0 |               0 |              5 |            6 |             1 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            0 |                    7 |               0 |              2 |            2 |             2 |            1 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            0 |                    2 |              15 |              1 |            1 |             3 |            9 |             6 |               1 |                0 |           0 |
| Fiducia        |             42 |            5 |                    2 |               1 |             22 |           14 |             7 |            0 |             1 |               2 |                2 |           0 |
| Gioia          |             29 |            4 |                    1 |               5 |              5 |           17 |            10 |            0 |             0 |               4 |                2 |           0 |
| Neutro         |             17 |            1 |                    2 |               1 |              1 |            2 |            11 |            2 |             1 |               2 |                1 |           2 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            1 |             0 |            0 |             0 |               0 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               4 |              1 |            0 |             1 |            4 |             3 |               0 |                1 |           0 |
| Sorpresa       |             16 |            0 |                    0 |               3 |              3 |            6 |             6 |            3 |             1 |               4 |                2 |           0 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            2 |             1 |               1 |                1 |           0 |

##### Run: `Case 4` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3636 |   0.3636 |     0.3636 |        11 |
| Anticipazione |      0.7143 |   0.5000 |     0.5882 |        10 |
| Disgusto      |      0.6429 |   0.4737 |     0.5455 |        19 |
| Fiducia       |      0.6400 |   0.7619 |     0.6957 |        42 |
| Gioia         |      0.5862 |   0.5862 |     0.5862 |        29 |
| Neutro        |      0.5333 |   0.4706 |     0.5000 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.2857 |   0.3333 |     0.3077 |         6 |
| Sorpresa      |      0.2500 |   0.3125 |     0.2778 |        16 |
| Tristezza     |      0.4000 |   1.0000 |     0.5714 |         2 |
| micro avg     |      0.5283 |   0.5490 |     0.5385 |       153 |
| macro avg     |      0.4416 |   0.4802 |     0.4436 |       153 |
| weighted avg  |      0.5413 |   0.5490 |     0.5385 |       153 |
| samples avg   |      0.5284 |   0.5698 |     0.5315 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  111 |    7 |    7 |    4 |      0.3636 |   0.3636 |     0.3636 |        11 |
| Anticipazione |  117 |    2 |    5 |    5 |      0.7143 |   0.5000 |     0.5882 |        10 |
| Disgusto      |  105 |    5 |   10 |    9 |      0.6429 |   0.4737 |     0.5455 |        19 |
| Fiducia       |   69 |   18 |   10 |   32 |      0.6400 |   0.7619 |     0.6957 |        42 |
| Gioia         |   88 |   12 |   12 |   17 |      0.5862 |   0.5862 |     0.5862 |        29 |
| Neutro        |  105 |    7 |    9 |    8 |      0.5333 |   0.4706 |     0.5000 |        17 |
| Paura         |  127 |    1 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  118 |    5 |    4 |    2 |      0.2857 |   0.3333 |     0.3077 |         6 |
| Sorpresa      |   98 |   15 |   11 |    5 |      0.2500 |   0.3125 |     0.2778 |        16 |
| Tristezza     |  124 |    3 |    0 |    2 |      0.4000 |   1.0000 |     0.5714 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            4 |                    0 |               0 |              7 |            3 |             0 |            0 |             0 |               1 |                0 |           1 |
| Anticipazione  |             10 |            1 |                    5 |               0 |              2 |            5 |             0 |            0 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            1 |                    1 |               9 |              0 |            0 |             2 |            0 |             5 |               4 |                1 |           2 |
| Fiducia        |             42 |            2 |                    1 |               0 |             32 |            7 |             2 |            0 |             0 |               4 |                0 |           3 |
| Gioia          |             29 |            3 |                    0 |               0 |              4 |           17 |             2 |            0 |             0 |               5 |                1 |           3 |
| Neutro         |             17 |            1 |                    0 |               2 |              4 |            3 |             8 |            0 |             1 |               3 |                0 |           3 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            1 |             0 |            0 |             1 |               1 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               3 |              1 |            0 |             0 |            0 |             2 |               1 |                2 |           0 |
| Sorpresa       |             16 |            1 |                    0 |               1 |              4 |            5 |             2 |            1 |             0 |               5 |                3 |           2 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             0 |               1 |                2 |           0 |

##### Run: `Case 4` | `Blind Text (ClassifierChain)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3000 |   0.5455 |     0.3871 |        11 |
| Anticipazione |      0.2917 |   0.7000 |     0.4118 |        10 |
| Disgusto      |      0.6087 |   0.7368 |     0.6667 |        19 |
| Fiducia       |      0.6222 |   0.6667 |     0.6437 |        42 |
| Gioia         |      0.5833 |   0.7241 |     0.6462 |        29 |
| Neutro        |      0.3214 |   0.5294 |     0.4000 |        17 |
| Paura         |      0.0667 |   1.0000 |     0.1250 |         1 |
| Rabbia        |      0.1667 |   0.1667 |     0.1667 |         6 |
| Sorpresa      |      0.3846 |   0.3125 |     0.3448 |        16 |
| Tristezza     |      0.1818 |   1.0000 |     0.3077 |         2 |
| micro avg     |      0.4253 |   0.6144 |     0.5027 |       153 |
| macro avg     |      0.3527 |   0.6382 |     0.4100 |       153 |
| weighted avg  |      0.4829 |   0.6144 |     0.5286 |       153 |
| samples avg   |      0.4734 |   0.6176 |     0.5093 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  104 |   14 |    5 |    6 |      0.3000 |   0.5455 |     0.3871 |        11 |
| Anticipazione |  102 |   17 |    3 |    7 |      0.2917 |   0.7000 |     0.4118 |        10 |
| Disgusto      |  101 |    9 |    5 |   14 |      0.6087 |   0.7368 |     0.6667 |        19 |
| Fiducia       |   70 |   17 |   14 |   28 |      0.6222 |   0.6667 |     0.6437 |        42 |
| Gioia         |   85 |   15 |    8 |   21 |      0.5833 |   0.7241 |     0.6462 |        29 |
| Neutro        |   93 |   19 |    8 |    9 |      0.3214 |   0.5294 |     0.4000 |        17 |
| Paura         |  114 |   14 |    0 |    1 |      0.0667 |   1.0000 |     0.1250 |         1 |
| Rabbia        |  118 |    5 |    5 |    1 |      0.1667 |   0.1667 |     0.1667 |         6 |
| Sorpresa      |  105 |    8 |   11 |    5 |      0.3846 |   0.3125 |     0.3448 |        16 |
| Tristezza     |  118 |    9 |    0 |    2 |      0.1818 |   1.0000 |     0.3077 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            6 |                    1 |               0 |              8 |            4 |             1 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            0 |                    7 |               0 |              2 |            2 |             1 |            0 |             0 |               0 |                0 |           0 |
| Disgusto       |             19 |            1 |                    1 |              14 |              1 |            2 |             3 |            5 |             2 |               2 |                1 |           0 |
| Fiducia        |             42 |           11 |                   10 |               1 |             28 |           11 |             4 |            2 |             1 |               1 |                2 |           1 |
| Gioia          |             29 |            5 |                    5 |               2 |              7 |           21 |             8 |            2 |             0 |               5 |                3 |           0 |
| Neutro         |             17 |            1 |                    3 |               1 |              2 |            3 |             9 |            1 |             1 |               3 |                2 |           0 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            1 |             0 |            1 |             1 |               0 |                0 |           0 |
| Rabbia         |              6 |            0 |                    1 |               5 |              2 |            0 |             0 |            4 |             1 |               0 |                2 |           0 |
| Sorpresa       |             16 |            1 |                    1 |               3 |              4 |            5 |             4 |            3 |             1 |               5 |                3 |           1 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            2 |             1 |               2 |                2 |           0 |

##### Run: `Case 4` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.4000 |   0.5455 |     0.4615 |        11 |
| Anticipazione |      0.4286 |   0.6000 |     0.5000 |        10 |
| Disgusto      |      0.6111 |   0.5789 |     0.5946 |        19 |
| Fiducia       |      0.6275 |   0.7619 |     0.6882 |        42 |
| Gioia         |      0.6552 |   0.6552 |     0.6552 |        29 |
| Neutro        |      0.3333 |   0.6471 |     0.4400 |        17 |
| Paura         |      0.0833 |   1.0000 |     0.1538 |         1 |
| Rabbia        |      0.3333 |   0.6667 |     0.4444 |         6 |
| Sorpresa      |      0.3125 |   0.3125 |     0.3125 |        16 |
| Tristezza     |      0.1053 |   1.0000 |     0.1905 |         2 |
| micro avg     |      0.4429 |   0.6340 |     0.5215 |       153 |
| macro avg     |      0.3890 |   0.6768 |     0.4441 |       153 |
| weighted avg  |      0.5138 |   0.6340 |     0.5553 |       153 |
| samples avg   |      0.4897 |   0.6395 |     0.5313 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  109 |    9 |    5 |    6 |      0.4000 |   0.5455 |     0.4615 |        11 |
| Anticipazione |  111 |    8 |    4 |    6 |      0.4286 |   0.6000 |     0.5000 |        10 |
| Disgusto      |  103 |    7 |    8 |   11 |      0.6111 |   0.5789 |     0.5946 |        19 |
| Fiducia       |   68 |   19 |   10 |   32 |      0.6275 |   0.7619 |     0.6882 |        42 |
| Gioia         |   90 |   10 |   10 |   19 |      0.6552 |   0.6552 |     0.6552 |        29 |
| Neutro        |   90 |   22 |    6 |   11 |      0.3333 |   0.6471 |     0.4400 |        17 |
| Paura         |  117 |   11 |    0 |    1 |      0.0833 |   1.0000 |     0.1538 |         1 |
| Rabbia        |  115 |    8 |    2 |    4 |      0.3333 |   0.6667 |     0.4444 |         6 |
| Sorpresa      |  102 |   11 |   11 |    5 |      0.3125 |   0.3125 |     0.3125 |        16 |
| Tristezza     |  110 |   17 |    0 |    2 |      0.1053 |   1.0000 |     0.1905 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            6 |                    0 |               0 |              9 |            4 |             1 |            1 |             0 |               0 |                2 |           0 |
| Anticipazione  |             10 |            1 |                    6 |               0 |              2 |            4 |             1 |            1 |             0 |               1 |                0 |           0 |
| Disgusto       |             19 |            0 |                    1 |              11 |              0 |            0 |             4 |            0 |             4 |               2 |                1 |           2 |
| Fiducia        |             42 |            4 |                    3 |               1 |             32 |            7 |             7 |            2 |             1 |               2 |                5 |           1 |
| Gioia          |             29 |            6 |                    2 |               3 |              4 |           19 |             7 |            3 |             0 |               7 |                7 |           0 |
| Neutro         |             17 |            2 |                    2 |               1 |              4 |            3 |            11 |            3 |             2 |               3 |                4 |           0 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             1 |               0 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               4 |              1 |            0 |             0 |            1 |             4 |               0 |                2 |           0 |
| Sorpresa       |             16 |            0 |                    1 |               2 |              4 |            6 |             3 |            2 |             3 |               5 |                5 |           1 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             2 |               1 |                2 |           0 |

#### Case 5: Step 2 Oracle (Train: +rel, Eval: no rel) (Checkpoint: `emotion_textual_corr_with_rel`, Eval Prompt: `Oracle (no rel)`)

| Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Blind Text (ClassifierChain)     | raw       |     0.4438 |     0.5500 |        0.5631 |        0.4186 |
| Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.3986 |     0.5000 |        0.5271 |        0.3333 |
| Blind Text (ClassifierChain)     | std       |     0.4247 |     0.5041 |        0.5483 |        0.3023 |
| Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4548 |     0.5333 |        0.5502 |        0.3411 |

##### Run: `Case 5` | `Blind Text (ClassifierChain)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.4000 |   0.5455 |     0.4615 |        11 |
| Anticipazione |      0.3571 |   0.5000 |     0.4167 |        10 |
| Disgusto      |      0.6500 |   0.6842 |     0.6667 |        19 |
| Fiducia       |      0.7368 |   0.6667 |     0.7000 |        42 |
| Gioia         |      0.6129 |   0.6552 |     0.6333 |        29 |
| Neutro        |      0.4706 |   0.4706 |     0.4706 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.6000 |   0.5000 |     0.5455 |         6 |
| Sorpresa      |      0.2778 |   0.3125 |     0.2941 |        16 |
| Tristezza     |      0.1667 |   0.5000 |     0.2500 |         2 |
| micro avg     |      0.5269 |   0.5752 |     0.5500 |       153 |
| macro avg     |      0.4272 |   0.4835 |     0.4438 |       153 |
| weighted avg  |      0.5583 |   0.5752 |     0.5631 |       153 |
| samples avg   |      0.5426 |   0.5775 |     0.5437 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  109 |    9 |    5 |    6 |      0.4000 |   0.5455 |     0.4615 |        11 |
| Anticipazione |  110 |    9 |    5 |    5 |      0.3571 |   0.5000 |     0.4167 |        10 |
| Disgusto      |  103 |    7 |    6 |   13 |      0.6500 |   0.6842 |     0.6667 |        19 |
| Fiducia       |   77 |   10 |   14 |   28 |      0.7368 |   0.6667 |     0.7000 |        42 |
| Gioia         |   88 |   12 |   10 |   19 |      0.6129 |   0.6552 |     0.6333 |        29 |
| Neutro        |  103 |    9 |    9 |    8 |      0.4706 |   0.4706 |     0.4706 |        17 |
| Paura         |  125 |    3 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  121 |    2 |    3 |    3 |      0.6000 |   0.5000 |     0.5455 |         6 |
| Sorpresa      |  100 |   13 |   11 |    5 |      0.2778 |   0.3125 |     0.2941 |        16 |
| Tristezza     |  122 |    5 |    1 |    1 |      0.1667 |   0.5000 |     0.2500 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            6 |                    1 |               0 |              7 |            3 |             0 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    5 |               0 |              1 |            4 |             0 |            0 |             0 |               1 |                0 |           1 |
| Disgusto       |             19 |            0 |                    1 |              13 |              1 |            1 |             3 |            0 |             3 |               2 |                0 |           1 |
| Fiducia        |             42 |            6 |                    5 |               1 |             28 |           11 |             1 |            0 |             0 |               3 |                2 |           2 |
| Gioia          |             29 |            4 |                    1 |               2 |              4 |           19 |             2 |            0 |             0 |               7 |                2 |           1 |
| Neutro         |             17 |            2 |                    2 |               0 |              1 |            0 |             8 |            0 |             0 |               4 |                1 |           2 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            0 |             0 |            0 |             0 |               0 |                0 |           1 |
| Rabbia         |              6 |            0 |                    0 |               4 |              2 |            0 |             1 |            2 |             3 |               0 |                1 |           0 |
| Sorpresa       |             16 |            0 |                    0 |               2 |              3 |            7 |             3 |            2 |             1 |               5 |                2 |           0 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               1 |                1 |           0 |

##### Run: `Case 5` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2105 |   0.3636 |     0.2667 |        11 |
| Anticipazione |      0.4545 |   0.5000 |     0.4762 |        10 |
| Disgusto      |      0.6190 |   0.6842 |     0.6500 |        19 |
| Fiducia       |      0.7429 |   0.6190 |     0.6753 |        42 |
| Gioia         |      0.6667 |   0.6207 |     0.6429 |        29 |
| Neutro        |      0.3182 |   0.4118 |     0.3590 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.4286 |   0.5000 |     0.4615 |         6 |
| Sorpresa      |      0.2143 |   0.3750 |     0.2727 |        16 |
| Tristezza     |      0.1111 |   0.5000 |     0.1818 |         2 |
| micro avg     |      0.4637 |   0.5425 |     0.5000 |       153 |
| macro avg     |      0.3766 |   0.4574 |     0.3986 |       153 |
| weighted avg  |      0.5280 |   0.5425 |     0.5271 |       153 |
| samples avg   |      0.5013 |   0.5491 |     0.5034 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  103 |   15 |    7 |    4 |      0.2105 |   0.3636 |     0.2667 |        11 |
| Anticipazione |  113 |    6 |    5 |    5 |      0.4545 |   0.5000 |     0.4762 |        10 |
| Disgusto      |  102 |    8 |    6 |   13 |      0.6190 |   0.6842 |     0.6500 |        19 |
| Fiducia       |   78 |    9 |   16 |   26 |      0.7429 |   0.6190 |     0.6753 |        42 |
| Gioia         |   91 |    9 |   11 |   18 |      0.6667 |   0.6207 |     0.6429 |        29 |
| Neutro        |   97 |   15 |   10 |    7 |      0.3182 |   0.4118 |     0.3590 |        17 |
| Paura         |  128 |    0 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  119 |    4 |    3 |    3 |      0.4286 |   0.5000 |     0.4615 |         6 |
| Sorpresa      |   91 |   22 |   10 |    6 |      0.2143 |   0.3750 |     0.2727 |        16 |
| Tristezza     |  119 |    8 |    1 |    1 |      0.1111 |   0.5000 |     0.1818 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            4 |                    0 |               0 |              4 |            3 |             3 |            0 |             0 |               0 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    5 |               0 |              1 |            3 |             1 |            0 |             0 |               1 |                0 |           1 |
| Disgusto       |             19 |            1 |                    1 |              13 |              0 |            0 |             3 |            0 |             2 |               8 |                1 |           1 |
| Fiducia        |             42 |            6 |                    2 |               0 |             26 |            6 |             4 |            0 |             0 |               5 |                2 |           3 |
| Gioia          |             29 |            5 |                    1 |               2 |              4 |           18 |             2 |            0 |             1 |               9 |                4 |           2 |
| Neutro         |             17 |            2 |                    2 |               2 |              2 |            3 |             7 |            0 |             1 |               5 |                0 |           1 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            0 |             1 |               0 |                1 |           0 |
| Rabbia         |              6 |            1 |                    0 |               3 |              0 |            0 |             2 |            0 |             3 |               1 |                2 |           0 |
| Sorpresa       |             16 |            1 |                    0 |               2 |              3 |            6 |             2 |            0 |             1 |               6 |                2 |           0 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            0 |             1 |               1 |                1 |           0 |

##### Run: `Case 5` | `Blind Text (ClassifierChain)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3478 |   0.7273 |     0.4706 |        11 |
| Anticipazione |      0.3750 |   0.6000 |     0.4615 |        10 |
| Disgusto      |      0.5417 |   0.6842 |     0.6047 |        19 |
| Fiducia       |      0.7568 |   0.6667 |     0.7089 |        42 |
| Gioia         |      0.7273 |   0.5517 |     0.6275 |        29 |
| Neutro        |      0.2941 |   0.5882 |     0.3922 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.4286 |   0.5000 |     0.4615 |         6 |
| Sorpresa      |      0.2258 |   0.4375 |     0.2979 |        16 |
| Tristezza     |      0.1429 |   0.5000 |     0.2222 |         2 |
| micro avg     |      0.4340 |   0.6013 |     0.5041 |       153 |
| macro avg     |      0.3840 |   0.5256 |     0.4247 |       153 |
| weighted avg  |      0.5373 |   0.6013 |     0.5483 |       153 |
| samples avg   |      0.4767 |   0.6059 |     0.5124 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  103 |   15 |    3 |    8 |      0.3478 |   0.7273 |     0.4706 |        11 |
| Anticipazione |  109 |   10 |    4 |    6 |      0.3750 |   0.6000 |     0.4615 |        10 |
| Disgusto      |   99 |   11 |    6 |   13 |      0.5417 |   0.6842 |     0.6047 |        19 |
| Fiducia       |   78 |    9 |   14 |   28 |      0.7568 |   0.6667 |     0.7089 |        42 |
| Gioia         |   94 |    6 |   13 |   16 |      0.7273 |   0.5517 |     0.6275 |        29 |
| Neutro        |   88 |   24 |    7 |   10 |      0.2941 |   0.5882 |     0.3922 |        17 |
| Paura         |  117 |   11 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  119 |    4 |    3 |    3 |      0.4286 |   0.5000 |     0.4615 |         6 |
| Sorpresa      |   89 |   24 |    9 |    7 |      0.2258 |   0.4375 |     0.2979 |        16 |
| Tristezza     |  121 |    6 |    1 |    1 |      0.1429 |   0.5000 |     0.2222 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            8 |                    1 |               0 |              5 |            4 |             5 |            2 |             0 |               1 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    6 |               0 |              1 |            1 |             1 |            0 |             0 |               2 |                0 |           2 |
| Disgusto       |             19 |            1 |                    1 |              13 |              1 |            1 |             3 |            2 |             5 |               4 |                0 |           0 |
| Fiducia        |             42 |            7 |                    6 |               3 |             28 |            8 |             6 |            3 |             0 |               8 |                2 |           0 |
| Gioia          |             29 |            7 |                    1 |               2 |              2 |           16 |             7 |            4 |             0 |              13 |                2 |           1 |
| Neutro         |             17 |            2 |                    3 |               2 |              1 |            0 |            10 |            0 |             0 |               5 |                2 |           0 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            0 |             0 |            0 |             0 |               0 |                0 |           1 |
| Rabbia         |              6 |            0 |                    0 |               2 |              2 |            0 |             1 |            3 |             3 |               0 |                1 |           0 |
| Sorpresa       |             16 |            2 |                    0 |               3 |              3 |            5 |             6 |            3 |             1 |               7 |                2 |           0 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               1 |                1 |           0 |

##### Run: `Case 5` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.3182 |   0.6364 |     0.4242 |        11 |
| Anticipazione |      0.3333 |   0.5000 |     0.4000 |        10 |
| Disgusto      |      0.5652 |   0.6842 |     0.6190 |        19 |
| Fiducia       |      0.6200 |   0.7381 |     0.6739 |        42 |
| Gioia         |      0.6061 |   0.6897 |     0.6452 |        29 |
| Neutro        |      0.4444 |   0.4706 |     0.4571 |        17 |
| Paura         |      0.1111 |   1.0000 |     0.2000 |         1 |
| Rabbia        |      0.4286 |   0.5000 |     0.4615 |         6 |
| Sorpresa      |      0.3000 |   0.3750 |     0.3333 |        16 |
| Tristezza     |      0.2000 |   1.0000 |     0.3333 |         2 |
| micro avg     |      0.4638 |   0.6275 |     0.5333 |       153 |
| macro avg     |      0.3927 |   0.6594 |     0.4548 |       153 |
| weighted avg  |      0.5008 |   0.6275 |     0.5502 |       153 |
| samples avg   |      0.5052 |   0.6189 |     0.5357 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  103 |   15 |    4 |    7 |      0.3182 |   0.6364 |     0.4242 |        11 |
| Anticipazione |  109 |   10 |    5 |    5 |      0.3333 |   0.5000 |     0.4000 |        10 |
| Disgusto      |  100 |   10 |    6 |   13 |      0.5652 |   0.6842 |     0.6190 |        19 |
| Fiducia       |   68 |   19 |   11 |   31 |      0.6200 |   0.7381 |     0.6739 |        42 |
| Gioia         |   87 |   13 |    9 |   20 |      0.6061 |   0.6897 |     0.6452 |        29 |
| Neutro        |  102 |   10 |    9 |    8 |      0.4444 |   0.4706 |     0.4571 |        17 |
| Paura         |  120 |    8 |    0 |    1 |      0.1111 |   1.0000 |     0.2000 |         1 |
| Rabbia        |  119 |    4 |    3 |    3 |      0.4286 |   0.5000 |     0.4615 |         6 |
| Sorpresa      |   99 |   14 |   10 |    6 |      0.3000 |   0.3750 |     0.3333 |        16 |
| Tristezza     |  119 |    8 |    0 |    2 |      0.2000 |   1.0000 |     0.3333 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            7 |                    1 |               0 |              9 |            3 |             0 |            0 |             1 |               0 |                0 |           0 |
| Anticipazione  |             10 |            2 |                    5 |               0 |              1 |            3 |             0 |            1 |             0 |               0 |                0 |           2 |
| Disgusto       |             19 |            0 |                    1 |              13 |              1 |            1 |             3 |            0 |             4 |               4 |                1 |           0 |
| Fiducia        |             42 |            8 |                    4 |               2 |             31 |           10 |             3 |            2 |             1 |               3 |                2 |           2 |
| Gioia          |             29 |            8 |                    1 |               1 |             10 |           20 |             1 |            2 |             0 |               8 |                2 |           1 |
| Neutro         |             17 |            2 |                    3 |               2 |              4 |            3 |             8 |            2 |             1 |               3 |                2 |           1 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             0 |               0 |                1 |           0 |
| Rabbia         |              6 |            0 |                    0 |               3 |              1 |            0 |             2 |            1 |             3 |               0 |                1 |           0 |
| Sorpresa       |             16 |            0 |                    1 |               3 |              3 |            7 |             2 |            2 |             0 |               6 |                4 |           0 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             0 |               1 |                2 |           0 |

#### Case 6: Step 2 Oracle (Train: +rel, Eval: +rel) (Checkpoint: `emotion_textual_corr_with_rel`, Eval Prompt: `Oracle (+ rel)`)

| Model                            | Scaling   |   Macro F1 |   Micro F1 |   Weighted F1 |   Subset Acc. |
|:---------------------------------|:----------|-----------:|-----------:|--------------:|--------------:|
| Blind Text (ClassifierChain)     | raw       |     0.4207 |     0.5029 |        0.5196 |        0.3333 |
| Dual-Encoder (FEEL-IT + ViT-224) | raw       |     0.4306 |     0.5333 |        0.5389 |        0.3333 |
| Blind Text (ClassifierChain)     | std       |     0.4504 |     0.5066 |        0.5515 |        0.3178 |
| Dual-Encoder (FEEL-IT + ViT-224) | std       |     0.4337 |     0.5000 |        0.5336 |        0.2868 |

##### Run: `Case 6` | `Blind Text (ClassifierChain)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2400 |   0.5455 |     0.3333 |        11 |
| Anticipazione |      0.5000 |   0.6000 |     0.5455 |        10 |
| Disgusto      |      0.4828 |   0.7368 |     0.5833 |        19 |
| Fiducia       |      0.6750 |   0.6429 |     0.6585 |        42 |
| Gioia         |      0.5862 |   0.5862 |     0.5862 |        29 |
| Neutro        |      0.4286 |   0.3529 |     0.3871 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.2727 |   0.5000 |     0.3529 |         6 |
| Sorpresa      |      0.2727 |   0.3750 |     0.3158 |        16 |
| Tristezza     |      0.2857 |   1.0000 |     0.4444 |         2 |
| micro avg     |      0.4508 |   0.5686 |     0.5029 |       153 |
| macro avg     |      0.3744 |   0.5339 |     0.4207 |       153 |
| weighted avg  |      0.4969 |   0.5686 |     0.5196 |       153 |
| samples avg   |      0.4916 |   0.5736 |     0.5070 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |   99 |   19 |    5 |    6 |      0.2400 |   0.5455 |     0.3333 |        11 |
| Anticipazione |  113 |    6 |    4 |    6 |      0.5000 |   0.6000 |     0.5455 |        10 |
| Disgusto      |   95 |   15 |    5 |   14 |      0.4828 |   0.7368 |     0.5833 |        19 |
| Fiducia       |   74 |   13 |   15 |   27 |      0.6750 |   0.6429 |     0.6585 |        42 |
| Gioia         |   88 |   12 |   12 |   17 |      0.5862 |   0.5862 |     0.5862 |        29 |
| Neutro        |  104 |    8 |   11 |    6 |      0.4286 |   0.3529 |     0.3871 |        17 |
| Paura         |  124 |    4 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  115 |    8 |    3 |    3 |      0.2727 |   0.5000 |     0.3529 |         6 |
| Sorpresa      |   97 |   16 |   10 |    6 |      0.2727 |   0.3750 |     0.3158 |        16 |
| Tristezza     |  122 |    5 |    0 |    2 |      0.2857 |   1.0000 |     0.4444 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            6 |                    0 |               0 |              6 |            3 |             0 |            0 |             0 |               1 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    6 |               0 |              2 |            2 |             0 |            0 |             0 |               1 |                0 |           0 |
| Disgusto       |             19 |            0 |                    1 |              14 |              1 |            0 |             3 |            1 |             8 |               2 |                0 |           0 |
| Fiducia        |             42 |           10 |                    3 |               3 |             27 |           10 |             0 |            0 |             0 |               6 |                2 |           1 |
| Gioia          |             29 |            7 |                    1 |               2 |              3 |           17 |             3 |            0 |             1 |              10 |                2 |           1 |
| Neutro         |             17 |            2 |                    2 |               3 |              2 |            2 |             6 |            0 |             0 |               3 |                1 |           3 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            0 |             0 |            0 |             0 |               0 |                0 |           1 |
| Rabbia         |              6 |            0 |                    0 |               4 |              2 |            0 |             0 |            3 |             3 |               0 |                1 |           0 |
| Sorpresa       |             16 |            2 |                    0 |               5 |              4 |            6 |             3 |            2 |             1 |               6 |                3 |           0 |
| Tristezza      |              2 |            0 |                    0 |               2 |              0 |            0 |             0 |            1 |             1 |               1 |                2 |           0 |

##### Run: `Case 6` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `raw`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2857 |   0.1818 |     0.2222 |        11 |
| Anticipazione |      0.4286 |   0.6000 |     0.5000 |        10 |
| Disgusto      |      0.6818 |   0.7895 |     0.7317 |        19 |
| Fiducia       |      0.5849 |   0.7381 |     0.6526 |        42 |
| Gioia         |      0.6429 |   0.6207 |     0.6316 |        29 |
| Neutro        |      0.2432 |   0.5294 |     0.3333 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.6667 |   0.3333 |     0.4444 |         6 |
| Sorpresa      |      0.3200 |   0.5000 |     0.3902 |        16 |
| Tristezza     |      0.3333 |   0.5000 |     0.4000 |         2 |
| micro avg     |      0.4792 |   0.6013 |     0.5333 |       153 |
| macro avg     |      0.4187 |   0.4793 |     0.4306 |       153 |
| weighted avg  |      0.5066 |   0.6013 |     0.5389 |       153 |
| samples avg   |      0.5168 |   0.5969 |     0.5364 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  113 |    5 |    9 |    2 |      0.2857 |   0.1818 |     0.2222 |        11 |
| Anticipazione |  111 |    8 |    4 |    6 |      0.4286 |   0.6000 |     0.5000 |        10 |
| Disgusto      |  103 |    7 |    4 |   15 |      0.6818 |   0.7895 |     0.7317 |        19 |
| Fiducia       |   65 |   22 |   11 |   31 |      0.5849 |   0.7381 |     0.6526 |        42 |
| Gioia         |   90 |   10 |   11 |   18 |      0.6429 |   0.6207 |     0.6316 |        29 |
| Neutro        |   84 |   28 |    8 |    9 |      0.2432 |   0.5294 |     0.3333 |        17 |
| Paura         |  128 |    0 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  122 |    1 |    4 |    2 |      0.6667 |   0.3333 |     0.4444 |         6 |
| Sorpresa      |   96 |   17 |    8 |    8 |      0.3200 |   0.5000 |     0.3902 |        16 |
| Tristezza     |  125 |    2 |    1 |    1 |      0.3333 |   0.5000 |     0.4000 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            2 |                    0 |               0 |             10 |            3 |             5 |            0 |             0 |               1 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    6 |               0 |              2 |            2 |             2 |            0 |             0 |               1 |                0 |           0 |
| Disgusto       |             19 |            0 |                    1 |              15 |              2 |            0 |             4 |            0 |             2 |               4 |                0 |           0 |
| Fiducia        |             42 |            2 |                    4 |               0 |             31 |            8 |            11 |            0 |             0 |               4 |                0 |           3 |
| Gioia          |             29 |            4 |                    1 |               1 |              9 |           18 |             4 |            0 |             0 |              11 |                1 |           0 |
| Neutro         |             17 |            0 |                    3 |               1 |              5 |            3 |             9 |            0 |             0 |               3 |                0 |           1 |
| Paura          |              1 |            0 |                    0 |               1 |              0 |            0 |             0 |            0 |             1 |               1 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               4 |              1 |            0 |             1 |            0 |             2 |               1 |                1 |           0 |
| Sorpresa       |             16 |            0 |                    0 |               2 |              5 |            7 |             3 |            0 |             0 |               8 |                2 |           0 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            0 |             0 |               1 |                1 |           0 |

##### Run: `Case 6` | `Blind Text (ClassifierChain)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2273 |   0.4545 |     0.3030 |        11 |
| Anticipazione |      0.3750 |   0.6000 |     0.4615 |        10 |
| Disgusto      |      0.7000 |   0.7368 |     0.7179 |        19 |
| Fiducia       |      0.5957 |   0.6667 |     0.6292 |        42 |
| Gioia         |      0.6061 |   0.6897 |     0.6452 |        29 |
| Neutro        |      0.4231 |   0.6471 |     0.5116 |        17 |
| Paura         |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |      0.5000 |   0.6667 |     0.5714 |         6 |
| Sorpresa      |      0.2500 |   0.3750 |     0.3000 |        16 |
| Tristezza     |      0.2222 |   1.0000 |     0.3636 |         2 |
| micro avg     |      0.4248 |   0.6275 |     0.5066 |       153 |
| macro avg     |      0.3899 |   0.5836 |     0.4504 |       153 |
| weighted avg  |      0.5019 |   0.6275 |     0.5515 |       153 |
| samples avg   |      0.4809 |   0.6318 |     0.5194 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  101 |   17 |    6 |    5 |      0.2273 |   0.4545 |     0.3030 |        11 |
| Anticipazione |  109 |   10 |    4 |    6 |      0.3750 |   0.6000 |     0.4615 |        10 |
| Disgusto      |  104 |    6 |    5 |   14 |      0.7000 |   0.7368 |     0.7179 |        19 |
| Fiducia       |   68 |   19 |   14 |   28 |      0.5957 |   0.6667 |     0.6292 |        42 |
| Gioia         |   87 |   13 |    9 |   20 |      0.6061 |   0.6897 |     0.6452 |        29 |
| Neutro        |   97 |   15 |    6 |   11 |      0.4231 |   0.6471 |     0.5116 |        17 |
| Paura         |  107 |   21 |    1 |    0 |      0.0000 |   0.0000 |     0.0000 |         1 |
| Rabbia        |  119 |    4 |    2 |    4 |      0.5000 |   0.6667 |     0.5714 |         6 |
| Sorpresa      |   95 |   18 |   10 |    6 |      0.2500 |   0.3750 |     0.3000 |        16 |
| Tristezza     |  120 |    7 |    0 |    2 |      0.2222 |   1.0000 |     0.3636 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            5 |                    1 |               0 |              9 |            4 |             1 |            2 |             0 |               1 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    6 |               0 |              2 |            3 |             0 |            0 |             0 |               2 |                0 |           0 |
| Disgusto       |             19 |            4 |                    1 |              14 |              1 |            0 |             4 |            6 |             5 |               3 |                1 |           0 |
| Fiducia        |             42 |            7 |                    6 |               2 |             28 |           11 |             3 |            5 |             0 |               6 |                2 |           0 |
| Gioia          |             29 |            6 |                    1 |               1 |              6 |           20 |             4 |            5 |             0 |              10 |                2 |           2 |
| Neutro         |             17 |            2 |                    3 |               0 |              3 |            2 |            11 |            2 |             0 |               4 |                2 |           0 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            0 |             0 |            0 |             0 |               0 |                0 |           1 |
| Rabbia         |              6 |            0 |                    0 |               4 |              1 |            0 |             1 |            4 |             4 |               0 |                2 |           0 |
| Sorpresa       |             16 |            0 |                    0 |               1 |              3 |            6 |             4 |            4 |             2 |               6 |                3 |           0 |
| Tristezza      |              2 |            0 |                    0 |               1 |              0 |            0 |             0 |            1 |             1 |               1 |                2 |           0 |

##### Run: `Case 6` | `Dual-Encoder (FEEL-IT + ViT-224)` | Scaling: `std`
**Classification Report**:

| Label         |   Precision |   Recall |   F1-Score |   Support |
|:--------------|------------:|---------:|-----------:|----------:|
| Amore         |      0.2000 |   0.3636 |     0.2581 |        11 |
| Anticipazione |      0.3750 |   0.6000 |     0.4615 |        10 |
| Disgusto      |      0.6842 |   0.6842 |     0.6842 |        19 |
| Fiducia       |      0.5882 |   0.7143 |     0.6452 |        42 |
| Gioia         |      0.5556 |   0.6897 |     0.6154 |        29 |
| Neutro        |      0.3750 |   0.5294 |     0.4390 |        17 |
| Paura         |      0.0476 |   1.0000 |     0.0909 |         1 |
| Rabbia        |      0.3333 |   0.5000 |     0.4000 |         6 |
| Sorpresa      |      0.3158 |   0.3750 |     0.3429 |        16 |
| Tristezza     |      0.2500 |   1.0000 |     0.4000 |         2 |
| micro avg     |      0.4215 |   0.6144 |     0.5000 |       153 |
| macro avg     |      0.3725 |   0.6456 |     0.4337 |       153 |
| weighted avg  |      0.4820 |   0.6144 |     0.5336 |       153 |
| samples avg   |      0.4774 |   0.6189 |     0.5132 |       153 |

**Per-Label 2x2 Confusion Matrices (`TN, FP, FN, TP`)**:

| Label         |   TN |   FP |   FN |   TP |   Precision |   Recall |   F1-Score |   Support |
|:--------------|-----:|-----:|-----:|-----:|------------:|---------:|-----------:|----------:|
| Amore         |  102 |   16 |    7 |    4 |      0.2000 |   0.3636 |     0.2581 |        11 |
| Anticipazione |  109 |   10 |    4 |    6 |      0.3750 |   0.6000 |     0.4615 |        10 |
| Disgusto      |  104 |    6 |    6 |   13 |      0.6842 |   0.6842 |     0.6842 |        19 |
| Fiducia       |   66 |   21 |   12 |   30 |      0.5882 |   0.7143 |     0.6452 |        42 |
| Gioia         |   84 |   16 |    9 |   20 |      0.5556 |   0.6897 |     0.6154 |        29 |
| Neutro        |   97 |   15 |    8 |    9 |      0.3750 |   0.5294 |     0.4390 |        17 |
| Paura         |  108 |   20 |    0 |    1 |      0.0476 |   1.0000 |     0.0909 |         1 |
| Rabbia        |  117 |    6 |    3 |    3 |      0.3333 |   0.5000 |     0.4000 |         6 |
| Sorpresa      |  100 |   13 |   10 |    6 |      0.3158 |   0.3750 |     0.3429 |        16 |
| Tristezza     |  121 |    6 |    0 |    2 |      0.2500 |   1.0000 |     0.4000 |         2 |

**10x10 Gold vs. Predicted Emotion Co-occurrence Matrix**:

| Gold_Emotion   |   Gold_Support |   Pred_Amore |   Pred_Anticipazione |   Pred_Disgusto |   Pred_Fiducia |   Pred_Gioia |   Pred_Neutro |   Pred_Paura |   Pred_Rabbia |   Pred_Sorpresa |   Pred_Tristezza |   Pred_None |
|:---------------|---------------:|-------------:|---------------------:|----------------:|---------------:|-------------:|--------------:|-------------:|--------------:|----------------:|-----------------:|------------:|
| Amore          |             11 |            4 |                    0 |               0 |              8 |            6 |             0 |            1 |             1 |               1 |                0 |           0 |
| Anticipazione  |             10 |            1 |                    6 |               0 |              2 |            2 |             1 |            2 |             0 |               1 |                0 |           0 |
| Disgusto       |             19 |            1 |                    1 |              13 |              1 |            0 |             4 |            3 |             5 |               2 |                0 |           0 |
| Fiducia        |             42 |            3 |                    4 |               2 |             30 |           14 |             3 |            3 |             2 |               6 |                2 |           0 |
| Gioia          |             29 |            8 |                    2 |               1 |              7 |           20 |             4 |            5 |             0 |               9 |                2 |           1 |
| Neutro         |             17 |            4 |                    4 |               0 |              5 |            2 |             9 |            3 |             0 |               1 |                2 |           0 |
| Paura          |              1 |            0 |                    0 |               0 |              0 |            0 |             0 |            1 |             1 |               0 |                0 |           0 |
| Rabbia         |              6 |            0 |                    0 |               4 |              1 |            0 |             1 |            2 |             3 |               0 |                1 |           0 |
| Sorpresa       |             16 |            1 |                    0 |               1 |              4 |            7 |             3 |            5 |             0 |               6 |                3 |           0 |
| Tristezza      |              2 |            0 |                    0 |               0 |              0 |            0 |             0 |            2 |             0 |               1 |                2 |           0 |

---

## 2. Task 2: Irony & Sarcasm Detection (3-Class Single-Label, `UmBERTo` + `ViT-224`)

- **Text Encoder**: `Musixmatch/umberto-commoncrawl-cased-v1` (`UmBERTo`, 768-d `[CLS]`)
- **Vision Encoder**: `google/vit-base-patch16-224` (`ViT-Base-224`, frozen 768-d `[CLS]`)
- **Multimodal Head (`mm_test`, $N=129$)**: `SGDClassifier(loss='squared_hinge', penalty='l2', class_weight='balanced', random_state=42)`

### 2.1 Textual Irony Benchmark (`me_test`, $N=405$)

| Regime                        | Checkpoint                  | Prompt                               |   N Test |   Macro F1 |   Micro F1 (= Acc) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:------------------------------|:----------------------------|:-------------------------------------|---------:|-----------:|-------------------:|--------------:|----------:|------------:|--------------:|
| Step 1 Base (Raw)             | irony_textual_no_corr       | Raw                                  |      405 |     0.4805 |             0.7383 |        0.7337 |    0.8460 |      0.3289 |        0.2667 |
| Step 2 Raw (Ablation)         | irony_textual_step2_raw     | Raw                                  |      405 |     0.5404 |             0.7432 |        0.7475 |    0.8389 |      0.4186 |        0.3636 |
| Step 2 Oracle (Train: no rel) | irony_textual_corr          | Oracle (Train: no rel, Eval: no rel) |      405 |     0.5418 |             0.7901 |        0.7790 |    0.8767 |      0.4328 |        0.3158 |
| Step 2 Oracle (Train: +rel)   | irony_textual_corr_with_rel | Oracle (Train: +rel, Eval: no rel)   |      405 |     0.5329 |             0.7901 |        0.7826 |    0.8769 |      0.4552 |        0.2667 |

#### Textual Case: `Step 1 Base (Raw)` (`irony_textual_no_corr`)
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8369 |   0.8553 |     0.8460 |       318 |
| Ironia       |      0.3425 |   0.3165 |     0.3289 |        79 |
| Sarcasmo     |      0.2857 |   0.2500 |     0.2667 |         8 |
| macro avg    |      0.4884 |   0.4739 |     0.4805 |       405 |
| weighted avg |      0.7296 |   0.7383 |     0.7337 |       405 |
| accuracy     |      0.7383 |   0.7383 |     0.7383 |       405 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |         272 |            44 |               2 |          318 |
| Ironia       |          51 |            25 |               3 |           79 |
| Sarcasmo     |           2 |             4 |               2 |            8 |

#### Textual Case: `Step 2 Raw (Ablation)` (`irony_textual_step2_raw`)
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8511 |   0.8270 |     0.8389 |       318 |
| Ironia       |      0.3871 |   0.4557 |     0.4186 |        79 |
| Sarcasmo     |      0.6667 |   0.2500 |     0.3636 |         8 |
| macro avg    |      0.6350 |   0.5109 |     0.5404 |       405 |
| weighted avg |      0.7570 |   0.7432 |     0.7475 |       405 |
| accuracy     |      0.7432 |   0.7432 |     0.7432 |       405 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |         263 |            54 |               1 |          318 |
| Ironia       |          43 |            36 |               0 |           79 |
| Sarcasmo     |           3 |             3 |               2 |            8 |

#### Textual Case: `Step 2 Oracle (Train: no rel)` (`irony_textual_corr`)
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8496 |   0.9057 |     0.8767 |       318 |
| Ironia       |      0.5273 |   0.3671 |     0.4328 |        79 |
| Sarcasmo     |      0.2727 |   0.3750 |     0.3158 |         8 |
| macro avg    |      0.5499 |   0.5492 |     0.5418 |       405 |
| weighted avg |      0.7753 |   0.7901 |     0.7790 |       405 |
| accuracy     |      0.7901 |   0.7901 |     0.7901 |       405 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |         288 |            24 |               6 |          318 |
| Ironia       |          48 |            29 |               2 |           79 |
| Sarcasmo     |           3 |             2 |               3 |            8 |

#### Textual Case: `Step 2 Oracle (Train: +rel)` (`irony_textual_corr_with_rel`)
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8584 |   0.8962 |     0.8769 |       318 |
| Ironia       |      0.5000 |   0.4177 |     0.4552 |        79 |
| Sarcasmo     |      0.2857 |   0.2500 |     0.2667 |         8 |
| macro avg    |      0.5480 |   0.5213 |     0.5329 |       405 |
| weighted avg |      0.7772 |   0.7901 |     0.7826 |       405 |
| accuracy     |      0.7901 |   0.7901 |     0.7901 |       405 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |         285 |            29 |               4 |          318 |
| Ironia       |          45 |            33 |               1 |           79 |
| Sarcasmo     |           2 |             4 |               2 |            8 |

---

### 2.2 Multimodal Irony Benchmark (`mm_test`, $N=129$ — All 6 Cases $\times$ `raw` vs `std`)

#### Master Comparison Table (All Multimodal Irony Runs)
| Case   | Regime                                      | Train Relation   | Eval Relation   | Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:-------|:--------------------------------------------|:-----------------|:----------------|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Blind Text (SGD, raw)                 | raw       |     0.4001 |              0.7209 |        0.7345 |    0.8627 |      0.2424 |        0.0952 |
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.4671 |              0.8140 |        0.7965 |    0.9014 |      0.5000 |        0.0000 |
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Blind Text (SGD, std)                 | std       |     0.4280 |              0.7364 |        0.7516 |    0.8586 |      0.4255 |        0.0000 |
| Case 1 | Step 1 Base (No corr)                       | None (Raw)       | None (Raw)      | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.5164 |              0.8062 |        0.7938 |    0.8920 |      0.4571 |        0.2000 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Blind Text (SGD, raw)                 | raw       |     0.4023 |              0.7209 |        0.7338 |    0.8800 |      0.0769 |        0.2500 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5561 |              0.8372 |        0.8274 |    0.9151 |      0.5714 |        0.1818 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Blind Text (SGD, std)                 | std       |     0.4631 |              0.7519 |        0.7659 |    0.8911 |      0.2400 |        0.2581 |
| Case 2 | Step 2 Raw (No corr)                        | None (Raw)       | None (Raw)      | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.4540 |              0.8062 |        0.7904 |    0.9005 |      0.4615 |        0.0000 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Blind Text (SGD, raw)                 | raw       |     0.3954 |              0.7287 |        0.7342 |    0.8529 |      0.3333 |        0.0000 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.4069 |              0.7597 |        0.7509 |    0.8708 |      0.3500 |        0.0000 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Blind Text (SGD, std)                 | std       |     0.3956 |              0.7132 |        0.7242 |    0.8529 |      0.2162 |        0.1176 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | no rel           | no rel          | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.3810 |              0.7287 |        0.7263 |    0.8502 |      0.2927 |        0.0000 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Blind Text (SGD, raw)                 | raw       |     0.3896 |              0.7209 |        0.7252 |    0.8431 |      0.3256 |        0.0000 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.4414 |              0.7752 |        0.7645 |    0.8696 |      0.4545 |        0.0000 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Blind Text (SGD, std)                 | std       |     0.3299 |              0.6744 |        0.6953 |    0.8358 |      0.1538 |        0.0000 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | no rel           | +rel            | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.3791 |              0.7209 |        0.7218 |    0.8447 |      0.2927 |        0.0000 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Blind Text (SGD, raw)                 | raw       |     0.4486 |              0.7442 |        0.7393 |    0.8667 |      0.1714 |        0.3077 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.4157 |              0.7674 |        0.7573 |    0.8750 |      0.3721 |        0.0000 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Blind Text (SGD, std)                 | std       |     0.4088 |              0.7364 |        0.7388 |    0.8515 |      0.3750 |        0.0000 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | +rel             | no rel          | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.3783 |              0.7364 |        0.7289 |    0.8558 |      0.2791 |        0.0000 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Blind Text (SGD, raw)                 | raw       |     0.3942 |              0.7519 |        0.7484 |    0.8750 |      0.3077 |        0.0000 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.4323 |              0.7907 |        0.7722 |    0.8868 |      0.4103 |        0.0000 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Blind Text (SGD, std)                 | std       |     0.4308 |              0.7597 |        0.7583 |    0.8670 |      0.4255 |        0.0000 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | +rel             | +rel            | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.4139 |              0.7597 |        0.7530 |    0.8696 |      0.3721 |        0.0000 |

#### Semiotic Subsets Breakdown (`ADD`, `PAR`, `DIV` Accuracy / Micro F1 on `mm_test`, $N=129$)
| Case   | Regime                                      | Model                                 | Scaling   |   ADD (N=68) |   PAR (N=23) |   DIV (N=38) |   Macro F1 |   Micro F1 (= Acc.) |
|:-------|:--------------------------------------------|:--------------------------------------|:----------|-------------:|-------------:|-------------:|-----------:|--------------------:|
| Case 1 | Step 1 Base (No corr)                       | Blind Text (SGD, raw)                 | raw       |       0.7941 |       1.0000 |       0.4211 |     0.4001 |              0.7209 |
| Case 1 | Step 1 Base (No corr)                       | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.8676 |       0.9565 |       0.6316 |     0.4671 |              0.8140 |
| Case 1 | Step 1 Base (No corr)                       | Blind Text (SGD, std)                 | std       |       0.7500 |       0.9130 |       0.6053 |     0.4280 |              0.7364 |
| Case 1 | Step 1 Base (No corr)                       | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.8676 |       0.9565 |       0.6053 |     0.5164 |              0.8062 |
| Case 2 | Step 2 Raw (No corr)                        | Blind Text (SGD, raw)                 | raw       |       0.8088 |       0.9565 |       0.4211 |     0.4023 |              0.7209 |
| Case 2 | Step 2 Raw (No corr)                        | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.8824 |       0.9565 |       0.6842 |     0.5561 |              0.8372 |
| Case 2 | Step 2 Raw (No corr)                        | Blind Text (SGD, std)                 | std       |       0.8382 |       0.9565 |       0.4737 |     0.4631 |              0.7519 |
| Case 2 | Step 2 Raw (No corr)                        | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.8529 |       0.9565 |       0.6316 |     0.4540 |              0.8062 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (SGD, raw)                 | raw       |       0.8235 |       0.9130 |       0.4474 |     0.3954 |              0.7287 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.8529 |       0.9130 |       0.5000 |     0.4069 |              0.7597 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (SGD, std)                 | std       |       0.8382 |       0.9130 |       0.3684 |     0.3956 |              0.7132 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.8382 |       0.9130 |       0.4211 |     0.3810 |              0.7287 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Blind Text (SGD, raw)                 | raw       |       0.8088 |       0.9130 |       0.4474 |     0.3896 |              0.7209 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.8235 |       0.9130 |       0.6053 |     0.4414 |              0.7752 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Blind Text (SGD, std)                 | std       |       0.7794 |       0.9130 |       0.3421 |     0.3299 |              0.6744 |
| Case 4 | Step 2 Oracle (Train: no rel, Eval: +rel)   | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.7941 |       0.9130 |       0.4737 |     0.3791 |              0.7209 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Blind Text (SGD, raw)                 | raw       |       0.8676 |       0.9130 |       0.4211 |     0.4486 |              0.7442 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.8529 |       0.9130 |       0.5263 |     0.4157 |              0.7674 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Blind Text (SGD, std)                 | std       |       0.8088 |       0.8261 |       0.5526 |     0.4088 |              0.7364 |
| Case 5 | Step 2 Oracle (Train: +rel, Eval: no rel)   | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.8235 |       0.9130 |       0.4737 |     0.3783 |              0.7364 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Blind Text (SGD, raw)                 | raw       |       0.8824 |       0.9130 |       0.4211 |     0.3942 |              0.7519 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.8824 |       0.9565 |       0.5263 |     0.4323 |              0.7907 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Blind Text (SGD, std)                 | std       |       0.8529 |       0.9130 |       0.5000 |     0.4308 |              0.7597 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel)     | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.8676 |       0.9130 |       0.4737 |     0.4139 |              0.7597 |

### 2.3 Multimodal Irony Detailed Reports by Case (`Case 1` – `Case 6`)

#### Case 1: Step 1 Base (No corr) (Checkpoint: `irony_textual_no_corr`, Eval Prompt: `None (Raw)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Blind Text (SGD, raw)                 | raw       |     0.4001 |              0.7209 |        0.7345 |    0.8627 |      0.2424 |        0.0952 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.4671 |              0.8140 |        0.7965 |    0.9014 |      0.5000 |        0.0000 |
| Blind Text (SGD, std)                 | std       |     0.4280 |              0.7364 |        0.7516 |    0.8586 |      0.4255 |        0.0000 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.5164 |              0.8062 |        0.7938 |    0.8920 |      0.4571 |        0.2000 |

##### Run: `Case 1` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8800 |   0.8462 |     0.8627 |       104 |
| Ironia       |      0.2667 |   0.2222 |     0.2424 |        18 |
| Sarcasmo     |      0.0714 |   0.1429 |     0.0952 |         7 |
| macro avg    |      0.4060 |   0.4037 |     0.4001 |       129 |
| weighted avg |      0.7505 |   0.7209 |     0.7345 |       129 |
| accuracy     |      0.7209 |   0.7209 |     0.7209 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          88 |             9 |               7 |          104 |
| Ironia       |           8 |             4 |               6 |           18 |
| Sarcasmo     |           4 |             2 |               1 |            7 |

##### Run: `Case 1` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8807 |   0.9231 |     0.9014 |       104 |
| Ironia       |      0.5000 |   0.5000 |     0.5000 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.4602 |   0.4744 |     0.4671 |       129 |
| weighted avg |      0.7798 |   0.8140 |     0.7965 |       129 |
| accuracy     |      0.8140 |   0.8140 |     0.8140 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          96 |             7 |               1 |          104 |
| Ironia       |           8 |             9 |               1 |           18 |
| Sarcasmo     |           5 |             2 |               0 |            7 |

##### Run: `Case 1` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.9043 |   0.8173 |     0.8586 |       104 |
| Ironia       |      0.3448 |   0.5556 |     0.4255 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.4164 |   0.4576 |     0.4280 |       129 |
| weighted avg |      0.7771 |   0.7364 |     0.7516 |       129 |
| accuracy     |      0.7364 |   0.7364 |     0.7364 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          85 |            15 |               4 |          104 |
| Ironia       |           6 |            10 |               2 |           18 |
| Sarcasmo     |           3 |             4 |               0 |            7 |

##### Run: `Case 1` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8716 |   0.9135 |     0.8920 |       104 |
| Ironia       |      0.4706 |   0.4444 |     0.4571 |        18 |
| Sarcasmo     |      0.3333 |   0.1429 |     0.2000 |         7 |
| macro avg    |      0.5585 |   0.5003 |     0.5164 |       129 |
| weighted avg |      0.7864 |   0.8062 |     0.7938 |       129 |
| accuracy     |      0.8062 |   0.8062 |     0.8062 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          95 |             7 |               2 |          104 |
| Ironia       |          10 |             8 |               0 |           18 |
| Sarcasmo     |           4 |             2 |               1 |            7 |

#### Case 2: Step 2 Raw (No corr) (Checkpoint: `irony_textual_step2_raw`, Eval Prompt: `None (Raw)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Blind Text (SGD, raw)                 | raw       |     0.4023 |              0.7209 |        0.7338 |    0.8800 |      0.0769 |        0.2500 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5561 |              0.8372 |        0.8274 |    0.9151 |      0.5714 |        0.1818 |
| Blind Text (SGD, std)                 | std       |     0.4631 |              0.7519 |        0.7659 |    0.8911 |      0.2400 |        0.2581 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.4540 |              0.8062 |        0.7904 |    0.9005 |      0.4615 |        0.0000 |

##### Run: `Case 2` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.9167 |   0.8462 |     0.8800 |       104 |
| Ironia       |      0.1250 |   0.0556 |     0.0769 |        18 |
| Sarcasmo     |      0.1600 |   0.5714 |     0.2500 |         7 |
| macro avg    |      0.4006 |   0.4910 |     0.4023 |       129 |
| weighted avg |      0.7651 |   0.7209 |     0.7338 |       129 |
| accuracy     |      0.7209 |   0.7209 |     0.7209 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          88 |             6 |              10 |          104 |
| Ironia       |           6 |             1 |              11 |           18 |
| Sarcasmo     |           2 |             1 |               4 |            7 |

##### Run: `Case 2` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8981 |   0.9327 |     0.9151 |       104 |
| Ironia       |      0.5882 |   0.5556 |     0.5714 |        18 |
| Sarcasmo     |      0.2500 |   0.1429 |     0.1818 |         7 |
| macro avg    |      0.5788 |   0.5437 |     0.5561 |       129 |
| weighted avg |      0.8197 |   0.8372 |     0.8274 |       129 |
| accuracy     |      0.8372 |   0.8372 |     0.8372 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          97 |             5 |               2 |          104 |
| Ironia       |           7 |            10 |               1 |           18 |
| Sarcasmo     |           4 |             2 |               1 |            7 |

##### Run: `Case 2` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.9184 |   0.8654 |     0.8911 |       104 |
| Ironia       |      0.4286 |   0.1667 |     0.2400 |        18 |
| Sarcasmo     |      0.1667 |   0.5714 |     0.2581 |         7 |
| macro avg    |      0.5045 |   0.5345 |     0.4631 |       129 |
| weighted avg |      0.8092 |   0.7519 |     0.7659 |       129 |
| accuracy     |      0.7519 |   0.7519 |     0.7519 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          90 |             3 |              11 |          104 |
| Ironia       |           6 |             3 |               9 |           18 |
| Sarcasmo     |           2 |             1 |               4 |            7 |

##### Run: `Case 2` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8879 |   0.9135 |     0.9005 |       104 |
| Ironia       |      0.4286 |   0.5000 |     0.4615 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.4388 |   0.4712 |     0.4540 |       129 |
| weighted avg |      0.7756 |   0.8062 |     0.7904 |       129 |
| accuracy     |      0.8062 |   0.8062 |     0.8062 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          95 |             9 |               0 |          104 |
| Ironia       |           8 |             9 |               1 |           18 |
| Sarcasmo     |           4 |             3 |               0 |            7 |

#### Case 3: Step 2 Oracle (Train: no rel, Eval: no rel) (Checkpoint: `irony_textual_corr`, Eval Prompt: `Oracle (no rel)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Blind Text (SGD, raw)                 | raw       |     0.3954 |              0.7287 |        0.7342 |    0.8529 |      0.3333 |        0.0000 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.4069 |              0.7597 |        0.7509 |    0.8708 |      0.3500 |        0.0000 |
| Blind Text (SGD, std)                 | std       |     0.3956 |              0.7132 |        0.7242 |    0.8529 |      0.2162 |        0.1176 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.3810 |              0.7287 |        0.7263 |    0.8502 |      0.2927 |        0.0000 |

##### Run: `Case 3` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8700 |   0.8365 |     0.8529 |       104 |
| Ironia       |      0.2917 |   0.3889 |     0.3333 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.3872 |   0.4085 |     0.3954 |       129 |
| weighted avg |      0.7421 |   0.7287 |     0.7342 |       129 |
| accuracy     |      0.7287 |   0.7287 |     0.7287 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          87 |            14 |               3 |          104 |
| Ironia       |           9 |             7 |               2 |           18 |
| Sarcasmo     |           4 |             3 |               0 |            7 |

##### Run: `Case 3` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8667 |   0.8750 |     0.8708 |       104 |
| Ironia       |      0.3182 |   0.3889 |     0.3500 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.3949 |   0.4213 |     0.4069 |       129 |
| weighted avg |      0.7431 |   0.7597 |     0.7509 |       129 |
| accuracy     |      0.7597 |   0.7597 |     0.7597 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          91 |            12 |               1 |          104 |
| Ironia       |          10 |             7 |               1 |           18 |
| Sarcasmo     |           4 |             3 |               0 |            7 |

##### Run: `Case 3` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8700 |   0.8365 |     0.8529 |       104 |
| Ironia       |      0.2105 |   0.2222 |     0.2162 |        18 |
| Sarcasmo     |      0.1000 |   0.1429 |     0.1176 |         7 |
| macro avg    |      0.3935 |   0.4005 |     0.3956 |       129 |
| weighted avg |      0.7362 |   0.7132 |     0.7242 |       129 |
| accuracy     |      0.7132 |   0.7132 |     0.7132 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          87 |            13 |               4 |          104 |
| Ironia       |           9 |             4 |               5 |           18 |
| Sarcasmo     |           4 |             2 |               1 |            7 |

##### Run: `Case 3` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8544 |   0.8462 |     0.8502 |       104 |
| Ironia       |      0.2609 |   0.3333 |     0.2927 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.3717 |   0.3932 |     0.3810 |       129 |
| weighted avg |      0.7252 |   0.7287 |     0.7263 |       129 |
| accuracy     |      0.7287 |   0.7287 |     0.7287 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          88 |            14 |               2 |          104 |
| Ironia       |          11 |             6 |               1 |           18 |
| Sarcasmo     |           4 |             3 |               0 |            7 |

#### Case 4: Step 2 Oracle (Train: no rel, Eval: +rel) (Checkpoint: `irony_textual_corr`, Eval Prompt: `Oracle (+ rel)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Blind Text (SGD, raw)                 | raw       |     0.3896 |              0.7209 |        0.7252 |    0.8431 |      0.3256 |        0.0000 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.4414 |              0.7752 |        0.7645 |    0.8696 |      0.4545 |        0.0000 |
| Blind Text (SGD, std)                 | std       |     0.3299 |              0.6744 |        0.6953 |    0.8358 |      0.1538 |        0.0000 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.3791 |              0.7209 |        0.7218 |    0.8447 |      0.2927 |        0.0000 |

##### Run: `Case 4` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8600 |   0.8269 |     0.8431 |       104 |
| Ironia       |      0.2800 |   0.3889 |     0.3256 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.3800 |   0.4053 |     0.3896 |       129 |
| weighted avg |      0.7324 |   0.7209 |     0.7252 |       129 |
| accuracy     |      0.7209 |   0.7209 |     0.7209 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          86 |            16 |               2 |          104 |
| Ironia       |           9 |             7 |               2 |           18 |
| Sarcasmo     |           5 |             2 |               0 |            7 |

##### Run: `Case 4` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8738 |   0.8654 |     0.8696 |       104 |
| Ironia       |      0.3846 |   0.5556 |     0.4545 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.4195 |   0.4736 |     0.4414 |       129 |
| weighted avg |      0.7581 |   0.7752 |     0.7645 |       129 |
| accuracy     |      0.7752 |   0.7752 |     0.7752 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          90 |            14 |               0 |          104 |
| Ironia       |           8 |            10 |               0 |           18 |
| Sarcasmo     |           5 |             2 |               0 |            7 |

##### Run: `Case 4` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8660 |   0.8077 |     0.8358 |       104 |
| Ironia       |      0.1429 |   0.1667 |     0.1538 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.3363 |   0.3248 |     0.3299 |       129 |
| weighted avg |      0.7181 |   0.6744 |     0.6953 |       129 |
| accuracy     |      0.6744 |   0.6744 |     0.6744 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          84 |            16 |               4 |          104 |
| Ironia       |           8 |             3 |               7 |           18 |
| Sarcasmo     |           5 |             2 |               0 |            7 |

##### Run: `Case 4` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8529 |   0.8365 |     0.8447 |       104 |
| Ironia       |      0.2609 |   0.3333 |     0.2927 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.3713 |   0.3900 |     0.3791 |       129 |
| weighted avg |      0.7240 |   0.7209 |     0.7218 |       129 |
| accuracy     |      0.7209 |   0.7209 |     0.7209 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          87 |            15 |               2 |          104 |
| Ironia       |          10 |             6 |               2 |           18 |
| Sarcasmo     |           5 |             2 |               0 |            7 |

#### Case 5: Step 2 Oracle (Train: +rel, Eval: no rel) (Checkpoint: `irony_textual_corr_with_rel`, Eval Prompt: `Oracle (no rel)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Blind Text (SGD, raw)                 | raw       |     0.4486 |              0.7442 |        0.7393 |    0.8667 |      0.1714 |        0.3077 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.4157 |              0.7674 |        0.7573 |    0.8750 |      0.3721 |        0.0000 |
| Blind Text (SGD, std)                 | std       |     0.4088 |              0.7364 |        0.7388 |    0.8515 |      0.3750 |        0.0000 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.3783 |              0.7364 |        0.7289 |    0.8558 |      0.2791 |        0.0000 |

##### Run: `Case 5` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8585 |   0.8750 |     0.8667 |       104 |
| Ironia       |      0.1765 |   0.1667 |     0.1714 |        18 |
| Sarcasmo     |      0.3333 |   0.2857 |     0.3077 |         7 |
| macro avg    |      0.4561 |   0.4425 |     0.4486 |       129 |
| weighted avg |      0.7348 |   0.7442 |     0.7393 |       129 |
| accuracy     |      0.7442 |   0.7442 |     0.7442 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          91 |            12 |               1 |          104 |
| Ironia       |          12 |             3 |               3 |           18 |
| Sarcasmo     |           3 |             2 |               2 |            7 |

##### Run: `Case 5` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8750 |   0.8750 |     0.8750 |       104 |
| Ironia       |      0.3200 |   0.4444 |     0.3721 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.3983 |   0.4398 |     0.4157 |       129 |
| weighted avg |      0.7501 |   0.7674 |     0.7573 |       129 |
| accuracy     |      0.7674 |   0.7674 |     0.7674 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          91 |            13 |               0 |          104 |
| Ironia       |          10 |             8 |               0 |           18 |
| Sarcasmo     |           3 |             4 |               0 |            7 |

##### Run: `Case 5` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8776 |   0.8269 |     0.8515 |       104 |
| Ironia       |      0.3000 |   0.5000 |     0.3750 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.3925 |   0.4423 |     0.4088 |       129 |
| weighted avg |      0.7493 |   0.7364 |     0.7388 |       129 |
| accuracy     |      0.7364 |   0.7364 |     0.7364 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          86 |            17 |               1 |          104 |
| Ironia       |           9 |             9 |               0 |           18 |
| Sarcasmo     |           3 |             4 |               0 |            7 |

##### Run: `Case 5` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8558 |   0.8558 |     0.8558 |       104 |
| Ironia       |      0.2400 |   0.3333 |     0.2791 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.3653 |   0.3964 |     0.3783 |       129 |
| weighted avg |      0.7234 |   0.7364 |     0.7289 |       129 |
| accuracy     |      0.7364 |   0.7364 |     0.7364 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          89 |            15 |               0 |          104 |
| Ironia       |          12 |             6 |               0 |           18 |
| Sarcasmo     |           3 |             4 |               0 |            7 |

#### Case 6: Step 2 Oracle (Train: +rel, Eval: +rel) (Checkpoint: `irony_textual_corr_with_rel`, Eval Prompt: `Oracle (+ rel)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   None F1 |   Ironia F1 |   Sarcasmo F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|----------:|------------:|--------------:|
| Blind Text (SGD, raw)                 | raw       |     0.3942 |              0.7519 |        0.7484 |    0.8750 |      0.3077 |        0.0000 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.4323 |              0.7907 |        0.7722 |    0.8868 |      0.4103 |        0.0000 |
| Blind Text (SGD, std)                 | std       |     0.4308 |              0.7597 |        0.7583 |    0.8670 |      0.4255 |        0.0000 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.4139 |              0.7597 |        0.7530 |    0.8696 |      0.3721 |        0.0000 |

##### Run: `Case 6` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8750 |   0.8750 |     0.8750 |       104 |
| Ironia       |      0.2857 |   0.3333 |     0.3077 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.3869 |   0.4028 |     0.3942 |       129 |
| weighted avg |      0.7453 |   0.7519 |     0.7484 |       129 |
| accuracy     |      0.7519 |   0.7519 |     0.7519 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          91 |            12 |               1 |          104 |
| Ironia       |           9 |             6 |               3 |           18 |
| Sarcasmo     |           4 |             3 |               0 |            7 |

##### Run: `Case 6` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8704 |   0.9038 |     0.8868 |       104 |
| Ironia       |      0.3810 |   0.4444 |     0.4103 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.4171 |   0.4494 |     0.4323 |       129 |
| weighted avg |      0.7549 |   0.7907 |     0.7722 |       129 |
| accuracy     |      0.7907 |   0.7907 |     0.7907 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          94 |            10 |               0 |          104 |
| Ironia       |          10 |             8 |               0 |           18 |
| Sarcasmo     |           4 |             3 |               0 |            7 |

##### Run: `Case 6` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8889 |   0.8462 |     0.8670 |       104 |
| Ironia       |      0.3448 |   0.5556 |     0.4255 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.4112 |   0.4672 |     0.4308 |       129 |
| weighted avg |      0.7647 |   0.7597 |     0.7583 |       129 |
| accuracy     |      0.7597 |   0.7597 |     0.7597 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          88 |            15 |               1 |          104 |
| Ironia       |           8 |            10 |               0 |           18 |
| Sarcasmo     |           3 |             4 |               0 |            7 |

##### Run: `Case 6` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| nan          |      0.8738 |   0.8654 |     0.8696 |       104 |
| Ironia       |      0.3200 |   0.4444 |     0.3721 |        18 |
| Sarcasmo     |      0.0000 |   0.0000 |     0.0000 |         7 |
| macro avg    |      0.3979 |   0.4366 |     0.4139 |       129 |
| weighted avg |      0.7491 |   0.7597 |     0.7530 |       129 |
| accuracy     |      0.7597 |   0.7597 |     0.7597 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_None |   Pred_Ironia |   Pred_Sarcasmo |   Total_True |
|:-------------|------------:|--------------:|----------------:|-------------:|
| nan          |          90 |            14 |               0 |          104 |
| Ironia       |           9 |             8 |               1 |           18 |
| Sarcasmo     |           4 |             3 |               0 |            7 |

---

## 3. Task 3: Semiotic Relation Classification (3-Class Single-Label, `UmBERTo` + `ViT-224`)

- **Text Encoder**: `Musixmatch/umberto-commoncrawl-cased-v1` (`UmBERTo`, 768-d `[CLS]`)
- **Vision Encoder**: `google/vit-base-patch16-224` (`ViT-Base-224`, frozen 768-d `[CLS]`)
- **Multimodal Head (`mm_test`, $N=129$)**: `SGDClassifier(loss='squared_hinge', penalty='l2', class_weight='balanced', random_state=42)`

### 3.1 Textual Encoder Training Benchmark (`mm_test`, $N=129$)

| Regime         | Checkpoint               | Prompt   |   N Test |   Macro F1 |   Micro F1 (= Acc) |   Weighted F1 |   Additiva F1 |   Divergente F1 |   Parallela F1 |
|:---------------|:-------------------------|:---------|---------:|-----------:|-------------------:|--------------:|--------------:|----------------:|---------------:|
| Stage A Raw    | relation_textual_no_corr | Raw      |      129 |     0.5886 |             0.6667 |        0.6476 |        0.7421 |          0.6349 |         0.3889 |
| Stage B Oracle | relation_textual_corr    | Oracle   |      129 |     0.5044 |             0.6434 |        0.5959 |        0.7229 |          0.6364 |         0.1538 |

#### Textual Encoder Stage: `Stage A Raw` (`relation_textual_no_corr`)
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6484 |   0.8676 |     0.7421 |        68 |
| Divergente   |      0.8000 |   0.5263 |     0.6349 |        38 |
| Parallela    |      0.5385 |   0.3043 |     0.3889 |        23 |
| macro avg    |      0.6623 |   0.5661 |     0.5886 |       129 |
| weighted avg |      0.6734 |   0.6667 |     0.6476 |       129 |
| accuracy     |      0.6667 |   0.6667 |     0.6667 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              59 |                 5 |                4 |           68 |
| Divergente   |              16 |                20 |                2 |           38 |
| Parallela    |              16 |                 0 |                7 |           23 |

#### Textual Encoder Stage: `Stage B Oracle` (`relation_textual_corr`)
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6122 |   0.8824 |     0.7229 |        68 |
| Divergente   |      0.7500 |   0.5526 |     0.6364 |        38 |
| Parallela    |      0.6667 |   0.0870 |     0.1538 |        23 |
| macro avg    |      0.6763 |   0.5073 |     0.5044 |       129 |
| weighted avg |      0.6625 |   0.6434 |     0.5959 |       129 |
| accuracy     |      0.6434 |   0.6434 |     0.6434 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              60 |                 7 |                1 |           68 |
| Divergente   |              17 |                21 |                0 |           38 |
| Parallela    |              21 |                 0 |                2 |           23 |

---

### 3.2 Multimodal Semiotic Relation Benchmark (`mm_test`, $N=129$ — `Stage A Raw` vs `Stage B Oracle` $\times$ `raw` vs `std`)

#### Master Comparison Table (All Multimodal Relation Runs)
| Case   | Regime                     | Prompting   | Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   Additiva F1 |   Divergente F1 |   Parallela F1 |
|:-------|:---------------------------|:------------|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|--------------:|----------------:|---------------:|
| Case 1 | Stage A Raw (No corr)      | None (Raw)  | Blind Text (SGD, raw)                 | raw       |     0.5366 |              0.6202 |        0.5981 |        0.7044 |          0.5625 |         0.3429 |
| Case 1 | Stage A Raw (No corr)      | None (Raw)  | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5779 |              0.6202 |        0.6179 |        0.7000 |          0.5556 |         0.4783 |
| Case 1 | Stage A Raw (No corr)      | None (Raw)  | Blind Text (SGD, std)                 | std       |     0.5784 |              0.6279 |        0.6186 |        0.7067 |          0.5397 |         0.4889 |
| Case 1 | Stage A Raw (No corr)      | None (Raw)  | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.6412 |              0.6667 |        0.6647 |        0.7133 |          0.6269 |         0.5833 |
| Case 2 | Stage B Oracle (With corr) | Oracle      | Blind Text (SGD, raw)                 | raw       |     0.6341 |              0.6667 |        0.6640 |        0.7123 |          0.6567 |         0.5333 |
| Case 2 | Stage B Oracle (With corr) | Oracle      | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.6088 |              0.6589 |        0.6525 |        0.7248 |          0.6364 |         0.4651 |
| Case 2 | Stage B Oracle (With corr) | Oracle      | Blind Text (SGD, std)                 | std       |     0.6020 |              0.6434 |        0.6378 |        0.6933 |          0.6364 |         0.4762 |
| Case 2 | Stage B Oracle (With corr) | Oracle      | Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.6027 |              0.6434 |        0.6371 |        0.6933 |          0.6269 |         0.4878 |

#### Target Classes Per-Class F1 Breakdown (`ADD`, `PAR`, `DIV` on `mm_test`, $N=129$)
| Case   | Regime                     | Model                                 | Scaling   |   ADD (n=68) |   PAR (n=23) |   DIV (n=38) |   Macro F1 |   Micro F1 (= Acc.) |
|:-------|:---------------------------|:--------------------------------------|:----------|-------------:|-------------:|-------------:|-----------:|--------------------:|
| Case 1 | Stage A Raw (No corr)      | Blind Text (SGD, raw)                 | raw       |       0.7044 |       0.3429 |       0.5625 |     0.5366 |              0.6202 |
| Case 1 | Stage A Raw (No corr)      | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.7000 |       0.4783 |       0.5556 |     0.5779 |              0.6202 |
| Case 1 | Stage A Raw (No corr)      | Blind Text (SGD, std)                 | std       |       0.7067 |       0.4889 |       0.5397 |     0.5784 |              0.6279 |
| Case 1 | Stage A Raw (No corr)      | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.7133 |       0.5833 |       0.6269 |     0.6412 |              0.6667 |
| Case 2 | Stage B Oracle (With corr) | Blind Text (SGD, raw)                 | raw       |       0.7123 |       0.5333 |       0.6567 |     0.6341 |              0.6667 |
| Case 2 | Stage B Oracle (With corr) | Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |       0.7248 |       0.4651 |       0.6364 |     0.6088 |              0.6589 |
| Case 2 | Stage B Oracle (With corr) | Blind Text (SGD, std)                 | std       |       0.6933 |       0.4762 |       0.6364 |     0.6020 |              0.6434 |
| Case 2 | Stage B Oracle (With corr) | Dual-Encoder (UmBERTo + ViT-224, std) | std       |       0.6933 |       0.4878 |       0.6269 |     0.6027 |              0.6434 |

### 3.3 Multimodal Semiotic Relation Detailed Reports by Case (`Case 1` & `Case 2`)

#### Case 1: Stage A Raw (No corr) (Checkpoint: `relation_textual_no_corr`, Eval Prompt: `None (Raw)`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   Additiva F1 |   Divergente F1 |   Parallela F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|--------------:|----------------:|---------------:|
| Blind Text (SGD, raw)                 | raw       |     0.5366 |              0.6202 |        0.5981 |        0.7044 |          0.5625 |         0.3429 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.5779 |              0.6202 |        0.6179 |        0.7000 |          0.5556 |         0.4783 |
| Blind Text (SGD, std)                 | std       |     0.5784 |              0.6279 |        0.6186 |        0.7067 |          0.5397 |         0.4889 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.6412 |              0.6667 |        0.6647 |        0.7133 |          0.6269 |         0.5833 |

##### Run: `Case 1` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6154 |   0.8235 |     0.7044 |        68 |
| Divergente   |      0.6923 |   0.4737 |     0.5625 |        38 |
| Parallela    |      0.5000 |   0.2609 |     0.3429 |        23 |
| macro avg    |      0.6026 |   0.5194 |     0.5366 |       129 |
| weighted avg |      0.6175 |   0.6202 |     0.5981 |       129 |
| accuracy     |      0.6202 |   0.6202 |     0.6202 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              56 |                 7 |                5 |           68 |
| Divergente   |              19 |                18 |                1 |           38 |
| Parallela    |              16 |                 1 |                6 |           23 |

##### Run: `Case 1` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6806 |   0.7206 |     0.7000 |        68 |
| Divergente   |      0.5882 |   0.5263 |     0.5556 |        38 |
| Parallela    |      0.4783 |   0.4783 |     0.4783 |        23 |
| macro avg    |      0.5824 |   0.5751 |     0.5779 |       129 |
| weighted avg |      0.6173 |   0.6202 |     0.6179 |       129 |
| accuracy     |      0.6202 |   0.6202 |     0.6202 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              49 |                10 |                9 |           68 |
| Divergente   |              15 |                20 |                3 |           38 |
| Parallela    |               8 |                 4 |               11 |           23 |

##### Run: `Case 1` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6463 |   0.7794 |     0.7067 |        68 |
| Divergente   |      0.6800 |   0.4474 |     0.5397 |        38 |
| Parallela    |      0.5000 |   0.4783 |     0.4889 |        23 |
| macro avg    |      0.6088 |   0.5683 |     0.5784 |       129 |
| weighted avg |      0.6302 |   0.6279 |     0.6186 |       129 |
| accuracy     |      0.6279 |   0.6279 |     0.6279 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              53 |                 8 |                7 |           68 |
| Divergente   |              17 |                17 |                4 |           38 |
| Parallela    |              12 |                 0 |               11 |           23 |

##### Run: `Case 1` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6800 |   0.7500 |     0.7133 |        68 |
| Divergente   |      0.7241 |   0.5526 |     0.6269 |        38 |
| Parallela    |      0.5600 |   0.6087 |     0.5833 |        23 |
| macro avg    |      0.6547 |   0.6371 |     0.6412 |       129 |
| weighted avg |      0.6716 |   0.6667 |     0.6647 |       129 |
| accuracy     |      0.6667 |   0.6667 |     0.6667 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              51 |                 7 |               10 |           68 |
| Divergente   |              16 |                21 |                1 |           38 |
| Parallela    |               8 |                 1 |               14 |           23 |

#### Case 2: Stage B Oracle (With corr) (Checkpoint: `relation_textual_corr`, Eval Prompt: `Oracle`)

| Model                                 | Scaling   |   Macro F1 |   Micro F1 (= Acc.) |   Weighted F1 |   Additiva F1 |   Divergente F1 |   Parallela F1 |
|:--------------------------------------|:----------|-----------:|--------------------:|--------------:|--------------:|----------------:|---------------:|
| Blind Text (SGD, raw)                 | raw       |     0.6341 |              0.6667 |        0.6640 |        0.7123 |          0.6567 |         0.5333 |
| Dual-Encoder (UmBERTo + ViT-224, raw) | raw       |     0.6088 |              0.6589 |        0.6525 |        0.7248 |          0.6364 |         0.4651 |
| Blind Text (SGD, std)                 | std       |     0.6020 |              0.6434 |        0.6378 |        0.6933 |          0.6364 |         0.4762 |
| Dual-Encoder (UmBERTo + ViT-224, std) | std       |     0.6027 |              0.6434 |        0.6371 |        0.6933 |          0.6269 |         0.4878 |

##### Run: `Case 2` | `Blind Text (SGD, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6667 |   0.7647 |     0.7123 |        68 |
| Divergente   |      0.7586 |   0.5789 |     0.6567 |        38 |
| Parallela    |      0.5455 |   0.5217 |     0.5333 |        23 |
| macro avg    |      0.6569 |   0.6218 |     0.6341 |       129 |
| weighted avg |      0.6721 |   0.6667 |     0.6640 |       129 |
| accuracy     |      0.6667 |   0.6667 |     0.6667 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              52 |                 7 |                9 |           68 |
| Divergente   |              15 |                22 |                1 |           38 |
| Parallela    |              11 |                 0 |               12 |           23 |

##### Run: `Case 2` | `Dual-Encoder (UmBERTo + ViT-224, raw)` | Scaling: `raw`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6667 |   0.7941 |     0.7248 |        68 |
| Divergente   |      0.7500 |   0.5526 |     0.6364 |        38 |
| Parallela    |      0.5000 |   0.4348 |     0.4651 |        23 |
| macro avg    |      0.6389 |   0.5938 |     0.6088 |       129 |
| weighted avg |      0.6615 |   0.6589 |     0.6525 |       129 |
| accuracy     |      0.6589 |   0.6589 |     0.6589 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              54 |                 6 |                8 |           68 |
| Divergente   |              15 |                21 |                2 |           38 |
| Parallela    |              12 |                 1 |               10 |           23 |

##### Run: `Case 2` | `Blind Text (SGD, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6341 |   0.7647 |     0.6933 |        68 |
| Divergente   |      0.7500 |   0.5526 |     0.6364 |        38 |
| Parallela    |      0.5263 |   0.4348 |     0.4762 |        23 |
| macro avg    |      0.6368 |   0.5840 |     0.6020 |       129 |
| weighted avg |      0.6490 |   0.6434 |     0.6378 |       129 |
| accuracy     |      0.6434 |   0.6434 |     0.6434 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              52 |                 7 |                9 |           68 |
| Divergente   |              17 |                21 |                0 |           38 |
| Parallela    |              13 |                 0 |               10 |           23 |

##### Run: `Case 2` | `Dual-Encoder (UmBERTo + ViT-224, std)` | Scaling: `std`
**Classification Report**:

| Label        |   Precision |   Recall |   F1-Score |   Support |
|:-------------|------------:|---------:|-----------:|----------:|
| Additiva     |      0.6341 |   0.7647 |     0.6933 |        68 |
| Divergente   |      0.7241 |   0.5526 |     0.6269 |        38 |
| Parallela    |      0.5556 |   0.4348 |     0.4878 |        23 |
| macro avg    |      0.6379 |   0.5840 |     0.6027 |       129 |
| weighted avg |      0.6466 |   0.6434 |     0.6371 |       129 |
| accuracy     |      0.6434 |   0.6434 |     0.6434 |       129 |

**3x3 Confusion Matrix**:

| True_Label   |   Pred_Additiva |   Pred_Divergente |   Pred_Parallela |   Total_True |
|:-------------|----------------:|------------------:|-----------------:|-------------:|
| Additiva     |              52 |                 8 |                8 |           68 |
| Divergente   |              17 |                21 |                0 |           38 |
| Parallela    |              13 |                 0 |               10 |           23 |
