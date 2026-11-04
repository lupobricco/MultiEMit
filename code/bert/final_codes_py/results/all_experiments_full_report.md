# MultiEMit Full Benchmark Experiments Report
Generated on: 2026-11-03 15:59:45

## 1. Unified Summary Metrics Table

| Task                                               | Stage             | Regime                                 | Prefix          | Model                                | Features   |   N Test |   Micro F1 |   Macro F1 |   Weighted F1 |   Accuracy | Notes                       |
|:---------------------------------------------------|:------------------|:---------------------------------------|:----------------|:-------------------------------------|:-----------|---------:|-----------:|-----------:|--------------:|-----------:|:----------------------------|
| Emotion Detection (10-Class Multi-Label)           | Textual (me_test) | Step 1: EMit A + ME Textual            | No              | FEEL-IT (fine-tuned)                 | -          |      405 |     0.5403 |     0.5092 |        0.5371 |     0.321  | threshold=0.40              |
| Emotion Detection (10-Class Multi-Label)           | Textual (me_test) | Step 2 Raw: Step 1 -> ME + MM Train    | No              | FEEL-IT (fine-tuned)                 | -          |      405 |     0.5383 |     0.5235 |        0.5347 |     0.3012 | threshold=0.40; A3 ablation |
| Emotion Detection (10-Class Multi-Label)           | Textual (me_test) | Step 2 Oracle: Step 1 -> ME + MM Train | Oracle (no rel) | FEEL-IT (fine-tuned)                 | -          |      405 |     0.553  |     0.5349 |        0.5507 |     0.3259 | threshold=0.40              |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)              | No              | Fine-tuned model (direct prediction) | -          |      129 |     0.4492 |     0.375  |        0.4442 |     0.3023 | nan                         |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)              | No              | Blind text (frozen CLS + head)       | raw        |      129 |     0.4314 |     0.3565 |        0.4147 |     0.2636 | convergence_warnings=0      |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)              | No              | Dual-Encoder (text + image)          | raw        |      129 |     0.429  |     0.4068 |        0.4229 |     0.2791 | convergence_warnings=0      |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)              | No              | Blind text (frozen CLS + head)       | std        |      129 |     0.3762 |     0.3255 |        0.4111 |     0.1163 | convergence_warnings=0      |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)              | No              | Dual-Encoder (text + image)          | std        |      129 |     0.4329 |     0.3787 |        0.4548 |     0.1395 | convergence_warnings=0      |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)         | No              | Fine-tuned model (direct prediction) | -          |      129 |     0.5217 |     0.437  |        0.5165 |     0.3411 | nan                         |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)         | No              | Blind text (frozen CLS + head)       | raw        |      129 |     0.4941 |     0.4233 |        0.4928 |     0.2868 | convergence_warnings=0      |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)         | No              | Dual-Encoder (text + image)          | raw        |      129 |     0.506  |     0.4343 |        0.4913 |     0.2868 | convergence_warnings=1      |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)         | No              | Blind text (frozen CLS + head)       | std        |      129 |     0.4624 |     0.4016 |        0.4959 |     0.2481 | convergence_warnings=0      |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)         | No              | Dual-Encoder (text + image)          | std        |      129 |     0.5053 |     0.4313 |        0.5255 |     0.2946 | convergence_warnings=0      |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM)      | Oracle (no rel) | Fine-tuned model (direct prediction) | -          |      129 |     0.6106 |     0.534  |        0.6114 |     0.4496 | nan                         |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM)      | Oracle (no rel) | Blind text (frozen CLS + head)       | raw        |      129 |     0.5913 |     0.5285 |        0.6026 |     0.3721 | convergence_warnings=0      |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM)      | Oracle (no rel) | Dual-Encoder (text + image)          | raw        |      129 |     0.5633 |     0.5224 |        0.5762 |     0.3488 | convergence_warnings=0      |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM)      | Oracle (no rel) | Blind text (frozen CLS + head)       | std        |      129 |     0.5189 |     0.4617 |        0.5547 |     0.2868 | convergence_warnings=0      |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM)      | Oracle (no rel) | Dual-Encoder (text + image)          | std        |      129 |     0.5435 |     0.4863 |        0.5659 |     0.3101 | convergence_warnings=0      |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM)      | Oracle (+ rel)  | Fine-tuned model (direct prediction) | -          |      129 |     0.5397 |     0.479  |        0.5485 |     0.4031 | nan                         |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM)      | Oracle (+ rel)  | Blind text (frozen CLS + head)       | raw        |      129 |     0.5183 |     0.4343 |        0.5456 |     0.3488 | convergence_warnings=0      |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM)      | Oracle (+ rel)  | Dual-Encoder (text + image)          | raw        |      129 |     0.5535 |     0.4967 |        0.5477 |     0.3566 | convergence_warnings=0      |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM)      | Oracle (+ rel)  | Blind text (frozen CLS + head)       | std        |      129 |     0.5213 |     0.455  |        0.5584 |     0.3488 | convergence_warnings=0      |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM)      | Oracle (+ rel)  | Dual-Encoder (text + image)          | std        |      129 |     0.5215 |     0.4537 |        0.5569 |     0.3101 | convergence_warnings=0      |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Textual (me_test) | Step 1: IronITA + ME Textual           | No              | UmBERTo (fine-tuned)                 | -          |      405 |     0.7235 |     0.4411 |        0.7222 |     0.7235 | nan                         |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Textual (me_test) | Step 2 Raw: Step 1 -> ME + MM Train    | No              | UmBERTo (fine-tuned)                 | -          |      405 |     0.7358 |     0.4312 |        0.7314 |     0.7358 | A3 ablation                 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Textual (me_test) | Step 2 Oracle: Step 1 -> ME + MM Train | Oracle (no rel) | UmBERTo (fine-tuned)                 | -          |      405 |     0.7605 |     0.4995 |        0.7612 |     0.7605 | nan                         |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)             | No              | Fine-tuned model (direct prediction) | -          |      129 |     0.7364 |     0.4224 |        0.7225 |     0.7364 | nan                         |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)             | No              | Blind text (frozen CLS + head)       | raw        |      129 |     0.3566 |     0.2617 |        0.4617 |     0.3566 | convergence_warnings=0      |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)             | No              | Dual-Encoder (text + image)          | raw        |      129 |     0.7907 |     0.437  |        0.7686 |     0.7907 | convergence_warnings=0      |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)             | No              | Blind text (frozen CLS + head)       | std        |      129 |     0.7597 |     0.433  |        0.7545 |     0.7597 | convergence_warnings=0      |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)             | No              | Dual-Encoder (text + image)          | std        |      129 |     0.8062 |     0.4458 |        0.782  |     0.8062 | convergence_warnings=0      |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)         | No              | Fine-tuned model (direct prediction) | -          |      129 |     0.7829 |     0.5316 |        0.7858 |     0.7829 | nan                         |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)         | No              | Blind text (frozen CLS + head)       | raw        |      129 |     0.7442 |     0.478  |        0.7606 |     0.7442 | convergence_warnings=0      |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)         | No              | Dual-Encoder (text + image)          | raw        |      129 |     0.7829 |     0.5279 |        0.7837 |     0.7829 | convergence_warnings=1      |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)         | No              | Blind text (frozen CLS + head)       | std        |      129 |     0.7287 |     0.5084 |        0.7484 |     0.7287 | convergence_warnings=0      |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)         | No              | Dual-Encoder (text + image)          | std        |      129 |     0.7597 |     0.5678 |        0.7731 |     0.7597 | convergence_warnings=1      |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM)      | Oracle (no rel) | Fine-tuned model (direct prediction) | -          |      129 |     0.7752 |     0.4538 |        0.7717 |     0.7752 | nan                         |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM)      | Oracle (no rel) | Blind text (frozen CLS + head)       | raw        |      129 |     0.7519 |     0.4876 |        0.7648 |     0.7519 | convergence_warnings=0      |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM)      | Oracle (no rel) | Dual-Encoder (text + image)          | raw        |      129 |     0.7674 |     0.4484 |        0.7657 |     0.7674 | convergence_warnings=0      |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM)      | Oracle (no rel) | Blind text (frozen CLS + head)       | std        |      129 |     0.6589 |     0.4279 |        0.6982 |     0.6589 | convergence_warnings=1      |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM)      | Oracle (no rel) | Dual-Encoder (text + image)          | std        |      129 |     0.8062 |     0.5424 |        0.8036 |     0.8062 | convergence_warnings=0      |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM)      | Oracle (+ rel)  | Fine-tuned model (direct prediction) | -          |      129 |     0.7364 |     0.4153 |        0.7377 |     0.7364 | nan                         |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM)      | Oracle (+ rel)  | Blind text (frozen CLS + head)       | raw        |      129 |     0.7519 |     0.4769 |        0.7549 |     0.7519 | convergence_warnings=0      |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM)      | Oracle (+ rel)  | Dual-Encoder (text + image)          | raw        |      129 |     0.7829 |     0.4481 |        0.7737 |     0.7829 | convergence_warnings=0      |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM)      | Oracle (+ rel)  | Blind text (frozen CLS + head)       | std        |      129 |     0.7209 |     0.4503 |        0.7335 |     0.7209 | convergence_warnings=0      |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM)      | Oracle (+ rel)  | Dual-Encoder (text + image)          | std        |      129 |     0.7209 |     0.4433 |        0.7413 |     0.7209 | convergence_warnings=0      |
| Semiotic Relation Detection (3-Class Single-Label) | Textual (mm_test) | Stage A: MM Train Raw                  | No              | UmBERTo (fine-tuned)                 | -          |      129 |     0.6512 |     0.469  |        0.5844 |     0.6512 | nan                         |
| Semiotic Relation Detection (3-Class Single-Label) | Textual (mm_test) | Stage B: MM Train Oracle               | Oracle          | UmBERTo (fine-tuned)                 | -          |      129 |     0.6279 |     0.5429 |        0.6076 |     0.6279 | nan                         |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw                  | No              | Fine-tuned model (direct prediction) | -          |      129 |     0.6512 |     0.469  |        0.5844 |     0.6512 | nan                         |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw                  | No              | Blind text (frozen CLS + head)       | raw        |      129 |     0.5969 |     0.5063 |        0.57   |     0.5969 | convergence_warnings=0      |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw                  | No              | Dual-Encoder (text + image)          | raw        |      129 |     0.6667 |     0.5729 |        0.6406 |     0.6667 | convergence_warnings=0      |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw                  | No              | Blind text (frozen CLS + head)       | std        |      129 |     0.6667 |     0.6011 |        0.6537 |     0.6667 | convergence_warnings=0      |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw                  | No              | Dual-Encoder (text + image)          | std        |      129 |     0.6512 |     0.6175 |        0.6501 |     0.6512 | convergence_warnings=0      |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle               | Oracle          | Fine-tuned model (direct prediction) | -          |      129 |     0.6279 |     0.5429 |        0.6076 |     0.6279 | nan                         |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle               | Oracle          | Blind text (frozen CLS + head)       | raw        |      129 |     0.6434 |     0.5408 |        0.6114 |     0.6434 | convergence_warnings=0      |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle               | Oracle          | Dual-Encoder (text + image)          | raw        |      129 |     0.6512 |     0.6242 |        0.6516 |     0.6512 | convergence_warnings=0      |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle               | Oracle          | Blind text (frozen CLS + head)       | std        |      129 |     0.6434 |     0.58   |        0.629  |     0.6434 | convergence_warnings=0      |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle               | Oracle          | Dual-Encoder (text + image)          | std        |      129 |     0.6357 |     0.6116 |        0.6346 |     0.6357 | convergence_warnings=0      |

---

## 2. Semiotic Relation Subsets Breakdown (ALL / ADD / PAR / DIV)

| Task                                               | Stage             | Regime                            | Prefix          | Model                                | Features   | Subset   |   N |   Micro F1 |   Macro F1 (classes with support) |   Weighted F1 |   Accuracy |
|:---------------------------------------------------|:------------------|:----------------------------------|:----------------|:-------------------------------------|:-----------|:---------|----:|-----------:|----------------------------------:|--------------:|-----------:|
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)         | No              | Fine-tuned model (direct prediction) | -          | ALL      | 129 |     0.4492 |                            0.375  |        0.4442 |     0.3023 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)         | No              | Fine-tuned model (direct prediction) | -          | ADD      |  68 |     0.5029 |                            0.4242 |        0.5025 |     0.3676 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)         | No              | Fine-tuned model (direct prediction) | -          | PAR      |  23 |     0.4314 |                            0.3165 |        0.4633 |     0.3043 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)         | No              | Fine-tuned model (direct prediction) | -          | DIV      |  38 |     0.3689 |                            0.3284 |        0.3879 |     0.1842 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)         | No              | Blind text (frozen CLS + head)       | raw        | ALL      | 129 |     0.4314 |                            0.3565 |        0.4147 |     0.2636 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)         | No              | Blind text (frozen CLS + head)       | raw        | ADD      |  68 |     0.4842 |                            0.4091 |        0.4649 |     0.3382 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)         | No              | Blind text (frozen CLS + head)       | raw        | PAR      |  23 |     0.4483 |                            0.2793 |        0.4951 |     0.3043 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)         | No              | Blind text (frozen CLS + head)       | raw        | DIV      |  38 |     0.3303 |                            0.2734 |        0.3237 |     0.1053 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)         | No              | Dual-Encoder (text + image)          | raw        | ALL      | 129 |     0.429  |                            0.4068 |        0.4229 |     0.2791 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)         | No              | Dual-Encoder (text + image)          | raw        | ADD      |  68 |     0.4624 |                            0.4391 |        0.4591 |     0.2941 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)         | No              | Dual-Encoder (text + image)          | raw        | PAR      |  23 |     0.5172 |                            0.2045 |        0.4773 |     0.3913 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)         | No              | Dual-Encoder (text + image)          | raw        | DIV      |  38 |     0.3168 |                            0.2337 |        0.3146 |     0.1842 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)         | No              | Blind text (frozen CLS + head)       | std        | ALL      | 129 |     0.3762 |                            0.3255 |        0.4111 |     0.1163 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)         | No              | Blind text (frozen CLS + head)       | std        | ADD      |  68 |     0.4292 |                            0.3827 |        0.4649 |     0.1471 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)         | No              | Blind text (frozen CLS + head)       | std        | PAR      |  23 |     0.2812 |                            0.2063 |        0.3869 |     0.0435 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)         | No              | Blind text (frozen CLS + head)       | std        | DIV      |  38 |     0.3252 |                            0.3316 |        0.3623 |     0.1053 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)         | No              | Dual-Encoder (text + image)          | std        | ALL      | 129 |     0.4329 |                            0.3787 |        0.4548 |     0.1395 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)         | No              | Dual-Encoder (text + image)          | std        | ADD      |  68 |     0.5    |                            0.4537 |        0.514  |     0.1471 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)         | No              | Dual-Encoder (text + image)          | std        | PAR      |  23 |     0.459  |                            0.2287 |        0.5028 |     0.1739 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 1 Base (EMit A + ME)         | No              | Dual-Encoder (text + image)          | std        | DIV      |  38 |     0.3036 |                            0.2424 |        0.3564 |     0.1053 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Fine-tuned model (direct prediction) | -          | ALL      | 129 |     0.5217 |                            0.437  |        0.5165 |     0.3411 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Fine-tuned model (direct prediction) | -          | ADD      |  68 |     0.6272 |                            0.5203 |        0.6208 |     0.4706 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Fine-tuned model (direct prediction) | -          | PAR      |  23 |     0.48   |                            0.2514 |        0.463  |     0.3913 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Fine-tuned model (direct prediction) | -          | DIV      |  38 |     0.3689 |                            0.2796 |        0.3758 |     0.0789 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Blind text (frozen CLS + head)       | raw        | ALL      | 129 |     0.4941 |                            0.4233 |        0.4928 |     0.2868 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Blind text (frozen CLS + head)       | raw        | ADD      |  68 |     0.5824 |                            0.5199 |        0.5804 |     0.3824 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Blind text (frozen CLS + head)       | raw        | PAR      |  23 |     0.5    |                            0.3016 |        0.5139 |     0.3913 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Blind text (frozen CLS + head)       | raw        | DIV      |  38 |     0.3333 |                            0.2783 |        0.3737 |     0.0526 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Dual-Encoder (text + image)          | raw        | ALL      | 129 |     0.506  |                            0.4343 |        0.4913 |     0.2868 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Dual-Encoder (text + image)          | raw        | ADD      |  68 |     0.5568 |                            0.4727 |        0.5382 |     0.3382 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Dual-Encoder (text + image)          | raw        | PAR      |  23 |     0.5424 |                            0.2897 |        0.5724 |     0.3913 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Dual-Encoder (text + image)          | raw        | DIV      |  38 |     0.3918 |                            0.3341 |        0.3907 |     0.1316 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Blind text (frozen CLS + head)       | std        | ALL      | 129 |     0.4624 |                            0.4016 |        0.4959 |     0.2481 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Blind text (frozen CLS + head)       | std        | ADD      |  68 |     0.5464 |                            0.5005 |        0.5692 |     0.3235 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Blind text (frozen CLS + head)       | std        | PAR      |  23 |     0.4839 |                            0.3855 |        0.5966 |     0.2609 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Blind text (frozen CLS + head)       | std        | DIV      |  38 |     0.3103 |                            0.2636 |        0.3683 |     0.1053 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Dual-Encoder (text + image)          | std        | ALL      | 129 |     0.5053 |                            0.4313 |        0.5255 |     0.2946 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Dual-Encoder (text + image)          | std        | ADD      |  68 |     0.596  |                            0.5291 |        0.6166 |     0.3088 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Dual-Encoder (text + image)          | std        | PAR      |  23 |     0.5614 |                            0.3523 |        0.5726 |     0.5217 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Dual-Encoder (text + image)          | std        | DIV      |  38 |     0.3306 |                            0.2995 |        0.3978 |     0.1316 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Fine-tuned model (direct prediction) | -          | ALL      | 129 |     0.6106 |                            0.534  |        0.6114 |     0.4496 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Fine-tuned model (direct prediction) | -          | ADD      |  68 |     0.6509 |                            0.5742 |        0.6532 |     0.5147 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Fine-tuned model (direct prediction) | -          | PAR      |  23 |     0.6275 |                            0.4202 |        0.6272 |     0.5217 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Fine-tuned model (direct prediction) | -          | DIV      |  38 |     0.5347 |                            0.3906 |        0.5468 |     0.2895 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Blind text (frozen CLS + head)       | raw        | ALL      | 129 |     0.5913 |                            0.5285 |        0.6026 |     0.3721 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Blind text (frozen CLS + head)       | raw        | ADD      |  68 |     0.6054 |                            0.5502 |        0.6113 |     0.3676 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Blind text (frozen CLS + head)       | raw        | PAR      |  23 |     0.64   |                            0.4032 |        0.6333 |     0.5652 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Blind text (frozen CLS + head)       | raw        | DIV      |  38 |     0.5455 |                            0.4381 |        0.5837 |     0.2632 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Dual-Encoder (text + image)          | raw        | ALL      | 129 |     0.5633 |                            0.5224 |        0.5762 |     0.3488 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Dual-Encoder (text + image)          | raw        | ADD      |  68 |     0.6235 |                            0.5773 |        0.6375 |     0.3824 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Dual-Encoder (text + image)          | raw        | PAR      |  23 |     0.5417 |                            0.2963 |        0.537  |     0.4348 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Dual-Encoder (text + image)          | raw        | DIV      |  38 |     0.4694 |                            0.3714 |        0.5046 |     0.2368 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Blind text (frozen CLS + head)       | std        | ALL      | 129 |     0.5189 |                            0.4617 |        0.5547 |     0.2868 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Blind text (frozen CLS + head)       | std        | ADD      |  68 |     0.5588 |                            0.5176 |        0.5992 |     0.2941 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Blind text (frozen CLS + head)       | std        | PAR      |  23 |     0.4912 |                            0.281  |        0.5155 |     0.3043 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Blind text (frozen CLS + head)       | std        | DIV      |  38 |     0.4587 |                            0.3909 |        0.526  |     0.2632 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Dual-Encoder (text + image)          | std        | ALL      | 129 |     0.5435 |                            0.4863 |        0.5659 |     0.3101 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Dual-Encoder (text + image)          | std        | ADD      |  68 |     0.5816 |                            0.5377 |        0.6137 |     0.3235 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Dual-Encoder (text + image)          | std        | PAR      |  23 |     0.5172 |                            0.3299 |        0.5624 |     0.3913 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Dual-Encoder (text + image)          | std        | DIV      |  38 |     0.4912 |                            0.344  |        0.5214 |     0.2368 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Fine-tuned model (direct prediction) | -          | ALL      | 129 |     0.5397 |                            0.479  |        0.5485 |     0.4031 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Fine-tuned model (direct prediction) | -          | ADD      |  68 |     0.5868 |                            0.5202 |        0.5944 |     0.4265 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Fine-tuned model (direct prediction) | -          | PAR      |  23 |     0.375  |                            0.1984 |        0.3889 |     0.3913 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Fine-tuned model (direct prediction) | -          | DIV      |  38 |     0.54   |                            0.43   |        0.5561 |     0.3684 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Blind text (frozen CLS + head)       | raw        | ALL      | 129 |     0.5183 |                            0.4343 |        0.5456 |     0.3488 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Blind text (frozen CLS + head)       | raw        | ADD      |  68 |     0.5806 |                            0.48   |        0.5858 |     0.4118 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Blind text (frozen CLS + head)       | raw        | PAR      |  23 |     0.3934 |                            0.262  |        0.4664 |     0.3478 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Blind text (frozen CLS + head)       | raw        | DIV      |  38 |     0.4815 |                            0.3594 |        0.5452 |     0.2368 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Dual-Encoder (text + image)          | raw        | ALL      | 129 |     0.5535 |                            0.4967 |        0.5477 |     0.3566 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Dual-Encoder (text + image)          | raw        | ADD      |  68 |     0.5614 |                            0.4949 |        0.54   |     0.3971 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Dual-Encoder (text + image)          | raw        | PAR      |  23 |     0.6296 |                            0.3555 |        0.6288 |     0.5217 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Dual-Encoder (text + image)          | raw        | DIV      |  38 |     0.4946 |                            0.3265 |        0.5119 |     0.1842 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Blind text (frozen CLS + head)       | std        | ALL      | 129 |     0.5213 |                            0.455  |        0.5584 |     0.3488 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Blind text (frozen CLS + head)       | std        | ADD      |  68 |     0.5373 |                            0.4881 |        0.574  |     0.3676 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Blind text (frozen CLS + head)       | std        | PAR      |  23 |     0.5172 |                            0.2401 |        0.5027 |     0.4348 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Blind text (frozen CLS + head)       | std        | DIV      |  38 |     0.4957 |                            0.454  |        0.5559 |     0.2632 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Dual-Encoder (text + image)          | std        | ALL      | 129 |     0.5215 |                            0.4537 |        0.5569 |     0.3101 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Dual-Encoder (text + image)          | std        | ADD      |  68 |     0.5572 |                            0.5077 |        0.5926 |     0.2941 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Dual-Encoder (text + image)          | std        | PAR      |  23 |     0.5    |                            0.2991 |        0.5493 |     0.4783 |
| Emotion Detection (10-Class Multi-Label)           | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Dual-Encoder (text + image)          | std        | DIV      |  38 |     0.4685 |                            0.4246 |        0.5312 |     0.2368 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)        | No              | Fine-tuned model (direct prediction) | -          | ALL      | 129 |     0.7364 |                            0.4224 |        0.7225 |     0.7364 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)        | No              | Fine-tuned model (direct prediction) | -          | ADD      |  68 |     0.8235 |                            0.5194 |        0.8515 |     0.8235 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)        | No              | Fine-tuned model (direct prediction) | -          | PAR      |  23 |     0.913  |                            0.9545 |        0.9545 |     0.913  |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)        | No              | Fine-tuned model (direct prediction) | -          | DIV      |  38 |     0.4737 |                            0.2718 |        0.3976 |     0.4737 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)        | No              | Blind text (frozen CLS + head)       | raw        | ALL      | 129 |     0.3566 |                            0.2617 |        0.4617 |     0.3566 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)        | No              | Blind text (frozen CLS + head)       | raw        | ADD      |  68 |     0.3824 |                            0.1905 |        0.5126 |     0.3824 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)        | No              | Blind text (frozen CLS + head)       | raw        | PAR      |  23 |     0.3043 |                            0.4667 |        0.4667 |     0.3043 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)        | No              | Blind text (frozen CLS + head)       | raw        | DIV      |  38 |     0.3421 |                            0.3346 |        0.3859 |     0.3421 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)        | No              | Dual-Encoder (text + image)          | raw        | ALL      | 129 |     0.7907 |                            0.437  |        0.7686 |     0.7907 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)        | No              | Dual-Encoder (text + image)          | raw        | ADD      |  68 |     0.8529 |                            0.3069 |        0.8259 |     0.8529 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)        | No              | Dual-Encoder (text + image)          | raw        | PAR      |  23 |     0.913  |                            0.9545 |        0.9545 |     0.913  |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)        | No              | Dual-Encoder (text + image)          | raw        | DIV      |  38 |     0.6053 |                            0.4177 |        0.5694 |     0.6053 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)        | No              | Blind text (frozen CLS + head)       | std        | ALL      | 129 |     0.7597 |                            0.433  |        0.7545 |     0.7597 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)        | No              | Blind text (frozen CLS + head)       | std        | ADD      |  68 |     0.7941 |                            0.3457 |        0.8015 |     0.7941 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)        | No              | Blind text (frozen CLS + head)       | std        | PAR      |  23 |     1      |                            1      |        1      |     1      |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)        | No              | Blind text (frozen CLS + head)       | std        | DIV      |  38 |     0.5526 |                            0.4019 |        0.5443 |     0.5526 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)        | No              | Dual-Encoder (text + image)          | std        | ALL      | 129 |     0.8062 |                            0.4458 |        0.782  |     0.8062 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)        | No              | Dual-Encoder (text + image)          | std        | ADD      |  68 |     0.8676 |                            0.3097 |        0.8335 |     0.8676 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)        | No              | Dual-Encoder (text + image)          | std        | PAR      |  23 |     0.913  |                            0.9545 |        0.9545 |     0.913  |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 1 Base (IronITA + ME)        | No              | Dual-Encoder (text + image)          | std        | DIV      |  38 |     0.6316 |                            0.4329 |        0.5933 |     0.6316 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Fine-tuned model (direct prediction) | -          | ALL      | 129 |     0.7829 |                            0.5316 |        0.7858 |     0.7829 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Fine-tuned model (direct prediction) | -          | ADD      |  68 |     0.8382 |                            0.567  |        0.8599 |     0.8382 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Fine-tuned model (direct prediction) | -          | PAR      |  23 |     0.913  |                            0.9545 |        0.9545 |     0.913  |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Fine-tuned model (direct prediction) | -          | DIV      |  38 |     0.6053 |                            0.4212 |        0.5734 |     0.6053 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Blind text (frozen CLS + head)       | raw        | ALL      | 129 |     0.7442 |                            0.478  |        0.7606 |     0.7442 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Blind text (frozen CLS + head)       | raw        | ADD      |  68 |     0.8382 |                            0.5273 |        0.8554 |     0.8382 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Blind text (frozen CLS + head)       | raw        | PAR      |  23 |     0.913  |                            0.9545 |        0.9545 |     0.913  |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Blind text (frozen CLS + head)       | raw        | DIV      |  38 |     0.4737 |                            0.4008 |        0.4825 |     0.4737 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Dual-Encoder (text + image)          | raw        | ALL      | 129 |     0.7829 |                            0.5279 |        0.7837 |     0.7829 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Dual-Encoder (text + image)          | raw        | ADD      |  68 |     0.8235 |                            0.5306 |        0.8472 |     0.8235 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Dual-Encoder (text + image)          | raw        | PAR      |  23 |     0.913  |                            0.9545 |        0.9545 |     0.913  |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Dual-Encoder (text + image)          | raw        | DIV      |  38 |     0.6316 |                            0.4395 |        0.5959 |     0.6316 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Blind text (frozen CLS + head)       | std        | ALL      | 129 |     0.7287 |                            0.5084 |        0.7484 |     0.7287 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Blind text (frozen CLS + head)       | std        | ADD      |  68 |     0.7941 |                            0.4957 |        0.826  |     0.7941 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Blind text (frozen CLS + head)       | std        | PAR      |  23 |     0.8696 |                            0.9302 |        0.9302 |     0.8696 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Blind text (frozen CLS + head)       | std        | DIV      |  38 |     0.5263 |                            0.4559 |        0.5259 |     0.5263 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Dual-Encoder (text + image)          | std        | ALL      | 129 |     0.7597 |                            0.5678 |        0.7731 |     0.7597 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Dual-Encoder (text + image)          | std        | ADD      |  68 |     0.8235 |                            0.5575 |        0.8505 |     0.8235 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Dual-Encoder (text + image)          | std        | PAR      |  23 |     0.8696 |                            0.9302 |        0.9302 |     0.8696 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Raw (Step 1 -> ME + MM)    | No              | Dual-Encoder (text + image)          | std        | DIV      |  38 |     0.5789 |                            0.5098 |        0.5729 |     0.5789 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Fine-tuned model (direct prediction) | -          | ALL      | 129 |     0.7752 |                            0.4538 |        0.7717 |     0.7752 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Fine-tuned model (direct prediction) | -          | ADD      |  68 |     0.8529 |                            0.4298 |        0.8517 |     0.8529 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Fine-tuned model (direct prediction) | -          | PAR      |  23 |     0.8696 |                            0.9302 |        0.9302 |     0.8696 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Fine-tuned model (direct prediction) | -          | DIV      |  38 |     0.5789 |                            0.4102 |        0.556  |     0.5789 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Blind text (frozen CLS + head)       | raw        | ALL      | 129 |     0.7519 |                            0.4876 |        0.7648 |     0.7519 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Blind text (frozen CLS + head)       | raw        | ADD      |  68 |     0.8382 |                            0.5363 |        0.8625 |     0.8382 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Blind text (frozen CLS + head)       | raw        | PAR      |  23 |     0.8696 |                            0.9302 |        0.9302 |     0.8696 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Blind text (frozen CLS + head)       | raw        | DIV      |  38 |     0.5263 |                            0.3772 |        0.514  |     0.5263 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Dual-Encoder (text + image)          | raw        | ALL      | 129 |     0.7674 |                            0.4484 |        0.7657 |     0.7674 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Dual-Encoder (text + image)          | raw        | ADD      |  68 |     0.8382 |                            0.4081 |        0.8404 |     0.8382 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Dual-Encoder (text + image)          | raw        | PAR      |  23 |     0.8696 |                            0.9302 |        0.9302 |     0.8696 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Dual-Encoder (text + image)          | raw        | DIV      |  38 |     0.5789 |                            0.4167 |        0.5632 |     0.5789 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Blind text (frozen CLS + head)       | std        | ALL      | 129 |     0.6589 |                            0.4279 |        0.6982 |     0.6589 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Blind text (frozen CLS + head)       | std        | ADD      |  68 |     0.7206 |                            0.4253 |        0.7799 |     0.7206 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Blind text (frozen CLS + head)       | std        | PAR      |  23 |     0.8261 |                            0.9048 |        0.9048 |     0.8261 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Blind text (frozen CLS + head)       | std        | DIV      |  38 |     0.4474 |                            0.3333 |        0.4474 |     0.4474 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Dual-Encoder (text + image)          | std        | ALL      | 129 |     0.8062 |                            0.5424 |        0.8036 |     0.8062 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Dual-Encoder (text + image)          | std        | ADD      |  68 |     0.8971 |                            0.6169 |        0.8985 |     0.8971 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Dual-Encoder (text + image)          | std        | PAR      |  23 |     0.8696 |                            0.9302 |        0.9302 |     0.8696 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (no rel) | Dual-Encoder (text + image)          | std        | DIV      |  38 |     0.6053 |                            0.4276 |        0.5805 |     0.6053 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Fine-tuned model (direct prediction) | -          | ALL      | 129 |     0.7364 |                            0.4153 |        0.7377 |     0.7364 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Fine-tuned model (direct prediction) | -          | ADD      |  68 |     0.7647 |                            0.3639 |        0.7931 |     0.7647 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Fine-tuned model (direct prediction) | -          | PAR      |  23 |     0.8696 |                            0.9302 |        0.9302 |     0.8696 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Fine-tuned model (direct prediction) | -          | DIV      |  38 |     0.6053 |                            0.4165 |        0.5705 |     0.6053 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Blind text (frozen CLS + head)       | raw        | ALL      | 129 |     0.7519 |                            0.4769 |        0.7549 |     0.7519 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Blind text (frozen CLS + head)       | raw        | ADD      |  68 |     0.8088 |                            0.4661 |        0.8303 |     0.8088 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Blind text (frozen CLS + head)       | raw        | PAR      |  23 |     0.8696 |                            0.9302 |        0.9302 |     0.8696 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Blind text (frozen CLS + head)       | raw        | DIV      |  38 |     0.5789 |                            0.3939 |        0.5431 |     0.5789 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Dual-Encoder (text + image)          | raw        | ALL      | 129 |     0.7829 |                            0.4481 |        0.7737 |     0.7829 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Dual-Encoder (text + image)          | raw        | ADD      |  68 |     0.8382 |                            0.4167 |        0.8419 |     0.8382 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Dual-Encoder (text + image)          | raw        | PAR      |  23 |     0.8696 |                            0.9302 |        0.9302 |     0.8696 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Dual-Encoder (text + image)          | raw        | DIV      |  38 |     0.6316 |                            0.4329 |        0.5933 |     0.6316 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Blind text (frozen CLS + head)       | std        | ALL      | 129 |     0.7209 |                            0.4503 |        0.7335 |     0.7209 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Blind text (frozen CLS + head)       | std        | ADD      |  68 |     0.7794 |                            0.432  |        0.8096 |     0.7794 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Blind text (frozen CLS + head)       | std        | PAR      |  23 |     0.8696 |                            0.9302 |        0.9302 |     0.8696 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Blind text (frozen CLS + head)       | std        | DIV      |  38 |     0.5263 |                            0.373  |        0.51   |     0.5263 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Dual-Encoder (text + image)          | std        | ALL      | 129 |     0.7209 |                            0.4433 |        0.7413 |     0.7209 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Dual-Encoder (text + image)          | std        | ADD      |  68 |     0.7794 |                            0.4278 |        0.8099 |     0.7794 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Dual-Encoder (text + image)          | std        | PAR      |  23 |     0.8261 |                            0.9048 |        0.9048 |     0.8261 |
| Irony & Sarcasm Detection (3-Class Single-Label)   | Multimodal split  | Step 2 Oracle (Step 1 -> ME + MM) | Oracle (+ rel)  | Dual-Encoder (text + image)          | std        | DIV      |  38 |     0.5526 |                            0.3992 |        0.5514 |     0.5526 |
| Semiotic Relation Detection (3-Class Single-Label) | Textual (mm_test) | Stage A: MM Train Raw             | No              | UmBERTo (fine-tuned)                 | -          | ALL      | 129 |     0.6512 |                            0.469  |        0.5844 |     0.6512 |
| Semiotic Relation Detection (3-Class Single-Label) | Textual (mm_test) | Stage A: MM Train Raw             | No              | UmBERTo (fine-tuned)                 | -          | ADD      |  68 |     0.8971 |                            0.9457 |        0.9457 |     0.8971 |
| Semiotic Relation Detection (3-Class Single-Label) | Textual (mm_test) | Stage A: MM Train Raw             | No              | UmBERTo (fine-tuned)                 | -          | PAR      |  23 |     0      |                            0      |        0      |     0      |
| Semiotic Relation Detection (3-Class Single-Label) | Textual (mm_test) | Stage A: MM Train Raw             | No              | UmBERTo (fine-tuned)                 | -          | DIV      |  38 |     0.6053 |                            0.7541 |        0.7541 |     0.6053 |
| Semiotic Relation Detection (3-Class Single-Label) | Textual (mm_test) | Stage B: MM Train Oracle          | Oracle          | UmBERTo (fine-tuned)                 | -          | ALL      | 129 |     0.6279 |                            0.5429 |        0.6076 |     0.6279 |
| Semiotic Relation Detection (3-Class Single-Label) | Textual (mm_test) | Stage B: MM Train Oracle          | Oracle          | UmBERTo (fine-tuned)                 | -          | ADD      |  68 |     0.7941 |                            0.8852 |        0.8852 |     0.7941 |
| Semiotic Relation Detection (3-Class Single-Label) | Textual (mm_test) | Stage B: MM Train Oracle          | Oracle          | UmBERTo (fine-tuned)                 | -          | PAR      |  23 |     0.2174 |                            0.3571 |        0.3571 |     0.2174 |
| Semiotic Relation Detection (3-Class Single-Label) | Textual (mm_test) | Stage B: MM Train Oracle          | Oracle          | UmBERTo (fine-tuned)                 | -          | DIV      |  38 |     0.5789 |                            0.7333 |        0.7333 |     0.5789 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw             | No              | Fine-tuned model (direct prediction) | -          | ALL      | 129 |     0.6512 |                            0.469  |        0.5844 |     0.6512 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw             | No              | Fine-tuned model (direct prediction) | -          | ADD      |  68 |     0.8971 |                            0.9457 |        0.9457 |     0.8971 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw             | No              | Fine-tuned model (direct prediction) | -          | PAR      |  23 |     0      |                            0      |        0      |     0      |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw             | No              | Fine-tuned model (direct prediction) | -          | DIV      |  38 |     0.6053 |                            0.7541 |        0.7541 |     0.6053 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw             | No              | Blind text (frozen CLS + head)       | raw        | ALL      | 129 |     0.5969 |                            0.5063 |        0.57   |     0.5969 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw             | No              | Blind text (frozen CLS + head)       | raw        | ADD      |  68 |     0.8088 |                            0.8943 |        0.8943 |     0.8088 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw             | No              | Blind text (frozen CLS + head)       | raw        | PAR      |  23 |     0.2174 |                            0.3571 |        0.3571 |     0.2174 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw             | No              | Blind text (frozen CLS + head)       | raw        | DIV      |  38 |     0.4474 |                            0.6182 |        0.6182 |     0.4474 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw             | No              | Dual-Encoder (text + image)          | raw        | ALL      | 129 |     0.6667 |                            0.5729 |        0.6406 |     0.6667 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw             | No              | Dual-Encoder (text + image)          | raw        | ADD      |  68 |     0.8529 |                            0.9206 |        0.9206 |     0.8529 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw             | No              | Dual-Encoder (text + image)          | raw        | PAR      |  23 |     0.2174 |                            0.3571 |        0.3571 |     0.2174 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw             | No              | Dual-Encoder (text + image)          | raw        | DIV      |  38 |     0.6053 |                            0.7541 |        0.7541 |     0.6053 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw             | No              | Blind text (frozen CLS + head)       | std        | ALL      | 129 |     0.6667 |                            0.6011 |        0.6537 |     0.6667 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw             | No              | Blind text (frozen CLS + head)       | std        | ADD      |  68 |     0.7794 |                            0.876  |        0.876  |     0.7794 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw             | No              | Blind text (frozen CLS + head)       | std        | PAR      |  23 |     0.3043 |                            0.4667 |        0.4667 |     0.3043 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw             | No              | Blind text (frozen CLS + head)       | std        | DIV      |  38 |     0.6842 |                            0.8125 |        0.8125 |     0.6842 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw             | No              | Dual-Encoder (text + image)          | std        | ALL      | 129 |     0.6512 |                            0.6175 |        0.6501 |     0.6512 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw             | No              | Dual-Encoder (text + image)          | std        | ADD      |  68 |     0.7353 |                            0.8475 |        0.8475 |     0.7353 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw             | No              | Dual-Encoder (text + image)          | std        | PAR      |  23 |     0.5217 |                            0.6857 |        0.6857 |     0.5217 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage A: MM Train Raw             | No              | Dual-Encoder (text + image)          | std        | DIV      |  38 |     0.5789 |                            0.7333 |        0.7333 |     0.5789 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle          | Oracle          | Fine-tuned model (direct prediction) | -          | ALL      | 129 |     0.6279 |                            0.5429 |        0.6076 |     0.6279 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle          | Oracle          | Fine-tuned model (direct prediction) | -          | ADD      |  68 |     0.7941 |                            0.8852 |        0.8852 |     0.7941 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle          | Oracle          | Fine-tuned model (direct prediction) | -          | PAR      |  23 |     0.2174 |                            0.3571 |        0.3571 |     0.2174 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle          | Oracle          | Fine-tuned model (direct prediction) | -          | DIV      |  38 |     0.5789 |                            0.7333 |        0.7333 |     0.5789 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle          | Oracle          | Blind text (frozen CLS + head)       | raw        | ALL      | 129 |     0.6434 |                            0.5408 |        0.6114 |     0.6434 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle          | Oracle          | Blind text (frozen CLS + head)       | raw        | ADD      |  68 |     0.8382 |                            0.912  |        0.912  |     0.8382 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle          | Oracle          | Blind text (frozen CLS + head)       | raw        | PAR      |  23 |     0.1739 |                            0.2963 |        0.2963 |     0.1739 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle          | Oracle          | Blind text (frozen CLS + head)       | raw        | DIV      |  38 |     0.5789 |                            0.7333 |        0.7333 |     0.5789 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle          | Oracle          | Dual-Encoder (text + image)          | raw        | ALL      | 129 |     0.6512 |                            0.6242 |        0.6516 |     0.6512 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle          | Oracle          | Dual-Encoder (text + image)          | raw        | ADD      |  68 |     0.7059 |                            0.8276 |        0.8276 |     0.7059 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle          | Oracle          | Dual-Encoder (text + image)          | raw        | PAR      |  23 |     0.5652 |                            0.7222 |        0.7222 |     0.5652 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle          | Oracle          | Dual-Encoder (text + image)          | raw        | DIV      |  38 |     0.6053 |                            0.7541 |        0.7541 |     0.6053 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle          | Oracle          | Blind text (frozen CLS + head)       | std        | ALL      | 129 |     0.6434 |                            0.58   |        0.629  |     0.6434 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle          | Oracle          | Blind text (frozen CLS + head)       | std        | ADD      |  68 |     0.7794 |                            0.876  |        0.876  |     0.7794 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle          | Oracle          | Blind text (frozen CLS + head)       | std        | PAR      |  23 |     0.3043 |                            0.4667 |        0.4667 |     0.3043 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle          | Oracle          | Blind text (frozen CLS + head)       | std        | DIV      |  38 |     0.6053 |                            0.7541 |        0.7541 |     0.6053 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle          | Oracle          | Dual-Encoder (text + image)          | std        | ALL      | 129 |     0.6357 |                            0.6116 |        0.6346 |     0.6357 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle          | Oracle          | Dual-Encoder (text + image)          | std        | ADD      |  68 |     0.6912 |                            0.8174 |        0.8174 |     0.6912 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle          | Oracle          | Dual-Encoder (text + image)          | std        | PAR      |  23 |     0.5217 |                            0.6857 |        0.6857 |     0.5217 |
| Semiotic Relation Detection (3-Class Single-Label) | Multimodal split  | Stage B: MM Train Oracle          | Oracle          | Dual-Encoder (text + image)          | std        | DIV      |  38 |     0.6053 |                            0.7541 |        0.7541 |     0.6053 |

---

## 3. Per-Task Classification Reports

### Textual Emotion Detection
```
=== Step 1: EMit A + ME Textual (No Correlations) ===
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


=== Step 2 Raw Ablation: Step 1 -> ME Textual + MM Train (No Correlations) ===
               precision    recall  f1-score   support

        Amore     0.6182    0.6939    0.6538        49
Anticipazione     0.4667    0.5600    0.5091        25
     Disgusto     0.5513    0.5811    0.5658        74
      Fiducia     0.5579    0.6386    0.5955        83
        Gioia     0.5152    0.5667    0.5397        90
       Neutro     0.4130    0.3654    0.3878        52
        Paura     0.8000    0.3333    0.4706        12
       Rabbia     0.4400    0.4074    0.4231        27
     Sorpresa     0.4831    0.4831    0.4831        89
    Tristezza     0.5263    0.7143    0.6061        42

    micro avg     0.5216    0.5562    0.5383       543
    macro avg     0.5372    0.5344    0.5235       543
 weighted avg     0.5221    0.5562    0.5347       543
  samples avg     0.5179    0.5276    0.5020       543


=== Step 2 Oracle: Step 1 -> ME Textual + MM Train (With Oracle Correlations) ===
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

### Multimodal Emotion Dual-Encoder
```
=== Step 1 Base (EMit A + ME) | prefix=No | Fine-tuned model (direct prediction) | features=- ===
               precision    recall  f1-score   support

        Amore     0.3684    0.6364    0.4667        11
Anticipazione     0.4706    0.8000    0.5926        10
     Disgusto     0.8000    0.4211    0.5517        19
      Fiducia     0.5588    0.4524    0.5000        42
        Gioia     0.3673    0.6207    0.4615        29
       Neutro     0.2963    0.4706    0.3636        17
        Paura     0.0000    0.0000    0.0000         1
       Rabbia     0.5000    0.1667    0.2500         6
     Sorpresa     0.3000    0.1875    0.2308        16
    Tristezza     0.2500    0.5000    0.3333         2

    micro avg     0.4244    0.4771    0.4492       153
    macro avg     0.3911    0.4255    0.3750       153
 weighted avg     0.4668    0.4771    0.4442       153
  samples avg     0.4367    0.4703    0.4370       153


=== Step 1 Base (EMit A + ME) | prefix=No | Blind text (frozen CLS + head) | features=raw ===
               precision    recall  f1-score   support

        Amore     0.3200    0.7273    0.4444        11
Anticipazione     0.4500    0.9000    0.6000        10
     Disgusto     0.4118    0.3684    0.3889        19
      Fiducia     0.6042    0.6905    0.6444        42
        Gioia     0.3469    0.5862    0.4359        29
       Neutro     0.2000    0.1765    0.1875        17
        Paura     0.1000    1.0000    0.1818         1
       Rabbia     0.2000    0.1667    0.1818         6
     Sorpresa     0.0000    0.0000    0.0000        16
    Tristezza     0.3333    1.0000    0.5000         2

    micro avg     0.3775    0.5033    0.4314       153
    macro avg     0.2966    0.5616    0.3565       153
 weighted avg     0.3702    0.5033    0.4147       153
  samples avg     0.3986    0.4948    0.4230       153


=== Step 1 Base (EMit A + ME) | prefix=No | Dual-Encoder (text + image) | features=raw ===
               precision    recall  f1-score   support

        Amore     0.2000    0.1818    0.1905        11
Anticipazione     0.5000    0.5000    0.5000        10
     Disgusto     0.5385    0.3684    0.4375        19
      Fiducia     0.5172    0.7143    0.6000        42
        Gioia     0.3488    0.5172    0.4167        29
       Neutro     0.2500    0.2941    0.2703        17
        Paura     0.3333    1.0000    0.5000         1
       Rabbia     0.2727    0.5000    0.3529         6
     Sorpresa     0.2105    0.2500    0.2286        16
    Tristezza     0.4000    1.0000    0.5714         2

    micro avg     0.3854    0.4837    0.4290       153
    macro avg     0.3571    0.5326    0.4068       153
 weighted avg     0.3899    0.4837    0.4229       153
  samples avg     0.3928    0.4755    0.4113       153


=== Step 1 Base (EMit A + ME) | prefix=No | Blind text (frozen CLS + head) | features=std ===
               precision    recall  f1-score   support

        Amore     0.2059    0.6364    0.3111        11
Anticipazione     0.1538    0.4000    0.2222        10
     Disgusto     0.3448    0.5263    0.4167        19
      Fiducia     0.5682    0.5952    0.5814        42
        Gioia     0.4412    0.5172    0.4762        29
       Neutro     0.3200    0.4706    0.3810        17
        Paura     0.0526    1.0000    0.1000         1
       Rabbia     0.2632    0.8333    0.4000         6
     Sorpresa     0.0833    0.1250    0.1000        16
    Tristezza     0.1538    1.0000    0.2667         2

    micro avg     0.2959    0.5163    0.3762       153
    macro avg     0.2587    0.6104    0.3255       153
 weighted avg     0.3642    0.5163    0.4111       153
  samples avg     0.3012    0.5155    0.3663       153


=== Step 1 Base (EMit A + ME) | prefix=No | Dual-Encoder (text + image) | features=std ===
               precision    recall  f1-score   support

        Amore     0.2222    0.5455    0.3158        11
Anticipazione     0.4118    0.7000    0.5185        10
     Disgusto     0.3529    0.3158    0.3333        19
      Fiducia     0.7073    0.6905    0.6988        42
        Gioia     0.4286    0.5172    0.4688        29
       Neutro     0.2500    0.4118    0.3111        17
        Paura     0.1000    1.0000    0.1818         1
       Rabbia     0.0909    0.1667    0.1176         6
     Sorpresa     0.2381    0.3125    0.2703        16
    Tristezza     0.4000    1.0000    0.5714         2

    micro avg     0.3726    0.5163    0.4329       153
    macro avg     0.3202    0.5660    0.3787       153
 weighted avg     0.4242    0.5163    0.4548       153
  samples avg     0.3366    0.5026    0.3858       153


=== Step 2 Raw (Step 1 -> ME + MM) | prefix=No | Fine-tuned model (direct prediction) | features=- ===
               precision    recall  f1-score   support

        Amore     0.5385    0.6364    0.5833        11
Anticipazione     0.4444    0.8000    0.5714        10
     Disgusto     0.5000    0.4211    0.4571        19
      Fiducia     0.7105    0.6429    0.6750        42
        Gioia     0.4286    0.6207    0.5070        29
       Neutro     0.4000    0.4706    0.4324        17
        Paura     0.0000    0.0000    0.0000         1
       Rabbia     0.5000    0.1667    0.2500         6
     Sorpresa     0.3333    0.3125    0.3226        16
    Tristezza     0.4000    1.0000    0.5714         2

    micro avg     0.4970    0.5490    0.5217       153
    macro avg     0.4255    0.5071    0.4370       153
 weighted avg     0.5103    0.5490    0.5165       153
  samples avg     0.5058    0.5517    0.5075       153


=== Step 2 Raw (Step 1 -> ME + MM) | prefix=No | Blind text (frozen CLS + head) | features=raw ===
               precision    recall  f1-score   support

        Amore     0.5455    0.5455    0.5455        11
Anticipazione     0.3478    0.8000    0.4848        10
     Disgusto     0.6250    0.2632    0.3704        19
      Fiducia     0.6122    0.7143    0.6593        42
        Gioia     0.4524    0.6552    0.5352        29
       Neutro     0.4545    0.2941    0.3571        17
        Paura     0.2500    1.0000    0.4000         1
       Rabbia     0.2308    0.5000    0.3158         6
     Sorpresa     0.3158    0.3750    0.3429        16
    Tristezza     0.1429    0.5000    0.2222         2

    micro avg     0.4492    0.5490    0.4941       153
    macro avg     0.3977    0.5647    0.4233       153
 weighted avg     0.4895    0.5490    0.4928       153
  samples avg     0.4477    0.5258    0.4643       153


=== Step 2 Raw (Step 1 -> ME + MM) | prefix=No | Dual-Encoder (text + image) | features=raw ===
               precision    recall  f1-score   support

        Amore     0.4000    0.5455    0.4615        11
Anticipazione     0.6364    0.7000    0.6667        10
     Disgusto     0.4615    0.3158    0.3750        19
      Fiducia     0.6316    0.8571    0.7273        42
        Gioia     0.4872    0.6552    0.5588        29
       Neutro     0.1538    0.2353    0.1860        17
        Paura     0.3333    1.0000    0.5000         1
       Rabbia     0.5000    0.3333    0.4000         6
     Sorpresa     0.3333    0.1250    0.1818        16
    Tristezza     0.2000    0.5000    0.2857         2

    micro avg     0.4693    0.5490    0.5060       153
    macro avg     0.4137    0.5267    0.4343       153
 weighted avg     0.4697    0.5490    0.4913       153
  samples avg     0.4596    0.5452    0.4762       153


=== Step 2 Raw (Step 1 -> ME + MM) | prefix=No | Blind text (frozen CLS + head) | features=std ===
               precision    recall  f1-score   support

        Amore     0.3600    0.8182    0.5000        11
Anticipazione     0.3182    0.7000    0.4375        10
     Disgusto     0.5000    0.3158    0.3871        19
      Fiducia     0.6327    0.7381    0.6813        42
        Gioia     0.5714    0.5517    0.5614        29
       Neutro     0.3500    0.4118    0.3784        17
        Paura     0.0333    1.0000    0.0645         1
       Rabbia     0.3000    0.5000    0.3750         6
     Sorpresa     0.2857    0.2500    0.2667        16
    Tristezza     0.2222    1.0000    0.3636         2

    micro avg     0.3927    0.5621    0.4624       153
    macro avg     0.3574    0.6286    0.4016       153
 weighted avg     0.4744    0.5621    0.4959       153
  samples avg     0.4152    0.5491    0.4498       153


=== Step 2 Raw (Step 1 -> ME + MM) | prefix=No | Dual-Encoder (text + image) | features=std ===
               precision    recall  f1-score   support

        Amore     0.3810    0.7273    0.5000        11
Anticipazione     0.5294    0.9000    0.6667        10
     Disgusto     0.4000    0.3158    0.3529        19
      Fiducia     0.7045    0.7381    0.7209        42
        Gioia     0.5676    0.7241    0.6364        29
       Neutro     0.2703    0.5882    0.3704        17
        Paura     0.0667    1.0000    0.1250         1
       Rabbia     0.2857    0.6667    0.4000         6
     Sorpresa     0.2308    0.1875    0.2069        16
    Tristezza     0.2000    1.0000    0.3333         2

    micro avg     0.4260    0.6209    0.5053       153
    macro avg     0.3636    0.6848    0.4313       153
 weighted avg     0.4811    0.6209    0.5255       153
  samples avg     0.5084    0.6357    0.5367       153


=== Step 2 Oracle (Step 1 -> ME + MM) | prefix=Oracle (no rel) | Fine-tuned model (direct prediction) | features=- ===
               precision    recall  f1-score   support

        Amore     0.4211    0.7273    0.5333        11
Anticipazione     0.6154    0.8000    0.6957        10
     Disgusto     0.7368    0.7368    0.7368        19
      Fiducia     0.7073    0.6905    0.6988        42
        Gioia     0.5938    0.6552    0.6230        29
       Neutro     0.4444    0.4706    0.4571        17
        Paura     0.0000    0.0000    0.0000         1
       Rabbia     0.7500    0.5000    0.6000         6
     Sorpresa     0.4118    0.4375    0.4242        16
    Tristezza     0.4000    1.0000    0.5714         2

    micro avg     0.5833    0.6405    0.6106       153
    macro avg     0.5081    0.6018    0.5340       153
 weighted avg     0.5958    0.6405    0.6114       153
  samples avg     0.6118    0.6525    0.6121       153


=== Step 2 Oracle (Step 1 -> ME + MM) | prefix=Oracle (no rel) | Blind text (frozen CLS + head) | features=raw ===
               precision    recall  f1-score   support

        Amore     0.4444    0.7273    0.5517        11
Anticipazione     0.7500    0.6000    0.6667        10
     Disgusto     0.5600    0.7368    0.6364        19
      Fiducia     0.6863    0.8333    0.7527        42
        Gioia     0.6875    0.7586    0.7213        29
       Neutro     0.4167    0.2941    0.3448        17
        Paura     0.5000    1.0000    0.6667         1
       Rabbia     0.1667    0.5000    0.2500         6
     Sorpresa     0.4000    0.3750    0.3871        16
    Tristezza     0.1818    1.0000    0.3077         2

    micro avg     0.5312    0.6667    0.5913       153
    macro avg     0.4793    0.6825    0.5285       153
 weighted avg     0.5695    0.6667    0.6026       153
  samples avg     0.5322    0.6486    0.5623       153


=== Step 2 Oracle (Step 1 -> ME + MM) | prefix=Oracle (no rel) | Dual-Encoder (text + image) | features=raw ===
               precision    recall  f1-score   support

        Amore     0.3889    0.6364    0.4828        11
Anticipazione     0.6154    0.8000    0.6957        10
     Disgusto     0.7500    0.6316    0.6857        19
      Fiducia     0.7222    0.6190    0.6667        42
        Gioia     0.6786    0.6552    0.6667        29
       Neutro     0.4667    0.4118    0.4375        17
        Paura     0.2500    1.0000    0.4000         1
       Rabbia     0.1429    0.3333    0.2000         6
     Sorpresa     0.3333    0.3125    0.3226        16
    Tristezza     0.5000    1.0000    0.6667         2

    micro avg     0.5460    0.5817    0.5633       153
    macro avg     0.4848    0.6400    0.5224       153
 weighted avg     0.5887    0.5817    0.5762       153
  samples avg     0.5161    0.5711    0.5230       153


=== Step 2 Oracle (Step 1 -> ME + MM) | prefix=Oracle (no rel) | Blind text (frozen CLS + head) | features=std ===
               precision    recall  f1-score   support

        Amore     0.5714    0.3636    0.4444        11
Anticipazione     0.3500    0.7000    0.4667        10
     Disgusto     0.5000    0.6842    0.5778        19
      Fiducia     0.5957    0.6667    0.6292        42
        Gioia     0.6364    0.7241    0.6774        29
       Neutro     0.3846    0.5882    0.4651        17
        Paura     0.0385    1.0000    0.0741         1
       Rabbia     0.3333    0.5000    0.4000         6
     Sorpresa     0.4375    0.4375    0.4375        16
    Tristezza     0.2857    1.0000    0.4444         2

    micro avg     0.4424    0.6275    0.5189       153
    macro avg     0.4133    0.6664    0.4617       153
 weighted avg     0.5158    0.6275    0.5547       153
  samples avg     0.4955    0.6331    0.5313       153


=== Step 2 Oracle (Step 1 -> ME + MM) | prefix=Oracle (no rel) | Dual-Encoder (text + image) | features=std ===
               precision    recall  f1-score   support

        Amore     0.2857    0.7273    0.4103        11
Anticipazione     0.6154    0.8000    0.6957        10
     Disgusto     0.5000    0.6842    0.5778        19
      Fiducia     0.6591    0.6905    0.6744        42
        Gioia     0.5714    0.6897    0.6250        29
       Neutro     0.3810    0.4706    0.4211        17
        Paura     0.0833    1.0000    0.1538         1
       Rabbia     0.3750    0.5000    0.4286         6
     Sorpresa     0.3810    0.5000    0.4324        16
    Tristezza     0.2857    1.0000    0.4444         2

    micro avg     0.4651    0.6536    0.5435       153
    macro avg     0.4138    0.7062    0.4863       153
 weighted avg     0.5132    0.6536    0.5659       153
  samples avg     0.5239    0.6641    0.5576       153


=== Step 2 Oracle (Step 1 -> ME + MM) | prefix=Oracle (+ rel) | Fine-tuned model (direct prediction) | features=- ===
               precision    recall  f1-score   support

        Amore     0.4545    0.4545    0.4545        11
Anticipazione     0.6364    0.7000    0.6667        10
     Disgusto     0.6190    0.6842    0.6500        19
      Fiducia     0.6857    0.5714    0.6234        42
        Gioia     0.5333    0.5517    0.5424        29
       Neutro     0.2647    0.5294    0.3529        17
        Paura     0.0000    0.0000    0.0000         1
       Rabbia     1.0000    0.3333    0.5000         6
     Sorpresa     0.5000    0.5000    0.5000        16
    Tristezza     0.5000    0.5000    0.5000         2

    micro avg     0.5247    0.5556    0.5397       153
    macro avg     0.5194    0.4825    0.4790       153
 weighted avg     0.5679    0.5556    0.5485       153
  samples avg     0.5446    0.5556    0.5336       153


=== Step 2 Oracle (Step 1 -> ME + MM) | prefix=Oracle (+ rel) | Blind text (frozen CLS + head) | features=raw ===
               precision    recall  f1-score   support

        Amore     0.2667    0.7273    0.3902        11
Anticipazione     0.5833    0.7000    0.6364        10
     Disgusto     0.5882    0.5263    0.5556        19
      Fiducia     0.6667    0.7143    0.6897        42
        Gioia     0.5714    0.5517    0.5614        29
       Neutro     0.2727    0.5294    0.3600        17
        Paura     0.0000    0.0000    0.0000         1
       Rabbia     0.2857    0.6667    0.4000         6
     Sorpresa     0.5833    0.4375    0.5000        16
    Tristezza     0.1667    0.5000    0.2500         2

    micro avg     0.4554    0.6013    0.5183       153
    macro avg     0.3985    0.5353    0.4343       153
 weighted avg     0.5264    0.6013    0.5456       153
  samples avg     0.5128    0.6047    0.5360       153


=== Step 2 Oracle (Step 1 -> ME + MM) | prefix=Oracle (+ rel) | Dual-Encoder (text + image) | features=raw ===
               precision    recall  f1-score   support

        Amore     0.3000    0.5455    0.3871        11
Anticipazione     0.7143    0.5000    0.5882        10
     Disgusto     0.6429    0.4737    0.5455        19
      Fiducia     0.6429    0.8571    0.7347        42
        Gioia     0.6400    0.5517    0.5926        29
       Neutro     0.4000    0.3529    0.3750        17
        Paura     0.3333    1.0000    0.5000         1
       Rabbia     0.2727    0.5000    0.3529         6
     Sorpresa     0.4444    0.2500    0.3200        16
    Tristezza     0.4000    1.0000    0.5714         2

    micro avg     0.5333    0.5752    0.5535       153
    macro avg     0.4791    0.6031    0.4967       153
 weighted avg     0.5549    0.5752    0.5477       153
  samples avg     0.5226    0.5814    0.5310       153


=== Step 2 Oracle (Step 1 -> ME + MM) | prefix=Oracle (+ rel) | Blind text (frozen CLS + head) | features=std ===
               precision    recall  f1-score   support

        Amore     0.4545    0.4545    0.4545        11
Anticipazione     0.6000    0.9000    0.7200        10
     Disgusto     0.6111    0.5789    0.5946        19
      Fiducia     0.6140    0.8333    0.7071        42
        Gioia     0.5000    0.6207    0.5538        29
       Neutro     0.2759    0.4706    0.3478        17
        Paura     0.0385    1.0000    0.0741         1
       Rabbia     0.2857    0.3333    0.3077         6
     Sorpresa     0.5385    0.4375    0.4828        16
    Tristezza     0.1818    1.0000    0.3077         2

    micro avg     0.4395    0.6405    0.5213       153
    macro avg     0.4100    0.6629    0.4550       153
 weighted avg     0.5119    0.6405    0.5584       153
  samples avg     0.5155    0.6395    0.5508       153


=== Step 2 Oracle (Step 1 -> ME + MM) | prefix=Oracle (+ rel) | Dual-Encoder (text + image) | features=std ===
               precision    recall  f1-score   support

        Amore     0.3889    0.6364    0.4828        11
Anticipazione     0.6429    0.9000    0.7500        10
     Disgusto     0.5217    0.6316    0.5714        19
      Fiducia     0.6538    0.8095    0.7234        42
        Gioia     0.5357    0.5172    0.5263        29
       Neutro     0.2727    0.3529    0.3077        17
        Paura     0.0526    1.0000    0.1000         1
       Rabbia     0.2727    0.5000    0.3529         6
     Sorpresa     0.5000    0.5000    0.5000        16
    Tristezza     0.1250    1.0000    0.2222         2

    micro avg     0.4429    0.6340    0.5215       153
    macro avg     0.3966    0.6848    0.4537       153
 weighted avg     0.5111    0.6340    0.5569       153
  samples avg     0.5009    0.6305    0.5337       153



```

### Textual Irony Detection
```
=== Step 1: IronITA + ME Textual (No Correlations) ===
              precision    recall  f1-score   support

        None     0.8307    0.8333    0.8320       318
      Ironia     0.3333    0.3418    0.3375        79
    Sarcasmo     0.2000    0.1250    0.1538         8

    accuracy                         0.7235       405
   macro avg     0.4547    0.4334    0.4411       405
weighted avg     0.7212    0.7235    0.7222       405


=== Step 2 Raw Ablation: Step 1 -> ME Textual + MM Train (No Correlations) ===
              precision    recall  f1-score   support

        None     0.8344    0.8553    0.8447       318
      Ironia     0.3623    0.3165    0.3378        79
    Sarcasmo     0.1000    0.1250    0.1111         8

    accuracy                         0.7358       405
   macro avg     0.4322    0.4323    0.4312       405
weighted avg     0.7278    0.7358    0.7314       405


=== Step 2 Oracle: Step 1 -> ME Textual + MM Train (With Oracle Correlations) ===
              precision    recall  f1-score   support

        None     0.8527    0.8553    0.8540       318
      Ironia     0.4595    0.4304    0.4444        79
    Sarcasmo     0.1667    0.2500    0.2000         8

    accuracy                         0.7605       405
   macro avg     0.4929    0.5119    0.4995       405
weighted avg     0.7624    0.7605    0.7612       405



```

### Multimodal Irony Dual-Encoder
```
=== Step 1 Base (IronITA + ME) | prefix=No | Fine-tuned model (direct prediction) | features=- ===
              precision    recall  f1-score   support

        None     0.8273    0.8750    0.8505       104
      Ironia     0.1667    0.1667    0.1667        18
    Sarcasmo     1.0000    0.1429    0.2500         7

    accuracy                         0.7364       129
   macro avg     0.6646    0.3948    0.4224       129
weighted avg     0.7445    0.7364    0.7225       129


=== Step 1 Base (IronITA + ME) | prefix=No | Blind text (frozen CLS + head) | features=raw ===
              precision    recall  f1-score   support

        None     0.9091    0.3846    0.5405       104
      Ironia     0.2222    0.1111    0.1481        18
    Sarcasmo     0.0526    0.5714    0.0964         7

    accuracy                         0.3566       129
   macro avg     0.3946    0.3557    0.2617       129
weighted avg     0.7668    0.3566    0.4617       129


=== Step 1 Base (IronITA + ME) | prefix=No | Dual-Encoder (text + image) | features=raw ===
              precision    recall  f1-score   support

        None     0.8545    0.9038    0.8785       104
      Ironia     0.4211    0.4444    0.4324        18
    Sarcasmo     0.0000    0.0000    0.0000         7

    accuracy                         0.7907       129
   macro avg     0.4252    0.4494    0.4370       129
weighted avg     0.7477    0.7907    0.7686       129


=== Step 1 Base (IronITA + ME) | prefix=No | Blind text (frozen CLS + head) | features=std ===
              precision    recall  f1-score   support

        None     0.8641    0.8558    0.8599       104
      Ironia     0.3913    0.5000    0.4390        18
    Sarcasmo     0.0000    0.0000    0.0000         7

    accuracy                         0.7597       129
   macro avg     0.4185    0.4519    0.4330       129
weighted avg     0.7512    0.7597    0.7545       129


=== Step 1 Base (IronITA + ME) | prefix=No | Dual-Encoder (text + image) | features=std ===
              precision    recall  f1-score   support

        None     0.8649    0.9231    0.8930       104
      Ironia     0.4444    0.4444    0.4444        18
    Sarcasmo     0.0000    0.0000    0.0000         7

    accuracy                         0.8062       129
   macro avg     0.4364    0.4558    0.4458       129
weighted avg     0.7593    0.8062    0.7820       129


=== Step 2 Raw (Step 1 -> ME + MM) | prefix=No | Fine-tuned model (direct prediction) | features=- ===
              precision    recall  f1-score   support

        None     0.8990    0.8558    0.8768       104
      Ironia     0.3793    0.6111    0.4681        18
    Sarcasmo     1.0000    0.1429    0.2500         7

    accuracy                         0.7829       129
   macro avg     0.7594    0.5366    0.5316       129
weighted avg     0.8320    0.7829    0.7858       129


=== Step 2 Raw (Step 1 -> ME + MM) | prefix=No | Blind text (frozen CLS + head) | features=raw ===
              precision    recall  f1-score   support

        None     0.9072    0.8462    0.8756       104
      Ironia     0.2941    0.2778    0.2857        18
    Sarcasmo     0.2000    0.4286    0.2727         7

    accuracy                         0.7442       129
   macro avg     0.4671    0.5175    0.4780       129
weighted avg     0.7833    0.7442    0.7606       129


=== Step 2 Raw (Step 1 -> ME + MM) | prefix=No | Dual-Encoder (text + image) | features=raw ===
              precision    recall  f1-score   support

        None     0.8900    0.8558    0.8725       104
      Ironia     0.4074    0.6111    0.4889        18
    Sarcasmo     0.5000    0.1429    0.2222         7

    accuracy                         0.7829       129
   macro avg     0.5991    0.5366    0.5279       129
weighted avg     0.8015    0.7829    0.7837       129


=== Step 2 Raw (Step 1 -> ME + MM) | prefix=No | Blind text (frozen CLS + head) | features=std ===
              precision    recall  f1-score   support

        None     0.8925    0.7981    0.8426       104
      Ironia     0.3000    0.5000    0.3750        18
    Sarcasmo     0.3333    0.2857    0.3077         7

    accuracy                         0.7287       129
   macro avg     0.5086    0.5279    0.5084       129
weighted avg     0.7795    0.7287    0.7484       129


=== Step 2 Raw (Step 1 -> ME + MM) | prefix=No | Dual-Encoder (text + image) | features=std ===
              precision    recall  f1-score   support

        None     0.8947    0.8173    0.8543       104
      Ironia     0.3548    0.6111    0.4490        18
    Sarcasmo     0.6667    0.2857    0.4000         7

    accuracy                         0.7597       129
   macro avg     0.6387    0.5714    0.5678       129
weighted avg     0.8070    0.7597    0.7731       129


=== Step 2 Oracle (Step 1 -> ME + MM) | prefix=Oracle (no rel) | Fine-tuned model (direct prediction) | features=- ===
              precision    recall  f1-score   support

        None     0.8900    0.8558    0.8725       104
      Ironia     0.4074    0.6111    0.4889        18
    Sarcasmo     0.0000    0.0000    0.0000         7

    accuracy                         0.7752       129
   macro avg     0.4325    0.4890    0.4538       129
weighted avg     0.7744    0.7752    0.7717       129


=== Step 2 Oracle (Step 1 -> ME + MM) | prefix=Oracle (no rel) | Blind text (frozen CLS + head) | features=raw ===
              precision    recall  f1-score   support

        None     0.9053    0.8269    0.8643       104
      Ironia     0.3333    0.5556    0.4167        18
    Sarcasmo     0.2500    0.1429    0.1818         7

    accuracy                         0.7519       129
   macro avg     0.4962    0.5084    0.4876       129
weighted avg     0.7899    0.7519    0.7648       129


=== Step 2 Oracle (Step 1 -> ME + MM) | prefix=Oracle (no rel) | Dual-Encoder (text + image) | features=raw ===
              precision    recall  f1-score   support

        None     0.8889    0.8462    0.8670       104
      Ironia     0.3929    0.6111    0.4783        18
    Sarcasmo     0.0000    0.0000    0.0000         7

    accuracy                         0.7674       129
   macro avg     0.4272    0.4858    0.4484       129
weighted avg     0.7714    0.7674    0.7657       129


=== Step 2 Oracle (Step 1 -> ME + MM) | prefix=Oracle (no rel) | Blind text (frozen CLS + head) | features=std ===
              precision    recall  f1-score   support

        None     0.8916    0.7115    0.7914       104
      Ironia     0.3030    0.5556    0.3922        18
    Sarcasmo     0.0769    0.1429    0.1000         7

    accuracy                         0.6589       129
   macro avg     0.4238    0.4700    0.4279       129
weighted avg     0.7652    0.6589    0.6982       129


=== Step 2 Oracle (Step 1 -> ME + MM) | prefix=Oracle (no rel) | Dual-Encoder (text + image) | features=std ===
              precision    recall  f1-score   support

        None     0.9020    0.8846    0.8932       104
      Ironia     0.4400    0.6111    0.5116        18
    Sarcasmo     0.5000    0.1429    0.2222         7

    accuracy                         0.8062       129
   macro avg     0.6140    0.5462    0.5424       129
weighted avg     0.8157    0.8062    0.8036       129


=== Step 2 Oracle (Step 1 -> ME + MM) | prefix=Oracle (+ rel) | Fine-tuned model (direct prediction) | features=- ===
              precision    recall  f1-score   support

        None     0.8763    0.8173    0.8458       104
      Ironia     0.3125    0.5556    0.4000        18
    Sarcasmo     0.0000    0.0000    0.0000         7

    accuracy                         0.7364       129
   macro avg     0.3963    0.4576    0.4153       129
weighted avg     0.7501    0.7364    0.7377       129


=== Step 2 Oracle (Step 1 -> ME + MM) | prefix=Oracle (+ rel) | Blind text (frozen CLS + head) | features=raw ===
              precision    recall  f1-score   support

        None     0.8713    0.8462    0.8585       104
      Ironia     0.3200    0.4444    0.3721        18
    Sarcasmo     0.3333    0.1429    0.2000         7

    accuracy                         0.7519       129
   macro avg     0.5082    0.4778    0.4769       129
weighted avg     0.7652    0.7519    0.7549       129


=== Step 2 Oracle (Step 1 -> ME + MM) | prefix=Oracle (+ rel) | Dual-Encoder (text + image) | features=raw ===
              precision    recall  f1-score   support

        None     0.8835    0.8750    0.8792       104
      Ironia     0.4000    0.5556    0.4651        18
    Sarcasmo     0.0000    0.0000    0.0000         7

    accuracy                         0.7829       129
   macro avg     0.4278    0.4769    0.4481       129
weighted avg     0.7681    0.7829    0.7737       129


=== Step 2 Oracle (Step 1 -> ME + MM) | prefix=Oracle (+ rel) | Blind text (frozen CLS + head) | features=std ===
              precision    recall  f1-score   support

        None     0.8660    0.8077    0.8358       104
      Ironia     0.3200    0.4444    0.3721        18
    Sarcasmo     0.1429    0.1429    0.1429         7

    accuracy                         0.7209       129
   macro avg     0.4429    0.4650    0.4503       129
weighted avg     0.7506    0.7209    0.7335       129


=== Step 2 Oracle (Step 1 -> ME + MM) | prefix=Oracle (+ rel) | Dual-Encoder (text + image) | features=std ===
              precision    recall  f1-score   support

        None     0.8763    0.8173    0.8458       104
      Ironia     0.3889    0.3889    0.3889        18
    Sarcasmo     0.0714    0.1429    0.0952         7

    accuracy                         0.7209       129
   macro avg     0.4455    0.4497    0.4433       129
weighted avg     0.7646    0.7209    0.7413       129



```

### Textual Semiotic Relation Detection
```
=== Stage A: Textual Semiotic Relation (No Correlations) ===
              precision    recall  f1-score   support

    Additiva     0.6162    0.8971    0.7305        68
  Divergente     0.7667    0.6053    0.6765        38
   Parallela     0.0000    0.0000    0.0000        23

    accuracy                         0.6512       129
   macro avg     0.4609    0.5008    0.4690       129
weighted avg     0.5506    0.6512    0.5844       129


=== Stage B: Textual Semiotic Relation (With Oracle Correlations) ===
              precision    recall  f1-score   support

    Additiva     0.6207    0.7941    0.6968        68
  Divergente     0.7097    0.5789    0.6377        38
   Parallela     0.4545    0.2174    0.2941        23

    accuracy                         0.6279       129
   macro avg     0.5950    0.5302    0.5429       129
weighted avg     0.6173    0.6279    0.6076       129



```

### Multimodal Relation Dual-Encoder
```
=== Stage A: MM Train Raw | prefix=No | Fine-tuned model (direct prediction) | features=- ===
              precision    recall  f1-score   support

    Additiva     0.6162    0.8971    0.7305        68
  Divergente     0.7667    0.6053    0.6765        38
   Parallela     0.0000    0.0000    0.0000        23

    accuracy                         0.6512       129
   macro avg     0.4609    0.5008    0.4690       129
weighted avg     0.5506    0.6512    0.5844       129


=== Stage A: MM Train Raw | prefix=No | Blind text (frozen CLS + head) | features=raw ===
              precision    recall  f1-score   support

    Additiva     0.5914    0.8088    0.6832        68
  Divergente     0.6296    0.4474    0.5231        38
   Parallela     0.5556    0.2174    0.3125        23

    accuracy                         0.5969       129
   macro avg     0.5922    0.4912    0.5063       129
weighted avg     0.5963    0.5969    0.5700       129


=== Stage A: MM Train Raw | prefix=No | Dual-Encoder (text + image) | features=raw ===
              precision    recall  f1-score   support

    Additiva     0.6517    0.8529    0.7389        68
  Divergente     0.7188    0.6053    0.6571        38
   Parallela     0.6250    0.2174    0.3226        23

    accuracy                         0.6667       129
   macro avg     0.6651    0.5585    0.5729       129
weighted avg     0.6667    0.6667    0.6406       129


=== Stage A: MM Train Raw | prefix=No | Blind text (frozen CLS + head) | features=std ===
              precision    recall  f1-score   support

    Additiva     0.6709    0.7794    0.7211        68
  Divergente     0.7027    0.6842    0.6933        38
   Parallela     0.5385    0.3043    0.3889        23

    accuracy                         0.6667       129
   macro avg     0.6374    0.5893    0.6011       129
weighted avg     0.6566    0.6667    0.6537       129


=== Stage A: MM Train Raw | prefix=No | Dual-Encoder (text + image) | features=std ===
              precision    recall  f1-score   support

    Additiva     0.6757    0.7353    0.7042        68
  Divergente     0.7097    0.5789    0.6377        38
   Parallela     0.5000    0.5217    0.5106        23

    accuracy                         0.6512       129
   macro avg     0.6285    0.6120    0.6175       129
weighted avg     0.6544    0.6512    0.6501       129


=== Stage B: MM Train Oracle | prefix=Oracle | Fine-tuned model (direct prediction) | features=- ===
              precision    recall  f1-score   support

    Additiva     0.6207    0.7941    0.6968        68
  Divergente     0.7097    0.5789    0.6377        38
   Parallela     0.4545    0.2174    0.2941        23

    accuracy                         0.6279       129
   macro avg     0.5950    0.5302    0.5429       129
weighted avg     0.6173    0.6279    0.6076       129


=== Stage B: MM Train Oracle | prefix=Oracle | Blind text (frozen CLS + head) | features=raw ===
              precision    recall  f1-score   support

    Additiva     0.6264    0.8382    0.7170        68
  Divergente     0.6667    0.5789    0.6197        38
   Parallela     0.8000    0.1739    0.2857        23

    accuracy                         0.6434       129
   macro avg     0.6977    0.5304    0.5408       129
weighted avg     0.6692    0.6434    0.6114       129


=== Stage B: MM Train Oracle | prefix=Oracle | Dual-Encoder (text + image) | features=raw ===
              precision    recall  f1-score   support

    Additiva     0.6957    0.7059    0.7007        68
  Divergente     0.6571    0.6053    0.6301        38
   Parallela     0.5200    0.5652    0.5417        23

    accuracy                         0.6512       129
   macro avg     0.6243    0.6255    0.6242       129
weighted avg     0.6530    0.6512    0.6516       129


=== Stage B: MM Train Oracle | prefix=Oracle | Blind text (frozen CLS + head) | features=std ===
              precision    recall  f1-score   support

    Additiva     0.6463    0.7794    0.7067        68
  Divergente     0.6389    0.6053    0.6216        38
   Parallela     0.6364    0.3043    0.4118        23

    accuracy                         0.6434       129
   macro avg     0.6405    0.5630    0.5800       129
weighted avg     0.6424    0.6434    0.6290       129


=== Stage B: MM Train Oracle | prefix=Oracle | Dual-Encoder (text + image) | features=std ===
              precision    recall  f1-score   support

    Additiva     0.6528    0.6912    0.6714        68
  Divergente     0.6571    0.6053    0.6301        38
   Parallela     0.5455    0.5217    0.5333        23

    accuracy                         0.6357       129
   macro avg     0.6185    0.6061    0.6116       129
weighted avg     0.6349    0.6357    0.6346       129



```

