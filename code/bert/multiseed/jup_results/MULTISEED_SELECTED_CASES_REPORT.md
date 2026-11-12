# MultiEMiT: Consolidated Multi-Seed Report — Paper Selected Configurations (Option B)

> **Scope**: Extracted multi-seed experimental tables (Seeds: `42`, `123`, `456`, locked test split `SPLIT_SEED = 42`) matching the exact configurations presented in the paper:
> 1. **Task 1 (Emotion) & Task 2 (Irony) — Textual Test Set (`me_test`, $N=405$, Fine-Tuned Classification Head)**:
>    - `Step 1 Base (Raw)`
>    - `Step 2 Raw (Ablation)` (without correlations)
>    - `Step 2 Oracle (Train: no rel)` (with correlations, no relation)
>    - `Step 2 Oracle (Train: +rel)` (ablation row: correlations + relation on multimodal training rows)
> 2. **Task 1 (Emotion) & Task 2 (Irony) — Multimodal Test Set (`mm_test`, $N=129$, `raw` unscaled features, `Blind Text` vs. `Dual-Encoder`)**:
>    - **Case 2 (`Text only` / `With images`)**: Step 2 trained on raw text, tested on raw text (`No corr`).
>    - **Case 3 (`... with correlations`)**: Step 2 trained with correlations (no relation), tested with correlations (no relation).
>    - **Case 6 (`Ablation: correlations + relation` / matched to Pixtral)**: Step 2 trained with correlations + relation, tested with correlations + relation.
> 3. **Task 3 (Semiotic Relation) — Multimodal Test Set (`mm_test`, $N=129$, `raw` unscaled features)**:
>    - **Relation-NoCorr (`Without correlations`, Stage A Raw)**: Fine-tuned head (reference) + Text only + Dual-Encoder.
>    - **Relation-Corr (`With correlations`, Stage B Oracle)**: Fine-tuned head (reference) + Text only + Dual-Encoder.
> *(Excluded: Cases 1, 4, 5, all `std`-scaled variants, and direct fine-tuned head predictions on the multimodal test set for Emotion/Irony.)*

---

## 0. Master Summary Table (All Paper Configurations — 3-Seed Means & Per-Seed Values)

| Task | Split / Modality | Case | Regime | Prompt | Model | Scaling | Macro F1 (s42) | Macro F1 (s123) | Macro F1 (s456) | Macro F1 (Mean ± Std) | Micro F1 / Acc (s42) | Micro F1 / Acc (s123) | Micro F1 / Acc (s456) | Micro F1 / Acc (Mean ± Std) | Weighted F1 (Mean ± Std) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Task 1: Emotion | Textual (me_test, N=405) | Textual | Step 1 Base (Raw) | Raw | FEEL-IT (SequenceClassification) | — | 0.5123 | 0.5063 | 0.4987 | 0.5058 ± 0.0068 | 0.5326 | 0.5324 | 0.5221 | 0.5290 ± 0.0060 | 0.5254 ± 0.0063 |
| Task 1: Emotion | Textual (me_test, N=405) | Textual | Step 2 Raw (Ablation) | Raw | FEEL-IT (SequenceClassification) | — | 0.5104 | 0.5296 | 0.5047 | 0.5149 ± 0.0130 | 0.5352 | 0.5258 | 0.5229 | 0.5280 ± 0.0064 | 0.5272 ± 0.0062 |
| Task 1: Emotion | Textual (me_test, N=405) | Textual | Step 2 Oracle (Train: no rel) | Oracle (Train: no rel, Eval: no rel) | FEEL-IT (SequenceClassification) | — | 0.5304 | 0.5399 | 0.5205 | 0.5303 ± 0.0097 | 0.5572 | 0.5620 | 0.5464 | 0.5552 ± 0.0080 | 0.5522 ± 0.0089 |
| Task 1: Emotion | Textual (me_test, N=405) | Textual | Step 2 Oracle (Train: +rel) | Oracle (Train: +rel, Eval: no rel) | FEEL-IT (SequenceClassification) | — | 0.5592 | 0.5449 | 0.5284 | 0.5442 ± 0.0154 | 0.5732 | 0.5541 | 0.5435 | 0.5569 ± 0.0150 | 0.5565 ± 0.0153 |
| Task 1: Emotion | Multimodal (mm_test, N=129) | Case 2 | Step 2 Raw (No corr) | None (Raw) | Blind Text (ClassifierChain) | raw | 0.3784 | 0.4530 | 0.4408 | 0.4240 ± 0.0400 | 0.4492 | 0.5029 | 0.5232 | 0.4918 ± 0.0382 | 0.4958 ± 0.0369 |
| Task 1: Emotion | Multimodal (mm_test, N=129) | Case 2 | Step 2 Raw (No corr) | None (Raw) | Dual-Encoder (FEEL-IT + ViT-224) | raw | 0.4761 | 0.4300 | 0.3912 | 0.4324 ± 0.0425 | 0.4938 | 0.4909 | 0.4403 | 0.4750 ± 0.0301 | 0.4757 ± 0.0260 |
| Task 1: Emotion | Multimodal (mm_test, N=129) | Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Oracle (no rel) | Blind Text (ClassifierChain) | raw | 0.4339 | 0.4233 | 0.4219 | 0.4264 ± 0.0065 | 0.5242 | 0.4593 | 0.5123 | 0.4986 ± 0.0345 | 0.5313 ± 0.0065 |
| Task 1: Emotion | Multimodal (mm_test, N=129) | Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Oracle (no rel) | Dual-Encoder (FEEL-IT + ViT-224) | raw | 0.5222 | 0.5090 | 0.4837 | 0.5050 ± 0.0195 | 0.5223 | 0.5354 | 0.5285 | 0.5287 ± 0.0066 | 0.5337 ± 0.0060 |
| Task 1: Emotion | Multimodal (mm_test, N=129) | Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel) | Oracle (+ rel) | Blind Text (ClassifierChain) | raw | 0.4250 | 0.4700 | 0.4207 | 0.4386 ± 0.0273 | 0.5174 | 0.5252 | 0.5029 | 0.5152 ± 0.0113 | 0.5368 ± 0.0171 |
| Task 1: Emotion | Multimodal (mm_test, N=129) | Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel) | Oracle (+ rel) | Dual-Encoder (FEEL-IT + ViT-224) | raw | 0.4744 | 0.4688 | 0.4306 | 0.4579 ± 0.0238 | 0.4867 | 0.5401 | 0.5333 | 0.5200 ± 0.0291 | 0.5305 ± 0.0314 |
| Task 2: Irony | Textual (me_test, N=405) | Textual | Step 1 Base (Raw) | Raw | UmBERTo (SequenceClassification) | — | 0.3970 | 0.4692 | 0.4805 | 0.4489 ± 0.0453 | 0.7309 | 0.7432 | 0.7383 | 0.7374 ± 0.0062 | 0.7316 ± 0.0041 |
| Task 2: Irony | Textual (me_test, N=405) | Textual | Step 2 Raw (Ablation) | Raw | UmBERTo (SequenceClassification) | — | 0.4485 | 0.5292 | 0.5404 | 0.5060 ± 0.0501 | 0.7506 | 0.7852 | 0.7432 | 0.7597 ± 0.0224 | 0.7526 ± 0.0119 |
| Task 2: Irony | Textual (me_test, N=405) | Textual | Step 2 Oracle (Train: no rel) | Oracle (Train: no rel, Eval: no rel) | UmBERTo (SequenceClassification) | — | 0.5419 | 0.5146 | 0.5418 | 0.5328 ± 0.0157 | 0.7531 | 0.7506 | 0.7901 | 0.7646 ± 0.0221 | 0.7641 ± 0.0130 |
| Task 2: Irony | Textual (me_test, N=405) | Textual | Step 2 Oracle (Train: +rel) | Oracle (Train: +rel, Eval: no rel) | UmBERTo (SequenceClassification) | — | 0.4962 | 0.5011 | 0.5329 | 0.5101 ± 0.0199 | 0.7259 | 0.7630 | 0.7901 | 0.7597 ± 0.0322 | 0.7590 ± 0.0213 |
| Task 2: Irony | Multimodal (mm_test, N=129) | Case 2 | Step 2 Raw (No corr) | None (Raw) | Blind Text (SGD, raw) | raw | 0.4784 | 0.4788 | 0.4023 | 0.4532 ± 0.0441 | 0.7597 | 0.7752 | 0.7209 | 0.7519 ± 0.0280 | 0.7581 ± 0.0211 |
| Task 2: Irony | Multimodal (mm_test, N=129) | Case 2 | Step 2 Raw (No corr) | None (Raw) | Dual-Encoder (UmBERTo + ViT-224, raw) | raw | 0.5083 | 0.5196 | 0.5561 | 0.5280 ± 0.0250 | 0.7829 | 0.8062 | 0.8372 | 0.8088 ± 0.0272 | 0.7999 ± 0.0248 |
| Task 2: Irony | Multimodal (mm_test, N=129) | Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Oracle (no rel) | Blind Text (SGD, raw) | raw | 0.4632 | 0.4642 | 0.3954 | 0.4409 ± 0.0394 | 0.7364 | 0.7209 | 0.7287 | 0.7287 ± 0.0078 | 0.7440 ± 0.0103 |
| Task 2: Irony | Multimodal (mm_test, N=129) | Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Oracle (no rel) | Dual-Encoder (UmBERTo + ViT-224, raw) | raw | 0.4321 | 0.5267 | 0.4069 | 0.4552 ± 0.0632 | 0.7519 | 0.7674 | 0.7597 | 0.7597 ± 0.0078 | 0.7614 ± 0.0148 |
| Task 2: Irony | Multimodal (mm_test, N=129) | Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel) | Oracle (+ rel) | Blind Text (SGD, raw) | raw | 0.4382 | 0.4424 | 0.3942 | 0.4249 ± 0.0267 | 0.7597 | 0.7674 | 0.7519 | 0.7597 ± 0.0078 | 0.7586 ± 0.0094 |
| Task 2: Irony | Multimodal (mm_test, N=129) | Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel) | Oracle (+ rel) | Dual-Encoder (UmBERTo + ViT-224, raw) | raw | 0.4463 | 0.4186 | 0.4323 | 0.4324 ± 0.0139 | 0.7907 | 0.7752 | 0.7907 | 0.7855 ± 0.0090 | 0.7680 ± 0.0068 |
| Task 3: Semiotic Relation | Multimodal (mm_test, N=129) | Relation-NoCorr (Ref. Head) | Stage A Raw | Raw | UmBERTo (SequenceClassification) | — | 0.5680 | 0.4649 | 0.5886 | 0.5405 ± 0.0663 | 0.6667 | 0.6512 | 0.6667 | 0.6615 ± 0.0090 | 0.6230 ± 0.0353 |
| Task 3: Semiotic Relation | Multimodal (mm_test, N=129) | Relation-Corr (Ref. Head) | Stage B Oracle | Oracle | UmBERTo (SequenceClassification) | — | 0.6541 | 0.4668 | 0.5044 | 0.5417 ± 0.0991 | 0.6977 | 0.6512 | 0.6434 | 0.6641 ± 0.0293 | 0.6212 ± 0.0551 |
| Task 3: Semiotic Relation | Multimodal (mm_test, N=129) | Relation-NoCorr | Stage A Raw (No corr) | None (Raw) | Blind Text (SGD, raw) | raw | 0.5958 | 0.5817 | 0.5366 | 0.5713 ± 0.0309 | 0.6357 | 0.6047 | 0.6202 | 0.6202 ± 0.0155 | 0.6119 ± 0.0173 |
| Task 3: Semiotic Relation | Multimodal (mm_test, N=129) | Relation-NoCorr | Stage A Raw (No corr) | None (Raw) | Dual-Encoder (UmBERTo + ViT-224, raw) | raw | 0.5595 | 0.5189 | 0.5779 | 0.5521 ± 0.0302 | 0.5736 | 0.5891 | 0.6202 | 0.5943 ± 0.0237 | 0.5933 ± 0.0213 |
| Task 3: Semiotic Relation | Multimodal (mm_test, N=129) | Relation-Corr | Stage B Oracle (With corr) | Oracle | Blind Text (SGD, raw) | raw | 0.5435 | 0.6323 | 0.6341 | 0.6033 ± 0.0518 | 0.5349 | 0.6822 | 0.6667 | 0.6279 ± 0.0809 | 0.6242 ± 0.0742 |
| Task 3: Semiotic Relation | Multimodal (mm_test, N=129) | Relation-Corr | Stage B Oracle (With corr) | Oracle | Dual-Encoder (UmBERTo + ViT-224, raw) | raw | 0.6047 | 0.5422 | 0.6088 | 0.5852 ± 0.0373 | 0.6512 | 0.5891 | 0.6589 | 0.6331 ± 0.0382 | 0.6251 ± 0.0376 |

---

## 1. Task 1: Emotion Detection (10-Class Multi-Label, `FEEL-IT` + `ViT-224`)

### 1.1 Textual Emotion Benchmark (`me_test`, $N=405$)

| Regime | Prompt | Macro F1 (s42) | Macro F1 (s123) | Macro F1 (s456) | Macro F1 (Mean ± Std) | Micro F1 (s42) | Micro F1 (s123) | Micro F1 (s456) | Micro F1 (Mean ± Std) | Weighted F1 (s42) | Weighted F1 (s123) | Weighted F1 (s456) | Weighted F1 (Mean ± Std) | Subset Acc. (s42) | Subset Acc. (s123) | Subset Acc. (s456) | Subset Acc. (Mean ± Std) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step 1 Base (Raw) | Raw | 0.5123 | 0.5063 | 0.4987 | 0.5058 ± 0.0068 | 0.5326 | 0.5324 | 0.5221 | 0.5290 ± 0.0060 | 0.5297 | 0.5283 | 0.5182 | 0.5254 ± 0.0063 | 0.3333 | 0.2914 | 0.3160 | 0.3136 ± 0.0211 |
| Step 2 Raw (Ablation) | Raw | 0.5104 | 0.5296 | 0.5047 | 0.5149 ± 0.0130 | 0.5352 | 0.5258 | 0.5229 | 0.5280 ± 0.0064 | 0.5341 | 0.5254 | 0.5221 | 0.5272 ± 0.0062 | 0.3210 | 0.3012 | 0.3235 | 0.3152 ± 0.0122 |
| Step 2 Oracle (Train: no rel) | Oracle (Train: no rel, Eval: no rel) | 0.5304 | 0.5399 | 0.5205 | 0.5303 ± 0.0097 | 0.5572 | 0.5620 | 0.5464 | 0.5552 ± 0.0080 | 0.5556 | 0.5589 | 0.5420 | 0.5522 ± 0.0089 | 0.3235 | 0.3185 | 0.3432 | 0.3284 ± 0.0131 |
| Step 2 Oracle (Train: +rel) | Oracle (Train: +rel, Eval: no rel) | 0.5592 | 0.5449 | 0.5284 | 0.5442 ± 0.0154 | 0.5732 | 0.5541 | 0.5435 | 0.5569 ± 0.0150 | 0.5733 | 0.5529 | 0.5434 | 0.5565 ± 0.0153 | 0.3333 | 0.3407 | 0.3333 | 0.3358 ± 0.0043 |

#### Textual Per-Emotion F1 Breakdown (Part 1: `Amore`, `Anticipazione`, `Disgusto`, `Fiducia`, `Gioia`)

| Regime | Amore F1 (s42) | Amore F1 (s123) | Amore F1 (s456) | Amore F1 (Mean ± Std) | Anticipazione F1 (s42) | Anticipazione F1 (s123) | Anticipazione F1 (s456) | Anticipazione F1 (Mean ± Std) | Disgusto F1 (s42) | Disgusto F1 (s123) | Disgusto F1 (s456) | Disgusto F1 (Mean ± Std) | Fiducia F1 (s42) | Fiducia F1 (s123) | Fiducia F1 (s456) | Fiducia F1 (Mean ± Std) | Gioia F1 (s42) | Gioia F1 (s123) | Gioia F1 (s456) | Gioia F1 (Mean ± Std) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step 1 Base (Raw) | 0.6538 | 0.6604 | 0.6042 | 0.6395 ± 0.0307 | 0.5000 | 0.5306 | 0.5000 | 0.5102 ± 0.0177 | 0.5897 | 0.5789 | 0.5714 | 0.5800 ± 0.0092 | 0.5644 | 0.5696 | 0.5783 | 0.5708 ± 0.0070 | 0.5714 | 0.5545 | 0.5842 | 0.5700 ± 0.0149 |
| Step 2 Raw (Ablation) | 0.6392 | 0.6337 | 0.6364 | 0.6364 ± 0.0028 | 0.4490 | 0.5714 | 0.5106 | 0.5103 ± 0.0612 | 0.5658 | 0.5854 | 0.5191 | 0.5567 ± 0.0341 | 0.5549 | 0.5521 | 0.5811 | 0.5627 ± 0.0160 | 0.5970 | 0.5444 | 0.5500 | 0.5638 ± 0.0289 |
| Step 2 Oracle (Train: no rel) | 0.6374 | 0.6126 | 0.6186 | 0.6228 ± 0.0129 | 0.4815 | 0.4898 | 0.4545 | 0.4753 ± 0.0184 | 0.6027 | 0.6275 | 0.5556 | 0.5952 ± 0.0365 | 0.6286 | 0.6415 | 0.6042 | 0.6247 ± 0.0190 | 0.6154 | 0.6030 | 0.6327 | 0.6170 ± 0.0149 |
| Step 2 Oracle (Train: +rel) | 0.6882 | 0.6226 | 0.6275 | 0.6461 ± 0.0365 | 0.5455 | 0.5238 | 0.4643 | 0.5112 ± 0.0420 | 0.5976 | 0.6275 | 0.5912 | 0.6054 ± 0.0194 | 0.6199 | 0.6163 | 0.6272 | 0.6211 ± 0.0056 | 0.6091 | 0.5814 | 0.5820 | 0.5908 ± 0.0158 |

#### Textual Per-Emotion F1 Breakdown (Part 2: `Neutro`, `Paura`, `Rabbia`, `Sorpresa`, `Tristezza`)

| Regime | Neutro F1 (s42) | Neutro F1 (s123) | Neutro F1 (s456) | Neutro F1 (Mean ± Std) | Paura F1 (s42) | Paura F1 (s123) | Paura F1 (s456) | Paura F1 (Mean ± Std) | Rabbia F1 (s42) | Rabbia F1 (s123) | Rabbia F1 (s456) | Rabbia F1 (Mean ± Std) | Sorpresa F1 (s42) | Sorpresa F1 (s123) | Sorpresa F1 (s456) | Sorpresa F1 (Mean ± Std) | Tristezza F1 (s42) | Tristezza F1 (s123) | Tristezza F1 (s456) | Tristezza F1 (Mean ± Std) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step 1 Base (Raw) | 0.4762 | 0.4356 | 0.4490 | 0.4536 ± 0.0207 | 0.3750 | 0.2353 | 0.3529 | 0.3211 ± 0.0751 | 0.4151 | 0.4615 | 0.4074 | 0.4280 ± 0.0293 | 0.4181 | 0.4199 | 0.3905 | 0.4095 ± 0.0165 | 0.5591 | 0.6170 | 0.5495 | 0.5752 ± 0.0365 |
| Step 2 Raw (Ablation) | 0.5051 | 0.3964 | 0.4646 | 0.4554 ± 0.0549 | 0.3333 | 0.5882 | 0.3750 | 0.4322 ± 0.1367 | 0.4906 | 0.4400 | 0.3902 | 0.4403 ± 0.0502 | 0.4592 | 0.4444 | 0.4550 | 0.4529 ± 0.0076 | 0.5102 | 0.5400 | 0.5652 | 0.5385 ± 0.0275 |
| Step 2 Oracle (Train: no rel) | 0.4536 | 0.4950 | 0.4536 | 0.4674 ± 0.0239 | 0.4000 | 0.4706 | 0.3750 | 0.4152 ± 0.0496 | 0.4444 | 0.4231 | 0.5000 | 0.4558 ± 0.0397 | 0.4717 | 0.4457 | 0.4339 | 0.4504 ± 0.0194 | 0.5686 | 0.5905 | 0.5773 | 0.5788 ± 0.0110 |
| Step 2 Oracle (Train: +rel) | 0.5347 | 0.4746 | 0.4381 | 0.4824 ± 0.0488 | 0.4706 | 0.5556 | 0.5556 | 0.5272 ± 0.0491 | 0.4615 | 0.4364 | 0.3704 | 0.4228 ± 0.0471 | 0.4848 | 0.4497 | 0.4656 | 0.4667 ± 0.0176 | 0.5806 | 0.5614 | 0.5625 | 0.5682 ± 0.0108 |

### 1.2 Multimodal Emotion Baselines — Case 2 (`No corr`), Case 3 (`Corr, no rel`) & Case 6 (`Corr + rel`), `raw` Features (`mm_test`, $N=129$)

| Case | Regime | Prompting | Model | Scaling | Macro F1 (s42) | Macro F1 (s123) | Macro F1 (s456) | Macro F1 (Mean ± Std) | Micro F1 (s42) | Micro F1 (s123) | Micro F1 (s456) | Micro F1 (Mean ± Std) | Weighted F1 (s42) | Weighted F1 (s123) | Weighted F1 (s456) | Weighted F1 (Mean ± Std) | Subset Acc. (s42) | Subset Acc. (s123) | Subset Acc. (s456) | Subset Acc. (Mean ± Std) |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Case 2 | Step 2 Raw (No corr) | None (Raw) | Blind Text (ClassifierChain) | raw | 0.3784 | 0.4530 | 0.4408 | 0.4240 ± 0.0400 | 0.4492 | 0.5029 | 0.5232 | 0.4918 ± 0.0382 | 0.4539 | 0.5099 | 0.5237 | 0.4958 ± 0.0369 | 0.3488 | 0.2713 | 0.2868 | 0.3023 ± 0.0410 |
| Case 2 | Step 2 Raw (No corr) | None (Raw) | Dual-Encoder (FEEL-IT + ViT-224) | raw | 0.4761 | 0.4300 | 0.3912 | 0.4324 ± 0.0425 | 0.4938 | 0.4909 | 0.4403 | 0.4750 ± 0.0301 | 0.4867 | 0.4944 | 0.4461 | 0.4757 ± 0.0260 | 0.3178 | 0.3566 | 0.2636 | 0.3127 ± 0.0467 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Oracle (no rel) | Blind Text (ClassifierChain) | raw | 0.4339 | 0.4233 | 0.4219 | 0.4264 ± 0.0065 | 0.5242 | 0.4593 | 0.5123 | 0.4986 ± 0.0345 | 0.5262 | 0.5291 | 0.5386 | 0.5313 ± 0.0065 | 0.3178 | 0.1395 | 0.2636 | 0.2403 ± 0.0914 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Oracle (no rel) | Dual-Encoder (FEEL-IT + ViT-224) | raw | 0.5222 | 0.5090 | 0.4837 | 0.5050 ± 0.0195 | 0.5223 | 0.5354 | 0.5285 | 0.5287 ± 0.0066 | 0.5275 | 0.5342 | 0.5395 | 0.5337 ± 0.0060 | 0.3023 | 0.4031 | 0.2791 | 0.3282 ± 0.0659 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel) | Oracle (+ rel) | Blind Text (ClassifierChain) | raw | 0.4250 | 0.4700 | 0.4207 | 0.4386 ± 0.0273 | 0.5174 | 0.5252 | 0.5029 | 0.5152 ± 0.0113 | 0.5368 | 0.5539 | 0.5196 | 0.5368 ± 0.0171 | 0.3101 | 0.2558 | 0.3333 | 0.2997 ± 0.0398 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel) | Oracle (+ rel) | Dual-Encoder (FEEL-IT + ViT-224) | raw | 0.4744 | 0.4688 | 0.4306 | 0.4579 ± 0.0238 | 0.4867 | 0.5401 | 0.5333 | 0.5200 ± 0.0291 | 0.4958 | 0.5568 | 0.5389 | 0.5305 ± 0.0314 | 0.3256 | 0.3798 | 0.3333 | 0.3463 ± 0.0293 |

#### Multimodal Per-Emotion F1 Breakdown — Case 2, Case 3 & Case 6 (`raw`) (Part 1: `Amore` – `Gioia`)

| Case | Regime | Model | Amore F1 (s42) | Amore F1 (s123) | Amore F1 (s456) | Amore F1 (Mean ± Std) | Anticipazione F1 (s42) | Anticipazione F1 (s123) | Anticipazione F1 (s456) | Anticipazione F1 (Mean ± Std) | Disgusto F1 (s42) | Disgusto F1 (s123) | Disgusto F1 (s456) | Disgusto F1 (Mean ± Std) | Fiducia F1 (s42) | Fiducia F1 (s123) | Fiducia F1 (s456) | Fiducia F1 (Mean ± Std) | Gioia F1 (s42) | Gioia F1 (s123) | Gioia F1 (s456) | Gioia F1 (Mean ± Std) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Case 2 | Step 2 Raw (No corr) | Blind Text (ClassifierChain) | 0.3750 | 0.6316 | 0.5714 | 0.5260 ± 0.1342 | 0.5600 | 0.4286 | 0.3636 | 0.4507 ± 0.1000 | 0.4500 | 0.4571 | 0.5854 | 0.4975 ± 0.0762 | 0.6133 | 0.6667 | 0.6552 | 0.6451 ± 0.0281 | 0.5000 | 0.4928 | 0.4800 | 0.4909 ± 0.0101 |
| Case 2 | Step 2 Raw (No corr) | Dual-Encoder (FEEL-IT + ViT-224) | 0.3200 | 0.4444 | 0.3590 | 0.3745 ± 0.0637 | 0.6364 | 0.5517 | 0.4211 | 0.5364 ± 0.1085 | 0.4242 | 0.4516 | 0.3784 | 0.4181 ± 0.0370 | 0.6506 | 0.6410 | 0.6154 | 0.6357 ± 0.0182 | 0.5915 | 0.4918 | 0.5455 | 0.5429 ± 0.0499 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (ClassifierChain) | 0.4103 | 0.5517 | 0.3077 | 0.4232 ± 0.1225 | 0.5263 | 0.2195 | 0.3846 | 0.3768 ± 0.1536 | 0.6667 | 0.6667 | 0.7000 | 0.6778 ± 0.0192 | 0.6105 | 0.6585 | 0.6667 | 0.6452 ± 0.0303 | 0.5750 | 0.6176 | 0.6000 | 0.5975 ± 0.0214 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (FEEL-IT + ViT-224) | 0.4706 | 0.3000 | 0.2759 | 0.3488 ± 0.1061 | 0.5000 | 0.6316 | 0.5263 | 0.5526 ± 0.0696 | 0.6500 | 0.5455 | 0.5714 | 0.5890 ± 0.0544 | 0.6593 | 0.7191 | 0.7111 | 0.6965 ± 0.0324 | 0.5714 | 0.5588 | 0.5614 | 0.5639 ± 0.0067 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel) | Blind Text (ClassifierChain) | 0.3333 | 0.4103 | 0.3333 | 0.3590 ± 0.0444 | 0.5714 | 0.4000 | 0.5455 | 0.5056 ± 0.0924 | 0.6286 | 0.7895 | 0.5833 | 0.6671 ± 0.1083 | 0.6420 | 0.6667 | 0.6585 | 0.6557 ± 0.0126 | 0.5672 | 0.5429 | 0.5862 | 0.5654 ± 0.0217 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel) | Dual-Encoder (FEEL-IT + ViT-224) | 0.1905 | 0.2703 | 0.2222 | 0.2277 ± 0.0402 | 0.5263 | 0.5000 | 0.5000 | 0.5088 ± 0.0152 | 0.6667 | 0.6667 | 0.7317 | 0.6883 ± 0.0376 | 0.6133 | 0.7143 | 0.6526 | 0.6601 ± 0.0509 | 0.5517 | 0.6087 | 0.6316 | 0.5973 ± 0.0411 |

#### Multimodal Per-Emotion F1 Breakdown — Case 2, Case 3 & Case 6 (`raw`) (Part 2: `Neutro` – `Tristezza`)

| Case | Regime | Model | Neutro F1 (s42) | Neutro F1 (s123) | Neutro F1 (s456) | Neutro F1 (Mean ± Std) | Paura F1 (s42) | Paura F1 (s123) | Paura F1 (s456) | Paura F1 (Mean ± Std) | Rabbia F1 (s42) | Rabbia F1 (s123) | Rabbia F1 (s456) | Rabbia F1 (Mean ± Std) | Sorpresa F1 (s42) | Sorpresa F1 (s123) | Sorpresa F1 (s456) | Sorpresa F1 (Mean ± Std) | Tristezza F1 (s42) | Tristezza F1 (s123) | Tristezza F1 (s456) | Tristezza F1 (Mean ± Std) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Case 2 | Step 2 Raw (No corr) | Blind Text (ClassifierChain) | 0.2286 | 0.3529 | 0.4571 | 0.3462 ± 0.1144 | 0.0000 | 0.0000 | 0.0000 | 0.0000 ± 0.0000 | 0.4286 | 0.3000 | 0.3846 | 0.3711 ± 0.0653 | 0.2286 | 0.4000 | 0.4103 | 0.3463 ± 0.1021 | 0.4000 | 0.8000 | 0.5000 | 0.5667 ± 0.2082 |
| Case 2 | Step 2 Raw (No corr) | Dual-Encoder (FEEL-IT + ViT-224) | 0.2759 | 0.3810 | 0.2857 | 0.3142 ± 0.0580 | 0.5000 | 0.0000 | 0.0000 | 0.1667 ± 0.2887 | 0.3750 | 0.3077 | 0.3333 | 0.3387 ± 0.0340 | 0.1875 | 0.3636 | 0.1739 | 0.2417 ± 0.1058 | 0.8000 | 0.6667 | 0.8000 | 0.7556 ± 0.0770 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (ClassifierChain) | 0.3333 | 0.1875 | 0.4138 | 0.3115 ± 0.1147 | 0.0000 | 0.0769 | 0.1176 | 0.0649 ± 0.0597 | 0.4286 | 0.4211 | 0.3636 | 0.4044 ± 0.0355 | 0.4242 | 0.5000 | 0.4151 | 0.4464 ± 0.0466 | 0.3636 | 0.3333 | 0.2500 | 0.3157 ± 0.0588 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (FEEL-IT + ViT-224) | 0.2222 | 0.4000 | 0.3810 | 0.3344 ± 0.0976 | 1.0000 | 0.6667 | 0.6667 | 0.7778 ± 0.1925 | 0.3636 | 0.3077 | 0.3529 | 0.3414 ± 0.0297 | 0.3846 | 0.2941 | 0.4571 | 0.3786 ± 0.0817 | 0.4000 | 0.6667 | 0.3333 | 0.4667 ± 0.1764 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel) | Blind Text (ClassifierChain) | 0.4000 | 0.3571 | 0.3871 | 0.3814 ± 0.0220 | 0.0000 | 0.1818 | 0.0000 | 0.0606 ± 0.1050 | 0.4000 | 0.2400 | 0.3529 | 0.3310 ± 0.0822 | 0.4848 | 0.5405 | 0.3158 | 0.4471 ± 0.1170 | 0.2222 | 0.5714 | 0.4444 | 0.4127 ± 0.1768 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel) | Dual-Encoder (FEEL-IT + ViT-224) | 0.2791 | 0.4000 | 0.3333 | 0.3375 ± 0.0606 | 0.6667 | 0.5000 | 0.0000 | 0.3889 ± 0.3469 | 0.3333 | 0.2857 | 0.4444 | 0.3545 ± 0.0815 | 0.3448 | 0.4571 | 0.3902 | 0.3974 ± 0.0565 | 0.5714 | 0.2857 | 0.4000 | 0.4190 ± 0.1438 |

### 1.3 Multimodal Emotion Semiotic Subsets (`ADD`, `PAR`, `DIV`) — Case 2, Case 3 & Case 6 (`raw`)

#### Semiotic Subsets Micro F1

| Case | Regime | Model | Scaling | ADD (N=68) (s42) | ADD (N=68) (s123) | ADD (N=68) (s456) | ADD (N=68) (Mean ± Std) | PAR (N=23) (s42) | PAR (N=23) (s123) | PAR (N=23) (s456) | PAR (N=23) (Mean ± Std) | DIV (N=38) (s42) | DIV (N=38) (s123) | DIV (N=38) (s456) | DIV (N=38) (Mean ± Std) | Macro F1 (s42) | Macro F1 (s123) | Macro F1 (s456) | Macro F1 (Mean ± Std) | Micro F1 (s42) | Micro F1 (s123) | Micro F1 (s456) | Micro F1 (Mean ± Std) |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Case 2 | Step 2 Raw (No corr) | Blind Text (ClassifierChain) | raw | 0.5371 | 0.5587 | 0.5584 | 0.5514 ± 0.0123 | 0.4082 | 0.5357 | 0.5862 | 0.5100 ± 0.0918 | 0.3168 | 0.3925 | 0.4286 | 0.3793 ± 0.0570 | 0.3784 | 0.4530 | 0.4408 | 0.4240 ± 0.0400 | 0.4492 | 0.5029 | 0.5232 | 0.4918 ± 0.0382 |
| Case 2 | Step 2 Raw (No corr) | Dual-Encoder (FEEL-IT + ViT-224) | raw | 0.5862 | 0.5780 | 0.5181 | 0.5608 ± 0.0372 | 0.5882 | 0.5085 | 0.4561 | 0.5176 ± 0.0665 | 0.2737 | 0.3265 | 0.2947 | 0.2983 ± 0.0266 | 0.4761 | 0.4300 | 0.3912 | 0.4324 ± 0.0425 | 0.4938 | 0.4909 | 0.4403 | 0.4750 ± 0.0301 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (ClassifierChain) | raw | 0.5625 | 0.5278 | 0.5274 | 0.5392 ± 0.0202 | 0.5714 | 0.4242 | 0.5556 | 0.5171 ± 0.0808 | 0.4272 | 0.3676 | 0.4643 | 0.4197 ± 0.0488 | 0.4339 | 0.4233 | 0.4219 | 0.4264 ± 0.0065 | 0.5242 | 0.4593 | 0.5123 | 0.4986 ± 0.0345 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (FEEL-IT + ViT-224) | raw | 0.5650 | 0.6012 | 0.5730 | 0.5797 ± 0.0190 | 0.6000 | 0.5455 | 0.5172 | 0.5542 ± 0.0421 | 0.4000 | 0.4124 | 0.4536 | 0.4220 ± 0.0281 | 0.5222 | 0.5090 | 0.4837 | 0.5050 ± 0.0195 | 0.5223 | 0.5354 | 0.5285 | 0.5287 ± 0.0066 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel) | Blind Text (ClassifierChain) | raw | 0.5109 | 0.5784 | 0.5435 | 0.5443 ± 0.0338 | 0.5000 | 0.5246 | 0.5600 | 0.5282 ± 0.0302 | 0.5370 | 0.4286 | 0.4107 | 0.4588 ± 0.0684 | 0.4250 | 0.4700 | 0.4207 | 0.4386 ± 0.0273 | 0.5174 | 0.5252 | 0.5029 | 0.5152 ± 0.0113 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel) | Dual-Encoder (FEEL-IT + ViT-224) | raw | 0.5062 | 0.5876 | 0.5667 | 0.5535 ± 0.0423 | 0.4898 | 0.4906 | 0.5424 | 0.5076 ± 0.0301 | 0.4494 | 0.4860 | 0.4717 | 0.4690 ± 0.0184 | 0.4744 | 0.4688 | 0.4306 | 0.4579 ± 0.0238 | 0.4867 | 0.5401 | 0.5333 | 0.5200 ± 0.0291 |

#### Semiotic Subsets Macro F1

| Case | Regime | Model | Scaling | ADD Macro F1 (s42) | ADD Macro F1 (s123) | ADD Macro F1 (s456) | ADD Macro F1 (Mean ± Std) | PAR Macro F1 (s42) | PAR Macro F1 (s123) | PAR Macro F1 (s456) | PAR Macro F1 (Mean ± Std) | DIV Macro F1 (s42) | DIV Macro F1 (s123) | DIV Macro F1 (s456) | DIV Macro F1 (Mean ± Std) | Macro F1 (s42) | Macro F1 (s123) | Macro F1 (s456) | Macro F1 (Mean ± Std) |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Case 2 | Step 2 Raw (No corr) | Blind Text (ClassifierChain) | raw | 0.4541 | 0.5185 | 0.4884 | 0.4870 ± 0.0322 | 0.2452 | 0.3322 | 0.4716 | 0.3497 ± 0.1142 | 0.2467 | 0.3127 | 0.3788 | 0.3128 ± 0.0660 | 0.3784 | 0.4530 | 0.4408 | 0.4240 ± 0.0400 |
| Case 2 | Step 2 Raw (No corr) | Dual-Encoder (FEEL-IT + ViT-224) | raw | 0.5615 | 0.4988 | 0.4485 | 0.5029 ± 0.0566 | 0.2959 | 0.3157 | 0.2458 | 0.2858 ± 0.0360 | 0.1643 | 0.2373 | 0.2048 | 0.2021 ± 0.0366 | 0.4761 | 0.4300 | 0.3912 | 0.4324 ± 0.0425 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (ClassifierChain) | raw | 0.4962 | 0.5142 | 0.4686 | 0.4930 ± 0.0230 | 0.2857 | 0.4569 | 0.3457 | 0.3628 ± 0.0869 | 0.2762 | 0.3506 | 0.2940 | 0.3069 ± 0.0389 | 0.4339 | 0.4233 | 0.4219 | 0.4264 ± 0.0065 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (FEEL-IT + ViT-224) | raw | 0.5539 | 0.5714 | 0.5248 | 0.5500 ± 0.0235 | 0.3463 | 0.2167 | 0.2746 | 0.2792 ± 0.0649 | 0.2814 | 0.2871 | 0.2743 | 0.2809 ± 0.0064 | 0.5222 | 0.5090 | 0.4837 | 0.5050 ± 0.0195 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel) | Blind Text (ClassifierChain) | raw | 0.4368 | 0.5403 | 0.4708 | 0.4826 ± 0.0527 | 0.2546 | 0.3214 | 0.3556 | 0.3105 ± 0.0513 | 0.4522 | 0.3148 | 0.2339 | 0.3337 ± 0.1104 | 0.4250 | 0.4700 | 0.4207 | 0.4386 ± 0.0273 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel) | Dual-Encoder (FEEL-IT + ViT-224) | raw | 0.4649 | 0.5190 | 0.4753 | 0.4864 ± 0.0287 | 0.2639 | 0.2442 | 0.3095 | 0.2726 ± 0.0335 | 0.2502 | 0.2966 | 0.2763 | 0.2744 ± 0.0233 | 0.4744 | 0.4688 | 0.4306 | 0.4579 ± 0.0238 |

---

## 2. Task 2: Irony & Sarcasm Detection (3-Class Single-Label, `UmBERTo` + `ViT-224`)

### 2.1 Textual Irony Benchmark (`me_test`, $N=405$)

| Regime | Prompt | Macro F1 (s42) | Macro F1 (s123) | Macro F1 (s456) | Macro F1 (Mean ± Std) | Micro F1 (= Acc) (s42) | Micro F1 (= Acc) (s123) | Micro F1 (= Acc) (s456) | Micro F1 (= Acc) (Mean ± Std) | Weighted F1 (s42) | Weighted F1 (s123) | Weighted F1 (s456) | Weighted F1 (Mean ± Std) | None F1 (s42) | None F1 (s123) | None F1 (s456) | None F1 (Mean ± Std) | Ironia F1 (s42) | Ironia F1 (s123) | Ironia F1 (s456) | Ironia F1 (Mean ± Std) | Sarcasmo F1 (s42) | Sarcasmo F1 (s123) | Sarcasmo F1 (s456) | Sarcasmo F1 (Mean ± Std) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step 1 Base (Raw) | Raw | 0.3970 | 0.4692 | 0.4805 | 0.4489 ± 0.0453 | 0.7309 | 0.7432 | 0.7383 | 0.7374 ± 0.0062 | 0.7268 | 0.7342 | 0.7337 | 0.7316 ± 0.0041 | 0.8380 | 0.8449 | 0.8460 | 0.8430 ± 0.0043 | 0.3529 | 0.3404 | 0.3289 | 0.3408 ± 0.0120 | 0.0000 | 0.2222 | 0.2667 | 0.1630 ± 0.1429 |
| Step 2 Raw (Ablation) | Raw | 0.4485 | 0.5292 | 0.5404 | 0.5060 ± 0.0501 | 0.7506 | 0.7852 | 0.7432 | 0.7597 ± 0.0224 | 0.7440 | 0.7662 | 0.7475 | 0.7526 ± 0.0119 | 0.8532 | 0.8726 | 0.8389 | 0.8549 ± 0.0169 | 0.3673 | 0.3817 | 0.4186 | 0.3892 ± 0.0264 | 0.1250 | 0.3333 | 0.3636 | 0.2740 ± 0.1299 |
| Step 2 Oracle (Train: no rel) | Oracle (Train: no rel, Eval: no rel) | 0.5419 | 0.5146 | 0.5418 | 0.5328 ± 0.0157 | 0.7531 | 0.7506 | 0.7901 | 0.7646 ± 0.0221 | 0.7583 | 0.7551 | 0.7790 | 0.7641 ± 0.0130 | 0.8466 | 0.8485 | 0.8767 | 0.8573 ± 0.0169 | 0.4458 | 0.4286 | 0.4328 | 0.4357 ± 0.0090 | 0.3333 | 0.2667 | 0.3158 | 0.3053 ± 0.0346 |
| Step 2 Oracle (Train: +rel) | Oracle (Train: +rel, Eval: no rel) | 0.4962 | 0.5011 | 0.5329 | 0.5101 ± 0.0199 | 0.7259 | 0.7630 | 0.7901 | 0.7597 ± 0.0322 | 0.7412 | 0.7533 | 0.7826 | 0.7590 ± 0.0213 | 0.8264 | 0.8589 | 0.8769 | 0.8541 ± 0.0256 | 0.4516 | 0.3776 | 0.4552 | 0.4281 ± 0.0438 | 0.2105 | 0.2667 | 0.2667 | 0.2480 ± 0.0324 |

### 2.2 Multimodal Irony Baselines — Case 2 (`No corr`), Case 3 (`Corr, no rel`) & Case 6 (`Corr + rel`), `raw` Features (`mm_test`, $N=129$)

| Case | Regime | Prompting | Model | Scaling | Macro F1 (s42) | Macro F1 (s123) | Macro F1 (s456) | Macro F1 (Mean ± Std) | Micro F1 (= Acc.) (s42) | Micro F1 (= Acc.) (s123) | Micro F1 (= Acc.) (s456) | Micro F1 (= Acc.) (Mean ± Std) | Weighted F1 (s42) | Weighted F1 (s123) | Weighted F1 (s456) | Weighted F1 (Mean ± Std) | None F1 (s42) | None F1 (s123) | None F1 (s456) | None F1 (Mean ± Std) | Ironia F1 (s42) | Ironia F1 (s123) | Ironia F1 (s456) | Ironia F1 (Mean ± Std) | Sarcasmo F1 (s42) | Sarcasmo F1 (s123) | Sarcasmo F1 (s456) | Sarcasmo F1 (Mean ± Std) |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Case 2 | Step 2 Raw (No corr) | None (Raw) | Blind Text (SGD, raw) | raw | 0.4784 | 0.4788 | 0.4023 | 0.4532 ± 0.0441 | 0.7597 | 0.7752 | 0.7209 | 0.7519 ± 0.0280 | 0.7715 | 0.7690 | 0.7338 | 0.7581 ± 0.0211 | 0.8824 | 0.8762 | 0.8800 | 0.8795 ± 0.0031 | 0.3529 | 0.3784 | 0.0769 | 0.2694 ± 0.1672 | 0.2000 | 0.1818 | 0.2500 | 0.2106 ± 0.0353 |
| Case 2 | Step 2 Raw (No corr) | None (Raw) | Dual-Encoder (UmBERTo + ViT-224, raw) | raw | 0.5083 | 0.5196 | 0.5561 | 0.5280 ± 0.0250 | 0.7829 | 0.8062 | 0.8372 | 0.8088 ± 0.0272 | 0.7791 | 0.7932 | 0.8274 | 0.7999 ± 0.0248 | 0.8750 | 0.8920 | 0.9151 | 0.8940 ± 0.0201 | 0.4500 | 0.4444 | 0.5714 | 0.4886 ± 0.0718 | 0.2000 | 0.2222 | 0.1818 | 0.2013 ± 0.0202 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Oracle (no rel) | Blind Text (SGD, raw) | raw | 0.4632 | 0.4642 | 0.3954 | 0.4409 ± 0.0394 | 0.7364 | 0.7209 | 0.7287 | 0.7287 ± 0.0078 | 0.7548 | 0.7432 | 0.7342 | 0.7440 ± 0.0103 | 0.8657 | 0.8586 | 0.8529 | 0.8591 ± 0.0064 | 0.3333 | 0.2581 | 0.3333 | 0.3082 ± 0.0435 | 0.1905 | 0.2759 | 0.0000 | 0.1554 ± 0.1412 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Oracle (no rel) | Dual-Encoder (UmBERTo + ViT-224, raw) | raw | 0.4321 | 0.5267 | 0.4069 | 0.4552 ± 0.0632 | 0.7519 | 0.7674 | 0.7597 | 0.7597 ± 0.0078 | 0.7551 | 0.7783 | 0.7509 | 0.7614 ± 0.0148 | 0.8614 | 0.8687 | 0.8708 | 0.8670 ± 0.0049 | 0.4348 | 0.4615 | 0.3500 | 0.4154 ± 0.0582 | 0.0000 | 0.2500 | 0.0000 | 0.0833 ± 0.1443 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel) | Oracle (+ rel) | Blind Text (SGD, raw) | raw | 0.4382 | 0.4424 | 0.3942 | 0.4249 ± 0.0267 | 0.7597 | 0.7674 | 0.7519 | 0.7597 ± 0.0078 | 0.7606 | 0.7669 | 0.7484 | 0.7586 ± 0.0094 | 0.8657 | 0.8725 | 0.8750 | 0.8711 ± 0.0048 | 0.4490 | 0.4545 | 0.3077 | 0.4037 ± 0.0832 | 0.0000 | 0.0000 | 0.0000 | 0.0000 ± 0.0000 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel) | Oracle (+ rel) | Dual-Encoder (UmBERTo + ViT-224, raw) | raw | 0.4463 | 0.4186 | 0.4323 | 0.4324 ± 0.0139 | 0.7907 | 0.7752 | 0.7907 | 0.7855 ± 0.0090 | 0.7717 | 0.7601 | 0.7722 | 0.7680 ± 0.0068 | 0.8774 | 0.8774 | 0.8868 | 0.8805 ± 0.0054 | 0.4615 | 0.3784 | 0.4103 | 0.4167 ± 0.0420 | 0.0000 | 0.0000 | 0.0000 | 0.0000 ± 0.0000 |

### 2.3 Multimodal Irony Semiotic Subsets Accuracy (`ADD`, `PAR`, `DIV`) — Case 2, Case 3 & Case 6 (`raw`)

| Case | Regime | Model | Scaling | ADD (N=68) (s42) | ADD (N=68) (s123) | ADD (N=68) (s456) | ADD (N=68) (Mean ± Std) | PAR (N=23) (s42) | PAR (N=23) (s123) | PAR (N=23) (s456) | PAR (N=23) (Mean ± Std) | DIV (N=38) (s42) | DIV (N=38) (s123) | DIV (N=38) (s456) | DIV (N=38) (Mean ± Std) | Macro F1 (s42) | Macro F1 (s123) | Macro F1 (s456) | Macro F1 (Mean ± Std) | Micro F1 (= Acc.) (s42) | Micro F1 (= Acc.) (s123) | Micro F1 (= Acc.) (s456) | Micro F1 (= Acc.) (Mean ± Std) |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Case 2 | Step 2 Raw (No corr) | Blind Text (SGD, raw) | raw | 0.8235 | 0.8529 | 0.8088 | 0.8284 ± 0.0225 | 0.9565 | 0.9130 | 0.9565 | 0.9420 ± 0.0251 | 0.5263 | 0.5526 | 0.4211 | 0.5000 ± 0.0696 | 0.4784 | 0.4788 | 0.4023 | 0.4532 ± 0.0441 | 0.7597 | 0.7752 | 0.7209 | 0.7519 ± 0.0280 |
| Case 2 | Step 2 Raw (No corr) | Dual-Encoder (UmBERTo + ViT-224, raw) | raw | 0.8971 | 0.8529 | 0.8824 | 0.8775 ± 0.0225 | 0.8261 | 0.9130 | 0.9565 | 0.8986 ± 0.0664 | 0.5526 | 0.6579 | 0.6842 | 0.6316 ± 0.0696 | 0.5083 | 0.5196 | 0.5561 | 0.5280 ± 0.0250 | 0.7829 | 0.8062 | 0.8372 | 0.8088 ± 0.0272 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Blind Text (SGD, raw) | raw | 0.8529 | 0.8088 | 0.8235 | 0.8284 ± 0.0225 | 0.9130 | 0.9130 | 0.9130 | 0.9130 ± 0.0000 | 0.4211 | 0.4474 | 0.4474 | 0.4386 ± 0.0152 | 0.4632 | 0.4642 | 0.3954 | 0.4409 ± 0.0394 | 0.7364 | 0.7209 | 0.7287 | 0.7287 ± 0.0078 |
| Case 3 | Step 2 Oracle (Train: no rel, Eval: no rel) | Dual-Encoder (UmBERTo + ViT-224, raw) | raw | 0.8382 | 0.8382 | 0.8529 | 0.8431 ± 0.0085 | 0.9130 | 0.8696 | 0.9130 | 0.8986 ± 0.0251 | 0.5000 | 0.5789 | 0.5000 | 0.5263 ± 0.0456 | 0.4321 | 0.5267 | 0.4069 | 0.4552 ± 0.0632 | 0.7519 | 0.7674 | 0.7597 | 0.7597 ± 0.0078 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel) | Blind Text (SGD, raw) | raw | 0.8382 | 0.8382 | 0.8824 | 0.8529 ± 0.0255 | 0.9565 | 0.9130 | 0.9130 | 0.9275 ± 0.0251 | 0.5000 | 0.5526 | 0.4211 | 0.4912 ± 0.0662 | 0.4382 | 0.4424 | 0.3942 | 0.4249 ± 0.0267 | 0.7597 | 0.7674 | 0.7519 | 0.7597 ± 0.0078 |
| Case 6 | Step 2 Oracle (Train: +rel, Eval: +rel) | Dual-Encoder (UmBERTo + ViT-224, raw) | raw | 0.8824 | 0.8676 | 0.8824 | 0.8775 ± 0.0085 | 0.9130 | 0.9130 | 0.9565 | 0.9275 ± 0.0251 | 0.5526 | 0.5263 | 0.5263 | 0.5351 ± 0.0152 | 0.4463 | 0.4186 | 0.4323 | 0.4324 ± 0.0139 | 0.7907 | 0.7752 | 0.7907 | 0.7855 ± 0.0090 |

---

## 3. Task 3: Semiotic Relation Classification (3-Class Single-Label, `UmBERTo` + `ViT-224`)

> *Note*: Renamed from `Case 1 / Case 2` to **`Relation-NoCorr` (`Stage A Raw`)** and **`Relation-Corr` (`Stage B Oracle`)** to avoid confusion with Emotion/Irony Case 2.

### 3.1 Fine-Tuned UmBERTo Classification Head Reference (`mm_test`, $N=129$)

| Regime | Prompt | Macro F1 (s42) | Macro F1 (s123) | Macro F1 (s456) | Macro F1 (Mean ± Std) | Micro F1 (= Acc) (s42) | Micro F1 (= Acc) (s123) | Micro F1 (= Acc) (s456) | Micro F1 (= Acc) (Mean ± Std) | Weighted F1 (s42) | Weighted F1 (s123) | Weighted F1 (s456) | Weighted F1 (Mean ± Std) | Additiva F1 (s42) | Additiva F1 (s123) | Additiva F1 (s456) | Additiva F1 (Mean ± Std) | Divergente F1 (s42) | Divergente F1 (s123) | Divergente F1 (s456) | Divergente F1 (Mean ± Std) | Parallela F1 (s42) | Parallela F1 (s123) | Parallela F1 (s456) | Parallela F1 (Mean ± Std) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage A Raw | Raw | 0.5680 | 0.4649 | 0.5886 | 0.5405 ± 0.0663 | 0.6667 | 0.6512 | 0.6667 | 0.6615 ± 0.0090 | 0.6390 | 0.5825 | 0.6476 | 0.6230 ± 0.0353 | 0.7453 | 0.7381 | 0.7421 | 0.7419 ± 0.0036 | 0.6462 | 0.6567 | 0.6349 | 0.6459 ± 0.0109 | 0.3125 | 0.0000 | 0.3889 | 0.2338 ± 0.2060 |
| Stage B Oracle | Oracle | 0.6541 | 0.4668 | 0.5044 | 0.5417 ± 0.0991 | 0.6977 | 0.6512 | 0.6434 | 0.6641 ± 0.0293 | 0.6844 | 0.5832 | 0.5959 | 0.6212 ± 0.0551 | 0.7547 | 0.7337 | 0.7229 | 0.7371 ± 0.0162 | 0.6129 | 0.6667 | 0.6364 | 0.6386 ± 0.0270 | 0.5946 | 0.0000 | 0.1538 | 0.2495 ± 0.3086 |

### 3.2 Multimodal Semiotic Relation Baselines — `Relation-NoCorr` & `Relation-Corr`, `raw` Features (`mm_test`, $N=129$)

| Case | Regime | Prompting | Model | Scaling | Macro F1 (s42) | Macro F1 (s123) | Macro F1 (s456) | Macro F1 (Mean ± Std) | Micro F1 (= Acc.) (s42) | Micro F1 (= Acc.) (s123) | Micro F1 (= Acc.) (s456) | Micro F1 (= Acc.) (Mean ± Std) | Weighted F1 (s42) | Weighted F1 (s123) | Weighted F1 (s456) | Weighted F1 (Mean ± Std) | Additiva F1 (s42) | Additiva F1 (s123) | Additiva F1 (s456) | Additiva F1 (Mean ± Std) | Divergente F1 (s42) | Divergente F1 (s123) | Divergente F1 (s456) | Divergente F1 (Mean ± Std) | Parallela F1 (s42) | Parallela F1 (s123) | Parallela F1 (s456) | Parallela F1 (Mean ± Std) |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Relation-NoCorr | Stage A Raw (No corr) | None (Raw) | Blind Text (SGD, raw) | raw | 0.5958 | 0.5817 | 0.5366 | 0.5713 ± 0.0309 | 0.6357 | 0.6047 | 0.6202 | 0.6202 ± 0.0155 | 0.6313 | 0.6063 | 0.5981 | 0.6119 ± 0.0173 | 0.7183 | 0.6519 | 0.7044 | 0.6915 ± 0.0351 | 0.5333 | 0.5833 | 0.5625 | 0.5597 ± 0.0251 | 0.5357 | 0.5098 | 0.3429 | 0.4628 ± 0.1047 |
| Relation-NoCorr | Stage A Raw (No corr) | None (Raw) | Dual-Encoder (UmBERTo + ViT-224, raw) | raw | 0.5595 | 0.5189 | 0.5779 | 0.5521 ± 0.0302 | 0.5736 | 0.5891 | 0.6202 | 0.5943 ± 0.0237 | 0.5812 | 0.5807 | 0.6179 | 0.5933 ± 0.0213 | 0.6230 | 0.6846 | 0.7000 | 0.6692 ± 0.0408 | 0.5556 | 0.5538 | 0.5556 | 0.5550 ± 0.0010 | 0.5000 | 0.3182 | 0.4783 | 0.4321 ± 0.0993 |
| Relation-Corr | Stage B Oracle (With corr) | Oracle | Blind Text (SGD, raw) | raw | 0.5435 | 0.6323 | 0.6341 | 0.6033 ± 0.0518 | 0.5349 | 0.6822 | 0.6667 | 0.6279 ± 0.0809 | 0.5385 | 0.6700 | 0.6640 | 0.6242 ± 0.0742 | 0.5839 | 0.7355 | 0.7123 | 0.6773 ± 0.0816 | 0.3797 | 0.6471 | 0.6567 | 0.5612 ± 0.1572 | 0.6667 | 0.5143 | 0.5333 | 0.5714 ± 0.0830 |
| Relation-Corr | Stage B Oracle (With corr) | Oracle | Dual-Encoder (UmBERTo + ViT-224, raw) | raw | 0.6047 | 0.5422 | 0.6088 | 0.5852 ± 0.0373 | 0.6512 | 0.5891 | 0.6589 | 0.6331 ± 0.0382 | 0.6407 | 0.5823 | 0.6525 | 0.6251 ± 0.0376 | 0.7190 | 0.6759 | 0.7248 | 0.7065 ± 0.0267 | 0.5714 | 0.4857 | 0.6364 | 0.5645 ± 0.0756 | 0.5238 | 0.4651 | 0.4651 | 0.4847 ± 0.0339 |

### 3.3 Semiotic Relation Target Classes Breakdown (`ADD`, `PAR`, `DIV` F1) — `raw` Features (`mm_test`, $N=129$)

| Case | Regime | Model | Scaling | ADD (n=68) (s42) | ADD (n=68) (s123) | ADD (n=68) (s456) | ADD (n=68) (Mean ± Std) | PAR (n=23) (s42) | PAR (n=23) (s123) | PAR (n=23) (s456) | PAR (n=23) (Mean ± Std) | DIV (n=38) (s42) | DIV (n=38) (s123) | DIV (n=38) (s456) | DIV (n=38) (Mean ± Std) | Macro F1 (s42) | Macro F1 (s123) | Macro F1 (s456) | Macro F1 (Mean ± Std) | Micro F1 (= Acc.) (s42) | Micro F1 (= Acc.) (s123) | Micro F1 (= Acc.) (s456) | Micro F1 (= Acc.) (Mean ± Std) |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Relation-NoCorr | Stage A Raw (No corr) | Blind Text (SGD, raw) | raw | 0.7183 | 0.6519 | 0.7044 | 0.6915 ± 0.0351 | 0.5357 | 0.5098 | 0.3429 | 0.4628 ± 0.1047 | 0.5333 | 0.5833 | 0.5625 | 0.5597 ± 0.0251 | 0.5958 | 0.5817 | 0.5366 | 0.5713 ± 0.0309 | 0.6357 | 0.6047 | 0.6202 | 0.6202 ± 0.0155 |
| Relation-NoCorr | Stage A Raw (No corr) | Dual-Encoder (UmBERTo + ViT-224, raw) | raw | 0.6230 | 0.6846 | 0.7000 | 0.6692 ± 0.0408 | 0.5000 | 0.3182 | 0.4783 | 0.4321 ± 0.0993 | 0.5556 | 0.5538 | 0.5556 | 0.5550 ± 0.0010 | 0.5595 | 0.5189 | 0.5779 | 0.5521 ± 0.0302 | 0.5736 | 0.5891 | 0.6202 | 0.5943 ± 0.0237 |
| Relation-Corr | Stage B Oracle (With corr) | Blind Text (SGD, raw) | raw | 0.5839 | 0.7355 | 0.7123 | 0.6773 ± 0.0816 | 0.6667 | 0.5143 | 0.5333 | 0.5714 ± 0.0830 | 0.3797 | 0.6471 | 0.6567 | 0.5612 ± 0.1572 | 0.5435 | 0.6323 | 0.6341 | 0.6033 ± 0.0518 | 0.5349 | 0.6822 | 0.6667 | 0.6279 ± 0.0809 |
| Relation-Corr | Stage B Oracle (With corr) | Dual-Encoder (UmBERTo + ViT-224, raw) | raw | 0.7190 | 0.6759 | 0.7248 | 0.7065 ± 0.0267 | 0.5238 | 0.4651 | 0.4651 | 0.4847 ± 0.0339 | 0.5714 | 0.4857 | 0.6364 | 0.5645 ± 0.0756 | 0.6047 | 0.5422 | 0.6088 | 0.5852 ± 0.0373 | 0.6512 | 0.5891 | 0.6589 | 0.6331 ± 0.0382 |
