# Comprehensive IEEE Transactions Peer Review, Audit & Critique: MultiEMiT

**Target Manuscript**: *MultiEMiT: Multimodal Emotion, Irony, and Semiotic Relation Detection in Italian Social TV*  
**Audited Venue**: *IEEE Transactions on Affective Computing* (also suitable for *IEEE Transactions on Computational Social Systems* / *IEEE/ACM Transactions on Audio, Speech, and Language Processing*)  
**Evaluation Perspective**: Lead Associate Editor, Senior Reviewer, and Academic Co-Author  
**Report Destination**: `docs/IEEE_PEER_REVIEW_AND_AUDIT_REPORT.md`  
**Ground-Truth Reference Sources**:
- `code/bert/multiseed/jup_results/MULTISEED_SELECTED_CASES_REPORT.md` (Option B: 3-seed averages, seeds `42, 123, 456`, fixed split `42`, unscaled `raw` features)
- `datasets/annotations/fleiss_kappa_summary.csv` & `code/compute_fleiss_kappa_multiemit.py`
- `datasets/corpora/DATASET_LABEL_DISTRIBUTIONS.md`
- `datasets/corpora/correlations/eng/irony_multimodal_corr.csv`, `emotions_multimodal_corr.csv`, `relation_multimodal_corr.csv`
- Manuscript files: `paper/draft.tex`, `paper/1_introduction.tex`, `paper/2_related_works.tex`, `paper/3_dataset_analysis.tex`, `paper/4_experimental_design.tex`, `paper/5_results_discussion.tex`, `paper/6_conclusion.tex`

---

## 1. Executive Summary & IEEE Verdict

### Overall Recommendation: **Minor to Moderate Revision**
The manuscript presents **MultiEMiT**, an original, timely, and methodologically sound multimodal dataset of Italian Social TV commentary annotated across five pragmatic layers (Plutchik emotions, irony/sarcasm, offensiveness, broadcast target orientation, and Kruk's semiotic text–image relations). The experimental investigation is transparent and methodologically honest, contrasting compact discriminative Dual-Encoders against a 12B generative vision–language model (Pixtral 12B) under a strictly locked multi-seed protocol (3 seeds: `42, 123, 456`; split seed: `42`; unscaled `raw` features).

Crucially, our mathematical and empirical audit confirms that the baseline tables (Tables III, IV, V, VI, VII, VIII, IX, and X) in the manuscript align with the official multi-seed experimental outputs (`MULTISEED_SELECTED_CASES_REPORT.md`) with 100% numerical fidelity.

However, prior to publication in *IEEE Transactions on Affective Computing*, the authors must address:
1. **Critical Figure Subpanel Referencing Bug**: A severe mislabeling in Section V-C (line 255), where Semiotic Relation confusion matrix discussions cite panels `(d)`, `(e)`, and `(f)` instead of `(e)`, `(g)`, and `(h)` (panel `(d)` is an Irony matrix).
2. **Table I Fleiss' $\kappa$ Discrepancies**: Five arithmetic/rounding mismatches against the gold summary CSV (`Anticipation`, `Joy`, `Sadness`, `Neutral`, and `Offensive`) and hyphen-minus formatting.
3. **LaTeX Float and Label Hygiene**: Conflicting duplicate/quadruplicate `\label` declarations on Tables II and VI.
4. **Literature Gap**: Complete absence of related work on Vision–Language Models (VLMs / MLLMs) in affective computing in Section II, creating an unmotivated leap to Pixtral 12B in Section IV.
5. **Abstract Lacks Quantitative Benchmarks**: Purely qualitative narrative without the headline metrics that substantiate an IEEE paper.
6. **Unstated Methodological Nuances**: Explicitly articulating modality representation asymmetry (fine-tuned text encoder vs. off-the-shelf ImageNet ViT) and corpus-level Pearson correlation prompt computation scope for zero-shot Pixtral.

### Core Strengths:
* **Unique and Theoretically Grounded Dataset**: Fills an acute gap in non-English multimodal affective computing by combining Plutchik's psychoevolutionary dyads with Kruk's semiotic document intent on live television discourse.
* **Controlled Human Disambiguation Study**: Empirically demonstrates on $N=628$ paired posts that visual context boosts human agreement by $+15.6\%$ cumulative Fleiss' $\kappa$ (surging $+40.0\%$ on Irony and $+48.6\%$ on Offensiveness).
* **Nuanced & Honest Modeling Takeaways**: Instead of claiming blanket multimodal superiority, the paper reveals sharp task-dependent dynamics: visual features improve irony ($+7.5$ Macro $F_1$) unconditionally, require auxiliary task guidance on emotion ($+8.1$ Macro $F_1$), and fail to surpass text alone on semiotic relations under frozen concatenation.
* **Flawless Baseline Alignment**: Every metric in Tables III, IV, V, VI, VII, VIII, IX, and X matches the locked ground-truth logs in `MULTISEED_SELECTED_CASES_REPORT.md` to three decimal places.

---

## 2. Review and Challenge of Prior Investigation Findings

As required by our skeptical reviewer protocol, we re-audited every claim made in the previous worker's evaluation report against the source code and raw data files.

### 2.1 Refutation: Unironic-Parallel Pearson Correlation in Section III
* **Prior Claim**: The prior investigator claimed that Section III, line 194 erroneously reported Unironic-Parallel as `Parallel $r = 0.171$` instead of `$r = 0.168$`.
* **Evidence Cited by Prior Investigator**: Calculated Pearson correlation on `multimodal_full.csv` was claimed to be $0.1684$.
* **What the Code Actually Shows**:
  Evaluating Pearson correlation on the official project file `datasets/corpora/correlations/eng/irony_multimodal_corr.csv`:
  ```python
  import pandas as pd
  df = pd.read_csv('datasets/corpora/correlations/eng/irony_multimodal_corr.csv')
  r = df['None'].corr(df['Parallela'])
  print(r)  # Output: 0.171341
  ```
  $0.171341$ rounds exactly to **$0.171$**.
* **Corrected Finding**: **The manuscript's text `$r = 0.171$` is 100% correct.** The prior investigator's claim of an error was an erroneous false positive caused by calculating correlations on an unofficial or unaligned dataframe.

### 2.2 Extension: Complete Audit of Table I (Fleiss' $\kappa$)
* **Prior Claim**: The prior investigator identified only two issues in Table I: Anticipation visual gain (`+0.105` vs `+0.104`) and Joy text-only (`0.452` vs `0.451`).
* **What the Code Actually Shows**:
  Re-executing `code/compute_fleiss_kappa_multiemit.py` and auditing all cells against `datasets/annotations/fleiss_kappa_summary.csv` revealed **THREE additional discrepancies** missed by the prior worker:
  1. `Sadness` (line 76): Table I reports `0.273` for Paired Text-Only. The ground-truth CSV records `0.272` (and $0.312 - 0.272 = +0.040$, whereas the table reports $+0.039$).
  2. `Neutral` (line 77): Table I reports `0.360` for Paired With-Image. The ground-truth CSV records `0.359` (and $0.359 - 0.274 = +0.085$).
  3. `Offensive / Hate Speech` (line 93): Table I reports Delta Gain as `+0.180 (+48.6%)`. However, $0.549 - 0.370 = 0.179$ (and the CSV records `0.179`, $48.56\%$).
  4. Negative cells: Lines 68 and 72 use short hyphens (`-0.111`, `-0.062`) instead of LaTeX math minus (`$-0.111$`, `$-0.062$`).
* **Corrected Finding**: Table I contains five numerical discrepancies and formatting flaws that must be harmonized simultaneously.

### 2.3 Verification: Figure 9 Subpanel Mismatch (Section V-C, line 255)
* **Prior Claim**: Discussion of Semiotic Relation confusion matrices in Section V-C cites panels `(d)`, `(e)`, and `(f)` instead of `(e)`, `(g)`, and `(h)`.
* **Code Verification**:
  Inspecting `paper/imgs/confusion_matrices_irony_relation.svg`:
  - Panel `(a)`: `Irony: Text Only` (axes_1)
  - Panel `(b)`: `Irony: Text Only + Corr.` (axes_2)
  - Panel `(c)`: `Irony: With Images` (axes_3)
  - Panel `(d)`: `Irony: With Images + Corr.` (axes_4)
  - Panel `(e)`: `Relation: Text Only` (axes_5)
  - Panel `(f)`: `Relation: Text Only + Corr.` (axes_6)
  - Panel `(g)`: `Relation: With Images` (axes_7)
  - Panel `(h)`: `Relation: With Images + Corr.` (axes_8)
  In `paper/5_results_discussion.tex`, line 255:
  1. *"In the text-only probe (Fig.~\ref{fig:confusion_matrices}d)..."* $\rightarrow$ Text-only relation is panel **`(e)`**, not `(d)`. (Panel `(d)` is Irony with images and correlations).
  2. *"Fusing visual features in the Dual-Encoder (Fig.~\ref{fig:confusion_matrices}e) reduces this majority-class pull: misclassifications into Additive drop to 36.2% for Parallel (8.3 posts) and 38.6% for Divergent (14.7 posts)..."* $\rightarrow$ Dual-Encoder with images is panel **`(g)`**, not `(e)`.
  3. *"When correlation prompts are supplied (Fig.~\ref{fig:confusion_matrices}f), the model sharpens its Additive predictions (77.5% recall)..."* $\rightarrow$ Dual-Encoder with images and correlation prompts is panel **`(h)`**, not `(f)`.
* **Corrected Finding**: Verified and confirmed with exact SVG panel IDs.

### 2.4 Explanation: Multiple Label Declarations on Table VI
* **Prior Claim**: Table VI has quadruple duplicate labels (`tab:subset_all`, `tab:subset_emotion`, `tab:subset_irony`, `tab:subset_relation`).
* **Context Found**: The author declared four labels because the table covers all three tasks and is referenced from three distinct sections:
  - Line 113: `Table~\ref{tab:subset_emotion}` (in Section V-A)
  - Line 245: `Table~\ref{tab:subset_irony}` (in Section V-B)
  - Line 324: `Table~\ref{tab:subset_relation}` (in Section V-C)
* **Corrected Finding**: Consolidating Table VI to a single label `\label{tab:semiotic_subsets_all}` requires updating the three in-text references to `Table~\ref{tab:semiotic_subsets_all}` to prevent broken references.

---

## 3. Chapter-by-Chapter Technical Audit & Critique

### Front Matter & Abstract (`paper/draft.tex`)
* **Author Block & Headers**:
  - Contains placeholder names (`Author Author`, `Co-Author One`, `Co-Author Two`) and `\thanks{TODO}` tags.
  - Header mark: `\markboth{IEEE Transactions on Affective Computing,~Vol.~XX, No.~X, October~2026}`.
* **Abstract Critique**:
  - The abstract clearly describes the motivation (Social TV commentary, brief text, memes), outlines the 5 pragmatic layers, introduces both modeling paradigms (BERT-ViT Dual-Encoders and Pixtral 12B), and summarizes RQ1–RQ3.
  - **Major Weakness**: It is purely narrative, with zero quantitative numbers. Top-tier IEEE Transactions abstracts must highlight headline empirical metrics.
  - **Actionable Fix**: Insert concrete quantitative metrics into line 37:
    > *"...For irony detection, visual features provide an immediate and substantial boost (+7.5 points in Macro $F_1$, from 0.453 to 0.528). In contrast, for emotion recognition, images alone yield little benefit, helping only when combined with auxiliary task labels (+8.1 points, reaching 0.505). For semiotic relations, frozen feature concatenation fails to outperform text baselines, whereas fine-tuned Pixtral 12B achieves the highest overall score (0.613 Macro $F_1$) despite minority-class collapse."*
* **Keywords**: Add `"Semiotic relation detection"` and `"Correlation prompting"`.

---

### Chapter 1: Introduction (`paper/1_introduction.tex`)
* **Narrative Flow & Structure**:
  - Engaging opening linking Social TV, second-screen interactions, and the M.EMO.RAI project.
  - Directly anchors the modeling hypothesis in empirical human behavior: annotator agreement increased by $+15.6\%$ Fleiss' $\kappa$ when images were provided.
  - Clear, disciplined formulation of RQ1 (Visual Context), RQ2 (Correlation Prompting), and RQ3 (Architectural Paradigms).
* **Missing Details & Polish**:
  - **Contribution 1 Scale**: Contribution 1 (line 21) mentions three-annotator adjudication but fails to state the exact dataset size ($N=2,668$ total instances: $2,023$ text-only, $645$ multimodal). Stating sample size in Contribution 1 is an IEEE requirement.
  - **Typography**: Change hyphen `"text-image"` to en-dash `"text--image"` in line 8.

---

### Chapter 2: Related Work (`paper/2_related_works.tex`)
* **Literature Grounding**:
  - Thorough historical progression across Italian sentiment/emotion (SENTIPOLC, FEEL-IT, MultiEmotions-It, EMiT), Social TV analytics (M.EMO.RAI), and irony detection (IronITA).
  - Excellent coverage of Kruk's multimodal intent framework (Additive, Parallel, Divergent) and multimodal sarcasm (Schifanella et al., Cai et al.).
* **Major Blind Spot: Absence of Vision–Language Models (VLMs) in Affective Computing**:
  - Section II reviews only classic discriminative/textual models, making the benchmarking of Pixtral 12B in Section IV unmotivated.
  - **Actionable Fix**: Insert a dedicated paragraph reviewing Vision–Language Models in affective and subjective NLP:
    ```latex
    Recent advances in multimodal foundation models and Vision--Language Models (VLMs) have redefined multimodal affective computing. Contrastive vision--language pre-training, pioneered by CLIP \cite{radford2021clip}, established unified embedding spaces capable of zero-shot transfer, while revealing modality gaps that hinder linear probe performance on figurative tasks \cite{liang2022mindgap}. Concurrently, autoregressive multimodal large language models such as LLaVA \cite{liu2024llava} and Pixtral \cite{pixtral} integrate high-capacity vision encoders with autoregressive decoders, enabling joint reasoning over text and visual media through instruction tuning. While these generative architectures demonstrate remarkable zero-shot reasoning on broad benchmarks, their capability to decode non-literal pragmatic phenomena---such as sarcasm, subtle emotional blends, and semiotic dissonance in non-English social media commentary---remains under-explored. In this work, we contrast compact discriminative dual-encoders against Pixtral 12B to examine how parameter scale and generative versus discriminative objectives navigate complex multimodal Social TV discourse.
    ```
* **Stylistic Polish**:
  - Hyphenate compound adjectives: `"large-scale"`, `"language-specific"`, `"eight-emotion"`, `"multilevel"`, `"three-class"`, `"four-class"`.

---

### Chapter 3: Dataset Analysis (`paper/3_dataset_analysis.tex`)
* **Data Sources & Protocol**:
  - Details *Propaganda Live* (satirical political commentary, Poland-Belarus border crisis) and *Pechino Express* (adventure reality show).
  - Web platform Sensitag, 3 annotators, majority voting (2/3).
  - Informative pruning example in Fig. 3 (*"Terrible, I'm ashamed of being European"*).
* **Table I (Fleiss' $\kappa$, lines 54–104) Corrections**:
  - Line 68 (`Love`): change `-0.111` to `$-0.111$`.
  - Line 69 (`Anticipation`): change `+0.105` to `+0.104` ($0.573 - 0.469 = 0.104$).
  - Line 72 (`Joy`): change `0.452` to `0.451`, and `-0.062` to `$-0.062$`.
  - Line 76 (`Sadness`): change `0.273` to `0.272`.
  - Line 77 (`Neutral`): change `0.360` to `0.359`.
  - Line 93 (`Offensive`): change `+0.180 (+48.6%)` to `+0.179 (+48.6%)` ($0.549 - 0.370 = 0.179$).
* **Table II (Label Distributions, lines 114–163)**:
  - All numbers verified against `DATASET_LABEL_DISTRIBUTIONS.md` (100% match).
  - Delete duplicate label `\label{table:multi_dist}` (lines 117–118) to prevent label warnings.
  - Replace hyphens `-` with en-dashes `--` in the textual semiotic relation cells.
* **Pearson Correlations (lines 169–196)**:
  - All verified: Trust-Container ($r=0.229$), Anticipation-Container ($r=0.181$), Surprise-Content ($r=0.319$), Sadness-Content ($r=0.214$), Disgust-Content ($r=0.205$), Disgust-Sarcasm ($r=0.400$), Offensiveness-Sarcasm ($r=0.169$), Anger-Sarcasm ($r=0.158$), Irony-Divergent ($r=0.407$), Irony-Content ($r=0.272$), Irony-Joy ($r=0.185$), Unironic-Container ($r=0.294$), Unironic-Trust ($r=0.286$), Unironic-Additive ($r=0.260$), Unironic-Parallel ($r=0.171$).

---

### Chapter 4: Experimental Design (`paper/4_experimental_design.tex`)
* **Pipeline Architecture**:
  - 80/20 fixed split (`SPLIT_SEED=42`), 3 training seeds (`42, 123, 456`), two-step fine-tuning (FEEL-IT, UmBERTo), frozen ViT-Base-224 features ($d=1{,}536$), and linear SVM probes (`SGDClassifier` with `ClassifierChain`).
  - Pixtral 12B setup: RSLoRA ($r=16, \alpha=32$), regex parsing, $n=2$ truncation.
* **Methodological Nuances to Highlight**:
  - *Modality Asymmetry*: The text encoder undergoes domain fine-tuning in Step 2 on in-domain text (`me_train + mm_train`, 2,134 posts), whereas ViT is completely frozen with off-the-shelf ImageNet weights without in-domain fine-tuning.
  - *Correlation Prompt Scope*: Pixtral zero-shot prompts include Pearson correlation rules calculated over the entire multimodal dataset ($N=645$). In a non-oracle deployment, these rules should be computed strictly on `mm_train` ($N=516$).
* **Formatting**:
  - Line 7: format `$d=1536$` as `$d=1{,}536$`.
  - Line 25: `336 \times 336` $\rightarrow$ `$336 \times 336$`.

---

### Chapter 5: Results and Discussion (`paper/5_results_discussion.tex`)
* **Numerical Verification (100% Match with Single Source of Truth)**:
  - **Table III (Emotion Baselines)**:
    - Textual: Step 1 (`0.506 / 0.529`), Step 2 Raw (`0.515 / 0.528`), Step 2 Oracle (`0.530 / 0.555`).
    - Multimodal: Text only (`0.424 / 0.492`), Text + Corr (`0.426 / 0.499`), With images (`0.432 / 0.475`), With images + Corr (`0.505 / 0.529`).
    - Ablation (+rel): Step 2 Textual (`0.544 / 0.557`), Text + Corr + Rel (`0.439 / 0.515`), With images + Corr + Rel (`0.458 / 0.520`).
  - **Table VII (Irony Baselines)**:
    - Textual: Step 1 (`0.449 / 0.737`), Step 2 Raw (`0.506 / 0.760`), Step 2 Oracle (`0.533 / 0.765`).
    - Multimodal: Text only (`0.453 / 0.752`), Text + Corr (`0.441 / 0.729`), With images (`0.528 / 0.809`), With images + Corr (`0.455 / 0.760`).
    - Ablation (+rel): Step 2 Textual (`0.510 / 0.760`), Text + Corr + Rel (`0.425 / 0.760`), With images + Corr + Rel (`0.432 / 0.786`).
  - **Table IX (Semiotic Relation Baselines)**:
    - Ref Head: Without Corr (`0.541 / 0.662`), With Corr (`0.542 / 0.664`).
    - Linear Probe: Text only (`0.571 / 0.620`), Text + Corr (`0.603 / 0.628`), With images (`0.552 / 0.594`), With images + Corr (`0.585 / 0.633`).
  - **Table VI (Semiotic Subsets Breakdown)**:
    - Dual-Encoder and Pixtral subset metrics for ADD, PAR, DIV across all three tasks match the official experimental outputs.
  - **Pixtral Benchmarks (Tables IV, V, VIII, X)**: Fully verified against experimental logs; truncation peaks at $n=2$ ($0.389$ Macro $F_1$).

* **CRITICAL FINDING: Figure 9 Subpanel Misreferencing Bug (Line 255)**:
  - In line 255:
    > *"In the text-only probe (Fig.~\ref{fig:confusion_matrices}d)... Fusing visual features in the Dual-Encoder (Fig.~\ref{fig:confusion_matrices}e)... When correlation prompts are supplied (Fig.~\ref{fig:confusion_matrices}f)..."*
  - **Actionable Fix**:
    - Change `Fig.~\ref{fig:confusion_matrices}d` $\rightarrow$ `Fig.~\ref{fig:confusion_matrices}e` (Relation Text Only: 43.0% Divergent and 46.4% Parallel misclassified as Additive).
    - Change `Fig.~\ref{fig:confusion_matrices}e` $\rightarrow$ `Fig.~\ref{fig:confusion_matrices}g` (Relation With Images: Parallel misclassification drops to 36.2%, Divergent to 38.6%).
    - Change `Fig.~\ref{fig:confusion_matrices}f` $\rightarrow$ `Fig.~\ref{fig:confusion_matrices}h` (Relation With Images + Corr: Additive recall reaches 77.5%, Divergent correct 49.1%).
* **Table VI Quadruple Labels**: Consolidate lines 118–121 to `\label{tab:semiotic_subsets_all}` and update in-text call-sites in lines 113, 245, and 324.
* **Reporting Standard Deviations**: Baseline tables currently report only 3-seed means; incorporating $\pm \text{Std}$ in key rows or footnoting seed variances further reinforces IEEE statistical rigor.

---

### Chapter 6: Conclusion (`paper/6_conclusion.tex`)
* **Synthesis**:
  - Crisp and accurate synthesis of RQ1, RQ2, and RQ3.
  - Concrete future work: expanding broadcast genres, restoring broadcast provenance metadata, exploring end-to-end multimodal fine-tuning (e.g. CLIP/SigLIP), and digital TV engagement schemas.
* **Recommendation**: Mention adaptive-resolution vision architectures (e.g., NaViT) to address the meme aspect ratio distortion limitation highlighted in Section V-D.

---

## 4. Priority Action Items for Co-Authoring & Polishing

| # | Priority | File | Lines | Issue Description | Verified Actionable Fix |
|---|---|---|---|---|---|
| **1** | **CRITICAL** | `paper/5_results_discussion.tex` | 255 | Figure 9 subpanel mismatch: text discusses relation confusion matrices using panels `(d)`, `(e)`, and `(f)`. Panel `(d)` is Irony; relation panels are `(e)`, `(g)`, and `(h)`. | Replace `Fig.~\ref{fig:confusion_matrices}d` with `(e)`, `Fig.~\ref{fig:confusion_matrices}e` with `(g)`, and `Fig.~\ref{fig:confusion_matrices}f` with `(h)`. |
| **2** | **HIGH** | `paper/3_dataset_analysis.tex` | 68, 69, 72, 76, 77, 93 | Table I Fleiss' $\kappa$ arithmetic discrepancies & hyphen formatting. | 1. Anticipation gain: `+0.105` $\rightarrow$ `+0.104`<br>2. Joy text-only: `0.452` $\rightarrow$ `0.451`<br>3. Sadness text-only: `0.273` $\rightarrow$ `0.272`<br>4. Neutral with-image: `0.360` $\rightarrow$ `0.359`<br>5. Offensive delta: `+0.180` $\rightarrow$ `+0.179`<br>6. Replace negative hyphens with `$-$` (`$-0.111$`, `$-0.062$`). |
| **3** | **HIGH** | `paper/2_related_works.tex` | after 14 | Missing Vision–Language Literature: Pixtral 12B benchmark is ungrounded in Section II. | Add the proposed VLM literature paragraph citing CLIP \cite{radford2021clip}, LLaVA \cite{liu2024llava}, and Pixtral \cite{pixtral}. |
| **4** | **HIGH** | `paper/draft.tex` | 37 | Abstract lacks headline quantitative benchmarks. | Insert key empirical gains ($+7.5$ irony F1 boost, $+8.1$ emotion multimodal gain, $0.613$ Pixtral relation F1). |
| **5** | **MEDIUM** | `paper/3_dataset_analysis.tex` | 117–118 | Table II duplicate label declarations (`table:text_dist` and `table:multi_dist`). | Consolidate to `\label{tab:dataset_distributions}` and replace hyphens `-` with en-dashes `--`. |
| **6** | **MEDIUM** | `paper/5_results_discussion.tex` | 118–121, 113, 245, 324 | Table VI quadruple label declarations (`tab:subset_all`, `tab:subset_emotion`, `tab:subset_irony`, `tab:subset_relation`). | Consolidate to `\label{tab:semiotic_subsets_all}` and update in-text references at lines 113, 245, 324. |
| **7** | **MEDIUM** | `paper/1_introduction.tex` | 21 | Contribution 1 omits exact corpus scale. | Ground sample size: $N=2,668$ total instances ($2,023$ textual, $645$ multimodal). |
| **8** | **LOW** | `paper/4_experimental_design.tex` | 7, 25 | Typographical and math-mode formatting. | Format `$d=1536$` as `$d=1{,}536$` and enclose `336 \times 336` in math mode `$336 \times 336$`. |
| **9** | **LOW** | `paper/draft.tex` | 24–28 | Placeholder author blocks and `TODO` acknowledgments. | Fill in author affiliations, IEEE memberships, and institutional grant details. |

---

## 5. Unstated Limitations to Add to Section V-D

Insert the following ready-to-paste LaTeX block into `paper/5_results_discussion.tex` (under `\begin{itemize}` in line 330):

```latex
\item \textbf{Modality Representation Asymmetry:} In our Dual-Encoder architecture, the text encoder is sequentially fine-tuned on in-domain Social TV text before feature extraction, whereas the vision encoder (ViT-Base-224) remains completely frozen with generic ImageNet pre-training. This creates an asymmetric representation space where language representations are specialized for broadcast discourse while visual features represent domain-agnostic semantic primitives. Future work should investigate whether end-to-end multimodal fine-tuning or multimodal contrastive adaptation (e.g., CLIP-style pre-training) can overcome this imbalance.
\item \textbf{Scope of Correlation Prompting Statistics:} In our zero-shot Pixtral experiments, the Pearson correlation rules supplied in the prompt were calculated over the complete multimodal partition ($N = 645$). While this design serves our objective of establishing an upper-bound oracle probe of cross-layer dependencies, a fully non-oracle deployment must derive all verbalized association rules strictly from the training partition ($N = 516$) or learn them end-to-end via multi-task heads.
\item \textbf{Temporal Drift and Meme Degradation:} Social TV commentary relies heavily on ephemeral cultural references, second-screen inside jokes, and rapidly evolving memes tied to specific broadcast dates (e.g., late 2020 and late 2021). Classifiers evaluated in this setting are susceptible to temporal degradation as slang, political figures, and visual meme templates rotate over television seasons.
```

---

## 6. Remaining Questions & Gaps

* **Pixtral Multi-Seed Fine-Tuning**: While Dual-Encoder baselines report 3-seed means and standard deviations over training seeds (`42, 123, 456`), Pixtral 12B was evaluated over 3 generation runs at two decoding temperatures ($T=0.1$ and $T=0.9$) on a single LoRA training run. Multi-seed LoRA training would provide an even more balanced comparison against discriminative probes, though computational costs on 12B models are high.
* **Formal Significance Testing**: Paired bootstrap resampling or McNemar tests across the $N=129$ test split are not reported. While 3-seed agreement provides strong consistency evidence, reporting p-values would further appease strict IEEE reviewers.
* **Non-Oracle Correlation Heads**: Evaluating a secondary predictive model that estimates the auxiliary layers (e.g. predicting Target and Irony first, then feeding predictions into Emotion) would demonstrate how much of the oracle correlation gain translates to real-world deployment.
