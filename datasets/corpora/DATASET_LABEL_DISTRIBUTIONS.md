# MultiEMiT & External Corpora — Exact Label Distributions (`SEED = 42`)

This document provides the exact label distributions across **Training** ($80\%$) and **Testing / Validation** ($20\%$) splits for all datasets in `datasets/corpora/` used by the MultiEMiT experimental pipeline (`SPLIT_SEED = 42`, `test_size = 0.20`).

- **Multimodal Dataset** (`correctly_split/multimodal_train_seed42.csv` & `correctly_split/multimodal_test_seed42.csv`): Pre-split on `random_state=42` ($N_{\text{train}} = 516$, $N_{\text{test}} = 129$, Total $= 645$).
- **Textual Dataset** (`correctly_split/textual_only_cleaned.csv`): Split dynamically via `train_test_split(me_df, test_size=0.20, random_state=42)` in `train_textual_*.py` ($N_{\text{train}} = 1,618$, $N_{\text{test}} = 405$, Total $= 2,023$).
- **Combined MultiEMiT Textual Fine-Tuning Pool** (`me_train + mm_train`): Used during Step 2 of `train_textual_emotion.py` and `train_textual_irony.py` ($N_{\text{combined\_train}} = 1,618 + 516 = 2,134$), evaluated strictly on held-out `me_test` ($405$) or `mm_test` ($129$) with zero data leakage.

---

## 1. Dataset Partition Summary (`SEED = 42`)

| Dataset | Source File(s) in `datasets/corpora/` | Train ($80\%$) | Test / Val ($20\%$) | Total ($100\%$) | Split Method (`SEED = 42`) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **MultiEMiT Textual (`me`)** | `correctly_split/textual_only_cleaned.csv` | $1,618$ | $405$ | $2,023$ | `train_test_split(test_size=0.2, random_state=42)` |
| **MultiEMiT Multimodal (`mm`)** | `correctly_split/multimodal_train_seed42.csv`<br>`correctly_split/multimodal_test_seed42.csv` | $516$ | $129$ | $645$ | Pre-split (`random_state=42`) |
| **MultiEMiT Total (`me + mm`)** | Combined MultiEMiT Corpus | $2,134$ | $534$ | $2,668$ | Disjoint union of `me` and `mm` splits |
| **EMit-A (External Emotion)** | `emit/cleaned_emit_train_A.csv` | $4,772$ | $1,194$ | $5,966$ | `train_test_split(test_size=0.2, random_state=42)` |
| **IronITA (External Irony)** | `ironita/ironita_full.xlsx` | $3,879$ | $970$ | $4,849$ | `train_test_split(test_size=0.2, random_state=42, stratify=label)` |

---

## 2. MultiEMiT Textual Dataset (`textual_only_cleaned.csv`, Split on `SEED = 42`)

- **Total Samples**: $2,023$
- **Train Split (`me_train`)**: $1,618$ samples ($79.98\%$)
- **Test Split (`me_test`)**: $405$ samples ($20.02\%$)

### 2.1 Emotion Distribution (10-Class Multi-Label)

*Percentages indicate the proportion of posts in that split annotated with each emotion (`1` vs `0`). Since a post can carry multiple emotions, column percentages sum to more than $100\%$.*

| Emotion Label | `me_train` ($N = 1,618$) | `me_train` ($\%$) | `me_test` ($N = 405$) | `me_test` ($\%$) | Total Textual ($N = 2,023$) | Total ($\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fiducia** *(Trust)* | $364$ | $22.50\%$ | $83$ | $20.49\%$ | $447$ | $22.10\%$ |
| **Sorpresa** *(Surprise)* | $344$ | $21.26\%$ | $89$ | $21.98\%$ | $433$ | $21.40\%$ |
| **Gioia** *(Joy)* | $343$ | $21.20\%$ | $90$ | $22.22\%$ | $433$ | $21.40\%$ |
| **Disgusto** *(Disgust)* | $284$ | $17.55\%$ | $74$ | $18.27\%$ | $358$ | $17.70\%$ |
| **Amore** *(Love)* | $213$ | $13.16\%$ | $49$ | $12.10\%$ | $262$ | $12.95\%$ |
| **Neutro** *(Neutral)* | $189$ | $11.68\%$ | $52$ | $12.84\%$ | $241$ | $11.91\%$ |
| **Tristezza** *(Sadness)* | $186$ | $11.50\%$ | $42$ | $10.37\%$ | $228$ | $11.27\%$ |
| **Rabbia** *(Anger)* | $109$ | $6.74\%$ | $27$ | $6.67\%$ | $136$ | $6.72\%$ |
| **Anticipazione** *(Anticipation)* | $105$ | $6.49\%$ | $25$ | $6.17\%$ | $130$ | $6.43\%$ |
| **Paura** *(Fear)* | $41$ | $2.53\%$ | $12$ | $2.96\%$ | $53$ | $2.62\%$ |
| **Total Emotion Annotations** | **$2,178$** | **Avg $1.35$/post** | **$543$** | **Avg $1.34$/post** | **$2,721$** | **Avg $1.34$/post** |

#### Multi-Label Emotion Cardinality (Emotions per Post in Textual Dataset)

| Number of Active Emotions | `me_train` ($N = 1,618$) | `me_test` ($N = 405$) | Total Textual ($N = 2,023$) |
| :--- | :---: | :---: | :---: |
| **$0$ Emotions** *(unlabeled / neutral-implicit)* | $79$ ($4.88\%$) | $19$ ($4.69\%$) | $98$ ($4.84\%$) |
| **$1$ Emotion** | $1,032$ ($63.78\%$) | $259$ ($63.95\%$) | $1,291$ ($63.82\%$) |
| **$2$ Emotions** | $396$ ($24.47\%$) | $99$ ($24.44\%$) | $495$ ($24.47\%$) |
| **$3$ Emotions** | $93$ ($5.75\%$) | $26$ ($6.42\%$) | $119$ ($5.88\%$) |
| **$4$ Emotions** | $15$ ($0.93\%$) | $2$ ($0.49\%$) | $17$ ($0.84\%$) |
| **$5$ Emotions** | $3$ ($0.19\%$) | $0$ ($0.00\%$) | $3$ ($0.15\%$) |

---

### 2.2 Irony & Sarcasm Distribution (3-Class Single-Label)

*Mapped via `row_to_irony_class`: `Sarcasmo = 2`, `Ironia = 1`, `None = 0`. In the textual corpus, `Ironia` and `Sarcasmo` are strictly mutually exclusive ($0$ overlap).*

| Irony Class | `me_train` ($N = 1,618$) | `me_train` ($\%$) | `me_test` ($N = 405$) | `me_test` ($\%$) | Total Textual ($N = 2,023$) | Total ($\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0`: None** *(Neither)* | $1,248$ | $77.13\%$ | $318$ | $78.52\%$ | $1,566$ | $77.41\%$ |
| **`1`: Ironia** *(Irony)* | $344$ | $21.26\%$ | $79$ | $19.51\%$ | $423$ | $20.91\%$ |
| **`2`: Sarcasmo** *(Sarcasm)* | $26$ | $1.61\%$ | $8$ | $1.98\%$ | $34$ | $1.68\%$ |
| **Total** | **$1,618$** | **$100.00\%$** | **$405$** | **$100.00\%$** | **$2,023$** | **$100.00\%$** |

---

### 2.3 Broadcast Target Combinations (`D1`) & Auxiliary Pragmatic Annotations (Textual Dataset)

#### Mutually Exclusive Broadcast Target Breakdown (`Contenuto` vs. `Contenitore`, `D1`)

| Target Combination | `me_train` ($N = 1,618$) | `me_train` ($\%$) | `me_test` ($N = 405$) | `me_test` ($\%$) | Total Textual ($N = 2,023$) | Total ($\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Target** (`Contenuto=0, Contenitore=0`) | $238$ | $14.71\%$ | $54$ | $13.33\%$ | $292$ | $14.43\%$ |
| **Only `Contenuto`** (`Contenuto=1, Contenitore=0`) | $756$ | $46.72\%$ | $193$ | $47.65\%$ | $949$ | $46.91\%$ |
| **Only `Contenitore`** (`Contenuto=0, Contenitore=1`) | $577$ | $35.66\%$ | $144$ | $35.56\%$ | $721$ | $35.64\%$ |
| **Both Targets** (`Contenuto=1, Contenitore=1`) | $47$ | $2.90\%$ | $14$ | $3.46\%$ | $61$ | $3.02\%$ |
| **Total** | **$1,618$** | **$100.00\%$** | **$405$** | **$100.00\%$** | **$2,023$** | **$100.00\%$** |

#### Marginal Pragmatic & Target Counts (Textual Dataset)

| Pragmatic / Target Feature | `me_train` ($N = 1,618$) | `me_train` ($\%$) | `me_test` ($N = 405$) | `me_test` ($\%$) | Total Textual ($N = 2,023$) | Total ($\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Contenuto** *(Show Content Target — Marginal)* | $803$ | $49.63\%$ | $207$ | $51.11\%$ | $1,010$ | $49.93\%$ |
| **Contenitore** *(Show Container/Event Target — Marginal)* | $624$ | $38.57\%$ | $158$ | $39.01\%$ | $782$ | $38.66\%$ |
| **Citazione** *(Quotation)* | $130$ | $8.03\%$ | $40$ | $9.88\%$ | $170$ | $8.40\%$ |
| **Offensivita** *(Offensiveness)* | $36$ | $2.22\%$ | $9$ | $2.22\%$ | $45$ | $2.22\%$ |
| **Hate speech** | $1$ | $0.06\%$ | $0$ | $0.00\%$ | $1$ | $0.05\%$ |
| **Incomprensibile** | $0$ | $0.00\%$ | $0$ | $0.00\%$ | $0$ | $0.00\%$ |

---

## 3. MultiEMiT Multimodal Dataset (`multimodal_train_seed42.csv` & `multimodal_test_seed42.csv`)

- **Total Samples**: $645$
- **Train Split (`mm_train`)**: $516$ samples ($80.00\%$)
- **Test Split (`mm_test`)**: $129$ samples ($20.00\%$)

### 3.1 Emotion Distribution (10-Class Multi-Label)

| Emotion Label | `mm_train` ($N = 516$) | `mm_train` ($\%$) | `mm_test` ($N = 129$) | `mm_test` ($\%$) | Total Multimodal ($N = 645$) | Total ($\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fiducia** *(Trust)* | $172$ | $33.33\%$ | $42$ | $32.56\%$ | $214$ | $33.18\%$ |
| **Gioia** *(Joy)* | $139$ | $26.94\%$ | $29$ | $22.48\%$ | $168$ | $26.05\%$ |
| **Neutro** *(Neutral)* | $73$ | $14.15\%$ | $17$ | $13.18\%$ | $90$ | $13.95\%$ |
| **Sorpresa** *(Surprise)* | $66$ | $12.79\%$ | $16$ | $12.40\%$ | $82$ | $12.71\%$ |
| **Disgusto** *(Disgust)* | $59$ | $11.43\%$ | $19$ | $14.73\%$ | $78$ | $12.09\%$ |
| **Anticipazione** *(Anticipation)* | $51$ | $9.88\%$ | $10$ | $7.75\%$ | $61$ | $9.46\%$ |
| **Amore** *(Love)* | $39$ | $7.56\%$ | $11$ | $8.53\%$ | $50$ | $7.75\%$ |
| **Rabbia** *(Anger)* | $28$ | $5.43\%$ | $6$ | $4.65\%$ | $34$ | $5.27\%$ |
| **Tristezza** *(Sadness)* | $23$ | $4.46\%$ | $2$ | $1.55\%$ | $25$ | $3.88\%$ |
| **Paura** *(Fear)* | $3$ | $0.58\%$ | $1$ | $0.78\%$ | $4$ | $0.62\%$ |
| **Total Emotion Annotations** | **$653$** | **Avg $1.27$/post** | **$153$** | **Avg $1.19$/post** | **$806$** | **Avg $1.25$/post** |

#### Multi-Label Emotion Cardinality (Emotions per Post in Multimodal Dataset)

| Number of Active Emotions | `mm_train` ($N = 516$) | `mm_test` ($N = 129$) | Total Multimodal ($N = 645$) |
| :--- | :---: | :---: | :---: |
| **$1$ Emotion** | $405$ ($78.49\%$) | $108$ ($83.72\%$) | $513$ ($79.53\%$) |
| **$2$ Emotions** | $85$ ($16.47\%$) | $18$ ($13.95\%$) | $103$ ($15.97\%$) |
| **$3$ Emotions** | $26$ ($5.04\%$) | $3$ ($2.33\%$) | $29$ ($4.50\%$) |

---

### 3.2 Irony & Sarcasm Distribution (3-Class Single-Label)

*Mapped via `row_to_irony_class`: `Sarcasmo = 2`, `Ironia = 1`, `None = 0` (`Ironia` and `Sarcasmo` are strictly mutually exclusive in the corpus).*

| Irony Class | `mm_train` ($N = 516$) | `mm_train` ($\%$) | `mm_test` ($N = 129$) | `mm_test` ($\%$) | Total Multimodal ($N = 645$) | Total ($\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0`: None** *(Neither)* | $412$ | $79.84\%$ | $104$ | $80.62\%$ | $516$ | $80.00\%$ |
| **`1`: Ironia** *(Irony)* | $91$ | $17.64\%$ | $18$ | $13.95\%$ | $109$ | $16.90\%$ |
| **`2`: Sarcasmo** *(Sarcasm)* | $13$ | $2.52\%$ | $7$ | $5.43\%$ | $20$ | $3.10\%$ |
| **Total** | **$516$** | **$100.00\%$** | **$129$** | **$100.00\%$** | **$645$** | **$100.00\%$** |

---

### 3.3 Semiotic Image-Text Relation Distribution (3-Class Single-Label, `D6`)

*In `multimodal_train_seed42.csv` and `multimodal_test_seed42.csv`, every single row is strictly one-hot across `Additiva`, `Divergente`, and `Parallela` ($0$ double-labeled rows).*

| Relation Class (`row_to_relation_class`) | `mm_train` ($N = 516$) | `mm_train` ($\%$) | `mm_test` ($N = 129$) | `mm_test` ($\%$) | Total Multimodal ($N = 645$) | Total ($\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0`: Additiva (`ADD`)** | $321$ | $62.21\%$ | $68$ | $52.71\%$ | $389$ | $60.31\%$ |
| **`1`: Divergente (`DIV`)** | $120$ | $23.26\%$ | $38$ | $29.46\%$ | $158$ | $24.50\%$ |
| **`2`: Parallela (`PAR`)** | $75$ | $14.53\%$ | $23$ | $17.83\%$ | $98$ | $15.19\%$ |
| **Total** | **$516$** | **$100.00\%$** | **$129$** | **$100.00\%$** | **$645$** | **$100.00\%$** |

---

### 3.4 Cross-Tabulation: Semiotic Relation $\times$ Irony Class (Multimodal Dataset, `D4`)

*Highlights the strong empirical coupling between `Divergente` image-text relations and `Ironia`/`Sarcasmo`.*

| Split | Semiotic Relation | `0: None` | `1: Ironia` | `2: Sarcasmo` | Row Total | $\%$ Irony/Sarcasm within Relation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Train (`mm_train`, $N=516$)** | **Additiva (`0`)** | $283$ | $32$ | $6$ | $321$ | $11.84\%$ |
| | **Divergente (`1`)** | $58$ | $55$ | $7$ | $120$ | **$51.67\%$** |
| | **Parallela (`2`)** | $71$ | $4$ | $0$ | $75$ | $5.33\%$ |
| **Test (`mm_test`, $N=129$)** | **Additiva (`0`)** | $61$ | $4$ | $3$ | $68$ | $10.29\%$ |
| | **Divergente (`1`)** | $20$ | $14$ | $4$ | $38$ | **$47.37\%$** |
| | **Parallela (`2`)** | $23$ | $0$ | $0$ | $23$ | $0.00\%$ |
| **Total (`mm`, $N=645$)** | **Additiva (`0`)** | $344$ | $36$ | $9$ | $389$ | $11.57\%$ |
| | **Divergente (`1`)** | $78$ | $69$ | $11$ | $158$ | **$50.63\%$** |
| | **Parallela (`2`)** | $94$ | $4$ | $0$ | $98$ | $4.08\%$ |

---

### 3.5 Broadcast Target Combinations (`D1`) & Auxiliary Pragmatic Annotations (Multimodal Dataset)

#### Mutually Exclusive Broadcast Target Breakdown (`Contenuto` vs. `Contenitore`, `D1`)

| Target Combination | `mm_train` ($N = 516$) | `mm_train` ($\%$) | `mm_test` ($N = 129$) | `mm_test` ($\%$) | Total Multimodal ($N = 645$) | Total ($\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Target** (`Contenuto=0, Contenitore=0`) | $52$ | $10.08\%$ | $6$ | $4.65\%$ | $58$ | $8.99\%$ |
| **Only `Contenuto`** (`Contenuto=1, Contenitore=0`) | $148$ | $28.68\%$ | $44$ | $34.11\%$ | $192$ | $29.77\%$ |
| **Only `Contenitore`** (`Contenuto=0, Contenitore=1`) | $311$ | $60.27\%$ | $77$ | $59.69\%$ | $388$ | $60.16\%$ |
| **Both Targets** (`Contenuto=1, Contenitore=1`) | $5$ | $0.97\%$ | $2$ | $1.55\%$ | $7$ | $1.09\%$ |
| **Total** | **$516$** | **$100.00\%$** | **$129$** | **$100.00\%$** | **$645$** | **$100.00\%$** |

#### Marginal Pragmatic & Target Counts (Multimodal Dataset)

| Pragmatic / Target Feature | `mm_train` ($N = 516$) | `mm_train` ($\%$) | `mm_test` ($N = 129$) | `mm_test` ($\%$) | Total Multimodal ($N = 645$) | Total ($\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Contenitore** *(Show Container/Event Target — Marginal)* | $316$ | $61.24\%$ | $79$ | $61.24\%$ | $395$ | $61.24\%$ |
| **Contenuto** *(Show Content Target — Marginal)* | $153$ | $29.65\%$ | $46$ | $35.66\%$ | $199$ | $30.85\%$ |
| **Citazione** *(Quotation)* | $40$ | $7.75\%$ | $10$ | $7.75\%$ | $50$ | $7.75\%$ |
| **Offensivita** *(Offensiveness)* | $4$ | $0.78\%$ | $2$ | $1.55\%$ | $6$ | $0.93\%$ |
| **Hate speech** | $1$ | $0.19\%$ | $0$ | $0.00\%$ | $1$ | $0.16\%$ |
| **Incomprensibile** | $0$ | $0.00\%$ | $0$ | $0.00\%$ | $0$ | $0.00\%$ |

---

## 4. Combined MultiEMiT Textual Fine-Tuning Pool (`me_train + mm_train`, $N = 2,134$)

In Step 2 of `train_textual_emotion.py` and `train_textual_irony.py`, the textual encoder is fine-tuned on the combined training splits of the textual and multimodal corpora (`pd.concat([me_train, mm_train])`, $N = 2,134$) and evaluated on `me_test` ($N = 405$).

### 4.1 Combined Emotion Distribution

| Emotion Label | Combined Train (`me_train + mm_train`, $N = 2,134$) | Combined Train ($\%$) | Combined Test (`me_test + mm_test`, $N = 534$) | Combined Test ($\%$) | Full MultiEMiT ($N = 2,668$) | Full ($\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fiducia** | $536$ | $25.12\%$ | $125$ | $23.41\%$ | $661$ | $24.78\%$ |
| **Gioia** | $482$ | $22.59\%$ | $119$ | $22.28\%$ | $601$ | $22.53\%$ |
| **Sorpresa** | $410$ | $19.21\%$ | $105$ | $19.66\%$ | $515$ | $19.30\%$ |
| **Disgusto** | $343$ | $16.07\%$ | $93$ | $17.42\%$ | $436$ | $16.34\%$ |
| **Neutro** | $262$ | $12.28\%$ | $69$ | $12.92\%$ | $331$ | $12.41\%$ |
| **Amore** | $252$ | $11.81\%$ | $60$ | $11.24\%$ | $312$ | $11.69\%$ |
| **Tristezza** | $209$ | $9.79\%$ | $44$ | $8.24\%$ | $253$ | $9.48\%$ |
| **Anticipazione** | $156$ | $7.31\%$ | $35$ | $6.55\%$ | $191$ | $7.16\%$ |
| **Rabbia** | $137$ | $6.42\%$ | $33$ | $6.18\%$ | $170$ | $6.37\%$ |
| **Paura** | $44$ | $2.06\%$ | $13$ | $2.43\%$ | $57$ | $2.14\%$ |

### 4.2 Combined Irony & Sarcasm Distribution

| Irony Class | Combined Train (`me_train + mm_train`, $N = 2,134$) | Combined Train ($\%$) | Combined Test (`me_test + mm_test`, $N = 534$) | Combined Test ($\%$) | Full MultiEMiT ($N = 2,668$) | Full ($\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0`: None** | $1,660$ | $77.79\%$ | $422$ | $79.03\%$ | $2,082$ | $78.04\%$ |
| **`1`: Ironia** | $435$ | $20.38\%$ | $97$ | $18.16\%$ | $532$ | $19.94\%$ |
| **`2`: Sarcasmo** | $39$ | $1.83\%$ | $15$ | $2.81\%$ | $54$ | $2.02\%$ |
| **Total** | **$2,134$** | **$100.00\%$** | **$534$** | **$100.00\%$** | **$2,668$** | **$100.00\%$** |

---

## 5. External Pre-Training Datasets (Step 1 Warm-Up Splits on `SEED = 42`)

### 5.1 EMit-A (`emit/cleaned_emit_train_A.csv`, Split 80/20 on `SEED = 42`)

Used in Step 1 of `train_textual_emotion.py` via `train_test_split(emit_df, test_size=0.20, random_state=42)`.

| Emotion Label | `emit_train` ($N = 4,772$) | `emit_train` ($\%$) | `emit_val` ($N = 1,194$) | `emit_val` ($\%$) | Total EMit-A ($N = 5,966$) | Total ($\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fiducia** | $1,341$ | $28.10\%$ | $324$ | $27.14\%$ | $1,665$ | $27.91\%$ |
| **Neutro** | $1,058$ | $22.17\%$ | $264$ | $22.11\%$ | $1,322$ | $22.16\%$ |
| **Disgusto** | $702$ | $14.71\%$ | $172$ | $14.41\%$ | $874$ | $14.65\%$ |
| **Gioia** | $527$ | $11.04\%$ | $123$ | $10.30\%$ | $650$ | $10.90\%$ |
| **Amore** | $513$ | $10.75\%$ | $120$ | $10.05\%$ | $633$ | $10.61\%$ |
| **Sorpresa** | $466$ | $9.77\%$ | $125$ | $10.47\%$ | $591$ | $9.91\%$ |
| **Tristezza** | $438$ | $9.18\%$ | $107$ | $8.96\%$ | $545$ | $9.14\%$ |
| **Anticipazione** | $433$ | $9.07\%$ | $114$ | $9.55\%$ | $547$ | $9.17\%$ |
| **Rabbia** | $293$ | $6.14\%$ | $74$ | $6.20\%$ | $367$ | $6.15\%$ |
| **Paura** | $75$ | $1.57\%$ | $16$ | $1.34\%$ | $91$ | $1.53\%$ |

### 5.2 IronITA (`ironita/ironita_full.xlsx`, Stratified Split 80/20 on `SEED = 42`)

Used in Step 1 of `train_textual_irony.py` via `train_test_split(iron_df, test_size=0.20, random_state=42, stratify=iron_df["label"])`. Labels are mapped via `map_ironita_row` (`sarcasm == 1 -> 2`, `irony == 1 -> 1`, else `0`).

| Irony Class | `iron_train` ($N = 3,879$) | `iron_train` ($\%$) | `iron_val` ($N = 970$) | `iron_val` ($\%$) | Total IronITA ($N = 4,849$) | Total ($\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0`: None** | $1,913$ | $49.32\%$ | $478$ | $49.28\%$ | $2,391$ | $49.31\%$ |
| **`1`: Ironia** | $1,063$ | $27.40\%$ | $266$ | $27.42\%$ | $1,329$ | $27.41\%$ |
| **`2`: Sarcasmo** | $903$ | $23.28\%$ | $226$ | $23.30\%$ | $1,129$ | $23.28\%$ |
| **Total** | **$3,879$** | **$100.00\%$** | **$970$** | **$100.00\%$** | **$4,849$** | **$100.00\%$** |
