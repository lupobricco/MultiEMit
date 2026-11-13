# MultiEMiT Pearson Correlation Audit & Verification Report

**Dataset Source**: `datasets/splits/multimodal_full.csv` ($N = 645$)

## Section III Pearson Correlation Claims Verification

| Section | Variables | Exact $r$ | Rounded $r$ | Paper Reported | Status | Description |
|---|---|---|---|---|---|---|
| Emotions (Line 192) | Trust & Container | `0.229420` | `0.229` | `0.229` | **MATCH** | Program-directed Trust vs Container |
| Emotions (Line 192) | Anticipation & Container | `0.180993` | `0.181` | `0.181` | **MATCH** | Program-directed Anticipation vs Container |
| Emotions (Line 192) | Surprise & Content | `0.319429` | `0.319` | `0.319` | **MATCH** | Content-reactive Surprise vs Content |
| Emotions (Line 192) | Sadness & Content | `0.213668` | `0.214` | `0.214` | **MATCH** | Content-reactive Sadness vs Content |
| Emotions (Line 192) | Disgust & Content | `0.205230` | `0.205` | `0.205` | **MATCH** | Content-reactive Disgust vs Content |
| Emotions (Line 192) | Sarcasm & Disgust | `0.400005` | `0.400` | `0.400` | **MATCH** | Sarcasm hostile pattern vs Disgust |
| Emotions (Line 192) | Sarcasm & Offensiveness | `0.169007` | `0.169` | `0.169` | **MATCH** | Sarcasm vs Offensive language |
| Emotions (Line 192) | Sarcasm & Anger | `0.157933` | `0.158` | `0.158` | **MATCH** | Sarcasm vs Anger |
| Irony (Line 194) | Irony & Divergent | `0.406915` | `0.407` | `0.407` | **MATCH** | Irony vs Divergent semiotic relation |
| Irony (Line 194) | Irony & Content | `0.272034` | `0.272` | `0.272` | **MATCH** | Irony vs Content target |
| Irony (Line 194) | Irony & Joy | `0.184847` | `0.185` | `0.185` | **MATCH** | Irony vs Joy (playful tension) |
| Irony (Line 194) | Unironic & Container | `0.294356` | `0.294` | `0.294` | **MATCH** | Sincere posts vs Container |
| Irony (Line 194) | Unironic & Trust | `0.286466` | `0.286` | `0.286` | **MATCH** | Sincere posts vs Trust |
| Irony (Line 194) | Unironic & Additive | `0.259848` | `0.260` | `0.260` | **MATCH** | Sincere posts vs Additive relation |
| Irony (Line 194) | Unironic & Parallel | `0.168445` | `0.168` | `0.171` | **MINOR_DIFF** | Sincere posts vs Parallel relation |
| Relations (Line 196) | Divergent & Irony | `0.406915` | `0.407` | `0.407` | **MATCH** | Divergent relation vs Irony |
| Relations (Line 196) | Divergent & Content | `0.251732` | `0.252` | `0.252` | **MATCH** | Divergent relation vs Content target |
| Relations (Line 196) | Divergent & Surprise | `0.193856` | `0.194` | `0.194` | **MATCH** | Divergent relation vs Surprise |
| Relations (Line 196) | Additive & Container | `0.193665` | `0.194` | `0.194` | **MATCH** | Additive relation vs Container target |

### Audit Note on Unironic-Parallel Correlation
- On `datasets/splits/multimodal_full.csv` ($N = 645$), `Unironic` ($N=516$) vs `Parallel` ($N=98$) yields $r = 0.168445$ (rounds to $0.168$).
- The paper text and Figure 8 heatmap report $r = 0.171$, which corresponds to the legacy unaligned split ($N=100$ Parallel instances, $r = 0.171341$).
- Cross-check against legacy corpus `datasets/corpora/correlations/eng/irony_multimodal_corr.csv` ($N = 645$ with 100 Parallel instances): `Unironic` vs `Parallel` yields exact $r = 0.171341$ (rounds to `0.171`, matching Figure 8 and paper text exactly).
- The difference is negligible ($\Delta = 0.003$) and both substantiate the exact same substantive empirical takeaway: sincere commentary positively associates with literal, parallel text-image relations.
