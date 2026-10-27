import re

with open('5_results_discussion.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add Emotion Complete Comparison Table
emotion_table = """
When evaluating Pixtral 12B, the fine-tuned version achieved an F1-score of 0.426. While lower than the best textual baseline, it is competitive with the discriminative Dual-Encoder, highlighting that generative LLMs can effectively learn multi-label distributions through LoRA fine-tuning.

\\begin{table}[htb!]
\\centering
\\caption{Complete Comparison: Pixtral vs Dual-Encoder in Emotion Detection ($T=0.1$ and $T=0.9$)}
\\label{tab:emotion_pixtral}
\\begin{tabular}{l l c c}
\\hline
\\textbf{Model} & \\textbf{Configuration} & \\textbf{T = 0.1 F1} & \\textbf{T = 0.9 F1} \\\\
\\hline
\\multicolumn{4}{l}{\\textit{Discriminative Multimodal (Dual-Encoder)}} \\\\
BERT-ViT & FT (With Corr.) & \\textbf{0.450} & \\textbf{0.450} \\\\
BERT-ViT & FT (No Corr.) & 0.430 & 0.430 \\\\
\\hline
\\multicolumn{4}{l}{\\textit{Generative (Pixtral 12B)}} \\\\
Pixtral (long\\_v1) & FT ($n=2$) & 0.426 & 0.359 \\\\
Pixtral (short) & FT ($n=2$) & 0.397 & 0.373 \\\\
Pixtral (short) & FT (all predictions) & 0.318 & 0.312 \\\\
Pixtral (short) & ZS (With Corr.) & 0.291 & 0.251 \\\\
Pixtral (long\\_v1) & ZS (With Corr.) & 0.275 & 0.266 \\\\
Pixtral (medium) & ZS (With Corr.) & 0.269 & 0.262 \\\\
Pixtral (long\\_v2) & ZS (No Corr.) & 0.266 & 0.237 \\\\
Pixtral (long\\_v1) & ZS (No Corr.) & 0.261 & 0.261 \\\\
Pixtral (short) & ZS (No Corr.) & 0.261 & 0.291 \\\\
Pixtral (long\\_v3) & ZS (With Corr.) & 0.258 & 0.262 \\\\
\\hline
\\end{tabular}
\\end{table}
"""

content = content.replace(
    "When evaluating Pixtral 12B, the fine-tuned version achieved an F1-score of 0.426. While lower than the best textual baseline, it is competitive with the discriminative Dual-Encoder, highlighting that generative LLMs can effectively learn multi-label distributions through LoRA fine-tuning.",
    emotion_table.strip()
)

# 2. Replace Emotion Subset Analysis
emotion_subset_old = """For the Dual-Encoder, the performance on the subsets was:
\\begin{itemize}
    \\item Without correlations: ADD: 0.47, PAR: 0.52, DIV: 0.38
    \\item With correlations: ADD: 0.54, PAR: 0.60, DIV: 0.42
\\end{itemize}
For the fine-tuned Pixtral model:
\\begin{itemize}
    \\item Fine-tuned: ADD: 0.33, PAR: 0.41, DIV: 0.41
\\end{itemize}"""

emotion_subset_new = """Table \\ref{tab:subset_emotion} presents the performance breakdown.
\\begin{table}[htb!]
\\centering
\\caption{Emotion Detection Micro F1-score across Semiotic Subsets}
\\label{tab:subset_emotion}
\\begin{tabular}{l c c c}
\\hline
\\textbf{Model Setting} & \\textbf{ADD} & \\textbf{PAR} & \\textbf{DIV} \\\\
\\hline
\\multicolumn{4}{l}{\\textit{Dual-Encoder}} \\\\
No correlations & 0.47 & 0.52 & 0.38 \\\\
With correlations & \\textbf{0.54} & \\textbf{0.60} & 0.42 \\\\
\\hline
\\multicolumn{4}{l}{\\textit{Pixtral 12B}} \\\\
Zero-Shot (No corr.) & 0.29 & 0.23 & \\textbf{0.44} \\\\
Zero-Shot (With corr.) & 0.32 & 0.30 & 0.43 \\\\
Fine-tuned (No corr.) & 0.33 & 0.41 & 0.41 \\\\
\\hline
\\end{tabular}
\\end{table}"""

content = content.replace(emotion_subset_old, emotion_subset_new.strip())

# 3. Replace Irony Subset Analysis
irony_subset_old = """Analyzing the subsets for Irony Detection, the Dual-Encoder achieved:
\\begin{itemize}
    \\item Without correlations: ADD: 0.83, PAR: 1.00, DIV: 0.55
    \\item With correlations: ADD: 0.88, PAR: 1.00, DIV: 0.68
\\end{itemize}
Fine-tuned Pixtral achieved:
\\begin{itemize}
    \\item Fine-tuned: ADD: 0.87, PAR: 1.00, DIV: 0.58
\\end{itemize}"""

irony_subset_new = """Analyzing the subsets for Irony Detection (Table \\ref{tab:subset_irony}), we observe distinct behaviors.
\\begin{table}[htb!]
\\centering
\\caption{Irony Detection Micro F1-score across Semiotic Subsets}
\\label{tab:subset_irony}
\\begin{tabular}{l c c c}
\\hline
\\textbf{Model Setting} & \\textbf{ADD} & \\textbf{PAR} & \\textbf{DIV} \\\\
\\hline
\\multicolumn{4}{l}{\\textit{Dual-Encoder}} \\\\
No correlations & 0.83 & \\textbf{1.00} & 0.55 \\\\
With correlations & \\textbf{0.88} & \\textbf{1.00} & \\textbf{0.68} \\\\
\\hline
\\multicolumn{4}{l}{\\textit{Pixtral 12B}} \\\\
Zero-Shot (No corr.) & 0.14 & 0.00 & 0.18 \\\\
Zero-Shot (With corr.) & 0.57 & 0.78 & 0.57 \\\\
Fine-tuned (No corr.) & 0.87 & \\textbf{1.00} & 0.58 \\\\
\\hline
\\end{tabular}
\\end{table}"""

content = content.replace(irony_subset_old, irony_subset_new.strip())


# 4. Replace Relation Subset Analysis
relation_subset_old = """The subset performance for Relation Detection further highlights architectural differences. The Dual-Encoder (with correlations) scored: ADD: 0.77, PAR: 0.59, DIV: 0.72.
Fine-tuned Pixtral scored: ADD: 0.71, PAR: 0.28, DIV: 0.60."""

relation_subset_new = """The subset performance for Relation Detection (Table \\ref{tab:subset_relation}) further highlights architectural differences.
\\begin{table}[htb!]
\\centering
\\caption{Relation Detection Micro F1-score across Semiotic Subsets}
\\label{tab:subset_relation}
\\begin{tabular}{l c c c}
\\hline
\\textbf{Model Setting} & \\textbf{ADD} & \\textbf{PAR} & \\textbf{DIV} \\\\
\\hline
\\multicolumn{4}{l}{\\textit{Dual-Encoder}} \\\\
No correlations & 0.73 & \\textbf{0.61} & 0.59 \\\\
With correlations & \\textbf{0.77} & 0.59 & \\textbf{0.72} \\\\
\\hline
\\multicolumn{4}{l}{\\textit{Pixtral 12B}} \\\\
Zero-Shot (No corr.) & 0.62 & 0.00 & 0.56 \\\\
Zero-Shot (With corr.) & 0.61 & 0.00 & 0.56 \\\\
Three-Shot (With corr.) & 0.60 & 0.17 & 0.64 \\\\
Fine-tuned (No corr.) & 0.71 & 0.28 & 0.60 \\\\
\\hline
\\end{tabular}
\\end{table}"""

content = content.replace(relation_subset_old, relation_subset_new.strip())

with open('5_results_discussion.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print("Tables updated successfully.")
