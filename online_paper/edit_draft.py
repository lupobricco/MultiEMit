import re

with open('draft.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the index of \section{Dataset Analysis}
start_idx = content.find('\\section{Dataset Analysis}')

# Find the index of \section*{Acknowledgments}
end_idx = content.find('\\section*{Acknowledgments}')

new_content = content[:start_idx] + """\\input{3_dataset_analysis.tex}
\\input{4_experimental_design.tex}
\\input{5_results_discussion.tex}
\\input{6_conclusion.tex}

""" + content[end_idx:]

# Also, add the missing bib items before \end{thebibliography}
bib_idx = new_content.find('\\end{thebibliography}')
extra_bib = """\\bibitem{multisampledropout} H. Inoue, "Multi-Sample Dropout for Accelerated Training and Better Generalization," arXiv preprint arXiv:1905.09788, 2019.
\\bibitem{pixtral} Mistral AI, "Pixtral 12B," [Online]. Available: https://mistral.ai/news/pixtral-12b/, 2024.
"""
new_content = new_content[:bib_idx] + extra_bib + new_content[bib_idx:]

with open('draft.tex', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Successfully updated draft.tex")
