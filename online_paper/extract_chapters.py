import os

draft_path = 'draft.tex'
intro_path = '1_introduction.tex'
related_path = '2_related_works.tex'

with open(draft_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

intro_start = -1
related_start = -1
inputs_start = -1

for i, line in enumerate(lines):
    if line.startswith('\\section{Introduction}'):
        intro_start = i
    elif line.startswith('\\section{Related Works}'):
        related_start = i
    elif line.startswith('\\input{3_dataset_analysis.tex}'):
        inputs_start = i

if intro_start != -1 and related_start != -1 and inputs_start != -1:
    intro_content = "".join(lines[intro_start:related_start])
    related_content = "".join(lines[related_start:inputs_start])
    
    with open(intro_path, 'w', encoding='utf-8') as f:
        f.write(intro_content)
        
    with open(related_path, 'w', encoding='utf-8') as f:
        f.write(related_content)
        
    new_draft_content = "".join(lines[:intro_start]) + \
                        "\\input{1_introduction.tex}\n" + \
                        "\\input{2_related_works.tex}\n" + \
                        "".join(lines[inputs_start:])
                        
    with open(draft_path, 'w', encoding='utf-8') as f:
        f.write(new_draft_content)
        
    print("Successfully extracted 1_introduction.tex and 2_related_works.tex")
else:
    print("Could not find section markers")
