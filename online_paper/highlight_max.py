import re

with open('5_results_discussion.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Emotion max
content = content.replace(
    r"Pixtral (long\_v1) & FT ($n=2$) & 0.426 & 0.359 \\",
    r"Pixtral (long\_v1) & FT ($n=2$) & \textbf{0.426} & 0.359 \\"
)
content = content.replace(
    r"Pixtral (short) & FT ($n=2$) & 0.397 & 0.373 \\",
    r"Pixtral (short) & FT ($n=2$) & 0.397 & \textbf{0.373} \\"
)

# Irony max
content = content.replace(
    r"Pixtral (long\_v2) & ZS (With Corr.) & 0.512 & 0.396 \\",
    r"Pixtral (long\_v2) & ZS (With Corr.) & \textbf{0.512} & 0.396 \\"
)
content = content.replace(
    r"Pixtral (long\_v2) & FT (all predictions) & 0.405 & 0.454 \\",
    r"Pixtral (long\_v2) & FT (all predictions) & 0.405 & \textbf{0.454} \\"
)

# Relation max
content = content.replace(
    r"Pixtral (long\_v1) & FT (all predictions) & 0.613 & 0.574 \\",
    r"Pixtral (long\_v1) & FT (all predictions) & \textbf{0.613} & \textbf{0.574} \\"
)

with open('5_results_discussion.tex', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated Pixtral max highlights.")
