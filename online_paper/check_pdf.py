import pypdf

reader = pypdf.PdfReader('draft.pdf')
print(f"Total pages: {len(reader.pages)}")
for i, p in enumerate(reader.pages):
    txt = p.extract_text()
    tables = [f"TABLE {t}" for t in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X'] if f"TABLE {t}" in txt]
    sections = [s for s in ['I.', 'II.', 'III.', 'IV.', 'V.', 'VI.', 'REFERENCES'] if s in txt]
    print(f"Page {i+1}: length={len(txt)}, tables={tables}, sections={sections}")

