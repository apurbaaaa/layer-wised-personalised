import re

with open('text') as f:
    content = f.read()
lines = content.split('\n')

# Extract all cite keys
cites = set()
cite_lines = {}
for i, line in enumerate(lines, 1):
    for m in re.finditer(r'\\cite\{([^}]+)\}', line):
        for key in m.group(1).split(','):
            k = key.strip()
            cites.add(k)
            if k not in cite_lines:
                cite_lines[k] = i

# Extract all bibitem keys
bibitems = set()
bib_lines = {}
for i, line in enumerate(lines, 1):
    m = re.search(r'\\bibitem\{([^}]+)\}', line)
    if m:
        k = m.group(1).strip()
        bibitems.add(k)
        bib_lines[k] = i

print('=== CITE KEYS (used in text) ===')
for k in sorted(cites):
    print(f'  {k}  (line {cite_lines[k]})')
print(f'Total: {len(cites)}')

print()
print('=== BIBITEM KEYS (defined in bibliography) ===')
for k in sorted(bibitems):
    print(f'  {k}  (line {bib_lines[k]})')
print(f'Total: {len(bibitems)}')

print()
missing_bib = cites - bibitems
if missing_bib:
    print('=== CITED BUT NO BIBITEM (will show [?]) ===')
    for k in sorted(missing_bib):
        print(f'  {k}  (cited at line {cite_lines[k]})')
else:
    print('=== All citations have matching bibitems ===')

print()
unused_bib = bibitems - cites
if unused_bib:
    print('=== BIBITEM DEFINED BUT NEVER CITED ===')
    for k in sorted(unused_bib):
        print(f'  {k}  (defined at line {bib_lines[k]})')
else:
    print('=== All bibitems are cited ===')
