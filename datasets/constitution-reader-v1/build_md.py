import re, html, json
ROMAN = ['I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII','XIII','XIV','XV','XVI','XVII','XVIII','XIX','XX','XXI','XXII','XXIII','XXIV','XXV','XXVI','XXVII']

def clean(t):
    t = re.sub(r'<a\b[^>]*>|</a>', '', t)
    t = re.sub(r'<[^>]+>', '', t)
    t = html.unescape(t).replace('\xa0', ' ')
    t = t.replace('*', '')
    return re.sub(r'\s+', ' ', t).strip()

def blocks(s):
    """(tag, text) for every h2/h3/p in order, from the main content."""
    out = []
    for m in re.finditer(r'<(h2|h3|p)\b[^>]*>(.*?)</\1>', s, re.S):
        tag, inner = m.group(1), m.group(2)
        cls = re.search(r'class="([^"]*)"', m.group(0)[:m.group(0).find('>')])
        out.append((tag, clean(inner), cls.group(1) if cls else ''))
    return out

sections = []  # (title, [paragraphs])

# ── the seven articles
s = open('constitution-transcript.html').read()
s = s[s.find('We the People'):]
s = s[:s.find('The Word, "the," being interlined')]
bl = blocks('<p>' + s)  # the preamble p was cut in half; restore its open tag
sections.append(('Preamble', [bl[0][1]]))
article = None
for tag, text, cls in bl[1:]:
    if tag == 'h2':
        article = 'Article ' + re.search(r'Article\.? ?([IVX]+)', text).group(1)
        cur = None
    elif tag == 'h3':
        sec = re.search(r'Section\.? ?(\d+)', text).group(1)
        cur = f'{article}, Section {sec}'
        sections.append((cur, []))
    elif tag == 'p' and text:
        if cur is None:  # an article with no sections (V, VI, VII)
            cur = article; sections.append((cur, []))
        sections[-1][1].append(text)

# ── amendments 1–10
s = open('bill-of-rights-transcript.html').read()
s = s[s.find('<h3>Amendment I</h3>'):]
s = s[:s.find('</section>')] if '</section>' in s else s
cur = None
for tag, text, cls in blocks(s):
    if tag == 'h3' and text.startswith('Amendment'):
        cur = text.strip(); sections.append((cur, []))
    elif tag == 'p' and cur and text and not text.startswith('Note'):
        sections[-1][1].append(text)
    if cur == 'Amendment X' and tag == 'p' and text:
        break

# ── amendments 11–27
s = open('amendments-11-27.html').read()
s = s[s.find('<h3><a id="xi"'):]
cur = None
for tag, text, cls in blocks(s):
    if tag in ('h2', 'h3') and text.upper().startswith('AMENDMENT'):
        n = text.split()[1]
        cur = f'Amendment {n}'; sections.append((cur, []))
    elif tag == 'h3' and cur and text.startswith('Section'):
        sections[-1][1].append('**' + text.strip() + '**')
    elif tag == 'p' and cur and text:
        if 'smaller' in cls or text.startswith('Note') or text.startswith('Passed by Congress'):
            continue
        if text.startswith('Superseded by') or text.startswith('Changed by'):
            continue
        sections[-1][1].append(text)
    if cur == 'Amendment XXVII' and tag == 'p' and text and not text.startswith('Note'):
        break

# a section heading inside an amendment reads better on the same line as its text
md = []
for title, paras in sections:
    body = []
    i = 0
    while i < len(paras):
        p = paras[i]
        if p.startswith('**') and i + 1 < len(paras):
            body.append(f'{p[2:-2]} {paras[i+1]}'); i += 2
        else:
            body.append(p); i += 1
    md.append(f'## {title}\n\n' + '\n\n'.join(body) + '\n')
text = '# The Constitution of the United States, and its Amendments\n\n' + '\n'.join(md)
open('constitution.md', 'w').write(text)
for k, (t, p) in enumerate(sections, 1):
    words = sum(len(x.split()) for x in p)
    print(f'§{k:>2} {words:>4}w  {t}')
print('total words', sum(sum(len(x.split()) for x in p) for _, p in sections))
