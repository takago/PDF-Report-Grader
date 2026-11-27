"""Human-readable, line-oriented logs shared by CLI and WebUI."""
import re
from pathlib import Path

def clean(value):
    return ''.join(c for c in str(value) if c.isprintable() or c in '\n\t')

def detail(timing, label, value):
    if getattr(timing, 'level', 'debug') == 'simple': return
    lines=clean(value).splitlines() or ['']
    print(f'  {label}: {lines[0]}', flush=True)
    for line in lines[1:]:
        if line.strip(): print('    '+line, flush=True)

def pdf_label(path):
    name=Path(path).name
    match=re.search(r'(?<![A-Za-z0-9])([0-9][A-Za-z]{2}[0-9]+[-_][0-9]{2,3})(?![0-9])',name)
    return match.group(1).replace('_','-') if match else name

def student_label(row):
    return ' / '.join(clean(row.get(k,'')) for k in ('受講名列番号','所属クラス','学籍番号') if row.get(k) is not None and str(row.get(k,''))) or '照合情報なし'
