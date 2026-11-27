"""Optional WebUI progress channel; CLI runs without it."""
import json
import os
from pathlib import Path
import pymupdf as pdf


def notify(path, page=1, count=None):
    target = os.environ.get('REPORT_GRADER_PREVIEW_STATE')
    if not target:
        return
    target = Path(target)
    tmp = target.with_suffix('.tmp')
    try:
        if path is None:
            data = {}
        else:
            path = Path(path).resolve()
            if count is None:
                with pdf.open(path) as doc:
                    count = len(doc)
            data = dict(path=str(path), page=page, count=count)
        tmp.write_text(json.dumps(data), encoding='utf-8')
        tmp.replace(target)
    except Exception:
        # Preview must never interrupt grading.
        try: tmp.unlink(missing_ok=True)
        except OSError: pass


def render_page(path, page, large=False):
    with pdf.open(path) as doc:
        if doc.needs_pass or not 1 <= page <= len(doc):
            raise ValueError('Invalid page')
        p = doc[page-1]
        scale = min((1800 if large else 850)/max(p.rect.width,p.rect.height), 3)
        return p.get_pixmap(matrix=pdf.Matrix(scale,scale), alpha=False).tobytes('png')
