"""Create a separate, printable copy of the first page with an AI grading header."""
from pathlib import Path
import math
import os
import tempfile
import pymupdf as pdf


class GradePDFWriter:
    def __init__(self, config):
        path = config.get('annotation_font', '')
        size = config.get('annotation_font_size', 10)
        if not isinstance(path, str):
            raise ValueError('grading.annotation_font must be a string')
        if type(size) not in (int, float) or not math.isfinite(size) or not 6 <= size <= 24:
            raise ValueError('grading.annotation_font_size must be 6..24')
        self.size = size
        # MuPDF ships an embeddable CJK font, so no OS font is required.
        self.font = pdf.Font(fontfile=path) if path else pdf.Font('japan')

    def wrap(self, text, width):
        lines = []
        for paragraph in text.replace('\r\n', '\n').replace('\r', '\n').expandtabs(4).split('\n'):
            line = ''; used = 0.0
            for ch in paragraph:
                if not self.font.has_glyph(ord(ch)) and not ch.isspace():
                    raise ValueError(f'採点用フォントに文字 U+{ord(ch):04X} がありません')
                advance = self.font.text_length(ch, fontsize=self.size)
                if advance > width:
                    raise ValueError('採点コメントの表示幅不足')
                if line and used + advance > width:
                    lines.append(line); line = ''; used = 0.0
                line += ch; used += advance
            lines.append(line)
        return lines

    def write(self, source, target, row, maximum):
        source, target = Path(source), Path(target)
        if source.resolve() == target.resolve() or target.is_symlink():
            raise ValueError('採点済PDFの保存先が不正です')
        score = row.get('AI点数')
        heading = f'AI採点（暫定）：{score:g} / {maximum:g} 点' if score is not None else 'AI採点：要確認（点数未確定）'
        if score is not None and row.get('AI採点状態') == 'REVIEW':
            heading += ' ／ 要確認'
        reason = str(row.get('AI採点理由') or '理由なし').replace(' / ', '\n')
        with pdf.open(source) as result, pdf.open() as header_doc:
            if not result.is_pdf or result.needs_pass or not len(result):
                raise ValueError('採点済PDFを生成できないPDFです')
            p = result[0]
            width, height = p.rect.width, p.rect.height
            rotation = p.rotation
            media = p.mediabox
            crop = p.cropbox
            # CropBox is expressed by PyMuPDF with a top-down Y axis;
            # PDF boxes use bottom-up Y coordinates.
            visible = pdf.Rect(crop.x0, media.y1-crop.y1,
                               crop.x1, media.y1-crop.y0)
            unit = height / (visible.width if rotation in (90, 270) else visible.height)
            margin = min(24, width / 10)
            lines = self.wrap(heading+'\n'+reason, width-2*margin)
            ascent = max(self.font.ascender, 1.0)*self.size
            descent = max(-self.font.descender, 0.3)*self.size
            step = ascent + descent + 2
            header = math.ceil(2*margin + len(lines)*step)
            if width > 14400 or height+header > 14400:
                raise ValueError('採点済PDFのページサイズが上限を超えます')
            page = header_doc.new_page(width=width, height=header)
            page.draw_rect(page.rect, color=None, fill=(1,1,1))
            tw = pdf.TextWriter(page.rect)
            for i, line in enumerate(lines):
                if line:
                    tw.append((margin,margin+ascent+i*step),line,font=self.font,fontsize=self.size)
            tw.write_text(page,color=(0.75,0.08,0.08))
            page.draw_line((margin,header-margin/2),(width-margin,header-margin/2),color=(0.75,0.08,0.08),width=0.6)
            # Import only the new header as a form XObject. The original
            # document, AcroForm, widget appearances and annotations stay intact.
            header_doc.subset_fonts()
            amount = header / unit
            if rotation == 0:
                visible.y1 += amount
            elif rotation == 90:
                visible.x0 -= amount
            elif rotation == 180:
                visible.y0 -= amount
            else:
                visible.x1 += amount
            media |= visible
            def box_value(rect):
                return '['+' '.join(format(v,'.12f').rstrip('0').rstrip('.') or '0' for v in rect)+']'
            result.xref_set_key(p.xref, 'MediaBox', box_value(media))
            result.xref_set_key(p.xref, 'CropBox', box_value(visible))
            p = result.reload_page(p)
            area = pdf.Rect(0,0,width,header) * p.derotation_matrix
            p.set_rotation(0)
            p.show_pdf_page(area, header_doc, 0, rotate=rotation, keep_proportion=False)
            p.set_rotation(rotation)
            target.parent.mkdir(exist_ok=True)
            with tempfile.NamedTemporaryFile(suffix='.pdf',dir=target.parent,delete=False) as f:
                temp=Path(f.name)
            try:
                result.save(temp,garbage=4,deflate=True)
                os.replace(temp,target)
            finally:
                temp.unlink(missing_ok=True)
