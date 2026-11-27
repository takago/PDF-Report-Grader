import hashlib
import tempfile
import unittest
from pathlib import Path
import pymupdf as pdf
from report_grader.annotated_pdf import GradePDFWriter


class AnnotatedPDFTests(unittest.TestCase):
    def test_wrapping_preserved_widget_and_unchanged_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'original.pdf'; dst=Path(tmp)/'graded.pdf'
            with pdf.open() as d:
                p=d.new_page(width=300,height=400)
                p.insert_text((20,20),'original content')
                w=pdf.Widget(); w.field_name='answer'; w.field_type=pdf.PDF_WIDGET_TYPE_TEXT
                w.rect=pdf.Rect(20,50,200,80); w.field_value='Student answer'; p.add_widget(w)
                p=d.new_page(width=300,height=400); p.insert_text((20,20),'second page')
                d.save(src)
            digest=hashlib.sha256(src.read_bytes()).digest()
            reason='タイムチャートの波形を確認してください．'*30
            writer=GradePDFWriter({})
            writer.write(src,dst,{'AI点数':2,'AI採点理由':reason,'AI採点状態':'REVIEW'},3)
            self.assertEqual(hashlib.sha256(src.read_bytes()).digest(),digest)
            with pdf.open(dst) as d, pdf.open(src) as original:
                self.assertEqual(len(d),2)
                self.assertEqual(d[0].rect.width,300)
                self.assertGreater(d[0].rect.height,400)
                self.assertEqual(d[1].rect,original[1].rect)
                self.assertEqual(d[1].get_pixmap().samples,original[1].get_pixmap().samples)
                text=d[0].get_text()
                first_page=d[0]; widget=next(first_page.widgets())
                self.assertEqual(widget.field_name,'answer')
                self.assertEqual(widget.field_value,'Student answer')
                self.assertEqual(widget.rect, pdf.Rect(20,50+d[0].rect.height-400,200,80+d[0].rect.height-400))
                self.assertIn('要確認',text)
                self.assertIn(reason,''.join(text.splitlines()))
                self.assertTrue(any(d.extract_font(f[0])[3] for f in d[0].get_fonts() if 'Droid' in f[3]))
                widget.field_value='Edited answer'; widget.update()
                self.assertEqual(next(first_page.widgets()).field_value,'Edited answer')
            writer.write(src,dst,{'AI点数':None,'AI採点理由':'確認が必要','AI採点状態':'REVIEW'},3)
            with pdf.open(dst) as d:
                self.assertIn('点数未確定',d[0].get_text())

    def test_rotated_cropped_pages_keep_rendering_and_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'original.pdf'; dst=Path(tmp)/'graded.pdf'
            for rotation in (0,90,180,270):
                with self.subTest(rotation=rotation):
                    with pdf.open() as d:
                        p=d.new_page(width=300,height=400)
                        p.insert_text((60,80),'Original content')
                        w=pdf.Widget(); w.field_name='answer'; w.field_type=pdf.PDF_WIDGET_TYPE_TEXT
                        w.rect=pdf.Rect(60,90,200,120); w.field_value='abc'; p.add_widget(w)
                        p.add_rect_annot(pdf.Rect(70,150,90,170))
                        p.set_rotation(rotation); p.set_cropbox(pdf.Rect(20,30,280,370))
                        d.save(src)
                    GradePDFWriter({}).write(src,dst,{'AI点数':3,'AI採点理由':'テスト'},3)
                    with pdf.open(src) as original, pdf.open(dst) as result:
                        p=result[0]; before=original[0].get_pixmap(); after=p.get_pixmap()
                        self.assertEqual(p.rotation,rotation)
                        self.assertEqual(before.width,after.width)
                        self.assertEqual(before.samples,after.samples[-len(before.samples):])
                        self.assertEqual(next(p.widgets()).field_value,'abc')
                        self.assertEqual(len(list(p.annots())),1)

    def test_source_protection_and_font_settings(self):
        with self.assertRaises(ValueError): GradePDFWriter({'annotation_font_size':0})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a.pdf'
            with self.assertRaises(ValueError): GradePDFWriter({}).write(path,path,{},3)

    def test_cli_graded_link_and_rerun(self):
        import json
        import subprocess
        import sys
        from urllib.parse import unquote
        from zipfile import ZipFile
        from openpyxl import Workbook, load_workbook
        project=Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); template=p/'template.pdf'; rubric=p/'rubric.txt'; roster=p/'roster.xlsx'
            with pdf.open() as d:
                page=d.new_page()
                page.insert_text((20,15),'Report w01')
                for i,name in enumerate(['stu_id','stu_class','stu_num','stu_name','Q1']):
                    w=pdf.Widget(); w.field_name=name; w.field_type=pdf.PDF_WIDGET_TYPE_TEXT
                    w.rect=pdf.Rect(20,20+i*30,200,45+i*30); page.add_widget(w)
                d.save(template)
            rubric.write_text('3点満点．',encoding='utf-8')
            plan=p/'plan.json'; plan.write_text(json.dumps({'version':2,'signature':hashlib.sha256(template.read_bytes()+b'\0'+rubric.read_bytes()).hexdigest(),'plan':{'maximum':3,'minimum_submitted':1,'missing_score':0,'unrelated_score':1,'rules':[],'warnings':['全体の警告']}}))
            book=Workbook(); book.active.append(['受講名列番号','学籍番号','所属クラス','学生氏名']); book.active.append([1,'1234567','2CC1001','試験 太郎']); book.save(roster); book.close()
            source=p/'reports.zip'
            with ZipFile(source,'w') as z: z.write(template,'1234567.pdf')
            args=[sys.executable,str(project/'grade_reports.py'),str(template),str(source),str(roster),'--config',str(project/'examples/config.example.toml'),'--grading',str(rubric),'--grading-plan',str(plan)]
            root=p/'out'/'reports'
            for _ in range(2):
                subprocess.run(args,cwd=p,check=True,capture_output=True,text=True)
                with_file=load_workbook(root/'reports.xlsx'); sheet=with_file['Results']
                cols={c.value:c.column for c in sheet[1]}
                self.assertEqual(sheet.cell(2,cols['AI点数']).value,3)
                self.assertEqual(sheet.cell(2,cols['AI採点理由']).value,'減点なし')
                self.assertEqual(sheet.cell(2,cols['AI採点状態']).value,'OK')
                for name in ('点数','AI点数'):
                    self.assertEqual(sheet.cell(2,cols[name]).alignment.horizontal,'center')
                rules={str(item.sqref): [r.formula[0] for r in sheet.conditional_formatting[item]] for item in sheet.conditional_formatting}
                self.assertIn('AND(ISNUMBER(E2),E2=0)', rules['E2'])
                self.assertIn('AND(ISNUMBER(G2),G2=0)', rules['G2'])
                self.assertIn('AND(H2<>"",H2<>"減点なし")', rules['H2'])
                link=sheet.cell(2,cols['採点済PDF']).hyperlink.target
                self.assertTrue((root/unquote(link)).is_file())
                self.assertEqual(len(list((root/'graded').glob('*.pdf'))),1)
                self.assertEqual(with_file['Unmatched'].max_row,1)
                with_file.close()
