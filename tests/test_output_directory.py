import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile
from urllib.parse import unquote

import pymupdf as pdf
from openpyxl import Workbook, load_workbook
from report_grader.collector import prepare_reports

PROJECT = Path(__file__).resolve().parent.parent

class OutputDirectoryTests(unittest.TestCase):
    def test_zip_output_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            source = p / 'incoming' / 'reports.zip'
            source.parent.mkdir()
            with ZipFile(source, 'w') as z:
                z.writestr('sub/answer.pdf', b'pdf bytes')
            root = prepare_reports(source, p / 'outputs')
            self.assertEqual(root, p / 'outputs' / 'reports')
            self.assertEqual((root/'answer.pdf').read_bytes(), b'pdf bytes')
            self.assertFalse(source.with_suffix('').exists())
            (root/'answer.pdf').rename(root/'renamed.pdf')
            self.assertEqual(prepare_reports(source, p/'outputs'), root)
            self.assertFalse((root/'answer.pdf').exists())
            with ZipFile(source,'a') as z:
                z.writestr('another.pdf', b'changed')
            with self.assertRaisesRegex(ValueError, 'ZIPの内容が変わっています'):
                prepare_reports(source,p/'outputs')

    def test_folder_input_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); root=p/'existing'; root.mkdir()
            self.assertEqual(prepare_reports(root,p/'outputs'),root)
            self.assertFalse((p/'outputs').exists())

    def test_cli_output_and_relative_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            template=p/'template.pdf'
            with pdf.open() as doc:
                page=doc.new_page()
                for i, name in enumerate(['stu_id','stu_class','stu_num','stu_name','Q1']):
                    w=pdf.Widget(); w.field_name=name; w.field_type=pdf.PDF_WIDGET_TYPE_TEXT
                    w.rect=pdf.Rect(20,20+i*30,200,45+i*30)
                    page.add_widget(w)
                doc.save(template)
            roster=p/'students.xlsx'; book=Workbook()
            book.active.append(['受講名列番号','学籍番号','所属クラス','学生氏名'])
            book.active.append([1,'1234567','2CC1001','試験 太郎'])
            book.save(roster); book.close()
            source=p/'reports.zip'
            with ZipFile(source,'w') as z:
                z.write(template,'1234567.pdf')
            args=[sys.executable,str(PROJECT/'grade_reports.py'),str(template),str(source),str(roster),'--config',str(PROJECT/'examples/config.example.toml')]
            subprocess.run(args,cwd=p,check=True,capture_output=True,text=True)
            root=p/'out'/'reports'
            out=root/'reports.xlsx'; self.assertTrue(out.is_file())
            self.assertTrue((root/'_inputs'/'template.pdf').is_file())
            with_file=load_workbook(out)
            sheet=with_file['Results']
            col=next(c.column for c in sheet[1] if c.value=='filename')
            link=sheet.cell(2,col).hyperlink.target
            self.assertTrue((root/unquote(link)).is_file())
            with_file.close()
            subprocess.run([*args,'--output-dir',str(p/'custom')],cwd=p,check=True,capture_output=True,text=True)
            self.assertTrue((p/'custom'/'reports'/'reports.xlsx').exists())
            source.unlink()
            subprocess.run([args[0],args[1],args[2],str(root),*args[4:]],cwd=p,check=True,capture_output=True,text=True)
            self.assertTrue((root/'reports.backup-001.xlsx').exists())
