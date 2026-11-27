import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from report_grader.log_output import detail, pdf_label, student_label

class LogTests(unittest.TestCase):
    def test_multiline(self):
        out=StringIO()
        with redirect_stdout(out):
            detail(SimpleNamespace(level='normal'),'P1','first\n\nsecond\x1b')
            detail(SimpleNamespace(level='simple'),'P2','hidden')
        self.assertEqual(out.getvalue(),'  P1: first\n    second\n')
    def test_label(self):
        self.assertEqual(pdf_label('prefix_2CC1-001_12345.pdf'),'2CC1-001')
        self.assertEqual(pdf_label('other.pdf'),'other.pdf')
        self.assertEqual(student_label({'受講名列番号':1,'所属クラス':'2CC1','学籍番号':'123'}),'1 / 2CC1 / 123')
