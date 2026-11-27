import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pymupdf as pdf
from report_grader.grading import Grader


class PlanReuseTests(unittest.TestCase):
    def test_automatic_reuse_and_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            template=root/'template.pdf'
            with pdf.open() as doc:
                doc.new_page()
                doc.save(template)
            rubric=root/'grading.txt'
            rubric.write_text('3点満点',encoding='utf-8')
            plan=dict(maximum=3,minimum_submitted=1,missing_score=0,unrelated_score=1,rules=[],warnings=[])
            path=root/'grading-plan.json'
            def create(approved=None):
                return Grader(template,rubric,[],None,{},root/'result.xlsx',None,approved)
            with patch.object(Grader,'ask_json',return_value=plan) as ask:
                create(); self.assertEqual(ask.call_count,1)
                create(); self.assertEqual(ask.call_count,1)
                rubric.write_text('採点基準変更',encoding='utf-8')
                create(); self.assertEqual(ask.call_count,2)
                path.write_text('{broken',encoding='utf-8')
                create(); self.assertEqual(ask.call_count,3)
                envelope=json.loads(path.read_text()); envelope['version']=0
                path.write_text(json.dumps(envelope))
                create(); self.assertEqual(ask.call_count,4)
                envelope=json.loads(path.read_text()); envelope['plan']['maximum']=-1
                path.write_text(json.dumps(envelope))
                create(); self.assertEqual(ask.call_count,5)
                with pdf.open(template) as doc:
                    doc[0].insert_text((20,20),'changed'); doc.saveIncr()
                with self.assertRaises(ValueError): create(path)
                self.assertEqual(ask.call_count,5)
                create(); self.assertEqual(ask.call_count,6)
                create(); self.assertEqual(ask.call_count,6)
