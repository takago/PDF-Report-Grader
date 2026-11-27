import io
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from report_grader.visual_analysis import Vision, RetryableResponse, OutputLimit, log_api_error
from report_grader.timing import OFF

class VisionLimitsTest(unittest.TestCase):
    def test_separate_limits(self):
        with tempfile.TemporaryDirectory() as d:
            v=Vision(dict(model='mock',max_tokens=1024,stream=False,cache_dir=d),OFF)
            def response():
                return io.BytesIO(json.dumps({'choices':[{'message':{'content':'test'},'finish_reason':'stop'}]}).encode())
            with patch('report_grader.visual_analysis.urlopen',side_effect=[response(),response()]) as call:
                image=np.zeros((10,10,3),dtype=np.uint8)
                v.ask('description',[image]);v.ask('grading',[image],purpose='grading')
                self.assertEqual([json.loads(c.args[0].data)['max_tokens'] for c in call.call_args_list],[1024,4096])
            self.assertEqual(Vision({}).max_tokens,2048)
            self.assertEqual(Vision({'grading_max_tokens':8192}).grading_max_tokens,8192)
    def test_log_separates_errors(self):
        for exc,expected in [(RetryableResponse('Invalid structured VLM response'),'有効な判定結果'),(OutputLimit('finish_reason=length',''),'生成上限')]:
            for level in ('normal','debug'):
                buf=io.StringIO()
                with redirect_stdout(buf): log_api_error(SimpleNamespace(level=level),'retry',exc)
                self.assertIn(expected,buf.getvalue())
                self.assertEqual(str(exc) in buf.getvalue(),level=='debug')

    def test_description_coverage_limit_and_resolution(self):
        self.assertEqual(Vision({}).description_dpi,150)
        self.assertEqual(Vision({'dpi':100}).description_dpi,100)
        self.assertEqual(Vision({}).description_max_chars,1000)
        with tempfile.TemporaryDirectory() as d:
            v=Vision(dict(model='mock',stream=False,cache_dir=d),OFF)
            answer='波形と写真。'*80
            response=io.BytesIO(json.dumps({'choices':[{'message':{'content':answer},'finish_reason':'stop'}]}).encode())
            with patch('report_grader.visual_analysis.urlopen',return_value=response) as call:
                self.assertEqual(v.ask('全領域を確認',[np.zeros((200,200,3),dtype=np.uint8)]),answer)
                prompt=json.loads(call.call_args.args[0].data)['messages'][1]['content'][0]['text']
                self.assertIn('1000文字以内',prompt)
                self.assertNotIn('概要のみ',prompt)
                self.assertIn('別の領域',prompt)
