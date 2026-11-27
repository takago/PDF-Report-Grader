import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from report_grader.grading import Grader
from report_grader.timing import OFF

GOOD={'decision':'PASS','reason':'確認済み'}

def sse(obj=GOOD,finish='stop',done=True):
    events=[{'choices':[{'delta':{'reasoning_content':'thinking'},'finish_reason':None}]}, {'choices':[{'delta':{'content':json.dumps(obj,ensure_ascii=False)},'finish_reason':finish}]}, {'choices':[],'usage':{'completion_tokens':12}}]
    return io.BytesIO((''.join('data: '+json.dumps(e)+'\n\n' for e in events)+('data: [DONE]\n\n' if done else '')).encode())

class TransportTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.g=Grader.__new__(Grader); self.g.config={'retry_interval':0,'show_stream_output':False}; self.g.model='mock'; self.g.url='http://localhost/v1/chat/completions'; self.g.effort='auto'; self.g.cache=Path(self.tmp.name); self.g.timing=OFF
    def ask(self): return self.g.ask_json('test',{},self.g.judgement)
    def test_timeout_retry_and_cache(self):
        with patch('report_grader.grading.urlopen',side_effect=[TimeoutError('read timed out'),sse()]) as call:
            self.assertEqual(self.ask(),GOOD); self.assertEqual(self.ask(),GOOD); self.assertEqual(call.call_count,2)
            bodies=[json.loads(c.args[0].data) for c in call.call_args_list]
            self.assertEqual([b['seed'] for b in bodies],[42,43]); self.assertTrue(bodies[0]['stream']); self.assertEqual(bodies[0]['temperature'],0.2)
    def test_invalid_schema_and_truncated(self):
        with patch('report_grader.grading.urlopen',side_effect=[sse({'wrong':1}),sse(finish='length'),sse()]) as call:
            self.assertEqual(self.ask(),GOOD); self.assertEqual(call.call_count,3)
    def test_incomplete_no_cache(self):
        with patch('report_grader.grading.urlopen',side_effect=[sse(done=False) for _ in range(3)]) as call:
            with self.assertRaises(ValueError): self.ask()
            self.assertEqual(call.call_count,3); self.assertEqual(list(self.g.cache.glob('*.json')),[])
    def test_auth_not_retried(self):
        with patch('report_grader.grading.urlopen',side_effect=HTTPError(self.g.url,401,'Unauthorized',{},None)) as call:
            with self.assertRaises(HTTPError): self.ask()
            self.assertEqual(call.call_count,1)
    def test_nonstream(self):
        self.g.config.update(stream=False,retry_change_seed=False)
        result={'choices':[{'message':{'content':json.dumps(GOOD)},'finish_reason':'stop'}]}
        with patch('report_grader.grading.urlopen',side_effect=[TimeoutError(),io.BytesIO(json.dumps(result).encode())]) as call:
            self.assertEqual(self.ask(),GOOD)
            bodies=[json.loads(c.args[0].data) for c in call.call_args_list]
            self.assertEqual([b['seed'] for b in bodies],[42,42]); self.assertNotIn('stream',bodies[0])
    def test_unknown_not_cached(self):
        with patch('report_grader.grading.urlopen',side_effect=[sse({'decision':'UNKNOWN','reason':'不明'}),sse()]) as call:
            self.assertEqual(self.ask()['decision'],'UNKNOWN'); self.assertEqual(self.ask(),GOOD); self.assertEqual(call.call_count,2)
    def test_usage_disabled(self):
        self.g.config['stream_include_usage']=False
        with patch('report_grader.grading.urlopen',return_value=sse()) as call:
            self.assertEqual(self.ask(),GOOD); self.assertNotIn('stream_options',json.loads(call.call_args.args[0].data))
