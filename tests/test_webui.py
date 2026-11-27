import re
import tempfile
import time
import unittest
from pathlib import Path
from zipfile import ZipFile
from fastapi.testclient import TestClient
import pymupdf as pdf
from openpyxl import Workbook
from report_grader.webui import create_app, discover, ROOT

class WebUITests(unittest.TestCase):
    def test_discover_api_and_real_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'in/2CC1').mkdir(parents=True)
            (p/'out').mkdir()
            # Run actual collector using project sources, but isolated data and cwd.
            (p/'grade_reports.py').write_text(f'import sys\nsys.path.insert(0, {str(ROOT)!r})\nfrom report_grader.collector import main\nmain()\n')
            (p/'config.toml').write_text((ROOT/'examples/config.example.toml').read_text(encoding='utf-8'), encoding='utf-8')
            template=p/'in/kadai_w01.pdf'
            with pdf.open() as d:
                page=d.new_page()
                for i,n in enumerate(['stu_id','stu_class','stu_num','stu_name','Q1']):
                    w=pdf.Widget();w.field_name=n;w.field_type=pdf.PDF_WIDGET_TYPE_TEXT;w.rect=pdf.Rect(20,20+i*30,200,45+i*30);page.add_widget(w)
                d.save(template)
            b=Workbook();b.active.append(['受講名列番号','学籍番号','所属クラス','学生氏名']);b.active.append([1,'1234567','2CC1001','試験 太郎']);b.save(p/'in/2CC1.xlsx');b.close()
            z=p/'in/2CC1/DATA_w01_test.zip'
            with ZipFile(z,'w') as a:a.write(template,'1234567.pdf')
            app=create_app(p)
            with TestClient(app,base_url='http://127.0.0.1:7860') as c:
                token=re.search("const token='([^']+)'",c.get('/').text)[1]
                h={'X-Local-Token':token}
                self.assertEqual(c.get('/api/tasks').status_code,403)
                self.assertEqual(c.get('/api/tasks',headers={**h,'Origin':'http://evil.test'}).status_code,403)
                rows=c.get('/api/tasks',headers=h).json();self.assertEqual(len(rows),1);id=rows[0]['id']
                self.assertEqual(c.put('/api/rubric',params={'id':id},json={'text':'3点満点'},headers=h).status_code,200)
                self.assertEqual(c.get('/api/rubric',params={'id':id},headers=h).json()['text'],'3点満点')
                self.assertEqual(c.post('/api/start',json={'ids':[id],'grading':False},headers=h).status_code,200)
                deadline=time.monotonic()+20
                while time.monotonic()<deadline:
                    s=c.get('/api/state',headers=h).json()
                    if not s['active']:break
                    time.sleep(.1)
                self.assertFalse(s['active']);self.assertEqual(s['items'][0]['status'],'done',s['logs'])
                result=p/'out/2CC1/DATA_w01_test/DATA_w01_test.xlsx';self.assertTrue(result.is_file())
                z.unlink();self.assertTrue(discover(p)[0]['directory']);self.assertTrue(discover(p)[0]['done'])
                self.assertEqual(c.get('/api/rubric',params={'id':'../../config.toml'},headers=h).status_code,404)

    def test_discover_natural_week_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for cls in ('2CC3', '2CC1'):
                (p/'in'/cls).mkdir(parents=True)
                (p/'in'/f'{cls}.xlsx').touch()
                for week in ('w10', 'w2', 'w05'):
                    (p/'in'/cls/f'DATA_{week}_test.zip').touch()
                (p/'out'/cls/'DATA_w01_test'/'valid').mkdir(parents=True)
                (p/'out'/cls/'DATA_w2_test'/'valid').mkdir(parents=True)
            rows = discover(p)
            self.assertEqual([(r['cls'], r['week']) for r in rows],
                [(cls, week) for cls in ('2CC1', '2CC3')
                 for week in ('w01', 'w2', 'w05', 'w10')])
            self.assertTrue(rows[0]['directory'])
            self.assertFalse(rows[1]['directory'])

    def test_cancel_queue(self):
        from report_grader.webui import Runner, Start
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'grade_reports.py').write_text('import time\nprint("started",flush=True)\ntime.sleep(60)\n')
            runner=Runner(p)
            r=dict(id='a',template='t.pdf',source='s.zip',roster='r.xlsx',rubric='g.txt',result=str(p/'out/a'),directory=False)
            runner.start([r,{**r,'id':'b'}],Start(ids=['a','b'],grading=False))
            deadline=time.monotonic()+5
            while time.monotonic()<deadline and not any('started' in x['text'] for x in runner.state(0)['logs']):time.sleep(.02)
            self.assertTrue(any('started' in x['text'] for x in runner.state(0)['logs']))
            runner.stop();runner.thread.join(timeout=8)
            s=runner.state(0)
            self.assertFalse(s['active']);self.assertEqual([x['status'] for x in s['items']],['cancelled','cancelled'])

    def test_preview_api_current_document_only(self):
        import json
        from report_grader.preview import render_page
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); folder=root/'out'/'task'; folder.mkdir(parents=True)
            path=folder/'sample.pdf'
            with pdf.open() as d:
                d.new_page().insert_text((50,50),'Page one')
                d.new_page().insert_text((50,50),'Page two')
                d.save(path)
            app=create_app(root); runner=app.state.runner
            with TestClient(app,base_url='http://127.0.0.1:7860') as c:
                h={'X-Local-Token':re.search("const token='([^']+)'",c.get('/').text)[1]}
                runner.active=True; runner.preview_root=folder; runner.preview_file=root/'state.json'
                runner.preview_file.write_text(json.dumps(dict(path=str(path),page=2,count=2)))
                current=c.get('/api/state',headers=h).json()['preview']
                self.assertEqual((current['page'],current['count']),(2,2))
                query=dict(id=current['id'],page=2)
                self.assertEqual(c.get('/api/preview',params=query).status_code,403)
                res=c.get('/api/preview',params=query,headers=h)
                self.assertEqual(res.status_code,200); self.assertEqual(res.content,render_page(path,2))
                self.assertEqual(c.get('/api/preview',params={**query,'page':3},headers=h).status_code,404)
                self.assertEqual(c.get('/api/preview',params={**query,'id':-1},headers=h).status_code,404)
                runner.preview_file.write_text(json.dumps(dict(path=str(root/'outside.pdf'),page=1,count=1)))
                (root/'outside.pdf').write_bytes(path.read_bytes())
                self.assertIsNone(c.get('/api/state',headers=h).json()['preview'])
                self.assertEqual(c.get('/api/preview',params=query,headers=h).status_code,404)
                runner.active=False
