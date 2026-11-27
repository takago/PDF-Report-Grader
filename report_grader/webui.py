"""Local-only UI; CLI remains the single collection engine."""
import argparse
import codecs
import json
import tempfile
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import webbrowser
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from .preview import render_page
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
WEEK = re.compile(r'^DATA_(w\d+)(?:_.+)?$', re.I)


def inside(path, base):
    return path.resolve().is_relative_to(base.resolve()) and not path.is_symlink()


def discover(root):
    inp, out = root/'in', root/'out'
    rows = []
    for roster in sorted(inp.glob('*.xlsx')):
        if not roster.is_file() or not inside(roster, inp) or roster.name.startswith('~$'):
            continue
        cls = roster.stem
        sources = {}
        for z in sorted((inp/cls).glob('*')):
            if z.is_file() and z.suffix.lower()=='.zip' and inside(z, inp):
                sources[z.stem] = z
        for d in sorted((out/cls).glob('DATA_*')):
            if d.is_dir() and inside(d, out) and ((d/'valid').is_dir() or (d/'invalid').is_dir() or (d/'review').is_dir() or (d/(d.name+'.xlsx')).is_file()):
                sources.setdefault(d.name, d)
        for stem, source in sources.items():
            m = WEEK.fullmatch(stem)
            if not m:
                continue
            week = m[1].lower()
            result = out/cls/stem
            template = inp/f'kadai_{week}.pdf'
            rubric = inp/f'grading_{week}.txt'
            if not inside(result, out) or not inside(template, inp) or not inside(rubric, inp):
                continue
            xlsx = result/(stem+'.xlsx')
            plan = result/'grading-plan.json'
            rows.append(dict(id=f'{cls}/{stem}', cls=cls, week=week, name=stem,
                source=str(source), template=str(template), roster=str(roster), rubric=str(rubric), result=str(result),
                ready=template.is_file(), rubric_exists=rubric.is_file(),
                done=xlsx.is_file() and inside(xlsx,out), plan=plan.is_file() and inside(plan,out),
                directory=source.is_dir()))
    # Sort the merged ZIP and extracted-directory entries by class and week.
    return sorted(rows, key=lambda r: (r["cls"].casefold(), int(r["week"][1:]), r["name"].casefold(), r["id"]))


class Start(BaseModel):
    ids: list[str]
    grading: bool = True
    plan_only: bool = False


class Edit(BaseModel):
    text: str


class Runner:
    def __init__(self, root):
        self.root = root
        self.lock = threading.RLock()
        self.logs = deque(maxlen=2500)
        self.seq = 0
        self.active = False
        self.cancelled = False
        self.proc = None
        self.items = []
        self.progress = ''
        self.generation = 0
        self.tail = ''
        self.thread = None
        self.preview_file = None
        self.preview_root = None
        self.preview_key = None
        self.preview_id = 0

    def ingest(self, text):
        if not text:
            return
        with self.lock:
            self.seq += 1
            self.logs.append((self.seq, text))
            lines = (self.tail + text).split('\n')
            self.tail = lines.pop()[-12000:]
            for line in lines:
                m = re.search(r'(?:Grading|AI採点) \[(\d+)/(\d+)\]', line)
                if m:
                    self.progress = f'採点 {m[1]} / {m[2]} 名'
                elif (m := re.match(r'^PDF解析 \[(\d+)/(\d+)\]', line)):
                    self.progress = f'PDF解析 {m[1]} / {m[2]} 件'
                elif re.match(r'^\[\d+\] .*: (VALID|INVALID)', line):
                    self.progress = 'PDF解析 ' + line.split(']')[0][1:] + ' 件'

    def log(self, line):
        self.ingest(line + '\n')

    def start(self, rows, opts):
        with self.lock:
            if self.active:
                raise HTTPException(409, '実行中です')
            self.active, self.cancelled = True, False
            self.logs.clear()
            self.generation += 1
            self.tail = ''
            self.progress = ''
            self.items = [dict(id=r['id'], status='queued') for r in rows]
            self.thread = threading.Thread(target=self.run, args=(rows, opts), daemon=True)
            self.thread.start()

    def run(self, rows, opts):
        try:
            for i, r in enumerate(rows):
                with self.lock:
                    if self.cancelled:
                        break
                    self.items[i]['status'] = 'running'
                    self.progress = ''
                self.log('=== ' + r['id'] + ' ===')
                cmd = [sys.executable, '-u', str(self.root/'grade_reports.py'), r['template'], r['source'], r['roster'], '--config', str(self.root/'config.toml')]
                if not r['directory']:
                    cmd += ['--output-dir', str(Path(r['result']).parent)]
                if opts.grading:
                    cmd += ['--grading', r['rubric']]
                if opts.plan_only:
                    cmd += ['--plan-only']
                try:
                    with tempfile.TemporaryDirectory(prefix="grader-preview-") as preview_dir:
                        self.run_process(r, i, cmd, preview_dir)
                except Exception as e:
                    self.log(f'実行失敗: {e}')
                    with self.lock:
                        self.items[i]['status'] = 'error'
        finally:
            with self.lock:
                for item in self.items:
                    if item['status'] in ('queued','running'):
                        item['status'] = 'cancelled' if self.cancelled else 'error'
                self.active = False
                self.proc = None
                self.preview_file = None

    def run_process(self, r, i, cmd, preview_dir):
        with self.lock:
            self.preview_file = Path(preview_dir)/"state.json"
            self.preview_root = Path(r["result"]).resolve()
            self.preview_key = None
            if self.cancelled:
                self.items[i]['status'] = 'cancelled'
                return
            self.proc = subprocess.Popen(cmd, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                start_new_session=True,
                env={**os.environ, 'PYTHONUNBUFFERED':'1', 'REPORT_GRADER_PREVIEW_STATE':str(self.preview_file)})
            p = self.proc
        with p.stdout:
            decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
            while (chunk := p.stdout.read1(4096)):
                self.ingest(decoder.decode(chunk))
            self.ingest(decoder.decode(b'', final=True))
            if self.tail:
                self.ingest('\n')
        code = p.wait()
        with self.lock:
            self.proc = None
            self.items[i]['status'] = 'cancelled' if self.cancelled else ('done' if code == 0 else 'error')
        self.log(f'終了コード: {code}')
    def stop(self):
        with self.lock:
            if not self.active: return
            self.cancelled = True
            p = self.proc
            if p and p.poll() is None:
                try: os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                def force():
                    try: p.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try: os.killpg(p.pid, signal.SIGKILL)
                        except ProcessLookupError: pass
                threading.Thread(target=force, daemon=True).start()

    def preview(self):
        # Only subprocess-authored state; never accept a filesystem path from HTTP.
        with self.lock:
            if not self.active or self.preview_file is None:
                return None
            try:
                data = json.loads(self.preview_file.read_text(encoding='utf-8'))
                path = Path(data['path'])
                if not inside(path, self.preview_root) or not inside(path, self.root/'out') or path.suffix.lower() != '.pdf' or not path.is_file():
                    return None
                key = (self.generation, str(path))
                if key != self.preview_key:
                    self.preview_id += 1
                    self.preview_key = key
                return dict(id=self.preview_id, name=path.name, page=data['page'], count=data['count']), path
            except (OSError, ValueError, KeyError, TypeError):
                return None

    def state(self, after):
        with self.lock:
            preview = self.preview()
            return dict(preview=preview[0] if preview else None, generation=self.generation, active=self.active, items=[dict(x) for x in self.items], progress=self.progress,
                logs=[dict(seq=n,text=s) for n,s in self.logs if n>after], cursor=self.seq)


def create_app(root=ROOT, port=7860):
    root = Path(root).resolve()
    for name in ('in','out'): (root/name).mkdir(exist_ok=True)
    token = secrets.token_urlsafe(32)
    runner = Runner(root)
    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            runner.stop()
            if runner.thread:
                runner.thread.join(timeout=6)
    app = FastAPI(title="PDF Report Grader", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.runner = runner

    @app.middleware('http')
    async def guard(request: Request, call_next):
        allowed = {f'127.0.0.1:{port}', f'localhost:{port}'}
        if request.headers.get('host') not in allowed:
            return PlainTextResponse('Invalid host', 403)
        origin = request.headers.get('origin')
        if origin and origin not in {f'http://{h}' for h in allowed}:
            return PlainTextResponse('Invalid origin', 403)
        if request.url.path.startswith('/api/') and request.headers.get('x-local-token') != token:
            return PlainTextResponse('Forbidden', 403)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        return response

    def lookup(id):
        for r in discover(root):
            if r['id'] == id: return r
        raise HTTPException(404, '課題が見つかりません')

    @app.get('/', response_class=HTMLResponse)
    def index():
        return (ROOT/'web/index.html').read_text(encoding='utf-8').replace('__LOCAL_TOKEN__', token)

    @app.get('/api/tasks')
    def tasks(): return discover(root)

    @app.get('/api/state')
    def state(after: int = 0): return runner.state(after)

    @app.get('/api/preview')
    def preview(id: int, page: int = 1, large: bool = False):
        current = runner.preview()
        if not current or current[0]['id'] != id:
            raise HTTPException(404, 'プレビュー対象が変わりました')
        try:
            content = render_page(current[1], page, large)
        except Exception:
            raise HTTPException(404, 'プレビューを生成できません')
        return Response(content, media_type='image/png')

    @app.post('/api/start')
    def start(opts: Start):
        if not opts.ids or len(opts.ids) != len(set(opts.ids)):
            raise HTTPException(400, '課題を選択してください（重複不可）')
        if opts.plan_only and not opts.grading:
            raise HTTPException(400, 'AI採点を有効にしてください')
        rows = [lookup(id) for id in opts.ids]
        for r in rows:
            if not r['ready'] or (opts.grading and not r['rubric_exists']):
                raise HTTPException(400, r['id']+' の入力ファイルが不足しています')
        if not (root/'config.toml').is_file(): raise HTTPException(400, 'config.toml がありません')
        runner.start(rows, opts)
        return {'ok':True}

    @app.post('/api/stop')
    def stop(): runner.stop(); return {'ok':True}

    @app.get('/api/rubric')
    def rubric(id: str):
        p = Path(lookup(id)['rubric'])
        return {'text':p.read_text(encoding='utf-8') if p.exists() else ''}

    @app.put('/api/rubric')
    def edit(id: str, body: Edit):
        with runner.lock:
            if runner.active: raise HTTPException(409, '実行中は編集できません')
            p = Path(lookup(id)['rubric'])
            if len(body.text)>100000: raise HTTPException(400, '採点基準が長すぎます')
            tmp = p.with_name('.'+p.name+'.'+secrets.token_hex(8))
            try:
                with tmp.open('x', encoding='utf-8') as f: f.write(body.text)
                tmp.replace(p)
            finally: tmp.unlink(missing_ok=True)
        return {'ok':True}

    @app.get('/api/plan')
    def plan(id: str):
        r=lookup(id)
        if not r['plan']: raise HTTPException(404, '採点プランがありません')
        return {'text':(Path(r['result'])/'grading-plan.json').read_text(encoding='utf-8')}

    @app.post('/api/open')
    def open_result(id: str, kind: str):
        r=lookup(id); p=Path(r['result'])
        if kind=='xlsx': p=p/(p.name+'.xlsx')
        elif kind!='folder': raise HTTPException(400, '不正な種類')
        if not p.exists() or not inside(p, root/'out'): raise HTTPException(404, '結果がありません')
        try: subprocess.Popen(['xdg-open',str(p)], stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        except OSError as e: raise HTTPException(500, str(e)) from e
        return {'ok':True}

    return app


def main():
    import uvicorn
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=7860)
    parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args()
    app=create_app(ROOT,args.port)
    if not args.no_browser:
        threading.Timer(1, lambda:webbrowser.open(f'http://127.0.0.1:{args.port}/')).start()
    uvicorn.run(app,host='127.0.0.1',port=args.port,access_log=False)
