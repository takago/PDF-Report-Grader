#!/usr/bin/env python3
"""Collect AcroForm reports against a required template."""
import argparse
import tomllib
from .visual_analysis import Analyzer, TemplateUncertain, TemplateMismatch
from .timing import Timing, OFF
from .log_output import detail, pdf_label, student_label
from .preview import notify as preview_notify
import hashlib
import json
import stat
import os
from urllib.parse import quote
import re
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import pymupdf as pdf
from openpyxl import Workbook, load_workbook
from openpyxl.utils.exceptions import InvalidFileException
from zipfile import BadZipFile, ZipFile
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.formatting.rule import FormulaRule
from openpyxl.utils import get_column_letter
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

STUDENT = ['stu_id', 'stu_class', 'stu_num', 'stu_name']
META = ['status', 'reason', 'pdf_bytes', 'filled_fields', 'empty_fields', 'form_text_chars', 'extra_text_chars', 'extra_text_bytes', 'extra_image_count', 'extra_image_bytes', 'extra_annotation_count']

def count_chars(s):
    return sum(not c.isspace() for c in s)

def outside(rect, boxes):
    r = pdf.Rect(rect)
    return not any(b.contains(r.tl + (r.br-r.tl)*0.5) for b in boxes)

def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFC', s) if not c.isspace())

def inspect(doc, reference=None):
    fields, values, pages = {}, {}, []
    for i, p in enumerate(doc):
        boxes = []
        for w in p.widgets() or []:
            name = w.field_name
            if not name:
                continue
            fields.setdefault(name, []).append((i, w.field_type, tuple(w.rect)))
            val = '' if w.field_value is None else str(w.field_value)
            if name in values and values[name] != val:
                raise ValueError(f'同名フィールドの値が不一致: {name}')
            values[name] = val
            boxes.append(w.rect)
        if reference is not None and i < len(reference):
            boxes += reference[i]['boxes']
        # Keep individual characters and positions to distinguish template text from additions.
        chars = Counter()
        for block in p.get_text('rawdict', flags=pdf.TEXTFLAGS_RAWDICT & ~pdf.TEXT_PRESERVE_IMAGES)['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    for c in span['chars']:
                        if not c['c'].isspace() and outside(c['bbox'], boxes):
                            x,y = c['origin']
                            chars[(c['c'], round(x), round(y))] += 1
        images = []
        for im in p.get_image_info(hashes=True, xrefs=True):
            if outside(im['bbox'], boxes):
                xref = im['xref']
                size = len(doc.xref_stream_raw(xref) or b'') if xref else im['size']
                images.append((im['digest'], size))
        ann = Counter()
        for a in p.annots() or []:
            if outside(a.rect, boxes):
                ann[(a.type[0], norm(a.info.get('content') or ''), tuple(round(v) for v in a.rect))] += 1
        pages.append(dict(size=(p.rect.width,p.rect.height,p.rotation), boxes=boxes, chars=chars, images=images, ann=ann))
    return fields, values, pages

def schema_equal(a,b):
    if a.keys()!=b.keys(): return False
    for n in a:
        x,y=sorted(a[n]),sorted(b[n])
        if len(x)!=len(y): return False
        if any(u[:2]!=v[:2] or any(abs(c-d)>2 for c,d in zip(u[2],v[2])) for u,v in zip(x,y)): return False
    return True

def char_diff(base, actual):
    remaining=actual.copy(); missing=0
    for (c,x,y), n in base.items():
        for _ in range(n):
            found=False
            for dx,dy in [(0,0)]+[(a,b) for a in range(-2,3) for b in range(-2,3) if (a,b)!=(0,0)]:
                k=(c,x+dx,y+dy)
                if remaining[k]>0:
                    remaining[k]-=1; found=True; break
            if not found: missing+=1
    return missing,+remaining

def analyze(path, tf, tp, tasks, timing=OFF):
    row=dict(filename=path.name,pdf_bytes=path.stat().st_size,status='INVALID',reason='')
    try:
        with timing.stage("PDF open"):
            d=pdf.open(path)
        with d:
            if not d.is_pdf: raise ValueError('PDFではありません')
            if d.needs_pass: raise ValueError('暗号化PDF')
            with timing.stage("Form and object extraction"):
                f,v,p=inspect(d,tp)
        with timing.stage("Structural comparison"):
            row.update({n:v.get(n,'') for n in tf})
            counts=[count_chars(v.get(n,'')) for n in tasks]
            row.update({n+'_chars':c for n,c in zip(tasks,counts)})
            row.update(filled_fields=sum(c>0 for c in counts),empty_fields=sum(c==0 for c in counts),form_text_chars=sum(counts))
            same_pages=len(p)==len(tp) and all(all(abs(x-y)<=2 for x,y in zip(a['size'],b['size'])) for a,b in zip(p,tp))
            same_schema=schema_equal(tf,f)
            extra=Counter(); ic=ib=ac=0
            for i,a in enumerate(p):
                b=tp[i] if i<len(tp) else dict(chars=Counter(),images=[],ann=Counter())
                _,e=char_diff(b['chars'],a['chars']); extra.update(e)
                old=Counter(h for h,s in b['images'])
                for h,s in a['images']:
                    if old[h]: old[h]-=1
                    else: ic+=1; ib+=s
                added=a['ann']-b['ann']; ac+=sum(added.values())
                # Annotation text may also appear in page text; count only separately in annotation count.
            row.update(extra_text_chars=sum(extra.values()),extra_text_bytes=sum(len(c.encode('utf-8'))*n for (c,x,y),n in extra.items()),extra_image_count=ic,extra_image_bytes=ib,extra_annotation_count=ac)
            if same_pages and same_schema:
                row.update(status='VALID',reason='フォーム構造・ページ構成一致')
            else:
                why=[]
                if not same_pages: why.append('ページ構成不一致')
                if not same_schema: why.append('フォーム構造不一致')
                row.update(status='INVALID',reason=' / '.join(why) or '雛形との不一致')
            if not same_schema:
                # Missing fields are unknown, not blank answers.
                for n in tasks:
                    if n not in f: row[n+'_chars']=''
                for k in ('filled_fields','empty_fields','form_text_chars'): row[k]=''
    except Exception as e:
        row.update(status='INVALID',reason=f'解析不能: {e}')
    return row

def natural(s):
    return [(0,int(x)) if x.isdigit() else (1,x.casefold()) for x in re.split(r'(\d+)',str(s))]

def normalize_student(row):
    cls = norm(unicodedata.normalize('NFKC', row.get('stu_class', ''))).upper()
    name = norm(row.get('stu_name', ''))
    if not cls or not name:
        return None
    course_num = norm(unicodedata.normalize('NFKC', row.get('course_num', '')))
    if not re.fullmatch(r'[0-9]{1,3}', course_num):
        return None
    stem = f'{course_num.zfill(3)}_{cls}_{name}'
    # Reject unsafe or unportable names rather than silently changing names.
    if re.search(r'[<>:"/\\|?*\x00-\x1f\x7f]', stem) or len((stem+'.pdf').encode('utf-8')) > 240:
        return None
    return stem + '.pdf'

ROSTER = ['受講名列番号', '学籍番号', '所属クラス', '学生氏名']
PDF_STUDENT = ['pdf_'+n for n in STUDENT]

def identity(value):
    return norm(unicodedata.normalize('NFKC', value))

def read_roster(path):
    if path.suffix.lower() != '.xlsx':
        raise ValueError('名簿はXLSX形式で指定してください')
    wb = load_workbook(path, read_only=True, data_only=False)
    try:
        ws = wb.worksheets[0]
        rows = ws.iter_rows()
        header = next(rows, ())
        columns = {}
        for i, cell in enumerate(header):
            name = str(cell.value or '').strip()
            if name in ROSTER:
                if name in columns:
                    raise ValueError(f'名簿の必須列が重複しています: {name}')
                columns[name] = i
        if not set(ROSTER).issubset(columns):
            raise ValueError('名簿の必須列が不足しています: '+', '.join(k for k in ROSTER if k not in columns))
        roster = []
        ids, names = defaultdict(set), defaultdict(set)
        for line, cells in enumerate(rows, 2):
            row = {}
            for k, i in columns.items():
                cell = cells[i]
                v = cell.value
                if v is None:
                    row[k] = ''
                elif cell.data_type in ('f', 'e'):
                    raise ValueError(f'名簿の必須列には数式・エラー値を使用できません: {line}行目 {k}')
                elif isinstance(v, str):
                    row[k] = v
                elif isinstance(v, (int, float)) and not isinstance(v, bool) and v == int(v):
                    row[k] = str(int(v))
                    # Preserve explicit zero-padding in Excel integer formats.
                    if v >= 0 and re.fullmatch(r'0+', cell.number_format or ''):
                        row[k] = row[k].zfill(len(cell.number_format))
                else:
                    raise ValueError(f'名簿の値が文字列または整数ではありません: {line}行目 {k}')
            if not any(v.strip() for v in row.values()): continue
            sid, name = identity(row['学籍番号']), identity(row['学生氏名'])
            if not sid or not name:
                raise ValueError(f'名簿の学籍番号または氏名が空: {line}行目')
            idx = len(roster)
            ids[sid].add(idx); names[name].add(idx)
            roster.append(row)
        if not roster: raise ValueError('名簿に学生がいません')
        return roster, ids, names
    finally:
        wb.close()

def match_student(row, ids, names):
    a = ids.get(identity(row.get('stu_id','')), set())
    b = names.get(identity(row.get('stu_name','')), set())
    if a and b:
        common = a & b
        if len(common)==1: return next(iter(common)), 'BOTH', ''
        return None, 'CONFLICT' if not common else 'AMBIGUOUS', '学籍番号と氏名の照合が矛盾または曖昧'
    if len(a)==1: return next(iter(a)), 'ID', '氏名不一致または未記入'
    if len(b)==1: return next(iter(b)), 'NAME', '学籍番号不一致または未記入'
    return None, 'AMBIGUOUS' if a or b else 'NONE', '名簿上の学生を一意に特定できません'

PROVENANCE = '.submission-names.json'


def file_digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_provenance(root):
    path = root / PROVENANCE
    if path.is_symlink():
        raise ValueError(f'管理ファイルがシンボリックリンクです: {path}')
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or any(
        not isinstance(k, str) or not isinstance(v, dict)
        or not isinstance(v.get('name'), str) or not isinstance(v.get('sha256'), str)
        for k, v in data.items()
    ):
        raise ValueError('提出ファイル名の管理情報が不正です')
    return data


def save_provenance(root, records):
    # Atomic replacement; student identifiers are local metadata.
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=root,
                                     prefix='.submission-', delete=False) as f:
        tmp = Path(f.name)
        try:
            json.dump(records, f, ensure_ascii=False, indent=2)
            f.write('\n')
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
    try:
        tmp.replace(root / PROVENANCE)
    finally:
        tmp.unlink(missing_ok=True)


def compile_filename_patterns(config):
    patterns = config.get('filename_patterns')
    if not isinstance(patterns, list) or not patterns:
        raise ValueError('[matching].filename_patterns に正規表現の配列を指定してください')
    compiled = []
    for i, pattern in enumerate(patterns, 1):
        if not isinstance(pattern, str):
            raise ValueError(f'filename_patterns[{i}]: 文字列が必要です')
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f'filename_patterns[{i}]: {exc}') from exc
        if regex.groups != 1:
            raise ValueError(f'filename_patterns[{i}]: キャプチャグループは1個必要です')
        compiled.append(regex)
    return compiled


def filename_candidates(filename, ids, patterns):
    stem = unicodedata.normalize('NFKC', Path(filename).stem)
    hits = set()
    for pattern in patterns:
        for match in pattern.finditer(stem):
            sid = identity(match.group(1) or '')
            if sid:
                hits.update(ids.get(sid, ()))
    return hits


def identify_submission(row, original_name, roster, ids, names, patterns):
    hits = filename_candidates(original_name, ids, patterns)
    if len(hits) > 1:
        return None, 'AMBIGUOUS', 'ファイル名に複数の学籍番号候補'
    if len(hits) == 1:
        idx = next(iter(hits))
        student = roster[idx]
        differences = []
        for field, label in [('stu_id', '学籍番号'), ('stu_name', '学生氏名')]:
            if identity(row.get(field, '')) != identity(student[label]):
                differences.append(label)
        why = 'PDF学生情報不一致・未記入: ' + '、'.join(differences) if differences else ''
        return idx, 'FILENAME', why
    return match_student(row, ids, names)


def backup_existing_output(output):
    """Keep the previous workbook intact before replacing it."""
    if not output.exists():
        return None
    for n in range(1, 100000):
        backup = output.with_name(f'{output.stem}.backup-{n:03d}.xlsx')
        try:
            with backup.open('xb') as dst:
                try:
                    with output.open('rb') as src:
                        shutil.copyfileobj(src, dst)
                except BaseException:
                    backup.unlink(missing_ok=True)
                    raise
            return backup
        except FileExistsError:
            continue
    raise ValueError('XLSXバックアップ名を確保できません')


def prepare_reports(source, output_dir=Path("out")):
    """Extract PDFs only, once per archive; never trust ZIP member paths."""
    if source.is_dir():
        return source.resolve()
    if not source.is_file() or source.suffix.lower() != '.zip':
        raise ValueError('提出物はZIPファイルまたはフォルダで指定してください')
    source = source.resolve()
    output_dir = Path(output_dir).resolve()
    root = output_dir / source.stem
    if root == source:
        raise ValueError('ZIPと出力先が衝突しています')
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = '.source-zip.sha256'
    with source.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
        if digest is None:
            h = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024*1024), b''):
                h.update(chunk)
            digest = h.hexdigest()
    if root.exists() or root.is_symlink():
        if root.is_symlink() or not root.is_dir() or (root/marker).is_symlink() or not (root/marker).is_file():
            raise ValueError(f'展開先が既に存在します: {root}（別名に変更してください）')
        if (root/marker).read_text(encoding='ascii').strip() != digest:
            raise ValueError(f'ZIPの内容が変わっています: {root}（旧フォルダを退避して再実行してください）')
        if not (root / PROVENANCE).is_file():
            raise ValueError(f'旧版の展開先に元ファイル名の管理情報がありません: {root}（旧フォルダを退避しZIPから再展開してください）')
        return root
    with tempfile.TemporaryDirectory(prefix='.reports-', dir=root.parent) as tmp:
        stage = Path(tmp)/'extracted'
        stage.mkdir()
        with ZipFile(source) as archive:
            entries = []
            total = 0
            for info in archive.infolist():
                name = info.filename.replace(chr(92), '/')
                parts = name.split('/')
                if name.startswith('/') or re.match(r'^[A-Za-z]:', name) or '..' in parts or stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError(f'安全でないZIPエントリ: {info.filename!r}')
                if info.is_dir() or name.endswith('/') or parts[0] == '__MACOSX' or parts[-1].startswith('._') or not parts[-1].lower().endswith('.pdf'):
                    continue
                if info.flag_bits & 1:
                    raise ValueError('暗号化されたZIP内PDFには対応していません')
                entries.append((info, parts[-1]))
                total += info.file_size
            if not entries:
                raise ValueError('ZIP内にPDFがありません')
            if total > shutil.disk_usage(stage).free:
                raise ValueError('ZIP展開に必要な空き容量が不足しています')
            records = {}
            for info, name in entries:
                original_name = name
                if re.search(r'[<>:"/\\|?*\x00-\x1f\x7f]', name) or len(name.encode('utf-8')) > 230:
                    name = 'submission.pdf'
                target = stage/name
                original = target
                n = 2
                while target.exists():
                    target = original.with_name(f'{original.stem}_{n}{original.suffix}')
                    n += 1
                with archive.open(info) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
                records[target.name] = {'name': original_name, 'sha256': file_digest(target)}
            save_provenance(stage, records)
        (stage/marker).write_text(digest+'\n', encoding='ascii')
        if root.exists() or root.is_symlink():
            raise ValueError(f'展開先が既に存在します: {root}')
        stage.rename(root)
    return root

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('template',type=Path); ap.add_argument('reports',type=Path,help='提出PDF群のZIPまたはフォルダ')
    ap.add_argument('roster',type=Path,help='受講者名簿XLSX')
    ap.add_argument('--dry-run',action='store_true',help='XLSXのみ生成しPDFは移動しない')
    ap.add_argument('--config',type=Path,default=Path(__file__).resolve().parent.parent / 'config.toml')
    ap.add_argument('--grading',type=Path,help='自然言語の採点基準UTF-8テキスト')
    ap.add_argument('--grading-plan',type=Path,help='確認済み採点ルールJSON（--gradingと併用）')
    ap.add_argument('--plan-only',action='store_true',help='採点ルールのみ生成（--grading必須）')
    ap.add_argument('--output-dir',type=Path,help='ZIP入力時の結果保存先（既定: out）')
    args=ap.parse_args(argv); template=args.template.resolve()
    if (args.grading_plan or args.plan_only) and not args.grading: ap.error('--grading が必要です')
    try:
        config = tomllib.loads(args.config.read_text(encoding='utf-8'))
        matching = config.get('matching', {})
        if not isinstance(matching, dict):
            raise ValueError('[matching] はテーブルで指定してください')
        patterns = compile_filename_patterns(matching)
        output_config = config.get('output', {})
        if not isinstance(output_config, dict):
            raise ValueError('[output] はテーブルで指定してください')
        directory = output_config.get('directory', 'out')
        if not isinstance(directory, str) or not directory.strip():
            raise ValueError('output.directory は空でない文字列で指定してください')
        output_dir = args.output_dir if args.output_dir is not None else Path(directory)
        if args.reports.is_dir() and args.output_dir is not None:
            raise ValueError('フォルダ入力は指定フォルダ内で処理するため --output-dir は使用できません')
    except (OSError, ValueError) as e:
        ap.error(str(e))
    logging_config=config.get("logging",{})
    level=logging_config.get("level","normal")
    if level not in ("simple","normal","debug"): ap.error("logging.level: simple / normal / debug")
    timing=Timing(level=="debug" and logging_config.get("show_timing",True))
    timing.level=level
    with timing.stage("Template rendering"):
        visual=Analyzer(template,config.get('vlm',{}),timing)
    with pdf.open(template) as d:
        if not d.is_pdf or d.needs_pass: ap.error('雛形PDFを読み込めません')
        tf,_,tp=inspect(d)
    if not tf: ap.error('雛形にフォームがありません')
    tasks=[n for n in tf if not n.startswith('stu_')]
    try:
        roster, ids, names = read_roster(args.roster)
    except (OSError, ValueError, InvalidFileException, BadZipFile) as e:
        ap.error(str(e))
    try:
        root = prepare_reports(args.reports, output_dir)
        output = root / (root.name + '.xlsx')
        plan_path = root / 'grading-plan.json'
        inputs = root / '_inputs'
        if inputs.is_symlink() or (inputs.exists() and not inputs.is_dir()):
            raise ValueError(f'入力控えの保存先が不正です: {inputs}')
        for target in (output, plan_path, inputs/'template.pdf', inputs/'grading.txt'):
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise ValueError(f'出力先が不正です: {target}')
        if output.resolve() == args.roster.resolve():
            raise ValueError('出力XLSXと名簿XLSXのパスが同一です．名簿を保護するため中止します')
        provenance = load_provenance(root)
        for name in ('valid', 'invalid', 'review', 'normal', 'graded'):
            if (root/name).is_symlink():
                raise ValueError(f'振り分け先がシンボリックリンクです: {root/name}')
        # Keep copies separate from the submission scan; never copy API settings.
        copies = [(template, inputs/'template.pdf')]
        if args.grading: copies.append((args.grading.resolve(), inputs/'grading.txt'))
        originals = {p.resolve() for p in (template, args.roster, args.config, args.grading, args.grading_plan) if p is not None}
        for target in (output, plan_path):
            if target.resolve() in originals and not (target == plan_path and args.grading_plan and target.resolve() == args.grading_plan.resolve()):
                raise ValueError(f'入力と出力が衝突します: {target}')
        for src, dst in copies:
            if dst.resolve() in originals and src.resolve() != dst.resolve():
                raise ValueError(f'入力と出力が衝突します: {dst}')
        inputs.mkdir(exist_ok=True)
    except (OSError, ValueError, BadZipFile, RuntimeError, NotImplementedError) as e:
        ap.error(str(e))
    grader=None
    pdf_writer=None
    if args.grading:
        from .grading import Grader, COLUMNS
        try:
            grading_config=config.get("grading",{})
            annotate=grading_config.get('annotate_pdf',True)
            if type(annotate) is not bool: raise ValueError('grading.annotate_pdf must be boolean')
            if annotate and not args.dry_run and not args.plan_only:
                from .annotated_pdf import GradePDFWriter
                pdf_writer=GradePDFWriter(grading_config)
                if (root/'graded').exists() and not (root/'graded').is_dir():
                    raise ValueError('graded の保存先が不正です')
            grader=Grader(template,args.grading,tasks,visual,grading_config,output,timing,args.grading_plan)
        except Exception as e: ap.error(f"採点基準の解析失敗: {e}")
    try:
        for src, dst in copies:
            if src.resolve() == dst.resolve(): continue
            with tempfile.NamedTemporaryFile(dir=inputs, delete=False) as f:
                temp=Path(f.name)
            try:
                shutil.copyfile(src,temp)
                temp.replace(dst)
            finally:
                temp.unlink(missing_ok=True)
    except OSError as e:
        ap.error(f'入力控えの保存失敗: {e}')
    print(f'Result folder: {root}', flush=True)
    if args.plan_only: return
    if not args.grading and plan_path.exists():
        print('注意: AI採点なしの実行です．以前の採点プラン・採点基準の控えは残ります．', flush=True)
    students=ROSTER
    extra_pages=[f'extra_p{i+1}' for i in range(len(tp))]
    visual_meta=['extraction_method','alignment','alignment_score','template_match','extra_area_percent','vlm_status','vlm_error']
    headers=ROSTER+['点数','備考']+(COLUMNS+(['採点済PDF','採点済PDFエラー'] if pdf_writer else []) if grader else [])+['filename','status','match','reason']+META[2:]+visual_meta+[n+'_chars' for n in tasks]+PDF_STUDENT+tasks+extra_pages
    if len(headers)!=len(set(headers)): ap.error('XLSX列名が衝突しています')
    row_records={}; rows=[]; plans=[]; row_paths={}; candidates=defaultdict(list); invalid_matches=defaultdict(list)
    folders=(root,root/'valid',root/'invalid',root/'review',root/'normal')
    total=sum(1 for folder in folders if folder.is_dir() for path in folder.iterdir() if path.is_file() and not path.is_symlink() and path.suffix.lower()=='.pdf' and path.resolve()!=template)
    for folder in folders:
        if not folder.is_dir(): continue
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.is_symlink() or path.suffix.lower()!='.pdf' or path.resolve()==template: continue
            print(f"\nPDF解析 [{len(rows)+1}/{total}] {pdf_label(path)}",flush=True)
            preview_notify(path)
            timing.begin(path, path.name)
            row=analyze(path,tf,tp,tasks,timing)
            row['vlm_status']='SKIPPED' if visual.vision.enabled else 'DISABLED'
            row['vlm_error']=''
            try: visual.run(path,row,tf,tasks)
            except TemplateMismatch as e:
                row.update(status='INVALID',reason=str(e))
                row['template_match']='MISMATCH'
                row.setdefault('alignment','FAILED')
            except TemplateUncertain as e:
                row.update(status='REVIEW',reason=str(e))
                row.setdefault('alignment','FAILED')
                row['template_match']=(row.get('template_match','')+' / UNKNOWN').strip(' /')
                for n in tasks: row[n+'_chars']=''
                for k in ('filled_fields','empty_fields','form_text_chars'): row[k]=''
            except Exception as e:
                row['alignment']='FAILED'; row['reason']+=f' / 画像解析失敗: {e}'
                if row['status']!='VALID' and not row['reason'].startswith('解析不能:'):
                    row['status']='REVIEW'; row['reason']=f'画像解析失敗: {e}'
                    for n in tasks: row[n+'_chars']=''
                    for k in ('filled_fields','empty_fields','form_text_chars'): row[k]=''
            timing.page=''
            rows.append(row); row_paths[id(row)]=path
            with timing.stage("Roster matching and provenance"):
                row.update({'pdf_'+n:row.get(n,'') for n in STUDENT})
                key = path.relative_to(root).as_posix()
                digest = file_digest(path)
                record = provenance.get(key, {})
                if record.get('sha256') != digest:
                    record = {'name': path.name, 'sha256': digest}
                row_records[id(row)] = record
                idx, method, why=identify_submission(row,record['name'],roster,ids,names,patterns)
                row['match']=method
                if why: row['reason'] += ' / '+why
                if idx is not None:
                    row.update(roster[idx])
                    (candidates if row['status']=='VALID' else invalid_matches)[idx].append(row)
                elif row['status']=='VALID': row['status']='INVALID'
            print(f"  結果: {row['status']}",flush=True)
            if row['status']!='VALID': detail(timing,"理由",row.get("reason",""))
            timing.subtotal(path)
    timing.current=None
    result_rows=[]; unmatched=[]; selected=set()
    for idx, student in enumerate(roster):
        group=candidates[idx]; bad=invalid_matches[idx]
        if len(group)==1:
            row=group[0]; result_rows.append(row); selected.add(id(row))
            for other in bad: other['reason']+=' / 同一学生の適合PDFを優先採用'
        elif len(group)>1:
            result_rows.append(dict(student,status='DUPLICATE',reason=f'適合PDFが{len(group)}件: Unmatchedで確認'))
            for row in group:
                row['status']='INVALID'; row['reason']+=' / 重複提出: 自動採用なし'
        elif len(bad)==1:
            row=bad[0]; result_rows.append(row); selected.add(id(row))
        elif len(bad)>1:
            result_rows.append(dict(student,status='DUPLICATE',reason=f'規格外・要確認PDFが{len(bad)}件: Unmatchedで確認'))
            for row in bad: row['reason']+=' / 重複提出: 自動採用なし'
        else:
            result_rows.append(dict(student,status='MISSING',reason='照合できた提出PDFなし', **{'点数': 0, '備考': '未提出'}))
    unmatched=[r for r in rows if id(r) not in selected]
    if grader:
        for i,row in enumerate(result_rows,1):
            print(f"\nAI採点 [{i}/{len(result_rows)}] {student_label(row)}",flush=True)
            preview_notify(row_paths.get(id(row)))
            row.update(grader.grade(row,row_paths.get(id(row))))
            score=row.get("AI点数")
            print(f"  AI点数: {score if score is not None else '未確定'} / {grader.plan['maximum']} 点 ({row.get('AI採点状態')})",flush=True)
            detail(timing,"採点理由",row.get("AI採点理由",""))
    for row in rows:
        path=row_paths[id(row)]; desired=None
        if '受講名列番号' in row:
            canonical=dict(course_num=row['受講名列番号'],stu_class=row['所属クラス'],stu_name=row['学生氏名'])
            desired=normalize_student(canonical)
            if desired is None: row['reason']+=' / 改名不可: 受講名列番号・所属クラス・氏名またはファイル名不適合'
        target=root/row['status'].lower()/(desired or path.name)
        if path.parent==target.parent and (path==target or re.fullmatch(re.escape(target.stem)+r'_[0-9]+'+re.escape(target.suffix),path.name)): continue
        if args.dry_run: continue
        original=target; n=2
        while target.exists() or any(t==target for _,t,_ in plans):
            target=original.with_name(f'{original.stem}_{n}{original.suffix}'); n+=1
        plans.append((path,target,row))
    graded_paths={}
    preview_notify(None)
    def fill_sheet(ws, rows):
        ws.freeze_panes = f'{get_column_letter(len(students)+4)}2'
        ws.sheet_view.showGridLines = True
        ws.append(headers)
        ws.row_dimensions[1].height = 32
        red = PatternFill('solid', fgColor='FADADD')
        green = PatternFill('solid', fgColor='DDF0DF')
        for col, name in enumerate(headers, 1):
            cell = ws.cell(1, col)
            cell.data_type = 's'
            letter = get_column_letter(col)
            if name == 'filename' or name in students or name in PDF_STUDENT:
                color = 'E2E8F0'
                width = 25 if name == 'filename' else 14
            elif name in ('点数', '備考'):
                color = 'FFF0C2'
                width = 10 if name == '点数' else 25
            elif grader and name in COLUMNS+['採点済PDF','採点済PDFエラー']:
                color = 'DDF0DF'
                width = 45 if name in ('AI採点理由','採点済PDFエラー') else 25 if name == '採点済PDF' else 14
            elif name in META or name in visual_meta or name == 'match':
                color = 'DCE8F7'
                width = 42 if name in ('reason','vlm_error') else max(14, min(24, len(name)+2))
            elif name in tasks or name in extra_pages:
                color = 'FCE8CF'
                width = 40
            else:
                color = 'DDF0DF'
                width = max(13, min(24, len(name)+2))
            cell.fill = PatternFill('solid', fgColor=color)
            cell.font = Font(bold=True, color='24364B')
            cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
            ws.column_dimensions[letter].width = width
        for idx, row in enumerate(rows, 2):
            ws.row_dimensions[idx].height = 30
            for col, name in enumerate(headers, 1):
                value = row.get(name, None if name in ('点数', '備考') else '')
                cell = ws.cell(idx, col)
                if isinstance(value, str):
                    # Preserve literal text, including formula-like answers and leading zeros.
                    clean = ILLEGAL_CHARACTERS_RE.sub('\uFFFD', value)
                    if len(clean) > 32767:
                        clean = clean[:32766] + '…'
                    cell.value = clean
                    cell.data_type = 's'
                    cell.number_format = '@'
                else:
                    cell.value = value
                    cell.number_format = 'General' if name in ('点数', '備考', 'AI点数') else ('0.000' if name == 'extra_area_percent' else '#,##0')
                cell.alignment = Alignment(vertical='center', wrap_text=name in tasks or name in extra_pages or name in ('reason', '備考', 'vlm_error', 'AI採点理由', '採点済PDFエラー'), horizontal='center' if name in ('点数', 'AI点数') else 'right' if isinstance(value, int) else 'left')
                if (name == 'filename' and id(row) in row_paths) or (name == '採点済PDF' and id(row) in graded_paths):
                    actual = (row_paths if name == 'filename' else graded_paths)[id(row)]
                    try:
                        relative = Path(os.path.relpath(actual, output.parent)).as_posix()
                        target_uri = quote(relative, safe='/')
                    except ValueError:
                        # Windows paths on different drives cannot be relative.
                        target_uri = actual.as_uri()
                    cell.hyperlink = target_uri
                    cell.font = Font(color='0563C1', underline='single')
        ws.auto_filter.ref = ws.dimensions
        if rows:
            end = len(rows)+1
            status = get_column_letter(headers.index('status')+1)
            for value, fill, color in [('VALID', green, '166534'), ('REVIEW', PatternFill('solid',fgColor='FFF0C2'), '92400E'), ('INVALID', red, '991B1B'), ('MISSING', red, '991B1B'), ('DUPLICATE', red, '991B1B')]:
                ws.conditional_formatting.add(f'{status}2:{status}{end}', FormulaRule(formula=[f'{status}2="{value}"'], fill=fill, font=Font(color=color)))
            match = get_column_letter(headers.index('match')+1)
            ws.conditional_formatting.add(f'{match}2:{match}{end}', FormulaRule(formula=[f'OR({match}2="ID",{match}2="NAME")'], fill=PatternFill('solid',fgColor='FFF0C2')))
            reason = get_column_letter(headers.index('reason')+1)
            ws.conditional_formatting.add(f'{match}2:{match}{end}', FormulaRule(formula=[f'AND({match}2="FILENAME",ISNUMBER(SEARCH("PDF学生情報不一致",{reason}2)))'], fill=PatternFill('solid',fgColor='FFF0C2')))
            vlm = get_column_letter(headers.index('vlm_status')+1)
            for value,fill,color in [('OK',green,'166534'),('PARTIAL',PatternFill('solid',fgColor='FFF0C2'),'92400E'),('ERROR',red,'991B1B')]:
                ws.conditional_formatting.add(f'{vlm}2:{vlm}{end}', FormulaRule(formula=[f'{vlm}2="{value}"'],fill=fill,font=Font(color=color)))
            if grader:
                ai = get_column_letter(headers.index('AI採点状態')+1)
                ws.conditional_formatting.add(f'{ai}2:{ai}{end}', FormulaRule(formula=[f'{ai}2="REVIEW"'],fill=PatternFill('solid',fgColor='FFF0C2')))
            # Keep highlights responsive to teacher edits and distinguish blank scores from zero.
            for name in ('点数', 'AI点数'):
                if name in headers:
                    letter = get_column_letter(headers.index(name)+1)
                    ws.conditional_formatting.add(f'{letter}2:{letter}{end}', FormulaRule(formula=[f'AND(ISNUMBER({letter}2),{letter}2=0)'], fill=red))
            if 'AI採点理由' in headers:
                letter = get_column_letter(headers.index('AI採点理由')+1)
                ws.conditional_formatting.add(f'{letter}2:{letter}{end}', FormulaRule(formula=[f'AND({letter}2<>"",{letter}2<>"減点なし")'], fill=PatternFill('solid', fgColor='FFF0C2')))
            empty = get_column_letter(headers.index('empty_fields')+1)
            ws.conditional_formatting.add(f'{empty}2:{empty}{end}', FormulaRule(formula=[f'AND(ISNUMBER({empty}2),{empty}2<>0)'], fill=red, font=Font(color='991B1B')))
            for name in tasks:
                letter = get_column_letter(headers.index(name+'_chars')+1)
                ws.conditional_formatting.add(f'{letter}2:{letter}{end}', FormulaRule(formula=[f'AND(ISNUMBER({letter}2),{letter}2=0)'], fill=red, font=Font(color='991B1B')))
    def write_xlsx():
        wb=Workbook()
        wb.active.title='Results'
        fill_sheet(wb.active,result_rows)
        fill_sheet(wb.create_sheet('Unmatched'),unmatched)
        with tempfile.NamedTemporaryFile(suffix='.xlsx', dir=output.parent, delete=False) as f:
            temp = Path(f.name)
        try:
            wb.save(temp)
            temp.replace(output)
        finally:
            temp.unlink(missing_ok=True)
            wb.close()
    try:
        backup = backup_existing_output(output)
        if backup:
            print(f'Previous XLSX backup: {backup}', flush=True)
    except (OSError, ValueError) as e:
        ap.error(f'既存XLSXのバックアップ失敗: {e}')
    with timing.stage("XLSX output"):
        write_xlsx()
    if not args.dry_run:
        # Persist original names before any move, including already classified PDFs.
        provenance = {row_paths[id(r)].relative_to(root).as_posix(): row_records[id(r)] for r in rows}
        save_provenance(root, provenance)
    for src,dst,row in ([] if args.dry_run else plans):
        timing.current=src
        timing.say(f"Move: {src.name} -> {dst.name}")
        try:
            with timing.stage("File rename and journal"):
                dst.parent.mkdir(exist_ok=True)
                # Journal both names before moving, so interruption cannot lose identity.
                old_key, new_key = src.relative_to(root).as_posix(), dst.relative_to(root).as_posix()
                provenance[new_key] = row_records[id(row)]
                save_provenance(root, provenance)
                shutil.move(str(src),str(dst))
                provenance.pop(old_key, None)
                row['filename']=dst.name
                row_paths[id(row)] = dst
        except OSError as e:
            row['reason']+=f' / 移動失敗: {e}'
        timing.subtotal(src)
    timing.current=None
    if not args.dry_run:
        save_provenance(root, {row_paths[id(r)].relative_to(root).as_posix(): row_records[id(r)] for r in rows})
    if pdf_writer:
        used=set()
        for row in result_rows:
            source=row_paths.get(id(row))
            if source is None: continue
            target=root/'graded'/source.name
            n=2
            while target in used:
                target=target.with_name(f'{source.stem}_{n}{source.suffix}'); n+=1
            used.add(target)
            try:
                with timing.stage("Graded PDF output"):
                    pdf_writer.write(source,target,row,grader.plan['maximum'])
                row['採点済PDF']=target.name
                graded_paths[id(row)]=target
            except Exception as e:
                row['採点済PDFエラー']=str(e)
                print(f'Graded PDF failed: {source.name}: {e}',flush=True)
        print(f'Graded PDFs: {len(graded_paths)}',flush=True)
    with timing.stage("XLSX output"):
        write_xlsx()
    if not args.dry_run:
        for name in ('valid', 'invalid', 'review'):
            (root/name).mkdir(exist_ok=True)
        for name in ('normal',):
            try: (root/name).rmdir()
            except OSError: pass
    print('PDF: ' + ' / '.join(f'{k}: {sum(r["status"]==k for r in rows)}' for k in ('VALID','REVIEW','INVALID')))
    print('Students: ' + ' / '.join(f'{k}: {sum(r["status"]==k for r in result_rows)}' for k in ('VALID','REVIEW','INVALID','MISSING','DUPLICATE')))
    print('VLM PDFs: ' + ' / '.join(f'{k}: {sum(r.get("vlm_status")==k for r in rows)}' for k in ('OK','PARTIAL','ERROR','SKIPPED','DISABLED')))
    print('VLM operations: ' + ' / '.join(f'{k}: {sum(r.get("_vlm_events",[]).count(k) for r in rows)}' for k in ('OK','PARTIAL','ERROR','SKIPPED')))
    if grader: grader.print_visual_summary()
    print(output)
    timing.summary()

if __name__=='__main__': main()
