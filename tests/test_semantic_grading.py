import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock
from report_grader.grading import Grader, validate_plan, cacheable
from report_grader.timing import OFF

RULE={'label':'波形未完成','kind':'semantic_condition','condition':'波形が未完成','pages':[1],'fields':[],'threshold':0,'deduction':1}
OTHER=dict(RULE,label='写真不足',pages=[2],condition='回路写真なし')
ROW={'status':'VALID','Q2':'観察結果','Q2_chars':4,'extra_p1':'cntの波形が追加記入されています','extra_p2':'回路写真あり','_description_status':{1:'OK',2:'OK'}}
def judgement(rule,decision,pages=(),reason='根拠',source='description',quote=None):
    if decision=='PASS':
        page=rule['pages'][0]; pages=pages or [page]
        evidence=[dict(source=source,page=page,field='',quote=quote or ROW[f'extra_p{page}'])]
    else: evidence=[]
    return dict(label=rule['label'],decision=decision,pages=list(pages),reason=reason,evidence=evidence)
def batch(*items): return {'judgements':list(items)}
class SemanticTest(unittest.TestCase):
    def make(self,mode='auto',enabled=True):
        g=Grader.__new__(Grader);g.visual_mode=mode;g.visual_stats=Counter();g.timing=OFF
        g.tasks=['Q2'];g.context=[{'page':1},{'page':2}];g.visual=SimpleNamespace(vision=SimpleNamespace(enabled=enabled,grading_requests=0))
        g.ask_json=Mock();g.image_batch=Mock();return g
    def test_batch_all_evidence(self):
        g=self.make();g.ask_json.return_value=batch(judgement(RULE,'PASS'),judgement(OTHER,'PASS'))
        out=g.semantic_batch('pdf',ROW,[RULE,OTHER]);self.assertEqual(out[RULE['label']][0],'PASS');self.assertEqual(g.ask_json.call_count,1)
        data=g.ask_json.call_args.args[1];self.assertEqual(data['evidence']['answers']['Q2']['text'],'観察結果');self.assertIn('波形',data['evidence']['page_descriptions'][0]['description']);g.image_batch.assert_not_called()
    def test_need_image_grouped(self):
        g=self.make();g.ask_json.return_value=batch(judgement(RULE,'NEED_IMAGE',[1]),judgement(OTHER,'NEED_IMAGE',[2]));g.image_batch.return_value={r['label']:judgement(r,'PASS',source='image') for r in (RULE,OTHER)}
        out=g.semantic_batch('pdf',ROW,[RULE,OTHER]);self.assertEqual(out[RULE['label']][0],'PASS');self.assertEqual(g.image_batch.call_count,1);self.assertEqual(g.image_batch.call_args.args[3],[1,2])
    def test_disagreement_no_deduction(self):
        g=self.make('text');g.ask_json.side_effect=[batch(judgement(RULE,'DEDUCT')),batch(judgement(RULE,'PASS'))];self.assertEqual(g.semantic_batch('pdf',ROW,[RULE])[RULE['label']][0],'UNKNOWN');g.image_batch.assert_not_called()
    def test_confirmed_deduction(self):
        g=self.make();g.ask_json.side_effect=[batch(judgement(RULE,'DEDUCT')),batch(judgement(RULE,'DEDUCT'))];self.assertEqual(g.semantic_batch('pdf',ROW,[RULE])[RULE['label']][0],'DEDUCT');g.image_batch.assert_not_called()
    def test_image_disagreement(self):
        g=self.make();g.ask_json.return_value=batch(judgement(RULE,'NEED_IMAGE',[1]));g.image_batch.side_effect=[{RULE['label']:judgement(RULE,'DEDUCT',[1])},{RULE['label']:judgement(RULE,'PASS',[1],source='image')}];self.assertEqual(g.semantic_batch('pdf',ROW,[RULE])[RULE['label']][0],'UNKNOWN')
    def test_disabled_and_text(self):
        for mode,enabled in [('text',True),('auto',False)]:
            g=self.make(mode,enabled);g.ask_json.return_value=batch(judgement(RULE,'NEED_IMAGE',[1]));self.assertEqual(g.semantic_batch('pdf',ROW,[RULE])[RULE['label']][0],'UNKNOWN');g.image_batch.assert_not_called()
    def test_image_first(self):
        g=self.make('image');g.image_batch.return_value={RULE['label']:judgement(RULE,'PASS',source='image')};self.assertEqual(g.semantic_batch('pdf',ROW,[RULE])[RULE['label']][0],'PASS');g.ask_json.assert_not_called()
    def test_schema(self):
        validate=Grader.batch_validator([RULE],2)
        for obj in [batch(),batch(judgement(RULE,'NEED_IMAGE')),batch(judgement(RULE,'NEED_IMAGE',[3])),batch(judgement(OTHER,'PASS'))]:
            with self.assertRaises(ValueError): validate(obj)
        self.assertFalse(cacheable(batch(judgement(RULE,'NEED_IMAGE',[1]))))
    def test_old_plan_rejected(self):
        plan=dict(maximum=3,minimum_submitted=1,missing_score=0,unrelated_score=1,rules=[dict(RULE,kind='text_condition')],warnings=[])
        with self.assertRaises(ValueError):validate_plan(plan,['Q2'],2)
    def test_score_and_review(self):
        g=self.make('text');g.plan=dict(maximum=3,minimum_submitted=1,missing_score=0,unrelated_score=1,rules=[RULE],warnings=[]);g.ask_json.side_effect=[batch(judgement(RULE,'DEDUCT')),batch(judgement(RULE,'PASS'))];out=g.grade(ROW,'pdf');self.assertEqual(out['AI点数'],3);self.assertEqual(out['AI採点状態'],'REVIEW')

    def test_missing_visual_evidence_image_check(self):
        g=self.make(); photo=dict(OTHER,pages=[1]); row=dict(ROW,extra_p1='追加記入なし')
        g.plan=dict(maximum=3,minimum_submitted=1,missing_score=0,unrelated_score=1,rules=[RULE,photo],warnings=[])
        # The model must request image verification; Python does not interpret prose.
        g.ask_json.return_value=batch(judgement(RULE,'NEED_IMAGE',[1]),judgement(photo,'NEED_IMAGE',[1]))
        deductions={r['label']:judgement(r,'DEDUCT',[1],reason='指定欄は白紙') for r in [RULE,photo]}
        g.image_batch.side_effect=[deductions,deductions]
        out=g.grade(row,'pdf'); self.assertEqual(out['AI点数'],1); self.assertEqual(out['AI採点状態'],'OK'); self.assertEqual(g.image_batch.call_count,2)
    def test_missing_images_review(self):
        g=self.make(enabled=False); g.plan=dict(maximum=3,minimum_submitted=1,missing_score=0,unrelated_score=1,rules=[RULE],warnings=[])
        g.ask_json.return_value=batch(judgement(RULE,'NEED_IMAGE',[1]))
        out=g.grade(ROW,'pdf'); self.assertEqual(out['AI採点状態'],'REVIEW'); self.assertNotEqual(out['AI採点理由'],'減点なし')
    def test_no_semantic_override_of_text_pass(self):
        g=self.make(); rule=dict(RULE,label='課題3の描き込み不足',condition='図に描き込みがない',pages=[2]); row=dict(ROW,extra_p2='指定位置に赤い円が追加されています')
        g.ask_json.return_value=batch(judgement(rule,'PASS',reason='指定位置に赤い円が追加されている',quote=row['extra_p2']))
        self.assertEqual(g.semantic_batch('pdf',row,[rule])[rule['label']][0],'PASS');g.image_batch.assert_not_called()
    def test_image_pass_resolves_prior_uncertainty(self):
        g=self.make(); rule=dict(RULE,label='課題3の描き込み不足',condition='図に描き込みがない',pages=[2]);g.plan=dict(maximum=3,minimum_submitted=1,missing_score=0,unrelated_score=1,rules=[rule],warnings=[])
        g.ask_json.return_value=batch(judgement(rule,'NEED_IMAGE',[2],reason='根拠不足'))
        g.image_batch.return_value={rule['label']:judgement(rule,'PASS',source='image',quote='指定位置に赤い円',reason='指定位置に赤い円がある')}
        self.assertEqual(g.grade(ROW,'pdf'),{'AI点数':3,'AI採点理由':'減点なし','AI採点状態':'OK'})
    def test_text_rule_with_grounded_answer(self):
        g=self.make();rule=dict(RULE,label='説明不足',condition='観察結果の記述がない',pages=[],fields=['Q2']); j=dict(label=rule['label'],decision='PASS',reason='観察結果が記載されている',pages=[],evidence=[dict(source='form',page=0,field='Q2',quote='観察結果')]);g.ask_json.return_value=batch(j)
        self.assertEqual(g.semantic_batch('pdf',ROW,[rule])[rule['label']][0],'PASS');g.image_batch.assert_not_called()
    def test_uncertain_image_does_not_pass(self):
        g=self.make('image');g.image_batch.return_value={RULE['label']:judgement(RULE,'UNKNOWN',reason='判読不能')}
        self.assertEqual(g.semantic_batch('pdf',ROW,[RULE])[RULE['label']][0],'UNKNOWN')

    def test_image_confirmed_blank_photo_deduction(self):
        g=self.make(); photo=dict(OTHER,pages=[1]);g.plan=dict(maximum=3,minimum_submitted=1,missing_score=0,unrelated_score=1,rules=[photo],warnings=[])
        g.ask_json.return_value=batch(judgement(photo,'NEED_IMAGE',[1],reason='写真への言及なし'))
        a=judgement(photo,'DEDUCT',[1],reason='写真欄に画像なし'); b=judgement(photo,'DEDUCT',[1],reason='指定枠内は白紙')
        g.image_batch.side_effect=[{photo['label']:a},{photo['label']:b}]
        out=g.grade(ROW,'pdf');self.assertEqual(out['AI点数'],2);self.assertEqual(out['AI採点状態'],'OK');self.assertIn('指定枠内は白紙',out['AI採点理由']);self.assertNotIn('言及なし',out['AI採点理由']);self.assertNotIn('再検証',out['AI採点理由'])

    def test_image_schema_rejects_unseen_pages(self):
        validate=Grader.batch_validator([RULE],2,[1])
        with self.assertRaises(ValueError):validate(batch(judgement(RULE,'PASS',[2],source='image')))
        with self.assertRaises(ValueError):validate(batch(judgement(RULE,'DEDUCT')))
        self.assertEqual(validate(batch(judgement(RULE,'PASS',source='image',quote='赤い円')))['judgements'][0]['decision'],'PASS')

    def test_stage_prompts(self):
        g=self.make(); text=g.semantic_prompt(); image=g.semantic_prompt(image=True)
        self.assertIn('欠如を断定できない',text);self.assertIn('NEED_IMAGEと対象ページ',text);self.assertIn('画像確認後はNEED_IMAGEを使わない',image);self.assertNotIn('テキスト段階では',image)

    def test_internal_error_not_in_final_reason(self):
        from report_grader.visual_analysis import RetryableResponse
        for stage in ('initial','verification'):
            g=self.make();g.plan=dict(maximum=3,minimum_submitted=1,missing_score=0,unrelated_score=1,rules=[RULE],warnings=[])
            g.ask_json.return_value=batch(judgement(RULE,'NEED_IMAGE',[1]))
            err=RetryableResponse('Invalid structured VLM response')
            g.image_batch.side_effect=[err] if stage=='initial' else [{RULE['label']:judgement(RULE,'DEDUCT',[1])},err]
            out=g.grade(ROW,'pdf');self.assertEqual(out['AI採点状態'],'REVIEW');self.assertEqual(out['AI点数'],3)
            self.assertIn('有効な判定結果',out['AI採点理由']);self.assertNotIn('RetryableResponse',out['AI採点理由']);self.assertNotIn('Invalid structured',out['AI採点理由'])

    def test_blank_form_quote_valid_in_both_stages(self):
        rule=dict(RULE,fields=['Q2']); item=dict(label=rule['label'],decision='DEDUCT',reason='未記入',pages=[1],evidence=[dict(source='form',page=0,field='Q2',quote='')])
        for pages in (None,[1]):
            validate=Grader.batch_validator([rule],2,pages)
            self.assertEqual(validate(batch(item)),batch(item))
            for bad in (dict(source='form',page=1,field='Q2',quote=''),dict(source='form',page=0,field='',quote=''),dict(source='form',page=0,field='Q2',quote=None),dict(source='description',page=1,field='',quote=''),dict(source='image',page=1,field='',quote=' ')):
                with self.assertRaises(ValueError): validate(batch(dict(item,evidence=[bad])))

    def test_blank_submission_scores_minimum(self):
        g=self.make(); g.tasks=['Q1','Q2','Q3','Q7']; row=dict(ROW,extra_p1='追加記入なし',extra_p2='追加記入なし')
        rules=[]; items=[]
        for n,(field,points,page) in enumerate([('Q1',1,1),('Q2',1,1),('Q3',2,1),('',2,1),('',1,2),('',1,2)]):
            r=dict(RULE,label=f'未記入{n}',deduction=points,fields=[field] if field else [],pages=[page]); rules.append(r)
            if field: row[field]='';row[field+'_chars']=0
            ev=dict(source='form',page=0,field=field,quote='') if field else dict(source='description',page=page,field='',quote='追加記入なし')
            items.append(dict(label=r['label'],decision='DEDUCT',reason='未記入',pages=[page],evidence=[ev]))
        for label in ('周期','デューティ比'):
            r=dict(RULE,label=label); rules.append(r); items.append(dict(label=label,decision='PASS',reason='未記入の減点と重複しない',pages=[1],evidence=[dict(source='description',page=1,field='',quote='追加記入なし')]))
        row['Q7_chars']=0
        g.plan=dict(maximum=10,minimum_submitted=1,missing_score=0,unrelated_score=1,rules=rules+[dict(RULE,label='考察不足',kind='char_count_below',fields=['Q7'],threshold=200)],warnings=[])
        g.ask_json.side_effect=lambda prompt,data,validator: validator(batch(*(i for i in items if i['label'] in {r['label'] for r in data['rules']})))
        out=g.grade(row,'pdf'); self.assertEqual(out['AI点数'],1);self.assertEqual(out['AI採点状態'],'OK');self.assertEqual(g.ask_json.call_count,2);g.image_batch.assert_not_called()
