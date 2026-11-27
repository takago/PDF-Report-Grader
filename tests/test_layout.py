import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import cv2
import numpy as np
import pymupdf as pdf
from report_grader.layout import page_boxes, rectangle_transform
from report_grader.visual_analysis import Analyzer, register, render, TemplateMismatch, TemplateUncertain
from report_grader.grading import Grader
from report_grader.collector import analyze, inspect

# Long fixed lines with two different fonts produce very different ink pixels.
LINES=[('Embedded assignment w01',40,40),('Explain the operation of the first counter.',40,150),('Describe the result of the second experiment.',40,340),('Record the measured value of the third signal.',40,530)]
BOXES=[(42,70,240,115),(310,72,550,120),(42,185,550,295),(42,375,550,480),(42,565,550,700)]

def make_doc(path,font='helv',widgets=True,wrong=False,answers=False):
    with pdf.open() as d:
        p=d.new_page(width=595,height=842)
        for text,x,y in LINES:
            if wrong: text=text.replace('w01','w02').replace('counter','motor').replace('experiment','circuit').replace('signal','voltage')
            p.insert_text((x,y),text,fontname=font,fontsize=14)
        for i,b in enumerate(BOXES):
            r=pdf.Rect(b); p.draw_rect(r,color=(.5,.5,.5),width=1)
            if widgets:
                w=pdf.Widget(); w.field_name='Q'+str(i); w.field_type=pdf.PDF_WIDGET_TYPE_TEXT; w.rect=r; w.border_width=1; w.border_color=(.5,.5,.5); p.add_widget(w)
            elif answers: p.insert_text((r.x0+15,r.y0+20),'ANSWER '+str(i),fontsize=11)
        d.save(path)

def scale_file(source,dest,raster=False):
    with pdf.open(source) as d,pdf.open() as out:
        p=out.new_page(width=612,height=792); r=pdf.Rect(35,35,577,757)
        if raster:
            pix=d[0].get_pixmap(dpi=150,alpha=False); p.insert_image(r,pixmap=pix,keep_proportion=False)
        else: p.show_pdf_page(r,d,0,keep_proportion=False)
        out.save(dest)

class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.template=self.root/'template.pdf'
        make_doc(self.template); self.a=Analyzer(self.template,{'dpi':150})
        self.addCleanup(self.a.doc.close)
        self.tasks=['Q'+str(i) for i in range(len(BOXES))]
    def run_file(self,path):
        row={'status':'INVALID','reason':'flattened','vlm_status':'DISABLED'}
        self.a.run(path,row,{},self.tasks); return row
    def test_vector_scaled_and_answers(self):
        src=self.root/'source.pdf'; out=self.root/'letter.pdf'
        make_doc(src,widgets=False,answers=True); scale_file(src,out)
        row=self.run_file(out)
        self.assertEqual(row['status'],'VALID'); self.assertIn('RECT',row['alignment']);self.assertIn('PIXEL',row['template_match'])
        self.assertEqual([row[n] for n in self.tasks],['ANSWER '+str(i) for i in range(len(BOXES))])
    def test_wrong_assignment_same_rectangles(self):
        src=self.root/'wrong.pdf'; make_doc(src,font='cour',widgets=False,wrong=True)
        with self.assertRaises(TemplateUncertain): self.run_file(src)
    def test_raster_rectangles_without_identity_ocr(self):
        src=self.root/'source.pdf'; out=self.root/'raster.pdf'
        make_doc(src,widgets=False); scale_file(src,out,raster=True)
        self.a.vision.enabled=True
        def ask(prompt,images,purpose='description',**kwargs):
            self.assertIn(purpose,('ocr','description'))
            return 'EMPTY' if purpose=='ocr' else '追加記入なし'
        self.a.vision.ask=Mock(side_effect=ask)
        row=self.run_file(out); self.assertEqual(row['status'],'VALID');self.assertIn('PIXEL',row['template_match'])
    def test_changed_font_low_score_not_accepted(self):
        src=self.root/'source.pdf'; out=self.root/'letter.pdf'
        make_doc(src,font='cour',widgets=False); scale_file(src,out)
        self.a.vision.enabled=True; self.a.vision.ask=Mock()
        with self.assertRaises(TemplateUncertain): self.run_file(out)
        self.a.vision.ask.assert_not_called()
    def test_form_structure_bypasses_pixel_gate(self):
        src=self.root/'filled.pdf'; make_doc(src)
        with pdf.open(src) as d:
            p=d[0]
            for w in p.widgets():
                w.field_value='ANSWER'; w.update()
            # Obscure fixed ink: form validation must not be overturned by image comparison.
            p.draw_rect(pdf.Rect(0,0,595,842),color=None,fill=(1,1,1),overlay=True)
            d.saveIncr()
        with pdf.open(self.template) as d: tf,_,tp=inspect(d)
        row=analyze(src,tf,tp,self.tasks)
        self.assertEqual(row['status'],'VALID')
        self.a.vision.enabled=True
        self.a.vision.ask=Mock(return_value='追加記入あり')
        self.a.run(src,row,tf,self.tasks)
        self.assertEqual(row['status'],'VALID'); self.assertEqual(row['template_match'],'P1: FORM')
        self.assertEqual(row['alignment_score'],''); self.assertEqual(row['filled_fields'],len(self.tasks))
        self.assertEqual(row['Q0'],'ANSWER'); self.a.vision.ask.assert_called_once()
        np.testing.assert_allclose(row['_alignment_matrices'][1],[[1,0,0],[0,1,0]])
    def test_added_ink_does_not_reduce_fixed_coverage(self):
        ref=np.full((300,400,3),255,np.uint8)
        cv2.line(ref,(20,100),(380,100),(0,0,0),2)
        src=ref.copy()
        cv2.rectangle(src,(20,103),(380,106),(0,0,0),-1)
        _,_,_,score=register(ref,src,np.full(ref.shape[:2],255,np.uint8))
        self.assertEqual(score,1.0)
    def test_review_not_graded(self):
        g=Grader.__new__(Grader);g.plan={'warnings':[]}
        r=g.grade({'status':'REVIEW','reason':'位置合わせ不能'})
        self.assertIsNone(r['AI点数']);self.assertEqual(r['AI採点状態'],'REVIEW')
    def test_sparse_rectangles_rejected(self):
        boxes=np.array([[10,10,100,50],[20,200,100,300]])
        self.assertIsNone(rectangle_transform(boxes,boxes,(800,600,3)))
