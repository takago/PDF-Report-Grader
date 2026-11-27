"""Document geometry for page alignment."""
import cv2
import numpy as np
import pymupdf as pdf


def pixels(rect, page, shape):
    r=pdf.Rect(rect)*page.rotation_matrix
    h,w=shape[:2]
    return np.array([r.x0*w/page.rect.width,r.y0*h/page.rect.height,
                     r.x1*w/page.rect.width,r.y1*h/page.rect.height])


def unique_boxes(boxes, shape):
    h,w=shape[:2]; result=[]
    for b in boxes:
        b=np.asarray(b,dtype=float); bw,bh=b[2:]-b[:2]
        if bw<max(12,w*.025) or bh<max(9,h*.008): continue
        if bw>w*.98 or bh>h*.98 or np.any(b[:2]<-2) or b[2]>w+2 or b[3]>h+2: continue
        if any(np.max(np.abs(b-a))<max(4,w*.003) for a in result): continue
        result.append(b)
    # Prefer large, widely distributed frames, not small glyph boxes.
    return np.array(sorted(result,key=lambda b: -np.prod(b[2:]-b[:2]))[:80],dtype=float).reshape(-1,4)


def raster_boxes(image):
    h,w=image.shape[:2]
    g=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    ink=(g<210).astype(np.uint8)*255
    horizontal=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((1,max(12,round(w*.022))),np.uint8))
    vertical=cv2.morphologyEx(ink,cv2.MORPH_OPEN,np.ones((max(10,round(h*.008)),1),np.uint8))
    # Bridge only small gaps at corners; do not connect distant text blocks.
    lines=cv2.morphologyEx(horizontal|vertical,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(lines,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    boxes=[]
    for c in contours:
        x,y,bw,bh=cv2.boundingRect(c)
        if bw<w*.025 or bh<h*.008: continue
        x1,y1=x+bw-1,y+bh-1
        t=max(3,round(w*.003))
        def hcover(v):
            return np.mean(np.any(horizontal[max(0,v-t):min(h,v+t+1),x:x1+1]>0,axis=0))
        def vcover(v):
            return np.mean(np.any(vertical[y:y1+1,max(0,v-t):min(w,v+t+1)]>0,axis=1))
        if min(hcover(y),hcover(y1),vcover(x),vcover(x1))>=.72:
            boxes.append([x,y,x1,y1])
    return unique_boxes(boxes,image.shape)


def page_boxes(page,image,template=False):
    boxes=[]
    if template:
        for widget in page.widgets() or []:
            if widget.border_width and widget.border_color is not None:
                boxes.append(pixels(widget.rect,page,image.shape))
    # PDF printers often retain vector rectangles even when widgets disappear.
    try:
        paths=page.get_drawings()
    except (ValueError, RuntimeError):
        paths=[]
    for path in paths:
        if path.get('type') not in ('s','fs') or path.get('color') is None: continue
        for item in path['items']:
            if item[0]=='re': boxes.append(pixels(item[1],page,image.shape))
    boxes.extend(raster_boxes(image))
    return unique_boxes(boxes,image.shape)


def rectangle_transform(ref_boxes,src_boxes,shape):
    """Robust consensus of independent x/y scale and translation hypotheses.

    Requires >=3 one-to-one matched frames, spread in both dimensions, and
    sub-frame residuals. Rotation/shear or sparse frames fall back to SIFT.
    """
    h,w=shape[:2]
    if len(ref_boxes)<3 or len(src_boxes)<3: return None
    r=np.asarray(ref_boxes); s=np.asarray(src_boxes)
    rc=(r[:,:2]+r[:,2:])/2; sc=(s[:,:2]+s[:,2:])/2
    sizes_r=r[:,2:]-r[:,:2]; sizes_s=s[:,2:]-s[:,:2]
    initial=np.array([max(7,w*.008),max(7,h*.008)]*2)
    final=np.array([max(4,w*.0025),max(4,h*.0025)]*2)
    def matches(scale,offset,tol):
        transformed=s*np.tile(scale,2)+np.tile(offset,2)
        d=np.max(np.abs(r[:,None,:]-transformed[None,:,:])/tol,axis=2)
        pairs=[]; used_r=set(); used_s=set()
        for k in np.argsort(d,axis=None):
            i,j=np.unravel_index(k,d.shape)
            if d[i,j]>1: break
            if i in used_r or j in used_s: continue
            used_r.add(i); used_s.add(j); pairs.append((i,j))
        return pairs
    def fit(pairs):
        a=np.array([s[j] for i,j in pairs]); b=np.array([r[i] for i,j in pairs]); scales=[]; offsets=[]
        for k in range(2):
            x=a[:,[k,k+2]].ravel(); y=b[:,[k,k+2]].ravel()
            slope,shift=np.linalg.lstsq(np.column_stack((x,np.ones(len(x)))),y,rcond=None)[0]
            scales.append(slope); offsets.append(shift)
        return np.array(scales),np.array(offsets)
    def plausible(scale,offset):
        return np.all((scale>=.65)&(scale<=1.5)) and np.all(np.abs(offset)<[w*.3,h*.3])
    # Deduplicate similar hypotheses before evaluation.
    hypotheses={(100,100,0,0):(np.ones(2),np.zeros(2))}
    for i in range(len(r)):
        for j in range(len(s)):
            scale=sizes_r[i]/sizes_s[j]; offset=rc[i]-sc[j]*scale
            if plausible(scale,offset):
                key=tuple(np.round(np.r_[scale*100,offset/[w,h]*100]).astype(int))
                hypotheses.setdefault(key,(scale,offset))
    candidates=[]
    for scale,offset in hypotheses.values():
        pairs=matches(scale,offset,initial)
        if len(pairs)<3: continue
        scale,offset=fit(pairs)
        if not plausible(scale,offset): continue
        pairs=matches(scale,offset,final)
        if len(pairs)<max(3,int(np.ceil(len(r)*.45))): continue
        scale,offset=fit(pairs)
        if not plausible(scale,offset): continue
        pairs=matches(scale,offset,final)
        if len(pairs)<max(3,int(np.ceil(len(r)*.45))): continue
        rb=np.array([r[i] for i,j in pairs]); sb=np.array([s[j] for i,j in pairs])
        # Do not accept a set of adjacent tiny boxes as page-wide evidence.
        if np.ptp(rb[:,[0,2]])<w*.25 or np.ptp(rb[:,[1,3]])<h*.25: continue
        residual=float(np.mean(np.abs(rb-(sb*np.tile(scale,2)+np.tile(offset,2)))))
        candidates.append((len(pairs),residual,scale,offset))
    if not candidates: return None
    candidates.sort(key=lambda c:(-c[0],c[1]))
    count,err,scale,offset=candidates[0]
    # Equal-consensus but substantially different transforms are ambiguous.
    for n,e,s2,o2 in candidates[1:]:
        if n<count or e>err+1.5: break
        if np.max(np.abs(np.r_[offset-o2,(scale-s2)*[w,h]]))>max(10,w*.008): return None
    return np.array([[scale[0],0,offset[0]],[0,scale[1],offset[1]]]),count,err


