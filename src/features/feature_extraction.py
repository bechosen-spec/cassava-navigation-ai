"""Shared native-resolution geometry and image descriptors for GT/predicted polygons.
Coordinates use pixel centres; x increases right, y down. Empty geometry is NaN.
Union areas avoid double-counting overlapping instances. Original data is read-only.
"""
from pathlib import Path
import hashlib
import cv2
import numpy as np
import pandas as pd
import yaml
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import shannon_entropy

ROOT = Path(__file__).resolve().parents[2]
NAMES = ['path', 'cassava_leaves', 'ridge']

def config():
    return yaml.safe_load((ROOT/'configs/features.yaml').read_text())

def manifest():
    base = ROOT/config()['dataset_root']
    rows = []
    for split in ['train','valid','test']:
        for p in sorted((base/'images'/split).rglob('*')):
            if p.suffix.lower() not in {'.jpg','.jpeg','.png'} or any(x.startswith(('.', '__MACOSX')) for x in p.relative_to(base).parts):
                continue
            label = base/'labels'/split/(p.stem+'.txt')
            if not label.exists():
                raise ValueError(f'Missing annotation: {p}')
            rows.append(dict(image_id=f'{split}/{p.stem}',split=split,image_path=str(p.relative_to(ROOT)),label_path=str(label.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    return pd.DataFrame(rows)

def polygons(label):
    result = {i:[] for i in range(3)}
    for line in Path(label).read_text().splitlines():
        v = np.array([float(x) for x in line.split()])
        if len(v)<7 or (len(v)-1)%2 or not np.isfinite(v).all() or v[0] not in result or (v[1:]<0).any() or (v[1:]>1).any():
            raise ValueError(f'Technically invalid polygon in {label}')
        p = v[1:].reshape(-1,2)
        if cv2.contourArea(p.astype(np.float32)) <= 0:
            raise ValueError(f'Zero-area polygon in {label}')
        result[int(v[0])].append(p)
    return result

def masks_from_polygons(polys,h,w):
    masks = []
    for c in range(3):
        m = np.zeros((h,w),np.uint8)
        for p in polys[c]:
            q = np.rint(np.asarray(p)*[w-1,h-1]).astype(np.int32)
            # Fill independently: OpenCV's collective even/odd fill can erase overlaps.
            cv2.fillPoly(m,[q],1)
        masks.append(m)
    return masks

def extract(image, polys):
    cfg=config(); h,w=image.shape[:2]; masks=masks_from_polygons(polys,h,w); f={}
    def put(name,value): f[name]=float(value)
    for c,name in enumerate(NAMES):
        m=masks[c]; yy,xx=np.nonzero(m); points=[np.asarray(p)*[w-1,h-1] for p in polys[c]]
        areas=[cv2.contourArea(p.astype(np.float32)) for p in points]
        values=dict(polygon_count=len(points),mask_area=m.sum(),area_ratio=m.mean(),largest_polygon_area=max(areas) if areas else np.nan,mean_polygon_area=np.mean(areas) if areas else np.nan,centroid_x=xx.mean() if len(xx) else np.nan,centroid_y=yy.mean() if len(yy) else np.nan,bounding_width=np.ptp(xx)+1 if len(xx) else np.nan,bounding_height=np.ptp(yy)+1 if len(yy) else np.nan,perimeter=sum(cv2.arcLength(p.astype(np.float32),True) for p in points) if points else np.nan,leftmost=xx.min() if len(xx) else np.nan,rightmost=xx.max() if len(xx) else np.nan,topmost=yy.min() if len(yy) else np.nan,bottommost=yy.max() if len(yy) else np.nan)
        values['aspect_ratio']=values['bounding_width']/values['bounding_height']
        for k,v in values.items(): put(f'{name}_{k}',v)
    path,leaf,ridge=masks; start=int(h*cfg['lower_roi_start']); roi=path[start:]
    left=[];right=[]; widths=[]
    for row in roi:
        x=np.flatnonzero(row)
        if len(x): left.append(x.min());right.append(x.max());widths.append(x.max()-x.min()+1)
    put('path_left_boundary',np.median(left) if left else np.nan)
    put('path_right_boundary',np.median(right) if right else np.nan)
    put('path_center_x',(f['path_left_boundary']+f['path_right_boundary'])/2)
    put('path_width_lower',np.median(widths)/w if widths else np.nan)
    put('normalized_path_offset',(f['path_center_x']-w/2)/(w/2))
    put('absolute_path_deviation',abs(f['normalized_path_offset']))
    put('path_continuity',np.any(roi,axis=1).mean())
    put('missing_path',not path.any())
    for y in cfg['width_rows']:
        x=np.flatnonzero(path[min(h-1,int(y*h))]);put(f'path_width_y{y}',(x.max()-x.min()+1)/w if len(x) else np.nan)
    for name,m in [('cassava',leaf),('ridge',ridge)]:
        put(name+'_coverage',m.mean());put(name+'_left_coverage',m[:,:w//2].mean());put(name+'_right_coverage',m[:,w//2:].mean())
        put(name+'_imbalance',f[name+'_left_coverage']-f[name+'_right_coverage'])
    put('cassava_lower_obstruction',leaf[start:].mean());put('cassava_density',len(polys[1])/(h*w/1e6))
    y,x=np.nonzero(ridge)
    orientation=np.nan
    if len(x)>2:
        vals,vecs=np.linalg.eigh(np.cov(np.vstack([x,y])))
        if vals[-1]>1.05*vals[0]: orientation=np.degrees(np.arctan2(vecs[1,-1],vecs[0,-1]))%180
    put('ridge_orientation',orientation)
    put('ridge_proximity_to_path',cv2.distanceTransform(1-path,cv2.DIST_L2,5)[ridge.astype(bool)].mean()/np.hypot(w,h) if path.any() and ridge.any() else np.nan)
    put('ridge_path_overlap',(ridge & path).sum()/ridge.sum() if ridge.any() else np.nan)
    put('ridge_path_centroid_distance',np.hypot(f['ridge_centroid_x']-f['path_centroid_x'],f['ridge_centroid_y']-f['path_centroid_y'])/np.hypot(w,h))
    rgb=image.astype(float)/255; hsv=cv2.cvtColor(image,cv2.COLOR_RGB2HSV).astype(float)/[179,255,255]
    for space,a in [('rgb',rgb),('hsv',hsv)]:
        for j in range(3):
            put(f'{space}_{j}_mean',a[:,:,j].mean());put(f'{space}_{j}_std',a[:,:,j].std())
            hist=np.histogram(a[:,:,j],bins=cfg['histogram_bins'],range=(0,1))[0]/(h*w)
            for b,v in enumerate(hist):put(f'{space}_{j}_hist_{b}',v)
    r,g,b=rgb.transpose(2,0,1)
    put('excess_green',(2*g-r-b).mean());put('excess_red',(1.4*r-g).mean());put('normalized_green_red',np.mean((g-r)/(g+r+1e-8)))
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY); small=cv2.resize(gray,(cfg['texture_size'],)*2,interpolation=cv2.INTER_AREA)
    quant=(small.astype(float)*cfg['glcm_levels']/256).astype(np.uint8)
    glcm=graycomatrix(quant,[1],[0,np.pi/4,np.pi/2,3*np.pi/4],levels=cfg['glcm_levels'],symmetric=True,normed=True)
    for prop in ['contrast','correlation','energy','homogeneity']:put('glcm_'+prop,graycoprops(glcm,prop).mean())
    put('texture_entropy',shannon_entropy(small))
    lbp=local_binary_pattern(small,8,1,method='uniform')
    for i,v in enumerate(np.histogram(lbp,bins=np.arange(11))[0]/lbp.size):put(f'lbp_hist_{i}',v)
    put('lbp_mean',lbp.mean());put('lbp_std',lbp.std())
    edges=cv2.Canny(gray,*cfg['canny_thresholds']);put('edge_density',(edges>0).mean())
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    put('edge_contour_count',len(contours));put('edge_contour_mean_area',np.mean([cv2.contourArea(c) for c in contours]) if contours else np.nan)
    put('edge_contour_mean_perimeter',np.mean([cv2.arcLength(c,True) for c in contours]) if contours else np.nan)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=40,minLineLength=30,maxLineGap=10)
    angle=np.nan
    if lines is not None:
        lines=lines.reshape(-1,4);d=lines[:,2:]-lines[:,:2];angles=np.arctan2(d[:,1],d[:,0])%np.pi
        hist,bins=np.histogram(angles,bins=18,range=(0,np.pi),weights=np.linalg.norm(d,axis=1));angle=np.degrees((bins[hist.argmax()]+bins[hist.argmax()+1])/2)
    put('edge_dominant_orientation',angle)
    f.update({k+'_missing':int(not np.isfinite(v)) for k,v in list(f.items())})
    return f

def run(predicted=None):
    rows=[]; mf=manifest(); (ROOT/'data/processed').mkdir(exist_ok=True)
    for i,row in enumerate(mf.to_dict('records')):
        image=cv2.cvtColor(cv2.imread(str(ROOT/row['image_path'])),cv2.COLOR_BGR2RGB)
        ps=polygons(ROOT/row['label_path']) if predicted is None else predicted[row['image_id']]
        rows.append(dict(image_id=row['image_id'],split=row['split'],image_width=image.shape[1],image_height=image.shape[0],**extract(image,ps)))
        if i%100==0: print(f'Features: {i}/{len(mf)}',flush=True)
    df=pd.DataFrame(rows); name='features_ground_truth.csv' if predicted is None else 'features_predicted_masks.csv'
    df.to_csv(ROOT/'data/processed'/name,index=False);mf.to_csv(ROOT/'data/processed/image_manifest.csv',index=False)
    return df

if __name__=='__main__': run()
