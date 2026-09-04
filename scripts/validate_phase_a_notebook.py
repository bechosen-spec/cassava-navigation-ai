"""Validate notebook syntax and embedded helpers using temporary synthetic fixtures; no training."""
import ast, tempfile, sys, zipfile, json, importlib, os
from pathlib import Path
import nbformat
import numpy as np
from PIL import Image
import pandas as pd
import yaml
notebook=Path(__file__).resolve().parents[1]/'notebooks/03_phase_A_colab.ipynb'
nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb)
helper_sources={}
for i,cell in enumerate(nb.cells):
    if cell.cell_type!='code':continue
    compile(cell.source,f'cell_{i}','exec')
    assert '/Users/' not in cell.source
    tree=ast.parse(cell.source)
    if cell.source.startswith('# Reusable helper:'):
        path=cell.source.splitlines()[0].split(': ',1)[1]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='write_text':
                value=node.args[0].value;compile(value,path,'exec');helper_sources[path]=value
assert len(helper_sources)==5
base=Path(tempfile.mkdtemp(prefix='phase_a_notebook_check_'))
project=base/'project';project.mkdir()
os.environ['MPLCONFIGDIR']=str(base/'matplotlib-cache')
for path,source in helper_sources.items():
    dest=project/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(source)
sys.path.insert(0,str(project/'src'))
from phase_a_io import safe_extract, discover_root, validate_dataset
unsafe=base/'unsafe.zip'
with zipfile.ZipFile(unsafe,'w') as z:z.writestr('../escape.txt','bad')
try:safe_extract(unsafe,base/'rejected',100000)
except ValueError:pass
else:raise AssertionError('ZIP traversal was not rejected')
assert not (base/'escape.txt').exists()
# Build a fixture with unique images and the supported val alias.
fixture=base/'fixture'
rng=np.random.default_rng(42)
for split in ['train','val','test']:
    for i in range(2):
        ip=fixture/'data/images'/split/f'image_{i}.png';ip.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(rng.integers(0,255,(64,64,3),dtype=np.uint8)).save(ip)
        label=fixture/'data/labels'/split/f'image_{i}.txt';label.parent.mkdir(parents=True,exist_ok=True)
        # A valid very-large path and very-small leaf are retained; invalid class excluded.
        label.write_text('0 0.0 0.0 0.8 0.0 0.8 1.0 0.0 1.0\n1 0.1 0.1 0.1001 0.1 0.1001 0.1001 0.1 0.1001\n9 0 0 1 0 1 1\n')
archive=base/'arbitrary_name.zip'
with zipfile.ZipFile(archive,'w') as z:
    for p in fixture.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(fixture))
    z.writestr('data/images/train/.ipynb_checkpoints/duplicate.jpg',b'ignored')
info=safe_extract(archive,project/'data/extracted',10000000)
assert info['ignored_archive_members']==1
root=discover_root(project/'data/extracted');mf=validate_dataset(root,project,info)
assert mf.groupby('split').size().to_dict()=={'train':2,'valid':2,'test':2}
quality=pd.read_csv(project/'reports/phaseA_pretraining_quality_check.csv')
assert (quality.issue=='extremely_small_polygon').sum()==6
assert (quality.issue=='extremely_large_polygon').sum()==6
assert (quality.issue=='invalid_class_id').sum()==6
assert len((root/'labels/train/image_0.txt').read_text().splitlines())==3
assert len((project/mf.iloc[0].label_path).read_text().splitlines())==2
cfg=dict(seed=42,dataset_root=str(root),lower_roi_start=.65,width_rows=[.5,.65,.8,.95],texture_size=64,glcm_levels=32,histogram_bins=16,canny_thresholds=[100,200])
(project/'configs/features.yaml').write_text(yaml.safe_dump(cfg))
from features.feature_extraction import extract,run,masks_from_polygons
features=run();assert len(features)==6
assert {'original_image_path','extraction_status','warning_flags','path_width'}<=set(features)
empty=extract(np.full((64,64,3),128,np.uint8),{0:[],1:[],2:[]})
assert np.isnan(empty['path_center_x']) and empty['path_center_x_missing']==1
p=np.array([[.1,.1],[.9,.1],[.9,.9],[.1,.9]])
np.testing.assert_array_equal(masks_from_polygons({0:[p],1:[],2:[]},64,64)[0],masks_from_polygons({0:[p,p],1:[],2:[]},64,64)[0])
from navigation.target_generation import generate
cfg=dict(smoothing={'enabled':False},minimum_path_area_ratio=.02,minimum_valid_path_width=.1,minimum_path_continuity=.6,stop_uncertain_threshold=.6,left_threshold=-.15,right_threshold=.15)
t=generate(pd.DataFrame([dict(image_id='empty',split='train',image_width=64,**empty)]),cfg)
assert t.iloc[0].discrete_target=='stop_or_uncertain' and np.isnan(t.iloc[0].continuous_target)
from phase_a_segmentation import metric_values
class FakeMetrics:
    results_dict={'metrics/precision(M)':.75,'metrics/recall(M)':.5,'metrics/mAP50(M)':.6}
values=metric_values(FakeMetrics(),'mask')
assert values['mask_precision']==.75 and np.isnan(values['mask_map50_95'])
print(json.dumps(dict(status='passed',cells=len(nb.cells),code_cells=sum(c.cell_type=='code' for c in nb.cells),embedded_modules=len(helper_sources),checks=['nbformat schema','all Python cells compile','all embedded modules compile','no local Mac paths','ZIP traversal rejected','arbitrary ZIP filename','val alias','hidden artifacts ignored','size warnings retained','technical row excluded only in derived copy','original labels unchanged','feature metadata retained','union overlap safe','missing path is uncertain/NaN','metric API fallback preserves missingness'],gpu_training_executed=False),indent=2))
