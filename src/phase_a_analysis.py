"""Reproducible quality analysis; selection statistics use training rows only."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import yaml
from features.feature_extraction import ROOT,manifest,polygons,masks_from_polygons
from navigation.target_generation import run as targets
plt.rcParams.update({'figure.dpi':130,'savefig.dpi':220,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def save(fig,path):
    fig.tight_layout();fig.savefig(path);plt.close(fig)

def run():
    f=pd.read_csv(ROOT/'data/processed/features_ground_truth.csv'); t=targets();mf=manifest()
    out=ROOT/'reports';figdir=ROOT/'figures/navigation_targets'; qdir=ROOT/'figures/feature_quality';qdir.mkdir(exist_ok=True)
    numeric=f.drop(columns=['image_id','split','image_width','image_height']);train=numeric[f.split=='train']
    missing=pd.DataFrame({'feature':numeric.columns,'missing_count':numeric.isna().sum().values,'missing_rate':numeric.isna().mean().values})
    missing.to_csv(out/'feature_missing_values.csv',index=False)
    summary=pd.DataFrame({'feature':numeric.columns,'train_unique':train.nunique().values,'train_dominant_fraction':[train[c].value_counts(dropna=False,normalize=True).iloc[0] for c in train]})
    summary['constant']=summary.train_unique<=1;summary['near_constant']=summary.train_dominant_fraction>=.99
    summary.to_csv(out/'feature_extraction_summary.csv',index=False)
    summary[summary.constant].to_csv(out/'constant_features.csv',index=False);summary[summary.near_constant & ~summary.constant].to_csv(out/'near_constant_features.csv',index=False)
    corr=train.corr();corr.to_csv(out/'feature_correlations_train.csv')
    pairs=[(a,b,corr.loc[a,b]) for i,a in enumerate(corr) for b in corr.columns[i+1:] if abs(corr.loc[a,b])>=.95]
    pd.DataFrame(pairs,columns=['feature_a','feature_b','pearson_r']).to_csv(out/'high_correlation_pairs.csv',index=False)
    # Exclude all path geometry: direct target construction inputs/proxies.
    # Also exclude ridge-to-path relations and position features that recreate centre.
    forbidden=lambda c: c.startswith(('path_','normalized_path','absolute_path','missing_path','ridge_path','ridge_proximity'))
    broad=[c for c in train if not forbidden(c) and not c.endswith('_missing') and train[c].nunique()>1 and train[c].isna().mean()<.2]
    selected=[]
    for c in broad:
        if all(not np.isfinite(corr.loc[c,d]) or abs(corr.loc[c,d])<.95 for d in selected):selected.append(c)
    anfis=[c for c in ['cassava_imbalance','ridge_imbalance','ridge_orientation','cassava_lower_obstruction','edge_density'] if c in selected]
    (ROOT/'configs/recommended_features.yaml').write_text(yaml.safe_dump(dict(selection_split='train',classical_ml=selected,anfis=anfis,excluded='All path-derived geometry; direct formula leakage. GT-mask features remain oracle-only until predicted masks are available.')))
    dictionary=[]
    for c in numeric:
        category='colour' if c.startswith(('rgb','hsv','excess','normalized_green')) else 'texture' if c.startswith(('glcm','lbp','texture')) else 'edge' if c.startswith('edge') else 'navigation_geometry' if c.startswith(('cassava_','ridge_path','ridge_proximity','normalized_path','absolute_path','missing_path')) or c in ['path_center_x','path_width_lower','path_continuity','path_left_boundary','path_right_boundary'] or c.startswith('path_width_y') else 'segmentation_geometry'
        unit='dimensionless; see formula'
        formula='Defined by extract() in src/features/feature_extraction.py: '+c
        if c.endswith('_missing'): formula=f'1 if {c[:-8]} is NaN/nonfinite, otherwise 0';unit='0 or 1'
        elif '_hist_' in c: formula='bin count / number of pixels; equal-width bins';unit='[0,1]'
        elif c.endswith('area_ratio') or 'coverage' in c or c=='edge_density':formula='foreground pixel count / region pixel count';unit='[0,1]'
        elif c.endswith('centroid_x'):formula='mean x coordinate of foreground pixel centres';unit='pixels'
        elif c.endswith('centroid_y'):formula='mean y coordinate of foreground pixel centres';unit='pixels'
        elif c.endswith('polygon_count'):formula='number of valid annotated instances';unit='count'
        elif c.endswith('mask_area'):formula='count of union foreground pixels';unit='pixels squared'
        elif c.endswith('largest_polygon_area'):formula='max absolute shoelace area across polygons';unit='pixels squared'
        elif c.endswith('mean_polygon_area'):formula='mean absolute shoelace area across polygons';unit='pixels squared'
        elif c.endswith('perimeter') and not c.startswith('edge'):formula='sum of closed polygon Euclidean edge lengths';unit='pixels'
        elif c.endswith('bounding_width'):formula='max(x)-min(x)+1 over union mask';unit='pixels'
        elif c.endswith('bounding_height'):formula='max(y)-min(y)+1 over union mask';unit='pixels'
        elif c.endswith('aspect_ratio'):formula='bounding_width / bounding_height'
        elif c.endswith(('leftmost','rightmost','topmost','bottommost')):formula='min x, max x, min y or max y of union mask as named';unit='pixels'
        elif c.startswith('path_width'):formula='outer right-left+1 divided by image width; lower ROI uses median row span';unit='[0,1]'
        elif c=='path_center_x':formula='(median lower-ROI left boundary + median lower-ROI right boundary)/2';unit='pixels'
        elif c in ['path_left_boundary','path_right_boundary']:formula='median outer boundary coordinate across occupied lower-ROI rows';unit='pixels'
        elif c=='normalized_path_offset':formula='(path_center_x - width/2)/(width/2)';unit='[-1,1]'
        elif c=='absolute_path_deviation':formula='abs(normalized_path_offset)';unit='[0,1]'
        elif c=='path_continuity':formula='fraction lower-ROI rows with >=1 path pixel';unit='[0,1]'
        elif c=='missing_path':formula='1 if no path pixel else 0';unit='0 or 1'
        elif c.endswith('imbalance'):formula='left-half coverage minus right-half coverage';unit='[-1,1]'
        elif c=='cassava_lower_obstruction':formula='leaf union pixel count in lower ROI / ROI area';unit='[0,1]'
        elif c=='cassava_density':formula='leaf instance count / image megapixels';unit='instances/Mpixel'
        elif c=='ridge_orientation':formula='principal eigenvector angle of ridge pixel covariance; NaN if eigenvalue ratio <=1.05';unit='degrees [0,180)'
        elif c=='ridge_proximity_to_path':formula='mean ridge-pixel distance to nearest path pixel / image diagonal';unit='[0,1]'
        elif c=='ridge_path_overlap':formula='ridge and path intersection area / ridge area';unit='[0,1]'
        elif c=='ridge_path_centroid_distance':formula='Euclidean centroid distance / image diagonal';unit='[0,1]'
        elif c.startswith(('rgb','hsv')):formula='mean or population std of channel normalized to [0,1]; HSV hue is linear, not circular';unit='[0,1]'
        elif c=='excess_green':formula='mean(2G-R-B), RGB in [0,1]';unit='[-2,2]'
        elif c=='excess_red':formula='mean(1.4R-G), RGB in [0,1]';unit='[-1,1.4]'
        elif c=='normalized_green_red':formula='mean((G-R)/(G+R+1e-8))';unit='[-1,1]'
        elif c.startswith('glcm'):formula='skimage graycoprops '+c[5:]+'; symmetric normalized 32-level GLCM, distance 1, mean over 0/45/90/135 degrees'
        elif c=='texture_entropy':formula='-sum(p*log2(p)) of grayscale histogram';unit='bits [0,8]'
        elif c.startswith('lbp'):formula='mean/std of uniform LBP(P=8,R=1), resized 256x256 grayscale';unit='[0,9]'
        elif c=='edge_contour_count':formula='number of Canny contours (RETR_LIST)';unit='count'
        elif c=='edge_contour_mean_area':formula='mean cv2.contourArea of Canny contours';unit='pixels squared'
        elif c=='edge_contour_mean_perimeter':formula='mean closed Canny contour perimeter';unit='pixels'
        elif c=='edge_dominant_orientation':formula='centre of maximum length-weighted Hough-line angle bin (18 bins)';unit='degrees [0,180)'
        dictionary.append(dict(feature_name=c,category=category,meaning=c.replace('_',' '),formula=formula,source='image' if category in ['colour','texture','edge'] else 'ground-truth polygons (oracle); same implementation for predictions',unit_range=unit,suitable_for_classical_ml='yes' if c in selected else 'no',suitable_for_ANFIS='yes' if c in anfis else 'no'))
    pd.DataFrame(dictionary).to_csv(ROOT/'data/processed/feature_dictionary.csv',index=False)
    counts=t.discrete_target.value_counts().reindex(['left','forward','right','stop_or_uncertain'],fill_value=0)
    counts.rename_axis('target').to_csv(out/'navigation_target_counts.csv')
    trainstats=f[f.split=='train'][['normalized_path_offset','path_width_lower','path_area_ratio','path_continuity']].describe(percentiles=[.05,.1,.25,.5,.75,.9,.95]);trainstats.to_csv(out/'target_threshold_training_distributions.csv')
    for col,label in [('normalized_offset','Continuous offset'),('path_width','Path width / image width'),('path_area_ratio','Path area ratio')]:
        fig,ax=plt.subplots(figsize=(7,4));ax.hist(t[col].dropna(),bins=30,color='#287c72');ax.set(xlabel=label,ylabel='Images');save(fig,figdir/f'{col}_distribution.png')
    fig,ax=plt.subplots(figsize=(7,4));counts.plot.bar(ax=ax,color='#287c72',rot=15);ax.set(ylabel='Images',xlabel='Provisional target');save(fig,figdir/'target_class_distribution.png')
    fig,ax=plt.subplots(figsize=(7,4));ax.scatter(t.path_width,t.normalized_offset,s=8,alpha=.4);ax.set(xlabel='Path width / image width',ylabel='Normalized offset');save(fig,figdir/'offset_vs_width.png')
    chosen=t.groupby('discrete_target',group_keys=False).sample(n=1,random_state=42).index.tolist()
    chosen+=t.drop(index=chosen).sample(n=30-len(chosen),random_state=42).index.tolist()
    for n,i in enumerate(chosen):
        row=t.iloc[i];m=mf.set_index('image_id').loc[row.image_id];im=cv2.cvtColor(cv2.imread(str(ROOT/m.image_path)),cv2.COLOR_BGR2RGB);h,w=im.shape[:2];mask=masks_from_polygons(polygons(ROOT/m.label_path),h,w)[0]
        fig,ax=plt.subplots(figsize=(7,7));ax.imshow(im);overlay=np.zeros((h,w,4));overlay[mask.astype(bool)]=[0,.9,.6,.35];ax.imshow(overlay);ax.axvline(w/2,color='white',linestyle='--',label='Image centre')
        fr=f.iloc[i]
        for key,color,label in [('path_center_x','yellow','Path centre'),('path_left_boundary','cyan','Left boundary'),('path_right_boundary','magenta','Right boundary')]:
            if np.isfinite(fr[key]):ax.vlines(fr[key],.65*h,h,color=color,label=label)
        ax.axhline(.65*h,color='white',linestyle=':');ax.set_title(f'{row.discrete_target} | offset={row.normalized_offset:.3f}\n{row.image_id[:45]}');ax.legend(loc='upper right',fontsize=7);ax.axis('off');save(fig,figdir/f'diagnostic_{n+1:02d}.png')
    diagnostic=t.loc[chosen,['image_id','split','discrete_target']].copy();diagnostic['figure']=[f'diagnostic_{n+1:02d}.png' for n in range(30)];diagnostic.to_csv(out/'navigation_diagnostic_manifest.csv',index=False)
    display=['path_width_lower','path_area_ratio','cassava_imbalance','ridge_orientation','edge_density','normalized_path_offset']
    for group,labels in [('split',f.split),('target',t.discrete_target)]:
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for ax,c in zip(axes.flat,display):
            groups=list(labels.unique());ax.boxplot([f.loc[labels==g,c].dropna() for g in groups],tick_labels=groups);ax.set_title(c);ax.tick_params(axis='x',rotation=20)
        save(fig,qdir/f'distributions_by_{group}.png')
    f.assign(target=t.discrete_target).groupby('split')[list(numeric)].agg(['mean','std','median']).to_csv(out/'feature_distributions_by_split.csv')
    f.assign(target=t.discrete_target).groupby('target')[list(numeric)].agg(['mean','std','median']).to_csv(out/'feature_distributions_by_target.csv')
    fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(corr.loc[display,display],vmin=-1,vmax=1,cmap='coolwarm');ax.set_xticks(range(6),display,rotation=60,ha='right');ax.set_yticks(range(6),display);fig.colorbar(im,ax=ax);save(fig,qdir/'correlations_train.png')
    hashes=mf.groupby('sha256').split.nunique(); cross=int((hashes>1).sum())
    pd.DataFrame([dict(check='exact_cross_split_hashes',count=cross),dict(check='duplicate_image_ids',count=mf.image_id.duplicated().sum()),dict(check='feature_target_id_mismatches',count=(f.image_id!=t.image_id).sum())]).to_csv(out/'leakage_checks.csv',index=False)
    assert cross==0 and not mf.image_id.duplicated().any() and f.image_id.equals(t.image_id)
    assert mf.split.value_counts().to_dict()=={'train':602,'valid':120,'test':80}
    rate=numeric.isna().mean().mean(); rawrate=numeric[[c for c in numeric if not c.endswith('_missing')]].isna().mean().mean()
    (out/'feature_quality_report.md').write_text(f'''# Phase A feature quality\n\nProcessed {len(f)} images: 602 train, 120 valid, 80 test. Original nominal 603 train includes one excluded checkpoint duplicate; no split assignments changed.\n\n{len(numeric.columns)} features including explicit missing flags. Missing rate {rate:.4%}; excluding missing flags {rawrate:.4%}. Missing geometry remains NaN; valid zero coverage/counts remain zero. Train-derived constant/near-constant reports and correlations are separate CSVs. Near-constant means dominant value >=99%.\n\nClassical ML: {len(selected)} recommended features, listed in configs/recommended_features.yaml. Greedy correlation pruning uses train only at |r| >=0.95; no target outcomes or test tuning. ANFIS: {', '.join(anfis)}. Missing-value imputation/scaling must be fitted on train only in Phase B.\n\nAll path-derived features (including offset, centroid, boundaries, widths, area and continuity) are excluded from primary recommended inputs because targets are constructed from them. Using offset to predict offset is identity leakage; using width/continuity to predict stop reproduces the rule. Such geometry can only be used in explicitly labelled rule-reconstruction experiments. Ridge-to-path relations are excluded too.\n\nGT-mask features are oracle inputs unavailable to the robot. Predicted-mask features and honest out-of-fold train predictions are required for end-to-end evaluation. A segmentation model trained on all train images produces in-sample train masks, not out-of-fold features.\n\nExact cross-split image duplicates: {cross}. This does not rule out adjacent video frames or shared field sessions; grouping metadata is unavailable. Preserve existing splits but investigate temporal/site leakage before performance claims. Feature distributions by split/target are descriptive only.\n\nRaster masks use independent union fills at native resolution. Polygon areas/perimeters are continuous shoelace/edge calculations; very small polygons may rasterize to a pixel and are retained. Row widths are outer spans and can bridge disjoint islands. Continuity measures row occupancy, not physical connectivity or traversability. Hue statistics are linear. Texture uses 256-pixel resampling.\n''')
    (out/'navigation_target_report.md').write_text(f'''# Provisional navigation targets\n\nLabels are geometry-derived, not manually recorded robot controls. Continuous offset is the primary research target: (path_center_x - width/2)/(width/2). Negative means image-left, positive image-right; this is not a calibrated wheel/steering sign. Centre is the midpoint of median outer boundaries over occupied rows in the lower 35% of the image.\n\n{counts.to_string()}\n\nContinuous target is available for {t.continuous_target.notna().sum()} of {len(t)} images. Missing centres remain NaN. Low-quality finite offsets remain available but warning flags must be used to define a training/evaluation policy.\n\nThresholds in configs/navigation_targets.yaml are provisional. A +/-0.15 deadband is an interpretable image-centre tolerance; area >=0.02, width >=0.10 and occupancy >=0.60 reject weak geometry. Training-only distributions are in target_threshold_training_distributions.csv; thresholds are reviewed against these, not optimized on test outcomes. Confidence is min(clipped area/min-area, clipped width/min-width, continuity); it is a heuristic, not a calibrated probability. Stop/uncertain combines threshold failures, missing centre, and confidence below 0.60. Smoothing disabled: no validated temporal order.\n\nThirty reproducible diagnostic overlays include image centre, path union, lower ROI, available path centre/boundaries, offset and class. Their IDs are in navigation_diagnostic_manifest.csv. Missing path geometry cannot be drawn.\n\nThe control team must later validate these targets against the intended physical controller. Phase B models are research navigation-prediction models, not direct motor-control systems. Perspective, camera calibration, obstacle depth, multiple disjoint paths, mask errors and temporal/site leakage remain unresolved. No physical robot safety or controller accuracy is claimed.\n''')
    print(dict(images=len(f),features=len(numeric.columns),missing_rate=rate,ml_count=len(selected),anfis=anfis,targets=counts.to_dict()))

if __name__=='__main__':run()
