"""Provisional image-space research labels; these are not robot commands."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[2]

def generate(features,cfg):
    if cfg['smoothing']['enabled']:
        raise ValueError('Temporal smoothing requires validated sequence IDs; unavailable here.')
    rows=[]
    for f in features.to_dict('records'):
        flags=[]
        for key,limit,flag in [('path_area_ratio',cfg['minimum_path_area_ratio'],'low_area'),('path_width_lower',cfg['minimum_valid_path_width'],'narrow_path'),('path_continuity',cfg['minimum_path_continuity'],'discontinuous_path')]:
            if not np.isfinite(f[key]) or f[key]<limit: flags.append(flag)
        offset=f['normalized_path_offset']
        if not np.isfinite(offset): flags.append('missing_path_center')
        quality=min(np.clip(f['path_area_ratio']/cfg['minimum_path_area_ratio'],0,1),np.clip(f['path_width_lower']/cfg['minimum_valid_path_width'],0,1) if np.isfinite(f['path_width_lower']) else 0,f['path_continuity'])
        if quality<cfg['stop_uncertain_threshold']: flags.append('low_quality')
        label='stop_or_uncertain' if flags else ('left' if offset<cfg['left_threshold'] else 'right' if offset>cfg['right_threshold'] else 'forward')
        rows.append(dict(image_id=f['image_id'],split=f['split'],path_center_x=f['path_center_x'],image_center_x=f['image_width']/2,normalized_offset=offset,absolute_offset=abs(offset),path_width=f['path_width_lower'],path_area_ratio=f['path_area_ratio'],path_continuity=f['path_continuity'],discrete_target=label,continuous_target=offset,target_confidence=quality,target_status='provisional_geometry_derived',warning_flags=';'.join(flags),continuous_target_missing=int(not np.isfinite(offset))))
    return pd.DataFrame(rows)

def run():
    f=pd.read_csv(ROOT/'data/processed/features_ground_truth.csv');cfg=yaml.safe_load((ROOT/'configs/navigation_targets.yaml').read_text())
    feature_cfg=yaml.safe_load((ROOT/'configs/features.yaml').read_text())
    if cfg['lower_roi_start']!=feature_cfg['lower_roi_start']: raise ValueError('ROI mismatch: regenerate features with the target ROI')
    t=generate(f,cfg);t.to_csv(ROOT/'data/processed/navigation_targets.csv',index=False);return t
if __name__=='__main__':run()
