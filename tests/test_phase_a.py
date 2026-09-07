"""Geometry and target safeguards independent of dataset outcomes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import unittest
import numpy as np
import yaml
from features.feature_extraction import extract,masks_from_polygons,ROOT
from navigation.target_generation import generate
import pandas as pd

class GeometryTests(unittest.TestCase):
    def test_empty_path_is_missing_not_forward(self):
        f=extract(np.full((64,64,3),128,np.uint8),{0:[],1:[],2:[]})
        self.assertTrue(np.isnan(f['path_center_x']))
        self.assertEqual(f['path_center_x_missing'],1)
        cfg=yaml.safe_load((ROOT/'configs/navigation_targets.yaml').read_text())
        t=generate(pd.DataFrame([dict(image_id='x',split='train',image_width=64,**f)]),cfg)
        self.assertEqual(t.iloc[0].discrete_target,'stop_or_uncertain')
        self.assertTrue(np.isnan(t.iloc[0].continuous_target))
    def test_union_does_not_cancel_overlap(self):
        p=np.array([[.1,.1],[.9,.1],[.9,.9],[.1,.9]])
        a=masks_from_polygons({0:[p],1:[],2:[]},64,64)[0]
        b=masks_from_polygons({0:[p,p],1:[],2:[]},64,64)[0]
        np.testing.assert_array_equal(a,b)
    def test_image_left_negative(self):
        p=np.array([[.05,.1],[.4,.1],[.4,1],[.05,1]])
        f=extract(np.full((64,64,3),128,np.uint8),{0:[p],1:[],2:[]})
        self.assertLess(f['normalized_path_offset'],0)
        cfg=yaml.safe_load((ROOT/'configs/navigation_targets.yaml').read_text())
        t=generate(pd.DataFrame([dict(image_id='x',split='train',image_width=64,**f)]),cfg)
        self.assertEqual(t.iloc[0].discrete_target,'left')
if __name__=='__main__': unittest.main()
