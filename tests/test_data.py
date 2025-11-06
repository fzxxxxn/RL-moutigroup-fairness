import json
from pathlib import Path
import pandas as pd
from rlmgf.data import DatasetConfig, load_dataset, tensorize_split

def test_dataset_load():
    cfg = DatasetConfig(csv='data/demo/demo.csv', label='label',
                        features=['f0','f1','f2','f3'],
                        groups=['g_frl','g_race','g_hisp','g_intersection'])
    df = load_dataset(cfg)
    assert not df.empty
    (Xtr, ytr, idtr), (Xte, yte, idte) = tensorize_split(df, cfg, 0.2, True)
    assert Xtr.shape[0] > 0 and Xte.shape[0] > 0
