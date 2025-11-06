from typing import List, Tuple, Optional
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

class DatasetConfig(BaseModel):
    csv: str
    label: str
    features: List[str]
    groups: List[str]

def load_dataset(cfg: DatasetConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.csv)
    expected = set([cfg.label, *cfg.features, *cfg.groups])
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

def tensorize_split(df: pd.DataFrame, cfg: DatasetConfig, test_size: float = 0.2, stratify: bool = True):
    X = df[cfg.features].values
    y = df[cfg.label].values
    strat = y if stratify else None
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values, test_size=test_size, random_state=42, stratify=strat
    )
    return (X_train, y_train, idx_train), (X_test, y_test, idx_test)

def group_vectors(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    return df[cfg.groups]
