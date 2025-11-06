import pandas as pd

def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def ensure_schema(df: pd.DataFrame):
    assert 'label' in df.columns and 'group' in df.columns
    return df
