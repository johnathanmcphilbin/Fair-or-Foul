import pandas as pd
from pathlib import Path

def load_calls_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    return pd.read_csv(p)

def save_processed(df, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
