import pandas as pd
import numpy as np


def z_score(series: pd.Series) -> pd.Series:
    return(series - series.mean()) / series.std(ddof=0)


def min_max(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


def normalize_features(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for col in cols:
        if col not in df.columns:
            print(f"[normalize] WARNING: '{col}' not found, skipping")
            continue
        df[f"{col}_z"] = z_score(df[col])
    return df