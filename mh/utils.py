
from __future__ import annotations
import pandas as pd
import numpy as np
import warnings
import country_converter as coco

_cc = coco.CountryConverter()

def to_iso3(country_series: pd.Series) -> pd.Series:
    def _map(x):
        try:
            return _cc.convert(names=x, to='ISO3')
        except Exception:
            return np.nan
    return country_series.astype(str).map(_map)

def coalesce_first(df: pd.DataFrame, cols: list[str], out: str) -> pd.DataFrame:
    df[out] = None
    for c in cols:
        if c in df:
            df[out] = df[out].fillna(df[c])
    return df

def ensure_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def latest_year(df: pd.DataFrame) -> int | None:
    if "year" not in df:
        return None
    vals = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
    return int(vals.max()) if len(vals) else None
