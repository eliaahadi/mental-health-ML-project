
from __future__ import annotations
from typing import Optional
import pandas as pd
import numpy as np
from .config import ColumnConfig, CANONICAL_COLUMNS
from .utils import to_iso3, ensure_numeric

def ingest_csv(path: str, config_path: str) -> pd.DataFrame:
    cfg = ColumnConfig.from_yaml(config_path)
    df = pd.read_csv(path)
    mapping = cfg.canonicalize(df.columns.tolist())
    df = df.rename(columns=mapping)
    # Keep only canonical columns that exist
    keep = [c for c in CANONICAL_COLUMNS if c in df.columns]
    df = df[keep].copy()
    # Standardize data types
    if "year" in df:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    for col in ["prevalence_total","prevalence_male","prevalence_female",
                "prevalence_depression","prevalence_anxiety"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "country" in df:
        df["country_iso3"] = to_iso3(df["country"])
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with no country or year
    for c in ["country_iso3","year"]:
        if c not in df:
            raise ValueError(f"Missing required column '{c}'. Check your columns.yaml mapping and ingest step.")
    df = df.dropna(subset=["country_iso3","year"]).copy()
    # Deduplicate
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "sex"])
    return df

def gender_gap(df: pd.DataFrame) -> pd.DataFrame:
    # Compute female - male prevalence for each indicator when both exist
    out = []
    keys = ["country","country_iso3","year"]
    indicators = ["prevalence_total","prevalence_depression","prevalence_anxiety"]
    for name in indicators:
        if {"sex", name}.issubset(df.columns):
            wide = df.pivot_table(index=keys, columns="sex", values=name, aggfunc="mean")
            if {"female","male"}.issubset(set(wide.columns.astype(str).str.lower())):
                # normalize col names
                cols = {c:c.lower() for c in wide.columns}
                wide = wide.rename(columns=cols)
                wide[name+"_gap_fm"] = wide.get("female") - wide.get("male")
                part = wide.reset_index()[keys + [name+"_gap_fm"]]
                out.append(part)
    if not out:
        raise ValueError("Sex-stratified columns not found. Provide 'sex' and male/female prevalence columns.")
    # Merge all gaps
    res = out[0]
    for part in out[1:]:
        res = pd.merge(res, part, on=["country","country_iso3","year"], how="outer")
    return res

def join_external(base: pd.DataFrame, external: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    return pd.merge(base, external, on=on, how="left")
