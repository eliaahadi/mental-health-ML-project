
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
    # Derive total prevalence when only sex-specific columns are present
    if "prevalence_total" not in df.columns and {"prevalence_male", "prevalence_female"}.issubset(df.columns):
        df["prevalence_total"] = df[["prevalence_male", "prevalence_female"]].mean(axis=1)
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
    produced = set()
    keys = ["country","country_iso3","year"]
    indicators = ["prevalence_total","prevalence_depression","prevalence_anxiety"]
    # First try the long format (sex column available)
    if "sex" in df.columns:
        for name in indicators:
            if {"sex", name}.issubset(df.columns):
                wide = df.pivot_table(index=keys, columns="sex", values=name, aggfunc="mean")
                normalized_cols = {c: str(c).lower() for c in wide.columns}
                wide = wide.rename(columns=normalized_cols)
                if {"female", "male"}.issubset(wide.columns):
                    gap_col = name + "_gap_fm"
                    if gap_col in produced:
                        continue
                    wide[gap_col] = wide["female"] - wide["male"]
                    part = wide.reset_index()[keys + [gap_col]]
                    out.append(part)
                    produced.add(gap_col)

    # Fallback for wide datasets that already have *_male/*_female columns
    male_cols = [c for c in df.columns if c.endswith("_male")]
    for male_col in male_cols:
        base = male_col[:-5]
        female_col = f"{base}_female"
        if female_col not in df.columns:
            continue
        indicator_name = base
        if indicator_name.endswith("_"):
            indicator_name = indicator_name.rstrip("_")
        indicator_name = indicator_name or male_col[:-5]
        gap_name = indicator_name + "_gap_fm"
        if gap_name in produced:
            continue
        part = df[keys].copy()
        part[gap_name] = df[female_col] - df[male_col]
        out.append(part)
        produced.add(gap_name)

    if not out:
        raise ValueError("Sex-stratified columns not found. Provide 'sex' column or *_male/*_female pairs.")
    # Merge all gaps
    res = out[0]
    for part in out[1:]:
        res = pd.merge(res, part, on=["country","country_iso3","year"], how="outer")
    return res

def join_external(base: pd.DataFrame, external: pd.DataFrame, on: list[str]) -> pd.DataFrame:
    return pd.merge(base, external, on=on, how="left")
