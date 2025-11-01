from __future__ import annotations

import logging
import warnings
from functools import lru_cache

import json
import numpy as np
import pandas as pd

try:  # Python 3.9+
    from importlib import resources
except ImportError:  # pragma: no cover - fallback for older interpreters
    import importlib_resources as resources  # type: ignore[import]

try:
    import country_converter as coco
except ImportError:  # pragma: no cover - optional dependency
    coco = None  # type: ignore[assignment]

_cc = coco.CountryConverter() if coco else None
if _cc is not None:
    logging.getLogger("country_converter").setLevel(logging.ERROR)
else:
    warnings.warn(
        "country_converter not installed; using bundled ISO3 fallback mapping. "
        "Install country_converter for more comprehensive coverage.",
        ImportWarning,
        stacklevel=2,
    )


def _normalize_key(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


@lru_cache(maxsize=1)
def _fallback_iso_map() -> dict[str, str]:
    # Lazy-load bundled fallback map generated from CountryConverter metadata.
    data = resources.files("mh").joinpath("_country_iso_fallback.json")
    with data.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return {_normalize_key(name): code for name, code in raw.items()}


def _fallback_lookup(name: str) -> str | None:
    if not isinstance(name, str):
        return None
    key = _normalize_key(name)
    if not key:
        return None
    return _fallback_iso_map().get(key)


def to_iso3(country_series: pd.Series) -> pd.Series:
    def _map(x):
        if isinstance(x, str):
            name = x.strip()
        else:
            name = ""
        if not name:
            return np.nan

        code: str | None = None
        if _cc is not None:
            try:
                code = _cc.convert(names=name, to="ISO3", not_found=None)
            except Exception:  # pragma: no cover - defensive guard
                code = None
            if isinstance(code, (list, tuple)):
                code = code[0] if len(code) else None
            if isinstance(code, str) and code.lower().strip() in {"not found", "notfound", ""}:
                code = None
        if not isinstance(code, str) or not code.strip():
            code = _fallback_lookup(name)
        return code if isinstance(code, str) and code else np.nan

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
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def latest_year(df: pd.DataFrame) -> int | None:
    if "year" not in df:
        return None
    vals = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
    return int(vals.max()) if len(vals) else None


def read_parquet_safe(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    """
    Read a parquet file with a defensive fallback for environments where
    pyarrow's dataset reader raises `Repetition level histogram size mismatch`.
    """
    try:
        return pd.read_parquet(path, columns=columns)
    except OSError as err:
        message = str(err)
        if "Repetition level histogram size mismatch" not in message:
            raise
        warnings.warn(
            "pyarrow dataset reader failed, attempting fallback read via pyarrow.parquet",
            RuntimeWarning,
            stacklevel=2,
        )
        try:
            import pyarrow as pa  # type: ignore[import]
            import pyarrow.parquet as pq  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - should not happen given requirements
            raise err from exc
        try:
            table = pq.read_table(path, use_legacy_dataset=True, use_threads=False)
        except TypeError:
            table = pq.read_table(path, use_threads=False)
        except OSError as err2:
            if "Repetition level histogram size mismatch" not in str(err2):
                raise
            try:
                table = pq.ParquetFile(path).read(use_threads=False)
            except OSError as err3:
                if "Repetition level histogram size mismatch" not in str(err3):
                    raise
                try:
                    import fastparquet  # type: ignore  # noqa: F401
                    return pd.read_parquet(path, columns=columns, engine="fastparquet")
                except Exception:
                    raise err3
        frame = table.to_pandas()
        if columns:
            missing = [c for c in columns if c not in frame.columns]
            if missing:
                raise KeyError(f"Columns not found in parquet file: {missing}")
            frame = frame[columns]
        return frame
