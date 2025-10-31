
from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from statsmodels.tsa.statespace.sarimax import SARIMAX

def forecast_country(df: pd.DataFrame, country_iso3: str, indicator: str, steps: int = 5, order=(1,1,1), seasonal_order=(0,0,0,0)) -> pd.DataFrame:
    d = df[(df["country_iso3"] == country_iso3) & (~df[indicator].isna())].sort_values("year")
    if d.empty or d["year"].nunique() < 5:
        raise ValueError("Not enough data points to forecast. Need >=5 years of data.")
    y = d.set_index("year")[indicator].astype(float)
    model = SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=steps)
    idx = range(int(d["year"].max())+1, int(d["year"].max())+1+steps)
    out = pd.DataFrame({"year": list(idx), indicator: pred.predicted_mean.values, "country_iso3": country_iso3})
    return out

def cluster_countries(df: pd.DataFrame, year: int | str = "latest", features: list[str] = None, n_clusters: int = 5) -> pd.DataFrame:
    if features is None or not features:
        raise ValueError("Provide features to cluster.")
    d = df.copy()
    if year == "latest":
        d = d.sort_values(["country_iso3","year"]).groupby("country_iso3").tail(1)
    else:
        d = d[d["year"] == int(year)]
    X = d[features].copy()
    X = X.fillna(X.median(numeric_only=True))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(Xs)
    out = d[["country","country_iso3","year"]].copy()
    out["cluster"] = labels
    return out
