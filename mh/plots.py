
from __future__ import annotations
import pandas as pd
import plotly.express as px

def choropleth(df: pd.DataFrame, indicator: str, year: str | int = "latest"):
    d = df.copy()
    if year == "latest":
        d = d.sort_values(["country_iso3","year"]).groupby("country_iso3").tail(1)
    else:
        d = d[d["year"] == int(year)]
    fig = px.choropleth(
        d,
        locations="country_iso3",
        color=indicator,
        hover_name="country",
        color_continuous_scale="Viridis",
        projection="natural earth",
    )
    fig.update_layout(title=f"{indicator} by country ({year})")
    return fig

def country_trend(df: pd.DataFrame, country_iso3: str, indicator: str):
    d = df[df["country_iso3"] == country_iso3].sort_values("year")
    fig = px.line(d, x="year", y=indicator, title=f"{indicator} over time — {country_iso3}")
    return fig
