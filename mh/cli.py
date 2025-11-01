
from __future__ import annotations
import typer
import yaml
from typing import Optional, List
import pandas as pd
from .data import ingest_csv, clean, gender_gap, join_external
from .plots import choropleth, country_trend
from .models import forecast_country, cluster_countries
from .utils import to_iso3, read_parquet_safe
from pathlib import Path

app = typer.Typer(add_completion=False)

@app.command()
def ingest(input: str, output: str, config: str):
    df = ingest_csv(input, config)
    df.to_parquet(output, index=False)
    typer.echo(f"Wrote standardized data -> {output}")

@app.command()
def clean_data(input: str, output: str):
    df = read_parquet_safe(input)
    df = clean(df)
    df.to_parquet(output, index=False)
    typer.echo(f"Wrote clean data -> {output}")

@app.command("gender-gap")
def gender_gap_cmd(input: str, output: str):
    df = read_parquet_safe(input)
    gg = gender_gap(df)
    gg.to_csv(output, index=False)
    typer.echo(f"Wrote gender gap report -> {output}")

@app.command("visualize")
def visualize(input: str, out_html: str, indicator: str = "prevalence_total", year: str = "latest"):
    df = read_parquet_safe(input)
    fig = choropleth(df, indicator=indicator, year=year)
    fig.write_html(out_html)
    typer.echo(f"Wrote map -> {out_html}")

@app.command("trend")
def trend(input: str, country_iso3: str, indicator: str, out_html: str):
    df = read_parquet_safe(input)
    fig = country_trend(df, country_iso3, indicator)
    fig.write_html(out_html)
    typer.echo(f"Wrote trend -> {out_html}")

@app.command("forecast")
def forecast(input: str, country: str, indicator: str = "prevalence_total", steps: int = 5, out_png: str = "reports/forecast.png"):
    df = read_parquet_safe(input)
    country_key = country.strip()
    country_iso3 = None
    if "country_iso3" in df.columns:
        mask_iso = df["country_iso3"].astype(str).str.lower() == country_key.lower()
        if mask_iso.any():
            country_iso3 = df.loc[mask_iso, "country_iso3"].iloc[0]
        else:
            mask_name = df["country"].astype(str).str.lower() == country_key.lower()
            if mask_name.any():
                country_iso3 = df.loc[mask_name, "country_iso3"].iloc[0]
    if country_iso3 is None:
        try:
            iso_guess = to_iso3(pd.Series([country_key])).iloc[0]
            if isinstance(iso_guess, str) and iso_guess:
                country_iso3 = iso_guess
        except Exception:
            country_iso3 = None
    if country_iso3 is None:
        raise typer.BadParameter(f"Could not resolve '{country}' to a country ISO3 code. Try passing the ISO3 code explicitly.")

    fut = forecast_country(df, country_iso3=country_iso3, indicator=indicator, steps=steps)
    # Simple PNG output via plotly for consistency with HTML
    import plotly.express as px
    hist = df[(df["country_iso3"] == country_iso3)][["year", indicator]].dropna().sort_values("year")
    hist["type"] = "observed"
    fut2 = fut.rename(columns={indicator:"value"})
    hist2 = hist.rename(columns={indicator:"value"})
    hist2["type"] = "observed"
    fut2["type"] = "forecast"
    plot_df = pd.concat([hist2, fut2], ignore_index=True, sort=False)
    fig = px.line(plot_df, x="year", y="value", color="type", title=f"{indicator} forecast — {country}")
    try:
        fig.write_image(out_png, scale=2)
        typer.echo(f"Wrote forecast -> {out_png}")
    except ValueError as err:
        if "kaleido" in str(err).lower():
            fallback = Path(out_png).with_suffix(".html")
            fig.write_html(str(fallback))
            typer.secho(
                f"Kaleido missing; wrote HTML fallback instead -> {fallback}. "
                f"Install kaleido for PNG export (`pip install kaleido`).",
                err=True,
            )
        else:
            raise

@app.command("cluster")
def cluster(input: str, year: str = "latest", k: int = 5, out_csv: str = "reports/clusters.csv", features: List[str] = typer.Option(None)):
    df = read_parquet_safe(input)
    if features is None or len(features) == 0:
        features = ["prevalence_total","prevalence_depression","prevalence_anxiety"]
    res = cluster_countries(df, year=year, features=features, n_clusters=k)
    res.to_csv(out_csv, index=False)
    typer.echo(f"Wrote clusters -> {out_csv}")

@app.command("join-external")
def join_external_cmd(base: str, external: str, key: str = "country_iso3 year", out: str = "data/processed/joined.parquet"):
    key_cols = key.split()
    b = read_parquet_safe(base)
    e = pd.read_csv(external)
    # try to ensure year numeric if present
    for c in ["year","Year","YEAR"]:
        if c in e.columns:
            e[c if c=="year" else "year"] = pd.to_numeric(e[c], errors="coerce")
            if c != "year":
                e = e.rename(columns={c:"year"})
            break
    # Normalize country keys if present
    if "country_iso3" not in e.columns:
        for c in ["iso_code","ISO3","iso3","Code"]:
            if c in e.columns:
                e = e.rename(columns={c:"country_iso3"})
                break
    joined = join_external(b, e, on=key_cols)
    joined.to_parquet(out, index=False)
    typer.echo(f"Wrote joined dataset -> {out}")

if __name__ == "__main__":
    app()
