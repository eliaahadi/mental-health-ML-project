
import streamlit as st
import pandas as pd
from mh.plots import choropleth, country_trend
from mh.utils import latest_year

st.set_page_config(page_title="Global Mental Health Indicators", layout="wide")
st.title("🧠 Global Mental Health Indicators")

data_path = st.text_input("Path to processed data (.parquet)", "data/processed/clean.parquet")
if not data_path:
    st.stop()

@st.cache_data
def load_df(p):
    return pd.read_parquet(p)

try:
    df = load_df(data_path)
except Exception as e:
    st.error(f"Could not load {data_path}: {e}")
    st.stop()

st.sidebar.header("Controls")
indicator = st.sidebar.selectbox("Indicator", [c for c in df.columns if c.startswith("prevalence")])
yr = st.sidebar.selectbox("Year", ["latest"] + sorted([int(x) for x in df["year"].dropna().unique()]))
st.write(f"### Choropleth — {indicator} ({yr})")
fig = choropleth(df, indicator=indicator, year=yr)
st.plotly_chart(fig, use_container_width=True)

st.write("### Country trend")
iso_list = sorted(df["country_iso3"].dropna().unique())
iso = st.selectbox("Country (ISO‑3)", iso_list)
trend_fig = country_trend(df, country_iso3=iso, indicator=indicator)
st.plotly_chart(trend_fig, use_container_width=True)

# Optional gender gap panel
gap_cols = [c for c in df.columns if c.endswith("_gap_fm")]
if gap_cols:
    st.write("### Gender gap (female - male) — latest year")
    latest = df.sort_values(["country_iso3","year"]).groupby("country_iso3").tail(1)
    show = latest[["country","country_iso3"] + gap_cols].sort_values(gap_cols[0], ascending=False).head(20)
    st.dataframe(show)
