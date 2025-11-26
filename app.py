import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="📊 Dashboard Prediksi Iklim", layout="wide")

# ========== 1️⃣ LOAD DATA ==========
@st.cache_data
def load_data():
    df = pd.read_excel("DATA BMKG_JAKARTA PUSAT 2015-2025.xlsx", sheet_name="Data Harian - Table")
    df = df.loc[:, ~df.columns.duplicated()]
    if "kecepatan_angin" in df.columns:
        df = df.rename(columns={"kecepatan_angin":"FF_X"})
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], dayfirst=True)
    df["Tahun"] = df["Tanggal"].dt.year
    df["Bulan"] = df["Tanggal"].dt.month
    return df

df = load_data()

wilayah = "Jakarta Pusat"
st.title(f"🌦️ Dashboard Analisis & Prediksi Iklim — {wilayah}")


# ========== 2️⃣ Sidebar Filter ==========
st.sidebar.header("🔎 Filter Data")

selected_year = st.sidebar.multiselect(
    "Pilih Tahun",
    sorted(df["Tahun"].unique()),
    default=df["Tahun"].unique()
)

selected_month = st.sidebar.multiselect(
    "Pilih Bulan",
    range(1, 13),
    default=range(1, 13)
)

df = df[df["Tahun"].isin(selected_year)]
df = df[df["Bulan"].isin(selected_month)]

possible_vars = ["Tn","Tx","Tavg","kelembaban","curah_hujan","matahari","FF_X","DDD_X"]
available_vars = [v for v in possible_vars if v in df.columns]

label = {
    "Tn": "Suhu Minimum (°C)",
    "Tx": "Suhu Maksimum (°C)",
    "Tavg": "Suhu Rata-rata (°C)",
    "kelembaban": "Kelembaban (%)",
    "curah_hujan": "Curah Hujan (mm)",
    "matahari": "Durasi Matahari (jam)",
    "FF_X": "Kecepatan Angin (m/s)",
    "DDD_X": "Arah Angin (°)"
}

# ========== 3️⃣ Agregasi ==========
agg_dict = {v:"mean" for v in available_vars}
if "curah_hujan" in available_vars:
    agg_dict["curah_hujan"] = "sum"

monthly = df.groupby(["Tahun","Bulan"]).agg(agg_dict).reset_index()


# ========== 4️⃣ Model ==========
models = {}
metrics = {}

for v in available_vars:
    X = monthly[["Tahun","Bulan"]]
    y = monthly[v]

    Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.2, random_state=42)

    m = RandomForestRegressor(n_estimators=180, random_state=42)
    m.fit(Xtr,ytr)
    pred = m.predict(Xts)

    models[v] = m
    metrics[v] = (mean_squared_error(yts,pred)**0.5, r2_score(yts,pred))

# ========== 5️⃣ Card Statistik ==========
c1,c2,c3 = st.columns(3)
c1.metric("📏 Data Historis", f"{len(df):,} record")
c2.metric("📅 Rentang Tahun", f"{df['Tahun'].min()} - {df['Tahun'].max()}")
c3.metric("📦 Variabel Iklim", len(available_vars))


# ========== 6️⃣ Grafik Tren ==========
st.subheader("📈 Tren Data Historis")
var_plot = st.selectbox("Pilih Variabel", [label[v] for v in available_vars])

key = [k for k,v in label.items() if v==var_plot][0]

monthly["Tanggal"] = pd.to_datetime(
    monthly["Tahun"].astype(str)+"-"+monthly["Bulan"].astype(str)+"-01"
)

fig1 = px.line(
    monthly,
    x="Tanggal",
    y=key,
    markers=True,
    title=var_plot,
    template="plotly_white"
)
st.plotly_chart(fig1, use_container_width=True)


# ========== 7️⃣ Prediksi 50 Tahun ==========
future = pd.DataFrame([(y,m) for y in range(2025,2076) for m in range(1,13)], columns=["Tahun","Bulan"])
for v in available_vars:
    future[f"Pred_{v}"] = models[v].predict(future[["Tahun","Bulan"]])

st.subheader("🔮 Prediksi 2025–2075")
var_pred = st.selectbox("Pilih Variabel Prediksi", [label[v] for v in available_vars])

key2 = [k for k,v in label.items() if v==var_pred][0]
future["Tanggal"] = pd.to_datetime(
    future["Tahun"].astype(str)+"-"+future["Bulan"].astype(str)+"-01"
)

fig2 = px.line(
    future,
    x="Tanggal",
    y=f"Pred_{key2}",
    title=f"Prediksi {var_pred}",
    template="plotly_white"
)
st.plotly_chart(fig2, use_container_width=True)


# ========== 8️⃣ Download ==========
csv = future.to_csv(index=False).encode("utf8")
st.download_button(
    "📥 Download Dataset Prediksi",
    data=csv,
    file_name="prediksi_jawa_timur.csv",
    mime="text/csv"
)
