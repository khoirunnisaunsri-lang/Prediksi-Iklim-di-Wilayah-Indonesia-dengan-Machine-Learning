import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Cek scikit-learn
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
except ModuleNotFoundError:
    st.error("❌ scikit-learn belum terinstal. Tambahkan 'scikit-learn' pada requirements.txt")
    st.stop()

# Judul
st.title("🌦️ Prediksi Iklim Indonesia (Tanpa Upload + Dengan Upload)")
st.write("Aplikasi tetap berjalan meskipun Anda tidak mengunggah file.")

# ============================================================
# 1. PILIHAN DATA: UPLOAD ATAU AUTO
# ============================================================
uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="Data Harian - Table")
    st.success("📁 Data berhasil di-load dari upload!")
else:
    st.warning("⚠️ Anda belum upload file — memakai DATA CONTOH otomatis.")
    
    # ============================================================
    # DATA DEFAULT (DUMMY) — TIDAK PERLU FILE LAGI
    # ============================================================
    date_rng = pd.date_range(start="2010-01-01", end="2024-12-31", freq="D")
    np.random.seed(42)

    df = pd.DataFrame({
        "Tanggal": date_rng,
        "Tn": np.random.uniform(20, 25, len(date_rng)),
        "Tx": np.random.uniform(28, 34, len(date_rng)),
        "Tavg": np.random.uniform(24, 29, len(date_rng)),
        "kelembaban": np.random.uniform(60, 95, len(date_rng)),
        "curah_hujan": np.random.uniform(0, 20, len(date_rng)),
        "matahari": np.random.uniform(2, 10, len(date_rng)),
        "FF_X": np.random.uniform(1, 10, len(date_rng)),
        "DDD_X": np.random.uniform(0, 360, len(date_rng)),
    })

# ============================================================
# 2. PRA-PROSES DATA
# ============================================================
df = df.loc[:, ~df.columns.duplicated()]

df['Tanggal'] = pd.to_datetime(df['Tanggal'])
df['Tahun'] = df['Tanggal'].dt.year
df['Bulan'] = df['Tanggal'].dt.month

# Variabel utama
possible_vars = ["Tn", "Tx", "Tavg", "kelembaban", "curah_hujan", "matahari", "FF_X", "DDD_X"]
available_vars = [v for v in possible_vars if v in df.columns]

akademis_label = {
    "Tn": "Suhu Minimum (°C)",
    "Tx": "Suhu Maksimum (°C)",
    "Tavg": "Suhu Rata-rata (°C)",
    "kelembaban": "Kelembaban Udara (%)",
    "curah_hujan": "Curah Hujan (mm)",
    "matahari": "Durasi Penyinaran Matahari (jam)",
    "FF_X": "Kecepatan Angin Maksimum (m/s)",
    "DDD_X": "Arah Angin Maksimum (°)"
}

# ============================================================
# 3. AGREGASI BULANAN
# ============================================================
agg_dict = {v: 'mean' for v in available_vars}
agg_dict["curah_hujan"] = "sum"

monthly_df = df.groupby(['Tahun', 'Bulan']).agg(agg_dict).reset_index()

st.subheader("📊 Data Bulanan")
st.dataframe(monthly_df.head(20))

# ============================================================
# 4. TRAIN MODEL
# ============================================================
X = monthly_df[['Tahun', 'Bulan']]
models = {}
metrics = {}

for var in available_vars:
    y = monthly_df[var]

    if len(y) < 5:
        continue

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    models[var] = model
    metrics[var] = {
        "rmse": np.sqrt(mean_squared_error(y_test, pred)),
        "r2": r2_score(y_test, pred)
    }

# ============================================================
# 5. EVALUASI
# ============================================================
st.subheader("📈 Evaluasi Model")
for var in models.keys():
    m = metrics[var]
    st.write(f"**{akademis_label[var]}** — RMSE: {m['rmse']:.3f} | R²: {m['r2']:.3f}")

# ============================================================
# 6. PREDIKSI MANUAL
# ============================================================
st.subheader("🔮 Prediksi Manual")
tahun_input = st.number_input("Tahun", 2025, 2100, 2035)
bulan_input = st.selectbox("Bulan", list(range(1, 13)))

input_df = pd.DataFrame([[tahun_input, bulan_input]], columns=["Tahun", "Bulan"])

for var in models.keys():
    pred_val = models[var].predict(input_df)[0]
    st.success(f"{akademis_label[var]} bulan {bulan_input}/{tahun_input}: **{pred_val:.2f}**")

# ============================================================
# 7. PREDIKSI 2025–2075
# ============================================================
years = list(range(2025, 2076))
months = list(range(1, 13))
future_df = pd.DataFrame([(y, m) for y in years for m in months], columns=["Tahun", "Bulan"])

for var in models.keys():
    future_df[f"Pred_{var}"] = models[var].predict(future_df[['Tahun', 'Bulan']])

# ============================================================
# 8. GRAFIK
# ============================================================
st.subheader("📈 Grafik Prediksi")

monthly_df['Sumber'] = 'Historis'
future_df['Sumber'] = 'Prediksi'

gabungan = []

for var in models.keys():
    h = monthly_df[['Tahun', 'Bulan', var, 'Sumber']].rename(columns={var: "Nilai"})
    h['Variabel'] = akademis_label[var]

    p = future_df[['Tahun', 'Bulan', f"Pred_{var}", 'Sumber']].rename(columns={f"Pred_{var}": "Nilai"})
    p['Variabel'] = akademis_label[var]

    gabungan.append(pd.concat([h, p]))

merged = pd.concat(gabungan)
merged['Tanggal'] = pd.to_datetime(merged['Tahun'].astype(str) + "-" + merged['Bulan'].astype(str) + "-01")

pilih_var = st.selectbox("Pilih Variabel", list(akademis_label[v] for v in models.keys()))

fig = px.line(
    merged[merged["Variabel"] == pilih_var],
    x="Tanggal",
    y="Nilai",
    color="Sumber",
    title=f"Tren {pilih_var} (Historis vs Prediksi)"
)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 9. DOWNLOAD
# ============================================================
st.subheader("💾 Download Prediksi 2025–2075")
csv = future_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download CSV", csv, "prediksi_2025_2075.csv", "text/csv")
