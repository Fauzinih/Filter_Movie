import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🎬 Rekomendasi Film Berdasarkan Usia & Tahun (Decision Tree)")

# === 1. Load Data ===
@st.cache_data
def load_data():
    df = pd.read_csv('Movies.csv', encoding='mac_roman')
    df['Usia'] = df['Age'].replace('SU', '0').str.replace('+', '', regex=False).astype(int)
    return df

df = load_data()

# === 2. Sidebar Input ===
st.sidebar.header("🎯 Input Preferensi")
usia_pengguna = st.sidebar.slider("Usia Pengguna untuk prediksi", 0, 25, 15)
tahun_min = st.sidebar.slider("Tahun Rilis (Min)", 2000, 2021, 2010)
usia_latih = 17
# === 3. Label Berdasarkan Usia Latih
df['Boleh'] = df['Usia'].apply(lambda x: 1 if usia_latih >= x else 0)

# === 4. Training Model (pakai hanya Usia & Year)
fitur = ['Usia', 'Year']
X = df[fitur]
y = df['Boleh']
model = DecisionTreeClassifier(max_depth=2)
model.fit(X, y)

# === 5. Prediksi Semua Data
df['Prediksi'] = model.predict(X)

# === 6. Filter untuk Film yang Cocok Ditonton
film_boleh = df[
    (df['Usia'] <= usia_pengguna) &
    (df['Prediksi'] == 1) &
    (df['Year'] >= tahun_min)
]

# === 7. Tampilkan Rekomendasi Film
st.subheader("✅ Film yang Boleh Ditonton")
st.write(f"Model dilatih untuk batas usia ≤ {usia_latih}, prediksi untuk pengguna usia {usia_pengguna}, Tahun ≥ {tahun_min}")

if not film_boleh.empty:
    film_boleh_display = film_boleh[['Title', 'Year', 'Usia']].reset_index(drop=True)
    film_boleh_display.index += 1  # mulai dari 1
    film_boleh_display.index.name = 'No'
    st.dataframe(film_boleh_display)

else:
    st.warning("Tidak ada film yang cocok dengan kriteria.")

# === 8. Perhitungan Manual (Logika If-Else)
st.subheader("🧮 Perhitungan Manual Setiap Film")

def logika_manual(row):
    if row['Usia'] <= usia_latih:
        if row['Year'] > 2010:
            return "✅ Boleh (Usia ≤ {} dan Year > 2010)".format(usia_latih)
        else:
            return "❌ Tidak Boleh (Year ≤ 2010)"
    else:
        return "❌ Tidak Boleh (Usia > {})".format(usia_latih)

film_boleh_display = df.copy()
film_boleh_display['Penjelasan'] = film_boleh_display.apply(logika_manual, axis=1)

# Ambil hanya film yang usia ≤ usia_pengguna
hasil_pengguna = film_boleh_display[film_boleh_display['Usia'] <= usia_pengguna]

# Tampilkan hasil perhitungan manual
hasil_tampil = hasil_pengguna[['Title', 'Year', 'Usia', 'Penjelasan']].reset_index(drop=True)
hasil_tampil.index += 1
hasil_tampil.index.name = 'No'

st.dataframe(hasil_tampil)