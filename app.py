import joblib
from tensorflow.keras.models import load_model
from agent import salgin_analizi
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
lstm_model = load_model("lstm_model.keras")
sc = joblib.load("lstm_scaler.pkl")
st.set_page_config(
    page_title="Türkiye Salgın İstihbaratı",
    page_icon="🦠",
    layout="wide"
)

st.title("🦠 Türkiye Salgın İstihbarat Sistemi")
st.caption("Yapay zekâ destekli salgın izleme, tahmin ve erken uyarı paneli")

# --------------------------------------------------
# VERİ YÜKLEME
# --------------------------------------------------

st.sidebar.header("Veri Kaynakları")

# Proje klasöründeki CSV dosyalarını bul
from pathlib import Path

klasor = Path(__file__).resolve().parent
csv_dosyalar = list(klasor.glob("*.csv"))

if not csv_dosyalar:
    st.error("Proje klasöründe CSV dosyası bulunamadı.")
    st.stop()

# Kullanılacak veri dosyasını seç
dosya_adi = st.sidebar.selectbox(
    "Salgın veri setini seç",
    csv_dosyalar,
    format_func=lambda x: x.name
)

try:
    df = pd.read_csv(dosya_adi)
except UnicodeDecodeError:
    df = pd.read_csv(dosya_adi, encoding="latin1")

st.sidebar.success(f"{len(df)} satır yüklendi.")

# Sütun seçimi
sutunlar = df.columns.tolist()
#st.write("CSV sütunları:", sutunlar)
st.sidebar.header("Veri Sütunlarını Eşleştir")

tarih_col = st.sidebar.selectbox(
    "Tarih sütunu", ["Seçme"] + sutunlar,
    index=sutunlar.index("Hafta") + 1 if "Hafta" in sutunlar else 0
)

vaka_col = st.sidebar.selectbox(
    "Vaka sayısı sütunu", ["Seçme"] + sutunlar,
    index=sutunlar.index("New_cases") + 1 if "New_cases" in sutunlar else 0
)

bolge_col = st.sidebar.selectbox(
    "Bölge / İl sütunu", ["Seçme"] + sutunlar,
    index=sutunlar.index("Country") + 1 if "Country" in sutunlar else 0
)

enlem_col = st.sidebar.selectbox("Enlem sütunu", ["Seçme"] + sutunlar)
boylam_col = st.sidebar.selectbox("Boylam sütunu", ["Seçme"] + sutunlar)

if tarih_col == "Seçme" or vaka_col == "Seçme":
    st.warning("Devam etmek için tarih ve vaka sütunlarını seçmelisin.")
    st.stop()

# Verileri düzenle
df[tarih_col] = pd.to_datetime(df[tarih_col], errors="coerce")
df[vaka_col] = pd.to_numeric(df[vaka_col], errors="coerce")
df = df.dropna(subset=[tarih_col, vaka_col]).copy()

if df.empty:
    st.error("Seçilen tarih ve vaka sütunlarında kullanılabilir veri bulunamadı.")
    st.stop()

# --------------------------------------------------
# GENEL GÖSTERGELER
# --------------------------------------------------

son_tarih = df[tarih_col].max()
son_veriler = df[df[tarih_col] == son_tarih]

toplam_vaka = df[vaka_col].sum()
son_vaka = son_veriler[vaka_col].sum()

onceki_tarihler = sorted(df[tarih_col].unique())

if len(onceki_tarihler) >= 2:
    onceki_vaka = df[
        df[tarih_col] == onceki_tarihler[-2]
    ][vaka_col].sum()

    degisim = (
        (son_vaka - onceki_vaka) / onceki_vaka * 100
        if onceki_vaka != 0 else 0
    )
else:
    degisim = 0

st.subheader("Genel Salgın Göstergeleri")

c1, c2, c3 = st.columns(3)

c1.metric("Toplam kayıtlı vaka", f"{toplam_vaka:,.0f}")
c2.metric("Son tarih vaka değeri", f"{son_vaka:,.0f}")
c3.metric("Son iki tarih arasındaki değişim", f"%{degisim:.2f}")

st.caption(f"Verideki en son tarih: {son_tarih:%d.%m.%Y}")

# --------------------------------------------------
# 1. BÖLGESEL VAKA VE TREND HARİTASI
# --------------------------------------------------

st.divider()
st.header("1. Bölgesel Vaka ve Trend Haritası")

if bolge_col != "Seçme":
    bolge_ozet = (
        df.groupby(bolge_col)[vaka_col]
        .sum()
        .reset_index()
        .sort_values(vaka_col, ascending=False)
    )

    st.subheader("Bölgelere göre vaka dağılımı")

    fig_bolge = px.bar(
        bolge_ozet,
        x=bolge_col,
        y=vaka_col,
        color=vaka_col,
        title="Bölgelere Göre Toplam Vaka",
        labels={
            bolge_col: "Bölge",
            vaka_col: "Vaka sayısı"
        }
    )

    st.plotly_chart(fig_bolge, use_container_width=True)

    # Gerçek harita için koordinatlar gerekir.
    if enlem_col != "Seçme" and boylam_col != "Seçme":
        harita_df = df.copy()

        harita_df[enlem_col] = pd.to_numeric(
            harita_df[enlem_col], errors="coerce"
        )
        harita_df[boylam_col] = pd.to_numeric(
            harita_df[boylam_col], errors="coerce"
        )

        harita_df = harita_df.dropna(
            subset=[enlem_col, boylam_col]
        )

        harita_df = (
            harita_df.groupby(bolge_col)
            .agg({
                vaka_col: "sum",
                enlem_col: "mean",
                boylam_col: "mean"
            })
            .reset_index()
        )

        if not harita_df.empty:
            fig_map = px.scatter_mapbox(
                harita_df,
                lat=enlem_col,
                lon=boylam_col,
                size=vaka_col,
                color=vaka_col,
                hover_name=bolge_col,
                zoom=4,
                mapbox_style="open-street-map",
                title="Türkiye Bölgesel Vaka Haritası"
            )

            st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info(
            "Gerçek harita için veri setinde enlem ve boylam sütunları "
            "bulunmalıdır. Şimdilik bölgesel vaka grafiği gösteriliyor."
        )
else:
    st.warning(
        "Bölgesel analiz için veri setinde il veya bölge sütunu bulunmalı."
    )

# --------------------------------------------------
# 2. TAHMİN - GERÇEKLEŞME PANELİ
# --------------------------------------------------

# --------------------------------------------------
# 2. LSTM TAHMİN - GERÇEKLEŞME PANELİ
# --------------------------------------------------

st.divider()
st.header("2. LSTM Tahmin – Gerçekleşme Karşılaştırması")

ozellikler = [
    "New_cases", "New_deaths", "COVID-19",
    "Coronavirus", "Fever", "Cough", "COVID Symptoms"
]

if all(col in df.columns for col in ozellikler):

    lstm_veri = df.sort_values(tarih_col).copy()

    for col in ozellikler:
        lstm_veri[col] = pd.to_numeric(
            lstm_veri[col], errors="coerce"
        )

    lstm_veri = lstm_veri.dropna(subset=ozellikler)

    if len(lstm_veri) >= 2:
        x = sc.transform(lstm_veri[ozellikler].values)
        x = x[:-1].reshape(-1, 1, 7)

        tahmin = lstm_model.predict(x, verbose=0).flatten()

        grafik_df = pd.DataFrame({
            "Tarih": lstm_veri[tarih_col].iloc[1:].values,
            "Gerçek Vaka": lstm_veri["New_cases"].iloc[1:].values,
            "LSTM Tahmini": tahmin
        })

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=grafik_df["Tarih"],
            y=grafik_df["Gerçek Vaka"],
            mode="lines+markers",
            name="Gerçek Vaka"
        ))

        fig.add_trace(go.Scatter(
            x=grafik_df["Tarih"],
            y=grafik_df["LSTM Tahmini"],
            mode="lines+markers",
            name="LSTM Tahmini"
        ))

        fig.update_layout(
            title="Gerçek ve Tahmin Edilen Vaka Sayıları",
            xaxis_title="Tarih",
            yaxis_title="Vaka Sayısı",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Tahmin için yeterli veri bulunamadı.")

else:
    st.warning("LSTM için gerekli 7 özellik CSV dosyasında bulunamadı.")

st.header("3. Erken Uyarı Sinyali Paneli")

esik = st.sidebar.slider(
    "Erken uyarı artış eşiği (%)",
    min_value=5,
    max_value=100,
    value=20
)

haftalik = df.copy()

haftalik["Hafta"] = (
    haftalik[tarih_col]
    .dt.to_period("W")
    .dt.start_time
)

if bolge_col != "Seçme":
    haftalik_ozet = (
        haftalik.groupby([bolge_col, "Hafta"])[vaka_col]
        .sum()
        .reset_index()
    )
    haftalik_ozet["Haftalık değişim (%)"] = (
        haftalik_ozet.groupby(bolge_col)[vaka_col]
        .pct_change() * 100
    )

    son_haftalar = (
        haftalik_ozet.sort_values("Hafta")
        .groupby(bolge_col)
        .tail(1)
        .copy()
    )
else:
    haftalik_ozet = (
        haftalik.groupby("Hafta")[vaka_col]
        .sum()
        .reset_index()
    )
    haftalik_ozet["Haftalık değişim (%)"] = (
        haftalik_ozet[vaka_col].pct_change() * 100
    )
    son_haftalar = haftalik_ozet.tail(1).copy()

uyari_sayisi = (
    son_haftalar["Haftalık değişim (%)"] > esik
).sum()

if uyari_sayisi > 0:
    st.error(
        f"{uyari_sayisi} bölge veya zaman diliminde "
        f"%{esik} üzerinde artış sinyali tespit edildi."
    )
else:
    st.success("Belirlenen eşiğin üzerinde artış sinyali bulunmadı.")

st.caption(
    "Bu panel basit bir haftalık vaka artış kuralı kullanır. "
    "Tek başına salgın teşhisi veya resmî erken uyarı değildir."
)

st.dataframe(
    son_haftalar,
    use_container_width=True
)

# --------------------------------------------------
# 4. YAPAY ZEKÂ BİLGİLENDİRME METNİ
# --------------------------------------------------


# --------------------------------------------------
# 4. YAPAY ZEKÂ SALGIN RAPORU
# --------------------------------------------------

st.subheader("4. Yapay Zekâ Salgın Değerlendirmesi")

soru = st.text_area(
    "Salgın hakkında neyi analiz etmek istiyorsun?",
    value="Türkiye'deki salgın verilerini ve vaka değişimlerini değerlendir."
)

if st.button("Yapay Zekâ Raporu Oluştur"):

    with st.spinner("Salgın verileri analiz ediliyor..."):

        try:
            rapor = salgin_analizi(soru)
            st.markdown(rapor)

        except Exception as e:
            st.error(f"Rapor oluşturulamadı: {e}")

# --------------------------------------------------
# 5. YAZILI TARTIŞMA
# --------------------------------------------------

st.divider()
st.header("5. Model ve Ajan Değerlendirmesi")

with st.expander("En tahmin edici sinyaller"):
    st.write("""
    Salgın tahmininde haftalık vaka değişimi, önceki haftaların vaka
    düzeyi ve zaman içindeki eğilimler değerlendirilebilir.

    Ancak hangi değişkenlerin modelimizde en güçlü tahmin ediciler
    olduğu, Gradient Boosting özellik önemleri ve model değerlendirme
    sonuçları incelenmeden kesin olarak söylenemez.
    """)

with st.expander("Model ve ajan sınırlamaları"):
    st.write("""
    - Eğitim verisinin kapsamı ve güncelliği model başarısını etkiler.
    - Eksik veya hatalı veriler yanlış tahminlere yol açabilir.
    - LSTM tahminleri kesin gelecek değerleri değildir.
    - DistilBERT metin sınıflandırması tıbbi teşhis yerine geçmez.
    - RAG ajanı yalnızca erişebildiği kaynaklar kadar güvenilirdir.
    - Yapay zekâ tarafından oluşturulan raporlar kaynaklarla
      karşılaştırılarak doğrulanmalıdır.
    """)

with st.expander("Etik konular: gözetim, yanlış bilgilendirme ve gizlilik"):
    st.write("""
    - Gözetim: Kişilerin konum ve sağlık verileri gereksiz yere
      izlenmemelidir.
    - Yanlış bilgilendirme: Doğrulanmamış tahminler kesin bilgi gibi
      sunulmamalıdır.
    - Gizlilik: Kişisel sağlık verileri anonimleştirilmeli ve
      yetkisiz kişilerle paylaşılmamalıdır.
    - Şeffaflık: Raporlarda kullanılan veri kaynakları ve belirsizlikler
      açıkça belirtilmelidir.
    - İnsan denetimi: Önemli halk sağlığı kararları yalnızca yapay
      zekâ çıktısına dayandırılmamalıdır.
    """)

with st.expander("Gelecekteki iyileştirmeler"):
    st.write("""
    - Daha güncel ve güvenilir veri kaynakları eklenebilir.
    - İl ve bölge düzeyinde coğrafi veri kapsamı geliştirilebilir.
    - LSTM ve Gradient Boosting modelleri karşılaştırılabilir.
    - Model başarısı farklı zaman dönemlerinde test edilebilir.
    - Erken uyarı eşikleri uzman görüşüyle belirlenebilir.
    - Kaynak gösteren ve belirsizlik bildiren RAG ajanı geliştirilebilir.
    """)

st.divider()
st.caption(
    "Türkiye Salgın İstihbarat Sistemi | Akademik proje prototipi"
)
