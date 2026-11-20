
import streamlit as st
import os
import zipfile
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# Ortam değişkenlerini yükle
load_dotenv()

# API anahtarını al
api_key = st.secrets.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY"))

if not api_key:
    raise ValueError("OpenRouter API anahtarı bulunamadı.")

# OpenRouter bağlantısı
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# Veritabanı yolları
db_path = "./epidemic_db"
zip_path = "./epidemic_db.zip"

# Veritabanı klasörü yoksa ZIP dosyasını çıkar
if not os.path.exists(db_path):
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(".")
    else:
        raise FileNotFoundError("epidemic_db.zip bulunamadı.")

# ChromaDB bağlantısı
client_db = chromadb.PersistentClient(path=db_path)

# Koleksiyonu aç
col = client_db.get_collection("epidemic_data")


def salgin_analizi(soru):

    sonuc = col.query(
        query_texts=[soru],
        n_results=5
    )

    belgeler = sonuc.get("documents", [[]])[0]
    kaynaklar = sonuc.get("metadatas", [[]])[0]

    if not belgeler:
        return "İlgili salgın verisi bulunamadı."

    context = "\n".join(belgeler)

    prompt = f"""
Sen profesyonel bir salgın istihbarat asistanısın.

Türkçe bir salgın durum raporu hazırla.

Kurallar:
- Yalnızca verilen verilere dayan.
- Olmayan rakamları uydurma.
- Tarihsel verileri güncelmiş gibi gösterme.
- Veri yetersizse bunu açıkça belirt.
- Risk değerlendirmesini kesin tanı gibi sunma.

Rapor başlıkları:

## Mevcut Salgın Durumu
## Vaka ve Ölüm Bilgileri
## Haftalık Değişim
## Dikkat Çeken Bulgular
## Güven ve Belirsizlik
## Kaynaklar

SALGIN VERİLERİ:
{context}

KAYNAKLAR:
{kaynaklar}

SORU:
{soru}
"""

    response = client.chat.completions.create(
        model="openrouter/free",
        messages=[
            {
                "role": "system",
                "content": "Kaynak odaklı ve dikkatli bir salgın istihbarat asistanısın."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

