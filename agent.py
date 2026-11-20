
import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()

# API anahtarını al
api_key = os.getenv("OPENROUTER_API_KEY")

# OpenRouter bağlantısını oluştur
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

# ChromaDB bağlantısı
client_db = chromadb.PersistentClient(path="./epidemic_db")
col = client_db.get_collection("epidemic_data")

def salgin_analizi(soru):

    if not api_key:
        return "OpenRouter API anahtarı tanımlanmamış."

    # ChromaDB'den ilgili belgeleri getir
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
