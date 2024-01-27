# Python resmi imajını kullan
FROM python:3.10

# Çalışma dizinini ayarla
WORKDIR /app

# Gerekli Python paketlerini yükle
COPY requirements.txt .
RUN pip install -r requirements.txt

# Uygulama dosyalarını kopyala
COPY . .

# Uygulamanın çalışacağı portu belirt
EXPOSE 5000

# Uygulamayı başlat
CMD ["python", "app.py"]