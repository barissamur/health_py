# Temel alınacak imaj
FROM python:3.10

# Çalışma dizini ayarı
WORKDIR /app

# Gerekli Python paketlerini yükleyin
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Uygulama dosyalarını kopyalayın
COPY . /app

# Uygulamayı çalıştırın
CMD ["python", "1_production.py"]
