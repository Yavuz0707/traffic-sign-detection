# 🚦 Trafik İşareti Tanıma Sistemi

YOLOv8 kullanarak otonom araçlar için gerçek zamanlı trafik işareti tanıma projesi.

## 📋 Proje Hakkında

Bu proje, derin öğrenme tabanlı YOLOv8 (You Only Look Once) algoritması kullanarak 10 farklı trafik işaretini tespit eden bir görüntü işleme uygulamasıdır. Otonom araçlar, güvenlik sistemleri ve akıllı ulaşım sistemleri için tasarlanmıştır.

### 🎯 Tespit Edilen Trafik İşaretleri

- 🚧 **Hump** (Tümsek)
- 🚫 **No Entry** (Girilmez)
- 🚗 **No Overtaking** (Sollama Yapılmaz)
- 🛑 **No Stopping** (Durma Yasak)
- ↩️ **No U Turn** (U Dönüşü Yapılmaz)
- 🅿️ **Parking** (Park Yeri)
- 👷 **Roadwork** (Yol Çalışması)
- 🔄 **Roundabout** (Kavşak)
- 🔢 **Speed Limit 40** (Hız Sınırı 40)
- 🛑 **Stop** (Dur)

## ✨ Özellikler

- ⚡ **Gerçek Zamanlı Tespit**: YOLOv8'in hızlı algılama kabiliyeti
- 🎯 **Yüksek Doğruluk**: %79.81 mAP50 skoruyla güvenilir tespit
- 🔧 **Kolay Kullanım**: Basit eğitim ve test scripti
- 📊 **Detaylı Raporlama**: Eğitim metrikleri ve görselleştirme
- 🛠️ **Özelleştirilebilir**: Hyperparameter ayarlaması mevcut

## 📦 Kurulum

### Gereksinimler

- Python 3.8+
- CUDA (GPU kullanımı için - opsiyonel)

### Adımlar

1. Repoyu klonlayın:
```bash
git clone <repo-url>
cd project_5
```

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install ultralytics opencv-python
```

## 🚀 Kullanım

### Model Eğitimi

```bash
python train.py
```

**Eğitim Parametreleri:**
- **Epochs**: 2 (eğitim döngüsü sayısı)
- **Image Size**: 640x640
- **Batch Size**: 16
- **Optimizer**: SGD
- **Learning Rate**: 0.01
- **Device**: CPU (CUDA için `device='cuda'` yapabilirsiniz)

### Model Testi

Test etmek istediğiniz görüntüyü `test1.jpg` olarak kaydedin ve:

```bash
python test.py
```

Test sonucu `test_result.jpg` olarak kaydedilecektir.

## 📊 Model Performansı

Eğitim tamamlandıktan sonra elde edilen metrikler:

| Metrik | Değer |
|--------|-------|
| **Precision** | 69.73% |
| **Recall** | 76.22% |
| **mAP50** | 79.81% |
| **mAP50-95** | 63.58% |

## 📁 Proje Yapısı

```
project_5/
├── train.py                    # Model eğitim scripti
├── test.py                     # Model test scripti
├── yolov8n.pt                  # YOLOv8 nano ön-eğitimli model
├── README.md                   # Proje dokümantasyonu
├── traffic-sign-detection/     # Veri seti klasörü
│   ├── data.yaml              # Veri seti konfigürasyonu
│   ├── train/                 # Eğitim verileri
│   ├── valid/                 # Doğrulama verileri
│   └── test/                  # Test verileri
└── runs/                       # Eğitim sonuçları
    └── detect/
        └── traffic-sign-model/
            ├── weights/        # Eğitilmiş model ağırlıkları
            │   └── best.pt    # En iyi model
            └── results.csv     # Eğitim metrikleri
```

## 🧠 YOLOv8 Nedir?

**YOLO (You Only Look Once)**, bir görsel veya video karesi üzerindeki nesneleri tek bir seferde tarayarak hem türlerini (sınıflandırma) hem de konumlarını (koordinat belirleme) anlık olarak tespit eden, derin öğrenme tabanlı bir algoritmadır.

### Avantajları:
- 🚀 İnanılmaz hızlı işlem
- 🎯 Yüksek doğruluk oranı
- 🔄 Gerçek zamanlı çalışma
- 🎨 Kolay entegrasyon

## 🔧 Hyperparameter Ayarları

`train.py` dosyasında aşağıdaki parametreleri özelleştirebilirsiniz:

```python
epochs = 2              # Eğitim döngüsü sayısı
imgsz = 640            # Resim boyutu
batch = 16             # Batch size
lr0 = 0.01             # Başlangıç öğrenme oranı
optimizer = "SGD"      # Optimizasyon algoritması
weight_decay = 0.0005  # Ağırlık cezası (overfitting önleme)
momentum = 0.935       # Momentum değeri
patience = 50          # Erken durdurma sabır değeri
device = "cpu"         # cpu veya cuda
```

## 📸 Örnek Kullanım

```python
from ultralytics import YOLO

# Modeli yükle
model = YOLO("runs/detect/traffic-sign-model/weights/best.pt")

# Tespit yap
results = model("image.jpg")

# Sonuçları göster
results[0].show()
```

## 🤝 Katkıda Bulunma

İyileştirme önerileri ve katkılarınızı bekliyoruz!

## 📞 İletişim

Sorularınız ve önerileriniz için issue açabilirsiniz.

---

⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!
