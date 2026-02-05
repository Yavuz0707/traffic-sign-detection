"""
YOLO ile Trafik İşareti Tanıma Modeli Eğitimi

eo sensor : kamera , trafik kuralları , trafik işaretlerinin tanınması
otonom aracin en önemli görevi : cevreyi tanımak (görsel algılama) (trafik işaretleri, şeritler, yayalar, diğer araçlar)

YOLO?  -> tek bir kez bakmak. real time çalışır. hızlıdır.
plan propgram? -> veri bulma, yükleme, train etme, test etme, model kaydetme

kütüphanelerin kurulması ,içeriye aktarılması

ultralytics YOLO kütüphanesi : pip install ultralytics

YOLO (You Only Look Once), bir görsel veya video karesi üzerindeki nesneleri tek bir seferde tarayarak hem türlerini
 (sınıflandırma) hem de konumlarını (koordinat belirleme) anlık olarak tespit eden,
 derin öğrenme tabanlı bir algoritmadır. Geleneksel yöntemlerin aksine görüntüyü
 parçalara bölmek yerine bütüncül bir yaklaşımla analiz ettiği için inanılmaz hızlı çalışır; bu da onu otonom araçlar,
 güvenlik sistemleri ve üretim hatları gibi gerçek zamanlı hız gerektiren projelerin vazgeçilmez standardı haline getirmiştir.

"""

from ultralytics import YOLO

# model seç -> Yolov8 nano

model = YOLO("yolov8n.pt")  # önceden eğitilmiş model

# model eğitimi

model.train(
    data = "traffic-sign-detection/data.yaml", # yaml dosyasında veri yolları ve sınıf sayıları tanımalandı
    epochs = 2, # eğitim döngüsü sayısı
    imgsz = 640, # resim boyutu
    batch = 16, # batch size 16 resim de bir ben ne öğrendim diye bakar
    name = "traffic-sign-model", # kaydedilecek model ismi
    lr0 = 0.01, # başlangıçta ki öğrenme oranı
    optimizer = "SGD", # optimizasyon algoritması
    weight_decay = 0.0005, # ağırlık cezası (overfitting önleme amaçlı)
    momentum = 0.935, # momentum değeri SGD için
    patience = 50, # erken durdurma için sabır değeri
    workers = 2, # veri yükleme için işçi sayısı hızlandırma amaçlı
    device = "cpu", # eğitim için cihaz seçimi (cpu veya cuda)
    save = True, # eğitilen modeli kaydetme
    save_period = 1, # kaç epochta bir kaydetme
    val = True, # her epoch sonunda doğrulama yapma
    verbose = True # eğitim sürecini detaylı gösterme

)