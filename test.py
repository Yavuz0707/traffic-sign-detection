from ultralytics import YOLO
import cv2

# 1. Modeli yükleme
# Klasör yolunun doğruluğundan emin ol (runs/detect/...)
model = YOLO("runs/detect/traffic-sign-model/weights/best.pt") 

# 2. Test edilecek görselin yolu
image_path = "test1.jpg" 

# 3. Görseli yükleme ve model ile test etme
image = cv2.imread(image_path)
results = model(image_path)[0]  # Burada 'results' (çoğul) yaptık

# 4. Kutu çizimi 
for box in results.boxes:
    # Koordinatları alma
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # kutu koordinatları
    cls_id = int(box.cls[0])  # sınıf ID'si
    conf = box.conf[0].item()  # güven skoru
    label = model.names[cls_id]  # sınıf etiketi
    
    # Konsola bilgi yazdırma
    print(f"Sınıf: {label}, Güven: {conf:.2f}")
    
    # Görsel üzerine kutu çizimi
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Etiket ekleme
    cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) 

# 5. Sonucu gösterme ve kaydetme
cv2.imshow("Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("test_result.jpg", image)  # Sonucu kaydetme