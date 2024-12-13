import cv2
import tensorflow as tf
import numpy as np

# Konfigurasi parameter
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Load model yang sudah dilatih
model = tf.keras.models.load_model('model/final_model.keras')

# Ambil nama kelas dari model
class_indices = {0: "Cardboard", 1: "Food Organics", 2: "Glass", 3: "Metal", 4: "Paper", 5: "Plastic", 6: "Vegetation"}
# (Pastikan nama kelas sesuai dengan training Anda)

# Fungsi untuk memproses gambar
def preprocess_image(frame):
    img = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))  # Resize frame
    img = img / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Menambahkan dimensi batch
    return img

# Fungsi untuk prediksi dan menampilkan hasil
def classify_frame(frame):
    processed_img = preprocess_image(frame)
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions[0])
    class_name = class_indices[predicted_class]
    confidence = predictions[0][predicted_class] * 100
    return class_name, confidence

# Inisialisasi kamera
cap = cv2.VideoCapture(0)  # '0' adalah kamera default

if not cap.isOpened():
    print("Error: Kamera tidak dapat diakses.")
    exit()

print("Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Tidak dapat membaca frame dari kamera.")
        break

    # Klasifikasi frame
    class_name, confidence = classify_frame(frame)

    # Tampilkan hasil pada frame
    text = f"{class_name} ({confidence:.2f}%)"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Real-Time Classification", frame)

    # Keluar jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Lepaskan kamera dan tutup jendela
cap.release()
cv2.destroyAllWindows()
