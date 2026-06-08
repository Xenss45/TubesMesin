# --- REAL-TIME HAND SIGN RECOGNITION & DATASET COLLECTOR ---
# Menggunakan Kotak ROI Tetap untuk deteksi yang 100% stabil dan bebas error library
# Menggunakan Keras CNN model untuk prediksi real-time

import cv2
import numpy as np
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# Set Keras backend sebelum import
os.environ['KERAS_BACKEND'] = 'jax'
from keras import models

print("\n" + "="*70)
print("[OK] REAL-TIME HAND SIGN RECOGNITION & DATASET COLLECTOR (FIXED ROI)")
print("="*70 + "\n")

# --- 1. Konfigurasi & Inisialisasi ---
model_path = 'sign_language_cnn_model.h5'
dataset_dir = "dataset"

# Hardcode alfabet dan angka sesuai folder di dataset (36 kelas)
alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
label_map = {i: char for i, char in enumerate(alphabet)}
reverse_label_map = {char: i for i, char in enumerate(alphabet)}

# Muat Model
if not os.path.exists(model_path):
    print(f"[ERROR] Model file '{model_path}' tidak ditemukan!")
    sys.exit(1)

try:
    model = models.load_model(model_path)
    print("[OK] Model CNN berhasil dimuat.")
except Exception as e:
    print(f"[ERROR] Error memuat model: {e}")
    sys.exit(1)

# Inisialisasi Folder Dataset
dataset_dir = "dataset"
os.makedirs(dataset_dir, exist_ok=True)
for char in alphabet:
    os.makedirs(os.path.join(dataset_dir, char), exist_ok=True)

# Hitung jumlah file gambar di setiap folder saat awal aplikasi berjalan
def get_dataset_counts():
    counts = {}
    for char in alphabet:
        char_dir = os.path.join(dataset_dir, char)
        counts[char] = len([f for f in os.listdir(char_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    return counts

dataset_counts = get_dataset_counts()
active_letter_idx = 0  # Mulai dari huruf 'A'

# --- 2. Helper Functions ---
def segment_hand_skin(roi):
    """
    Mengubah background ROI menjadi hitam pekat menggunakan filter warna kulit (HSV)
    agar sesuai dengan gambar pada dataset training (yang berbackground hitam).
    """
    try:
        # Convert BGR ke HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Range warna kulit yang lebih toleran (S min 20, V min 40 untuk mengatasi bayangan/cahaya redup)
        lower_skin1 = np.array([0, 20, 40], dtype=np.uint8)
        upper_skin1 = np.array([25, 255, 255], dtype=np.uint8)
        
        lower_skin2 = np.array([160, 20, 40], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        
        # Buat mask warna kulit
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        # Operasi morfologi untuk menutup lubang kecil pada deteksi tangan
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Blackout background (selain warna kulit diubah menjadi hitam)
        segmented_roi = cv2.bitwise_and(roi, roi, mask=skin_mask)
        return segmented_roi
    except:
        return roi

def get_fixed_roi(frame, box_size=240):
    """
    Mengambil koordinat dan memotong area kotak ROI di tengah-kanan frame
    """
    h, w = frame.shape[:2]
    
    # Tempatkan kotak agak ke kanan agar pengguna mudah memposisikan tangan kanan
    x_min = int(w * 0.55) - int(box_size * 0.5)
    y_min = int(h * 0.5) - int(box_size * 0.5)
    x_max = x_min + box_size
    y_max = y_min + box_size
    
    # Validasi batas frame
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(w, x_max), min(h, y_max)
    
    roi = frame[y_min:y_max, x_min:x_max]
    return roi, (x_min, y_min, x_max, y_max)

def preprocess_for_prediction(roi):
    """
    Ubah ROI menjadi format 28x28 grayscale dengan background hitam pekat
    """
    try:
        # 1. Segmentasikan warna kulit agar background di luar tangan menjadi hitam pekat
        segmented_roi = segment_hand_skin(roi)
        
        # 2. Konversi ke Grayscale
        gray = cv2.cvtColor(segmented_roi, cv2.COLOR_BGR2GRAY)
        
        # 3. Resize ke 28x28 piksel
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        
        # 4. Normalisasi
        normalized = resized / 255.0
        
        # Tambah dimensi batch dan channel -> (1, 28, 28, 1)
        input_image = np.expand_dims(normalized, axis=-1)
        input_image = np.expand_dims(input_image, axis=0)
        return input_image, segmented_roi
    except:
        return None, roi

# --- 3. Main Loop Kamera ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Kamera tidak dapat dibuka.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Mode Mirror (default: False agar arah tangan sesuai dengan dataset asli)
mirror_mode = False

print("\nKONTROL:")
print("  'q'     - Keluar dari program")
print("  's'     - Simpan screenshot TANGAN BERSIH (ROI) ke folder huruf aktif")
print("  'm'     - Aktifkan/Nonaktifkan Mirror kamera")
print("  '['     - Ganti huruf target ke SEBELUMNYA")
print("  ']'     - Ganti huruf target ke BERIKUTNYA")
print("\nProgram berjalan. Silakan arahkan tangan Anda ke dalam kotak biru...\n")

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Gagal mengambil gambar dari kamera.")
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # Efek cermin (Mirror) jika diaktifkan
        if mirror_mode:
            frame = cv2.flip(frame, 1)
        
        # Duplikat frame bersih (tanpa kotak/annotasi) untuk keperluan cropping dataset
        clean_frame = frame.copy()
        
        # Ambil ROI berdasarkan kotak tetap
        roi, (x_min, y_min, x_max, y_max) = get_fixed_roi(frame, box_size=240)
        clean_roi, _ = get_fixed_roi(clean_frame, box_size=240)
        
        if roi.size > 0:
            # Dataset Anda menggunakan orientasi tangan di mana jempol menunjuk ke kanan.
            # Jika mirror_mode = False (kamera normal), jempol secara alami sudah di kanan, jadi jangan di-flip.
            # Jika mirror_mode = True (kamera mirror), jempol terbalik ke kiri di layar, jadi kita harus membaliknya (flip) agar menunjuk ke kanan.
            if mirror_mode:
                roi = cv2.flip(roi, 1)
                clean_roi = cv2.flip(clean_roi, 1)
            # --- PREDIKSI MODEL ---
            input_image, segmented_roi = preprocess_for_prediction(roi)
            if input_image is not None:
                # Cek jika ROI kosong (jumlah piksel kulit terlalu sedikit)
                gray_seg = cv2.cvtColor(segmented_roi, cv2.COLOR_BGR2GRAY)
                skin_pixel_count = cv2.countNonZero(gray_seg)
                
                # Threshold piksel kulit minimal untuk menyatakan ada tangan di dalam kotak ROI
                # Ukuran kotak ROI adalah 240x240 = 57,600 piksel. 1500 piksel adalah ~2.6% dari total area.
                MIN_SKIN_PIXELS = 1500
                
                if skin_pixel_count < MIN_SKIN_PIXELS:
                    # Gambar kotak biru dan katakan kosong
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)  # Biru default (BGR: 255, 0, 0)
                    cv2.putText(frame, "Kotak Kosong", (x_min, max(y_min - 12, 25)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
                else:
                    prediction = model.predict(input_image, verbose=0)
                    predicted_idx = np.argmax(prediction)
                    predicted_char = label_map.get(predicted_idx, "?")
                    confidence = prediction[0][predicted_idx] * 100
                    
                    # Warnai Bounding Box berdasarkan confidence
                    box_color = (0, 255, 0) if confidence > 60 else (0, 165, 255)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 3)
                    
                    # Gambar Teks Prediksi di atas kotak tangan
                    pred_text = f"Prediksi: {predicted_char} ({confidence:.0f}%)"
                    cv2.putText(frame, pred_text, (x_min, max(y_min - 12, 25)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2, cv2.LINE_AA)
            
            # Tampilkan versi preprocessed 28x28 grayscale di kanan atas layar sebagai verifikasi
            try:
                gray_resized = cv2.resize(cv2.cvtColor(segmented_roi, cv2.COLOR_BGR2GRAY), (100, 100), interpolation=cv2.INTER_NEAREST)
                gray_bgr = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
                frame[10:110, w-110:w-10] = gray_bgr
                cv2.rectangle(frame, (w-110, 10), (w-10, 110), (255, 255, 255), 1)
                cv2.putText(frame, "Input CNN", (w-110, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            except:
                pass
        
        # --- HUD (HEADS-UP DISPLAY) OVERLAY ---
        active_char = alphabet[active_letter_idx]
        
        # Latar belakang semi transparan untuk HUD di sebelah kiri
        cv2.rectangle(frame, (10, 10), (320, 170), (0, 0, 0), -1)
        # Tambahkan transparansi ringan
        cv2.addWeighted(frame, 0.8, frame, 0.2, 0)
        
        cv2.putText(frame, f"REKAM DATASET (BEBAS WAJAH)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Huruf Target: {active_char}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Jumlah Tersimpan: {dataset_counts[active_char]} foto", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Mirror Kamera: {'ON' if mirror_mode else 'OFF'} (Press 'm')", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if mirror_mode else (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Ganti Huruf: Press '[' atau ']'", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Simpan Gambar: Press 'S'", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)
        
        # Tampilkan petunjuk umum
        cv2.putText(frame, "DEKATKAN tangan hingga MEMENUHI kotak biru", (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Press 'q' untuk keluar", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        # Tampilkan Window utama
        cv2.imshow('Hand Sign Recognition & Dataset Collector', frame)
        
        # --- MENANGANI INPUT KEYBOARD ---
        key = cv2.waitKey(1) & 0xFF
        
        # 1. Keluar
        if key == ord('q'):
            print("\n[OK] Program dihentikan oleh user.")
            break
            
        # 2. Ganti huruf target ke berikutnya ']'
        elif key == ord(']'):
            active_letter_idx = (active_letter_idx + 1) % len(alphabet)
            print(f"  Ganti huruf target ke: {alphabet[active_letter_idx]}")
            
        # 3. Ganti huruf target ke sebelumnya '['
        elif key == ord('['):
            active_letter_idx = (active_letter_idx - 1) % len(alphabet)
            print(f"  Ganti huruf target ke: {alphabet[active_letter_idx]}")
            
        # 4. Toggle Mirror Mode 'm'
        elif key == ord('m'):
            mirror_mode = not mirror_mode
            print(f"  Mirror Kamera: {'ON' if mirror_mode else 'OFF'}")
            
        # 5. Ambil screenshot ROI bersih
        elif key == ord('s'):
            if clean_roi is None or clean_roi.size == 0:
                print("[ERROR] Gagal menyimpan: Kotak deteksi kosong!")
            else:
                # Resize potongan tangan menjadi 200x200 BGR agar seragam dan bersih
                try:
                    # Terapkan segmentasi warna kulit juga pada screenshot yang disimpan agar sama persis
                    roi_segmented = segment_hand_skin(clean_roi)
                    roi_resized = cv2.resize(roi_segmented, (200, 200), interpolation=cv2.INTER_AREA)
                    
                    # Buat file path
                    timestamp = int(time.time() * 1000)
                    active_char = alphabet[active_letter_idx]
                    filename = f"dataset/{active_char}/hand_{timestamp}.jpg"
                    
                    # Simpan gambar
                    cv2.imwrite(filename, roi_resized)
                    
                    # Update counter
                    dataset_counts[active_char] += 1
                    print(f"[OK] Berhasil menyimpan data latih: {filename} (Total: {dataset_counts[active_char]} foto)")
                except Exception as e:
                    print(f"[ERROR] Gagal menyimpan screenshot: {e}")

except KeyboardInterrupt:
    print("\n[OK] Program dihentikan (Ctrl+C).")

finally:
    # Bersihkan sumber daya
    cap.release()
    cv2.destroyAllWindows()
    print("\n" + "="*70)
    print("[OK] Selesai. Semua sumber daya kamera dilepas.")
    print("="*70 + "\n")