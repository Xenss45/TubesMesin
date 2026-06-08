# --- GOOGLE COLAB MODEL TRAINING SCRIPT ---
# Script ini dirancang khusus untuk dijalankan di Google Colab.
# Salin seluruh kode di bawah ini ke dalam cell kedua di Google Colab Anda.

import os
import cv2
import numpy as np
import sys
import time
import csv

# Gunakan JAX backend untuk Keras agar cocok dengan script deteksi real-time
os.environ['KERAS_BACKEND'] = 'jax'
from keras import models, layers

# Hubungkan ke Google Drive (jika belum dihubungkan di cell sebelumnya)
from google.colab import drive
try:
    drive.mount('/content/drive')
except Exception as e:
    print("[INFO] Google Drive mungkin sudah terhubung.")

print("\n" + "="*70)
print("[OK] SCRIPT TRAINING MODEL BAHASA ISYARAT DI GOOGLE COLAB")
print("="*70 + "\n")

# --- 1. Konfigurasi Jalur Google Drive ---
# Arahkan dataset ke penyimpanan lokal Colab hasil ekstrak (agar training cepat)
dataset_dir = "/content/dataset"

# Tetap simpan model hasil training (.h5) ke Google Drive agar aman dan permanen
model_output_path = "/content/drive/MyDrive/tubesmesin/sign_language_cnn_model.h5"


# Hardcode alfabet dan angka sesuai folder di dataset (35/36 kelas)
alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
char_to_label = {char: idx for idx, char in enumerate(alphabet)}
num_classes = len(alphabet)  # 36 kelas

# --- 2. Helper Functions (Sama seperti di versi lokal) ---
def train_test_split_numpy(X, y, test_size=0.2, random_seed=42):
    np.random.seed(random_seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(len(X) * (1 - test_size))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

def to_categorical_numpy(y, num_classes=25):
    one_hot = np.zeros((len(y), num_classes), dtype=np.float32)
    one_hot[np.arange(len(y)), y] = 1.0
    return one_hot

# --- 3. Memuat Dataset Kustom dari Google Drive ---
def load_custom_dataset():
    X = []
    y = []
    
    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Folder dataset '{dataset_dir}' tidak ditemukan di Google Drive!")
        print("Silakan ambil gambar terlebih dahulu lewat script webcam.")
        return None, None

    print("[INFO] Memindai gambar di folder dataset kustom di Google Drive...")
    total_loaded = 0
    
    for char in alphabet:
        char_dir = os.path.join(dataset_dir, char)
        if not os.path.isdir(char_dir):
            continue
            
        label = char_to_label[char]
        files = [f for f in os.listdir(char_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for file_name in files:
            file_path = os.path.join(char_dir, file_name)
            try:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
                img_normalized = img_resized / 255.0
                
                X.append(img_normalized)
                y.append(label)
                total_loaded += 1
            except Exception as e:
                print(f"[WARN] Gagal memuat gambar {file_path}: {e}")

    if total_loaded == 0:
        print("[ERROR] Tidak ada foto kustom (.jpg/.png) yang ditemukan di folder dataset Google Drive.")
        return None, None
        
    print(f"[OK] Berhasil memuat {total_loaded} foto kustom dari Google Drive.")
    
    X = np.array(X, dtype=np.float32)
    X = np.expand_dims(X, axis=-1)
    y = np.array(y, dtype=np.int32)
    return X, y

# --- 4. Persiapan Data ---
X_custom, y_custom = load_custom_dataset()

if X_custom is None:
    print("[ERROR] Tidak ada data latihan yang ditemukan sama sekali!")
    sys.exit(1)

# Bagi 80/20 untuk train/validation
X_train, X_val, y_train, y_val = train_test_split_numpy(X_custom, y_custom, test_size=0.2)

print(f"\n[INFO] Total Dataset Akhir:")
print(f"       -> Gambar Latihan (Train): {X_train.shape[0]} sampel")
print(f"       -> Gambar Validasi (Val) : {X_val.shape[0]} sampel")

# Konversi label ke One-Hot Encoding
y_train_onehot = to_categorical_numpy(y_train, num_classes=num_classes)
y_val_onehot = to_categorical_numpy(y_val, num_classes=num_classes)

# --- 5. Membuat Model CNN ---
print("\n[INFO] Membangun arsitektur model CNN...")
model = models.Sequential([
    # Input Layer
    layers.Input(shape=(28, 28, 1)),
    
    # Data Augmentation Layers (Hanya aktif selama training)
    layers.RandomRotation(factor=0.08, fill_mode='constant', fill_value=0.0),
    layers.RandomTranslation(height_factor=0.08, width_factor=0.08, fill_mode='constant', fill_value=0.0),
    layers.RandomZoom(height_factor=0.08, fill_mode='constant', fill_value=0.0),
    
    # CNN Layers
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 6. Melatih Model ---
epochs = 15
batch_size = 32

print(f"\n[INFO] Memulai training model selama {epochs} epochs...")
try:
    history = model.fit(
        X_train, y_train_onehot,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val_onehot)
    )
    print("[OK] Pelatihan model selesai.")
except Exception as e:
    print(f"[ERROR] Terjadi kegagalan saat melatih model: {e}")
    sys.exit(1)

# --- 7. Menyimpan Model ke Google Drive ---
try:
    print(f"\n[INFO] Menyimpan model baru ke Google Drive: '{model_output_path}'...")
    model.save(model_output_path)
    print(f"[OK] Model baru berhasil disimpan langsung ke Google Drive Anda!")
except Exception as e:
    print(f"[ERROR] Gagal menyimpan file model ke Google Drive: {e}")

print("\n" + "="*70)
print("[OK] Selesai.")
print("="*70 + "\n")
