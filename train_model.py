# --- SCRIPT TRAINING BAHASA ISYARAT TANGAN ---
# Script ini digunakan untuk melatih model CNN menggunakan dataset kustom Anda
# (dari screenshot) dan menggabungkannya dengan dataset CSV (jika ada).

import cv2
import numpy as np
import os
import sys
import time
import csv

# Gunakan JAX backend untuk Keras agar cocok dengan script deteksi real-time
os.environ['KERAS_BACKEND'] = 'jax'
from keras import models, layers

print("\n" + "="*70)
print("[OK] SCRIPT TRAINING MODEL BAHASA ISYARAT")
print("="*70 + "\n")

# --- 1. Konfigurasi ---
dataset_dir = "dataset"
train_csv_path = 'sign_mnist_train.csv'
test_csv_path = 'sign_mnist_test.csv'
model_output_path = 'sign_language_cnn_model.h5'

# Hardcode alfabet dan angka sesuai folder di dataset (35 kelas)
alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
char_to_label = {char: idx for idx, char in enumerate(alphabet)}
num_classes = len(alphabet)  # 35 kelas

# --- 2. Helper Functions (Menggunakan Numpy & Standard Library agar bebas dependensi luar) ---
def train_test_split_numpy(X, y, test_size=0.2, random_seed=42):
    """
    Membagi dataset menjadi set training dan validation menggunakan numpy
    """
    np.random.seed(random_seed)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    split_idx = int(len(X) * (1 - test_size))
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

def to_categorical_numpy(y, num_classes=25):
    """
    Mengonversi label numerik menjadi format One-Hot Encoding
    """
    one_hot = np.zeros((len(y), num_classes), dtype=np.float32)
    one_hot[np.arange(len(y)), y] = 1.0
    return one_hot

# --- 3. Memuat Dataset Kustom (Screenshot) ---
def load_custom_dataset():
    X = []
    y = []
    
    if not os.path.exists(dataset_dir):
        print(f"[INFO] Folder '{dataset_dir}' tidak ditemukan. Membuat folder baru...")
        os.makedirs(dataset_dir, exist_ok=True)
        for char in alphabet:
            os.makedirs(os.path.join(dataset_dir, char), exist_ok=True)
        return None, None

    print("[INFO] Memindai gambar di folder dataset kustom...")
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
                # Load gambar dalam format grayscale
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                # Resize ke 28x28 (ukuran input model CNN)
                img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
                # Normalisasi nilai piksel (0 - 1)
                img_normalized = img_resized / 255.0
                
                X.append(img_normalized)
                y.append(label)
                total_loaded += 1
            except Exception as e:
                print(f"[WARN] Gagal memuat gambar {file_path}: {e}")

    if total_loaded == 0:
        print("[INFO] Tidak ada foto kustom (.jpg/.png) yang ditemukan di folder dataset.")
        return None, None
        
    print(f"[OK] Berhasil memuat {total_loaded} foto kustom dari folder dataset.")
    
    # Konversi ke numpy array dan tambahkan dimensi channel (28, 28, 1)
    X = np.array(X, dtype=np.float32)
    X = np.expand_dims(X, axis=-1)
    y = np.array(y, dtype=np.int32)
    return X, y

# --- 4. Memuat Dataset CSV menggunakan built-in csv module ---
def load_csv_dataset(csv_path):
    if not os.path.exists(csv_path):
        return None, None
        
    try:
        print(f"[INFO] Memuat data dari {csv_path}...")
        X = []
        y = []
        with open(csv_path, mode='r') as f:
            reader = csv.reader(f)
            # Skip header
            header = next(reader)
            
            for row in reader:
                if len(row) == 0:
                    continue
                # Kolom pertama adalah label
                label = int(row[0])
                # Kolom sisanya adalah piksel gambar 28x28
                pixels = np.array(row[1:], dtype=np.float32).reshape(28, 28, 1) / 255.0
                
                X.append(pixels)
                y.append(label)
                
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)
    except Exception as e:
        print(f"[WARN] Gagal memuat file CSV {csv_path}: {e}")
        return None, None

# --- 5. Persiapan Seluruh Data ---
X_custom, y_custom = load_custom_dataset()

# Hanya muat dataset CSV bawaan jika melatih model 25 kelas (Sign MNIST asli)
if num_classes == 25:
    X_csv_train, y_csv_train = load_csv_dataset(train_csv_path)
    X_csv_test, y_csv_test = load_csv_dataset(test_csv_path)
else:
    # Untuk dataset kustom 36 kelas (A-Z + 0-9), lewati data CSV lama agar label tidak tertukar
    X_csv_train, y_csv_train = None, None
    X_csv_test, y_csv_test = None, None

# Gabungkan data training
X_train_list = []
y_train_list = []

# Gabungkan data validasi
X_val_list = []
y_val_list = []

# Jika ada data kustom (screenshot), bagi 80/20 untuk train/val
if X_custom is not None:
    X_c_train, X_c_val, y_c_train, y_c_val = train_test_split_numpy(X_custom, y_custom, test_size=0.2)
    X_train_list.append(X_c_train)
    y_train_list.append(y_c_train)
    X_val_list.append(X_c_val)
    y_val_list.append(y_c_val)
    print(f"[INFO] Data Kustom untuk Training: {len(X_c_train)}, Validation: {len(X_c_val)}")

# Jika ada data CSV asli, gabungkan ke training & validation
if X_csv_train is not None:
    X_train_list.append(X_csv_train)
    y_train_list.append(y_csv_train)
if X_csv_test is not None:
    X_val_list.append(X_csv_test)
    y_val_list.append(y_csv_test)

# Validasi akhir ketersediaan data
if len(X_train_list) == 0:
    print("[ERROR] Tidak ada data latihan yang ditemukan sama sekali!")
    print("        Silakan ambil screenshot dulu lewat 'realtime_hand_recognition.py'")
    print("        atau letakkan file CSV dataset asli di folder ini.")
    sys.exit(1)

# Gabungkan semua data menggunakan numpy concatenate
X_train = np.concatenate(X_train_list, axis=0)
y_train = np.concatenate(y_train_list, axis=0)
X_val = np.concatenate(X_val_list, axis=0) if len(X_val_list) > 0 else X_train
y_val = np.concatenate(y_val_list, axis=0) if len(y_val_list) > 0 else y_train

print(f"\n[INFO] Total Dataset Akhir:")
print(f"       -> Gambar Latihan (Train): {X_train.shape[0]} sampel")
print(f"       -> Gambar Validasi (Val) : {X_val.shape[0]} sampel")

# Konversi label ke One-Hot Encoding
y_train_onehot = to_categorical_numpy(y_train, num_classes=num_classes)
y_val_onehot = to_categorical_numpy(y_val, num_classes=num_classes)

# --- 6. Membuat Model CNN ---
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
    layers.Dropout(0.5),  # Dropout untuk mencegah overfitting
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Tampilkan ringkasan model
model.summary()

# --- 7. Melatih Model ---
epochs = 35
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

# --- 8. Menyimpan Model ---
try:
    print(f"\n[INFO] Menyimpan model baru ke '{model_output_path}'...")
    model.save(model_output_path)
    print(f"[OK] Model baru berhasil disimpan! Sekarang Anda bisa langsung menggunakan")
    print(f"     'realtime_hand_recognition.py' dengan model baru ini.")
except Exception as e:
    print(f"[ERROR] Gagal menyimpan file model: {e}")

print("\n" + "="*70)
print("[OK] Selesai.")
print("="*70 + "\n")