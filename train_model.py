# --- SCRIPT TRAINING BAHASA ISYARAT TANGAN ---
# Script ini digunakan untuk melatih model CNN menggunakan dataset kustom Anda
# (dari screenshot) dan menggabungkannya dengan dataset CSV (jika ada).

import cv2
import numpy as np
import os
import sys
import time
import csv
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
evaluation_dir = 'evaluation_results'

# Hardcode alfabet dan angka sesuai folder di dataset (36 kelas)
alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
char_to_label = {char: idx for idx, char in enumerate(alphabet)}
label_to_char = {idx: char for char, idx in char_to_label.items()}
num_classes = len(alphabet)

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

def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[int(true_label), int(pred_label)] += 1
    return cm

def compute_classification_metrics(cm, active_labels):
    """Hitung precision, recall, F1 per kelas dari confusion matrix."""
    metrics = []
    total_correct = int(np.trace(cm))
    total_samples = int(cm.sum())

    for label_idx in active_labels:
        tp = cm[label_idx, label_idx]
        fp = cm[:, label_idx].sum() - tp
        fn = cm[label_idx, :].sum() - tp
        support = cm[label_idx, :].sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        metrics.append({
            'label': label_idx,
            'char': label_to_char[label_idx],
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': int(support),
        })

    macro_f1 = np.mean([m['f1'] for m in metrics]) if metrics else 0.0
    accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    return metrics, accuracy, macro_f1

def get_top_confused_pairs(cm, active_labels, top_n=10):
    pairs = []
    for true_label in active_labels:
        for pred_label in active_labels:
            if true_label == pred_label:
                continue
            count = cm[true_label, pred_label]
            if count > 0:
                pairs.append({
                    'true_char': label_to_char[true_label],
                    'pred_char': label_to_char[pred_label],
                    'count': int(count),
                })
    pairs.sort(key=lambda item: item['count'], reverse=True)
    return pairs[:top_n]

def predict_labels(model, X):
    probs = model.predict(X, verbose=0)
    return np.argmax(probs, axis=1)

def evaluate_model(model, X, y, dataset_name):
    print(f"\n{'=' * 70}")
    print(f"[EVAL] Evaluasi pada dataset: {dataset_name}")
    print(f"{'=' * 70}")

    y_pred = predict_labels(model, X)
    cm = confusion_matrix_np(y, y_pred, num_classes)
    active_labels = sorted(np.unique(np.concatenate([y, y_pred])).astype(int).tolist())
    class_metrics, accuracy, macro_f1 = compute_classification_metrics(cm, active_labels)
    confused_pairs = get_top_confused_pairs(cm, active_labels)

    print(f"  Accuracy : {accuracy * 100:.2f}%")
    print(f"  Macro F1 : {macro_f1 * 100:.2f}%")
    print(f"  Sampel   : {len(y)}")

    print("\n  Laporan per kelas (yang muncul di data ini):")
    print(f"  {'Kelas':<6} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Support':>8}")
    print(f"  {'-' * 40}")
    for item in class_metrics:
        print(
            f"  {item['char']:<6} "
            f"{item['precision'] * 100:>6.1f}% "
            f"{item['recall'] * 100:>6.1f}% "
            f"{item['f1'] * 100:>6.1f}% "
            f"{item['support']:>8}"
        )

    if confused_pairs:
        print("\n  Pasangan huruf yang paling sering tertukar:")
        for pair in confused_pairs:
            print(
                f"    '{pair['true_char']}' -> '{pair['pred_char']}' "
                f"({pair['count']} kali)"
            )

    return {
        'dataset_name': dataset_name,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'cm': cm,
        'active_labels': active_labels,
        'class_metrics': class_metrics,
        'confused_pairs': confused_pairs,
        'y_true': y,
        'y_pred': y_pred,
    }

def save_confusion_matrix_plot(cm, active_labels, save_path):
    labels = [label_to_char[idx] for idx in active_labels]
    sub_cm = cm[np.ix_(active_labels, active_labels)]

    fig_size = max(8, len(active_labels) * 0.45)
    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(sub_cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix (Validation)')
    plt.colorbar()
    tick_marks = np.arange(len(active_labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.xlabel('Prediksi')
    plt.ylabel('Label Sebenarnya')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_training_history_plot(history, save_path):
    history_dict = history.history
    epochs_ran = range(1, len(history_dict['loss']) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_ran, history_dict['loss'], label='Train Loss')
    if 'val_loss' in history_dict:
        plt.plot(epochs_ran, history_dict['val_loss'], label='Val Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_ran, history_dict['accuracy'], label='Train Accuracy')
    if 'val_accuracy' in history_dict:
        plt.plot(epochs_ran, history_dict['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_evaluation_report(train_eval, val_eval, history, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("LAPORAN EVALUASI MODEL BAHASA ISYARAT\n")
        f.write(f"Waktu evaluasi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history.get('val_accuracy', [0])[-1]
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history.get('val_loss', [0])[-1]

        f.write("RINGKASAN TRAINING TERAKHIR\n")
        f.write(f"  Train Accuracy (epoch terakhir): {final_train_acc * 100:.2f}%\n")
        f.write(f"  Val Accuracy   (epoch terakhir): {final_val_acc * 100:.2f}%\n")
        f.write(f"  Train Loss     (epoch terakhir): {final_train_loss:.4f}\n")
        f.write(f"  Val Loss       (epoch terakhir): {final_val_loss:.4f}\n\n")

        for eval_result in (train_eval, val_eval):
            f.write(f"{eval_result['dataset_name'].upper()}\n")
            f.write(f"  Accuracy : {eval_result['accuracy'] * 100:.2f}%\n")
            f.write(f"  Macro F1 : {eval_result['macro_f1'] * 100:.2f}%\n")
            f.write(f"  Sampel   : {len(eval_result['y_true'])}\n\n")

            f.write("  Per kelas:\n")
            f.write(f"  {'Kelas':<6} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Support':>8}\n")
            for item in eval_result['class_metrics']:
                f.write(
                    f"  {item['char']:<6} "
                    f"{item['precision'] * 100:>6.1f}% "
                    f"{item['recall'] * 100:>6.1f}% "
                    f"{item['f1'] * 100:>6.1f}% "
                    f"{item['support']:>8}\n"
                )

            if eval_result['confused_pairs']:
                f.write("\n  Pasangan huruf paling sering tertukar:\n")
                for pair in eval_result['confused_pairs']:
                    f.write(
                        f"    '{pair['true_char']}' -> '{pair['pred_char']}' "
                        f"({pair['count']} kali)\n"
                    )
            f.write("\n")

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

# --- 8. Evaluasi Model ---
os.makedirs(evaluation_dir, exist_ok=True)

print("\n[INFO] Menjalankan evaluasi model...")
train_eval = evaluate_model(model, X_train, y_train, "Training Set")
val_eval = evaluate_model(model, X_val, y_val, "Validation Set")

confusion_matrix_path = os.path.join(evaluation_dir, 'confusion_matrix.png')
history_plot_path = os.path.join(evaluation_dir, 'training_history.png')
report_path = os.path.join(evaluation_dir, 'evaluation_report.txt')

save_confusion_matrix_plot(val_eval['cm'], val_eval['active_labels'], confusion_matrix_path)
save_training_history_plot(history, history_plot_path)
save_evaluation_report(train_eval, val_eval, history, report_path)

print(f"\n[OK] Hasil evaluasi disimpan:")
print(f"     - {report_path}")
print(f"     - {confusion_matrix_path}")
print(f"     - {history_plot_path}")

# --- 9. Menyimpan Model ---
try:
    print(f"\n[INFO] Menyimpan model baru ke '{model_output_path}'...")
    model.save(model_output_path)
    print(f"[OK] Model baru berhasil disimpan! Sekarang Anda bisa langsung menggunakan")
    print(f"     'app.py' dengan model baru ini.")
except Exception as e:
    print(f"[ERROR] Gagal menyimpan file model: {e}")

print("\n" + "="*70)
print("[OK] Selesai.")
print("="*70 + "\n")