# --- SCRIPT TRAINING BAHASA ISYARAT TANGAN ---
# Script ini digunakan untuk melatih model CNN menggunakan dataset kustom Anda
# yang dikumpulkan lewat aplikasi (folder dataset/).

import cv2
import numpy as np
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Gunakan JAX backend untuk Keras agar konsisten dengan app.py
os.environ['KERAS_BACKEND'] = 'jax'
import keras
from keras import models, layers, callbacks, optimizers, regularizers

from dataset_utils import (
    audit_dataset,
    augment_batch,
    preprocess_gray_for_cnn,
    save_audit_report,
)

print("\n" + "="*70)
print("[OK] SCRIPT TRAINING MODEL BAHASA ISYARAT")
print("="*70 + "\n")

# --- 1. Konfigurasi ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(BASE_DIR, "dataset")
models_dir = os.path.join(BASE_DIR, "models")
model_output_path = os.path.join(models_dir, "sign_language_cnn_model.h5")
evaluation_dir = os.path.join(BASE_DIR, "evaluation_results")

# Ukuran input model CNN (wajib 64x64 — harus sama dengan app.py)
IMG_SIZE = 64

# Hardcode alfabet dan angka sesuai folder di dataset (36 kelas)
alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
char_to_label = {char: idx for idx, char in enumerate(alphabet)}
label_to_char = {idx: char for char, idx in char_to_label.items()}
num_classes = len(alphabet)

# --- 2. Helper Functions (Menggunakan Numpy & Standard Library agar bebas dependensi luar) ---
def train_test_split_numpy(X, y, test_size=0.2, random_seed=42):
    """Stratified split — tiap kelas proporsional di train & val."""
    np.random.seed(random_seed)
    train_idx, val_idx = [], []
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        np.random.shuffle(idx)
        n_val = max(1, int(round(len(idx) * test_size)))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    train_idx = np.array(train_idx, dtype=np.int32)
    val_idx = np.array(val_idx, dtype=np.int32)
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


class SignBatchDataset(keras.utils.PyDataset):
    """Batch loader; augmentasi hanya saat training."""

    def __init__(self, X, y_onehot, batch_size, augment=False):
        super().__init__()
        self.X = X
        self.y = y_onehot
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.X))
        batch_x = self.X[start:end]
        batch_y = self.y[start:end]
        if self.augment and np.random.rand() < 0.35:
            batch_x = augment_batch(batch_x)
        return batch_x, batch_y

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

bad_files = set()


def load_custom_dataset(skip_bad=True):
    X = []
    y = []
    skipped = 0

    if not os.path.exists(dataset_dir):
        print(f"[INFO] Folder '{dataset_dir}' tidak ditemukan. Membuat folder baru...")
        os.makedirs(dataset_dir, exist_ok=True)
        for char in alphabet:
            os.makedirs(os.path.join(dataset_dir, char), exist_ok=True)
        return None, None

    print("[INFO] Memuat gambar (center + resize 64x64, sama dengan app.py)...")
    total_loaded = 0

    for char in alphabet:
        char_dir = os.path.join(dataset_dir, char)
        if not os.path.isdir(char_dir):
            continue

        label = char_to_label[char]
        files = [
            f for f in os.listdir(char_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        for file_name in files:
            if skip_bad and (char, file_name) in bad_files:
                skipped += 1
                continue
            file_path = os.path.join(char_dir, file_name)
            try:
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img_normalized = preprocess_gray_for_cnn(img, IMG_SIZE)
                if img_normalized is None:
                    continue
                X.append(img_normalized)
                y.append(label)
                total_loaded += 1
            except Exception as e:
                print(f"[WARN] Gagal memuat gambar {file_path}: {e}")

    if total_loaded == 0:
        print("[INFO] Tidak ada foto valid yang bisa dimuat dari folder dataset.")
        return None, None

    print(f"[OK] {total_loaded} foto dimuat ({skipped} bermasalah diskip).")
    X = np.array(X, dtype=np.float32)
    X = np.expand_dims(X, axis=-1)
    y = np.array(y, dtype=np.int32)
    return X, y

def build_cnn_model():
    """CNN 64x64 — augmentasi di luar graph agar evaluasi konsisten."""

    def conv_block(x, filters, drop_rate=0.25):
        x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(drop_rate)(x)
        return x

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = conv_block(inputs, 32, 0.2)
    x = conv_block(x, 64, 0.3)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.45)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.08),
        metrics=["accuracy"],
    )
    return model


def main():
    global bad_files
    os.makedirs(evaluation_dir, exist_ok=True)

    print("[INFO] Mengaudit kualitas dataset...")
    audit_summary = audit_dataset(dataset_dir, alphabet, IMG_SIZE)
    audit_path = os.path.join(evaluation_dir, "dataset_audit.txt")
    save_audit_report(audit_summary, audit_path)
    print(
        f"[OK] Audit: {audit_summary['bad_images']} bermasalah "
        f"dari {audit_summary['total_images']} gambar -> {audit_path}"
    )
    bad_files = {(b["char"], b["file"]) for b in audit_summary["bad_files"]}

    X_custom, y_custom = load_custom_dataset()
    if X_custom is None or len(X_custom) == 0:
        print("[ERROR] Tidak ada data latihan yang ditemukan di folder dataset/!")
        print("        Kumpulkan dulu foto isyarat lewat aplikasi (mode 'Latih Data' di app.py).")
        sys.exit(1)

    X_train, X_val, y_train, y_val = train_test_split_numpy(X_custom, y_custom, test_size=0.2)
    print(f"[INFO] Data untuk Training: {len(X_train)}, Validation: {len(X_val)}")
    print(f"       Train: {X_train.shape[0]} sampel | Val: {X_val.shape[0]} sampel")

    y_train_onehot = to_categorical_numpy(y_train, num_classes=num_classes)
    y_val_onehot = to_categorical_numpy(y_val, num_classes=num_classes)

    print("\n[INFO] Membangun arsitektur model CNN (input 64x64)...")
    model = build_cnn_model()
    model.summary()

    epochs = 60
    batch_size = 32
    best_weights_path = os.path.join(evaluation_dir, "best_model.weights.h5")

    training_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            best_weights_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
    ]

    train_ds = SignBatchDataset(X_train, y_train_onehot, batch_size, augment=True)
    val_ds = SignBatchDataset(X_val, y_val_onehot, batch_size, augment=False)

    print(f"\n[INFO] Memulai training model maksimal {epochs} epochs (early stopping aktif)...")
    try:
        history = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=training_callbacks,
        )
        print("[OK] Pelatihan model selesai.")
    except Exception as e:
        print(f"[ERROR] Terjadi kegagalan saat melatih model: {e}")
        sys.exit(1)

    print("\n[INFO] Menjalankan evaluasi model (gambar tanpa augmentasi)...")
    train_eval = evaluate_model(model, X_train, y_train, "Training Set")
    val_eval = evaluate_model(model, X_val, y_val, "Validation Set")

    confusion_matrix_path = os.path.join(evaluation_dir, "confusion_matrix.png")
    history_plot_path = os.path.join(evaluation_dir, "training_history.png")
    report_path = os.path.join(evaluation_dir, "evaluation_report.txt")

    save_confusion_matrix_plot(val_eval["cm"], val_eval["active_labels"], confusion_matrix_path)
    save_training_history_plot(history, history_plot_path)
    save_evaluation_report(train_eval, val_eval, history, report_path)

    print(f"\n[OK] Hasil evaluasi disimpan:")
    print(f"     - {report_path}")
    print(f"     - {confusion_matrix_path}")
    print(f"     - {history_plot_path}")

    try:
        os.makedirs(models_dir, exist_ok=True)
        print(f"\n[INFO] Menyimpan model baru ke '{model_output_path}'...")
        model.save(model_output_path)
        print("[OK] Model baru berhasil disimpan. Jalankan app.py untuk pakai model ini.")
    except Exception as e:
        print(f"[ERROR] Gagal menyimpan file model: {e}")

    print("\n" + "=" * 70)
    print("[OK] Selesai.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()