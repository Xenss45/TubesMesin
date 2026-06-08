# Hand Sign Classifier - Real-time Recognition

Program untuk mengenali isyarat tangan (hand sign) secara real-time menggunakan model CNN dan input dari kamera webcam.

## 🎯 Program Utama - YANG BARU DAN LEBIH BAIK

### ⭐ `hand_recognition_with_landmarks.py` - **RECOMMENDED**

Program terbaru dengan fitur:
- ✓ **Hand landmarks visualization** - Visualisasi garis-garis yang membentuk tangan (contour + convex hull)
- ✓ **Real-time hand detection** - Menggunakan OpenCV skin color detection
- ✓ **28x28 preprocessed image display** - Lihat gambar yang sebenarnya diproses oleh model
- ✓ **Top 3 predictions** - Lihat confidence untuk 3 pilihan teratas
- ✓ **Accurate ROI extraction** - Ekstrak dari tangan yang terdeteksi (bukan area tengah frame)

**Jalankan:**
```bash
python hand_recognition_with_landmarks.py
```

---

## Persyaratan

- Python 3.14.5 ✓
- Webcam/Kamera yang terhubung
- File model: `sign_language_cnn_model.h5` (sudah ada)

## Dependencies yang Sudah Terinstall

✓ keras - Model Keras dengan JAX backend
✓ opencv-python - Image processing dan kamera
✓ numpy - Array operations
✓ jax - Backend untuk Keras
✓ h5py - Load model dari file .h5

## 🎮 Cara Menggunakan Program Terbaru

### 1. Jalankan Program

```bash
python hand_recognition_with_landmarks.py
```

### 2. Kontrol

- **'q'** - Keluar dari program
- **'s'** - Menyimpan screenshot
- **Arahkan tangan ke kamera** untuk deteksi

### 3. Apa yang Akan Anda Lihat

**Panel Visualisasi:**
- 🟢 **Garis-garis hijau** = Contour tangan terdeteksi
- 🔵 **Lingkaran kuning** = Landmarks (titik-titik penting tangan)
- 🔷 **Garis biru** = Convex hull (bentuk keseluruhan tangan)
- 📦 **Kotak ROI** = Area yang akan diproses model

**Info Debug:**
- **Top 3 Predictions** - Confidence untuk 3 huruf teratas
- **28x28 Input Image** - Gambar yang sebenarnya diproses model
- **Confidence Score** - Kepercayaan prediksi

---

## ⚠️ PENTING: Kenapa Confidence Tinggi (100%) Tapi Prediksi Masih Error?

Ini adalah **masalah umum dalam machine learning** yang disebut **"Overfitting"** atau **"Domain Mismatch"**:

### Penyebab:
1. **Training vs Testing Distribution Mismatch**
   - Model dilatih dengan dataset tertentu (background, pose, lighting)
   - Real-world data berbeda dari training data
   - Confidence tinggi = model sangat yakin, tapi yakin dengan hal yang salah

2. **Preprocessing Perbedaan**
   - Training menggunakan image tertentu
   - Real-time menggunakan skin segmentation → bisa berbeda karakteristik

3. **Model Limitation**
   - Model 28x28 sangat terbatas untuk detail
   - Beberapa huruf mirip (misalnya 'O' vs '0')

### Solusi:

#### A. **Improve Data Input** (Paling Penting)
- ✓ Pastikan pencahayaan yang baik dan konsisten
- ✓ Latar belakang kontras dengan tangan
- ✓ Posisi tangan di tengah frame, stable
- ✓ Hindari pose ekstrem atau tangan miring

#### B. **Adjust Confidence Threshold**
- Hanya percaya prediksi jika `confidence > 70%`
- Gunakan Top 2-3 predictions, ambil voting

#### C. **Improve Preprocessing**
- Cek apakah 28x28 image sudah jelas
- Bisa adjust skin color range di kode untuk deteksi lebih baik

#### D. **Retrain Model** (Jika perlu)
- Kumpulkan lebih banyak data real-world
- Augmentasi data (rotasi, scaling, lighting changes)
- Gunakan data augmentation saat training

---

## 📊 Cara Memahami Visualisasi

### Contour & Landmarks

```
     Yellow Circles = Detected Landmarks (titik penting tangan)
     Green Line = Contour (outline tangan)
     Blue Line = Convex Hull (bentuk utama)
     
     ●───●───●
     │   │   │
     ●───◆───●  ← ◆ = Wrist/Center
     │   │   │
     ●───●───●
```

### Input Image (28x28)

- Grayscale image (hitam putih)
- Normalized (0-1 range)
- Inilah yang **sebenarnya** diproses model
- **Kualitas 28x28 sangat rendah** - ini keterbatasan model

---

## 🔍 Debug Tips

### Jika Hand Tidak Terdeteksi:
- Cek pencahayaan
- Pastikan tangan dalam frame
- Jangan terlalu dekat/jauh
- Coba gerakkan tangan perlahan

### Jika Top 3 Predictions Aneh:
- Model bingung
- Coba ubah pose tangan
- Perbaiki pencahayaan/background
- Kemungkinan besar input 28x28 tidak jelas

### Jika Confidence Tinggi Tapi Prediksi Salah:
- **Ini normal!** Model overfitting
- Lihat Top 2-3 pilihan, ambil yang paling masuk akal
- Atau gunakan voting dari multiple frames

---

## 📁 File Program

**Aktif/Recommended:**
- `hand_recognition_with_landmarks.py` - **⭐ MAIN PROGRAM (GUNAKAN INI)**

**Alternative/Lama:**
- `realtime_hand_recognition_final.py` - Versi simple (tanpa landmark)
- `realtime_hand_recognition.py` - Versi lama
- `realtime_hand_recognition_v2.py` - Versi lama (ada kompatibilitas issue)

**Supporting:**
- `sign_language_cnn_model.h5` - Model CNN yang sudah dilatih
- `untitled3.py` - File reference dari training

---

## 📝 Alphabet yang Dikenali

```
A B C D E F G H I K L M N O P Q R S T U V W X Y
```

**Catatan**: Huruf J dan Z tidak termasuk dalam model

---

## 🎬 Program Flow

```
1. Load Model Keras (JAX Backend)
   ↓
2. Initialize Camera (640x480, 30fps)
   ↓
3. LOOP Real-time:
   ├─ Capture frame
   ├─ Mirror (flip horizontal)
   ├─ DETECT HAND:
   │  ├─ Convert BGR → HSV
   │  ├─ Skin color thresholding
   │  ├─ Morphology (close, open)
   │  └─ Find largest contour
   │
   ├─ EXTRACT LANDMARKS:
   │  ├─ Approximate polygon dari contour
   │  └─ Draw contour + convex hull
   │
   ├─ EXTRACT ROI:
   │  ├─ Get bounding box dari contour
   │  └─ Add padding
   │
   ├─ PREPROCESS:
   │  ├─ Grayscale
   │  ├─ Resize → 28×28
   │  └─ Normalize (0-1)
   │
   ├─ PREDICT:
   │  ├─ Model inference
   │  ├─ Get top 3 predictions
   │  └─ Extract confidence
   │
   ├─ VISUALIZE:
   │  ├─ Draw contour & landmarks
   │  ├─ Display 28×28 preprocessed
   │  ├─ Show top 3 predictions
   │  └─ Draw bounding box
   │
   └─ HANDLE INPUT:
      ├─ 'q' → Quit
      ├─ 's' → Screenshot
      └─ Loop continue
```

---

## 🔧 Technologi yang Digunakan

- **Model**: CNN (Convolutional Neural Network)
- **Framework**: Keras 3.14.1 dengan JAX Backend
- **Hand Detection**: OpenCV Skin Color Segmentation
- **Features**: Contour + Convex Hull Approximation
- **Input**: 28×28 Grayscale Image
- **Output**: 24 Classes (A-Y)

---

## Informasi Model

- **Input**: 28×28 pixel grayscale image
- **Output**: 24 classes (alphabet A-Y, tanpa J dan Z)
- **Format**: Keras H5 (dengan weights dan optimizer)
- **Backend**: JAX (compatible dengan Python 3.14)

---

## 📌 Tips untuk Hasil Lebih Baik

| Kategori | Tips |
|----------|------|
| **Pencahayaan** | Gunakan pencahayaan terang dan merata, hindari shadow |
| **Background** | Latar belakang kontras dengan tangan (bukan warna skin) |
| **Posisi Tangan** | Letakkan di tengah frame, tangan membentang jelas |
| **Stabilitas** | Tahan tangan stabil, tidak bergetar-getar |
| **Jarak** | Jarak optimal ~30-50cm dari camera |
| **Pose** | Isyarat jelas dan terdefinisi, hindari pose ekstrem |
| **Kecepatan** | Lakukan isyarat dengan kecepatan normal (tidak terlalu cepat) |
| **Confidence** | Tunggu hingga confidence stabil >70% sebelum ambil hasil |

---

## Output

Program akan menyimpan screenshot ke file `hand_sign_XXXX.jpg` ketika Anda menekan 's'.

```
hand_sign_1.jpg
hand_sign_2.jpg
hand_sign_3.jpg
```

---

## Troubleshooting

### Error: "Model file not found"
- ✓ Pastikan `sign_language_cnn_model.h5` ada di folder yang sama

### Error: "Cannot open camera"
- ✓ Webcam terhubung baik
- ✓ Check permission akses kamera di Windows Settings
- ✓ Tutup program lain yang gunakan camera

### Hand tidak terdeteksi sama sekali
- ✓ Cek pencahayaan - terlalu gelap
- ✓ Latar belakang yang kontras (bukan warna kulit)
- ✓ Tangan harus dalam frame
- ✓ Adjust skin color range jika diperlukan (lihat kode HSV range)

### Program sangat lambat
- ✓ Normal untuk run pertama (Keras compile)
- ✓ Tunggu 10-20 detik
- ✓ Run kedua akan lebih cepat

### Prediksi tidak akurat
- ✓ Lihat 28×28 input image - apakah jelas?
- ✓ Cek Top 3 predictions - apakah hasil yang benar ada di top 3?
- ✓ Perbaiki lighting, background, dan pose
- ✓ Gunakan confidence threshold lebih tinggi

### Confidence 100% tapi prediksi salah
- ⚠️ **Ini normal!** Model overfitting
- ✓ Lihat Top 3 predictions, ambil voting
- ✓ Atau gunakan multiple frames untuk confirmation
- ✓ Improve input data quality

---

## 🎓 Machine Learning Insights

### Kenapa Model Bisa Confident Tapi Salah?

```python
# Contoh skenario:
# Model dilatih dengan data A
# Real-world data adalah B
# Model sangat familiar dengan A
# Tapi menemui B yang berbeda

# Model akan output: "100% confident it's A!"
# Padahal seharusnya B

# Ini bukan bug - ini karakteristik machine learning
# Solusi: Improve training data atau adjust confidence threshold
```

### Domain Adaptation Solutions:

1. **Data Augmentation** - Tambah variety training data
2. **Transfer Learning** - Gunakan pre-trained model, fine-tune
3. **Ensemble Methods** - Combine multiple models
4. **Confidence Calibration** - Adjust confidence scores
5. **Active Learning** - Collect hard examples dan retrain

---

## Notes

- Program menggunakan JAX backend (compatible Python 3.14)
- OpenCV Skin Color Detection lebih reliable daripada MediaPipe di Python 3.14
- Model akan compile pada run pertama (normal)
- Confidence score bukan 100% guarantee - lihat Top 3 predictions
- 28×28 resolution adalah keterbatasan model - tidak bisa capture detail complex

---

**Status**: ✓ Fully Functional dengan Hand Detection  
**Last Updated**: 27 May 2026  
**Python Version**: 3.14.5  
**Recommendation**: Gunakan `hand_recognition_with_landmarks.py` untuk hasil terbaik
