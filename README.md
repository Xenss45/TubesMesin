# BicaraIsyarat - Penerjemah Bahasa Isyarat Real-Time

Aplikasi web untuk mengenali isyarat tangan (huruf & angka) secara real-time dari webcam
menggunakan model CNN, lalu merangkainya menjadi kata dan kalimat yang rapi.

- Input model: **64x64** grayscale
- Jumlah kelas: **36** (`0-9` dan `a-z`)
- Backend: Python + Flask + Keras (JAX)
- Frontend: HTML/CSS/JS (tanpa framework)

---

## Persyaratan

- Python 3.11+
- Webcam
- File model `models/sign_language_cnn_model.h5` (dihasilkan oleh `train_model.py`)

Dependensi ada di `requirements.txt` (Flask, OpenCV, Keras, JAX, NumPy, Matplotlib,
google-generativeai, python-dotenv, dll).

---

## Instalasi

```powershell
# 1. Buat virtual environment
python -m venv venv

# 2. Aktifkan (PowerShell)
.\venv\Scripts\Activate.ps1
# Jika diblokir, jalankan sekali:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 3. Install dependensi
pip install -r requirements.txt
```

---

## Menjalankan Aplikasi

```powershell
.\venv\Scripts\python.exe app.py
```

Lalu buka browser ke **http://127.0.0.1:5001** (port 5001 — port 5000 sering tertinggal proses zombie di Windows)

Tekan `CTRL+C` di terminal untuk berhenti.

---

## Dua Mode

Aplikasi punya dua tab di bagian atas:

### 1. Terjemah
Mode utama untuk menerjemahkan isyarat menjadi kalimat:
- Arahkan tangan ke kotak ROI di tengah kamera.
- **Tahan** isyarat sampai progress penuh untuk mengetik huruf.
- Huruf yang stabil dirangkai menjadi kata, lalu kalimat.
- Pilihan **prediksi kata** muncul otomatis (tekan `1`/`2`/`3` untuk memilih).
- Tombol **Rangkai** menyusun kalimat akhir yang rapi.

Preset **Mode** (sensitivitas):
- **Cepat** - tahan ~0,8 dtk, ambang kecocokan lebih rendah
- **Normal** - tahan ~1,2 dtk (default)
- **Akurat** - tahan ~1,7 dtk, ambang kecocokan lebih tinggi

### 2. Latih Data
Mode untuk mengumpulkan dataset:
- Pilih huruf target (dropdown atau tombol panah / `[` `]`).
- Tekan **Simpan Gambar** (atau `S`) untuk menyimpan foto ROI ke `dataset/<huruf>/`.
- Statistik jumlah foto per kelas tampil di bawah.

---

## Pintasan Keyboard

| Tombol | Fungsi |
|--------|--------|
| `S` | Simpan gambar ke dataset (mode Latih Data) |
| `[` `]` | Ganti huruf target |
| `Space` | Akhiri kata (spasi) |
| `Backspace` | Hapus huruf / kata terakhir |
| `1` `2` `3` | Pilih prediksi kata |
| `Enter` | Rangkai kalimat |
| `Esc` | Reset |

---

## Melatih Ulang Model

Setelah mengumpulkan data di folder `dataset/`:

```powershell
.\venv\Scripts\python.exe train_model.py
```

Proses training:
- Membaca semua foto di `dataset/<kelas>/`, resize ke 64x64 grayscale.
- Membagi data 80/20 (train/validation).
- Arsitektur CNN: 3 blok konvolusi (32-64-128) + BatchNorm + GlobalAveragePooling + Dense.
- Augmentasi: rotasi, translasi, zoom, kontras.
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint.
- Class weights otomatis untuk dataset tidak seimbang.

Hasil disimpan:
- `models/sign_language_cnn_model.h5` - model CNN siap dipakai `app.py`
- `evaluation_results/` - confusion matrix, grafik training, laporan evaluasi

> Penting: ukuran input model di `train_model.py` dan `app.py` (`IMG_SIZE = 64`) harus sama.
> Jika mengubahnya, latih ulang modelnya.

---

## Fitur AI Penyusun Kalimat (Opsional)

Penyusunan & koreksi kalimat berjalan dengan **sistem lokal** secara default
(koreksi typo + prediksi kata berbasis kamus).

Untuk hasil lebih pintar, aktifkan Gemini dengan mengisi `.env`:

```
GEMINI_API_KEY=isi_api_key_anda
```

Lalu restart `app.py`. Jika kosong, aplikasi tetap berjalan dengan sistem lokal.

---

## Struktur Proyek

```
TubesMesin/
├─ app.py                      # Web server Flask + prediksi + penyusun kalimat
├─ train_model.py              # Script training model CNN (64x64)
├─ hand_segmentation.py        # Masking tangan (MediaPipe)
├─ dataset_utils.py            # Preprocess + audit dataset
├─ models/                     # Semua model ML
│  ├─ sign_language_cnn_model.h5   # CNN bahasa isyarat (64x64)
│  └─ hand_landmarker.task         # MediaPipe hand landmark
├─ requirements.txt
├─ .env                        # GEMINI_API_KEY (opsional)
├─ dataset/                    # Foto latih per kelas (0-9, a-z)
├─ evaluation_results/         # Output evaluasi training
├─ templates/index.html        # Halaman web
└─ static/
   ├─ css/style.css
   └─ js/app.js
```

---

## Tips Akurasi

- Pencahayaan terang & merata, latar belakang kontras dengan warna kulit.
- Letakkan tangan di dalam kotak ROI, posisi stabil, jarak ~30-50 cm.
- Kumpulkan minimal 50-100 foto per huruf dengan variasi posisi/jarak/cahaya.
- Beri perhatian ekstra untuk huruf berpola mirip (E, O, 0, S).
- Gunakan tab "Latih Data" untuk menambah sampel, lalu latih ulang model.

---

## Troubleshooting

- **Kamera gagal diakses**: cek izin kamera browser & tutup aplikasi lain yang memakai kamera.
- **Prediksi selalu `?` / salah**: pastikan model sudah dilatih dengan input 64x64
  (jalankan `train_model.py`), dan perbaiki pencahayaan/posisi tangan.
- **`ModuleNotFoundError`**: pastikan venv aktif sebelum menjalankan
  (`pip install -r requirements.txt` di dalam venv).
