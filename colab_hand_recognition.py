# --- GOOGLE COLAB WEBCAM REAL-TIME RECOGNITION & DATASET COLLECTOR ---
# Script ini dirancang khusus untuk dijalankan di Google Colab.
# Jangan jalankan script ini secara lokal di PC/laptop Anda.
# 
# Salin seluruh kode di bawah ini ke dalam satu cell di Google Colab,
# lalu jalankan cell tersebut.

import os
import cv2
import numpy as np
import time
from base64 import b64decode, b64encode
import PIL.Image
import io

# Pastikan backend Keras diatur sebelum import
os.environ['KERAS_BACKEND'] = 'jax'
from keras import models
from google.colab import drive
drive.mount('/content/drive')

# --- 1. Konfigurasi ---
model_path = '/content/drive/MyDrive/tubesmesin/sign_language_cnn_model.h5'
dataset_dir = "/content/drive/MyDrive/tubesmesin/dataset"
alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
label_map = {i: char for i, char in enumerate(alphabet)}

# Muat Model
model = None
if not os.path.exists(model_path):
    print(f"[ERROR] File model '{model_path}' tidak ditemukan di Google Colab!")
    print("Silakan upload file model '.h5' Anda atau latih model terlebih dahulu.")
else:
    try:
        model = models.load_model(model_path)
        print("[OK] Model CNN berhasil dimuat di Google Colab.")
    except Exception as e:
        print(f"[ERROR] Gagal memuat model: {e}")

# Buat folder dataset jika belum ada
os.makedirs(dataset_dir, exist_ok=True)
for char in alphabet:
    os.makedirs(os.path.join(dataset_dir, char), exist_ok=True)


# --- 2. Helper Functions (Sama seperti di versi lokal) ---
def segment_hand_skin(roi):
    """
    Mengubah background ROI menjadi hitam menggunakan filter warna kulit (HSV)
    """
    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_skin1 = np.array([0, 45, 60], dtype=np.uint8)
        upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        lower_skin2 = np.array([165, 45, 60], dtype=np.uint8)
        upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
        
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        segmented_roi = cv2.bitwise_and(roi, roi, mask=skin_mask)
        return segmented_roi
    except:
        return roi

def get_fixed_roi_coords(w, h, box_size=240):
    """
    Mendapatkan koordinat kotak ROI (Tengah-Kanan)
    """
    x_min = int(w * 0.55) - int(box_size * 0.5)
    y_min = int(h * 0.5) - int(box_size * 0.5)
    x_max = x_min + box_size
    y_max = y_min + box_size
    return max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max)

def preprocess_for_prediction(roi):
    """
    Preprocessing ROI tangan ke 28x28 grayscale
    """
    try:
        segmented_roi = segment_hand_skin(roi)
        gray = cv2.cvtColor(segmented_roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        normalized = resized / 255.0
        
        input_image = np.expand_dims(normalized, axis=-1)
        input_image = np.expand_dims(input_image, axis=0)
        return input_image, segmented_roi
    except:
        return None, roi


# --- 3. Google Colab & JavaScript Webcam Integration ---
from IPython.display import display, Javascript, HTML, JSON
from google.colab import output

# Helper to convert JS base64 image data to OpenCV image
def js_to_image(js_string):
    image_bytes = b64decode(js_string.split(',')[1])
    jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=cv2.IMREAD_COLOR)
    return img

# Registrasi fungsi Python agar bisa dipanggil dari JavaScript
def run_prediction_js(image_base64):
    """
    Fungsi ini dipanggil oleh JavaScript untuk setiap frame webcam.
    Menerima base64 gambar, memproses prediksi, dan mengembalikan hasil ke JS.
    """
    try:
        frame = js_to_image(image_base64)
        h, w = frame.shape[:2]
        
        # Koordinat ROI
        x_min, y_min, x_max, y_max = get_fixed_roi_coords(w, h)
        roi = frame[y_min:y_max, x_min:x_max]
        
        if roi.size > 0:
            # Preprocess & Prediksi
            input_image, segmented_roi = preprocess_for_prediction(roi)
            if input_image is not None and model is not None:
                # Cek jika ROI kosong (jumlah piksel kulit terlalu sedikit)
                gray_seg = cv2.cvtColor(segmented_roi, cv2.COLOR_BGR2GRAY)
                skin_pixel_count = cv2.countNonZero(gray_seg)
                
                # Threshold piksel kulit minimal untuk menyatakan ada tangan di dalam kotak ROI
                # Ukuran kotak ROI adalah 240x240 = 57,600 piksel. 1500 piksel adalah ~2.6% dari total area.
                MIN_SKIN_PIXELS = 1500
                
                if skin_pixel_count < MIN_SKIN_PIXELS:
                    gray_resized = cv2.resize(gray_seg, (100, 100), interpolation=cv2.INTER_NEAREST)
                    _, buffer = cv2.imencode('.png', gray_resized)
                    mini_roi_base64 = "data:image/png;base64," + b64encode(buffer).decode('utf-8')
                    return {
                        "char": "?",
                        "confidence": 0.0,
                        "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max,
                        "mini_roi": mini_roi_base64
                    }
                
                prediction = model.predict(input_image, verbose=0)
                predicted_idx = np.argmax(prediction)
                predicted_char = label_map.get(predicted_idx, "?")
                confidence = float(prediction[0][predicted_idx])
                
                # Buat versi mini 100x100 dari segmented_roi untuk dikirim ke JS (verifikasi input CNN)
                gray_resized = cv2.resize(gray_seg, (100, 100), interpolation=cv2.INTER_NEAREST)
                _, buffer = cv2.imencode('.png', gray_resized)
                mini_roi_base64 = "data:image/png;base64," + b64encode(buffer).decode('utf-8')
                
                return {
                    "char": predicted_char,
                    "confidence": confidence,
                    "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max,
                    "mini_roi": mini_roi_base64
                }
        return {"char": "?", "confidence": 0.0, "x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max, "mini_roi": ""}
    except Exception as e:
        return {"error": str(e)}

def save_dataset_js(image_base64, target_char):
    """
    Fungsi ini dipanggil oleh JavaScript saat tombol 'Capture' ditekan.
    Memotong gambar ROI, mensegmentasi kulit, dan menyimpannya ke folder dataset.
    """
    try:
        if target_char not in alphabet:
            return {"status": "error", "message": "Karakter tidak valid"}
            
        frame = js_to_image(image_base64)
        h, w = frame.shape[:2]
        
        x_min, y_min, x_max, y_max = get_fixed_roi_coords(w, h)
        roi = frame[y_min:y_max, x_min:x_max]
        
        if roi.size > 0:
            # Segmentasi warna kulit & resize ke 200x200 BGR
            roi_segmented = segment_hand_skin(roi)
            roi_resized = cv2.resize(roi_segmented, (200, 200), interpolation=cv2.INTER_AREA)
            
            timestamp = int(time.time() * 1000)
            target_folder = os.path.join(dataset_dir, target_char)
            filename = os.path.join(target_folder, f"hand_{timestamp}.jpg")
            cv2.imwrite(filename, roi_resized)
            
            # Hitung jumlah file
            count = len([f for f in os.listdir(target_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            return {"status": "ok", "message": f"Tersimpan di {filename} (Total: {count} foto)", "count": count}
        return {"status": "error", "message": "Kotak tangan kosong!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_dataset_counts_js():
    counts = {}
    for char in alphabet:
        char_dir = os.path.join(dataset_dir, char)
        if os.path.exists(char_dir):
            counts[char] = len([f for f in os.listdir(char_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        else:
            counts[char] = 0
    return counts

# Daftarkan fungsi ke kernel Colab
output.register_callback('notebook.run_prediction', run_prediction_js)
output.register_callback('notebook.save_dataset', save_dataset_js)
output.register_callback('notebook.get_dataset_counts', get_dataset_counts_js)


# --- 4. Main HTML & Javascript GUI untuk Web Browser ---
def start_webcam_stream():
    # Buat HTML dropdown target huruf
    options_html = "".join([f'<option value="{c}">{c.upper()}</option>' for c in alphabet])

    html_code = f"""
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; display: flex; flex-direction: column; align-items: center; background: #121212; color: #fff; padding: 20px; border-radius: 12px; max-width: 680px; margin: auto; box-shadow: 0 8px 24px rgba(0,0,0,0.5);">
        <h3 style="margin: 0 0 15px 0; color: #00e5ff; font-weight: 600;">Google Colab Real-time Recognition & Dataset Collector</h3>
        
        <!-- Controls (Dropdown dan Button disembunyikan/dikecilkan karena sekarang pakai full keyboard shortcut HUD) -->
        <div style="display: flex; gap: 15px; margin-bottom: 15px; align-items: center; background: #1e1e1e; padding: 10px 20px; border-radius: 8px; width: 90%; justify-content: space-between;">
            <div>
                <label for="colab-char-select" style="font-size: 14px; margin-right: 8px; color: #ccc;">Target Huruf:</label>
                <select id="colab-char-select" style="background: #333; color: #fff; border: 1px solid #555; padding: 6px 12px; border-radius: 4px; font-weight: bold; font-size: 15px;">
                    {options_html}
                </select>
            </div>
            <button id="colab-btn-capture" style="background: #00e5ff; color: #000; border: none; padding: 8px 16px; border-radius: 6px; font-weight: bold; cursor: pointer; transition: background 0.3s;">
                Capture Image (Press 'S')
            </button>
        </div>

        <!-- Video and Canvas area -->
        <div style="position: relative; width: 640px; height: 480px; background: #000; border-radius: 8px; overflow: hidden; border: 2px solid #333;">
            <video id="colab-video" width="640" height="480" autoplay playsinline style="position: absolute; left:0; top:0; transform: scaleX(-1);"></video>
            <canvas id="colab-canvas" width="640" height="480" style="position: absolute; left:0; top:0; z-index: 10;"></canvas>
            
            <!-- CNN Input preview box -->
            <div style="position: absolute; right: 10px; top: 10px; width: 100px; height: 100px; border: 2px solid #fff; background: #222; z-index: 20; border-radius: 4px; overflow:hidden;">
                <img id="colab-mini-roi" width="100" height="100" style="display:block;"/>
            </div>
            <div style="position: absolute; right: 10px; top: 115px; background: rgba(0,0,0,0.7); padding: 2px 6px; font-size: 10px; color: #fff; border-radius: 3px; z-index:20;">Input CNN</div>
        </div>

        <!-- Logs/Output -->
        <div id="colab-log" style="margin-top: 15px; font-size: 13px; color: #aaa; width: 90%; text-align: center; min-height: 20px;">
            Menyiapkan webcam... silakan beri izin kamera di browser Anda.
        </div>
    </div>
    """

    js_code = """
    <script>
    (async function() {
        const logDiv = document.getElementById('colab-log');
        try {
            const video = document.getElementById('colab-video');
            const canvas = document.getElementById('colab-canvas');
            const ctx = canvas.getContext('2d');
            const select = document.getElementById('colab-char-select');
            const btnCapture = document.getElementById('colab-btn-capture');
            const imgMiniRoi = document.getElementById('colab-mini-roi');

        const alphabet = "0123456789abcdefghijklmnopqrstuvwxyz";
        let activeLetterIdx = 10; // Default ke 'a'
        select.value = alphabet[activeLetterIdx];
        
        let datasetCounts = {};
        // Inisialisasi default dengan 0 agar tidak pernah undefined
        for (let c of alphabet) {
            datasetCounts[c] = 0;
        }

        // Ambil data jumlah foto dari Python saat pertama kali dimuat
        try {
            const countsResult = await google.colab.kernel.invokeFunction('notebook.get_dataset_counts', [], {});
            if (countsResult && countsResult.data && countsResult.data['application/json']) {
                datasetCounts = countsResult.data['application/json'];
            }
        } catch (e) {
            console.error("Error fetching counts:", e);
        }

        // Buka Webcam
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
            video.srcObject = stream;
            await video.play();
            logDiv.innerText = "Webcam aktif. Gunakan shortcut '[', ']', dan 'S'!";
        } catch (err) {
            logDiv.style.color = "#ff4444";
            logDiv.innerText = "Gagal mengakses webcam: " + err.message;
            return;
        }

        // Setup canvas tersembunyi untuk mengambil frame raw
        const captureCanvas = document.createElement('canvas');
        captureCanvas.width = 640;
        captureCanvas.height = 480;
        const captureCtx = captureCanvas.getContext('2d');

        // Fungsi capture untuk save dataset
        async function captureAndSave() {
            logDiv.style.color = "#00e5ff";
            logDiv.innerText = "Menyimpan gambar...";
            
            // Gambar frame ke canvas tersembunyi (karena video di-mirror di CSS, kita balik juga agar sesuai orientasi)
            captureCtx.save();
            captureCtx.translate(640, 0);
            captureCtx.scale(-1, 1);
            captureCtx.drawImage(video, 0, 0, 640, 480);
            captureCtx.restore();

            const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.85);
            const targetChar = alphabet[activeLetterIdx];

            // Panggil Python untuk menyimpan dataset
            const result = await google.colab.kernel.invokeFunction('notebook.save_dataset', [dataUrl, targetChar], {});
            const resData = result && result.data ? result.data['application/json'] : null;
            
            if (resData && resData.status === "ok") {
                logDiv.style.color = "#00ff66";
                logDiv.innerText = resData.message;
                // Update jumlah foto di cache JavaScript
                datasetCounts[targetChar] = resData.count;
            } else {
                logDiv.style.color = "#ff4444";
                logDiv.innerText = "Gagal menyimpan: " + (resData ? resData.message : "koneksi error");
            }
            setTimeout(() => { logDiv.style.color = "#aaa"; }, 2000);
        }

        // Listener dropdown diubah manual
        select.onchange = () => {
            activeLetterIdx = alphabet.indexOf(select.value);
        };

        // Listener tombol dan keyboard
        btnCapture.onclick = captureAndSave;
        
        document.addEventListener('keydown', async (e) => {
            const key = e.key.toLowerCase();
            if (key === 's') {
                await captureAndSave();
            } else if (e.key === ']') {
                activeLetterIdx = (activeLetterIdx + 1) % alphabet.length;
                select.value = alphabet[activeLetterIdx];
            } else if (e.key === '[') {
                activeLetterIdx = (activeLetterIdx - 1 + alphabet.length) % alphabet.length;
                select.value = alphabet[activeLetterIdx];
            }
        });

        let lastPrediction = null;
        let isPredicting = false;

        // Loop Prediksi Real-time (Asynchronous, Non-Blocking Render)
        async function predictionLoop() {
            try {
                // Hapus canvas overlay lama
                ctx.clearRect(0, 0, 640, 480);

                // --- 1. GAMBAR HUD KLASIK (Sama seperti versi OpenCV lama) ---
                ctx.fillStyle = "rgba(0, 0, 0, 0.75)";
                ctx.fillRect(10, 10, 310, 170);

                const activeChar = alphabet[activeLetterIdx];
                const savedCount = datasetCounts[activeChar] || 0;

                ctx.fillStyle = "#00e5ff"; // Judul cyan
                ctx.font = "bold 13px 'Courier New', Courier, monospace";
                ctx.fillText("REKAM DATASET (BEBAS WAJAH)", 20, 35);

                ctx.fillStyle = "#ffffff"; // Putih untuk nilai
                ctx.font = "bold 15px 'Courier New', Courier, monospace";
                ctx.fillText(`Huruf Target: ${activeChar.toUpperCase()}`, 20, 65);

                ctx.fillStyle = "#cccccc";
                ctx.font = "13px 'Courier New', Courier, monospace";
                ctx.fillText(`Jumlah Tersimpan: ${savedCount} foto`, 20, 90);
                ctx.fillText("Mirror Kamera: ON (Di-mirror otomatis)", 20, 110);
                ctx.fillText("Ganti Huruf: Tekan '[' atau ']'", 20, 130);
                ctx.fillText("Simpan Gambar: Tekan 'S'", 20, 150);

                // Petunjuk Umum di bawah
                ctx.fillStyle = "#00e5ff";
                ctx.font = "bold 14px 'Courier New', Courier, monospace";
                ctx.fillText("DEKATKAN tangan hingga MEMENUHI kotak biru", 10, 435);
                ctx.fillStyle = "#aaaaaa";
                ctx.font = "12px 'Courier New', Courier, monospace";
                ctx.fillText("Gunakan tombol Stop (kotak hitam) di cell Colab untuk berhenti", 10, 455);

                // --- 2. GAMBAR BOUNDING BOX (Kompensasi Mirror) ---
                // Tentukan koordinat ROI tetap default (mirrored):
                const defX = 168;
                const defY = 120;
                const defW = 240;
                const defH = 240;

                let boxColor = "#0055ff"; // Biru default jika tidak ada hasil prediksi valid
                let predText = "";

                if (lastPrediction && (Date.now() - lastPrediction.timestamp < 1000)) {
                    // Tentukan warna bounding box berdasarkan hasil prediksi terbaru
                    if (lastPrediction.confidence > 0.60) {
                        boxColor = "#00ff66"; // Hijau jika percaya diri
                    } else if (lastPrediction.confidence > 0.10) {
                        boxColor = "#ffaa00"; // Oranye jika ragu-ragu
                    }
                    
                    if (lastPrediction.confidence > 0.10) {
                        predText = `Prediksi: ${lastPrediction.char.toUpperCase()} (${Math.round(lastPrediction.confidence * 100)}%)`;
                    }
                }

                // Gambar Bounding Box
                ctx.strokeStyle = boxColor;
                ctx.lineWidth = 3;
                ctx.strokeRect(defX, defY, defW, defH);

                // Gambar Teks Prediksi di atas kotak jika ada
                if (predText) {
                    ctx.fillStyle = boxColor;
                    ctx.font = "bold 18px 'Courier New', Courier, monospace";
                    ctx.fillText(predText, defX, defY - 12);
                }

                // --- 3. PANGGIL PYTHON SECARA ASINKRON (Non-Blocking) ---
                if (!isPredicting && !video.paused && !video.ended) {
                    isPredicting = true;
                    
                    // Ambil frame raw
                    captureCtx.save();
                    captureCtx.translate(640, 0);
                    captureCtx.scale(-1, 1);
                    captureCtx.drawImage(video, 0, 0, 640, 480);
                    captureCtx.restore();
                    
                    const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.6);

                    google.colab.kernel.invokeFunction('notebook.run_prediction', [dataUrl], {})
                        .then(result => {
                            const resData = result && result.data ? result.data['application/json'] : null;
                            if (resData && !resData.error) {
                                lastPrediction = {
                                    char: resData.char,
                                    confidence: resData.confidence,
                                    timestamp: Date.now()
                                };
                                
                                // Update mini ROI preview
                                if (resData.mini_roi) {
                                    imgMiniRoi.src = resData.mini_roi;
                                }
                                
                                // Hapus teks error jika deteksi berhasil
                                if (logDiv.innerText.startsWith("Python Error:")) {
                                    logDiv.style.color = "#aaa";
                                    logDiv.innerText = "Webcam aktif. Gunakan shortcut '[', ']', dan 'S'!";
                                }
                            } else if (resData && resData.error) {
                                logDiv.style.color = "#ff4444";
                                logDiv.innerText = "Python Error: " + resData.error;
                            }
                            isPredicting = false;
                        })
                        .catch(err => {
                            console.error("Prediction error:", err);
                            logDiv.style.color = "#ff4444";
                            logDiv.innerText = "JS Promise Error: " + err.message;
                            isPredicting = false;
                        });
                }
            } catch (err) {
                console.error("HUD Draw error:", err);
                logDiv.style.color = "#ff4444";
                logDiv.innerText = "Error HUD: " + err.message;
            }

            // Jalankan frame berikutnya (kecepatan render HUD yang super responsif)
            setTimeout(predictionLoop, 50);
        }

        // Mulai loop prediksi di sini (setelah semua variabel & fungsi terdefinisi)
        predictionLoop();

        } catch (err) {
            logDiv.style.color = "#ff4444";
            logDiv.innerText = "JS Crash: " + err.message + "\\n" + err.stack;
        }
    })();
    </script>
    """
    display(HTML(html_code + js_code))

# Jalankan Webcam Stream di Colab
start_webcam_stream()
