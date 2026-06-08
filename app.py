import os
import cv2
import numpy as np
import base64
import time
import sys
import warnings
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

warnings.filterwarnings('ignore')

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Set Keras backend sebelum import
os.environ['KERAS_BACKEND'] = 'jax'
from keras import models

app = Flask(__name__)
CORS(app)

model_path = 'sign_language_cnn_model.h5'
dataset_dir = "dataset"

# Hardcode alfabet dan angka sesuai folder di dataset (36 kelas)
alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
label_map = {i: char for i, char in enumerate(alphabet)}

# Load model secara global saat startup
model = None
if not os.path.exists(model_path):
    print(f"[ERROR] Model file '{model_path}' tidak ditemukan!")
else:
    try:
        model = models.load_model(model_path)
        print("[OK] Model CNN berhasil dimuat.")
    except Exception as e:
        print(f"[ERROR] Error memuat model: {e}")

# Helper to convert base64 image from frontend to OpenCV image
def base64_to_cv2(b64_string):
    try:
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
        img_data = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"[ERROR] Gagal memecahkan base64: {e}")
        return None

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

def get_fixed_roi_coords(w, h, box_size=240):
    """
    Mendapatkan koordinat area kotak ROI di tengah-kanan frame (seperti versi desktop)
    """
    x_min = int(w * 0.55) - int(box_size * 0.5)
    y_min = int(h * 0.5) - int(box_size * 0.5)
    x_max = x_min + box_size
    y_max = y_min + box_size
    
    # Validasi batas frame
    return max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max)

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
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return None, roi

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "Data gambar tidak dikirimkan"}), 400
        
        frame = base64_to_cv2(data['image'])
        if frame is None:
            return jsonify({"error": "Gagal membaca format base64"}), 400
        
        h, w = frame.shape[:2]
        x_min, y_min, x_max, y_max = get_fixed_roi_coords(w, h)
        
        # Potong ROI dari frame asli (sebelum di-mirror) agar posisinya pas dengan kotak di layar
        roi = frame[y_min:y_max, x_min:x_max]
        
        # Jika mirror_mode aktif, balikkan ROI secara horizontal agar sesuai representasi cermin
        mirror_mode = data.get('mirror_mode', True)
        if mirror_mode:
            roi = cv2.flip(roi, 1)
        
        if roi.size > 0:
            input_image, segmented_roi = preprocess_for_prediction(roi)
            if input_image is not None and model is not None:
                # Cek jika ROI kosong (jumlah piksel kulit terlalu sedikit)
                gray_seg = cv2.cvtColor(segmented_roi, cv2.COLOR_BGR2GRAY)
                skin_pixel_count = cv2.countNonZero(gray_seg)
                
                # Threshold piksel kulit minimal untuk menyatakan ada tangan di dalam kotak ROI
                # Ukuran kotak ROI adalah 240x240 = 57,600 piksel. 1500 piksel adalah ~2.6% dari total area.
                MIN_SKIN_PIXELS = 1500
                
                if skin_pixel_count < MIN_SKIN_PIXELS:
                    # Buat versi mini 100x100 grayscale untuk dikirim ke JS (verifikasi input CNN)
                    gray_resized = cv2.resize(gray_seg, (100, 100), interpolation=cv2.INTER_NEAREST)
                    _, buffer = cv2.imencode('.png', gray_resized)
                    mini_roi_base64 = "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')
                    
                    return jsonify({
                        "char": "?",
                        "confidence": 0.0,
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "mini_roi": mini_roi_base64
                    })
                
                prediction = model.predict(input_image, verbose=0)
                predicted_idx = np.argmax(prediction)
                predicted_char = label_map.get(predicted_idx, "?")
                confidence = float(prediction[0][predicted_idx])
                
                # Buat versi mini 100x100 grayscale untuk dikirim ke JS (verifikasi input CNN)
                gray_resized = cv2.resize(gray_seg, (100, 100), interpolation=cv2.INTER_NEAREST)
                _, buffer = cv2.imencode('.png', gray_resized)
                mini_roi_base64 = "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')
                
                return jsonify({
                    "char": predicted_char,
                    "confidence": confidence,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "mini_roi": mini_roi_base64
                })
        
        return jsonify({
            "char": "?",
            "confidence": 0.0,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "mini_roi": ""
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save_dataset', methods=['POST'])
def save_dataset():
    try:
        data = request.get_json()
        if not data or 'image' not in data or 'char' not in data:
            return jsonify({"error": "Parameter tidak lengkap"}), 400
        
        target_char = data['char'].lower()
        if target_char not in alphabet:
            return jsonify({"error": "Target karakter tidak valid"}), 400
            
        frame = base64_to_cv2(data['image'])
        if frame is None:
            return jsonify({"error": "Gagal membaca format base64"}), 400
            
        h, w = frame.shape[:2]
        x_min, y_min, x_max, y_max = get_fixed_roi_coords(w, h)
        
        # Potong ROI dari frame asli (sebelum di-mirror) agar posisinya pas dengan kotak di layar
        roi = frame[y_min:y_max, x_min:x_max]
        
        # Jika mirror_mode aktif, balikkan ROI secara horizontal agar sesuai representasi cermin
        mirror_mode = data.get('mirror_mode', True)
        if mirror_mode:
            roi = cv2.flip(roi, 1)
        
        if roi.size > 0:
            # Terapkan segmentasi warna kulit dan simpan dalam ukuran 200x200
            roi_segmented = segment_hand_skin(roi)
            roi_resized = cv2.resize(roi_segmented, (200, 200), interpolation=cv2.INTER_AREA)
            
            timestamp = int(time.time() * 1000)
            target_folder = os.path.join(dataset_dir, target_char)
            os.makedirs(target_folder, exist_ok=True)
            filename = os.path.join(target_folder, f"hand_{timestamp}.jpg")
            
            cv2.imwrite(filename, roi_resized)
            
            # Hitung jumlah file saat ini
            count = len([f for f in os.listdir(target_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            return jsonify({
                "status": "ok",
                "message": f"Berhasil menyimpan data latih: {filename}",
                "count": count,
                "char": target_char
            })
            
        return jsonify({"error": "Area kotak tangan kosong!"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/dataset_counts', methods=['GET'])
def get_dataset_counts():
    try:
        counts = {}
        for char in alphabet:
            char_dir = os.path.join(dataset_dir, char)
            if os.path.exists(char_dir):
                counts[char] = len([f for f in os.listdir(char_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            else:
                counts[char] = 0
        return jsonify(counts)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- ASISTEN PENERJEMAH BAHASA ISYARAT HELPERS ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if HAS_GEMINI and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

INDONESIAN_DICTIONARY = [
    "halo", "hello", "pagi", "siang", "sore", "malam", "apa", "kabar", "nama", "saya", 
    "kamu", "dia", "mereka", "kita", "bisa", "tidak", "suka", "makan", "minum", "tidur", 
    "belajar", "isyarat", "bahasa", "terima", "kasih", "sama", "sama", "tolong", "maaf", 
    "sehat", "sakit", "lapar", "haus", "senang", "sedih", "marah", "takut", "mau", 
    "ingin", "pergi", "pulang", "datang", "di", "ke", "dari", "ini", "itu", "ada", 
    "sudah", "belum", "sedang", "akan", "dan", "atau", "dengan", "untuk", "nasi", 
    "buku", "pulpen", "sekolah", "rumah", "jalan", "siapa", "mengapa", "bagaimana",
    "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan", "sepuluh"
]

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def local_autocorrect(word):
    word = word.lower().strip()
    if not word:
        return ""
    if word in INDONESIAN_DICTIONARY:
        return word
        
    best_word = word
    best_dist = 999
    
    for dict_word in INDONESIAN_DICTIONARY:
        dist = levenshtein_distance(word, dict_word)
        # Ambil yang jaraknya kecil, maksimal setengah panjang kata asal
        if dist < best_dist and dist <= max(1, len(word) // 2):
            best_dist = dist
            best_word = dict_word
            
    return best_word

def local_refine_sentence(raw_text):
    words = raw_text.split()
    corrected_words = []
    thought_steps = ["Menganalisis masukan kata secara lokal..."]
    
    for word in words:
        corrected = local_autocorrect(word)
        if corrected != word:
            thought_steps.append(f"Koreksi kata: '{word}' -> '{corrected}'")
            corrected_words.append(corrected)
        else:
            corrected_words.append(word)
            
    refined = " ".join(corrected_words)
    if refined:
        refined = refined.capitalize() + "."
    thought_steps.append("Penyusunan kalimat selesai.")
    
    return refined, thought_steps

def gemini_refine_sentence(raw_text):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = f"""
        Kamu adalah Asisten Penerjemah Bahasa Isyarat Indonesia.
        Tugasmu adalah merapikan urutan kata mentah hasil deteksi bahasa isyarat menjadi kalimat bahasa Indonesia yang rapi, padat, ber-tata bahasa benar, dan mudah dipahami.
        Biasanya kata mentah mengandung typo ejaan atau merupakan gabungan kata kerja/kata benda dasar (karena bahasa isyarat seringkali tidak menggunakan imbuhan secara penuh).
        
        Kata mentah yang dideteksi: "{raw_text}"
        
        Format output yang harus kamu berikan harus berupa JSON dengan struktur berikut:
        {{
            "thought": "Analisis singkat dalam menyusun kata menjadi kalimat bahasa Indonesia yang baik.",
            "refined": "Hasil akhir kalimat bahasa Indonesia yang sudah rapi dan benar (lengkap dengan tanda baca, diawali huruf kapital)."
        }}
        
        Ingat, kembalikan HANYA format JSON di atas, jangan tambahkan markdown atau teks penjelas lain di luar JSON.
        """
        
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        import json
        result = json.loads(response.text.strip())
        return result.get("refined", ""), [result.get("thought", "Analisis kata selesai.")]
    except Exception as e:
        print(f"[WARN] Gagal menggunakan Gemini API: {e}")
        # Fallback ke lokal
        return local_refine_sentence(raw_text)

@app.route('/refine_sentence', methods=['POST'])
def refine_sentence():
    try:
        data = request.get_json()
        if not data or 'raw_text' not in data:
            return jsonify({"error": "Data raw_text tidak dikirimkan"}), 400
            
        raw_text = data['raw_text']
        if not raw_text.strip():
            return jsonify({"refined": "", "thought": ["Input kosong."] })
            
        # Gunakan Gemini jika API Key ada, jika tidak, gunakan local fallback
        if HAS_GEMINI and GEMINI_API_KEY:
            refined, thought = gemini_refine_sentence(raw_text)
            source = "Sistem Awan"
        else:
            refined, thought = local_refine_sentence(raw_text)
            source = "Sistem Lokal"
            
        return jsonify({
            "refined": refined,
            "thought": thought,
            "source": source
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Pastikan folder templates dan static ada
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("\n" + "="*70)
    print("[OK] Menjalankan server lokal Pengenal Isyarat Tangan.")
    print("Buka browser dan buka: http://127.0.0.1:5000")
    print("="*70 + "\n")
    
    app.run(host='127.0.0.1', port=5000, debug=True)
