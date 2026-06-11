import os

# Sembunyikan log MediaPipe / TensorFlow Lite (harus sebelum import mediapipe)
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MP_VERBOSE", "0")

import cv2
import numpy as np
import base64
import time
import logging
import warnings
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), encoding="utf-8-sig")

warnings.filterwarnings('ignore')

# Sederhanakan output terminal: sembunyikan log request per-frame & warning bawaan
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Set Keras backend sebelum import
os.environ["KERAS_BACKEND"] = "jax"
import json
import shutil
import tempfile

import h5py
from keras import models

from hand_segmentation import init_hand_landmarker, is_mediapipe_ready, segment_hand
from dataset_utils import preprocess_gray_for_cnn


def load_cnn_model(path):
    """Muat model .h5; perbaiki quantization_config jika Keras menolak."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file '{path}' tidak ditemukan.")
    try:
        return models.load_model(path)
    except (TypeError, ValueError) as err:
        if "quantization_config" not in str(err):
            raise
        with h5py.File(path, "r") as src:
            config = json.loads(src.attrs["model_config"])

            def _strip_quant(obj):
                if isinstance(obj, dict):
                    obj.pop("quantization_config", None)
                    for v in obj.values():
                        _strip_quant(v)
                elif isinstance(obj, list):
                    for item in obj:
                        _strip_quant(item)

            _strip_quant(config)
            fixed = json.dumps(config).encode("utf-8")
        fd, tmp = tempfile.mkstemp(suffix=".h5")
        os.close(fd)
        try:
            shutil.copy2(path, tmp)
            with h5py.File(tmp, "r+") as dst:
                dst.attrs["model_config"] = fixed
            return models.load_model(tmp)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

app = Flask(__name__)
CORS(app)

MODELS_DIR = os.path.join(BASE_DIR, "models")
model_path = os.path.join(MODELS_DIR, "sign_language_cnn_model.h5")
_legacy_model_path = os.path.join(BASE_DIR, "sign_language_cnn_model.h5")
if not os.path.exists(model_path) and os.path.exists(_legacy_model_path):
    model_path = _legacy_model_path
dataset_dir = os.path.join(BASE_DIR, "dataset")

# Ukuran input model CNN — wajib 64x64 (sama dengan train_model.py)
REQUIRED_IMG_SIZE = 64
IMG_SIZE = REQUIRED_IMG_SIZE
PREVIEW_SIZE = 192  # ukuran tampilan preview UI saja (bukan input model)
# Ukuran kotak ROI (area tangan) pada frame kamera. Harus sama dengan nilai di app.js.
ROI_BOX_SIZE = 300
# Resolusi penyimpanan dataset (disimpan besar agar detail terjaga untuk training).
SAVE_SIZE = 256

# Hardcode alfabet dan angka sesuai folder di dataset (36 kelas)
alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
label_map = {i: char for i, char in enumerate(alphabet)}

# Model CNN global (nama cnn_model agar tidak bentrok dengan variabel Gemini)
cnn_model = None
if not os.path.exists(model_path):
    print(f"  [!] Model '{model_path}' tidak ditemukan. Jalankan train_model.py dulu.")
else:
    try:
        cnn_model = load_cnn_model(model_path)
        in_h = int(cnn_model.input_shape[1] or 0)
        in_w = int(cnn_model.input_shape[2] or 0)
        if in_h != REQUIRED_IMG_SIZE or in_w != REQUIRED_IMG_SIZE:
            print(
                f"  [!] Model harus input {REQUIRED_IMG_SIZE}x{REQUIRED_IMG_SIZE}, "
                f"file ini {in_w}x{in_h}. Jalankan train_model.py untuk buat model baru."
            )
            cnn_model = None
        else:
            dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 1), dtype=np.float32)
            cnn_model.predict(dummy, verbose=0)
            print(f"  [OK] Model dimuat (input {IMG_SIZE}x{IMG_SIZE}).")
    except Exception as e:
        print(f"  [!] Gagal memuat model: {e}")
        cnn_model = None

def bytes_to_cv2(img_bytes):
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return None


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

MIN_MEAN_BRIGHTNESS = 5

HAND_SEGMENTATION_MODE = "mediapipe"
if init_hand_landmarker():
    print("  [OK] MediaPipe HandLandmarker siap (masking green-screen).")
else:
    HAND_SEGMENTATION_MODE = "kulit"
    print("  [!] MediaPipe tidak siap — fallback deteksi warna kulit.")


def get_fixed_roi_coords(w, h, box_size=ROI_BOX_SIZE):
    """
    Mendapatkan koordinat kotak ROI di tengah frame (simetris, ramah untuk mirror).
    """
    box = min(box_size, w, h)
    x_min = int(w * 0.5 - box * 0.5)
    y_min = int(h * 0.5 - box * 0.5)
    x_max = x_min + box
    y_max = y_min + box
    
    # Validasi batas frame
    return max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max)


def is_hand_in_gray(gray_img):
    """Cek tangan dari citra grayscale 64x64 (lebih ringan dari full ROI)."""
    min_pixels = max(80, int(gray_img.shape[0] * gray_img.shape[1] * 0.015))
    if cv2.countNonZero(gray_img) < min_pixels:
        return False
    return float(np.mean(gray_img)) >= MIN_MEAN_BRIGHTNESS


def make_mini_roi_base64(gray_img):
    """Preview input CNN — sama persis dengan grayscale 64x64 (tanpa normalisasi min-max)."""
    if gray_img is None:
        return ""
    preview = cv2.resize(
        gray_img.astype(np.uint8), (PREVIEW_SIZE, PREVIEW_SIZE), interpolation=cv2.INTER_LINEAR
    )
    _, buffer = cv2.imencode('.jpg', preview, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

def empty_prediction_response(x_min, y_min, x_max, y_max, gray_seg=None, mini_roi=None):
    if mini_roi is None:
        mini_roi = make_mini_roi_base64(gray_seg) if gray_seg is not None else ""
    return jsonify({
        "char": "?",
        "confidence": 0.0,
        "hand_detected": False,
        "top3": [],
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "mini_roi": mini_roi
    })

def preprocess_for_prediction(roi):
    """
    Ubah ROI menjadi 64x64 grayscale. MediaPipe segmentasi di resolusi ROI penuh.
    """
    try:
        segmented = segment_hand(roi)
        gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
        normalized = preprocess_gray_for_cnn(gray, IMG_SIZE)
        if normalized is None:
            return None, None
        gray_64 = (normalized * 255).astype(np.uint8)
        input_image = np.expand_dims(normalized, axis=-1)
        input_image = np.expand_dims(input_image, axis=0)
        return input_image, gray_64
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return None, None

@app.route('/')
def index():
    return render_template('index.html', ai_source_status=AI_SOURCE_STATUS)

def _parse_predict_request():
    """Terima JPEG binary (FormData) atau JSON base64."""
    skip_preview = False
    roi_only = False
    frame = None

    if request.files and 'image' in request.files:
        skip_preview = request.form.get('skip_preview', '0') in ('1', 'true', 'yes')
        roi_only = request.form.get('roi_only', '0') in ('1', 'true', 'yes')
        frame = bytes_to_cv2(request.files['image'].read())
    else:
        data = request.get_json(silent=True)
        if data and 'image' in data:
            skip_preview = bool(data.get('skip_preview', False))
            roi_only = bool(data.get('roi_only', False))
            frame = base64_to_cv2(data['image'])

    return frame, skip_preview, roi_only


@app.route('/predict', methods=['POST'])
def predict_sign():
    try:
        frame, skip_preview, roi_only = _parse_predict_request()
        if frame is None:
            return jsonify({"error": "Data gambar tidak dikirimkan"}), 400

        h, w = frame.shape[:2]
        if roi_only:
            roi = frame
            x_min, y_min, x_max, y_max = 0, 0, w, h
        else:
            x_min, y_min, x_max, y_max = get_fixed_roi_coords(w, h)
            roi = frame[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            return empty_prediction_response(x_min, y_min, x_max, y_max)

        input_image, gray_input = preprocess_for_prediction(roi)
        mini_roi = "" if skip_preview else make_mini_roi_base64(gray_input)

        if input_image is None or cnn_model is None:
            return empty_prediction_response(x_min, y_min, x_max, y_max, mini_roi=mini_roi)

        if not is_hand_in_gray(gray_input):
            return empty_prediction_response(x_min, y_min, x_max, y_max, mini_roi=mini_roi)

        probs = cnn_model.predict(input_image, verbose=0)
        probs = np.asarray(probs[0])
        predicted_idx = int(np.argmax(probs))
        predicted_char = label_map.get(predicted_idx, "?")
        confidence = float(probs[predicted_idx])

        top_idx = np.argsort(probs)[::-1][:3]
        top3 = [
            {"char": label_map.get(int(i), "?"), "confidence": float(probs[i])}
            for i in top_idx
        ]

        return jsonify({
            "char": predicted_char,
            "confidence": confidence,
            "hand_detected": True,
            "top3": top3,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "mini_roi": mini_roi
        })
    except Exception as e:
        print(f"  [!] Error /predict: {e}")
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
        
        # Potong ROI dari frame asli (non-mirrored)
        roi = frame[y_min:y_max, x_min:x_max]
        
        if roi.size > 0:
            # Terapkan segmentasi warna kulit dan simpan dalam resolusi tinggi
            roi_segmented = segment_hand(roi)
            roi_resized = cv2.resize(roi_segmented, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
            
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
GEMINI_API_KEY = (os.environ.get("GEMINI_API_KEY") or "").strip()
GEMINI_MODEL = (os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash").strip()
gemini_cloud_ok = bool(HAS_GEMINI and GEMINI_API_KEY)


def gemini_error_reason(exc):
    msg = str(exc).lower()
    if "429" in msg or "quota" in msg or "rate" in msg:
        return "kuota API habis"
    if "401" in msg or "403" in msg or "api key" in msg or "permission" in msg:
        return "API key tidak valid"
    return "koneksi bermasalah"


def refresh_ai_source_status():
    global AI_SOURCE_STATUS
    if not HAS_GEMINI or not GEMINI_API_KEY:
        AI_SOURCE_STATUS = "Sistem Lokal"
    elif gemini_cloud_ok:
        AI_SOURCE_STATUS = "Sistem Awan (Siap)"
    else:
        AI_SOURCE_STATUS = "Sistem Awan (Offline)"
    return AI_SOURCE_STATUS


def mark_gemini_unavailable(exc):
    global gemini_cloud_ok
    if gemini_cloud_ok:
        gemini_cloud_ok = False
        reason = gemini_error_reason(exc)
        print(f"  [!] Gemini tidak dapat dipakai ({reason}). Memakai mode lokal.")
        refresh_ai_source_status()


AI_SOURCE_STATUS = refresh_ai_source_status()

if HAS_GEMINI and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
elif not HAS_GEMINI:
    print("  [!] Package google-generativeai belum terinstall. Fitur Gemini memakai fallback lokal.")
else:
    print("  [!] GEMINI_API_KEY kosong. Isi file .env lalu restart app.py untuk mengaktifkan Gemini.")

INDONESIAN_DICTIONARY = [
    "halo", "pagi", "siang", "sore", "malam", "apa", "kabar", "nama", "saya",
    "kamu", "dia", "mereka", "kita", "bisa", "tidak", "suka", "makan", "minum", "tidur",
    "belajar", "isyarat", "bahasa", "terima", "kasih", "sama", "tolong", "maaf",
    "sehat", "sakit", "lapar", "haus", "senang", "sedih", "marah", "takut", "mau",
    "ingin", "pergi", "pulang", "datang", "di", "ke", "dari", "ini", "itu", "ada",
    "sudah", "belum", "sedang", "akan", "dan", "atau", "dengan", "untuk", "nasi",
    "buku", "pulpen", "sekolah", "rumah", "jalan", "siapa", "mengapa", "bagaimana",
    "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan", "sepuluh"
]

ENGLISH_DICTIONARY = [
    "hello", "hi", "good", "morning", "afternoon", "evening", "night", "what", "how",
    "name", "i", "you", "he", "she", "they", "we", "can", "cannot", "like", "eat",
    "drink", "sleep", "learn", "sign", "language", "thank", "thanks", "please", "sorry",
    "healthy", "sick", "hungry", "thirsty", "happy", "sad", "angry", "afraid", "want",
    "need", "go", "come", "home", "school", "book", "pen", "food", "water", "help",
    "yes", "no", "the", "a", "an", "is", "are", "am", "my", "your", "this", "that",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"
]

DICTIONARIES = {
    "id": INDONESIAN_DICTIONARY,
    "en": ENGLISH_DICTIONARY,
}

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

def detect_language_local(words):
    id_matches = sum(1 for w in words if w.lower() in INDONESIAN_DICTIONARY)
    en_matches = sum(1 for w in words if w.lower() in ENGLISH_DICTIONARY)
    return "en" if en_matches > id_matches else "id"

def resolve_language(language, words):
    lang = (language or "auto").lower()
    if lang in ("id", "en"):
        return lang
    return detect_language_local(words)

def max_edit_distance(word_len):
    """Batas jarak edit untuk fuzzy match berdasarkan panjang kata."""
    if word_len <= 2:
        return 1
    if word_len <= 4:
        return 2
    return min(3, max(1, word_len // 2))

def local_autocorrect(word, lang="id"):
    word = word.lower().strip()
    if not word:
        return word

    dictionary = DICTIONARIES.get(lang, INDONESIAN_DICTIONARY)
    if word in dictionary:
        return word

    best_word = word
    best_dist = 999
    limit = max_edit_distance(len(word))

    for dict_word in dictionary:
        dist = levenshtein_distance(word, dict_word)
        if dist < best_dist and dist <= limit:
            best_dist = dist
            best_word = dict_word

    return best_word

def local_refine_sentence(raw_text, language="auto"):
    words = raw_text.split()
    detected_lang = resolve_language(language, words)
    corrected_words = []
    thought_steps = [f"Menganalisis masukan kata ({detected_lang.upper()})..."]

    for word in words:
        corrected = local_autocorrect(word, detected_lang)
        if corrected != word.lower():
            thought_steps.append(f"Koreksi kata: '{word}' -> '{corrected}'")
        corrected_words.append(corrected)

    refined = " ".join(corrected_words)
    if refined:
        refined = refined[0].upper() + refined[1:] + "."
    thought_steps.append("Penyusunan kalimat selesai.")

    return refined, thought_steps, detected_lang

CONTEXT_FOLLOWS = {
    "id": {
        "saya": ["mau", "ingin", "suka", "bisa", "tidak", "lapar", "haus"],
        "mau": ["makan", "minum", "tidur", "belajar", "pergi"],
        "ingin": ["makan", "minum", "belajar", "pergi"],
        "apa": ["kabar", "nama"],
        "terima": ["kasih"],
        "tidak": ["mau", "bisa", "suka"],
    },
    "en": {
        "i": ["want", "need", "like", "can", "am"],
        "want": ["eat", "drink", "sleep", "learn", "go"],
        "thank": ["you"],
        "how": ["are"],
    },
}

def score_prediction_candidate(partial, word, last_context, follows_map):
    """Skor kandidat prediksi: prefix match + fuzzy typo + bonus konteks."""
    partial = partial.lower()
    word = word.lower()
    score = None

    if word.startswith(partial):
        score = 130 - (len(word) - len(partial)) * 2
    else:
        dist = levenshtein_distance(partial, word)
        if dist <= max_edit_distance(len(partial)):
            score = 95 - dist * 18
            word_prefix = word[:len(partial)] if len(partial) <= len(word) else word
            if len(partial) >= 2 and levenshtein_distance(partial, word_prefix) <= 1:
                score += 12

    if score is None:
        return None

    if last_context and word in follows_map.get(last_context, []):
        score += 65

    return score

def local_predict_words(partial_word, context_words, language="auto"):
    partial = partial_word.lower().strip()
    context_words = [w.lower().strip() for w in context_words if w.strip()]
    detected_lang = resolve_language(language, context_words + [partial])

    if len(partial) < 2:
        return [], detected_lang

    dictionary = DICTIONARIES.get(detected_lang, INDONESIAN_DICTIONARY)
    follows_map = CONTEXT_FOLLOWS.get(detected_lang, {})
    last_context = context_words[-1] if context_words else ""

    candidates = {}
    for word in dictionary:
        score = score_prediction_candidate(partial, word, last_context, follows_map)
        if score is not None:
            candidates[word] = max(candidates.get(word, 0), score)

    corrected = local_autocorrect(partial, detected_lang)
    if corrected != partial:
        ac_score = 100
        if last_context and corrected in follows_map.get(last_context, []):
            ac_score += 65
        candidates[corrected] = max(candidates.get(corrected, 0), ac_score)

    ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
    suggestions = [word for word, _ in ranked[:3]]

    return suggestions, detected_lang

def gemini_predict_words(partial_word, context_words, language="auto"):
    import json

    partial = partial_word.lower().strip()
    context = " ".join(context_words).strip()
    lang_instruction = {
        "id": "Berikan prediksi kata Bahasa Indonesia.",
        "en": "Provide English word predictions.",
        "auto": "Deteksi bahasa dari konteks, lalu berikan prediksi kata dalam bahasa tersebut.",
    }

    try:
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = f"""
        Kamu adalah asisten prediksi kata untuk penerjemah bahasa isyarat.
        Pengguna mengetik kata huruf-per-huruf lewat deteksi isyarat tangan.
        Input SERING mengandung typo: huruf tertukar, kurang, lebih, atau salah urutan
        (contoh: "mkan" -> "makan", "nakan" -> "makan", "mkn" -> "makan").

        Tugas:
        1. Perbaiki typo secara implisit berdasarkan konteks kalimat.
        2. Prediksi 3 kata LENGKAP paling masuk akal untuk melanjutkan kalimat.
        3. Prioritaskan kata yang cocok dengan konteks, bukan hanya awalan huruf.

        Preferensi bahasa: {language}
        Instruksi: {lang_instruction.get(language, lang_instruction["auto"])}

        Konteks kalimat sejauh ini: "{context}"
        Kata parsial sedang diketik (bisa typo): "{partial}"

        Format output HANYA JSON:
        {{
            "detected_language": "id atau en",
            "suggestions": ["kata1", "kata2", "kata3"]
        }}
        """

        response = gemini_model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        result = json.loads(response.text.strip())
        suggestions = [s.lower().strip() for s in result.get("suggestions", []) if s]
        detected_lang = result.get("detected_language", "id")
        if detected_lang not in ("id", "en"):
            detected_lang = resolve_language(language, context_words + [partial])

        if len(suggestions) < 3:
            local_suggestions, _ = local_predict_words(partial_word, context_words, language)
            for word in local_suggestions:
                if word not in suggestions:
                    suggestions.append(word)
                if len(suggestions) >= 3:
                    break

        return suggestions[:3], detected_lang, "Sistem Awan"
    except Exception as e:
        mark_gemini_unavailable(e)
        suggestions, detected_lang = local_predict_words(partial_word, context_words, language)
        return suggestions, detected_lang, "Sistem Lokal"

def gemini_refine_sentence(raw_text, language="auto"):
    import json

    lang_instruction = {
        "id": "Rapikan menjadi kalimat Bahasa Indonesia yang benar.",
        "en": "Refine into a proper English sentence.",
        "auto": "Deteksi apakah input lebih cocok Bahasa Indonesia atau Inggris, lalu rapikan dalam bahasa tersebut.",
    }
    instruction = lang_instruction.get(language, lang_instruction["auto"])

    try:
        gemini_model = genai.GenerativeModel(GEMINI_MODEL)

        prompt = f"""
        Kamu adalah Asisten Penerjemah Bahasa Isyarat.
        Tugasmu merapikan urutan kata mentah hasil deteksi isyarat menjadi kalimat yang rapi dan mudah dipahami.
        Kata mentah sering mengandung typo ejaan atau kata dasar tanpa imbuhan lengkap.

        Preferensi bahasa: {language}
        Instruksi: {instruction}

        Kata mentah yang dideteksi: "{raw_text}"

        Format output HANYA JSON:
        {{
            "detected_language": "id atau en",
            "thought": "Analisis singkat proses koreksi dan penyusunan kalimat.",
            "refined": "Kalimat akhir yang sudah rapi (huruf kapital di awal, tanda baca di akhir).",
            "corrections": [{{"from": "kata_salah", "to": "kata_benar"}}]
        }}
        """

        response = gemini_model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )

        result = json.loads(response.text.strip())
        thought = [result.get("thought", "Analisis kata selesai.")]
        corrections = result.get("corrections", [])
        for corr in corrections:
            if corr.get("from") and corr.get("to"):
                thought.append(f"Koreksi: '{corr['from']}' -> '{corr['to']}'")

        detected_lang = result.get("detected_language", "id")
        if detected_lang not in ("id", "en"):
            detected_lang = resolve_language(language, raw_text.split())

        return result.get("refined", ""), thought, detected_lang, "Sistem Awan"
    except Exception as e:
        mark_gemini_unavailable(e)
        refined, thought, detected_lang = local_refine_sentence(raw_text, language)
        return refined, thought, detected_lang, "Sistem Lokal"

@app.route('/ai_status', methods=['GET'])
def ai_status():
    return jsonify({
        "status": AI_SOURCE_STATUS,
        "gemini_configured": bool(GEMINI_API_KEY),
        "gemini_available": gemini_cloud_ok,
        "model": GEMINI_MODEL if GEMINI_API_KEY else None,
    })


@app.route('/predict_words', methods=['POST'])
def predict_words():
    try:
        data = request.get_json()
        if not data or 'partial_word' not in data:
            return jsonify({"error": "Parameter partial_word wajib diisi"}), 400

        partial_word = (data.get('partial_word') or '').strip().lower()
        context_words = data.get('context_words', [])
        language = data.get('language', 'auto')

        if len(partial_word) < 2:
            return jsonify({
                "suggestions": [],
                "source": AI_SOURCE_STATUS,
                "detected_language": resolve_language(language, context_words)
            })

        if gemini_cloud_ok:
            suggestions, detected_lang, source = gemini_predict_words(
                partial_word, context_words, language
            )
        else:
            suggestions, detected_lang = local_predict_words(
                partial_word, context_words, language
            )
            source = "Sistem Lokal"

        return jsonify({
            "suggestions": suggestions,
            "source": source,
            "detected_language": detected_lang,
            "ai_status": AI_SOURCE_STATUS,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/refine_sentence', methods=['POST'])
def refine_sentence():
    try:
        data = request.get_json()
        if not data or 'raw_text' not in data:
            return jsonify({"error": "Data raw_text tidak dikirimkan"}), 400

        raw_text = data['raw_text']
        language = data.get('language', 'auto')

        if not raw_text.strip():
            return jsonify({"refined": "", "thought": ["Input kosong."], "detected_language": "id"})

        if gemini_cloud_ok:
            refined, thought, detected_lang, source = gemini_refine_sentence(raw_text, language)
        else:
            refined, thought, detected_lang = local_refine_sentence(raw_text, language)
            source = "Sistem Lokal"

        return jsonify({
            "refined": refined,
            "thought": thought,
            "source": source,
            "detected_language": detected_lang,
            "ai_status": AI_SOURCE_STATUS,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)

    port = int(os.environ.get("PORT", "5001"))
    seg_label = "MediaPipe" if is_mediapipe_ready() else "Warna kulit"
    print(f"\n  BicaraIsyarat siap  ->  http://127.0.0.1:{port}")
    print(f"  Masking tangan      : {seg_label}")
    print(f"  AI penyusun kalimat : {AI_SOURCE_STATUS}")
    print(f"  PID {os.getpid()}  (tekan CTRL+C untuk berhenti)\n")

    # use_reloader=False agar tidak ada output/booting dobel
    # threaded=False agar request diproses satu per satu (aman untuk model JAX)
    app.run(host='127.0.0.1', port=port, debug=False, use_reloader=False, threaded=False)
