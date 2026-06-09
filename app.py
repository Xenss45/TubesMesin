import os
import cv2
import numpy as np
import base64
import time
import sys
import warnings
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv(encoding="utf-8-sig")

warnings.filterwarnings('ignore')

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# Set Keras backend sebelum import
os.environ['KERAS_BACKEND'] = 'jax'
from model_utils import load_model_compatible

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
        model = load_model_compatible(model_path)
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

MIN_SKIN_PIXELS = 2500
MIN_MEAN_BRIGHTNESS = 5

def is_hand_in_roi(segmented_roi):
    """Cek apakah ROI berisi tangan (bukan gambar full hitam)."""
    gray_seg = cv2.cvtColor(segmented_roi, cv2.COLOR_BGR2GRAY)
    skin_pixel_count = cv2.countNonZero(gray_seg)
    if skin_pixel_count < MIN_SKIN_PIXELS:
        return False, gray_seg
    if float(np.mean(gray_seg)) < MIN_MEAN_BRIGHTNESS:
        return False, gray_seg
    return True, gray_seg

def make_mini_roi_base64(gray_seg):
    gray_resized = cv2.resize(gray_seg, (100, 100), interpolation=cv2.INTER_NEAREST)
    _, buffer = cv2.imencode('.png', gray_resized)
    return "data:image/png;base64," + base64.b64encode(buffer).decode('utf-8')

def empty_prediction_response(x_min, y_min, x_max, y_max, gray_seg=None):
    mini_roi = make_mini_roi_base64(gray_seg) if gray_seg is not None else ""
    return jsonify({
        "char": "?",
        "confidence": 0.0,
        "hand_detected": False,
        "x_min": x_min,
        "y_min": y_min,
        "x_max": x_max,
        "y_max": y_max,
        "mini_roi": mini_roi
    })

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
    return render_template('index.html', ai_source_status=AI_SOURCE_STATUS)

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
        
        # Potong ROI lalu flip horizontal (selalu mirror untuk tangan kanan + tampilan cermin)
        roi = frame[y_min:y_max, x_min:x_max]
        roi = cv2.flip(roi, 1)
        
        if roi.size > 0:
            input_image, segmented_roi = preprocess_for_prediction(roi)
            if input_image is not None and model is not None:
                hand_detected, gray_seg = is_hand_in_roi(segmented_roi)
                if not hand_detected:
                    return empty_prediction_response(x_min, y_min, x_max, y_max, gray_seg)
                
                prediction = model.predict(input_image, verbose=0)
                predicted_idx = np.argmax(prediction)
                predicted_char = label_map.get(predicted_idx, "?")
                confidence = float(prediction[0][predicted_idx])
                
                return jsonify({
                    "char": predicted_char,
                    "confidence": confidence,
                    "hand_detected": True,
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "mini_roi": make_mini_roi_base64(gray_seg)
                })
        
        return empty_prediction_response(x_min, y_min, x_max, y_max)
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
        
        # Potong ROI lalu flip horizontal (selalu mirror untuk tangan kanan + tampilan cermin)
        roi = frame[y_min:y_max, x_min:x_max]
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
GEMINI_API_KEY = (os.environ.get("GEMINI_API_KEY") or "").strip()
AI_SOURCE_STATUS = "Sistem Awan (Siap)" if HAS_GEMINI and GEMINI_API_KEY else "Sistem Lokal"

if HAS_GEMINI and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
elif not HAS_GEMINI:
    print("[WARN] Package google-generativeai belum terinstall. Fitur Gemini memakai fallback lokal.")
else:
    print("[WARN] GEMINI_API_KEY kosong. Isi file .env lalu restart app.py untuk mengaktifkan Gemini.")

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
    thought_steps = [f"Menganalisis masukan kata secara lokal ({detected_lang.upper()})..."]

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
        model = genai.GenerativeModel("gemini-2.5-flash")
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

        response = model.generate_content(
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

        return suggestions[:3], detected_lang
    except Exception as e:
        print(f"[WARN] Gagal prediksi kata Gemini: {e}")
        return local_predict_words(partial_word, context_words, language)

def gemini_refine_sentence(raw_text, language="auto"):
    import json

    lang_instruction = {
        "id": "Rapikan menjadi kalimat Bahasa Indonesia yang benar.",
        "en": "Refine into a proper English sentence.",
        "auto": "Deteksi apakah input lebih cocok Bahasa Indonesia atau Inggris, lalu rapikan dalam bahasa tersebut.",
    }
    instruction = lang_instruction.get(language, lang_instruction["auto"])

    try:
        model = genai.GenerativeModel("gemini-2.5-flash")

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

        response = model.generate_content(
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

        return result.get("refined", ""), thought, detected_lang
    except Exception as e:
        print(f"[WARN] Gagal menggunakan Gemini API: {e}")
        refined, thought, detected_lang = local_refine_sentence(raw_text, language)
        return refined, thought, detected_lang

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

        if HAS_GEMINI and GEMINI_API_KEY:
            suggestions, detected_lang = gemini_predict_words(partial_word, context_words, language)
            source = "Sistem Awan"
        else:
            suggestions, detected_lang = local_predict_words(partial_word, context_words, language)
            source = "Sistem Lokal"

        return jsonify({
            "suggestions": suggestions,
            "source": source,
            "detected_language": detected_lang
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

        if HAS_GEMINI and GEMINI_API_KEY:
            refined, thought, detected_lang = gemini_refine_sentence(raw_text, language)
            source = "Sistem Awan"
        else:
            refined, thought, detected_lang = local_refine_sentence(raw_text, language)
            source = "Sistem Lokal"

        return jsonify({
            "refined": refined,
            "thought": thought,
            "source": source,
            "detected_language": detected_lang
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
    print(f"[INFO] Status AI penyusun kalimat: {AI_SOURCE_STATUS}")
    print("Buka browser dan buka: http://127.0.0.1:5000")
    print("="*70 + "\n")
    
    app.run(host='127.0.0.1', port=5000, debug=True)
