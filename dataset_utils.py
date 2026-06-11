"""
Preprocessing citra tangan & audit kualitas dataset.
Dipakai oleh train_model.py dan app.py agar input CNN konsisten.
"""
import os
from datetime import datetime

import cv2
import numpy as np

# Ambang kualitas gambar (setelah center + resize 64x64)
MIN_HAND_PIXELS = 280
MAX_HAND_PIXEL_RATIO = 0.72
MIN_MEAN_BRIGHTNESS = 8.0
MAX_MEAN_BRIGHTNESS = 75.0
HAND_CANVAS_PAD_RATIO = 0.18
TARGET_HAND_MEAN = 42.0


def center_hand_gray(gray):
    """Pusatkan blob tangan di canvas persegi — posisi konsisten untuk CNN."""
    if gray is None:
        return None
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    nz = cv2.findNonZero(gray)
    if nz is None:
        return gray.copy()
    x, y, bw, bh = cv2.boundingRect(nz)
    hand = gray[y:y + bh, x:x + bw]
    side = max(bw, bh, 1)
    pad = max(4, int(side * HAND_CANVAS_PAD_RATIO))
    canvas_side = side + 2 * pad
    canvas = np.zeros((canvas_side, canvas_side), dtype=gray.dtype)
    yo = pad + (side - bh) // 2
    xo = pad + (side - bw) // 2
    canvas[yo:yo + bh, xo:xo + bw] = hand
    return canvas


def normalize_hand_brightness(gray_uint8):
    """Samakan kecerahan tangan agar model fokus bentuk, bukan terang/gelap."""
    out = gray_uint8.astype(np.float32)
    mask = out > 8
    if not np.any(mask):
        return gray_uint8
    mean = float(out[mask].mean())
    if mean < 1.0:
        return gray_uint8
    out = np.clip(out * (TARGET_HAND_MEAN / mean), 0, 255)
    return out.astype(np.uint8)


def preprocess_gray_for_cnn(gray, img_size=64):
    """Center → resize → samakan brightness → normalisasi 0–1."""
    centered = center_hand_gray(gray)
    if centered is None:
        return None
    resized = cv2.resize(centered, (img_size, img_size), interpolation=cv2.INTER_AREA)
    resized = normalize_hand_brightness(resized)
    return resized.astype(np.float32) / 255.0


def analyze_gray_sample(gray64):
    """Analisis kualitas satu gambar 64x64 uint8."""
    if gray64 is None:
        return {"ok": False, "issues": ["unreadable"]}
    if gray64.dtype != np.uint8:
        gray64 = np.clip(gray64, 0, 255).astype(np.uint8)
    nonzero = int(cv2.countNonZero(gray64))
    total = gray64.shape[0] * gray64.shape[1]
    ratio = nonzero / total
    hand_mask = gray64 > 8
    hand_mean = float(np.mean(gray64[hand_mask])) if np.any(hand_mask) else 0.0
    issues = []
    if nonzero < MIN_HAND_PIXELS:
        issues.append("tangan_terlalu_kecil")
    if ratio > MAX_HAND_PIXEL_RATIO:
        issues.append("mask_kebesaran")
    if hand_mean < MIN_MEAN_BRIGHTNESS:
        issues.append("terlalu_gelap")
    if hand_mean > MAX_MEAN_BRIGHTNESS:
        issues.append("terlalu_terang")
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "nonzero": nonzero,
        "ratio": ratio,
        "mean": hand_mean,
    }


def audit_dataset(dataset_dir, alphabet, img_size=64):
    """Scan semua gambar; laporkan statistik & file bermasalah."""
    per_class = {}
    bad_files = []
    all_means = []
    total = 0

    for char in alphabet:
        char_dir = os.path.join(dataset_dir, char)
        if not os.path.isdir(char_dir):
            per_class[char] = {"count": 0, "bad": 0, "mean_avg": 0.0}
            continue
        files = [
            f for f in os.listdir(char_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        means = []
        bad = 0
        for file_name in files:
            path = os.path.join(char_dir, file_name)
            gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                bad += 1
                bad_files.append({"char": char, "file": file_name, "issues": ["unreadable"]})
                continue
            proc = preprocess_gray_for_cnn(gray, img_size)
            if proc is None:
                bad += 1
                bad_files.append({"char": char, "file": file_name, "issues": ["preprocess_gagal"]})
                continue
            gray64 = (proc * 255).astype(np.uint8)
            info = analyze_gray_sample(gray64)
            means.append(info["mean"])
            total += 1
            if not info["ok"]:
                bad += 1
                bad_files.append({
                    "char": char,
                    "file": file_name,
                    "issues": info["issues"],
                    "nonzero": info["nonzero"],
                    "mean": round(info["mean"], 1),
                })
        per_class[char] = {
            "count": len(files),
            "bad": bad,
            "mean_avg": float(np.mean(means)) if means else 0.0,
        }
        if means:
            all_means.extend(means)

    summary = {
        "total_images": total,
        "bad_images": len(bad_files),
        "classes": per_class,
        "bad_files": bad_files,
        "mean_brightness_range": (
            float(min(all_means)) if all_means else 0.0,
            float(max(all_means)) if all_means else 0.0,
        ),
    }
    return summary


def save_audit_report(summary, report_path):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("AUDIT DATASET BAHASA ISYARAT\n")
        f.write(f"Waktu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total gambar   : {summary['total_images']}\n")
        f.write(f" Bermasalah    : {summary['bad_images']}\n")
        lo, hi = summary["mean_brightness_range"]
        f.write(f" Rentang mean   : {lo:.1f} – {hi:.1f}\n\n")
        f.write("Per kelas (count / bermasalah / mean rata-rata):\n")
        for char, info in sorted(summary["classes"].items()):
            f.write(
                f"  {char}: {info['count']} foto, {info['bad']} bermasalah, "
                f"mean={info['mean_avg']:.1f}\n"
            )
        if summary["bad_files"]:
            f.write("\nFile bermasalah (capture ulang jika bisa):\n")
            for item in summary["bad_files"][:200]:
                issues = ", ".join(item["issues"])
                extra = ""
                if "mean" in item:
                    extra = f" mean={item['mean']} nz={item.get('nonzero', '?')}"
                f.write(f"  dataset/{item['char']}/{item['file']}  [{issues}]{extra}\n")
            if len(summary["bad_files"]) > 200:
                f.write(f"  ... +{len(summary['bad_files']) - 200} file lain\n")


def augment_batch(X):
    """Augmentasi ringan — tanpa flip horizontal (isyarat tidak simetris)."""
    out = X.copy()
    for i in range(len(out)):
        img = out[i, :, :, 0]
        angle = np.random.uniform(-5, 5)
        h, w = img.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        shift_x = np.random.uniform(-0.04, 0.04) * w
        shift_y = np.random.uniform(-0.04, 0.04) * h
        m2 = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        img = cv2.warpAffine(img, m2, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        scale = np.random.uniform(0.94, 1.06)
        img = np.clip(img * scale, 0.0, 1.0)
        out[i, :, :, 0] = img
    return out
