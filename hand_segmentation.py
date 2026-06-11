"""
Segmentasi tangan MediaPipe — efek green screen (tangan di background hitam).
Fallback ke deteksi warna kulit jika landmark tidak terdeteksi.
"""
import contextlib
import os

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MP_VERBOSE", "0")


@contextlib.contextmanager
def _silence_cpp_logs():
    """MediaPipe/TFLite menulis langsung ke stderr fd — redam sementara."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull)
        os.close(saved_fd)

import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

MP_DETECT_MAX = 320
SEGMENT_MAX = 160
HAND_EDGE_MARGIN_PX = 6
HAND_HULL_PADDING_RATIO = 0.11
HAND_WRIST_EXTEND_RATIO = 0.30
HAND_THUMB_EXTEND_RATIO = 0.22
HAND_TIP_EXTEND_RATIO = 0.07
HAND_BBOX_MARGIN_X = 0.09
HAND_BBOX_MARGIN_Y = 0.10
HAND_BBOX_MARGIN_WRIST = 0.14
MIN_SKIN_PIXELS = 320

# Skeleton tangan MediaPipe (21 landmark)
_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 17), (5, 9), (9, 13), (13, 17),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
)

_HAND_MARGIN_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (HAND_EDGE_MARGIN_PX * 2 + 1, HAND_EDGE_MARGIN_PX * 2 + 1)
)
_SKIN_CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
_SKIN_OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

_hand_landmarker = None
_mediapipe_ready = False


def _ensure_hand_model():
    os.makedirs(os.path.dirname(HAND_MODEL_PATH), exist_ok=True)
    if os.path.exists(HAND_MODEL_PATH):
        return
    print("  [...] Mengunduh model MediaPipe hand_landmarker...")
    urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    print(f"  [OK] Model disimpan: {HAND_MODEL_PATH}")


def init_hand_landmarker():
    """Muat HandLandmarker sekali saat startup."""
    global _hand_landmarker, _mediapipe_ready
    try:
        _ensure_hand_model()
        base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.35,
            min_hand_presence_confidence=0.35,
            min_tracking_confidence=0.35,
        )
        with _silence_cpp_logs():
            _hand_landmarker = vision.HandLandmarker.create_from_options(options)
        _mediapipe_ready = True
        return True
    except Exception as e:
        print(f"  [!] MediaPipe HandLandmarker gagal: {e}")
        _mediapipe_ready = False
        return False


def is_mediapipe_ready():
    return _mediapipe_ready


def _glare_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array([0, 0, 215], dtype=np.uint8), np.array([180, 60, 255], dtype=np.uint8))


def _skin_color_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask_hsv = (
        cv2.inRange(hsv, np.array([0, 25, 50], dtype=np.uint8), np.array([22, 255, 255], dtype=np.uint8))
        | cv2.inRange(hsv, np.array([160, 25, 50], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
    )
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mask_ycrcb = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )
    return cv2.bitwise_and(mask_hsv, mask_ycrcb)


def _fill_holes(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return mask
    filled = mask.copy()
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] != -1:
            cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
    return filled


def _central_hand_component(mask, max_offset_ratio=0.38, side_margin=12):
    h, w = mask.shape
    cx0, cy0 = w / 2.0, h / 2.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100:
            continue
        m = cv2.moments(cnt)
        if m["m00"] < 1:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        if abs(cx - cx0) / w > max_offset_ratio or abs(cy - cy0) / h > max_offset_ratio:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        touches_side = x <= side_margin or x + bw >= w - side_margin
        candidates.append((area, touches_side, cnt))
    if not candidates:
        return np.zeros_like(mask)
    interior = [c for c in candidates if not c[1]]
    pool = interior if interior else candidates
    best_cnt = max(pool, key=lambda c: c[0])[2]
    out = np.zeros_like(mask)
    cv2.drawContours(out, [best_cnt], -1, 255, thickness=cv2.FILLED)
    return out


def _apply_mask_on_black(bgr, mask):
    out = np.zeros_like(bgr)
    out[mask > 0] = bgr[mask > 0]
    return out


def _extend_point(from_pt, to_pt, ratio):
    """Perpanjang to_pt menjauhi from_pt (ratio = fraksi panjang vektor)."""
    vec = to_pt - from_pt
    return to_pt + vec * ratio


def _clip_mask_to_hand_bbox(mask, pts_int, h, w):
    """Batasi mask agar tidak melebar jadi blob raksasa."""
    x, y, bw, bh = cv2.boundingRect(pts_int.reshape(-1, 1, 2))
    mx = max(4, int(bw * HAND_BBOX_MARGIN_X))
    my = max(4, int(bh * HAND_BBOX_MARGIN_Y))
    wrist = pts_int[0]
    palm_center = np.mean(pts_int[[5, 9, 13, 17]], axis=0)
    wrist_side = 1 if wrist[1] >= palm_center[1] else -1
    my_wrist = max(my, int(bh * HAND_BBOX_MARGIN_WRIST))
    y0 = max(0, y - my)
    y1 = min(h, y + bh + (my_wrist if wrist_side > 0 else my))
    x0 = max(0, x - mx)
    x1 = min(w, x + bw + mx)
    clipped = np.zeros_like(mask)
    clipped[y0:y1, x0:x1] = 255
    return cv2.bitwise_and(mask, clipped)


def _landmarks_to_mask(h, w, landmarks):
    """Mask mengikuti bentuk tangan — telapak aman, tidak melebar berlebihan."""
    pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)
    if pts.shape[0] < 21:
        return np.zeros((h, w), dtype=np.uint8)

    palm_mcps = pts[[5, 9, 13, 17]]
    palm_center = np.mean(palm_mcps, axis=0)
    extras = [
        _extend_point(palm_center, pts[0], HAND_WRIST_EXTEND_RATIO),
        _extend_point(pts[2], pts[1], HAND_THUMB_EXTEND_RATIO),
    ]
    for tip, joint in ((4, 3), (8, 7), (12, 11), (16, 15), (20, 19)):
        extras.append(_extend_point(pts[joint], pts[tip], HAND_TIP_EXTEND_RATIO))

    all_pts = np.vstack([pts, extras])
    all_int = np.clip(all_pts, [0, 0], [w - 1, h - 1]).astype(np.int32)
    pts_int = np.clip(pts, [0, 0], [w - 1, h - 1]).astype(np.int32)

    bw = float(np.max(all_pts[:, 0]) - np.min(all_pts[:, 0]))
    bh = float(np.max(all_pts[:, 1]) - np.min(all_pts[:, 1]))
    hand_span = max(bw, bh, 1.0)
    thickness = max(8, int(hand_span * 0.11))

    mask = np.zeros((h, w), dtype=np.uint8)
    for i, j in _HAND_CONNECTIONS:
        cv2.line(mask, tuple(pts_int[i]), tuple(pts_int[j]), 255, thickness, cv2.LINE_AA)

    hull = cv2.convexHull(all_int)
    cv2.fillConvexPoly(mask, hull, 255)
    mask = _fill_holes(mask)

    pad = max(HAND_EDGE_MARGIN_PX, int(hand_span * HAND_HULL_PADDING_RATIO))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = _clip_mask_to_hand_bbox(mask, pts_int, h, w)
    return _fill_holes(mask)


def segment_hand_skin(roi):
    """Fallback: warna kulit → background hitam."""
    try:
        h, w = roi.shape[:2]
        if max(h, w) > SEGMENT_MAX:
            work = cv2.resize(roi, (SEGMENT_MAX, SEGMENT_MAX), interpolation=cv2.INTER_AREA)
        else:
            work = roi

        glare = _glare_mask(work)
        skin = cv2.bitwise_and(_skin_color_mask(work), cv2.bitwise_not(glare))
        if cv2.countNonZero(skin) < MIN_SKIN_PIXELS:
            return np.zeros_like(roi)

        mask = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, _SKIN_CLOSE_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _SKIN_OPEN_KERNEL)
        mask = _fill_holes(mask)
        mask = _central_hand_component(mask)
        if cv2.countNonZero(mask) < 80:
            return np.zeros_like(roi)

        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            main = max(contours, key=cv2.contourArea)
            solid = np.zeros_like(mask)
            cv2.drawContours(solid, [main], -1, 255, thickness=cv2.FILLED)
            _, inner = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
            inner = cv2.bitwise_and(inner, solid)
            inner = cv2.bitwise_and(inner, cv2.bitwise_not(glare))
            mask = cv2.bitwise_or(mask, inner)
            mask = _fill_holes(mask)

        mask = cv2.dilate(mask, _HAND_MARGIN_KERNEL, iterations=1)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(glare))
        if cv2.countNonZero(mask) < 80:
            return np.zeros_like(roi)

        if work.shape[0] != h or work.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return _apply_mask_on_black(roi, mask)
    except Exception:
        return np.zeros_like(roi)


def segment_hand_mediapipe(roi_bgr):
    """MediaPipe landmark → convex hull mask → green screen."""
    if not _mediapipe_ready or _hand_landmarker is None:
        return np.zeros_like(roi_bgr)

    try:
        h, w = roi_bgr.shape[:2]
        if h < 8 or w < 8:
            return np.zeros_like(roi_bgr)

        detect = roi_bgr
        dh, dw = h, w
        if max(h, w) > MP_DETECT_MAX:
            scale = MP_DETECT_MAX / max(h, w)
            dw, dh = max(1, int(w * scale)), max(1, int(h * scale))
            detect = cv2.resize(roi_bgr, (dw, dh), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(detect, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with _silence_cpp_logs():
            result = _hand_landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return np.zeros_like(roi_bgr)

        mask_small = _landmarks_to_mask(dh, dw, result.hand_landmarks[0])
        if cv2.countNonZero(mask_small) < 50:
            return np.zeros_like(roi_bgr)

        if (dh, dw) != (h, w):
            mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            mask = mask_small

        return _apply_mask_on_black(roi_bgr, mask)
    except Exception:
        return np.zeros_like(roi_bgr)


def segment_hand(roi_bgr):
    """Utama: MediaPipe. Fallback: warna kulit."""
    out = segment_hand_mediapipe(roi_bgr)
    if cv2.countNonZero(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)) > 0:
        return out
    return segment_hand_skin(roi_bgr)
