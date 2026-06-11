// ============================================================
// BICARAISYARAT - FRONTEND LOGIC (Revamp)
// ============================================================

const alphabet = "0123456789abcdefghijklmnopqrstuvwxyz";
const ROI_BOX_SIZE = 300; // HARUS sama dengan ROI_BOX_SIZE di app.py

// --- Sensitivitas (preset) ---
const CONFIDENCE_HIGH_PCT = 60;   // hijau / deteksi "mantap"
const CONFIDENCE_MEDIUM_PCT = 45;

const SENSITIVITY = {
    fast:     { hold: 800,  conf: 0.55, label: "Cepat" },
    normal:   { hold: 900,  conf: 0.60, label: "Normal" },
    accurate: { hold: 1700, conf: 0.72, label: "Akurat" },
};
let holdDurationMs = SENSITIVITY.normal.hold;
let minConfidence = SENSITIVITY.normal.conf;

const PREDICTION_TICK_MS = 50;        // jadwalkan prediksi; antre jika request masih jalan
const ROI_CAPTURE_SIZE = 192;         // crop ROI untuk upload
const CAPTURE_MAX_WIDTH = 480;        // untuk simpan dataset (frame penuh)
const JPEG_QUALITY = 0.52;
const SMOOTH_BUFFER_SIZE = 4;         // voting huruf ke kalimat
const DISPLAY_BUFFER_SIZE = 5;        // smoothing tampilan huruf di UI
const DISPLAY_HOLD_MS = 550;          // tahan huruf terakhir (anti flicker ke "?")
const REPEAT_COOLDOWN_MS = 650;   // jeda sebelum huruf yang sama bisa diketik lagi
const WORD_SUGGESTION_DEBOUNCE_MS = 400;

// --- State umum ---
let activeLetterIdx = 10; // 'a'
let datasetCounts = {};
let isPredicting = false;
let predictPending = false;
let predictRequestId = 0;
let lastServerPreviewAt = 0;
let lastPrediction = null;
let displayBuffer = [];
let lastGoodDetectionAt = 0;
let smoothedDisplayConf = 0;
let currentMode = "translate";

// --- State penyusun kata ---
let currentWord = "";
let accumulatedWords = [];
let predictionBuffer = [];       // {char, conf}
let stableLetter = null;
let stableSince = 0;
let armed = true;                // siap commit huruf saat hold tercapai
let cooldownUntil = 0;
let isRefining = false;
let lastDetectedLanguage = "id";

// --- State prediksi kata ---
let wordSuggestions = [];
let isFetchingSuggestions = false;
let wordSuggestionDebounce = null;
let lastSuggestionRequest = "";
let lastScheduledKey = "";

// --- DOM: kamera ---
const video = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const loader = document.getElementById('webcam-loader');
const cnnPreview = document.getElementById('cnn-input-preview');
const cnnPreviewCtx = cnnPreview?.getContext('2d', { willReadFrequently: true });
const cameraStatus = document.getElementById('camera-status');

// --- DOM: deteksi ---
const detectedLetter = document.getElementById('detected-letter');
const detectionWrap = document.getElementById('detection-letter-wrap');
const confidencePercent = document.getElementById('confidence-percentage');
const confidenceProgress = document.getElementById('confidence-progress');
const top3List = document.getElementById('top3-list');

// --- DOM: tabs/panel ---
const tabTranslate = document.getElementById('tab-translate');
const tabCollect = document.getElementById('tab-collect');
const panelTranslate = document.getElementById('panel-translate');
const panelCollect = document.getElementById('panel-collect');

// --- DOM: composer ---
const activeCharDisplay = document.getElementById('active-char-display');
const currentWordField = document.getElementById('current-word-field');
const rawTextField = document.getElementById('raw-text-field');
const composerPreview = document.getElementById('composer-preview');
const refinedSentenceField = document.getElementById('refined-sentence-field');
const thoughtList = document.getElementById('thought-list');
const aiSource = document.getElementById('ai-source');
const defaultAiSourceStatus = aiSource?.dataset.defaultStatus || "Sistem Lokal";

function updateAiSourceChip(status, langLabel = "") {
    if (!aiSource || !status) return;
    const suffix = langLabel ? ` (${langLabel})` : "";
    aiSource.innerHTML = `<i class="fa-solid fa-microchip"></i> ${status}${suffix}`;
}
const progressCircle = document.querySelector('.progress-ring__circle');
const holdLabel = document.getElementById('hold-progress-label');
const wordSuggestionsEl = document.getElementById('word-suggestions');

// --- DOM: tombol composer ---
const btnSpace = document.getElementById('btn-space');
const btnBackspace = document.getElementById('btn-backspace');
const btnClearSentence = document.getElementById('btn-clear-sentence');
const btnRefine = document.getElementById('btn-refine');
const btnSpeak = document.getElementById('btn-speak');
const languageSelect = document.getElementById('language-select');
const sensitivitySelect = document.getElementById('sensitivity-select');

// --- DOM: collect ---
const selectChar = document.getElementById('char-select');
const targetBig = document.getElementById('target-big');
const btnPrev = document.getElementById('btn-prev-char');
const btnNext = document.getElementById('btn-next-char');
const btnCapture = document.getElementById('btn-capture');
const totalSamplesBadge = document.getElementById('total-samples');
const statsGrid = document.getElementById('stats-grid');

// --- Canvas tersembunyi untuk grab frame ---
const grabCanvas = document.createElement('canvas');
const grabCtx = grabCanvas.getContext('2d');
let videoFrameWidth = 640;
let videoFrameHeight = 480;

// ============================================================
// 1. INISIALISASI
// ============================================================

function populateDropdown() {
    selectChar.innerHTML = "";
    for (const char of alphabet) {
        const option = document.createElement('option');
        option.value = char;
        option.textContent = char.toUpperCase();
        selectChar.appendChild(option);
    }
    selectChar.value = alphabet[activeLetterIdx];
    updateTargetDisplay();
}

function getRoiCoords(w, h, boxSize = ROI_BOX_SIZE) {
    const box = Math.min(boxSize, w, h);
    const x_min = Math.max(0, Math.floor(w * 0.5 - box * 0.5));
    const y_min = Math.max(0, Math.floor(h * 0.5 - box * 0.5));
    return { x_min, y_min, box_size: box };
}

function syncWebcamLayout() {
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return;
    videoFrameWidth = vw;
    videoFrameHeight = vh;
    overlay.width = vw;
    overlay.height = vh;
    grabCanvas.width = vw;
    grabCanvas.height = vh;
    const stage = document.getElementById('webcam-stage');
    if (stage) stage.style.aspectRatio = `${vw} / ${vh}`;
}

async function setupWebcam() {
    try {
        loader.style.display = 'flex';
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
            audio: false
        });
        video.srcObject = stream;
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                syncWebcamLayout();
                loader.style.display = 'none';
                if (cameraStatus) cameraStatus.classList.add('online');
                resolve();
            };
        });
    } catch (err) {
        console.error("Error webcam:", err);
        if (cameraStatus) cameraStatus.classList.add('offline');
        loader.innerHTML = `
            <i class="fa-solid fa-triangle-exclamation" style="font-size:32px;color:#ef4444;"></i>
            <p style="color:#ef4444;font-weight:bold;">Gagal Mengakses Kamera</p>
            <p style="font-size:12px;color:#94a3b8;text-align:center;max-width:260px;">
                Pastikan izin kamera aktif dan tidak dipakai aplikasi lain.
            </p>`;
        throw err;
    }
}

async function fetchDatasetCounts() {
    try {
        const response = await fetch('/dataset_counts');
        if (response.ok) {
            datasetCounts = await response.json();
            updateStatsUI();
        }
    } catch (e) {
        console.error("Gagal mengambil statistik dataset:", e);
    }
}

function updateStatsUI() {
    statsGrid.innerHTML = "";
    let total = 0;
    for (const char of alphabet) {
        const count = datasetCounts[char] || 0;
        total += count;
        const box = document.createElement('div');
        box.className = `stat-box ${count > 0 ? 'has-data' : ''} ${alphabet[activeLetterIdx] === char ? 'active' : ''}`;
        box.id = `stat-box-${char}`;
        box.onclick = () => {
            activeLetterIdx = alphabet.indexOf(char);
            selectChar.value = char;
            updateTargetDisplay();
            updateActiveHighlight();
        };
        box.innerHTML = `<span class="stat-letter">${char}</span><span class="stat-count">${count}</span>`;
        statsGrid.appendChild(box);
    }
    if (totalSamplesBadge) totalSamplesBadge.textContent = `Total: ${total}`;
}

function updateActiveHighlight() {
    document.querySelectorAll('.stat-box').forEach(b => b.classList.remove('active'));
    const activeBox = document.getElementById(`stat-box-${alphabet[activeLetterIdx]}`);
    if (activeBox) activeBox.classList.add('active');
}

function updateTargetDisplay() {
    if (targetBig) targetBig.textContent = alphabet[activeLetterIdx].toUpperCase();
}

// ============================================================
// 2. MODE / TAB
// ============================================================

function switchMode(mode) {
    currentMode = mode;
    const isTranslate = mode === "translate";
    panelTranslate.classList.toggle('hidden', !isTranslate);
    panelCollect.classList.toggle('hidden', isTranslate);
    tabTranslate.classList.toggle('active', isTranslate);
    tabCollect.classList.toggle('active', !isTranslate);
    if (!isTranslate) updateStatsUI();
}

tabTranslate.addEventListener('click', () => switchMode('translate'));
tabCollect.addEventListener('click', () => switchMode('collect'));

// ============================================================
// 3. PREDIKSI & SMOOTHING
// ============================================================

function resetStableState() {
    predictionBuffer = [];
    displayBuffer = [];
    lastGoodDetectionAt = 0;
    smoothedDisplayConf = 0;
    stableLetter = null;
    stableSince = 0;
    armed = true;
    cooldownUntil = 0;
}

function pushDisplaySample(result) {
    if (!isValidPrediction(result)) return;
    displayBuffer.push({ char: result.char, conf: result.confidence || 0 });
    if (displayBuffer.length > DISPLAY_BUFFER_SIZE) displayBuffer.shift();
    lastGoodDetectionAt = Date.now();
}

function getSmoothedDisplay() {
    const handStale = Date.now() - lastGoodDetectionAt > DISPLAY_HOLD_MS;
    if (handStale || displayBuffer.length === 0) {
        return { char: '?', confidence: 0, hand_detected: false };
    }

    const tally = {};
    const confSum = {};
    for (const s of displayBuffer) {
        tally[s.char] = (tally[s.char] || 0) + 1;
        confSum[s.char] = (confSum[s.char] || 0) + s.conf;
    }
    let best = null;
    let bestCount = 0;
    for (const ch in tally) {
        if (tally[ch] > bestCount) {
            best = ch;
            bestCount = tally[ch];
        }
    }
    if (!best) {
        return { char: '?', confidence: 0, hand_detected: false };
    }

    const needed = Math.max(2, Math.ceil(displayBuffer.length * 0.45));
    if (bestCount < needed) {
        const last = displayBuffer[displayBuffer.length - 1];
        return {
            char: last.char,
            confidence: last.conf,
            hand_detected: true,
        };
    }

    return {
        char: best,
        confidence: confSum[best] / tally[best],
        hand_detected: true,
    };
}

// Voting: huruf paling sering muncul di buffer, dengan rata-rata confidence cukup
function computeSmoothedLetter() {
    if (predictionBuffer.length < Math.ceil(SMOOTH_BUFFER_SIZE * 0.6)) return null;
    const tally = {};
    const confSum = {};
    for (const p of predictionBuffer) {
        if (!p.char) continue;
        tally[p.char] = (tally[p.char] || 0) + 1;
        confSum[p.char] = (confSum[p.char] || 0) + p.conf;
    }
    let best = null, bestCount = 0;
    for (const ch in tally) {
        if (tally[ch] > bestCount) { best = ch; bestCount = tally[ch]; }
    }
    if (!best) return null;
    const needed = Math.ceil(predictionBuffer.length * 0.5);
    const avgConf = confSum[best] / tally[best];
    if (bestCount >= needed && avgConf >= minConfidence) return best;
    return null;
}

function isValidPrediction(result) {
    return result.hand_detected !== false && result.char && result.char !== "?";
}

function pushToBuffer(result) {
    const valid = isValidPrediction(result) && result.confidence >= (minConfidence - 0.1);
    predictionBuffer.push({ char: valid ? result.char : null, conf: result.confidence || 0 });
    if (predictionBuffer.length > SMOOTH_BUFFER_SIZE) predictionBuffer.shift();
}

function processStableLetter() {
    const now = Date.now();
    const smoothed = computeSmoothedLetter();

    if (!smoothed) {
        stableLetter = null;
        return;
    }

    // Huruf berubah -> mulai hitung hold baru
    if (smoothed !== stableLetter) {
        stableLetter = smoothed;
        stableSince = now;
        armed = now >= cooldownUntil;
        return;
    }

    // Masih dalam cooldown (baru saja commit huruf yang sama) -> tunggu lalu re-arm
    if (!armed) {
        if (now >= cooldownUntil) {
            armed = true;
            stableSince = now; // wajib tahan lagi untuk huruf berulang
        }
        return;
    }

    // Hold tercapai -> commit huruf
    if (now - stableSince >= holdDurationMs) {
        currentWord += stableLetter;
        armed = false;
        cooldownUntil = now + REPEAT_COOLDOWN_MS;
        triggerWordPulse();
        scheduleWordSuggestionsFetch();
    }
}

function getHoldProgressPercent() {
    if (!stableLetter || !armed) return 0;
    return Math.min(100, ((Date.now() - stableSince) / holdDurationMs) * 100);
}

function triggerWordPulse() {
    if (!currentWordField) return;
    currentWordField.classList.add('pulse-effect');
    setTimeout(() => currentWordField.classList.remove('pulse-effect'), 300);
}

function drawMirroredRegion(sx, sy, sw, sh, dw, dh) {
    grabCtx.save();
    grabCtx.scale(-1, 1);
    grabCtx.drawImage(video, sx, sy, sw, sh, -dw, 0, dw, dh);
    grabCtx.restore();
}

function getFrameBase64() {
    const vw = video.videoWidth || videoFrameWidth;
    const vh = video.videoHeight || videoFrameHeight;
    let tw = vw;
    let th = vh;
    if (vw > CAPTURE_MAX_WIDTH) {
        tw = CAPTURE_MAX_WIDTH;
        th = Math.max(1, Math.round(vh * (CAPTURE_MAX_WIDTH / vw)));
    }
    if (grabCanvas.width !== tw || grabCanvas.height !== th) {
        grabCanvas.width = tw;
        grabCanvas.height = th;
    }
    drawMirroredRegion(0, 0, vw, vh, tw, th);
    return grabCanvas.toDataURL('image/jpeg', JPEG_QUALITY);
}

/** Gambar ROI ke canvas (mirror). */
function captureRoiToCanvas() {
    const vw = video.videoWidth || videoFrameWidth;
    const vh = video.videoHeight || videoFrameHeight;
    const { x_min, y_min, box_size } = getRoiCoords(vw, vh);
    const out = ROI_CAPTURE_SIZE;
    if (grabCanvas.width !== out || grabCanvas.height !== out) {
        grabCanvas.width = out;
        grabCanvas.height = out;
    }
    drawMirroredRegion(x_min, y_min, box_size, box_size, out, out);
}

function roiCanvasToBlob() {
    return new Promise((resolve) => {
        grabCanvas.toBlob((blob) => resolve(blob), 'image/jpeg', JPEG_QUALITY);
    });
}

/** Placeholder hitam sampai preview segmented dari server tiba. */
function drawLocalCnnPreview() {
    if (!cnnPreviewCtx) return;
    if (lastServerPreviewAt) return;
    cnnPreviewCtx.fillStyle = '#05070d';
    cnnPreviewCtx.fillRect(0, 0, cnnPreview.width, cnnPreview.height);
}

function drawServerCnnPreview(dataUrl) {
    if (!cnnPreviewCtx || !dataUrl) return;
    const img = new Image();
    img.onload = () => {
        cnnPreviewCtx.drawImage(img, 0, 0, cnnPreview.width, cnnPreview.height);
        lastServerPreviewAt = Date.now();
    };
    img.src = dataUrl;
}

function localPreviewLoop() {
    drawLocalCnnPreview();
    requestAnimationFrame(localPreviewLoop);
}

function schedulePrediction() {
    if (video.paused || video.ended) return;
    captureRoiToCanvas();
    predictPending = true;
    if (!isPredicting) flushPrediction();
}

async function flushPrediction() {
    if (!predictPending || isPredicting) return;
    predictPending = false;
    isPredicting = true;
    const reqId = ++predictRequestId;

    const blob = await roiCanvasToBlob();
    if (!blob) {
        isPredicting = false;
        if (predictPending) flushPrediction();
        return;
    }

    const form = new FormData();
    form.append('image', blob, 'roi.jpg');
    form.append('roi_only', '1');
    form.append('skip_preview', '0');

    try {
        const response = await fetch('/predict', { method: 'POST', body: form });
        if (!response.ok) return;

        const result = await response.json();
        if (result.error) return;

        if (result.mini_roi) drawServerCnnPreview(result.mini_roi);

        pushDisplaySample(result);
        const display = getSmoothedDisplay();
        smoothedDisplayConf += (display.confidence - smoothedDisplayConf) * 0.35;

        lastPrediction = {
            char: display.char,
            confidence: smoothedDisplayConf,
            hand_detected: display.hand_detected,
            box: getRoiCoords(overlay.width, overlay.height),
            timestamp: Date.now()
        };

        updatePredictionUI(display);
        renderTop3(result.top3 || []);

        if (currentMode === 'translate') {
            pushToBuffer(result);
            processStableLetter();
        }
    } catch (err) {
        console.error('Gagal melakukan prediksi:', err);
    } finally {
        isPredicting = false;
        if (predictPending) flushPrediction();
    }
}

function predictionLoop() {
    setInterval(schedulePrediction, PREDICTION_TICK_MS);
}

function updatePredictionUI(result) {
    const handDetected = result.hand_detected !== false;
    const char = result.char;
    const pct = Math.round((result.confidence || 0) * 100);
    const letterEl = detectedLetter;
    if (letterEl && letterEl.textContent !== (handDetected && char && char !== '?' ? char : '?')) {
        letterEl.style.transform = 'scale(0.92)';
        requestAnimationFrame(() => { letterEl.style.transform = 'scale(1)'; });
    }

    detectionWrap.classList.remove('high', 'medium');
    if (handDetected && char && char !== "?") {
        detectedLetter.textContent = char;
        confidencePercent.textContent = `${pct}%`;
        confidenceProgress.style.width = `${pct}%`;
        if (pct >= CONFIDENCE_HIGH_PCT) {
            detectionWrap.classList.add('high');
            confidenceProgress.style.background = 'var(--green)';
        } else if (pct >= CONFIDENCE_MEDIUM_PCT) {
            detectionWrap.classList.add('medium');
            confidenceProgress.style.background = 'var(--orange)';
        } else {
            confidenceProgress.style.background = 'linear-gradient(90deg, var(--primary), var(--cyan))';
        }
    } else {
        detectedLetter.textContent = "?";
        confidencePercent.textContent = "0%";
        confidenceProgress.style.width = "0%";
    }
}

function renderTop3(top3) {
    if (!top3List) return;
    top3List.innerHTML = "";
    top3.forEach((item, i) => {
        const chip = document.createElement('span');
        chip.className = "top3-chip";
        chip.innerHTML = `<strong>${item.char}</strong> ${Math.round(item.confidence * 100)}%`;
        top3List.appendChild(chip);
    });
}

// ============================================================
// 4. SIMPAN DATASET
// ============================================================

async function captureAndSaveDataset() {
    const targetChar = alphabet[activeLetterIdx];
    const dataUrl = getFrameBase64();
    btnCapture.disabled = true;
    const original = btnCapture.innerHTML;
    btnCapture.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Menyimpan...`;
    try {
        const response = await fetch('/save_dataset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataUrl, char: targetChar })
        });
        if (response.ok) {
            const result = await response.json();
            if (result.status === "ok") {
                datasetCounts[targetChar] = result.count;
                updateStatsUI();
                updateActiveHighlight();
                btnCapture.innerHTML = `<i class="fa-solid fa-check"></i> Tersimpan! (${result.count})`;
            }
        }
    } catch (err) {
        console.error("Gagal menyimpan data latih:", err);
        btnCapture.innerHTML = `<i class="fa-solid fa-xmark"></i> Gagal`;
    } finally {
        setTimeout(() => { btnCapture.disabled = false; btnCapture.innerHTML = original; }, 1200);
    }
}

// ============================================================
// 5. OVERLAY (ROI BOX)
// ============================================================

function renderHUDLoop() {
    const fw = overlay.width || videoFrameWidth;
    const fh = overlay.height || videoFrameHeight;
    ctx.clearRect(0, 0, fw, fh);

    const { x_min, y_min, box_size } = getRoiCoords(fw, fh);

    let boxColor = "#6366f1";
    if (lastPrediction && (Date.now() - lastPrediction.timestamp < DISPLAY_HOLD_MS + 400)) {
        if (!lastPrediction.hand_detected) {
            boxColor = "#64748b";
        } else {
            const pct = Math.round(lastPrediction.confidence * 100);
            if (pct >= CONFIDENCE_HIGH_PCT) boxColor = "#10b981";
            else if (pct >= CONFIDENCE_MEDIUM_PCT) boxColor = "#f59e0b";
        }
    }

    ctx.strokeStyle = boxColor;
    ctx.lineWidth = 2.5;
    ctx.setLineDash([8, 6]);
    ctx.strokeRect(x_min, y_min, box_size, box_size);
    ctx.setLineDash([]);

    const len = 24;
    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    const corners = [
        [[x_min, y_min + len], [x_min, y_min], [x_min + len, y_min]],
        [[x_min + box_size - len, y_min], [x_min + box_size, y_min], [x_min + box_size, y_min + len]],
        [[x_min, y_min + box_size - len], [x_min, y_min + box_size], [x_min + len, y_min + box_size]],
        [[x_min + box_size - len, y_min + box_size], [x_min + box_size, y_min + box_size], [x_min + box_size, y_min + box_size - len]],
    ];
    for (const c of corners) {
        ctx.beginPath();
        ctx.moveTo(c[0][0], c[0][1]);
        ctx.lineTo(c[1][0], c[1][1]);
        ctx.lineTo(c[2][0], c[2][1]);
        ctx.stroke();
    }

    requestAnimationFrame(renderHUDLoop);
}

// ============================================================
// 6. PENYUSUN KALIMAT (UI)
// ============================================================

const RING_CIRCUMFERENCE = 87.96; // 2*PI*14
if (progressCircle) {
    progressCircle.style.strokeDasharray = `${RING_CIRCUMFERENCE} ${RING_CIRCUMFERENCE}`;
    progressCircle.style.strokeDashoffset = RING_CIRCUMFERENCE;
}

function setRingProgress(percent) {
    if (!progressCircle) return;
    progressCircle.style.strokeDashoffset = RING_CIRCUMFERENCE - (percent / 100 * RING_CIRCUMFERENCE);
}

function updateSentenceBuilderUI() {
    if (activeCharDisplay) activeCharDisplay.textContent = stableLetter ? stableLetter.toUpperCase() : "-";

    const rawSentence = accumulatedWords.join(" ");
    if (rawTextField) rawTextField.textContent = rawSentence;
    if (currentWordField) currentWordField.textContent = currentWord;
    if (composerPreview) composerPreview.classList.toggle("composer-empty", !rawSentence && !currentWord);

    const holdPct = getHoldProgressPercent();
    setRingProgress(holdPct);

    if (holdLabel) {
        if (stableLetter && armed) {
            const remaining = Math.max(0, (holdDurationMs - (Date.now() - stableSince)) / 1000);
            holdLabel.textContent = `Tahan ${remaining.toFixed(1)}s`;
        } else if (stableLetter && !armed) {
            holdLabel.textContent = "Tersimpan ✓";
        } else {
            holdLabel.textContent = "Tahan isyarat";
        }
    }
}

function renderWordSuggestions() {
    if (!wordSuggestionsEl) return;
    wordSuggestionsEl.innerHTML = "";

    if (!currentWord || currentWord.length < 2) {
        appendEmptySuggestion(currentWord.length === 1 ? "Ketik 1 huruf lagi..." : "Mulai mengetik isyarat...");
        return;
    }
    if (isFetchingSuggestions) {
        const btn = document.createElement("button");
        btn.className = "word-suggestion loading";
        btn.disabled = true;
        btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Mencari kata...`;
        wordSuggestionsEl.appendChild(btn);
        return;
    }
    if (wordSuggestions.length === 0) {
        appendEmptySuggestion("Tidak ada prediksi cocok");
        return;
    }

    const partial = currentWord.toLowerCase();
    wordSuggestions.slice(0, 3).forEach((word, i) => {
        const btn = document.createElement("button");
        const isTypoFix = partial.length >= 2 && !word.toLowerCase().startsWith(partial);
        btn.className = `word-suggestion${isTypoFix ? " word-suggestion-corrected" : ""}`;
        btn.title = isTypoFix ? `Koreksi "${partial}" → "${word}" (tekan ${i + 1})` : `Pilih "${word}" (tekan ${i + 1})`;
        const fix = isTypoFix ? `<span class="word-suggestion-fix" title="Auto-koreksi">✓</span>` : "";
        btn.innerHTML = `<span class="word-suggestion-key">${i + 1}</span><span class="word-suggestion-text">${word}</span>${fix}`;
        btn.addEventListener("click", () => selectWordSuggestion(i));
        wordSuggestionsEl.appendChild(btn);
    });
}

function appendEmptySuggestion(text) {
    const btn = document.createElement("button");
    btn.className = "word-suggestion empty";
    btn.disabled = true;
    btn.textContent = text;
    wordSuggestionsEl.appendChild(btn);
}

function scheduleWordSuggestionsFetch() {
    const partial = currentWord.toLowerCase().trim();
    const key = `${partial}|${accumulatedWords.join(" ")}`;
    if (partial.length < 2) {
        if (lastScheduledKey !== "") {
            lastScheduledKey = "";
            wordSuggestions = [];
            lastSuggestionRequest = "";
            if (wordSuggestionDebounce) { clearTimeout(wordSuggestionDebounce); wordSuggestionDebounce = null; }
            renderWordSuggestions();
        }
        return;
    }
    if (key === lastScheduledKey) return;
    lastScheduledKey = key;
    if (wordSuggestionDebounce) clearTimeout(wordSuggestionDebounce);
    wordSuggestionDebounce = setTimeout(fetchWordSuggestions, WORD_SUGGESTION_DEBOUNCE_MS);
}

async function fetchWordSuggestions() {
    const partial = currentWord.toLowerCase().trim();
    if (partial.length < 2) { wordSuggestions = []; renderWordSuggestions(); return; }

    const reqKey = `${partial}|${accumulatedWords.join(" ")}|${languageSelect ? languageSelect.value : "auto"}`;
    if (reqKey === lastSuggestionRequest && wordSuggestions.length > 0) { renderWordSuggestions(); return; }

    isFetchingSuggestions = true;
    renderWordSuggestions();
    try {
        const response = await fetch('/predict_words', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                partial_word: partial,
                context_words: accumulatedWords,
                language: languageSelect ? languageSelect.value : "auto"
            })
        });
        if (response.ok) {
            const result = await response.json();
            if (!result.error && partial === currentWord.toLowerCase().trim()) {
                wordSuggestions = result.suggestions || [];
                lastSuggestionRequest = reqKey;
                if (result.detected_language) lastDetectedLanguage = result.detected_language;
                if (result.ai_status) updateAiSourceChip(result.ai_status);
            }
        }
    } catch (e) {
        console.error("Gagal mengambil prediksi kata:", e);
    } finally {
        isFetchingSuggestions = false;
        renderWordSuggestions();
    }
}

function selectWordSuggestion(index) {
    if (index < 0 || index >= wordSuggestions.length) return;
    accumulatedWords.push(wordSuggestions[index].toLowerCase());
    currentWord = "";
    wordSuggestions = [];
    lastSuggestionRequest = "";
    lastScheduledKey = "";
    resetStableState();
    updateSentenceBuilderUI();
    renderWordSuggestions();
}

function handleSpaceAction() {
    if (currentWord) {
        accumulatedWords.push(currentWord.toLowerCase());
        currentWord = "";
        wordSuggestions = [];
        lastSuggestionRequest = "";
        lastScheduledKey = "";
        resetStableState();
        updateSentenceBuilderUI();
        renderWordSuggestions();
    }
}

function handleBackspaceAction() {
    if (currentWord) currentWord = currentWord.slice(0, -1);
    else if (accumulatedWords.length > 0) accumulatedWords.pop();
    resetStableState();
    scheduleWordSuggestionsFetch();
    updateSentenceBuilderUI();
}

function handleClearAction() {
    currentWord = "";
    accumulatedWords = [];
    wordSuggestions = [];
    lastSuggestionRequest = "";
    lastScheduledKey = "";
    resetStableState();
    if (refinedSentenceField) refinedSentenceField.textContent = "-";
    if (aiSource) aiSource.innerHTML = `<i class="fa-solid fa-microchip"></i> ${defaultAiSourceStatus}`;
    if (thoughtList) thoughtList.innerHTML = `<li class="empty-thought">Menunggu kata terkumpul...</li>`;
    updateSentenceBuilderUI();
    renderWordSuggestions();
}

async function refineSentenceWithAI() {
    if (isRefining) return;
    let rawSentence = accumulatedWords.join(" ");
    if (currentWord) rawSentence += (rawSentence ? " " : "") + currentWord;
    if (!rawSentence.trim()) return;

    isRefining = true;
    if (btnRefine) { btnRefine.disabled = true; btnRefine.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Merangkai...`; }
    if (thoughtList) thoughtList.innerHTML = `<li><i class="fa-solid fa-spinner fa-spin"></i> Menganalisis susunan kata...</li>`;

    try {
        const response = await fetch('/refine_sentence', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ raw_text: rawSentence, language: languageSelect ? languageSelect.value : "auto" })
        });
        if (response.ok) {
            const result = await response.json();
            if (!result.error) {
                if (refinedSentenceField) refinedSentenceField.textContent = result.refined || "-";
                const langLabel = result.detected_language === "en" ? "EN" : "ID";
                if (result.ai_status) updateAiSourceChip(result.ai_status, langLabel);
                else if (result.source) updateAiSourceChip(result.source, langLabel);
                if (result.detected_language) lastDetectedLanguage = result.detected_language;
                if (thoughtList) {
                    thoughtList.innerHTML = "";
                    (result.thought && result.thought.length ? result.thought : ["Penyusunan kalimat selesai."]).forEach(step => {
                        const li = document.createElement('li');
                        li.textContent = step;
                        thoughtList.appendChild(li);
                    });
                }
            }
        }
    } catch (e) {
        console.error("Gagal merangkai kalimat:", e);
        if (thoughtList) thoughtList.innerHTML = `<li style="color:#ef4444;">Gagal menghubungi server.</li>`;
    } finally {
        isRefining = false;
        if (btnRefine) { btnRefine.disabled = false; btnRefine.innerHTML = `<i class="fa-solid fa-language"></i> Rangkai`; }
    }
}

function speakSentence() {
    if (!refinedSentenceField) return;
    const text = refinedSentenceField.textContent;
    if (!text || text === "-" || text === "") return;
    if ('speechSynthesis' in window) {
        const utter = new SpeechSynthesisUtterance(text);
        const sel = languageSelect ? languageSelect.value : "auto";
        utter.lang = (sel === "en" || (sel === "auto" && lastDetectedLanguage === "en")) ? "en-US" : "id-ID";
        window.speechSynthesis.speak(utter);
    } else {
        alert("Text-to-Speech tidak didukung di browser ini.");
    }
}

// ============================================================
// 7. EVENT LISTENERS
// ============================================================

selectChar.addEventListener('change', () => {
    activeLetterIdx = alphabet.indexOf(selectChar.value);
    updateTargetDisplay();
    updateActiveHighlight();
});

btnPrev.addEventListener('click', () => changeTarget(-1));
btnNext.addEventListener('click', () => changeTarget(1));
btnCapture.addEventListener('click', captureAndSaveDataset);

function changeTarget(delta) {
    activeLetterIdx = (activeLetterIdx + delta + alphabet.length) % alphabet.length;
    selectChar.value = alphabet[activeLetterIdx];
    updateTargetDisplay();
    updateActiveHighlight();
}

if (btnSpace) btnSpace.addEventListener('click', handleSpaceAction);
if (btnBackspace) btnBackspace.addEventListener('click', handleBackspaceAction);
if (btnClearSentence) btnClearSentence.addEventListener('click', handleClearAction);
if (btnRefine) btnRefine.addEventListener('click', refineSentenceWithAI);
if (btnSpeak) btnSpeak.addEventListener('click', speakSentence);

if (languageSelect) languageSelect.addEventListener('change', () => {
    lastSuggestionRequest = ""; lastScheduledKey = "";
    scheduleWordSuggestionsFetch();
});

if (sensitivitySelect) sensitivitySelect.addEventListener('change', () => {
    const preset = SENSITIVITY[sensitivitySelect.value] || SENSITIVITY.normal;
    holdDurationMs = preset.hold;
    minConfidence = preset.conf;
    resetStableState();
});

document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
    const key = e.key.toLowerCase();

    if (key === 's') { e.preventDefault(); switchMode('collect'); captureAndSaveDataset(); }
    else if (key === '[') { e.preventDefault(); changeTarget(-1); }
    else if (key === ']') { e.preventDefault(); changeTarget(1); }
    else if (e.key === ' ' || e.code === 'Space') { e.preventDefault(); handleSpaceAction(); }
    else if (e.key === 'Backspace') { e.preventDefault(); handleBackspaceAction(); }
    else if (e.key === 'Enter') { e.preventDefault(); refineSentenceWithAI(); }
    else if (e.key === 'Escape') { e.preventDefault(); handleClearAction(); }
    else if (e.key >= '1' && e.key <= '3') { e.preventDefault(); selectWordSuggestion(parseInt(e.key, 10) - 1); }
});

// ============================================================
// 8. START
// ============================================================

async function fetchAiStatus() {
    try {
        const response = await fetch('/ai_status');
        if (response.ok) {
            const data = await response.json();
            if (data.status) updateAiSourceChip(data.status);
        }
    } catch (_) { /* chip tetap pakai nilai dari server saat render HTML */ }
}

async function startApp() {
    populateDropdown();
    requestAnimationFrame(localPreviewLoop);
    await fetchAiStatus();
    await fetchDatasetCounts();
    renderWordSuggestions();
    updateSentenceBuilderUI();
    try {
        await setupWebcam();
        window.addEventListener('resize', syncWebcamLayout);
        predictionLoop();
        setInterval(updateSentenceBuilderUI, 100);
        requestAnimationFrame(renderHUDLoop);
    } catch (err) {
        console.error("Gagal memulai aplikasi:", err);
    }
}

window.addEventListener('DOMContentLoaded', startApp);
