// GLOBAL STATE
const alphabet = "0123456789abcdefghijklmnopqrstuvwxyz";
let activeLetterIdx = 10; // Default: 'a'
let datasetCounts = {};
let isPredicting = false;
let lastPrediction = null;

// SENTENCE BUILDER STATE
let currentWord = "";
let accumulatedWords = [];
let stableLetter = null;
let stableLetterSince = null;
const HOLD_DURATION_MS = 1500;
const MIN_CONFIDENCE = 0.65;
const PREDICTION_INTERVAL_MS = 120;
let isRefining = false;
let hasTypedCurrentSign = false;
let lastDetectedLanguage = "id";

// WORD PREDICTION STATE
let wordSuggestions = [];
let isFetchingSuggestions = false;
let wordSuggestionDebounce = null;
let lastSuggestionRequest = "";
let lastScheduledPartial = "";
const WORD_SUGGESTION_DEBOUNCE_MS = 450;

// DOM ELEMENTS
const video = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const selectChar = document.getElementById('char-select');
const btnPrev = document.getElementById('btn-prev-char');
const btnNext = document.getElementById('btn-next-char');
const btnCapture = document.getElementById('btn-capture');
const loader = document.getElementById('webcam-loader');
const predictedLetter = document.getElementById('predicted-letter');
const confidencePercent = document.getElementById('confidence-percentage');
const confidenceProgress = document.getElementById('confidence-progress');
const cnnPreview = document.getElementById('cnn-input-preview');
const totalSamplesBadge = document.getElementById('total-samples');
const statsGrid = document.getElementById('stats-grid');

// DOM Sentence Builder
const activeCharDisplay = document.getElementById('active-char-display');
const currentWordField = document.getElementById('current-word-field');
const rawTextField = document.getElementById('raw-text-field');
const refinedSentenceField = document.getElementById('refined-sentence-field');
const thoughtList = document.getElementById('thought-list');
const aiSource = document.getElementById('ai-source');
const defaultAiSourceStatus = aiSource?.dataset.defaultStatus || "Sistem Lokal";
const progressCircle = document.querySelector('.progress-ring__circle');

// Sentence Builder Buttons
const btnSpace = document.getElementById('btn-space');
const btnBackspace = document.getElementById('btn-backspace');
const btnClearSentence = document.getElementById('btn-clear-sentence');
const btnRefine = document.getElementById('btn-refine');
const btnSpeak = document.getElementById('btn-speak');
const languageSelect = document.getElementById('language-select');
const holdProgressBar = document.getElementById('hold-progress-bar');
const holdProgressLabel = document.getElementById('hold-progress-label');
const wordSuggestionsEl = document.getElementById('word-suggestions');
const composerPreview = document.getElementById('composer-preview');

// Canvas tersembunyi untuk grab frame gambar mentah
const grabCanvas = document.createElement('canvas');
const grabCtx = grabCanvas.getContext('2d');
let videoFrameWidth = 640;
let videoFrameHeight = 480;

// --- 1. INISIALISASI ---

// Populasi Dropdown target karakter
function populateDropdown() {
    selectChar.innerHTML = "";
    for (let char of alphabet) {
        const option = document.createElement('option');
        option.value = char;
        option.textContent = char.toUpperCase();
        selectChar.appendChild(option);
    }
    selectChar.value = alphabet[activeLetterIdx];
}

function getRoiCoords(w, h, boxSize = 240) {
    const x_min = Math.max(0, Math.floor(w * 0.55 - boxSize * 0.5));
    const y_min = Math.max(0, Math.floor(h * 0.5 - boxSize * 0.5));
    const x_max = Math.min(w, x_min + boxSize);
    const y_max = Math.min(h, y_min + boxSize);
    return {
        x_min,
        y_min,
        box_size: Math.min(boxSize, x_max - x_min, y_max - y_min)
    };
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
    if (stage) {
        stage.style.aspectRatio = `${vw} / ${vh}`;
    }
}

// Buka webcam
async function setupWebcam() {
    try {
        loader.style.display = 'flex';
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        });
        video.srcObject = stream;
        
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                syncWebcamLayout();
                loader.style.display = 'none';
                resolve();
            };
        });
    } catch (err) {
        console.error("Error webcam:", err);
        loader.innerHTML = `
            <i class="fa-solid fa-triangle-exclamation" style="font-size: 32px; color: #ef4444;"></i>
            <p style="color: #ef4444; font-weight: bold; margin-top: 10px;">Gagal Mengakses Kamera</p>
            <p style="font-size: 12px; color: #9ca3af; text-align: center; max-width: 250px; margin-top: 5px;">
                Pastikan izin kamera browser aktif dan tidak ada aplikasi lain yang menggunakan kamera.
            </p>
        `;
        throw err;
    }
}

// Fetch total foto terkumpul saat pertama kali dimuat
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

// Update UI Grid Statistik
function updateStatsUI() {
    statsGrid.innerHTML = "";
    let totalSamples = 0;
    
    for (let char of alphabet) {
        const count = datasetCounts[char] || 0;
        totalSamples += count;
        
        const box = document.createElement('div');
        box.className = `stat-box ${count > 0 ? 'has-data' : ''} ${alphabet[activeLetterIdx] === char ? 'active' : ''}`;
        box.id = `stat-box-${char}`;
        box.onclick = () => {
            activeLetterIdx = alphabet.indexOf(char);
            selectChar.value = char;
            updateActiveHighlight();
        };
        
        box.innerHTML = `
            <span class="stat-letter">${char}</span>
            <span class="stat-count">${count}</span>
        `;
        statsGrid.appendChild(box);
    }
    
    totalSamplesBadge.textContent = `Total: ${totalSamples} Foto`;
}

// Update status highlight target aktif tanpa merender ulang seluruh grid
function updateActiveHighlight() {
    document.querySelectorAll('.stat-box').forEach(box => {
        box.classList.remove('active');
    });
    const activeBox = document.getElementById(`stat-box-${alphabet[activeLetterIdx]}`);
    if (activeBox) {
        activeBox.classList.add('active');
    }
}

// --- 2. LOGIKA UTAMA (PREDIKSI & DATASET) ---

function resetStableState() {
    stableLetter = null;
    stableLetterSince = null;
    hasTypedCurrentSign = false;
}

function isValidPrediction(result) {
    return result.hand_detected !== false
        && result.char
        && result.char !== "?"
        && result.confidence >= MIN_CONFIDENCE;
}

function getHoldProgressPercent() {
    if (!stableLetter || !stableLetterSince || hasTypedCurrentSign) return 0;
    const elapsed = Date.now() - stableLetterSince;
    return Math.min(100, (elapsed / HOLD_DURATION_MS) * 100);
}

function triggerWordPulse() {
    if (!currentWordField) return;
    currentWordField.classList.add('pulse-effect');
    setTimeout(() => currentWordField.classList.remove('pulse-effect'), 300);
}

function processStableLetter(result) {
    const now = Date.now();

    if (!isValidPrediction(result)) {
        resetStableState();
        return;
    }

    if (result.char === stableLetter) {
        if (!hasTypedCurrentSign && stableLetterSince && (now - stableLetterSince) >= HOLD_DURATION_MS) {
            currentWord += stableLetter;
            hasTypedCurrentSign = true;
            triggerWordPulse();
        }
        return;
    }

    stableLetter = result.char;
    stableLetterSince = now;
    hasTypedCurrentSign = false;
}

// Grab frame dan ubah ke format base64 (resolusi asli kamera, tanpa distorsi)
function getFrameBase64() {
    const w = video.videoWidth || videoFrameWidth;
    const h = video.videoHeight || videoFrameHeight;
    if (grabCanvas.width !== w || grabCanvas.height !== h) {
        grabCanvas.width = w;
        grabCanvas.height = h;
    }
    grabCtx.drawImage(video, 0, 0, w, h);
    return grabCanvas.toDataURL('image/jpeg', 0.85);
}

// Kirim frame ke server untuk deteksi/prediksi CNN
async function runPrediction() {
    if (isPredicting || video.paused || video.ended) return;
    
    isPredicting = true;
    const dataUrl = getFrameBase64();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: dataUrl
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            if (!result.error) {
                lastPrediction = {
                    char: result.char,
                    confidence: result.confidence,
                    hand_detected: result.hand_detected !== false,
                    x_min: result.x_min,
                    y_min: result.y_min,
                    x_max: result.x_max,
                    y_max: result.y_max,
                    timestamp: Date.now()
                };
                
                if (result.mini_roi) {
                    cnnPreview.src = result.mini_roi;
                }
                
                updatePredictionUI(result.char, result.confidence, result.hand_detected);
                processStableLetter(result);
                updateSentenceBuilderUI();
            }
        }
    } catch (err) {
        console.error("Gagal melakukan prediksi:", err);
    } finally {
        isPredicting = false;
    }
}

// Update prediksi utama di panel HUD kanan
function updatePredictionUI(char, confidence, handDetected = true) {
    const percentage = Math.round(confidence * 100);
    const wrapper = document.querySelector('.predicted-char-wrapper');
    
    if (handDetected && char && char !== "?") {
        predictedLetter.textContent = char;
        confidencePercent.textContent = `${percentage}%`;
        confidenceProgress.style.width = `${percentage}%`;
        
        // Atur style berdasarkan confidence
        wrapper.classList.remove('high-confidence', 'medium-confidence');
        if (percentage > 60) {
            wrapper.classList.add('high-confidence');
            confidenceProgress.style.background = 'var(--accent-green)';
        } else if (percentage > 30) {
            wrapper.classList.add('medium-confidence');
            confidenceProgress.style.background = 'var(--accent-orange)';
        } else {
            confidenceProgress.style.background = 'var(--accent-blue)';
        }
    } else {
        predictedLetter.textContent = "?";
        confidencePercent.textContent = "0%";
        confidenceProgress.style.width = "0%";
        wrapper.classList.remove('high-confidence', 'medium-confidence');
    }
}

// Kirim perintah simpan dataset ke Python backend
async function captureAndSaveDataset() {
    const targetChar = alphabet[activeLetterIdx];
    const dataUrl = getFrameBase64();
    
    btnCapture.disabled = true;
    btnCapture.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Menyimpan...`;
    
    try {
        const response = await fetch('/save_dataset', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: dataUrl,
                char: targetChar
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            if (result.status === "ok") {
                // Update counter dataset
                datasetCounts[targetChar] = result.count;
                updateStatsUI();
                
                // Beri efek notifikasi visual pada tombol
                btnCapture.style.background = 'var(--accent-green)';
                btnCapture.style.color = '#fff';
                btnCapture.innerHTML = `<i class="fa-solid fa-check"></i> Tersimpan! (${result.count})`;
            }
        }
    } catch (err) {
        console.error("Gagal menyimpan data latih:", err);
    } finally {
        setTimeout(() => {
            btnCapture.disabled = false;
            btnCapture.style.background = 'var(--accent-cyan)';
            btnCapture.style.color = '#fff';
            btnCapture.innerHTML = `<i class="fa-solid fa-camera-retro"></i> Simpan Gambar (Shortcut: S)`;
        }, 1500);
    }
}

// --- 3. GRAPHICS OVERLAY DRAW LOOP ---

// Render loop HUD overlay (kotak ROI tetap dan hasil prediksi)
function renderHUDLoop() {
    const fw = overlay.width || videoFrameWidth;
    const fh = overlay.height || videoFrameHeight;
    if (overlay.width !== fw || overlay.height !== fh) {
        overlay.width = fw;
        overlay.height = fh;
    }
    ctx.clearRect(0, 0, fw, fh);

    const { x_min, y_min, box_size } = getRoiCoords(fw, fh);
    
    let boxColor = "#3b82f6"; // Biru default
    let predText = "";
    
    if (lastPrediction && (Date.now() - lastPrediction.timestamp < 1000)) {
        if (!lastPrediction.hand_detected) {
            boxColor = "#64748b";
            predText = "Tidak ada tangan terdeteksi";
        } else {
            const percentage = Math.round(lastPrediction.confidence * 100);
            if (percentage > 60) {
                boxColor = "#10b981";
            } else if (percentage > 30) {
                boxColor = "#f59e0b";
            }

            if (lastPrediction.char && lastPrediction.char !== "?") {
                predText = `Prediksi: ${lastPrediction.char.toUpperCase()} (${percentage}%)`;
            }
        }
    }
    
    // Gambar kotak ROI tetap
    ctx.strokeStyle = boxColor;
    ctx.lineWidth = 3;
    ctx.lineJoin = "round";
    ctx.strokeRect(x_min, y_min, box_size, box_size);
    
    // Gambar sudut dekoratif HUD untuk estetika premium
    const len = 20;
    ctx.strokeStyle = boxColor;
    ctx.lineWidth = 5;
    
    // Pojok Kiri Atas
    ctx.beginPath();
    ctx.moveTo(x_min - 2, y_min - 2 + len);
    ctx.lineTo(x_min - 2, y_min - 2);
    ctx.lineTo(x_min - 2 + len, y_min - 2);
    ctx.stroke();
    
    // Pojok Kanan Atas
    ctx.beginPath();
    ctx.moveTo(x_min + box_size + 2 - len, y_min - 2);
    ctx.lineTo(x_min + box_size + 2, y_min - 2);
    ctx.lineTo(x_min + box_size + 2, y_min - 2 + len);
    ctx.stroke();
    
    // Pojok Kiri Bawah
    ctx.beginPath();
    ctx.moveTo(x_min - 2, y_min + box_size + 2 - len);
    ctx.lineTo(x_min - 2, y_min + box_size + 2);
    ctx.lineTo(x_min - 2 + len, y_min + box_size + 2);
    ctx.stroke();
    
    // Pojok Kanan Bawah
    ctx.beginPath();
    ctx.moveTo(x_min + box_size + 2 - len, y_min + box_size + 2);
    ctx.lineTo(x_min + box_size + 2, y_min + box_size + 2);
    ctx.lineTo(x_min + box_size + 2, y_min + box_size + 2 - len);
    ctx.stroke();
    
    // Gambar teks prediksi di atas kotak
    if (predText) {
        ctx.fillStyle = boxColor;
        ctx.font = "bold 16px 'Outfit', sans-serif";
        
        // Gambar background teks kecil untuk legibilitas
        const textWidth = ctx.measureText(predText).width;
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
        ctx.fillRect(x_min, y_min - 30, textWidth + 16, 24);
        
        ctx.fillStyle = boxColor;
        ctx.fillText(predText, x_min + 8, y_min - 13);
    }
    
    requestAnimationFrame(renderHUDLoop);
}

// --- 4. EVENT LISTENERS & SHORTCUTS ---

// Ganti huruf target
selectChar.addEventListener('change', () => {
    activeLetterIdx = alphabet.indexOf(selectChar.value);
    updateActiveHighlight();
});

btnPrev.addEventListener('click', () => {
    activeLetterIdx = (activeLetterIdx - 1 + alphabet.length) % alphabet.length;
    selectChar.value = alphabet[activeLetterIdx];
    updateActiveHighlight();
});

btnNext.addEventListener('click', () => {
    activeLetterIdx = (activeLetterIdx + 1) % alphabet.length;
    selectChar.value = alphabet[activeLetterIdx];
    updateActiveHighlight();
});

// Button capture
btnCapture.addEventListener('click', captureAndSaveDataset);

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Cek agar tidak terpicu ketika mengetik di form input/select (jika ada)
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') {
        return;
    }
    
    const key = e.key.toLowerCase();
    
    if (key === 's') {
        e.preventDefault();
        captureAndSaveDataset();
    } else if (key === '[') {
        e.preventDefault();
        activeLetterIdx = (activeLetterIdx - 1 + alphabet.length) % alphabet.length;
        selectChar.value = alphabet[activeLetterIdx];
        updateActiveHighlight();
    } else if (key === ']') {
        e.preventDefault();
        activeLetterIdx = (activeLetterIdx + 1) % alphabet.length;
        selectChar.value = alphabet[activeLetterIdx];
        updateActiveHighlight();
    } else if (e.key === ' ' || e.code === 'Space') {
        e.preventDefault();
        handleSpaceAction();
    } else if (e.key === 'Backspace') {
        e.preventDefault();
        handleBackspaceAction();
    } else if (e.key === 'Enter') {
        e.preventDefault();
        refineSentenceWithAI();
    } else if (e.key === 'Escape') {
        e.preventDefault();
        handleClearAction();
    } else if (e.key >= '1' && e.key <= '3') {
        e.preventDefault();
        selectWordSuggestion(parseInt(e.key, 10) - 1);
    }
});

// --- SENTENCE BUILDER CORE LOGIC & ACTIONS ---

const circleCircumference = 43.98;
if (progressCircle) {
    progressCircle.style.strokeDasharray = `${circleCircumference} ${circleCircumference}`;
    progressCircle.style.strokeDashoffset = circleCircumference;
}

function setProgress(percent) {
    if (!progressCircle) return;
    const offset = circleCircumference - (percent / 100 * circleCircumference);
    progressCircle.style.strokeDashoffset = offset;
}

function updateSentenceBuilderUI() {
    if (activeCharDisplay) {
        activeCharDisplay.textContent = stableLetter ? stableLetter.toUpperCase() : "-";
    }

    const rawSentence = accumulatedWords.join(" ");
    if (rawTextField) {
        rawTextField.textContent = rawSentence || (!currentWord ? "-" : "");
    }
    if (currentWordField) {
        if (currentWord) {
            currentWordField.textContent = currentWord;
            currentWordField.style.display = "";
        } else {
            currentWordField.textContent = "";
            currentWordField.style.display = "none";
        }
    }
    if (composerPreview) {
        composerPreview.classList.toggle("composer-empty", !rawSentence && !currentWord);
    }

    scheduleWordSuggestionsFetch();

    const holdPct = getHoldProgressPercent();
    setProgress(holdPct);

    if (holdProgressBar) {
        holdProgressBar.style.width = `${holdPct}%`;
    }
    if (holdProgressLabel) {
        if (stableLetter && !hasTypedCurrentSign) {
            const remaining = Math.max(0, (HOLD_DURATION_MS - (Date.now() - (stableLetterSince || Date.now()))) / 1000);
            holdProgressLabel.textContent = `Tahan ${remaining.toFixed(1)} dtk`;
        } else {
            holdProgressLabel.textContent = "Tahan ~1,5 dtk";
        }
    }
}

function renderWordSuggestions() {
    if (!wordSuggestionsEl) return;

    wordSuggestionsEl.innerHTML = "";

    if (!currentWord || currentWord.length < 2) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "word-suggestion empty";
        btn.disabled = true;
        btn.textContent = currentWord.length === 1
            ? "Ketik 1 huruf lagi untuk prediksi..."
            : "Mengetik isyarat...";
        wordSuggestionsEl.appendChild(btn);
        return;
    }

    if (isFetchingSuggestions) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "word-suggestion loading";
        btn.disabled = true;
        btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Mencari kata...`;
        wordSuggestionsEl.appendChild(btn);
        return;
    }

    if (wordSuggestions.length === 0) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "word-suggestion empty";
        btn.disabled = true;
        btn.textContent = "Tidak ada prediksi cocok";
        wordSuggestionsEl.appendChild(btn);
        return;
    }

    wordSuggestions.slice(0, 3).forEach((word, index) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "word-suggestion";
        btn.title = `Pilih "${word}" (tekan ${index + 1})`;
        btn.innerHTML = `<span class="word-suggestion-key">${index + 1}</span><span class="word-suggestion-text">${word}</span>`;
        btn.addEventListener("click", () => selectWordSuggestion(index));
        wordSuggestionsEl.appendChild(btn);
    });

    while (wordSuggestionsEl.children.length < 3) {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "word-suggestion empty";
        btn.disabled = true;
        btn.textContent = "—";
        wordSuggestionsEl.appendChild(btn);
    }
}

function scheduleWordSuggestionsFetch() {
    const partial = currentWord.toLowerCase().trim();
    const scheduleKey = `${partial}|${accumulatedWords.join(" ")}`;

    if (partial.length < 2) {
        if (lastScheduledPartial !== "") {
            lastScheduledPartial = "";
            wordSuggestions = [];
            lastSuggestionRequest = "";
            if (wordSuggestionDebounce) {
                clearTimeout(wordSuggestionDebounce);
                wordSuggestionDebounce = null;
            }
            renderWordSuggestions();
        }
        return;
    }

    if (scheduleKey === lastScheduledPartial) {
        return;
    }

    lastScheduledPartial = scheduleKey;
    if (wordSuggestionDebounce) {
        clearTimeout(wordSuggestionDebounce);
    }
    wordSuggestionDebounce = setTimeout(fetchWordSuggestions, WORD_SUGGESTION_DEBOUNCE_MS);
}

async function fetchWordSuggestions() {
    const partial = currentWord.toLowerCase().trim();
    if (partial.length < 2) {
        wordSuggestions = [];
        renderWordSuggestions();
        return;
    }

    const requestKey = `${partial}|${accumulatedWords.join(" ")}|${languageSelect ? languageSelect.value : "auto"}`;
    if (requestKey === lastSuggestionRequest && wordSuggestions.length > 0) {
        renderWordSuggestions();
        return;
    }

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
                lastSuggestionRequest = requestKey;
                if (result.detected_language) {
                    lastDetectedLanguage = result.detected_language;
                }
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
    lastScheduledPartial = "";
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
        lastScheduledPartial = "";
        updateSentenceBuilderUI();
        renderWordSuggestions();
    }
}

function handleBackspaceAction() {
    if (currentWord) {
        currentWord = currentWord.slice(0, -1);
    } else if (accumulatedWords.length > 0) {
        accumulatedWords.pop();
    }
    updateSentenceBuilderUI();
}

function handleClearAction() {
    currentWord = "";
    accumulatedWords = [];
    wordSuggestions = [];
    lastSuggestionRequest = "";
    lastScheduledPartial = "";
    resetStableState();
    if (refinedSentenceField) refinedSentenceField.textContent = "-";
    if (aiSource) aiSource.textContent = defaultAiSourceStatus;
    if (thoughtList) {
        thoughtList.innerHTML = `<li class="empty-thought">Menunggu kata terkumpul...</li>`;
    }
    updateSentenceBuilderUI();
    renderWordSuggestions();
}

async function refineSentenceWithAI() {
    if (isRefining) return;
    
    let rawSentence = accumulatedWords.join(" ");
    if (currentWord) {
        rawSentence += (rawSentence ? " " : "") + currentWord;
    }
    
    if (!rawSentence.trim()) return;
    
    isRefining = true;
    if (btnRefine) {
        btnRefine.disabled = true;
        btnRefine.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Merangkai...`;
    }
    
    if (thoughtList) {
        thoughtList.innerHTML = `<li><i class="fa-solid fa-spinner fa-spin"></i> Sistem sedang menganalisis susunan kata mentah...</li>`;
    }
    
    try {
        const response = await fetch('/refine_sentence', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                raw_text: rawSentence,
                language: languageSelect ? languageSelect.value : "auto"
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            if (!result.error) {
                if (refinedSentenceField) refinedSentenceField.textContent = result.refined;
                if (aiSource) {
                    const langLabel = result.detected_language === "en" ? "EN" : "ID";
                    aiSource.textContent = `${result.source} (${langLabel})`;
                }
                if (result.detected_language) {
                    lastDetectedLanguage = result.detected_language;
                }
                
                if (thoughtList) {
                    thoughtList.innerHTML = "";
                    if (result.thought && result.thought.length > 0) {
                        result.thought.forEach(step => {
                            const li = document.createElement('li');
                            li.textContent = step;
                            thoughtList.appendChild(li);
                        });
                    } else {
                        thoughtList.innerHTML = `<li>Penyusunan kalimat selesai.</li>`;
                    }
                }
            }
        }
    } catch (e) {
        console.error("Gagal merangkai kalimat:", e);
        if (thoughtList) {
            thoughtList.innerHTML = `<li style="color: #ef4444;">Gagal menghubungi server penyusun.</li>`;
        }
    } finally {
        isRefining = false;
        if (btnRefine) {
            btnRefine.disabled = false;
            btnRefine.innerHTML = `<i class="fa-solid fa-language"></i> Rangkai`;
        }
    }
}

function speakSentence() {
    if (!refinedSentenceField) return;
    const text = refinedSentenceField.textContent;
    if (!text || text === "-" || text === "") return;
    
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        const selectedLang = languageSelect ? languageSelect.value : "auto";
        const speakLang = selectedLang === "en" || (selectedLang === "auto" && lastDetectedLanguage === "en")
            ? "en-US"
            : "id-ID";
        utterance.lang = speakLang;
        window.speechSynthesis.speak(utterance);
        
        if (btnSpeak) {
            btnSpeak.style.background = 'var(--accent-green)';
            btnSpeak.style.color = '#fff';
            setTimeout(() => {
                btnSpeak.style.background = '#1F2937';
                btnSpeak.style.color = 'var(--text-primary)';
            }, 1000);
        }
    } else {
        alert("Text-to-Speech tidak didukung di browser Anda.");
    }
}

// Register Click Listeners for Sentence Builder Buttons
if (btnSpace) btnSpace.addEventListener('click', handleSpaceAction);
if (btnBackspace) btnBackspace.addEventListener('click', handleBackspaceAction);
if (btnClearSentence) btnClearSentence.addEventListener('click', handleClearAction);
if (btnRefine) btnRefine.addEventListener('click', refineSentenceWithAI);
if (btnSpeak) btnSpeak.addEventListener('click', speakSentence);
if (languageSelect) {
    languageSelect.addEventListener('change', () => {
        lastSuggestionRequest = "";
        lastScheduledPartial = "";
        scheduleWordSuggestionsFetch();
    });
}

// --- 5. START APP ---
async function startApp() {
    populateDropdown();
    await fetchDatasetCounts();
    renderWordSuggestions();
    
    try {
        await setupWebcam();
        window.addEventListener('resize', syncWebcamLayout);

        setInterval(runPrediction, PREDICTION_INTERVAL_MS);
        setInterval(updateSentenceBuilderUI, 50);
        requestAnimationFrame(renderHUDLoop);
    } catch (err) {
        console.error("Gagal memulai aplikasi:", err);
    }
}

// Start
window.addEventListener('DOMContentLoaded', startApp);
