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
let stableFramesCount = 0;
const REQUIRED_STABLE_FRAMES = 8; // Sekitar 1 detik (prediksi jalan tiap 120ms)
let isRefining = false;
let hasTypedCurrentSign = false; // Mencegah huruf terketik berulang kali tanpa melepas tangan

// DOM ELEMENTS
const video = document.getElementById('webcam');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const selectChar = document.getElementById('char-select');
const btnPrev = document.getElementById('btn-prev-char');
const btnNext = document.getElementById('btn-next-char');
const btnCapture = document.getElementById('btn-capture');
const mirrorMode = document.getElementById('mirror-mode');
const loader = document.getElementById('webcam-loader');
const predictedLetter = document.getElementById('predicted-letter');
const confidencePercent = document.getElementById('confidence-percentage');
const confidenceProgress = document.getElementById('confidence-progress');
const cnnPreview = document.getElementById('cnn-input-preview');
const totalSamplesBadge = document.getElementById('total-samples');
const statsGrid = document.getElementById('stats-grid');
const viewportContainer = document.querySelector('.webcam-viewport');

// DOM Sentence Builder
const activeCharDisplay = document.getElementById('active-char-display');
const currentWordField = document.getElementById('current-word-field');
const rawTextField = document.getElementById('raw-text-field');
const refinedSentenceField = document.getElementById('refined-sentence-field');
const thoughtList = document.getElementById('thought-list');
const aiSource = document.getElementById('ai-source');
const progressCircle = document.querySelector('.progress-ring__circle');

// Sentence Builder Buttons
const btnSpace = document.getElementById('btn-space');
const btnBackspace = document.getElementById('btn-backspace');
const btnClearSentence = document.getElementById('btn-clear-sentence');
const btnRefine = document.getElementById('btn-refine');
const btnSpeak = document.getElementById('btn-speak');

// Canvas tersembunyi untuk grab frame gambar mentah
const grabCanvas = document.createElement('canvas');
grabCanvas.width = 640;
grabCanvas.height = 480;
const grabCtx = grabCanvas.getContext('2d');

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

// Buka webcam
async function setupWebcam() {
    try {
        loader.style.display = 'flex';
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480,
                facingMode: 'user'
            },
            audio: false
        });
        video.srcObject = stream;
        
        // Sesuai mirror default
        if (mirrorMode.checked) {
            viewportContainer.classList.add('mirror-active');
        }
        
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
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

// Grab frame dan ubah ke format base64
function getFrameBase64() {
    // Kita capture video frame apa adanya
    grabCtx.drawImage(video, 0, 0, 640, 480);
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
                image: dataUrl,
                mirror_mode: mirrorMode.checked
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            if (!result.error) {
                lastPrediction = {
                    char: result.char,
                    confidence: result.confidence,
                    x_min: result.x_min,
                    y_min: result.y_min,
                    x_max: result.x_max,
                    y_max: result.y_max,
                    timestamp: Date.now()
                };
                
                // Update mini preview Input CNN
                if (result.mini_roi) {
                    cnnPreview.src = result.mini_roi;
                }
                
                updatePredictionUI(result.char, result.confidence);

                // --- KESTABILAN HURUF (DEBOUNCE) ---
                if (result.char && result.char !== "?" && result.confidence > 0.65) {
                    if (result.char === stableLetter) {
                        if (!hasTypedCurrentSign) {
                            stableFramesCount++;
                            if (stableFramesCount >= REQUIRED_STABLE_FRAMES) {
                                // Tambah huruf ke kata aktif
                                currentWord += stableLetter;
                                stableFramesCount = 0;
                                hasTypedCurrentSign = true; // Set flag agar tidak ngetik lagi terus-menerus
                                
                                // Visual blink/pulse effect pada kata aktif
                                currentWordField.classList.add('pulse-effect');
                                currentWordField.style.textShadow = '0 0 15px var(--accent-cyan)';
                                setTimeout(() => {
                                    currentWordField.classList.remove('pulse-effect');
                                    currentWordField.style.textShadow = '0 0 8px rgba(255, 255, 255, 0.2)';
                                }, 300);
                            }
                        }
                    } else {
                        stableLetter = result.char;
                        stableFramesCount = 1;
                        hasTypedCurrentSign = false; // Reset flag karena tanda sudah berubah
                    }
                } else {
                    stableLetter = null;
                    stableFramesCount = 0;
                    hasTypedCurrentSign = false; // Reset flag karena tangan turun
                }
                
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
function updatePredictionUI(char, confidence) {
    const percentage = Math.round(confidence * 100);
    const wrapper = document.querySelector('.predicted-char-wrapper');
    
    if (char && char !== "?") {
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
                char: targetChar,
                mirror_mode: mirrorMode.checked
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
    ctx.clearRect(0, 0, 640, 480);
    
    // Tentukan kotak ROI tetap default (240x240)
    // Sesuai dengan rumus Python:
    // x_min = int(w * 0.55) - 120 = 352 - 120 = 232
    // y_min = int(h * 0.5) - 120 = 240 - 120 = 120
    const x_min = 232;
    const y_min = 120;
    const box_size = 240;
    
    let boxColor = "#3b82f6"; // Biru default
    let predText = "";
    
    // Jika ada prediksi terbaru (kurang dari 1 detik yang lalu)
    if (lastPrediction && (Date.now() - lastPrediction.timestamp < 1000)) {
        const percentage = Math.round(lastPrediction.confidence * 100);
        if (percentage > 60) {
            boxColor = "#10b981"; // Hijau jika confident
        } else if (percentage > 30) {
            boxColor = "#f59e0b"; // Oranye jika ragu-ragu
        }
        
        if (lastPrediction.char && lastPrediction.char !== "?") {
            predText = `Prediksi: ${lastPrediction.char.toUpperCase()} (${percentage}%)`;
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

// Mirror mode toggle
mirrorMode.addEventListener('change', () => {
    if (mirrorMode.checked) {
        viewportContainer.classList.add('mirror-active');
    } else {
        viewportContainer.classList.remove('mirror-active');
    }
});

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
    }
});

// --- SENTENCE BUILDER CORE LOGIC & ACTIONS ---

const circleCircumference = 56.54;
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
    if (currentWordField) {
        currentWordField.textContent = currentWord ? currentWord.toUpperCase() : "-";
    }
    
    if (rawTextField) {
        const rawSentence = accumulatedWords.join(" ");
        rawTextField.textContent = rawSentence ? rawSentence : "-";
    }
    
    // Update progress ring
    if (stableLetter && stableFramesCount > 0) {
        const pct = (stableFramesCount / REQUIRED_STABLE_FRAMES) * 100;
        setProgress(pct);
    } else {
        setProgress(0);
    }
}

function handleSpaceAction() {
    if (currentWord) {
        accumulatedWords.push(currentWord.toLowerCase());
        currentWord = "";
        updateSentenceBuilderUI();
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
    if (refinedSentenceField) refinedSentenceField.textContent = "-";
    if (aiSource) aiSource.textContent = "Offline";
    if (thoughtList) {
        thoughtList.innerHTML = `<li class="empty-thought">Menunggu kata terkumpul...</li>`;
    }
    updateSentenceBuilderUI();
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
            body: JSON.stringify({ raw_text: rawSentence })
        });
        
        if (response.ok) {
            const result = await response.json();
            if (!result.error) {
                if (refinedSentenceField) refinedSentenceField.textContent = result.refined;
                if (aiSource) aiSource.textContent = result.source;
                
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
            btnRefine.innerHTML = `<i class="fa-solid fa-language"></i> Rangkai Kalimat`;
        }
    }
}

function speakSentence() {
    if (!refinedSentenceField) return;
    const text = refinedSentenceField.textContent;
    if (!text || text === "-" || text === "") return;
    
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'id-ID'; // Menggunakan suara Bahasa Indonesia
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

// --- 5. START APP ---
async function startApp() {
    populateDropdown();
    await fetchDatasetCounts();
    
    try {
        await setupWebcam();
        
        // Jalankan loop prediksi server setiap 120ms
        setInterval(runPrediction, 120);
        
        // Jalankan loop rendering overlay visual (FPS normal 60fps)
        requestAnimationFrame(renderHUDLoop);
    } catch (err) {
        console.error("Gagal memulai aplikasi:", err);
    }
}

// Start
window.addEventListener('DOMContentLoaded', startApp);
