(() => {
    const form = document.getElementById('whisper-upload-form');
    if (!form) return;

    const dropZone = document.getElementById('whisper-drop-zone');
    const fileInput = document.getElementById('whisper-file-input');
    const selectBtn = document.getElementById('whisper-select-btn');
    const fileList = document.getElementById('whisper-file-list');
    const submitBtn = document.getElementById('whisper-submit-btn');
    const submitSpinner = document.getElementById('whisper-submit-spinner');
    const submitText = document.getElementById('whisper-submit-text');
    const chunkSecondsInput = document.getElementById('whisper-chunk-seconds');
    const disableChunkingCheckbox = document.getElementById('whisper-disable-chunking');
    const utteranceToggle = document.getElementById('whisper-utterance-toggle');
    const modelInput = document.getElementById('whisper-model-name');
    const selectedFiles = [];
    const youtubeUrls = [];
    const STORAGE_KEY = 'whisper_chunk_seconds';
    const DISABLE_STORAGE_KEY = 'whisper_disable_chunking';
    const UTTERANCE_STORAGE_KEY = 'whisper_utterance_segmentation';
    const MODEL_STORAGE_KEY = 'whisper_model_name';

    function showAlert(message, type = 'error') {
        if (!message) return;

        if (type === 'success') {
            Toast.info(message, { position: 'top-center' });
        } else {
            Toast.alert(message, { position: 'top-center', ariaLive: 'assertive' });
        }
    }

    function loadStoredModelName() {
        if (!modelInput) return;
        try {
            const stored = localStorage.getItem(MODEL_STORAGE_KEY);
            if (stored) {
                modelInput.value = stored;
            }
        } catch (err) {
            console.warn('Whisper 모델 정보를 불러오지 못했습니다.', err);
        }
    }

    function persistModelName(value) {
        if (!modelInput) return;
        try {
            if (value) {
                localStorage.setItem(MODEL_STORAGE_KEY, value);
            } else {
                localStorage.removeItem(MODEL_STORAGE_KEY);
            }
        } catch (err) {
            console.warn('Whisper 모델 정보를 저장하지 못했습니다.', err);
        }
    }

    function setSubmitting(isSubmitting) {
        if (!submitBtn) return;
        submitBtn.disabled = isSubmitting;
        if (submitSpinner) {
            submitSpinner.classList.toggle('hidden', !isSubmitting);
            submitSpinner.classList.toggle('animate-spin', isSubmitting);
        }
        if (submitText) {
            submitText.textContent = isSubmitting ? '요청 중...' : '전사 시작';
        }
    }

    function renderFileList() {
        if (!fileList) return;
        const files = selectedFiles;
        if (!files.length) {
            fileList.classList.add('hidden');
            fileList.innerHTML = '';
            return;
        }
        fileList.classList.remove('hidden');
        fileList.innerHTML = files.map((file, index) => `
            <div class="flex items-center justify-between rounded-md border border-slate-200 bg-white px-3 py-2 gap-3">
                <div class="flex flex-col text-left">
                    <p class="text-sm font-semibold text-slate-800 truncate">${file.name}</p>
                    <span class="text-xs text-slate-500">${(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                </div>
                <button type="button" class="remove-file-btn text-slate-400 hover:text-slate-600" data-file-index="${index}">
                    <span class="material-icons text-base">close</span>
                </button>
            </div>
        `).join('');
        fileList.querySelectorAll('.remove-file-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const idx = Number(btn.dataset.fileIndex);
                if (!Number.isNaN(idx)) {
                    selectedFiles.splice(idx, 1);
                    renderFileList();
                }
            });
        });
    }

    function loadStoredChunkSeconds() {
        if (!chunkSecondsInput) return;
        try {
            const storedValue = localStorage.getItem(STORAGE_KEY);
            if (storedValue) {
                chunkSecondsInput.value = storedValue;
            }
            else {
                // Backwards compatibility: if older key 'whisper_chunk_minutes' exists, convert to seconds
                try {
                    const legacy = localStorage.getItem('whisper_chunk_minutes');
                    if (legacy) {
                        const mins = parseFloat(legacy);
                        if (!Number.isNaN(mins) && Number.isFinite(mins) && mins > 0) {
                            const secs = Math.max(5, Math.round(mins * 60));
                            chunkSecondsInput.value = String(secs);
                        }
                    }
                } catch (e) {
                    // ignore
                }
            }
        } catch (err) {
            console.warn('로컬스토리지에서 청크 길이를 불러오지 못했습니다.', err);
        }
    }

    function loadStoredDisableChunking() {
        if (!disableChunkingCheckbox) return;
        try {
            const stored = localStorage.getItem(DISABLE_STORAGE_KEY);
            disableChunkingCheckbox.checked = stored === '1';
        } catch (err) {
            console.warn('청크 비활성화 설정 불러오기 실패', err);
        }
    }

    function persistChunkSeconds(value) {
        if (!chunkSecondsInput) return;
        try {
            if (Number.isFinite(value) && value > 0) {
                localStorage.setItem(STORAGE_KEY, String(value));
            } else {
                localStorage.removeItem(STORAGE_KEY);
            }
        } catch (err) {
            console.warn('청크 길이 저장에 실패했습니다.', err);
        }
    }

    function persistDisableChunking(enabled) {
        if (!disableChunkingCheckbox) return;
        try {
            if (enabled) {
                localStorage.setItem(DISABLE_STORAGE_KEY, '1');
            } else {
                localStorage.removeItem(DISABLE_STORAGE_KEY);
            }
        } catch (err) {
            console.warn('청크 비활성화 설정 저장 실패', err);
        }
    }

    function loadStoredUtteranceSegmentation() {
        if (!utteranceToggle) return;
        try {
            const stored = localStorage.getItem(UTTERANCE_STORAGE_KEY);
            if (stored === '1') {
                utteranceToggle.checked = true;
            } else if (stored === '0') {
                utteranceToggle.checked = false;
            }
        } catch (err) {
            console.warn('발화 단위 전사 설정을 불러오지 못했습니다.', err);
        }
    }

    function persistUtteranceSegmentation(enabled) {
        if (!utteranceToggle) return;
        try {
            localStorage.setItem(UTTERANCE_STORAGE_KEY, enabled ? '1' : '0');
        } catch (err) {
            console.warn('발화 단위 전사 설정을 저장하지 못했습니다.', err);
        }
    }

    function syncSegmentationControls() {
        const useUtterance = utteranceToggle?.checked === true;
        const disableChunking = disableChunkingCheckbox?.checked === true;

        if (disableChunkingCheckbox) {
            disableChunkingCheckbox.disabled = false;
        }

        if (chunkSecondsInput) {
            chunkSecondsInput.disabled = useUtterance || disableChunking;
        }
    }

    function renderYoutubeList() {
        const list = document.getElementById('whisper-youtube-list');
        if (!list) return;
        if (!youtubeUrls.length) {
            list.classList.add('hidden');
            list.innerHTML = '';
            return;
        }
        list.classList.remove('hidden');
        list.innerHTML = youtubeUrls
            .map((url, index) => `
                <div class="flex items-center justify-between rounded-md border border-slate-200 bg-slate-50 px-3 py-2 gap-3">
                    <div class="flex flex-col text-left">
                        <p class="text-sm font-semibold text-slate-800 break-all">${url}</p>
                        <span class="text-xs text-slate-500">YouTube</span>
                    </div>
                    <button type="button" class="remove-youtube-btn text-slate-400 hover:text-slate-600" data-url-index="${index}">
                        <span class="material-icons text-base">close</span>
                    </button>
                </div>
            `)
            .join('');
        list.querySelectorAll('.remove-youtube-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const idx = Number(btn.dataset.urlIndex);
                if (!Number.isNaN(idx)) {
                    youtubeUrls.splice(idx, 1);
                    renderYoutubeList();
                }
            });
        });
    }

    function handleNewFiles(files) {
        if (!files?.length) return;
        Array.from(files).forEach(file => selectedFiles.push(file));
        selectedFiles.sort((a, b) => {
            const nameA = (a?.name || '').toLowerCase();
            const nameB = (b?.name || '').toLowerCase();
            if (nameA < nameB) return -1;
            if (nameA > nameB) return 1;
            return 0;
        });
        if (fileInput) {
            fileInput.value = '';
        }
        renderFileList();
    }

    function bindDropZone() {
        if (!dropZone) return;
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, event => {
                event.preventDefault();
                dropZone.classList.add('border-blue-400', 'bg-blue-50');
            });
        });
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, event => {
                event.preventDefault();
                dropZone.classList.remove('border-blue-400', 'bg-blue-50');
            });
        });
        dropZone.addEventListener('drop', event => {
            event.preventDefault();
            if (event.dataTransfer?.files?.length) {
                handleNewFiles(event.dataTransfer.files);
            }
        });
    }

    function bindSelectBtn() {
        if (!selectBtn || !fileInput) return;
        selectBtn.addEventListener('click', () => fileInput.click());
        dropZone?.addEventListener('click', evt => {
            const target = evt.target;
            const clickedButton = target instanceof Element && target.closest('#whisper-select-btn');
            if (!clickedButton) {
                fileInput.click();
            }
        });
        fileInput.addEventListener('change', () => {
            if (fileInput.files?.length) {
                handleNewFiles(fileInput.files);
            }
        });
        chunkSecondsInput?.addEventListener('input', () => {
            const val = parseFloat(chunkSecondsInput.value);
            if (Number.isFinite(val)) {
                persistChunkSeconds(val);
            }
        });
        disableChunkingCheckbox?.addEventListener('change', () => {
            const disabled = disableChunkingCheckbox.checked;
            persistDisableChunking(disabled);
            if (disabled) {
                utteranceToggle && persistUtteranceSegmentation(false);
                if (utteranceToggle) {
                    utteranceToggle.checked = false;
                }
            }
            syncSegmentationControls();
        });
        utteranceToggle?.addEventListener('change', () => {
            const enabled = utteranceToggle.checked;
            persistUtteranceSegmentation(enabled);
            if (enabled) {
                disableChunkingCheckbox && persistDisableChunking(false);
                if (disableChunkingCheckbox) {
                    disableChunkingCheckbox.checked = false;
                }
            }
            syncSegmentationControls();
        });
    }

    function addYoutubeUrlFromInput() {
        const input = document.getElementById('whisper-youtube-input');
        if (!input) return;
        const raw = (input.value || '').trim();
        if (!raw) {
            showAlert('YouTube 링크를 입력해 주세요.');
            input.focus();
            return;
        }
        const urlPattern = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\//i;
        if (!urlPattern.test(raw)) {
            showAlert('올바른 YouTube 링크를 입력해 주세요.');
            input.focus();
            return;
        }
        const normalized = raw.replace(/^\s+|\s+$/g, '');
        if (youtubeUrls.includes(normalized)) {
            showAlert('이미 추가된 링크입니다.');
            input.value = '';
            return;
        }
        youtubeUrls.push(normalized);
        input.value = '';
        renderYoutubeList();
        showAlert('');
    }

    function bindYoutubeControls() {
        const addBtn = document.getElementById('whisper-youtube-add-btn');
        const input = document.getElementById('whisper-youtube-input');
        if (addBtn) {
            addBtn.addEventListener('click', addYoutubeUrlFromInput);
        }
        if (input) {
            input.addEventListener('keydown', event => {
                if (event.key === 'Enter') {
                    event.preventDefault();
                    addYoutubeUrlFromInput();
                }
            });
        }
    }

    async function handleSubmit(event) {
        event.preventDefault();
        if (!selectedFiles.length && !youtubeUrls.length) {
            showAlert('파일을 추가하거나 YouTube 링크를 입력해 주세요.');
            return;
        }
        const useUtterance = utteranceToggle?.checked === true;
        let chunkSecondsValue = parseFloat(chunkSecondsInput?.value || '240');
        if (!Number.isFinite(chunkSecondsValue)) {
            chunkSecondsValue = 240;
        }
        const disableChunking = disableChunkingCheckbox?.checked === true;
        if (!useUtterance) {
            if (disableChunking) {
                // When disabling chunking, set chunkSeconds to 0 so the server will pass through the source file without splitting
                chunkSecondsValue = 0;
            } else if (chunkSecondsValue <= 0) {
                showAlert('청크 길이는 초 단위 양수로 입력해 주세요.');
                chunkSecondsInput?.focus();
                return;
            }
        }
        persistChunkSeconds(chunkSecondsValue);
        const formData = new FormData();
        selectedFiles.forEach(file => {
            formData.append('media_files', file);
        });
        youtubeUrls.forEach(url => {
            formData.append('youtube_urls', url);
        });
        formData.append('chunk_seconds', String(chunkSecondsValue));
        if (disableChunking && !useUtterance) {
            formData.append('disable_chunking', '1');
        }
        if (useUtterance) {
            formData.append('utterance_segmentation', '1');
            persistUtteranceSegmentation(true);
        } else {
            persistUtteranceSegmentation(false);
        }
        const modelName = (modelInput?.value || '').trim();
        if (modelName) {
            formData.append('model_name', modelName);
            persistModelName(modelName);
        } else {
            persistModelName('');
        }

        setSubmitting(true);
        showAlert('');
        try {
            const response = await fetch('/api/whisper/batch', {
                method: 'POST',
                body: formData,
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(payload.error || '배치 작업을 시작하지 못했습니다.');
            }
            
            const batchId = payload.batch_id;
            if (batchId) {
                window.location.href = `/whisper_only/batches/${encodeURIComponent(batchId)}`;
            }
        } catch (error) {
            console.error(error);
            showAlert(error.message || '요청 처리 중 오류가 발생했습니다.');
        } finally {
            setSubmitting(false);
        }
    }

    bindDropZone();
    bindSelectBtn();
    bindYoutubeControls();
    loadStoredChunkSeconds();
    loadStoredDisableChunking();
    loadStoredUtteranceSegmentation();
    loadStoredModelName();
    syncSegmentationControls();
    modelInput?.addEventListener('change', () => {
        const value = (modelInput.value || '').trim();
        persistModelName(value);
    });
    form.addEventListener('submit', handleSubmit);
})();
