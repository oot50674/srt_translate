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
    const selectedFiles = [];
    const STORAGE_KEY = 'whisper_chunk_seconds';

    function showAlert(message, type = 'error') {
        if (!message) return;

        if (type === 'success') {
            Toast.info(message, { position: 'top-center' });
        } else {
            Toast.alert(message, { position: 'top-center', ariaLive: 'assertive' });
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
        } catch (err) {
            console.warn('로컬스토리지에서 청크 길이를 불러오지 못했습니다.', err);
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
    }

    async function handleSubmit(event) {
        event.preventDefault();
        if (!selectedFiles.length) {
            showAlert('최소 한 개 이상의 파일을 선택해 주세요.');
            return;
        }
        const chunkSecondsValue = parseFloat(chunkSecondsInput?.value || '30');
        if (!Number.isFinite(chunkSecondsValue) || chunkSecondsValue <= 0) {
            showAlert('청크 길이는 초 단위 양수로 입력해 주세요.');
            chunkSecondsInput?.focus();
            return;
        }
        persistChunkSeconds(chunkSecondsValue);
        const formData = new FormData();
        selectedFiles.forEach(file => {
            formData.append('media_files', file);
        });
        formData.append('chunk_seconds', String(chunkSecondsValue));

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
            showAlert('작업을 준비했습니다. 곧 진행 화면으로 이동합니다.', 'success');
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
    loadStoredChunkSeconds();
    form.addEventListener('submit', handleSubmit);
})();
