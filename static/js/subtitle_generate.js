(() => {
    const form = document.getElementById('subtitle-generate-form');
    const modeRadios = document.querySelectorAll('input[name="transcription_mode"]');
    const translationLanguage = document.getElementById('translation-language');
    const alertBox = document.getElementById('subtitle-generate-alert');
    const submitBtn = document.getElementById('subtitle-submit-btn');
    const submitSpinner = document.getElementById('subtitle-submit-spinner');
    const submitText = document.getElementById('subtitle-submit-text');
    const formHint = document.getElementById('form-hint');
    const dropZone = document.getElementById('file-drop-zone');
    const videoFileInput = document.getElementById('video-file');
    const srtFileInput = document.getElementById('srt-file');
    const fileInput = document.getElementById('file-input');
    const selectBtn = document.getElementById('file-select-btn');
    const fileList = document.getElementById('file-list');
    const customPromptInput = document.querySelector('textarea[name="custom_prompt"]');
    const clearPromptBtn = document.getElementById('clear-custom-prompt-btn');
    const missingApiKey = String(document.body.dataset.missingApiKey || '').toLowerCase() === 'true';

    const STORAGE_KEY = 'subtitle_generate_custom_prompt';

    function showAlert(message, type = 'error') {
        if (!alertBox) return;
        if (!message) {
            alertBox.classList.add('hidden');
            alertBox.textContent = '';
            return;
        }
        const palette = type === 'success'
            ? 'text-emerald-800 bg-emerald-50 border-emerald-200'
            : type === 'warning'
                ? 'text-amber-800 bg-amber-50 border-amber-300'
                : 'text-red-800 bg-red-50 border-red-200';
        alertBox.className = `rounded-lg border px-4 py-3 text-sm mb-6 ${palette}`;
        alertBox.textContent = message;
        alertBox.classList.remove('hidden');
    }

    function setSubmitting(isSubmitting) {
        if (!submitBtn) return;
        submitBtn.disabled = isSubmitting;
        if (submitSpinner) {
            submitSpinner.classList.toggle('hidden', !isSubmitting);
            submitSpinner.classList.toggle('animate-spin', isSubmitting);
        }
        if (submitText) {
            submitText.textContent = isSubmitting ? '요청 보내는 중...' : '자막 생성 요청';
        }
        if (formHint) {
            formHint.textContent = isSubmitting ? '서버에서 작업을 준비하고 있습니다...' : '영상 길이와 네트워크에 따라 다소 시간이 걸릴 수 있습니다.';
        }
    }

    function toggleTranslationField() {
        if (!translationLanguage) return;
        const needsTranslation = Array.from(modeRadios).some(radio => radio.checked && radio.value === 'translate');
        translationLanguage.classList.toggle('hidden', !needsTranslation);
    }

    function isVideoFile(filename) {
        const videoExts = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'];
        return videoExts.some(ext => filename.toLowerCase().endsWith(ext));
    }

    function isSrtFile(filename) {
        return filename.toLowerCase().endsWith('.srt');
    }

    function processFiles(files) {
        if (!files || files.length === 0) return;

        let videoFile = null;
        let srtFile = null;

        // 파일을 타입별로 분류
        Array.from(files).forEach(file => {
            if (isVideoFile(file.name)) {
                videoFile = file;
            } else if (isSrtFile(file.name)) {
                srtFile = file;
            }
        });

        // DataTransfer 객체를 사용하여 각 input에 파일 할당
        if (videoFile) {
            const dt = new DataTransfer();
            dt.items.add(videoFile);
            videoFileInput.files = dt.files;
        }

        if (srtFile) {
            const dt = new DataTransfer();
            dt.items.add(srtFile);
            srtFileInput.files = dt.files;
        }

        // 파일 목록 UI 업데이트
        updateFileList(videoFile, srtFile);
    }

    function updateFileList(videoFile, srtFile) {
        if (!fileList) return;

        if (!videoFile && !srtFile) {
            fileList.classList.add('hidden');
            fileList.innerHTML = '';
            return;
        }

        fileList.classList.remove('hidden');
        fileList.innerHTML = '';

        if (videoFile) {
            const videoItem = document.createElement('div');
            videoItem.className = 'flex items-center gap-2 px-3 py-2 bg-blue-50 border border-blue-200 rounded-md';
            videoItem.innerHTML = `
                <span class="material-icons text-blue-600">videocam</span>
                <span class="flex-1 text-sm text-slate-700 truncate">${videoFile.name}</span>
                <button type="button" class="remove-video-btn text-slate-400 hover:text-slate-600">
                    <span class="material-icons text-lg">close</span>
                </button>
            `;
            fileList.appendChild(videoItem);

            videoItem.querySelector('.remove-video-btn').addEventListener('click', () => {
                videoFileInput.value = '';
                updateFileList(null, srtFileInput.files?.[0] || null);
            });
        }

        if (srtFile) {
            const srtItem = document.createElement('div');
            srtItem.className = 'flex items-center gap-2 px-3 py-2 bg-emerald-50 border border-emerald-200 rounded-md';
            srtItem.innerHTML = `
                <span class="material-icons text-emerald-600">subtitles</span>
                <span class="flex-1 text-sm text-slate-700 truncate">${srtFile.name}</span>
                <button type="button" class="remove-srt-btn text-slate-400 hover:text-slate-600">
                    <span class="material-icons text-lg">close</span>
                </button>
            `;
            fileList.appendChild(srtItem);

            srtItem.querySelector('.remove-srt-btn').addEventListener('click', () => {
                srtFileInput.value = '';
                updateFileList(videoFileInput.files?.[0] || null, null);
            });
        }
    }

    function validateForm() {
        const youtubeUrl = form.elements.namedItem('youtube_url')?.value.trim();
        const videoFileSelected = videoFileInput?.files && videoFileInput.files.length > 0;
        const mode = Array.from(modeRadios).find(radio => radio.checked)?.value || 'transcribe';
        const targetLanguage = form.elements.namedItem('target_language')?.value.trim();
        if (!youtubeUrl && !videoFileSelected) {
            throw new Error('YouTube 링크 또는 영상 파일을 입력해 주세요.');
        }
        if (mode === 'translate' && !targetLanguage) {
            throw new Error('번역할 언어를 입력해 주세요.');
        }
        if (missingApiKey) {
            throw new Error('Google API Key가 설정되지 않았습니다.');
        }
    }

    async function submitForm(event) {
        event.preventDefault();
        if (!form) return;
        try {
            validateForm();
        } catch (err) {
            showAlert(err.message, 'error');
            return;
        }
        const formData = new FormData(form);
        setSubmitting(true);
        showAlert('');
        try {
            const response = await fetch('/api/subtitle/jobs', {
                method: 'POST',
                body: formData
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(payload.error || '자막 생성 요청을 처리하지 못했습니다.');
            }
            const jobId = payload.job_id;
            if (jobId) {
                showAlert('작업이 시작되었습니다. 잠시 후 진행 화면으로 이동합니다.', 'success');
                window.location.href = `/subtitle_jobs/${encodeURIComponent(jobId)}`;
            }
        } catch (err) {
            console.error(err);
            showAlert(err.message || '요청 중 오류가 발생했습니다.', 'error');
        } finally {
            setSubmitting(false);
        }
    }

    function bindDropZone() {
        if (!dropZone) return;

        ['dragenter', 'dragover'].forEach(evt => {
            dropZone.addEventListener(evt, e => {
                e.preventDefault();
                dropZone.classList.add('border-blue-400', 'bg-blue-50');
            });
        });

        ['dragleave', 'drop'].forEach(evt => {
            dropZone.addEventListener(evt, e => {
                e.preventDefault();
                dropZone.classList.remove('border-blue-400', 'bg-blue-50');
            });
        });

        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            if (e.dataTransfer?.files) {
                processFiles(e.dataTransfer.files);
            }
        });
    }

    function bindFileSelect() {
        if (!selectBtn || !fileInput) return;

        selectBtn.addEventListener('click', () => fileInput.click());

        fileInput.addEventListener('change', () => {
            if (fileInput.files) {
                processFiles(fileInput.files);
            }
        });
    }

    function updateClearButtonVisibility() {
        if (!clearPromptBtn || !customPromptInput) return;
        const hasValue = customPromptInput.value.trim().length > 0;
        const hasSaved = localStorage.getItem(STORAGE_KEY) !== null;
        if (hasValue || hasSaved) {
            clearPromptBtn.classList.remove('hidden');
        } else {
            clearPromptBtn.classList.add('hidden');
        }
    }

    function loadCustomPrompt() {
        if (!customPromptInput) return;
        try {
            const saved = localStorage.getItem(STORAGE_KEY);
            if (saved) {
                customPromptInput.value = saved;
            }
            updateClearButtonVisibility();
        } catch (e) {
            console.warn('Failed to load custom prompt from localStorage:', e);
        }
    }

    function saveCustomPrompt() {
        if (!customPromptInput) return;
        try {
            const value = customPromptInput.value.trim();
            if (value) {
                localStorage.setItem(STORAGE_KEY, value);
            } else {
                localStorage.removeItem(STORAGE_KEY);
            }
            updateClearButtonVisibility();
        } catch (e) {
            console.warn('Failed to save custom prompt to localStorage:', e);
        }
    }

    function clearCustomPrompt() {
        if (!customPromptInput) return;
        try {
            customPromptInput.value = '';
            localStorage.removeItem(STORAGE_KEY);
            updateClearButtonVisibility();
            showAlert('저장된 프롬프트를 삭제했습니다.', 'success');
            setTimeout(() => showAlert(''), 2000);
        } catch (e) {
            console.warn('Failed to clear custom prompt from localStorage:', e);
        }
    }

    function bindCustomPromptStorage() {
        if (!customPromptInput) return;

        // 페이지 로드 시 저장된 프롬프트 불러오기
        loadCustomPrompt();

        // 입력 시 자동 저장 (디바운스 적용)
        let saveTimeout;
        customPromptInput.addEventListener('input', () => {
            clearTimeout(saveTimeout);
            saveTimeout = setTimeout(() => {
                saveCustomPrompt();
            }, 500); // 500ms 후 저장
            updateClearButtonVisibility();
        });

        // 폼 제출 시에도 저장
        customPromptInput.addEventListener('change', saveCustomPrompt);

        // 지우기 버튼 이벤트
        if (clearPromptBtn) {
            clearPromptBtn.addEventListener('click', clearCustomPrompt);
        }
    }

    modeRadios.forEach(radio => {
        radio.addEventListener('change', toggleTranslationField);
    });
    toggleTranslationField();
    bindDropZone();
    bindFileSelect();
    bindCustomPromptStorage();

    form?.addEventListener('submit', submitForm);
})();
