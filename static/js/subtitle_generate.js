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
    const voiceFileInput = document.getElementById('voice-file');
    const fileInput = document.getElementById('file-input');
    const selectBtn = document.getElementById('file-select-btn');
    const fileList = document.getElementById('file-list');
    const customPromptInput = document.querySelector('textarea[name="custom_prompt"]');
    const clearPromptBtn = document.getElementById('clear-custom-prompt-btn');
    const missingApiKey = String(document.body.dataset.missingApiKey || '').toLowerCase() === 'true';

    const STORAGE_KEYS = {
        transcription_mode: 'subtitle_generate_transcription_mode',
        target_language: 'subtitle_generate_target_language',
        chunk_minutes: 'subtitle_generate_chunk_minutes',
        model: 'subtitle_generate_model',
        custom_prompt: 'subtitle_generate_custom_prompt'
    };

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

    function isAudioFile(filename) {
        const audioExts = ['.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg', '.wma'];
        return audioExts.some(ext => filename.toLowerCase().endsWith(ext));
    }

    function processFiles(files) {
        if (!files || files.length === 0) return;

        // 기존에 업로드된 파일 확인
        let existingVideoFile = videoFileInput?.files?.[0] || null;
        let existingSrtFile = srtFileInput?.files?.[0] || null;
        let existingVoiceFile = voiceFileInput?.files?.[0] || null;

        let videoFile = existingVideoFile;
        let srtFile = existingSrtFile;
        let voiceFile = existingVoiceFile;

        // 새로 추가된 파일을 타입별로 분류
        Array.from(files).forEach(file => {
            if (isVideoFile(file.name)) {
                videoFile = file;
            } else if (isSrtFile(file.name)) {
                srtFile = file;
            } else if (isAudioFile(file.name)) {
                voiceFile = file;
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

        if (voiceFile && voiceFileInput) {
            const dt = new DataTransfer();
            dt.items.add(voiceFile);
            voiceFileInput.files = dt.files;
        }

        // 파일 목록 UI 업데이트
        updateFileList(videoFile, srtFile, voiceFile);
    }

    function updateFileList(videoFile, srtFile, voiceFile) {
        if (!fileList) return;

        if (!videoFile && !srtFile && !voiceFile) {
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
                updateFileList(null, srtFileInput.files?.[0] || null, voiceFileInput?.files?.[0] || null);
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
                updateFileList(videoFileInput?.files?.[0] || null, null, voiceFileInput?.files?.[0] || null);
            });
        }

        if (voiceFile) {
            const voiceItem = document.createElement('div');
            voiceItem.className = 'flex items-center gap-2 px-3 py-2 bg-purple-50 border border-purple-200 rounded-md';
            voiceItem.innerHTML = `
                <span class="material-icons text-purple-600">graphic_eq</span>
                <span class="flex-1 text-sm text-slate-700 truncate">${voiceFile.name}</span>
                <button type="button" class="remove-voice-btn text-slate-400 hover:text-slate-600">
                    <span class="material-icons text-lg">close</span>
                </button>
            `;
            fileList.appendChild(voiceItem);

            voiceItem.querySelector('.remove-voice-btn').addEventListener('click', () => {
                voiceFileInput.value = '';
                updateFileList(videoFileInput?.files?.[0] || null, srtFileInput?.files?.[0] || null, null);
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

    function getStorageValue(key) {
        try {
            return localStorage.getItem(STORAGE_KEYS[key]);
        } catch (e) {
            console.warn(`Failed to get ${key} from localStorage:`, e);
            return null;
        }
    }

    function setStorageValue(key, value) {
        try {
            if (value && value.trim()) {
                localStorage.setItem(STORAGE_KEYS[key], value.trim());
            } else {
                localStorage.removeItem(STORAGE_KEYS[key]);
            }
        } catch (e) {
            console.warn(`Failed to save ${key} to localStorage:`, e);
        }
    }

    function loadStoredValues() {
        // 전사 모드 로드
        const transcriptionMode = getStorageValue('transcription_mode');
        if (transcriptionMode) {
            const radio = document.querySelector(`input[name="transcription_mode"][value="${transcriptionMode}"]`);
            if (radio) {
                radio.checked = true;
                toggleTranslationField();
            }
        }

        // 번역 언어 로드
        const targetLanguage = getStorageValue('target_language');
        if (targetLanguage) {
            const langInput = form.elements.namedItem('target_language');
            if (langInput) langInput.value = targetLanguage;
        }

        // 청크 길이 로드
        const chunkMinutes = getStorageValue('chunk_minutes');
        if (chunkMinutes) {
            const chunkInput = form.elements.namedItem('chunk_minutes');
            if (chunkInput) chunkInput.value = chunkMinutes;
        }

        // 모델 로드
        const model = getStorageValue('model');
        if (model) {
            const modelInput = form.elements.namedItem('model');
            if (modelInput) modelInput.value = model;
        }

        // 커스텀 프롬프트 로드
        const customPrompt = getStorageValue('custom_prompt');
        if (customPrompt && customPromptInput) {
            customPromptInput.value = customPrompt;
        }

        updateClearButtonVisibility();
    }

    function saveFormValues() {
        // 전사 모드 저장
        const checkedMode = Array.from(modeRadios).find(radio => radio.checked);
        if (checkedMode) setStorageValue('transcription_mode', checkedMode.value);

        // 번역 언어 저장
        const langInput = form.elements.namedItem('target_language');
        if (langInput) setStorageValue('target_language', langInput.value);

        // 청크 길이 저장
        const chunkInput = form.elements.namedItem('chunk_minutes');
        if (chunkInput) setStorageValue('chunk_minutes', chunkInput.value);

        // 모델 저장
        const modelInput = form.elements.namedItem('model');
        if (modelInput) setStorageValue('model', modelInput.value);

        // 커스텀 프롬프트 저장
        if (customPromptInput) setStorageValue('custom_prompt', customPromptInput.value);
    }

    function updateClearButtonVisibility() {
        if (!clearPromptBtn || !customPromptInput) return;
        const hasValue = customPromptInput.value.trim().length > 0;
        const hasSaved = getStorageValue('custom_prompt') !== null;
        if (hasValue || hasSaved) {
            clearPromptBtn.classList.remove('hidden');
        } else {
            clearPromptBtn.classList.add('hidden');
        }
    }

    function clearCustomPrompt() {
        if (!customPromptInput) return;
        try {
            customPromptInput.value = '';
            localStorage.removeItem(STORAGE_KEYS.custom_prompt);
            updateClearButtonVisibility();
            showAlert('저장된 프롬프트를 삭제했습니다.', 'success');
            setTimeout(() => showAlert(''), 2000);
        } catch (e) {
            console.warn('Failed to clear custom prompt from localStorage:', e);
        }
    }

    function bindFormValueStorage() {
        // 페이지 로드 시 저장된 값들 불러오기
        loadStoredValues();

        // 입력 필드들에 change 이벤트 바인딩
        const inputsToWatch = [
            'target_language',
            'chunk_minutes',
            'model'
        ];

        inputsToWatch.forEach(name => {
            const input = form.elements.namedItem(name);
            if (input) {
                input.addEventListener('change', saveFormValues);
                input.addEventListener('input', saveFormValues);
            }
        });

        // 라디오 버튼들에 change 이벤트 바인딩
        modeRadios.forEach(radio => {
            radio.addEventListener('change', () => {
                saveFormValues();
                toggleTranslationField();
            });
        });

        // 커스텀 프롬프트 자동 저장
        if (customPromptInput) {
            let saveTimeout;
            customPromptInput.addEventListener('input', () => {
                clearTimeout(saveTimeout);
                saveTimeout = setTimeout(() => {
                    setStorageValue('custom_prompt', customPromptInput.value);
                    updateClearButtonVisibility();
                }, 500);
            });
            customPromptInput.addEventListener('change', () => {
                setStorageValue('custom_prompt', customPromptInput.value);
                updateClearButtonVisibility();
            });
        }

        // 지우기 버튼 이벤트
        if (clearPromptBtn) {
            clearPromptBtn.addEventListener('click', clearCustomPrompt);
        }

        // 폼 제출 시 모든 값 저장
        form?.addEventListener('submit', saveFormValues);
    }

    bindDropZone();
    bindFileSelect();
    bindFormValueStorage();

    form?.addEventListener('submit', submitForm);
})();
