(() => {
    const form = document.getElementById('subtitle-generate-form');
    const modeRadios = document.querySelectorAll('input[name="transcription_mode"]');
    const translationLanguage = document.getElementById('translation-language');
    const alertBox = document.getElementById('subtitle-generate-alert');
    const submitBtn = document.getElementById('subtitle-submit-btn');
    const submitSpinner = document.getElementById('subtitle-submit-spinner');
    const submitText = document.getElementById('subtitle-submit-text');
    const formHint = document.getElementById('form-hint');
    const dropZone = document.getElementById('video-drop-zone');
    const fileInput = document.getElementById('video-file');
    const uploadText = document.getElementById('video-upload-text');
    const selectBtn = document.getElementById('video-select-btn');
    const missingApiKey = String(document.body.dataset.missingApiKey || '').toLowerCase() === 'true';

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

    function validateForm() {
        const youtubeUrl = form.elements.namedItem('youtube_url')?.value.trim();
        const fileSelected = fileInput?.files && fileInput.files.length > 0;
        const mode = Array.from(modeRadios).find(radio => radio.checked)?.value || 'transcribe';
        const targetLanguage = form.elements.namedItem('target_language')?.value.trim();
        if (!youtubeUrl && !fileSelected) {
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
        if (!dropZone || !fileInput) return;
        ['dragenter', 'dragover'].forEach(evt => {
            dropZone.addEventListener(evt, e => {
                e.preventDefault();
                dropZone.classList.add('border-slate-400');
            });
        });
        ['dragleave', 'drop'].forEach(evt => {
            dropZone.addEventListener(evt, e => {
                e.preventDefault();
                dropZone.classList.remove('border-slate-400');
            });
        });
        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            if (e.dataTransfer?.files?.[0]) {
                fileInput.files = e.dataTransfer.files;
                if (uploadText) uploadText.textContent = e.dataTransfer.files[0].name;
            }
        });
    }

    function bindFileSelect() {
        if (!selectBtn || !fileInput || !uploadText) return;
        selectBtn.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', () => {
            if (fileInput.files && fileInput.files[0]) {
                uploadText.textContent = fileInput.files[0].name;
            } else {
                uploadText.textContent = '여기에 영상 파일을 끌어다 놓으세요';
            }
        });
    }

    modeRadios.forEach(radio => {
        radio.addEventListener('change', toggleTranslationField);
    });
    toggleTranslationField();
    bindDropZone();
    bindFileSelect();

    form?.addEventListener('submit', submitForm);
})();
