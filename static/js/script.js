$(function () {
    const DEFAULT_MODEL = 'gemini-2.5-flash';

    // 토스트 기본 위치를 상단 중앙으로 설정
    if (window.Toast && typeof window.Toast.setDefaults === 'function') {
        Toast.setDefaults({ position: 'top-center' });
    }

    const $selectBtn = $('#select-btn');
    const $fileInput = $('#srt-file');
    const $fileUploadText = $('#file-upload-text');
    const $dropZone = $('#drop-zone');
    const $form = $('#srt-upload-form');
    const $textArea = $('#srt-text');
    const $targetLangInput = $('#target-lang');
    const $chunkSizeInput = $('#chunk-size');
    const $customPromptInput = $('#custom-prompt');
    const $presetSelect = $('#preset-select');
    const $savePresetBtn = $('#save-preset-btn');
    const $newPresetBtn = $('#new-preset-btn');
    const $thinkingBudgetInput = $('#thinking-budget');
    const $disableThinkingCheckbox = $('#disable-thinking');
    const $apiKeyInput = $('#api-key');
    const $modelInput = $('#model-name');
    const $saveApiKeyBtn = $('#save-api-key-btn');
    const $openSettingsBtn = $('#open-settings-btn');
    const $settingsPanel = $('#settings-panel');
    const $settingsOverlay = $('#settings-overlay');
    const $settingsCloseBtn = $('#settings-close-btn');
    const $contextCompressionCheckbox = $('#context-compression');
    const $contextLimitInput = $('#context-limit');
    const $rpmLimitInput = $('#rpm-limit');
    const SETTINGS_PANEL_STORAGE_KEY = 'settingsPanelOpen';
    const $configSaveIndicator = $('#config-save-indicator');
    const $missingApiKeyAlert = $('#missing-api-key-alert');
    const $panelMissingApiKeyAlert = $('#panel-missing-api-key-alert');
    let missingApiKey = String($('body').data('missingApiKey')).toLowerCase() === 'true';

    if ($modelInput.length) {
        const storedModel = localStorage.getItem('modelName');
        $modelInput.val(storedModel || DEFAULT_MODEL);

        $modelInput.on('input', function () {
            const value = $(this).val().trim();
            if (value) {
                localStorage.setItem('modelName', value);
            } else {
                localStorage.removeItem('modelName');
            }
        });
    }

    if ($settingsPanel.length && $openSettingsBtn.length) {
        let panelOpen = localStorage.getItem(SETTINGS_PANEL_STORAGE_KEY) === 'true';
        if (missingApiKey) {
            panelOpen = true;
        }

        function applyPanelState(open) {
            const isOpen = !!open;
            $settingsPanel.toggleClass('translate-x-full', !isOpen);
            $settingsOverlay.toggleClass('hidden', !isOpen);
            $('body').toggleClass('overflow-hidden', isOpen);
            $openSettingsBtn.attr('aria-expanded', String(isOpen));
            $settingsPanel.attr('aria-hidden', String(!isOpen));
            if (isOpen) {
                localStorage.setItem(SETTINGS_PANEL_STORAGE_KEY, 'true');
            } else {
                localStorage.removeItem(SETTINGS_PANEL_STORAGE_KEY);
            }
        }

        function openPanel() {
            panelOpen = true;
            applyPanelState(panelOpen);
        }

        function closePanel() {
            panelOpen = false;
            applyPanelState(panelOpen);
        }

        applyPanelState(panelOpen);

        if (panelOpen && missingApiKey && $apiKeyInput.length) {
            setTimeout(() => {
                $apiKeyInput.trigger('focus');
            }, 300);
        }

        $openSettingsBtn.on('click', function () {
            openPanel();
            if ($apiKeyInput.length) {
                setTimeout(() => {
                    $apiKeyInput.trigger('focus');
                }, 200);
            }
        });

        if ($settingsCloseBtn.length) {
            $settingsCloseBtn.on('click', function () {
                closePanel();
            });
        }

        if ($settingsOverlay.length) {
            $settingsOverlay.on('click', function () {
                closePanel();
            });
        }

        $(document).on('keydown', function (e) {
            if (panelOpen && e.key === 'Escape') {
                closePanel();
            }
        });
    }

    // ---------------- Config (추가 설정) 관리 ----------------
    // 프리셋과 완전히 독립. 서버의 /api/config 엔드포인트와 동기화.
    let configLoaded = false;
    let saveTimer = null;
    const SAVE_DEBOUNCE_MS = 600;

    function applyConfigToUI(cfg) {
        if (!cfg || typeof cfg !== 'object') return;
        // 모델명
        if ($modelInput.length && cfg.model_name) {
            $modelInput.val(cfg.model_name);
        }
        // context compression
        if ($contextCompressionCheckbox.length && Object.prototype.hasOwnProperty.call(cfg, 'context_compression')) {
            $contextCompressionCheckbox.prop('checked', !!cfg.context_compression);
        }
        // context limit
        if ($contextLimitInput.length && cfg.context_limit) {
            $contextLimitInput.val(cfg.context_limit);
        }
        // rpm limit
        if ($rpmLimitInput.length && cfg.rpm_limit) {
            $rpmLimitInput.val(cfg.rpm_limit);
        }
        // API Key preview -> placeholder로 표시
        if ($apiKeyInput.length && cfg.api_key_preview) {
            $apiKeyInput.attr('placeholder', '저장됨: ' + cfg.api_key_preview);
        }
        // context-limit input 상태 적용 (compression 여부 따라)
        syncContextInputState();
    }

    async function loadConfig() {
        try {
            const res = await fetch('/api/config');
            if (!res.ok) throw new Error('config load failed');
            const data = await res.json();
            applyConfigToUI(data);
            configLoaded = true;
        } catch (e) {
            console.warn('Failed to load config', e);
        }
    }

    function gatherConfigPayload(extra = {}) {
        const payload = {};
        if ($modelInput.length && $modelInput.val().trim()) payload.model_name = $modelInput.val().trim();
        if ($contextCompressionCheckbox.length) payload.context_compression = $contextCompressionCheckbox.is(':checked') ? 1 : 0;
        if ($contextLimitInput.length && $contextLimitInput.val()) payload.context_limit = $contextLimitInput.val();
        if ($rpmLimitInput.length && $rpmLimitInput.val()) payload.rpm_limit = $rpmLimitInput.val();
        return Object.assign(payload, extra || {});
    }

    // 인디케이터 정책:
    // - 기본(대기): 완전히 숨김 (display: none)
    // - saving: "저장 중..." 텍스트 표시
    // - saved: "저장됨" 1.8초 표시 후 숨김
    // - error: "저장 실패" 유지 (사용자 후속 변경 시 재시도)
    function setConfigIndicator(state) {
        if (!$configSaveIndicator.length) return;
        const el = $configSaveIndicator;
        let text = '';
        let aria = '';
        let hideDelay = null;
        switch (state) {
            case 'saving':
                text = '저장 중...';
                aria = '설정 저장 중';
                break;
            case 'saved':
                text = '저장됨';
                aria = '설정 저장 완료';
                hideDelay = 1800;
                break;
            case 'error':
                text = '저장 실패';
                aria = '설정 저장 실패';
                break;
            default:
                // idle
                break;
        }
        if (!text) {
            el.text('').attr('aria-label', '설정 저장 상태: 대기').attr('title', '');
            el.css('display', 'none');
            return;
        }
        el.text(text)
          .attr('aria-label', aria)
          .attr('title', aria)
          .css('display', 'inline-block');
        if (hideDelay) {
            setTimeout(() => {
                // 상태가 여전히 saved 텍스트이면 숨김
                if (el.text() === '저장됨') {
                    el.text('').css('display', 'none').attr('title', '').attr('aria-label', '설정 저장 상태: 대기');
                }
            }, hideDelay);
        }
    }

    async function saveConfig(partial) {
        if (!partial || Object.keys(partial).length === 0) return;
        try {
            setConfigIndicator('saving');
            const body = JSON.stringify(partial);
            const res = await fetch('/api/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) throw new Error(data.error || '설정 저장 실패');
            setConfigIndicator('saved');
            loadConfig();
        } catch (e) {
            setConfigIndicator('error');
            console.error(e);
        }
    }

    function scheduleAutoSave() {
        if (saveTimer) clearTimeout(saveTimer);
        saveTimer = setTimeout(() => {
            const payload = gatherConfigPayload();
            if (Object.keys(payload).length > 0) {
                saveConfig(payload);
            }
        }, SAVE_DEBOUNCE_MS);
    }

    if ($saveApiKeyBtn.length && $apiKeyInput.length) {
        $saveApiKeyBtn.on('click', async function () {
            const apiKeyVal = ($apiKeyInput.val() || '').trim();
            if (!apiKeyVal) {
                // API Key 미입력시 인디케이터는 변하지 않고 포커스만 이동
                $apiKeyInput.trigger('focus');
                return;
            }
            $saveApiKeyBtn.prop('disabled', true).addClass('opacity-60 cursor-not-allowed');
            try {
                const payload = gatherConfigPayload({ api_key: apiKeyVal });
                await saveConfig(payload);
                $apiKeyInput.val(''); // 실제 키는 다시 표기하지 않음
                missingApiKey = false;
                $('body').attr('data-missing-api-key', 'false');
                if ($missingApiKeyAlert.length) $missingApiKeyAlert.remove();
                if ($panelMissingApiKeyAlert.length) $panelMissingApiKeyAlert.remove();
            } finally {
                $saveApiKeyBtn.prop('disabled', false).removeClass('opacity-60 cursor-not-allowed');
            }
        });
    }

    function toggleFileUpload(disabled) {
        if ($dropZone.length) {
            if (disabled) {
                $dropZone.addClass('opacity-50 pointer-events-none');
                $dropZone.removeClass('hover:border-slate-400');
            } else {
                $dropZone.removeClass('opacity-50 pointer-events-none');
                $dropZone.addClass('hover:border-slate-400');
            }
        }
        if ($fileInput.length) {
            $fileInput.prop('disabled', disabled);
        }
        if ($selectBtn.length) {
            $selectBtn.prop('disabled', disabled);
            if (disabled) {
                $selectBtn.addClass('opacity-50 cursor-not-allowed');
                $selectBtn.removeClass('hover:bg-slate-300 cursor-pointer');
            } else {
                $selectBtn.removeClass('opacity-50 cursor-not-allowed');
                $selectBtn.addClass('hover:bg-slate-300 cursor-pointer');
            }
        }
    }

    if ($textArea.length) {
        $textArea.on('input', function () {
            const hasText = $.trim($(this).val()) !== '';
            toggleFileUpload(hasText);
            if (hasText && $fileInput.length && $fileInput[0].files.length > 0) {
                $fileInput.val('');
                updateFileName(null);
            }
        });
        toggleFileUpload($.trim($textArea.val()) !== '');
    }

    if ($selectBtn.length && $fileInput.length) {
        $selectBtn.on('click', function (e) {
            e.stopPropagation();
            if (!$fileInput.prop('disabled')) {
                $fileInput.trigger('click');
            }
        });
    }

    function updateFileName(files) {
        if (files && files.length > 0) {
            if (files.length === 1) {
                $fileUploadText.text(`선택된 파일: ${files[0].name}`);
            } else {
                $fileUploadText.text(`${files.length}개 파일 선택됨`);
            }
        } else {
            $fileUploadText.text('여기에 .srt 파일을 끌어다 놓으세요');
        }
    }

    if ($fileInput.length) {
        $fileInput.on('click', function (e) {
            // 파일 입력 클릭 이벤트가 다시 드롭존으로 버블링되어
            // 무한 루프가 발생하지 않도록 전파를 중단한다.
            e.stopPropagation();
        });

        $fileInput.on('change', function () {
            if (this.files.length > 0) {
                updateFileName(this.files);
            } else {
                updateFileName(null);
            }
        });
    }

    if ($dropZone.length && $fileInput.length) {
        $dropZone.on('click', function (e) {
            if (!$fileInput.prop('disabled')) {
                // 사용자가 드롭존을 클릭하면 숨겨진 파일 입력을 눌러 파일 선택창을 연다.
                // 이때 발생하는 클릭 이벤트가 다시 드롭존으로 전파되어
                // 재귀적으로 호출되는 것을 방지하기 위해 기본 동작을 막는다.
                e.preventDefault();
                $fileInput[0].click();
            }
        });

        $dropZone.on('dragover', function (e) {
            if (!$fileInput.prop('disabled')) {
                e.preventDefault();
                e.stopPropagation();
                $dropZone.addClass('border-blue-500 bg-blue-50');
            }
        });

        $dropZone.on('dragleave', function (e) {
            if (!$fileInput.prop('disabled')) {
                e.preventDefault();
                e.stopPropagation();
                $dropZone.removeClass('border-blue-500 bg-blue-50');
            }
        });

        $dropZone.on('drop', function (e) {
            if (!$fileInput.prop('disabled')) {
                e.preventDefault();
                e.stopPropagation();
                $dropZone.removeClass('border-blue-500 bg-blue-50');
                const files = e.originalEvent.dataTransfer.files;
                if (files.length > 0) {
                    const valid = Array.from(files).every(f => f.name.toLowerCase().endsWith('.srt'));
                    if (valid) {
                        $fileInput[0].files = files;
                        updateFileName(files);
                    } else {
                        Toast.alert('SRT 파일만 업로드할 수 있습니다.');
                        $fileInput.val('');
                        updateFileName(null);
                    }
                }
            }
        });
    }

    async function fetchJson(url, options) {
        const res = await fetch(url, options);
        if (!res.ok) throw new Error('Request failed');
        return await res.json();
    }

    async function fetchPresets() {
        try {
            return await fetchJson('/api/presets');
        } catch {
            return [];
        }
    }

    async function fetchPreset(name) {
        try {
            return await fetchJson('/api/presets/' + encodeURIComponent(name));
        } catch {
            return null;
        }
    }

    async function savePreset(name, preset) {
        return fetchJson('/api/presets/' + encodeURIComponent(name), {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(preset)
        });
    }

    async function populatePresetOptions() {
        if (!$presetSelect.length) return;
        const presets = await fetchPresets();
        $presetSelect.empty().append('<option value="">프리셋 선택...</option>');
        $.each(presets, function (_, p) {
            $('<option>').val(p.name).text(p.name).appendTo($presetSelect);
        });
        
        // 마지막에 선택한 프리셋 자동 적용
        const lastSelectedPreset = localStorage.getItem('lastSelectedPreset');
        if (lastSelectedPreset) {
            const presetExists = presets.some(p => p.name === lastSelectedPreset);
            if (presetExists) {
                $presetSelect.val(lastSelectedPreset);
                await applyPreset(lastSelectedPreset);
            }
        }
    }

    async function applyPreset(name) {
        // NOTE: 프리셋은 번역 작업 시 사용할 기본 설정을 저장하는 용도입니다.
        // 추가 설정 패널의 값들은 프리셋과 별개의 독립적인 값으로,
        // 프리셋 선택 시 추가 설정이 자동으로 변경되지 않아야 합니다.
        // 아래 코드는 현재 프리셋 선택 시 추가 설정도 변경하는 잘못된 구현입니다.
        const p = await fetchPreset(name);
        if (!p) return;
        if ($targetLangInput.length) $targetLangInput.val(p.target_lang || '');
        if ($chunkSizeInput.length) $chunkSizeInput.val(p.batch_size || '');
        if ($customPromptInput.length) $customPromptInput.val(p.custom_prompt || '');
    }

    if ($presetSelect.length) {
        $presetSelect.on('change', function () {
            if (this.value) {
                // 선택한 프리셋을 localStorage에 저장
                localStorage.setItem('lastSelectedPreset', this.value);
                applyPreset(this.value);
            } else {
                // 프리셋 선택 해제시 저장된 프리셋 정보 삭제
                localStorage.removeItem('lastSelectedPreset');
            }
        });
    }

    // '저장' 버튼: 현재 선택된 프리셋이 있으면 갱신(update), 없으면 새로 만들기
    // 프리셋은 '번역 기본 파라미터'만 저장합니다.
    // 추가 설정(API 키, 모델, Thinking Budget, 컨텍스트 압축 등)은 세션/사용자 환경 설정이며 프리셋과 완전히 분리됩니다.
    if ($savePresetBtn.length) {
        $savePresetBtn.on('click', async function () {
            const current = $presetSelect.length ? $presetSelect.val() : null;
            const preset = {
                target_lang: $targetLangInput.val() || '',
                batch_size: $chunkSizeInput.val() || '',
                custom_prompt: $customPromptInput.val() || ''
                // NOTE: 추가 설정 필드는 절대 프리셋에 포함하지 않습니다.
            };

                if (current) {
                // 업데이트: 선택된 프리셋 이름으로 덮어쓰기
                    try {
                        await savePreset(current, preset);
                        await populatePresetOptions();
                        if ($presetSelect.length) $presetSelect.val(current);
                        Toast.info('프리셋이 갱신되었습니다.');
                    } catch (err) {
                        Toast.alert('프리셋 갱신에 실패했습니다.');
                    }
            } else {
                // 선택된 프리셋이 없으면 새 이름을 받아 생성
                const name = prompt('저장할 프리셋 이름을 입력하세요');
                if (!name) return;
                try {
                    await savePreset(name, preset);
                    await populatePresetOptions();
                    if ($presetSelect.length) $presetSelect.val(name);
                    Toast.info('프리셋이 저장되었습니다.');
                } catch (err) {
                    Toast.alert('프리셋 저장에 실패했습니다.');
                }
            }
        });
    }

    // '새로 만들기' 버튼: 항상 새 이름을 입력받아 새로운 프리셋 생성
    // 동일하게 번역 관련 3개 필드만 저장합니다.
    if ($newPresetBtn.length) {
        $newPresetBtn.on('click', async function () {
            const name = prompt('새 프리셋 이름을 입력하세요');
            if (!name) return;
            const preset = {
                target_lang: $targetLangInput.val() || '',
                batch_size: $chunkSizeInput.val() || '',
                custom_prompt: $customPromptInput.val() || ''
                // 추가 설정 필드 제외
            };
                try {
                    await savePreset(name, preset);
                    await populatePresetOptions();
                    if ($presetSelect.length) $presetSelect.val(name);
                    Toast.info('새 프리셋이 생성되었습니다.');
                } catch (err) {
                    Toast.alert('프리셋 생성에 실패했습니다.');
                }
        });
    }

    populatePresetOptions();

    if ($form.length) {
        $form.on('submit', async function (e) {
            e.preventDefault();
            const options = {
                target_lang: $targetLangInput.val() || '',
                batch_size: $chunkSizeInput.val() || '',
                custom_prompt: $customPromptInput.val() || '',
            thinking_budget: $disableThinkingCheckbox.is(':checked') ? '0' : $thinkingBudgetInput.val() || '',
            // Context compression options
            context_compression: $contextCompressionCheckbox.length ? ($contextCompressionCheckbox.is(':checked') ? '1' : '0') : '0',
            context_limit: $contextLimitInput.length ? $contextLimitInput.val() || '' : '',
                api_key: $apiKeyInput.length ? $apiKeyInput.val() || '' : '',
                model: $modelInput.length ? $modelInput.val().trim() : ''
            };
            const formData = new FormData();
            if ($textArea.length && $.trim($textArea.val()) !== '') {
                formData.append('srt_text', $textArea.val());
            } else if ($fileInput.length && $fileInput[0].files.length > 0) {
                for (const f of $fileInput[0].files) {
                    formData.append('srt_files', f, f.name);
                }
            } else {
                Toast.alert('SRT 파일을 선택하거나 텍스트를 입력하세요.');
                return;
            }

            formData.append('target_lang', options.target_lang);
            formData.append('batch_size', options.batch_size);
            formData.append('custom_prompt', options.custom_prompt);
            formData.append('thinking_budget', options.thinking_budget);
            formData.append('context_compression', options.context_compression);
            formData.append('context_limit', options.context_limit);
            formData.append('api_key', options.api_key);
            formData.append('model', options.model || DEFAULT_MODEL);

            const res = await fetch('/api/jobs', {
                method: 'POST',
                body: formData
            });
            const data = await res.json();
            if (data.job_id) {
                window.location.href = '/progress?job=' + encodeURIComponent(data.job_id);
            } else {
                Toast.alert(data.error || '작업 생성 실패');
            }
        });
    }

    // Thinking Budget 비활성화 체크박스 처리
    if ($disableThinkingCheckbox.length && $thinkingBudgetInput.length) {
        $disableThinkingCheckbox.on('change', function() {
            const isDisabled = $(this).is(':checked');
            $thinkingBudgetInput.prop('disabled', isDisabled);
            if (isDisabled) {
                $thinkingBudgetInput.addClass('opacity-50 bg-slate-100');
            } else {
                $thinkingBudgetInput.removeClass('opacity-50 bg-slate-100');
            }
        });
    }

    // Context Compression 토글 처리: 토글이 활성화되어야 컨텍스트 제한 입력을 사용할 수 있음
    function syncContextInputState() {
        if (!($contextCompressionCheckbox.length && $contextLimitInput.length)) return;
        const enabled = $contextCompressionCheckbox.is(':checked');
        $contextLimitInput.prop('disabled', !enabled);
        if (!enabled) {
            $contextLimitInput.addClass('opacity-50 bg-slate-100');
        } else {
            $contextLimitInput.removeClass('opacity-50 bg-slate-100');
        }
    }

    if ($contextCompressionCheckbox.length && $contextLimitInput.length) {
        // 초기 상태: 로컬스토리지에 저장된 값이 있으면 체크박스에 적용
        const storedCompression = localStorage.getItem('contextCompressionEnabled');
        if (storedCompression !== null) {
            $contextCompressionCheckbox.prop('checked', storedCompression === 'true');
        }

        // context-limit 값은 세션 범위로 저장/복원
        const SESSION_KEY = 'contextLimit';
        const storedLimit = sessionStorage.getItem(SESSION_KEY);
        if (storedLimit !== null) {
            $contextLimitInput.val(storedLimit);
        }

        // 초기 적용 (config 로드 전 임시)
        syncContextInputState();

        $contextCompressionCheckbox.on('change', function () {
            const enabled = $(this).is(':checked');
            // UI 동작
            syncContextInputState();
            // 사용자 설정 로컬 저장 (백엔드와 동기화됨)
            localStorage.setItem('contextCompressionEnabled', enabled ? 'true' : 'false');
            scheduleAutoSave();
        });

        // context-limit 변경 시 세션에 저장
        $contextLimitInput.on('input', function () {
            const v = $(this).val();
            if (v === '' || isNaN(Number(v))) {
                sessionStorage.removeItem(SESSION_KEY);
            } else {
                sessionStorage.setItem(SESSION_KEY, String(v));
            }
            scheduleAutoSave();
        });
    }

    // 모델명 / RPM / context-limit 자동 저장 이벤트
    if ($modelInput.length) {
        $modelInput.on('input', scheduleAutoSave);
    }
    if ($rpmLimitInput.length) {
        $rpmLimitInput.on('input', function () {
            const v = $(this).val();
            if (v && Number(v) > 0) scheduleAutoSave();
        });
    }

    // 초기 config 로드
    loadConfig();
});
