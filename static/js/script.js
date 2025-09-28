$(function () {
    const DEFAULT_MODEL = 'gemini-2.5-flash';

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
    const $thinkingBudgetInput = $('#thinking-budget');
    const $disableThinkingCheckbox = $('#disable-thinking');
    const $apiKeyInput = $('#api-key');
    const $modelInput = $('#model-name');

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
                        alert('SRT 파일만 업로드할 수 있습니다.');
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
        const p = await fetchPreset(name);
        if (!p) return;
        if ($targetLangInput.length) $targetLangInput.val(p.target_lang || '');
        if ($chunkSizeInput.length) $chunkSizeInput.val(p.batch_size || '');
        if ($customPromptInput.length) $customPromptInput.val(p.custom_prompt || '');

        if ($apiKeyInput.length) {
            if (p.api_key) {
                $apiKeyInput.val(p.api_key);
            } else {
                $apiKeyInput.val('');
            }
        }

        if ($modelInput.length) {
            // 프리셋은 모델명을 저장하지 않으므로 기존 입력값 유지.
            const storedModel = localStorage.getItem('modelName');
            if (!storedModel && !$modelInput.val()) {
                $modelInput.val(DEFAULT_MODEL);
            }
        }
        
        // Thinking Budget 설정 적용
        if ($thinkingBudgetInput.length) {
            const thinkingBudget = p.thinking_budget;
            if (thinkingBudget === '0' || thinkingBudget === 0) {
                $disableThinkingCheckbox.prop('checked', true).trigger('change');
            } else {
                $disableThinkingCheckbox.prop('checked', false).trigger('change');
                $thinkingBudgetInput.val(thinkingBudget || '8192');
            }
        }
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

    if ($savePresetBtn.length) {
        $savePresetBtn.on('click', async function () {
            const name = prompt('프리셋 이름을 입력하세요');
            if (!name) return;
            const preset = {
                target_lang: $targetLangInput.val() || '',
                batch_size: $chunkSizeInput.val() || '',
                custom_prompt: $customPromptInput.val() || '',
                thinking_budget: $disableThinkingCheckbox.is(':checked') ? '0' : $thinkingBudgetInput.val() || '',
                api_key: $apiKeyInput.length ? $apiKeyInput.val() || '' : ''
            };
            await savePreset(name, preset);
            await populatePresetOptions();
            if ($presetSelect.length) $presetSelect.val(name);
            alert('프리셋이 저장되었습니다.');
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
                alert('SRT 파일을 선택하거나 텍스트를 입력하세요.');
                return;
            }

            formData.append('target_lang', options.target_lang);
            formData.append('batch_size', options.batch_size);
            formData.append('custom_prompt', options.custom_prompt);
            formData.append('thinking_budget', options.thinking_budget);
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
                alert(data.error || '작업 생성 실패');
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
});
