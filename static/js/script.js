$(function () {
    const DEFAULT_MODEL = 'gemini-2.5-flash';
    
    // 탭 관련 요소들
    const $srtTab = $('#srt-tab');
    const $youtubeTab = $('#youtube-tab');
    const $srtContent = $('#srt-content');
    const $youtubeContent = $('#youtube-content');
    const $srtForm = $('#srt-upload-form');
    const $youtubeForm = $('#youtube-form');
    
    // YouTube 관련 요소들
    const $youtubeUrl = $('#youtube-url');
    const $extractBtn = $('#extract-btn');
    const $extractedPreview = $('#extracted-preview');
    const $extractedText = $('#extracted-text');
    const $translateBtn = $('#translate-btn');

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

    // 탭 전환 기능
    function switchTab(activeTab) {
        // 탭 버튼 스타일 업데이트
        $('.tab-button').removeClass('active');
        $(activeTab).addClass('active');
        
        // 콘텐츠 표시/숨김
        if (activeTab === '#srt-tab') {
            $srtContent.removeClass('hidden');
            $youtubeContent.addClass('hidden');
            $srtForm.removeClass('hidden');
            $youtubeForm.addClass('hidden');
        } else {
            $srtContent.addClass('hidden');
            $youtubeContent.removeClass('hidden');
            $srtForm.addClass('hidden');
            $youtubeForm.removeClass('hidden');
        }
    }
    
    // 탭 클릭 이벤트
    $srtTab.on('click', function() {
        switchTab('#srt-tab');
    });
    
    $youtubeTab.on('click', function() {
        switchTab('#youtube-tab');
    });
    
    // YouTube 자막 추출 기능
    $extractBtn.on('click', async function() {
        const url = $youtubeUrl.val().trim();
        if (!url) {
            alert('YouTube URL을 입력하세요.');
            return;
        }
        
        // 목표 언어 가져오기
        const targetLang = $('#youtube-target-lang').val() || 'ko';
        
        // 버튼 비활성화 및 로딩 상태
        $extractBtn.prop('disabled', true);
        const originalText = $extractBtn.find('span:last-child').text();
        $extractBtn.find('span:last-child').text('추출 중...');
        
        try {
            const response = await fetch('/api/youtube/extract', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    url: url,
                    target_lang: targetLang
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // 추출된 자막을 미리보기에 표시
                $extractedText.val(data.srt_content);
                $extractedPreview.removeClass('hidden');
                
                // SRT 텍스트 영역에도 복사
                $textArea.val(data.srt_content);
                
                // 자세한 정보와 함께 알림 표시
                const sourceInfo = data.source_transcript || '알 수 없음';
                const langInfo = data.language ? ` (${data.language})` : '';
                
                alert(`자막 추출 완료!
                
📊 추출 정보:
• 자막 개수: ${data.transcript_count}개
• 소스: ${sourceInfo}${langInfo}
• 비디오 ID: ${data.video_id}

자막이 미리보기 영역과 SRT 텍스트 영역에 복사되었습니다.`);
            } else {
                let errorMsg = '자막 추출 실패: ' + data.error;
                
                // 에러 타입별 추가 안내
                if (data.error_type === 'no_transcript') {
                    errorMsg += '\n\n💡 이 비디오에는 사용 가능한 자막이 없습니다. 다른 비디오를 시도해보세요.';
                } else if (data.error_type === 'transcripts_disabled') {
                    errorMsg += '\n\n💡 이 비디오는 자막이 비활성화되어 있습니다.';
                } else if (data.error_type === 'video_unavailable') {
                    errorMsg += '\n\n💡 비디오 URL이나 ID를 다시 확인하세요.';
                } else if (data.error_type === 'request_blocked' || data.error_type === 'ip_blocked') {
                    errorMsg += '\n\n💡 IP가 차단되었습니다. 잠시 후 다시 시도하거나 다른 네트워크를 사용해보세요.';
                }
                
                alert(errorMsg);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('자막 추출 중 오류가 발생했습니다.\n\n네트워크 연결을 확인하고 다시 시도해주세요.');
        } finally {
            // 버튼 상태 복원
            $extractBtn.prop('disabled', false);
            $extractBtn.find('span:last-child').text(originalText);
        }
    });
    
    // 번역 버튼 클릭 이벤트 (기존 폼 제출 로직과 통합)
    $translateBtn.on('click', function() {
        // 현재 활성화된 탭에 따라 다른 처리
        if (!$youtubeForm.hasClass('hidden')) {
            // YouTube 탭이 활성화된 경우
            if ($extractedText.val().trim() === '') {
                alert('먼저 YouTube 자막을 추출하세요.');
                return;
            }
            // 추출된 자막을 SRT 텍스트 영역에 복사
            $textArea.val($extractedText.val());
        }
        
        // 기존 폼 제출 로직 실행
        if ($form.length) {
            $form.trigger('submit');
        }
    });

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
