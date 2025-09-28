// Progress page script
$(async function () {
    const DEFAULT_MODEL = 'gemini-2.5-flash';
    const params = new URLSearchParams(window.location.search);
    const jobId = params.get('job');

    if (!jobId) {
        Toast.alert('번역 작업 ID가 없습니다.');
        window.location.href = '/';
        return;
    }

    const $translationContainer = $('#translation-container');
    if ($translationContainer.length) {
        $translationContainer.removeClass('h-[calc(100vh-350px)]');

        function adjustTranslationContainerHeight() {
            const containerTop = $translationContainer[0].getBoundingClientRect().top;
            const windowHeight = window.innerHeight;
            
            const $main = $translationContainer.closest('main');
            let mainPaddingBottom = 0;
            if ($main.length) {
                const mainStyle = window.getComputedStyle($main[0]);
                mainPaddingBottom = parseFloat(mainStyle.paddingBottom);
            }

            let newHeight = windowHeight - containerTop - mainPaddingBottom;
            const minHeight = 400; // min-h-[400px] from CSS

            if (newHeight < minHeight) {
                newHeight = minHeight;
            }
            
            $translationContainer.css('height', newHeight + 'px');
        }

        // 창 크기 변경 시 높이 조절
        $(window).on('resize', adjustTranslationContainerHeight);

        // 파일 목록 높이 변경 감지하여 높이 조절
        const fileListElement = document.getElementById('file-list');
        if (fileListElement) {
            const resizeObserver = new ResizeObserver(adjustTranslationContainerHeight);
            resizeObserver.observe(fileListElement);
        }

        // 초기 로드 시 높이 조절
        adjustTranslationContainerHeight();
    }

    let abortController;
    let isStopped = false;

    const $stopButton = $('#stop-translation-btn');
    $stopButton.on('click', function() {
        if (abortController) {
            isStopped = true;
            abortController.abort();
            $stopButton.prop('disabled', true).find('span').text('중지하는 중...');
        }
    });

    const jobRes = await fetch('/api/jobs/' + encodeURIComponent(jobId));
    if (!jobRes.ok) {
        Toast.alert('번역 데이터를 불러올 수 없습니다.');
        window.location.href = '/';
        return;
    }
    const progressData = await jobRes.json();
    console.log("Job 데이터:", progressData);
    
    let storedModelName = DEFAULT_MODEL;
    try {
        if (progressData.model) {
            localStorage.setItem('modelName', progressData.model);
            storedModelName = progressData.model;
        } else {
            const savedModel = localStorage.getItem('modelName');
            if (savedModel) {
                storedModelName = savedModel;
            }
        }
    } catch (err) {
        console.warn('모델명 로컬 저장 실패:', err);
    }

    const files = progressData.files || [];
    if (!files.length) {
        Toast.alert('번역할 파일이 없습니다.');
        window.location.href = '/';
        return;
    }

    // 파일 이름 오름차순으로 정렬
    files.sort((a, b) => a.name.localeCompare(b.name));

    const $tbody = $('#translation-body');
    const $progressBar = $('#progress-bar');
    const $progressText = $('#progress-text');
    const $fileList = $('#file-list');

    let rows = [];
    let lastFocus = -1;
    let lastFocusRow = null; // 최근 번역 완료 행을 추적하기 위한 변수
    const translatedFilesData = {}; // 번역된 파일 내용을 저장할 객체
    const fileRowOffsets = {}; // 각 파일의 행 시작 오프셋
    const fileSubtitleCounts = {}; // 각 파일의 자막 개수
    const translatedSubtitlesByFile = {}; // 파일별 번역 결과 저장
    const originalSubtitlesByFile = {}; // 파일별 원본 자막 텍스트 저장

    async function initTable(content, rowIndexOffset, fileName) {
        const res = await fetch('/parse_srt', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ srt_text: content })
        });
        const subtitles = await res.json();
        
        // 파일 제목 행 추가 (미리 표시해도 무방하므로 유지)
        const fileTitleRow = $(
            '<tr class="bg-slate-200">' +
                '<th colspan="2" class="px-4 sm:px-6 py-2 text-left text-slate-800 font-semibold">' +
                    fileName +
                '</th>' +
            '</tr>'
        );
        $tbody.append(fileTitleRow);
        
        // 기존에는 여기서 각 자막행을 placeholder 형태로 추가했으나,
        // "완료된 번역만" 표시하기 위해 더 이상 행을 생성하지 않습니다.
        // 필요한 메타데이터만 저장합니다.
        fileSubtitleCounts[fileName] = subtitles.length;
        translatedSubtitlesByFile[fileName] = [];
        originalSubtitlesByFile[fileName] = subtitles; // 자막 원문 저장
        
        return subtitles.length;
    }

    function updateProgress(received, total) {
        if (total > 0) {
            const percent = Math.round((received / total) * 100);
            $progressBar.css('width', percent + '%');
            $progressText.text(percent + '%');
        }
    }

    // 번역이 완료될 때마다 해당 자막을 즉시 테이블에 추가합니다.
    function updateRow(item, fileName) {
        // 행 생성
        const originalText = (originalSubtitlesByFile[fileName] && originalSubtitlesByFile[fileName][item.index - 1]) ? originalSubtitlesByFile[fileName][item.index - 1].text : (item.original || '');
        const originalHtml = `<div class="font-mono text-xs text-slate-500 mb-1">[${item.start}-${item.end}]</div>${originalText}`;
        const translatedHtml = item.error ?
            `<div class="font-mono text-xs text-red-500 mb-1">[${item.start}-${item.end}]</div><div class="text-red-600 font-medium">${item.translated}</div>` :
            `<div class="font-mono text-xs text-slate-500 mb-1">[${item.start}-${item.end}]</div>${item.translated}`;

        const tr = $(
            '<tr class="hover:bg-slate-50 transition-colors group">' +
                `<td class="px-4 sm:px-6 py-4 w-1/2 text-slate-700 text-sm">${originalHtml}</td>` +
                `<td class="px-4 sm:px-6 py-4 w-1/2 text-slate-700 text-sm">${translatedHtml}</td>` +
            '</tr>'
        );

        if (item.error) {
            tr.addClass('bg-red-50');
        }

        // 최근 번역 행 강조 처리
        if (lastFocusRow) {
            lastFocusRow.removeClass('recent-entry');
        }
        tr.addClass('recent-entry');
        lastFocusRow = tr;

        $tbody.append(tr);

        // 스크롤 이동
        if (!isStopped) {
            tr[0].scrollIntoView({ behavior: 'auto', block: 'center', inline: 'nearest' });
        }
    }

    async function processAllFiles() {
        // 모든 파일을 한번에 FormData에 담아서 전송
        const formData = new FormData();
        files.forEach(file => {
            const srtBlob = new Blob([file.text], { type: 'text/plain' });
            formData.append('srt_files', srtBlob, file.name);
        });
        formData.append('target_lang', progressData.target_lang);
        formData.append('batch_size', progressData.batch_size);
        formData.append('custom_prompt', progressData.custom_prompt);
        formData.append('thinking_budget', progressData.thinking_budget || '8192');
        // api_key는 이제 job이나 업로드 단계에서 직접 전달하지 않고 서버 config 사용
        formData.append('model', progressData.model || storedModelName);
        formData.append('job_id', jobId); // 기존 호환 (사용 안할 수도 있음)

        let response;
        try {
            response = await fetch('/upload_srt', {
                method: 'POST',
                body: formData,
                signal: abortController.signal
            });
        } catch(error) {
            if (error.name === 'AbortError') {
                fileListItems.forEach($li => {
                    const fileName = $li.data('fileName');
                    $li.html('<svg class="w-5 h-5 mr-2 inline-block text-red-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16ZM8.28 7.22a.75.75 0 0 0-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 1 0 1.06 1.06L10 11.06l1.72 1.72a.75.75 0 1 0 1.06-1.06L11.06 10l1.72-1.72a.75.75 0 0 0-1.06-1.06L10 8.94 8.28 7.22Z" clip-rule="evenodd" /></svg>' + `<span>${fileName} - 중지됨</span>`);
                });
                return { stopped: true };
            }
            throw error;
        }

        if (!response.ok) {
            fileListItems.forEach($li => {
                const fileName = $li.data('fileName');
                $li.text(`${fileName} - 오류 발생`);
            });
            return { stopped: false };
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let currentProcessingFile = null;
        let globalRowIndexOffset = 0;
        let totalSubtitles = 0;
        let receivedSubtitles = 0;

        try {
            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                
                const lines = buffer.split('\n');
                buffer = lines.pop(); 
                
                for (const line of lines) {
                    if (!line.trim()) continue;
                    
                    try {
                        const data = JSON.parse(line);
                        
                        if (data.total_files !== undefined) {
                            // 총 파일 수 정보 (필요시 사용)
                        } else if (data.current_file) {
                            // 현재 처리 중인 파일 변경 - 이때 테이블 초기화
                            currentProcessingFile = data.current_file;
                            const currentFile = files.find(f => f.name === currentProcessingFile);
                            
                            if (currentFile) {
                                // 현재 파일의 테이블 행 추가
                                fileRowOffsets[currentProcessingFile] = globalRowIndexOffset;
                                const subtitleCount = await initTable(currentFile.text, globalRowIndexOffset, currentProcessingFile);
                                globalRowIndexOffset += subtitleCount;
                                
                                // 총 자막 수 업데이트 (첫 번째 파일일 때만 0으로 초기화)
                                if (totalSubtitles === 0) {
                                    // 모든 파일의 자막 수를 미리 계산
                                    for (const file of files) {
                                        const res = await fetch('/parse_srt', {
                                            method: 'POST',
                                            headers: { 'Content-Type': 'application/json' },
                                            body: JSON.stringify({ srt_text: file.text })
                                        });
                                        const subtitles = await res.json();
                                        totalSubtitles += subtitles.length;
                                    }
                                    updateProgress(receivedSubtitles, totalSubtitles);
                                }
                            }
                            
                            const $li = fileListItems.find($li => $li.data('fileName') === currentProcessingFile);
                            if ($li) {
                                $li.html('<svg class="animate-spin h-5 w-5 mr-2 inline-block text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 714 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>' + `<span>${currentProcessingFile} - 처리중</span>`);
                                // 현재 처리 중인 파일만 표시 (확장 상태가 아닌 경우)
                                updateCurrentProcessingFile(currentProcessingFile);
                            }
                        } else if (data.file_completed) {
                            // 파일 처리 완료
                            const completedFile = data.file_completed;
                            const $li = fileListItems.find($li => $li.data('fileName') === completedFile);
                            if ($li) {
                                // 번역 결과를 SRT 형식으로 변환
                                const translatedSubtitles = translatedSubtitlesByFile[completedFile];
                                if (translatedSubtitles && translatedSubtitles.length > 0) {
                                    translatedSubtitles.sort((a, b) => a.index - b.index);
                                    const srtContent = translatedSubtitles.map(sub => {
                                        const text = sub.error ? sub.original : sub.translated;
                                        return `${sub.index}\n${sub.start} --> ${sub.end}\n${text}\n`;
                                    }).join('\n');
                                    
                                    // 파일 객체에 번역 결과 저장
                                    const fileObj = files.find(f => f.name === completedFile);
                                    if (fileObj) {
                                        fileObj.translatedText = srtContent;
                                    }

                                    // 다운로드 링크로 변경
                                    const blob = new Blob([srtContent], { type: 'text/plain;charset=utf-8' });
                                    const url = URL.createObjectURL(blob);
                                    const baseFileName = completedFile.replace(/\.srt$/i, '');
                                    const downloadFileName = `[번역] ${baseFileName}.srt`;

                                    $li.html(`<svg class="w-5 h-5 mr-2 inline-block text-green-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M16.704 4.153a.75.75 0 01.143 1.052l-8 10.5a.75.75 0 01-1.127.075l-4.5-4.5a.75.75 0 011.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 011.052-.143z" clip-rule="evenodd" /></svg><span>${completedFile} - 완료</span>`)
                                        .addClass('cursor-pointer text-blue-600 hover:text-blue-800 hover:bg-slate-100 rounded-md p-1 -m-1')
                                        .off('click')
                                        .on('click', function() {
                                            const a = document.createElement('a');
                                            a.href = url;
                                            a.download = downloadFileName;
                                            document.body.appendChild(a);
                                            a.click();
                                            document.body.removeChild(a);
                                        });
                                }
                            }
                        } else if (data.index !== undefined && currentProcessingFile) {
                            // 번역 결과 처리
                            receivedSubtitles++;
                            translatedSubtitlesByFile[currentProcessingFile].push(data);
                            updateRow(data, currentProcessingFile);
                            updateProgress(receivedSubtitles, totalSubtitles);
                        }
                    } catch (e) {
                        console.warn('JSON 파싱 오류:', line, e);
                    }
                }
            }
        } catch (error) {
            console.error('스트림 읽기 오류:', error);
            if (error.name === 'AbortError') {
                fileListItems.forEach($li => {
                    const fileName = $li.data('fileName');
                    $li.html('<svg class="w-5 h-5 mr-2 inline-block text-red-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 1 0 0-16 8 8 0 0 0 0 16ZM8.28 7.22a.75.75 0 0 0-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 1 0 1.06 1.06L10 11.06l1.72 1.72a.75.75 0 1 0 1.06-1.06L11.06 10l1.72-1.72a.75.75 0 0 0-1.06-1.06L10 8.94 8.28 7.22Z" clip-rule="evenodd" /></svg>' + `<span>${fileName} - 중지됨</span>`);
                });
                return { stopped: true };
            }
            fileListItems.forEach($li => {
                const fileName = $li.data('fileName');
                $li.text(`${fileName} - 오류 발생`);
            });
            return { stopped: false };
        }

        return { stopped: false };
    }

    async function loadScript(url) {
        return new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = url;
            script.onload = resolve;
            script.onerror = reject;
            document.head.appendChild(script);
        });
    }

    const fileListItems = [];
    files.forEach((file) => {
        const $li = $('<li>')
            .addClass('file-item p-1')
            .data('fileName', file.name)
            .text(`${file.name} - 대기중`);
        fileListItems.push($li);
        $fileList.append($li);
    });

    let isExpanded = false;
    let currentProcessingFileName = null;
    let $expandLi = null;
    
    function updateCurrentProcessingFile(fileName) {
        currentProcessingFileName = fileName;
        updateFileListVisibility();
    }
    
    function updateFileListVisibility() {
        if (isExpanded) {
            fileListItems.forEach($li => $li.show());
        } else {
            fileListItems.forEach($li => {
                const fileName = $li.data('fileName');
                if (fileName === currentProcessingFileName) {
                    $li.show();
                } else {
                    $li.hide();
                }
            });
        }
    }

    if (fileListItems.length > 1) {
        $expandLi = $('<li class="text-center"></li>');
        const $expandButton = $('<button class="text-sm text-blue-600 hover:underline p-2 w-full hover:bg-slate-50 rounded-md">▼ 전체 목록 보기</button>');
        $expandLi.append($expandButton);
        $fileList.append($expandLi);

        $expandButton.on('click', function() {
            isExpanded = !isExpanded;
            updateFileListVisibility();
            $expandButton.html(isExpanded ? '▲ 간단히 보기' : '▼ 전체 목록 보기');
        });
        
        // 처음엔 모든 파일 숨김
        fileListItems.forEach($li => $li.hide());
    }

    abortController = new AbortController();
    $stopButton.show().css('display', 'inline-flex');

    // 모든 파일을 한번에 처리
    const result = await processAllFiles();
    
    // 번역 완료 후 토글 버튼 숨기고 모든 파일 표시
    currentProcessingFileName = null;
    if ($expandLi) {
        $expandLi.hide();
    }
    fileListItems.forEach($li => $li.show());

    if (result.stopped) {
        $('#progress-container').parent().hide();
        Toast.alert('사용자에 의해 작업이 중지되었습니다.');
        $stopButton.hide();
    } else {
        $('#progress-container').parent().hide();
        const translatedFiles = files.filter(f => f.translatedText);

        if (translatedFiles.length > 1) {
            try {
                await loadScript('https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js');
                
                const $zipLi = $('<li class="mt-4"></li>');
                const $zipButton = $('<button>')
                    .addClass('inline-flex items-center justify-center w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-2 px-4 rounded-lg shadow-md transition-colors duration-300')
                    .html('<span>모두 ZIP으로 다운로드</span>');
                
                $zipButton.on('click', async function() {
                    const zip = new JSZip();
                    translatedFiles.forEach(file => {
                        const baseFileName = file.name.replace(/\.srt$/i, '');
                        const downloadFileName = `[번역] ${baseFileName}.srt`;
                        zip.file(downloadFileName, file.translatedText);
                    });
                    
                    const zipBlob = await zip.generateAsync({ type: "blob" });
                    const downloadUrl = URL.createObjectURL(zipBlob);
                    const $a = $('<a>').attr('href', downloadUrl).attr('download', 'translated_files.zip');
                    $('body').append($a);
                    $a[0].click();
                    $a.remove();
                    URL.revokeObjectURL(downloadUrl);
                });
                $zipLi.append($zipButton);
                $('#file-list').append($zipLi);

            } catch (e) {
                console.error('JSZip 로딩 실패:', e);
            }
        }
        
        await fetch('/api/jobs/' + encodeURIComponent(jobId), { method: 'DELETE' });
    }
});
