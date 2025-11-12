/**
 * 자막 보정 싱크 페이지 JavaScript
 */

$(document).ready(function () {
    // DOM 요소
    const $srtFile = $('#srt-file');
    const $srtFileBtn = $('#srt-file-btn');
    const $srtFileName = $('#srt-file-name');
    const $audioFile = $('#audio-file');
    const $audioFileBtn = $('#audio-file-btn');
    const $audioFileName = $('#audio-file-name');
    const $processBtn = $('#process-btn');
    const $downloadBtn = $('#download-btn');
    const $progressSection = $('#progress-section');
    const $progressBar = $('#progress-bar');
    const $progressText = $('#progress-text');
    const $statsSection = $('#stats-section');
    const $resultSection = $('#result-section');
    const $errorSection = $('#error-section');
    const $resultSrt = $('#result-srt');
    const $errorMessage = $('#error-message');

    // 통계 요소
    const $statEntries = $('#stat-entries');
    const $statChunks = $('#stat-chunks');
    const $statSegments = $('#stat-segments');
    const $statCorrections = $('#stat-corrections');
    const $statOverlaps = $('#stat-overlaps');

    // 설정 입력 요소
    const $chunkMode = $('input[name="chunk_mode"]');
    const $vadThreshold = $('#vad-threshold');
    const $minSpeechDuration = $('#min-speech-duration');
    const $minSilenceDuration = $('#min-silence-duration');
    const $speechPad = $('#speech-pad');
    const $gapThreshold = $('#gap-threshold');
    const $gapThresholdContainer = $('#gap-threshold-container');
    const $lookbackStart = $('#lookback-start');
    const $lookaheadStart = $('#lookahead-start');
    const $pad = $('#pad');

    let correctedSrtContent = '';

    /**
     * 에러 표시
     */
    function showError(message) {
        $errorSection.removeClass('hidden');
        $errorMessage.text(message);
        $progressSection.addClass('hidden');
    }

    /**
     * 에러 숨기기
     */
    function hideError() {
        $errorSection.addClass('hidden');
        $errorMessage.text('');
    }

    /**
     * 진행 상태 업데이트
     */
    function updateProgress(percent, text) {
        $progressSection.removeClass('hidden');
        $progressBar.css('width', percent + '%');
        $progressText.text(text);
    }

    /**
     * 통계 표시
     */
    function showStats(stats) {
        $statsSection.removeClass('hidden');
        $statEntries.text(stats.total_entries || '-');
        $statChunks.text(stats.total_chunks || '-');
        $statSegments.text(stats.vad_segments || '-');
        $statCorrections.text(stats.corrections_applied || '-');
        $statOverlaps.text(stats.overlaps_fixed || '0');
    }

    /**
     * 결과 표시
     */
    function showResult(srtContent) {
        correctedSrtContent = srtContent;
        $resultSection.removeClass('hidden');
        $resultSrt.val(srtContent);
    }

    /**
     * 설정 수집
     */
    function collectConfig() {
        return {
            chunk_mode: $('input[name="chunk_mode"]:checked').val() || 'grouped',
            vad_threshold: parseFloat($vadThreshold.val()) || 0.55,
            vad_min_speech_duration_ms: parseInt($minSpeechDuration.val()) || 200,
            vad_min_silence_duration_ms: parseInt($minSilenceDuration.val()) || 250,
            vad_speech_pad_ms: parseInt($speechPad.val()) || 80,
            gap_threshold_ms: parseInt($gapThreshold.val()) || 200,
            lookback_start_ms: parseInt($lookbackStart.val()) || 800,
            lookahead_start_ms: parseInt($lookaheadStart.val()) || 400,
            pad_ms: parseInt($pad.val()) || 80
        };
    }

    /**
     * 청크 모드에 따라 UI 업데이트
     */
    function updateChunkModeUI() {
        const chunkMode = $('input[name="chunk_mode"]:checked').val();

        if (chunkMode === 'individual') {
            // 개별 모드: 청크 간격 임계값 비활성화
            $gapThresholdContainer.addClass('opacity-50');
            $gapThreshold.prop('disabled', true);
        } else {
            // 그룹 모드: 청크 간격 임계값 활성화
            $gapThresholdContainer.removeClass('opacity-50');
            $gapThreshold.prop('disabled', false);
        }
    }

    /**
     * 파일 유효성 검사
     */
    function validateFiles() {
        const srtFile = $srtFile[0].files[0];
        const audioFile = $audioFile[0].files[0];

        if (!srtFile) {
            showError('SRT 파일을 선택해주세요.');
            return false;
        }

        if (!audioFile) {
            showError('오디오/비디오 파일을 선택해주세요.');
            return false;
        }

        // 파일 확장자 검사
        if (!srtFile.name.toLowerCase().endsWith('.srt')) {
            showError('SRT 파일 형식이 올바르지 않습니다.');
            return false;
        }

        return true;
    }

    /**
     * 자막 보정 처리
     */
    async function processSyncSubtitle() {
        // 유효성 검사
        if (!validateFiles()) {
            return;
        }

        // UI 초기화
        hideError();
        $statsSection.addClass('hidden');
        $resultSection.addClass('hidden');
        $processBtn.prop('disabled', true);

        updateProgress(10, 'VAD 모델 로드 중...');

        try {
            // FormData 생성
            const formData = new FormData();
            formData.append('srt_file', $srtFile[0].files[0]);
            formData.append('audio_file', $audioFile[0].files[0]);

            // 설정 추가
            const config = collectConfig();
            formData.append('config', JSON.stringify(config));

            updateProgress(30, '음성 구간 검출 중...');

            // API 호출
            const response = await fetch('/api/subtitle_sync/process', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || '처리 중 오류가 발생했습니다.');
            }

            updateProgress(70, '자막 보정 중...');

            const result = await response.json();

            updateProgress(100, '완료!');

            // 통계 표시
            if (result.stats) {
                showStats(result.stats);
            }

            // 결과 표시
            if (result.corrected_srt) {
                showResult(result.corrected_srt);
            }

            // 진행 바 숨기기 (2초 후)
            setTimeout(() => {
                $progressSection.addClass('hidden');
            }, 2000);

        } catch (error) {
            console.error('처리 오류:', error);
            showError(error.message || '알 수 없는 오류가 발생했습니다.');
        } finally {
            $processBtn.prop('disabled', false);
        }
    }

    /**
     * SRT 파일 다운로드
     */
    function downloadSrt() {
        if (!correctedSrtContent) {
            alert('다운로드할 자막이 없습니다.');
            return;
        }

        // 파일명 생성
        const originalFileName = $srtFile[0].files[0]?.name || 'subtitle.srt';
        const newFileName = originalFileName.replace('.srt', '_synced.srt');

        // Blob 생성
        const blob = new Blob([correctedSrtContent], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);

        // 다운로드 링크 생성 및 클릭
        const a = document.createElement('a');
        a.href = url;
        a.download = newFileName;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    /**
     * 커스텀 파일 선택 버튼
     */
    $srtFileBtn.on('click', function () {
        $srtFile.click();
    });

    $audioFileBtn.on('click', function () {
        $audioFile.click();
    });

    // 파일 선택 시 파일명 표시
    $srtFile.on('change', function () {
        const file = this.files[0];
        if (file) {
            $srtFileName.text(file.name);
            $srtFileName.removeClass('text-slate-500').addClass('text-slate-900');
        } else {
            $srtFileName.text('파일을 선택하세요');
            $srtFileName.removeClass('text-slate-900').addClass('text-slate-500');
        }
        hideError();
    });

    $audioFile.on('change', function () {
        const file = this.files[0];
        if (file) {
            $audioFileName.text(file.name);
            $audioFileName.removeClass('text-slate-500').addClass('text-slate-900');
        } else {
            $audioFileName.text('파일을 선택하세요');
            $audioFileName.removeClass('text-slate-900').addClass('text-slate-500');
        }
        hideError();
    });

    /**
     * 이벤트 리스너
     */
    $processBtn.on('click', function () {
        processSyncSubtitle();
    });

    $downloadBtn.on('click', function () {
        downloadSrt();
    });

    // 드래그 앤 드롭 지원 (선택 사항)
    function setupDragAndDrop($input, $button) {
        $button.on('dragover', function (e) {
            e.preventDefault();
            e.stopPropagation();
            $(this).addClass('border-[#0c77f2] bg-blue-50');
        });

        $button.on('dragleave', function (e) {
            e.preventDefault();
            e.stopPropagation();
            $(this).removeClass('border-[#0c77f2] bg-blue-50');
        });

        $button.on('drop', function (e) {
            e.preventDefault();
            e.stopPropagation();
            $(this).removeClass('border-[#0c77f2] bg-blue-50');

            const files = e.originalEvent.dataTransfer.files;
            if (files.length > 0) {
                $input[0].files = files;
                $input.trigger('change');
            }
        });
    }

    setupDragAndDrop($srtFile, $srtFileBtn);
    setupDragAndDrop($audioFile, $audioFileBtn);

    // 청크 모드 변경 시 UI 업데이트
    $chunkMode.on('change', updateChunkModeUI);

    // 초기 UI 상태 설정
    updateChunkModeUI();

    console.log('자막 보정 싱크 페이지 초기화 완료');
});
