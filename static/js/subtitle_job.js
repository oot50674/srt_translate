(() => {
    const jobId = document.body.dataset.jobId;
    if (!jobId) {
        return;
    }

    const STATUS_STYLES = {
        pending: 'bg-slate-200 text-slate-700',
        running: 'bg-blue-100 text-blue-800',
        completed: 'bg-green-100 text-green-800',
        failed: 'bg-red-100 text-red-800',
        cancelled: 'bg-amber-100 text-amber-700'
    };
    const SEGMENT_STATUS_STYLES = {
        pending: 'bg-slate-100 text-slate-600',
        processing: 'bg-amber-100 text-amber-700',
        completed: 'bg-green-100 text-green-800',
        error: 'bg-red-100 text-red-700',
        cancelled: 'bg-amber-50 text-amber-700'
    };
    const TERMINAL_STATES = new Set(['completed', 'failed', 'cancelled']);
    const POLL_INTERVAL = 4000;

    const statusPill = document.getElementById('job-status-pill');
    const phaseText = document.getElementById('job-phase-text');
    const progressBar = document.getElementById('job-progress-bar');
    const progressText = document.getElementById('job-progress-text');
    const metaContainer = document.getElementById('job-meta');
    const segmentsContainer = document.getElementById('segments-container');
    const segmentsPlaceholder = document.getElementById('segments-placeholder');
    const segmentCounter = document.getElementById('segment-counter');
    const alertBox = document.getElementById('job-alert');
    const logList = document.getElementById('job-log');
    const refreshBtn = document.getElementById('refresh-btn');
    const downloadSrtLink = document.getElementById('download-srt');
    const stopBtn = document.getElementById('stop-job-btn');
    const retrySelectedBtn = document.getElementById('retry-selected-btn');

    let pollTimer = null;
    let terminalReached = false;
    let latestJobData = null;
    const selectedSegments = new Set();
    const RETRY_ALLOWED_STATES = new Set(['completed', 'failed']);
    let autoSelectedErrors = false;

    function setAlert(message, type = 'info') {
        if (!alertBox) return;
        if (!message) {
            alertBox.classList.add('hidden');
            alertBox.textContent = '';
            return;
        }
        const palette = type === 'error'
            ? 'text-red-800 bg-red-50 border-red-200'
            : type === 'success'
                ? 'text-emerald-800 bg-emerald-50 border-emerald-200'
                : 'text-amber-800 bg-amber-50 border-amber-300';
        alertBox.className = `rounded-lg border px-4 py-3 text-sm ${palette}`;
        alertBox.textContent = message;
    }

    function formatTime(value) {
        if (!value && value !== 0) return '-';
        const date = new Date(value * 1000);
        return date.toLocaleTimeString();
    }

    function canRetry(job) {
        return Boolean(job && RETRY_ALLOWED_STATES.has(job.status) && !job.stop_requested);
    }

    function renderMeta(job) {
        if (!metaContainer) return;
        const entries = [];
        entries.push({
            label: '청크 길이',
            value: `${job.chunk_minutes || 0}분`
        });
        const isVideoSource = job.source_has_video !== false;
        entries.push({
            label: '소스 유형',
            value: isVideoSource ? '영상' : '오디오'
        });
        let modeLabel = '전사';
        if (job.mode === 'translate') {
            modeLabel = `번역 (${job.target_language || '-'})`;
        } else if (job.mode === 'whisper_only') {
            modeLabel = 'Whisper 전사';
        }
        entries.push({
            label: '모드',
            value: modeLabel
        });
        entries.push({
            label: '원본 소스',
            value: job.source_file || job.youtube_url || '-'
        });
        entries.push({
            label: '처리된 세그먼트',
            value: `${job.processed_segments}/${job.total_segments}`
        });
        metaContainer.innerHTML = entries.map(entry => `
            <div class="rounded-lg border border-slate-200 p-4">
                <p class="text-sm text-slate-500">${entry.label}</p>
                <p class="text-base font-semibold text-slate-900 mt-1">${entry.value}</p>
            </div>
        `).join('');
    }

    function syncSegmentSelection(job) {
        if (!canRetry(job)) {
            selectedSegments.clear();
            return;
        }
        const available = new Set((job.segments || []).map(seg => seg.index));
        Array.from(selectedSegments).forEach(idx => {
            if (!available.has(idx)) {
                selectedSegments.delete(idx);
            }
        });
    }

    function renderSegments(segments, job) {
        syncSegmentSelection(job);
        const canRetryNow = canRetry(job);
        if (!segmentsContainer) return;
        if (!segments || !segments.length) {
            segmentsPlaceholder?.classList.remove('hidden');
            segmentsContainer.innerHTML = '';
            segmentCounter.textContent = '';
            updateRetryButton(job);
            return;
        }
        segmentCounter.textContent = `${job.processed_segments}/${job.total_segments} 완료`;
        segmentsPlaceholder?.classList.add('hidden');
        segmentsContainer.innerHTML = segments.map(segment => {
            const badgeStyle = SEGMENT_STATUS_STYLES[segment.status] || 'bg-slate-100 text-slate-600';
            const speechInfo = segment.speech_segments?.length
                ? `${segment.speech_segments.length}개 발화`
                : '발화 정보 없음';
            const downloadHref = `/api/subtitle/jobs/${encodeURIComponent(job.job_id)}/segments/${segment.index}/download`;
            const downloadClasses = segment.download_available
                ? 'rounded-md border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-700 hover:bg-slate-50 transition-colors'
                : 'rounded-md border border-slate-200 px-3 py-2 text-sm font-semibold text-slate-400 pointer-events-none opacity-60';
            const checkboxId = `segment-select-${segment.index}`;
            const isChecked = selectedSegments.has(segment.index);
            const checkboxAttrs = [
                `type="checkbox"`,
                `class="segment-select-checkbox h-4 w-4 text-[#0c77f2]"`,
                `data-segment-index="${segment.index}"`,
                `id="${checkboxId}"`,
                `aria-label="세그먼트 #${segment.index} 재시도 선택"`,
            ];
            if (isChecked) checkboxAttrs.push('checked');
            if (!canRetryNow) checkboxAttrs.push('disabled');
            return `
                <div class="rounded-xl border border-slate-200 p-5 flex flex-col gap-2 bg-white">
                    <div class="flex items-start justify-between gap-3">
                        <div class="flex items-start gap-3">
                            <input ${checkboxAttrs.join(' ')} />
                            <div>
                                <p class="font-semibold text-slate-900">세그먼트 #${segment.index}</p>
                                <p class="text-xs text-slate-500">${segment.start_time.toFixed(1)}s ~ ${segment.end_time.toFixed(1)}s · ${speechInfo}</p>
                            </div>
                        </div>
                        <span class="text-xs font-semibold px-3 py-1 rounded-full ${badgeStyle}">${segment.status}</span>
                    </div>
                    <p class="text-sm text-slate-600">${segment.message || ''}</p>
                    <div class="flex flex-wrap gap-2">
                        <a href="${downloadHref}" class="inline-flex items-center gap-2 ${downloadClasses}">
                            다운로드
                        </a>
                    </div>
                </div>
            `;
        }).join('');
        segmentsContainer.querySelectorAll('.segment-select-checkbox').forEach(checkbox => {
            checkbox.addEventListener('change', () => {
                const idx = Number(checkbox.dataset.segmentIndex);
                if (Number.isNaN(idx)) return;
                if (checkbox.checked) {
                    selectedSegments.add(idx);
                } else {
                    selectedSegments.delete(idx);
                }
                updateRetryButton(job);
            });
        });
        updateRetryButton(job);
    }

    function renderLogs(logs) {
        if (!logList) return;
        if (!logs || !logs.length) {
            logList.innerHTML = '<li class="text-slate-400 text-sm">로그가 아직 없습니다.</li>';
            return;
        }
        logList.innerHTML = logs.slice().reverse().map(entry => {
            const timeText = formatTime(entry.timestamp);
            const levelClass = entry.level === 'error' ? 'text-red-600' : 'text-slate-600';
        return `
                <li class="flex items-start gap-2">
                    <span class="text-xs text-slate-400 w-24 flex-shrink-0">${timeText}</span>
                    <span class="text-xs uppercase font-semibold ${levelClass} flex-shrink-0">${entry.level}</span>
                    <span class="flex-1 min-w-0 text-slate-700 break-all">${entry.message}</span>
                </li>
            `;
        }).join('');
    }

    function toggleDownload(job) {
        if (!downloadSrtLink) return;
        if (job.transcript_ready) {
            downloadSrtLink.href = `/api/subtitle/jobs/${encodeURIComponent(job.job_id)}/download/srt`;
            downloadSrtLink.classList.remove('pointer-events-none', 'opacity-50', 'bg-slate-300');
            downloadSrtLink.classList.add('bg-[#0c77f2]', 'hover:bg-blue-600', 'text-white');
        } else {
            downloadSrtLink.removeAttribute('href');
            downloadSrtLink.classList.add('pointer-events-none', 'opacity-50', 'bg-slate-300');
        }
    }

    function toggleStopButton(job) {
        if (!stopBtn) return;
        if (TERMINAL_STATES.has(job.status)) {
            stopBtn.disabled = true;
            stopBtn.textContent = job.status === 'cancelled' ? '작업 중지됨' : '작업 종료';
            stopBtn.classList.add('opacity-60', 'cursor-not-allowed');
        } else if (job.stop_requested) {
            stopBtn.disabled = true;
            stopBtn.textContent = '중지 요청됨';
            stopBtn.classList.add('opacity-60', 'cursor-not-allowed');
        } else {
            stopBtn.disabled = false;
            stopBtn.textContent = '작업 중지';
            stopBtn.classList.remove('opacity-60', 'cursor-not-allowed');
        }
    }

    function updateRetryButton(job) {
        if (!retrySelectedBtn) return;
        const can = canRetry(job);
        const selectionCount = selectedSegments.size;
        retrySelectedBtn.disabled = !can || selectionCount === 0;
        const baseText = selectionCount > 0 ? `선택 세그먼트 재시도 (${selectionCount})` : '선택 세그먼트 재시도';
        retrySelectedBtn.textContent = baseText;
        if (!can) {
            retrySelectedBtn.classList.add('opacity-60', 'cursor-not-allowed');
        } else {
            retrySelectedBtn.classList.remove('opacity-60', 'cursor-not-allowed');
        }
    }

    function ensureAutoSelectFailedSegments(job) {
        if (!job) {
            return;
        }
        if (!TERMINAL_STATES.has(job.status)) {
            autoSelectedErrors = false;
            return;
        }
        if (autoSelectedErrors) {
            return;
        }
        const segments = job.segments || [];
        let added = false;
        segments.forEach(segment => {
            if (segment.status === 'error' && !selectedSegments.has(segment.index)) {
                selectedSegments.add(segment.index);
                added = true;
            }
        });
        if (added) {
            updateRetryButton(job);
        }
        autoSelectedErrors = true;
    }

    function updateJob(job) {
        ensureAutoSelectFailedSegments(job);
        latestJobData = job;
        const percent = Math.round((job.progress || 0) * 100);
        progressText.textContent = `${percent}%`;
        progressBar.style.width = `${percent}%`;
        phaseText.textContent = job.message || '';
        const pillStyle = STATUS_STYLES[job.status] || STATUS_STYLES.pending;
        statusPill.className = `rounded-full px-3 py-1 text-xs font-semibold ${pillStyle}`;
        statusPill.textContent = job.status;
        renderMeta(job);
        renderSegments(job.segments, job);
        renderLogs(job.logs);
        toggleDownload(job);
        toggleStopButton(job);
        updateRetryButton(job);

        if (job.status === 'failed' && job.error) {
            setAlert(job.error, 'error');
        } else if (job.status === 'cancelled') {
            setAlert(job.message || '작업이 중지되었습니다.', 'warning');
        } else if (job.status === 'completed') {
            setAlert('작업이 완료되었습니다.', 'success');
        } else if (job.stop_requested) {
            setAlert('사용자 요청에 따라 작업 중지를 준비하고 있습니다.', 'warning');
        } else {
            setAlert('');
        }
    }

    async function fetchJob() {
        try {
            const res = await fetch(`/api/subtitle/jobs/${encodeURIComponent(jobId)}`);
            if (!res.ok) {
                throw new Error('상태를 불러오지 못했습니다.');
            }
            const data = await res.json();
            updateJob(data);
            if (TERMINAL_STATES.has(data.status)) {
                terminalReached = true;
                return;
            }
        } catch (err) {
            console.warn(err);
            setAlert('상태를 불러오지 못했습니다. 잠시 후 다시 시도하세요.', 'warning');
        } finally {
            if (!terminalReached) {
                pollTimer = setTimeout(fetchJob, POLL_INTERVAL);
            }
        }
    }

    refreshBtn?.addEventListener('click', () => {
        if (pollTimer) clearTimeout(pollTimer);
        terminalReached = false;
        fetchJob();
    });

    stopBtn?.addEventListener('click', async () => {
        if (stopBtn.disabled) return;
        const confirmed = window.confirm('현재 작업을 중지할까요? 진행 중인 세그먼트 이후의 작업은 중단됩니다.');
        if (!confirmed) return;
        stopBtn.disabled = true;
        stopBtn.textContent = '중지 요청 중...';
        stopBtn.classList.add('opacity-60', 'cursor-not-allowed');
        try {
            const res = await fetch(`/api/subtitle/jobs/${encodeURIComponent(jobId)}/stop`, {
                method: 'POST'
            });
            if (!res.ok) {
                const payload = await res.json().catch(() => ({}));
                throw new Error(payload.error || '작업 중지 요청에 실패했습니다.');
            }
            setAlert('작업 중지 요청을 보냈습니다. 잠시만 기다려 주세요.', 'warning');
            if (pollTimer) clearTimeout(pollTimer);
            terminalReached = false;
            fetchJob();
        } catch (err) {
            console.error(err);
            stopBtn.disabled = false;
            stopBtn.textContent = '작업 중지';
            stopBtn.classList.remove('opacity-60', 'cursor-not-allowed');
            setAlert(err.message || '작업 중지 요청에 실패했습니다.', 'error');
        }
    });

    retrySelectedBtn?.addEventListener('click', async () => {
        if (!latestJobData) return;
        if (retrySelectedBtn.disabled) return;
        if (selectedSegments.size === 0) {
            setAlert('재시도할 세그먼트를 선택해 주세요.', 'warning');
            return;
        }
        const segments = Array.from(selectedSegments).sort((a, b) => a - b);
        retrySelectedBtn.disabled = true;
        retrySelectedBtn.textContent = '재시도 요청 중...';
        try {
            const res = await fetch(`/api/subtitle/jobs/${encodeURIComponent(jobId)}/retry`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ segments })
            });
            const payload = await res.json().catch(() => ({}));
            if (!res.ok) {
                throw new Error(payload.error || '세그먼트 재시도 요청에 실패했습니다.');
            }
            selectedSegments.clear();
            setAlert('선택한 세그먼트 재시도 요청을 보냈습니다.', 'success');
            if (pollTimer) clearTimeout(pollTimer);
            terminalReached = false;
            fetchJob();
        } catch (err) {
            console.error(err);
            setAlert(err.message || '세그먼트 재시도 요청에 실패했습니다.', 'error');
            updateRetryButton(latestJobData);
        }
    });

    fetchJob();
})();
