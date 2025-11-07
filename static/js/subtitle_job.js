(() => {
    const jobId = document.body.dataset.jobId;
    if (!jobId) {
        return;
    }

    const STATUS_STYLES = {
        pending: 'bg-slate-200 text-slate-700',
        running: 'bg-blue-100 text-blue-800',
        completed: 'bg-green-100 text-green-800',
        failed: 'bg-red-100 text-red-800'
    };
    const SEGMENT_STATUS_STYLES = {
        pending: 'bg-slate-100 text-slate-600',
        processing: 'bg-amber-100 text-amber-700',
        completed: 'bg-green-100 text-green-800',
        error: 'bg-red-100 text-red-700'
    };
    const TERMINAL_STATES = new Set(['completed', 'failed']);
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

    let pollTimer = null;
    let terminalReached = false;

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

    function renderMeta(job) {
        if (!metaContainer) return;
        const entries = [];
        entries.push({
            label: '청크 길이',
            value: `${job.chunk_minutes || 0}분`
        });
        entries.push({
            label: '모드',
            value: job.mode === 'translate' ? `번역 (${job.target_language || '-'})` : '전사'
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

    function renderSegments(segments, job) {
        if (!segmentsContainer) return;
        if (!segments || !segments.length) {
            segmentsPlaceholder?.classList.remove('hidden');
            segmentsContainer.innerHTML = '';
            segmentCounter.textContent = '';
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
            return `
                <div class="rounded-xl border border-slate-200 p-5 flex flex-col gap-2 bg-white">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="font-semibold text-slate-900">세그먼트 #${segment.index}</p>
                            <p class="text-xs text-slate-500">${segment.start_time.toFixed(1)}s ~ ${segment.end_time.toFixed(1)}s · ${speechInfo}</p>
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
                <li class="flex items-center gap-2">
                    <span class="text-xs text-slate-400 w-24">${timeText}</span>
                    <span class="text-xs uppercase font-semibold ${levelClass}">${entry.level}</span>
                    <span class="flex-1 text-slate-700">${entry.message}</span>
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

    function updateJob(job) {
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

        if (job.status === 'failed' && job.error) {
            setAlert(job.error, 'error');
        } else if (job.status === 'completed') {
            setAlert('작업이 완료되었습니다.', 'success');
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

    fetchJob();
})();
