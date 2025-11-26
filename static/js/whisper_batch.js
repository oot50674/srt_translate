(() => {
    const batchId = document.body.dataset.batchId;
    if (!batchId) return;

    const POLL_INTERVAL = 4000;
    const RECONNECT_DELAY = 5000;
    let pollTimer = null;
    let reconnectTimer = null;
    let websocket = null;
    let websocketFailures = 0;
    let finished = false;
    let usePollingFallback = !('WebSocket' in window);
    let ignoreSocketAlerts = false;

    const alertBox = document.getElementById('batch-alert');
    const statusText = document.getElementById('whisper-batch-status-text');
    const progressBar = document.getElementById('whisper-batch-progress-bar');
    const progressText = document.getElementById('whisper-batch-progress-text');
    const totalCountEl = document.getElementById('whisper-total-count');
    const completeCountEl = document.getElementById('whisper-complete-count');
    const failedCountEl = document.getElementById('whisper-failed-count');
    const filesList = document.getElementById('whisper-files-list');
    const filesPlaceholder = document.getElementById('whisper-files-placeholder');
    const updatedAtEl = document.getElementById('whisper-updated-at');

    const STATUS_BADGE = {
        pending: 'bg-slate-100 text-slate-600',
        running: 'bg-blue-100 text-blue-700',
        completed: 'bg-emerald-100 text-emerald-700',
        failed: 'bg-red-100 text-red-700',
        cancelled: 'bg-amber-100 text-amber-700',
    };

    const SEGMENT_STATUS_BADGE = {
        pending: 'bg-slate-200 text-slate-600',
        running: 'bg-blue-200 text-blue-800',
        completed: 'bg-emerald-200 text-emerald-800',
        failed: 'bg-red-200 text-red-800',
    };

    const SEGMENT_STATUS_LABEL = {
        pending: '대기',
        running: '진행 중',
        completed: '완료',
        failed: '실패',
    };

    function setAlert(message, type = 'error') {
        if (!alertBox) return;
        if (!message) {
            alertBox.classList.add('hidden');
            alertBox.textContent = '';
            return;
        }
        const palette = type === 'error'
            ? 'text-red-800 bg-red-50 border-red-200'
            : 'text-emerald-800 bg-emerald-50 border-emerald-200';
        alertBox.className = `rounded-lg border px-4 py-3 text-sm ${palette}`;
        alertBox.textContent = message;
        alertBox.classList.remove('hidden');
    }

    function formatTime(value) {
        if (!value && value !== 0) return '-';
        const date = new Date(value * 1000);
        return date.toLocaleTimeString();
    }

    function updateSummary(data) {
        const overallPercent = Math.round((data.overall_progress || 0) * 100);
        if (progressBar) {
            progressBar.style.width = `${overallPercent}%`;
        }
        if (progressText) {
            progressText.textContent = `${overallPercent}%`;
        }
        if (statusText) {
            let label = '진행 중';
            if (data.completed_items === data.total_items) {
                label = '모든 파일을 전사했습니다.';
            } else if (data.completed_items + data.failed_items === data.total_items) {
                label = '일부 파일에서 오류가 발생했습니다.';
            }
            statusText.textContent = label;
        }
        if (totalCountEl) totalCountEl.textContent = data.total_items ?? 0;
        if (completeCountEl) completeCountEl.textContent = data.completed_items ?? 0;
        if (failedCountEl) failedCountEl.textContent = data.failed_items ?? 0;
        if (updatedAtEl) updatedAtEl.textContent = `업데이트: ${formatTime(data.updated_at)}`;
    }

    function buildSegmentRows(item) {
        if (!item.segment_progress?.length) {
            return '';
        }
        const rows = item.segment_progress.map((segment) => {
            const rawPercent = (segment.progress || 0) * 100;
            const clampedPercent = Math.max(0, Math.min(100, rawPercent));
            const percentText = `${clampedPercent.toFixed(1)}%`;
            const badgeClass = SEGMENT_STATUS_BADGE[segment.status] || SEGMENT_STATUS_BADGE.pending;
            const label = SEGMENT_STATUS_LABEL[segment.status] || segment.status || '대기';
            return `
                <div class="space-y-1 rounded-lg bg-white/70 p-3">
                    <div class="flex items-center justify-between text-xs text-slate-500">
                        <span>세그먼트 #${segment.segment_index}</span>
                        <span class="text-[11px] font-semibold px-2 py-0.5 rounded-full ${badgeClass}">${label}</span>
                    </div>
                    <div class="flex items-center justify-between text-[11px] text-slate-400">
                        <span>${segment.start_timecode || '-'} ~ ${segment.end_timecode || '-'}</span>
                        <span>${percentText}</span>
                    </div>
                    <div class="w-full rounded-full bg-slate-200 h-1.5 overflow-hidden">
                        <div class="h-full rounded-full bg-slate-500" style="width:${clampedPercent}%;"></div>
                    </div>
                    <p class="text-[11px] text-slate-500">${segment.message || ''}</p>
                </div>
            `;
        }).join('');
        return `
            <div class="rounded-lg border border-slate-100 bg-slate-50 p-3 space-y-2">
                <p class="text-xs font-semibold text-slate-500">세그먼트 진행률</p>
                <div class="space-y-2 max-h-64 overflow-y-auto pr-1">${rows}</div>
            </div>
        `;
    }

    function buildFileCard(item) {
        const percent = Math.round((item.progress || 0) * 100);
        const badgeClass = STATUS_BADGE[item.status] || STATUS_BADGE.pending;
        const downloadBtn = item.transcript_ready
            ? `<a href="${item.download_url}" class="rounded-md bg-[#0c77f2] text-white text-xs font-semibold px-3 py-2 hover:bg-blue-600 transition-colors">SRT 다운로드</a>`
            : '';
        const hasRemovalCount = Number.isFinite(item.silent_entries_removed);
        const removalText = hasRemovalCount
            ? ` / 환각 제거 ${item.silent_entries_removed}개`
            : '';
        const segmentsMarkup = buildSegmentRows(item);
        return `
            <div class="rounded-lg border border-slate-200 p-4 bg-white flex flex-col gap-3">
                <div class="flex items-center justify-between gap-3">
                    <div class="flex flex-col">
                        <p class="text-base font-semibold text-slate-900">${item.file_name}</p>
                        <p class="text-xs text-slate-500">${(item.message || '') + removalText}</p>
                    </div>
                    <span class="text-xs font-semibold px-3 py-1 rounded-full ${badgeClass}">${item.status}</span>
                </div>
                <div class="flex items-center justify-between text-xs text-slate-500">
                    <span>진행률</span>
                    <span>${percent}%</span>
                </div>
                <div class="w-full rounded-full bg-slate-200 h-2 overflow-hidden">
                    <div class="h-full rounded-full bg-[#0c77f2]" style="width:${percent}%;"></div>
                </div>
                ${segmentsMarkup}
                <div class="flex items-center justify-between">
                    <span class="text-xs text-slate-400">업데이트: ${formatTime(item.updated_at)}</span>
                    ${downloadBtn}
                </div>
            </div>
        `;
    }

    function renderFiles(items) {
        if (!filesList || !filesPlaceholder) return;
        if (!items?.length) {
            filesPlaceholder.classList.remove('hidden');
            filesList.innerHTML = '';
            return;
        }
        filesPlaceholder.classList.add('hidden');
        filesList.innerHTML = items.map(buildFileCard).join('');
    }


    function handleData(data) {
        setAlert('');
        updateSummary(data);
        renderFiles(data.items);
        const total = data.total_items || 0;
        const finishedCount = (data.completed_items || 0) + (data.failed_items || 0);
        if (total && finishedCount >= total) {
            finished = true;
            stopPolling();
            disconnectWebsocket();
        }
    }

    async function fetchState() {
        if (!usePollingFallback || finished) {
            return;
        }
        try {
            const response = await fetch(`/api/whisper/batch/${encodeURIComponent(batchId)}`);
            if (!response.ok) {
                throw new Error('상태를 불러오지 못했습니다.');
            }
            const payload = await response.json();
            handleData(payload);
        } catch (error) {
            console.error(error);
            setAlert(error.message || '상태 확인 중 오류가 발생했습니다.');
        } finally {
            schedulePoll();
        }
    }

    function schedulePoll() {
        if (!usePollingFallback || finished) {
            return;
        }
        if (pollTimer) {
            clearTimeout(pollTimer);
        }
        pollTimer = setTimeout(fetchState, POLL_INTERVAL);
    }

    function stopPolling() {
        usePollingFallback = false;
        if (pollTimer) {
            clearTimeout(pollTimer);
            pollTimer = null;
        }
    }

    function startPollingFallback() {
        if (usePollingFallback) {
            if (!pollTimer) {
                fetchState();
            }
            return;
        }
        usePollingFallback = true;
        disconnectWebsocket();
        fetchState();
    }

    function disconnectWebsocket() {
        // Suppress transient socket error alerts for intentional disconnects
        ignoreSocketAlerts = true;
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
        if (websocket) {
            try {
                websocket.close();
            } catch (err) {
                console.warn(err);
            }
            websocket = null;
        }
    }

    function scheduleReconnect() {
        if (usePollingFallback || finished || reconnectTimer) {
            return;
        }
        reconnectTimer = setTimeout(() => {
            reconnectTimer = null;
            connectWebsocket();
        }, RECONNECT_DELAY);
    }

    function connectWebsocket() {
        if (usePollingFallback || finished || websocket) {
            return;
        }
        if (!('WebSocket' in window)) {
            startPollingFallback();
            return;
        }

        const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
        const socket = new WebSocket(`${protocol}://${window.location.host}/ws/whisper/batch/${encodeURIComponent(batchId)}`);
        websocket = socket;

        socket.addEventListener('open', () => {
            websocketFailures = 0;
            stopPolling();
            // new connection, re-enable socket alerts
            ignoreSocketAlerts = false;
        });

        socket.addEventListener('message', (event) => {
            try {
                const payload = JSON.parse(event.data);
                if (payload?.type === 'batch_state' && payload.payload) {
                    handleData(payload.payload);
                } else if (payload?.type === 'error') {
                    setAlert(payload.error || '실시간 업데이트 오류가 발생했습니다.');
                }
            } catch (err) {
                console.error(err);
            }
        });

        const handleSocketDrop = () => {
            if (finished) {
                return;
            }
            websocket = null;
            websocketFailures += 1;
            if (websocketFailures >= 3) {
                startPollingFallback();
            } else {
                scheduleReconnect();
            }
        };

        socket.addEventListener('close', handleSocketDrop);
        socket.addEventListener('error', () => {
            if (!ignoreSocketAlerts && !finished) {
                setAlert('실시간 연결이 원활하지 않습니다. 재시도합니다.');
            }
            handleSocketDrop();
        });
    }

    if (usePollingFallback) {
        fetchState();
    } else {
        connectWebsocket();
    }
})();
