(() => {
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

    const registerComponent = () => {
        if (!window.Alpine || registerComponent.__registered) {
            return;
        }
        registerComponent.__registered = true;
        Alpine.data('whisperBatchState', (batchId) => ({
            batchId,
            alert: null,
            finished: false,
            summary: {
                modelName: '-',
                totalItems: 0,
                completedItems: 0,
                failedItems: 0,
                overallPercent: 0,
                statusLabel: '진행 중',
            },
            files: [],
            POLL_INTERVAL: 4000,
            RECONNECT_DELAY: 5000,
            pollTimer: null,
            reconnectTimer: null,
            websocket: null,
            websocketFailures: 0,
            usePollingFallback: !('WebSocket' in window),
            ignoreSocketAlerts: false,
            segmentScrollMap: new Map(),
            boundCleanup: null,
            init() {
                if (!this.batchId) {
                    this.setAlert('배치 ID가 유효하지 않습니다.');
            return;
        }
                this.boundCleanup = this.cleanup.bind(this);
                window.addEventListener('beforeunload', this.boundCleanup);
                if (this.usePollingFallback) {
                    this.fetchState();
                } else {
                    this.connectWebsocket();
    }
            },
            cleanup() {
                this.disconnectWebsocket();
                this.stopPolling();
                if (this.boundCleanup) {
                    window.removeEventListener('beforeunload', this.boundCleanup);
                    this.boundCleanup = null;
                }
            },
            setAlert(message, type = 'error') {
                this.alert = message ? { message, type } : null;
            },
            clearAlert() {
                this.alert = null;
            },
            get hasAlert() {
                return Boolean(this.alert?.message);
            },
            get alertClass() {
                if (this.alert?.type === 'success') {
                    return 'text-emerald-800 bg-emerald-50 border-emerald-200';
                }
                return 'text-red-800 bg-red-50 border-red-200';
            },
            get statusLabel() {
                return this.summary.statusLabel;
            },
            get overallPercentText() {
                return `${this.summary.overallPercent}%`;
            },
            get filesEmpty() {
                return this.files.length === 0;
            },
            clampPercent(progress) {
                const numeric = Number(progress);
                const value = Number.isFinite(numeric) ? numeric : 0;
                return Math.max(0, Math.min(100, value * 100));
            },
            filePercent(item) {
                return Math.round(this.clampPercent(item?.progress ?? 0));
            },
            segmentPercent(segment) {
                return this.clampPercent(segment?.progress ?? 0);
            },
            segmentPercentText(segment) {
                return `${this.segmentPercent(segment).toFixed(1)}%`;
            },
            segmentProgressWidth(segment) {
                return `${this.segmentPercent(segment)}%`;
            },
            fileSubtitle(item) {
                const base = item?.message || '';
                const hasRemovalCount = Number.isFinite(item?.silent_entries_removed);
                const removalText = hasRemovalCount ? ` / 환각 제거 ${item.silent_entries_removed}개` : '';
                return `${base}${removalText}`;
            },
            hasSegments(item) {
                return Array.isArray(item?.segment_progress) && item.segment_progress.length > 0;
            },
            statusBadgeClass(status) {
                return STATUS_BADGE[status] || STATUS_BADGE.pending;
            },
            segmentBadgeClass(status) {
                return SEGMENT_STATUS_BADGE[status] || SEGMENT_STATUS_BADGE.pending;
            },
            segmentStatusLabel(status) {
                return SEGMENT_STATUS_LABEL[status] || status || '대기';
            },
            cacheSegmentScroll() {
                if (!this.$refs?.filesList) return;
                // 세그먼트 리스트의 스크롤 위치를 저장해 새 데이터에서도 위치를 유지한다.
                this.segmentScrollMap.clear();
                const cards = this.$refs.filesList.querySelectorAll('.file-card');
                cards.forEach((card) => {
                    const fileName = card?.dataset?.fileName;
            const container = card.querySelector('.segment-list-container');
            if (fileName && container) {
                        this.segmentScrollMap.set(fileName, container.scrollTop);
            }
        });
            },
            restoreSegmentScroll() {
                if (!this.$refs?.filesList || this.segmentScrollMap.size === 0) return;
                // DOM 갱신 이후에 저장된 스크롤 값을 다시 적용한다.
                this.$nextTick(() => {
                    const cards = this.$refs.filesList.querySelectorAll('.file-card');
                    cards.forEach((card) => {
                        const fileName = card?.dataset?.fileName;
            const container = card.querySelector('.segment-list-container');
                        if (fileName && container && this.segmentScrollMap.has(fileName)) {
                            container.scrollTop = this.segmentScrollMap.get(fileName);
            }
        });
                });
            },
            updateSummary(data = {}) {
                const totalItems = data?.total_items ?? 0;
                const completedItems = data?.completed_items ?? 0;
                const failedItems = data?.failed_items ?? 0;
                const percent = Math.round(this.clampPercent(data?.overall_progress ?? 0));
                let statusLabel = '진행 중';
                if (totalItems && completedItems === totalItems) {
                    statusLabel = '모든 파일을 전사했습니다.';
                } else if (totalItems && completedItems + failedItems === totalItems) {
                    statusLabel = '일부 파일에서 오류가 발생했습니다.';
                }
                this.summary = {
                    modelName: data?.model_name || 'large-v3',
                    totalItems,
                    completedItems,
                    failedItems,
                    overallPercent: percent,
                    statusLabel,
                };
            },
            updateFiles(items = []) {
                if (!Array.isArray(items)) {
                    this.files = [];
                    return;
                }
                this.cacheSegmentScroll();
                this.files = items.map((file) => ({ ...file }));
                this.restoreSegmentScroll();
            },
            handleData(data) {
                this.clearAlert();
                this.updateSummary(data);
                this.updateFiles(data?.items ?? []);
                const total = data?.total_items || 0;
                const finishedCount = (data?.completed_items || 0) + (data?.failed_items || 0);
        if (total && finishedCount >= total) {
                    this.finished = true;
                    this.stopPolling();
                    this.disconnectWebsocket();
        }
            },
            async fetchState() {
                if (!this.usePollingFallback || this.finished) {
            return;
        }
        try {
                    const response = await fetch(`/api/whisper/batch/${encodeURIComponent(this.batchId)}`);
            if (!response.ok) {
                throw new Error('상태를 불러오지 못했습니다.');
            }
            const payload = await response.json();
                    this.handleData(payload);
        } catch (error) {
                    const message = error?.message || '상태 확인 중 오류가 발생했습니다.';
                    this.setAlert(message);
        } finally {
                    this.schedulePoll();
        }
            },
            schedulePoll() {
                if (!this.usePollingFallback || this.finished) {
            return;
        }
                if (this.pollTimer) {
                    clearTimeout(this.pollTimer);
        }
                this.pollTimer = window.setTimeout(() => this.fetchState(), this.POLL_INTERVAL);
            },
            stopPolling() {
                if (this.pollTimer) {
                    clearTimeout(this.pollTimer);
                    this.pollTimer = null;
        }
            },
            startPollingFallback() {
                if (this.usePollingFallback) {
                    if (!this.pollTimer) {
                        this.fetchState();
            }
            return;
        }
                this.usePollingFallback = true;
                this.disconnectWebsocket();
                this.fetchState();
            },
            disconnectWebsocket() {
                // 의도적인 종료 시에는 경고 토스트를 숨긴다.
                this.ignoreSocketAlerts = true;
                if (this.reconnectTimer) {
                    clearTimeout(this.reconnectTimer);
                    this.reconnectTimer = null;
        }
                if (this.websocket) {
            try {
                        this.websocket.close();
                    } catch (error) {
                        console.warn(error);
            }
                    this.websocket = null;
        }
            },
            scheduleReconnect() {
                if (this.usePollingFallback || this.finished || this.reconnectTimer) {
            return;
        }
                this.reconnectTimer = window.setTimeout(() => {
                    this.reconnectTimer = null;
                    this.connectWebsocket();
                }, this.RECONNECT_DELAY);
            },
            connectWebsocket() {
                if (this.usePollingFallback || this.finished || this.websocket || !this.batchId) {
            return;
        }
        if (!('WebSocket' in window)) {
                    this.startPollingFallback();
            return;
        }

        const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
                const socket = new WebSocket(`${protocol}://${window.location.host}/ws/whisper/batch/${encodeURIComponent(this.batchId)}`);
                this.websocket = socket;

        socket.addEventListener('open', () => {
                    this.websocketFailures = 0;
                    this.usePollingFallback = false;
                    this.stopPolling();
                    this.ignoreSocketAlerts = false;
        });

        socket.addEventListener('message', (event) => {
            try {
                const payload = JSON.parse(event.data);
                if (payload?.type === 'batch_state' && payload.payload) {
                            this.handleData(payload.payload);
                } else if (payload?.type === 'error') {
                            this.setAlert(payload.error || '실시간 업데이트 오류가 발생했습니다.');
                }
                    } catch (error) {
                        console.error(error);
            }
        });

        const handleSocketDrop = () => {
                    if (this.finished) {
                return;
            }
                    this.websocket = null;
                    this.websocketFailures += 1;
                    if (this.websocketFailures >= 3) {
                        this.startPollingFallback();
            } else {
                        this.scheduleReconnect();
            }
        };

        socket.addEventListener('close', handleSocketDrop);
        socket.addEventListener('error', () => {
                    if (!this.ignoreSocketAlerts && !this.finished) {
                        this.setAlert('실시간 연결이 원활하지 않습니다. 재시도합니다.');
            }
            handleSocketDrop();
        });
            },
        }));
    };

    const initializeIfLate = () => {
        if (window.Alpine?.initTree) {
            window.Alpine.initTree(document.body);
        }
    };

    document.addEventListener('alpine:init', registerComponent);

    if (window.Alpine) {
        registerComponent();
        initializeIfLate();
    }
})();
