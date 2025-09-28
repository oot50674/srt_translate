// Toast Popup Library - Pure JS
(function () {
  /**
   * @typedef {object} ToastOptions
   * @property {string} [position='top-right'] - 토스트 팝업이 표시될 위치. 'top-right', 'top-left', 'bottom-right', 'bottom-left', 'top-center', 'bottom-center'
   * @property {number|false} [duration=4000] - 토스트가 자동으로 사라지는 시간 (ms). 0 또는 false로 설정하면 자동으로 닫히지 않음.
   * @property {boolean} [dismissible=true] - 닫기 버튼 표시 여부.
   * @property {number} [maxVisible=5] - 한 위치에 동시에 표시될 수 있는 최대 토스트 개수.
   * @property {string} [ariaLive='polite'] - 스크린 리더를 위한 aria-live 속성 값. 'polite', 'assertive'.
   * @property {string} [title] - 토스트의 제목.
   * @property {boolean} [focus=false] - 토스트 생성 시 포커스를 줄지 여부. (주로 confirm에서 사용)
   * @property {Array<ToastAction>} [actions] - 사용자 정의 액션 버튼 배열.
   */

  /**
   * @typedef {object} ToastAction
   * @property {string} label - 버튼에 표시될 텍스트.
   * @property {string} [className='secondary'] - 버튼에 적용할 CSS 클래스.
   * @property {function(Event): void} onClick - 버튼 클릭 시 실행될 함수.
   */

  /**
   * @typedef {ToastOptions} ConfirmToastOptions
   * @property {function(): void} onConfirm - '확인' 버튼 클릭 시 실행될 콜백.
   * @property {function(): void} [onCancel] - '취소' 버튼 클릭 시 실행될 콜백.
   * @property {string} [okText='확인'] - '확인' 버튼의 텍스트.
   * @property {string} [cancelText='취소'] - '취소' 버튼의 텍스트.
   */

  // 기본 옵션 정의
  const DEFAULTS = {
    position: 'top-right', // 토스트 위치
    duration: 4000, // 표시 시간(ms); confirm은 무시됨
    dismissible: true, // 닫기 버튼 표시 여부
    maxVisible: 5, // 한 번에 표시할 최대 토스트 개수
    ariaLive: 'polite', // 접근성용 aria-live 속성
  };

  // 위치별 CSS 클래스 매핑
  const POS_CLASS = {
    'top-right': 'tp-top-right',
    'top-left': 'tp-top-left',
    'bottom-right': 'tp-bottom-right',
    'bottom-left': 'tp-bottom-left',
    'top-center': 'tp-top-center',
    'bottom-center': 'tp-bottom-center',
  };

  // 상태 관리 객체
  const state = {
    defaults: { ...DEFAULTS },
    containers: new Map(), // position별 컨테이너 div 저장
  };

  // position에 맞는 컨테이너 div 반환 (없으면 생성)
  function getContainer(position) {
    const pos = POS_CLASS[position] ? position : DEFAULTS.position;
    if (state.containers.has(pos)) return state.containers.get(pos);
    const el = document.createElement('div');
    el.className = `tp-container ${POS_CLASS[pos]}`;
    el.dataset.position = pos;
    document.body.appendChild(el);
    state.containers.set(pos, el);
    return el;
  }

  // 최대 표시 개수 초과 시 오래된 토스트 자동 닫기
  function clampVisible(container, max) {
    const items = Array.from(container.children);
    if (items.length <= max) return;
    const excess = items.length - max;
    const isBottom = (container.dataset.position || '').includes('bottom');
    if (isBottom) {
      // bottom 위치: 오래된 토스트가 앞쪽
      for (let i = 0; i < excess; i++) hideToast(items[i]);
    } else {
      // top 위치: 오래된 토스트가 뒤쪽
      for (let i = 0; i < excess; i++) hideToast(items[items.length - 1 - i]);
    }
  }

  // 액션 버튼 생성 함수
  function createButton(label, className, onClick) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `tp-btn ${className || ''}`.trim();
    btn.textContent = label;
    if (typeof onClick === 'function') btn.addEventListener('click', onClick);
    return btn;
  }

  // 토스트 닫기(애니메이션 후 제거)
  function hideToast(el) {
    if (!el || el.__hiding) return;
    el.__hiding = true;
    el.classList.add('tp-hide');
    const remove = () => el.remove();
    el.addEventListener('animationend', remove, { once: true });
    // 애니메이션 이벤트 누락 대비 타임아웃
    setTimeout(remove, 400);
  }

  // 토스트 DOM 생성 및 표시, 옵션에 따라 동작 제어
  function makeToast({ type, title, message, options }) {
    const opts = { ...state.defaults, ...(options || {}) };
    const sticky = opts.duration === 0 || opts.duration === false; // duration이 0이면 자동 닫힘 없음
    const container = getContainer(opts.position);

    // 토스트 요소 생성
    const toast = document.createElement('div');
    toast.className = `tp-toast tp-${type}`;
    toast.setAttribute('role', type === 'alert' ? 'alert' : type === 'confirm' ? 'alertdialog' : 'status');
    toast.setAttribute('aria-live', opts.ariaLive);
    toast.tabIndex = -1;

    // 타이틀 영역
    if (title) {
      const t = document.createElement('div');
      t.className = 'tp-title';
      t.textContent = title;
      toast.appendChild(t);
    } else {
      // 타이틀이 없을 때 레이아웃 유지를 위해 빈 div 추가
      const s = document.createElement('div');
      s.className = 'tp-title';
      s.style.display = 'none';
      toast.appendChild(s);
    }

    // 메시지 영역
    const msg = document.createElement('div');
    msg.className = 'tp-message';
    if (typeof message === 'string') msg.textContent = message;
    else if (message instanceof Node) msg.appendChild(message);
    toast.appendChild(msg);

    // 닫기 버튼 (confirm 제외)
    if ((opts.dismissible || sticky) && type !== 'confirm') {
      const close = document.createElement('button');
      close.className = 'tp-close';
      close.setAttribute('aria-label', '닫기');
      close.innerHTML = '\u2715';
      close.addEventListener('click', () => hideToast(toast));
      toast.appendChild(close);
    }

    // 액션 버튼 영역
    const actions = document.createElement('div');
    actions.className = 'tp-actions';

    let timer = null;
    let resolved = false;

    // 자동 닫힘 타이머 설정
    function armAutoHide() {
      if (type === 'confirm' || sticky) return; // confirm 또는 sticky는 자동 닫힘 없음
      if (opts.duration && opts.duration > 0) {
        timer = setTimeout(() => hideToast(toast), opts.duration);
      }
    }

    // 자동 닫힘 타이머 해제
    function clearAutoHide() {
      if (timer) { clearTimeout(timer); timer = null; }
    }

    toast.addEventListener('mouseenter', clearAutoHide);
    toast.addEventListener('mouseleave', armAutoHide);

    // confirm 타입: 확인/취소 버튼 및 키보드 이벤트
    if (type === 'confirm') {
      const okLabel = opts.okText || '확인';
      const cancelLabel = opts.cancelText || '취소';
      const onOk = () => {
        if (resolved) return; resolved = true;
        if (typeof opts.onConfirm === 'function') opts.onConfirm();
        hideToast(toast);
      };
      const onCancel = () => {
        if (resolved) return; resolved = true;
        if (typeof opts.onCancel === 'function') opts.onCancel();
        hideToast(toast);
      };
      actions.appendChild(createButton(cancelLabel, 'secondary', onCancel));
      actions.appendChild(createButton(okLabel, 'primary', onOk));

      toast.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') { e.stopPropagation(); onCancel(); }
        if (e.key === 'Enter') { e.stopPropagation(); onOk(); }
      });

      // confirm도 duration 지정 시 자동 취소
      if (opts.duration && opts.duration > 0) {
        timer = setTimeout(() => { if (!resolved) onCancel(); }, opts.duration);
      }
    } else if (Array.isArray(opts.actions)) {
      // 커스텀 액션 버튼
      opts.actions.slice(0, 3).forEach((a, i) => {
        const btn = createButton(a.label || `Action ${i + 1}`, a.className || 'secondary', (ev) => {
          try { a.onClick && a.onClick(ev); } finally { hideToast(toast); }
        });
        actions.appendChild(btn);
      });
    }

    if (actions.childElementCount > 0) toast.appendChild(actions);

    // 컨테이너 위치에 따라 토스트 삽입 (top: prepend, bottom: append)
    const isBottom = (container.dataset.position || '').includes('bottom');
    if (isBottom) container.appendChild(toast); else container.prepend(toast);
    clampVisible(container, opts.maxVisible);

    // 접근성: confirm 또는 focus 옵션 시 포커스 이동
    if (type === 'confirm' || opts.focus) {
      setTimeout(() => { try { toast.focus(); } catch { } }, 0);
    }

    // 자동 닫힘 시작
    armAutoHide();

    return {
      element: toast,
      close: () => hideToast(toast),
    };
  }

  // 외부에서 사용할 API 객체
  const TOASTAPI = {
    /**
     * 라이브러리의 기본 옵션을 설정하거나 확장합니다.
     * @param {Partial<ToastOptions>} next - 덮어쓸 기본 옵션 객체.
     */
    setDefaults(next) {
      state.defaults = { ...state.defaults, ...(next || {}) };
    },

    /**
     * 화면에 표시된 모든 토스트를 지웁니다.
     * 특정 위치의 토스트만 지울 수도 있습니다.
     * @param {string} [position] - 지울 토스트의 위치. 지정하지 않으면 모든 토스트를 지웁니다.
     */
    clear(position) {
      if (position) {
        const c = state.containers.get(position);
        if (c) Array.from(c.children).forEach(hideToast);
      } else {
        state.containers.forEach(c => Array.from(c.children).forEach(hideToast));
      }
    },

    /**
     * 정보(info) 타입의 토스트를 표시합니다.
     * @param {string|Node} message - 토스트에 표시할 메시지.
     * @param {ToastOptions} [options={}] - 토스트에 대한 추가 옵션.
     * @returns {{element: HTMLElement, close: function(): void}} 생성된 토스트 인스턴스.
     */
    info(message, options = {}) {
      const { title = '정보', ...rest } = options;
      return makeToast({ type: 'info', title, message, options: rest });
    },

    /**
     * 경고(alert) 타입의 토스트를 표시합니다.
     * @param {string|Node} message - 토스트에 표시할 메시지.
     * @param {ToastOptions} [options={}] - 토스트에 대한 추가 옵션.
     * @returns {{element: HTMLElement, close: function(): void}} 생성된 토스트 인스턴스.
     */
    alert(message, options = {}) {
      const { title = '알림', ...rest } = options;
      // 알림은 assertive로 aria-live 설정
      rest.ariaLive = rest.ariaLive || 'assertive';
      return makeToast({ type: 'alert', title, message, options: rest });
    },

    /**
     * 확인(confirm) 타입의 토스트를 표시합니다.
     * @param {string|Node} message - 토스트에 표시할 메시지.
     * @param {function(): void} onConfirm - '확인' 버튼을 눌렀을 때 실행될 콜백.
     * @param {function(): void} [onCancel] - '취소' 버튼을 눌렀을 때 실행될 콜백.
     * @param {ConfirmToastOptions} [options={}] - 토스트에 대한 추가 옵션.
     * @returns {{element: HTMLElement, close: function(): void}} 생성된 토스트 인스턴스.
     */
    confirm(message, onConfirm, onCancel, options = {}) {
      const { title = '확인', ...rest } = options;
      rest.onConfirm = onConfirm;
      rest.onCancel = onCancel;
      return makeToast({ type: 'confirm', title, message, options: rest });
    },
  };

  // 전역(window)에 API 노출
  if (typeof window !== 'undefined') {
    window.Toast = TOASTAPI;
  }
})();
