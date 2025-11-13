# 자막 번역기 (SRT Translator)
Flask 기반 AI 자막 번역 & 생성 웹 애플리케이션

<img width="1430" height="1279" alt="image" src="https://github.com/user-attachments/assets/eccaa152-1bd5-42e0-b023-674104111e86" />

업로드 화면<br><br>

<img width="1369" height="1247" alt="image" src="https://github.com/user-attachments/assets/baebd883-fc0d-4e15-97b8-df84d79d2e68" />
진행화면

## 주요 기능

### 1. 자막 번역
- 다중 SRT 파일 번역 및 다운로드
- 실시간 번역 진행 상황 표시
- 컨텍스트 압축/토큰 제한을 통한 장문 번역 지원
- 번역 완료 후 대화 히스토리를 JSON 로그로 자동 보관
- **YouTube 링크 맥락 분석**: 번역 청크마다 최소 1장의 스냅샷(10개 엔트리당 1장 추가)을 추출해 모델이 장면을 이해하도록 보조
- **Thinking Budget 설정**: 숫자 입력 또는 `auto`로 Gemini의 사고 과정 토큰 예산 제어
- **프리셋 관리**: 자주 사용하는 번역 설정을 저장하고 불러오기

### 2. 자막 생성
- YouTube 링크 또는 비디오 파일에서 자막 자동 생성
- Whisper + Gemini 듀얼 전사 시스템으로 고품질 자막 생성
- Silero VAD 기반 정밀 음성 구간 검출
- 원본 언어 전사 또는 타겟 언어 번역 선택 가능

### 3. 자막 보정 싱크 ⭐ NEW
- **Silero VAD 기반 자막 타이밍 자동 보정**
- 음성 구간에 맞춰 자막 시간 정확도 향상
- 청크 단위 경계 스냅으로 안정적인 보정
- 고급 설정으로 세밀한 파라미터 조정 가능
- 실시간 보정 통계 제공

## 자막 생성 파이프라인

이 프로젝트는 긴 영상을 다음과 같은 순서로 처리해 자막을 만듭니다.

### 기본 파이프라인 (영상 파일만 업로드)

1. **영상 분할**: 영상을 일정 길이로 나누고 Silero VAD로 각 조각 안의 발화 구간을 표시합니다.
2. **Whisper 보조 전사**: 분할된 조각을 Whisper에 통과시켜 "시작·종료 시각 + 대략적인 문장"을 얻습니다. 이 정보는 다음 단계의 힌트로만 사용합니다.
3. **LLM 정밀 전사/번역**: 조각 영상과 Whisper가 준 힌트를 함께 Gemini에 전달해, 실제 음성을 들은 뒤 더 정확한 문장(또는 번역)을 JSON으로 돌려받습니다.
4. **SRT 저장**: LLM이 돌려준 문장을 전체 타임라인 순으로 정렬해 표준 SRT 파일로 저장합니다.

덕분에 Whisper가 빠르게 타임스탬프를 제시하고, LLM이 의미와 문장을 다듬으면서 자막 품질을 확보할 수 있습니다.

### SRT 파일 기반 파이프라인 (영상 + SRT 파일 함께 업로드)

영상과 함께 SRT 자막 파일을 업로드하면 **Silero VAD와 Whisper 처리를 건너뛰고** 다음과 같이 처리합니다:

1. **SRT 파싱**: 업로드된 SRT 파일의 타임스탬프를 읽어 세그먼트 구간을 결정합니다.
2. **영상 분할**: SRT 타임스탬프를 기준으로 영상을 청크 단위로 나눕니다.
3. **LLM 정밀 전사/번역**: 분할된 영상을 Gemini에 전달하여 정밀한 전사 또는 번역을 수행합니다.
4. **SRT 저장**: LLM 결과를 정렬하여 최종 SRT 파일로 저장합니다.

이 방식은 이미 타임스탬프가 있는 자막을 개선하거나 번역할 때 유용하며, VAD/Whisper 단계를 생략하여 처리 시간을 단축할 수 있습니다.

## 사용 팁

- 업로드 폼에서 컨텍스트 압축을 켜고 토큰 한도를 지정하면 히스토리가 자동 요약됩니다.
- 번역이 끝나면 `logs/` 디렉터리에 `history_<job_id>_*.json` 파일이 생성되어 대화 내역을 확인할 수 있습니다.
- 진행 화면의 ZIP 다운로드 버튼으로 여러 파일의 번역본을 한 번에 저장할 수 있습니다.
- YouTube 링크를 입력하면 각 번역 청크마다 스냅샷을 첨부한 채 바로 번역을 요청하여 모델이 장면 정보를 참고합니다(엔트리 10개마다 스냅샷 1장 추가; 컨텍스트 압축이 실행되면 오래된 이미지 참조는 자동 정리).
- 생성된 스냅샷 파일은 `snapshots/` 폴더 아래에 작업/청크별로 저장되니, 번역 후에도 그대로 열람할 수 있습니다.
- **SRT 파일 함께 업로드**: 자막 생성 페이지에서 영상과 함께 SRT 파일을 업로드하면 VAD/Whisper 단계를 건너뛰고 SRT의 타임스탬프를 기준으로 세그먼트를 나누어 LLM 전사/번역만 수행합니다. 기존 자막을 개선하거나 번역할 때 유용합니다.

- **Thinking Budget 설정**:
  - 숫자 입력 (예: `512`, `1024`, `2048`): 해당 토큰 수만큼 사고 과정 예산 설정
  - `auto`: Gemini가 자동으로 최적의 예산을 결정
  - 비활성화 체크박스: Thinking 기능 완전 비활성화
- 자주 사용하는 설정은 프리셋으로 저장하여 빠르게 재사용할 수 있습니다.

## 설치 방법

### 원클릭 설치&실행

1. 프로젝트를 clone 혹은 코드를 다운로드 받은 후 `install_app.bat` 파일 실행 → 가상환경 세팅 후 프로젝트 실행.

2. 설치 완료 후에는 `start_app.bat`으로 실행.

### 수동 설치 가이드

1. 가상환경 생성:
  ```bash
  python -m venv .venv
  ```

2. 가상환경 활성화:
  ```bash
  .venv\Scripts\activate
  ```

3. 의존성 설치:
  ```bash
  pip install -r requirements.txt
  ```

4. 애플리케이션 실행:
  ```bash
  python app.py
  ```

5. 웹 브라우저에서 애플리케이션 접속:
  ```
  http://127.0.0.1:6789
  ```

### Docker 배포

Docker를 사용하여 CUDA GPU를 지원하는 환경에서 배포할 수 있습니다.

#### 사전 요구사항

- Docker 및 Docker Compose 설치
- NVIDIA GPU 및 NVIDIA Container Toolkit 설치
- 호스트 시스템에 CUDA 드라이버 설치

#### 배포 방법

1. **이미지 빌드 및 컨테이너 실행**:
   ```bash
   docker compose up --build -d
   ```
   
   또는 Docker Compose v1 사용 시:
   ```bash
   docker-compose up --build -d
   ```

2. **애플리케이션 접속**:
   ```
   http://localhost:6789
   ```

3. **컨테이너 로그 확인**:
   ```bash
   docker compose logs -f
   ```

4. **컨테이너 중지**:
   ```bash
   docker compose down
   ```

#### 주의사항

- Windows에서 Docker Desktop을 사용하는 경우 WSL2 백엔드 사용을 권장합니다.
- GPU가 제대로 인식되지 않으면 `nvidia-smi` 명령으로 호스트 GPU 상태를 확인하세요.
- 컨테이너 내부에서 생성된 파일(`generated_subtitles/`, `snapshots/` 등)은 볼륨 마운트를 통해 호스트에서도 접근 가능합니다.

## 기술 스택

### 백엔드
- **Flask 3.0.2**: 웹 애플리케이션 프레임워크
- **Google Generative AI SDK**: Gemini API 클라이언트
- **OpenAI Whisper**: 음성 전사
- **Silero VAD**: 음성 구간 검출
- **FFmpeg**: 비디오/오디오 처리
- **PyTorch**: 딥러닝 모델 실행

### 프론트엔드
- **Tailwind CSS**: UI 스타일링
- **jQuery**: DOM 조작 및 AJAX
- **Material Icons**: 아이콘

### 데이터베이스
- **SQLite**: 설정, 프리셋, 작업 관리

## 주요 파일 구조

```
project/
├── app.py                          # Flask 애플리케이션 엔트리 포인트
├── requirements.txt                # Python 의존성 목록
├── Dockerfile                       # Docker 이미지 빌드 설정
├── docker-compose.yml              # Docker Compose 설정 (GPU 지원)
├── .dockerignore                   # Docker 빌드 시 제외할 파일 목록
├── module/                         # 주요 기능 모듈
│   ├── database_module.py          # 데이터베이스 관리
│   ├── ffmpeg_module.py            # 비디오/오디오 처리
│   ├── gemini_module.py            # Google Gemini API 클라이언트
│   ├── srt_module.py               # SRT 파일 파싱
│   ├── Whisper_util.py             # Whisper 전사
│   ├── silero_vad.py               # 음성 구간 검출
│   ├── subtitle_generation.py      # 자막 생성 파이프라인
│   ├── subtitle_sync.py            # 자막 보정 싱크 로직
│   └── video_split.py              # 비디오 분할
├── static/                         # 정적 파일 (CSS, JS)
│   ├── css/
│   └── js/
│       ├── script.js               # 자막 번역 페이지
│       ├── subtitle_generate.js    # 자막 생성 페이지
│       └── subtitle_sync.js        # 자막 보정 싱크 페이지
└── templates/                      # HTML 템플릿
    ├── index.html                  # 자막 번역 페이지
    ├── subtitle_generate.html      # 자막 생성 페이지
    ├── subtitle_sync.html          # 자막 보정 싱크 페이지
    └── subtitle_job.html           # 자막 생성 작업 모니터링
```

## 사용 방법

### 자막 번역 (`/`)

1. **API 키 설정**: 우측 상단 "추가 설정" 버튼을 클릭하여 Google API 키를 입력하고 저장
2. **번역 설정**:
   - 타겟 언어: 번역할 언어 지정 (예: 한국어, English)
   - 청크 크기: 한 번에 번역할 자막 수
   - Thinking Budget: `auto`, `512`, `1024`, `2048` 등 선택 또는 직접 입력
   - 커스텀 프롬프트: 번역 스타일 또는 추가 지시사항 입력
3. **SRT 파일 업로드**: 드래그 앤 드롭 또는 파일 선택
4. **번역 시작**: "번역 시작하기" 버튼 클릭
5. **진행 상황 확인**: 실시간으로 번역 진행 상황 모니터링
6. **결과 다운로드**: 개별 파일 다운로드 또는 ZIP으로 일괄 다운로드

### 자막 생성 (`/subtitle_generate`)

1. **영상 소스 선택**: YouTube 링크 입력 또는 비디오 파일 업로드
2. **전사 모드 선택**:
   - 원본 언어로만 전사
   - 특정 언어로 번역
3. **청크 설정**: 영상 분할 길이 지정 (1-5분)
4. **모델 선택**: Gemini 모델명 입력 (기본: gemini-flash-latest)
5. **자막 생성 요청**: 버튼 클릭 후 작업 진행 페이지로 이동
6. **완료 후 다운로드**: SRT 파일 다운로드

### 자막 보정 싱크 (`/subtitle_sync`)

1. **파일 업로드**:
   - SRT 자막 파일 선택
   - 오디오/비디오 파일 선택
2. **고급 설정 (선택)**:
   - **VAD 설정**: 임계값, 최소 음성/무음 길이, 패딩
   - **경계 보정 설정**: 청크 간격, 탐색 범위, 경계 패딩
3. **자막 보정 시작**: 버튼 클릭
4. **통계 확인**: 총 엔트리, 청크 수, VAD 세그먼트, 보정된 청크
5. **결과 다운로드**: 보정된 SRT 파일 다운로드 (`_synced.srt`)

## 자막 보정 싱크 상세 설명

### 작동 원리

1. **청크 묶기**: 엔트리 간 간격이 `gap_threshold_ms` 이하인 자막들을 하나의 청크로 묶습니다
2. **VAD 세그먼트 추출**: Silero VAD로 오디오에서 실제 음성 구간을 검출합니다
3. **경계 스냅**: 각 청크의 시작/끝을 가장 가까운 VAD 세그먼트에 맞춥니다
4. **안전 제약**: 청크 간 겹침 방지, 최소 길이 보장, 이동량 제한
5. **SRT 출력**: 보정된 타이밍으로 새 SRT 파일 생성

### 주요 파라미터

**VAD 설정**
- `threshold` (0.0-1.0): 음성 감지 민감도 (기본: 0.55)
- `min_speech_duration_ms`: 최소 음성 길이 (기본: 200ms)
- `min_silence_duration_ms`: 최소 무음 길이 (기본: 250ms)
- `speech_pad_ms`: 음성 앞뒤 여유 시간 (기본: 80ms)

**경계 보정 설정**
- `gap_threshold_ms`: 청크 묶기 간격 임계값 (기본: 200ms)
- `lookback_start_ms`: 시작 지점 뒤로 탐색 범위 (기본: 800ms)
- `lookahead_start_ms`: 시작 지점 앞으로 탐색 범위 (기본: 400ms)
- `pad_ms`: 보정 경계 패딩 (기본: 80ms)

### 사용 시나리오

- 자동 생성된 자막의 타이밍 정확도 향상
- 수동 작성 자막과 실제 음성 동기화
- 번역된 자막의 타이밍 재조정
- 오디오 편집 후 자막 싱크 복구
