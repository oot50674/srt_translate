# 번역 서비스 웹 애플리케이션

Flask를 이용한 간단한 번역 서비스 웹 애플리케이션입니다.

## 기능

- 텍스트 번역 (여러 언어 지원)
- SRT 파일 번역 및 다운로드
- 사용자 친화적인 인터페이스
- Google API 키 및 모델명 설정 가능
- 실시간 번역 진행 상황 표시

## 설치 방법

1. 가상환경 활성화:
```bash
.venv\Scripts\activate
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

3. 애플리케이션 실행:
```bash
python app.py
```

4. 웹 브라우저에서 애플리케이션 접속:
```
http://127.0.0.1:5000
```

## 실행 명령어

1. 애플리케이션 실행:
```bash
python app.py
```

2. 실행 후 브라우저에서 접속:
```
http://127.0.0.1:5000
```

## 개발 환경

- Python 3.13.2
- Flask 3.0.2
- Google Generative AI SDK

## 주요 파일 구조

```
project/
├── app.py                # Flask 애플리케이션 엔트리 포인트
├── requirements.txt      # Python 의존성 목록
├── module/               # 주요 기능 모듈
│   ├── database_module.py
│   ├── gemini_module.py
│   ├── ollama_module.py
│   └── srt_module.py
├── static/               # 정적 파일 (CSS, JS)
│   ├── css/
│   └── js/
└── templates/            # HTML 템플릿
```

## 사용 방법

1. Google API 키를 입력하여 번역 기능 활성화.
2. 번역할 SRT 파일을 업로드.
3. 번역 모델을 선택하거나 기본값 사용.
4. 번역 진행 상황을 확인하며 결과 다운로드.

## 기여

이 프로젝트에 기여하고 싶다면, 이슈를 생성하거나 풀 리퀘스트를 제출해주세요. 감사합니다!

## 가상환경 설정 및 활성화

1. 가상환경 생성:
```bash
python -m venv .venv
```

2. 가상환경 활성화:
- Windows:
```bash
.venv\Scripts\activate
```
- macOS/Linux:
```bash
source .venv/bin/activate
```

3. 가상환경 비활성화:
```bash
deactivate
```
