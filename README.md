# 번역 서비스 웹 애플리케이션
Flask를 이용한 번역 서비스 웹 애플리케이션입니다

<img width="1675" height="1061" alt="image" src="https://github.com/user-attachments/assets/2db12764-7abe-40a3-a6c2-f0124202b4a6" /><br><br>
<img width="1675" height="1061" alt="image" src="https://github.com/user-attachments/assets/941c2995-15d7-492f-9dac-053be9546fa9" />
업로드 화면<br><br>
  
<img width="1369" height="1247" alt="image" src="https://github.com/user-attachments/assets/baebd883-fc0d-4e15-97b8-df84d79d2e68" />
진행화면

## 기능

- 텍스트 번역 (여러 언어 지원)
- 다중 SRT 파일 번역 및 다운로드
- 사용자 친화적인 인터페이스
- Google API 키 및 모델명 설정 가능
- 실시간 번역 진행 상황 표시
- 컨텍스트 압축/토큰 제한을 통한 장문 번역 지원
- 번역 완료 후 대화 히스토리를 JSON 로그로 자동 보관

## 사용 팁

- 업로드 폼에서 컨텍스트 압축을 켜고 토큰 한도를 지정하면 히스토리가 자동 요약됩니다.
- 번역이 끝나면 `logs/` 디렉터리에 `history_<job_id>_*.json` 파일이 생성되어 대화 내역을 확인할 수 있습니다.
- 진행 화면의 ZIP 다운로드 버튼으로 여러 파일의 번역본을 한 번에 저장할 수 있습니다.

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

## 개발 환경

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
│   └── srt_module.py
├── static/               # 정적 파일 (CSS, JS)
│   ├── css/
│   └── js/
└── templates/            # HTML 템플릿
```

## 사용 방법

1. Google API 키를 입력하여 .env 파일에 저장.
2. 번역할 SRT 파일을 업로드.
3. 번역 모델을 선택하거나 기본값 사용.
4. 번역 진행 상황을 확인하며 결과 다운로드.

## 기여

이 프로젝트에 기여하고 싶다면, 이슈를 생성하거나 풀 리퀘스트를 제출해주세요. 감사합니다!
