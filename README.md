# 번역 서비스 웹 애플리케이션
Flask를 이용한 번역 서비스 웹 애플리케이션입니다

<img width="1607" height="1089" alt="image" src="https://github.com/user-attachments/assets/d3f0aa88-0f1e-47c2-8ee4-9b741121de04" /><br><br>
<img width="1607" height="1089" alt="image" src="https://github.com/user-attachments/assets/94f129ef-4dc7-4989-b17d-2429d3eb2cf8" />

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
- **Thinking Budget 설정**: 숫자 입력 또는 `auto`로 Gemini의 사고 과정 토큰 예산 제어
- **프리셋 관리**: 자주 사용하는 번역 설정을 저장하고 불러오기

## 사용 팁

- 업로드 폼에서 컨텍스트 압축을 켜고 토큰 한도를 지정하면 히스토리가 자동 요약됩니다.
- 번역이 끝나면 `logs/` 디렉터리에 `history_<job_id>_*.json` 파일이 생성되어 대화 내역을 확인할 수 있습니다.
- 진행 화면의 ZIP 다운로드 버튼으로 여러 파일의 번역본을 한 번에 저장할 수 있습니다.
- **Thinking Budget 설정**:
  - 숫자 입력 (예: `512`, `1024`, `2048`): 해당 토큰 수만큼 사고 과정 예산 설정
  - `auto`: Gemini가 자동으로 최적의 예산을 결정
  - 비활성화 체크박스: Thinking 기능 완전 비활성화
- 자주 사용하는 설정은 프리셋으로 저장하여 빠르게 재사용할 수 있습니다.

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

## bat 파일 사용한 설치

1. 코드를 clone 혹은 다운로드 받은 후 `install_app.bat` 파일 실행 → 가상환경 세팅 후 프로젝트 실행.

2. 설치 완료 후에는 `start_app.bat`으로 실핼.

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

### 프리셋 관리
- 자주 사용하는 번역 설정을 프리셋으로 저장
- 프리셋 선택 시 저장된 설정 자동 적용
- "저장" 버튼으로 현재 프리셋 업데이트
- "새로 만들기" 버튼으로 새 프리셋 생성

## 기여

이 프로젝트에 기여하고 싶다면, 이슈를 생성하거나 풀 리퀘스트를 제출해주세요. 감사합니다!
