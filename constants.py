import os

# 기본 모델 설정
DEFAULT_MODEL = "gemini-flash-latest"

# 컨텍스트 유지 설정
DEFAULT_CONTEXT_KEEP_RECENT = 50

# 디렉토리 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_LOG_DIR = os.path.join(BASE_DIR, 'logs')
SNAPSHOT_ROOT_DIR = os.path.join(BASE_DIR, 'snapshots')
