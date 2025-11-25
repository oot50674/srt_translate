FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124

# 국내 미러로 변경하여 apt 속도 향상
RUN sed -i 's|archive.ubuntu.com|mirror.kakao.com|g' /etc/apt/sources.list

# apt 업데이트는 한 번만, 불필요한 upgrade 제거
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        git \
        ffmpeg \
        libsndfile1 \
        portaudio19-dev \
        build-essential \
        ca-certificates \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# venv 생성
RUN python3.11 -m venv /opt/venv \
    && /opt/venv/bin/python -m ensurepip

ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

# 먼저 requirements만 복사하여 레이어 캐시 활용
COPY requirements.txt ./

# 의존성 변경 여부에 따라 캐시 활용 가능
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install gunicorn

EXPOSE 6789

CMD ["python", "app.py"]
