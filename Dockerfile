FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # PyTorch CUDA 12.4 빌드 인덱스 URL
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
        python3.10 \
        python3.10-dev \
        python3.10-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# venv 생성 (Ubuntu 22.04 기본 파이썬은 3.10)
RUN python3.10 -m venv /opt/venv \
    && /opt/venv/bin/python -m ensurepip

ENV PATH="/opt/venv/bin:${PATH}"

# pip로 설치한 nvidia-cudnn-cu12 / nvidia-cublas-cu12 라이브러리 경로를 미리 등록
ENV NVIDIA_PIP_LIB="/opt/venv/lib/python3.10/site-packages/nvidia"
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${NVIDIA_PIP_LIB}/cudnn/lib:${NVIDIA_PIP_LIB}/cublas/lib:${LD_LIBRARY_PATH}"

WORKDIR /app

# 먼저 requirements만 복사하여 레이어 캐시 활용
COPY requirements.txt ./

# 의존성 변경 여부에 따라 캐시 활용 가능
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install gunicorn

EXPOSE 6789

CMD ["python", "app.py"]
