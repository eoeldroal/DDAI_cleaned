# =============================================================================
# DDAI - GSPO Training Docker Image
# =============================================================================
#
# CUDA 12.4 기반 이미지 + Miniconda + LNS 환경
# Qwen2.5-VL + Gemini Flash VLM Judge 지원
#
# 빌드:
#   docker build -t ddai-gspo:latest .
#
# 실행:
#   docker run -it --gpus all --shm-size=16g \
#       -v /path/to/DDAI_cleaned:/workspace \
#       -e GEMINI_API_KEY=$GEMINI_API_KEY \
#       -e WANDB_API_KEY=$WANDB_API_KEY \
#       -w /workspace \
#       ddai-gspo:latest bash
#
# =============================================================================

# CUDA 12.4 devel 이미지 (nvcc, CUDA 헤더 포함 - flash-attn 빌드에 필요)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 기본 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    git \
    git-lfs \
    vim \
    htop \
    tmux \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Miniconda 설치
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH

# Conda 설정 및 ToS 동의
RUN conda config --set channel_priority flexible \
    && conda init bash \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# environment.yml 복사 및 환경 생성
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml \
    && conda clean -a -y

# flash-attn 설치 (torch 설치 후 별도 빌드 필요)
RUN $CONDA_DIR/envs/LNS/bin/pip install --no-cache-dir flash-attn==2.6.3 --no-build-isolation

# google-generativeai 설치 (Gemini SDK - environment.yml에 누락됨)
RUN $CONDA_DIR/envs/LNS/bin/pip install --no-cache-dir google-generativeai

# LNS 환경 자동 활성화 설정
RUN echo "conda activate LNS" >> ~/.bashrc

# 작업 디렉토리 설정
WORKDIR /workspace

# 기본 쉘을 bash로 설정 (conda 활성화 지원)
SHELL ["/bin/bash", "-c"]

# 컨테이너 시작 시 LNS 환경 활성화
ENTRYPOINT ["/bin/bash", "-c", "source ~/.bashrc && exec bash"]
CMD ["bash"]
