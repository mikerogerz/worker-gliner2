ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS python-base

WORKDIR /app

ARG MODEL_NAME=""
ARG BASE_PATH="/runpod-volume"

ENV MODEL_NAME=$MODEL_NAME \
	BASE_PATH=$BASE_PATH

ENV PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	WORKERS=1 \
	THREADS=8 \
	PYTORCH_ALLOC_CONF=expandable_segments:True

ENV HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
	HF_HOME="${BASE_PATH}/.cache/huggingface" \
	HF_HUB_ENABLE_HF_TRANSFER=0 \
	HF_HUB_OFFLINE=1 \
	TRANSFORMERS_OFFLINE=1

# Update the base OS
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
	--mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
	set -eux; \
	apt-get update; \
	apt-get upgrade -y; \
	apt install --no-install-recommends -y  \
		git; \
	apt-get autoremove -y

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
	python3 -m pip install --upgrade pip && \
	python3 -m pip install --upgrade -r requirements.txt

COPY src .

CMD ["python3", "-u", "handler.py"]