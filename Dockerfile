# you will also find guides on how best to write your Dockerfile

FROM docker.io/huggingface/downloader:0.17.3@sha256:b5bc7fb168c85737634c00c4099b19e251eb2e55b3523b4e389d98876b35f109

FROM docker.io/library/python:3.10@sha256:bd4097b62c6752da37c3f1ca2e330b5e76d8deb4848cf1795a2473bf7910a91e

RUN --mount=target=pre-requirements.txt,source=pre-requirements.txt 	pip install --no-cache-dir -r pre-requirements.txt

RUN pip install --no-cache-dir 	gradio[oauth]==4.16.0 	"uvicorn>=0.14.0" 	spaces==0.22.0

RUN --mount=target=requirements.txt,source=requirements.txt 	pip install --no-cache-dir -r requirements.txt

WORKDIR /home/user/app

RUN pip install --no-cache-dir pip==22.3.1 && 	pip install --no-cache-dir 	datasets 	"huggingface-hub>=0.19" "hf-transfer>=0.1.4" "protobuf<4" "click<8.1" "pydantic~=1.0"

RUN useradd -m -u 1000 user

RUN --mount=target=/root/packages.txt,source=packages.txt 	apt-get update && 	xargs -r -a /root/packages.txt apt-get install -y 	&& rm -rf /var/lib/apt/lists/*

COPY --link --chown=1000 --from=lfs /app /home/user/app

RUN apt-get update && apt-get install -y 	git 	git-lfs 	ffmpeg 	libsm6 	libxext6 	cmake 	libgl1-mesa-glx 	&& rm -rf /var/lib/apt/lists/* 	&& git lfs install

RUN pip freeze > /tmp/freeze.txt

COPY --link --chown=1000 ./ /home/user/app

COPY --from=pipfreeze --link --chown=1000 /tmp/freeze.txt .
