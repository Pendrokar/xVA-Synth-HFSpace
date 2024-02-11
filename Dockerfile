FROM python:3.9

RUN --mount=target=pre-requirements.txt,source=pre-requirements.txt 	pip install --no-cache-dir -r pre-requirements.txt

RUN pip install --no-cache-dir pip==22.3.1 && 	pip install --no-cache-dir 	datasets 	"huggingface-hub>=0.19" "hf-transfer>=0.1.4" "protobuf<4" "click<8.1" "pydantic~=1.0"

RUN pip install --no-cache-dir 	gradio[oauth]==4.18.0 	"uvicorn>=0.14.0" 	spaces

RUN useradd -m -u 1000 user

RUN apt-get update && apt-get install -y 	git 	git-lfs 	ffmpeg 	libsm6 	libxext6 	cmake 	libgl1-mesa-glx 	&& rm -rf /var/lib/apt/lists/* 	&& git lfs install

RUN --mount=target=requirements.txt,source=requirements.txt 	pip install --no-cache-dir -r requirements.txt

COPY --link --chown=1000 --from=lfs /app /home/user/app

WORKDIR /home/user/app

RUN --mount=target=/root/packages.txt,source=packages.txt 	apt-get update && 	xargs -r -a /root/packages.txt apt-get install -y 	&& rm -rf /var/lib/apt/lists/*

RUN pip freeze > /tmp/freeze.txt

COPY --link --chown=1000 ./ /home/user/app

COPY --from=pipfreeze --link --chown=1000 /tmp/freeze.txt .

CMD ["python", "app.py"]
