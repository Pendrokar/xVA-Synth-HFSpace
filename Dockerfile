# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Try and run pip command after setting the user with `USER user` to avoid permission issues with Python
RUN pip install --no-cache-dir --upgrade pip

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

FROM docker.io/huggingface/downloader:0.17.3@sha256:b5bc7fb168c85737634c00c4099b19e251eb2e55b3523b4e389d98876b35f109

FROM docker.io/library/python:3.9@sha256:5e11e0165c7e02fcb4a15772bd25e28266ed9c4e90fded5e8cc7a921affd7826

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
