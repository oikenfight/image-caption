# FROM nvidia/cuda:9.2-cudnn7-runtime
FROM nvidia/cuda:8.0-devel

MAINTAINER Yuta Oikawa

RUN apt-get upgrade -y && apt-get update -y
RUN apt-get install -y \
    wget \
    curl \
    git \
    zsh \
    vim \
    tree \
    gcc \
    openssl \
    zlib1g-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    language-pack-ja-base \
    language-pack-ja \
    ibus-mozc \
    build-essential \
    python3-dev \
    python3-pip \
    zip \
    unzip

RUN update-locale LANG=ja_JP.UTF-8 LANGUAGE=ja_JP:ja
ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
ENV LC_CTYPE ja_JP.UTF-8

# デフォルトだと /usr/local/cuda にパスが設定されているようなので、明示的に/usr/local/cuda-8.0 にパスを通す必要があるそう
ENV CUDA_PATH=/usr/local/cuda-8.0
ENV PATH=/usr/local/cuda-8.0/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

# cupy は cuda のバージョンに合ったものを明示的に指定しないと認識出来ないため、cupy-cuda80 と指定してインストール
RUN pip3 install --upgrade pip
RUN pip3 install \
    setuptools \
    cupy-cuda80 \
    chainer \
    scipy \
    pillow \
    h5py \
    --no-cache-dir

RUN mkdir workspace

CMD ["/bin/bash"]
