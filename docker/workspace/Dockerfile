FROM nvidia/cuda:8.0-devel

RUN apt-get upgrade -y && apt-get update -y
RUN apt-get install -y \
    wget \
    curl \
    git \
    zsh \
    vim \
    python3-dev \
    python3-pip \
    language-pack-ja-base \
    language-pack-ja \
    ibus-mozc

RUN update-locale LANG=ja_JP.UTF-8 LANGUAGE=ja_JP:ja
ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
ENV LC_CTYPE ja_JP.UTF-8

RUN pip3 install --upgrade pip
RUN pip3 install
    setuptools \
    cupy \
    chainer \
    --no-cache-dir

RUN python3 -c "import chainer; print(chainer.cuda.available)"

CMD ["/bin/zsh"]
