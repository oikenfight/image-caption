FROM nvidia/cuda:latest

RUN apt-get update
RUN apt-get install -y wget curl git zsh vim
RUN apt-get -y install language-pack-ja-base language-pack-ja ibus-mozc

RUN update-locale LANG=ja_JP.UTF-8 LANGUAGE=ja_JP:ja
ENV LANG ja_JP.UTF-8
ENV LC_ALL ja_JP.UTF-8
ENV LC_CTYPE ja_JP.UTF-8

RUN wget https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh
RUN bash Anaconda3-4.4.0-Linux-x86_64.sh -b
ENV PATH /root/anaconda3/bin:$PATH
RUN echo $PATH

CMD ["/bin/zsh"]

RUN pip install chainer

