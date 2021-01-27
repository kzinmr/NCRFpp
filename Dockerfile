FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

WORKDIR /app

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV APP_ROOT /app
ENV WORK_DIR /app/workspace

ENV DEBIAN_FRONTEND noninteractive

RUN mkdir -p $APP_ROOT
WORKDIR $APP_ROOT

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

RUN apt-get update -y && apt-get install -y \
    git \
    wget \
    curl \
    cmake \
    unzip \
    gcc \
    g++ \
    mecab \
    libmecab-dev
RUN ln -s /etc/mecabrc /usr/local/etc/mecabrc

COPY model /app/model
COPY utils /app/utils
COPY ./main.py /app/
COPY ./train.config /app/
COPY ./decode.config /app/

CMD ["python3", "/app/main.py", "--config=/app/train.config"]