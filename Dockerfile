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

RUN rm -fr NCRFpp
RUN git clone https://github.com/kzinmr/NCRFpp.git && cd NCRFpp
RUN mkdir -p /app/NCRFpp/workspace/data
RUN mkdir -p /app/NCRFpp/workspace/models
COPY workspace/data /app/NCRFpp/workspace/data
COPY workspace/models /app/NCRFpp/workspace/models
COPY ./train.config /app/NCRFpp
COPY ./decode.config /app/NCRFpp

CMD ["python3", "/app/NCRFpp/main.py", "--config=/app/NCRFpp/train.config"]