FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu16.04

# http://bugs.python.org/issue19846
ENV LANG C.UTF-8
# https://github.com/docker-library/python/issues/147
ENV PYTHONIOENCODING UTF-8
ENV EDITOR vim

RUN apt-get update && apt-get install -y --no-install-recommends \
	ca-certificates \
	curl \
	netbase \
	wget \
	git \
	openssh-client \
	ssh \
	vim \
&& rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
	python \
	python-dev \
	python-pip \
	python-setuptools \
	python-tk \
	python-lxml \
	python-pil \
	protobuf-compiler \
&& rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /workspace

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# TF object detection API

RUN mkdir -p /tensorflow/models

RUN git clone https://github.com/mateuszbuda/models.git /tensorflow/models

WORKDIR /tensorflow/models/research

RUN protoc object_detection/protos/*.proto --python_out=.

ENV PYTHONPATH /tensorflow/models/research:/tensorflow/models/research/slim

