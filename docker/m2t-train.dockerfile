FROM nvcr.io/nvidia/pytorch:22.11-py3
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing
RUN apt-get install -y libsndfile1-dev && \
    apt-get install -y ffmpeg 

RUN python -m pip install --no-cache-dir --upgrade pip setuptools

RUN python -m pip install --no-cache-dir --upgrade pip setuptools
COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt
COPY train-requirements.txt train-requirements.txt
RUN python -m pip install -r train-requirements.txt

RUN python -m pip install multiprocess==0.70.15 --no-deps
RUN python -m pip uninstall -y transformer-engine
RUN python -m pip install protobuf==3.20.2

RUN mkdir -p /m2t/
COPY m2t /m2t/m2t
COPY setup.py /m2t
RUN ls -la /m2t/*
WORKDIR /m2t
RUN python -m pip install -e .
RUN patch -u /usr/local/lib/python3.8/dist-packages/transformers/modeling_utils.py /m2t/m2t/modeling_utils.patch
ENV PYTHONPATH "${PYTHONPATH}:/m2t"
