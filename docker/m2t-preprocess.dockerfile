FROM apache/beam_python3.10_sdk:2.48.0

RUN apt-get update --fix-missing
RUN apt-get install -y libsndfile1-dev && \
    apt-get install -y ffmpeg && \
    apt-get install -y cmake

RUN python -m pip install --no-cache-dir --upgrade pip setuptools

RUN python -m pip uninstall -y numpy
RUN python -m pip install --no-cache-dir --upgrade pip setuptools
RUN python -m pip install cython==0.29.35 numpy
RUN python -m pip install git+https://github.com/CPJKU/madmom.git
RUN python -m pip install librosa google-cloud-storage==2.10.0
RUN python -m pip install ffmpeg-python
RUN python -m pip install openai
COPY requirements.txt requirements.txt
COPY dataflow-requirements.txt dataflow-requirements.txt

RUN python -m pip install -r requirements.txt && \
    python -m pip install -r dataflow-requirements.txt
RUN mkdir -p /m2t/
COPY m2t /m2t/m2t
COPY setup.py /m2t
RUN ls -la /m2t/*
WORKDIR /m2t
RUN python -m pip install -e .
