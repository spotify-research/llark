# same env as shama-jukebox-noconda.dockerfile, but with different entrypoint
FROM apache/beam_python3.7_sdk:2.24.0
ENV INSTALLER_DIR="/tmp/installer_dir"

# The base image has TensorFlow 2.2.0, which requires CUDA 10.1 and cuDNN 7.6.
# You can download cuDNN from NVIDIA website
# https://developer.nvidia.com/cudnn
COPY cudnn-10.1-linux-x64-v8.0.5.39.tar $INSTALLER_DIR/cudnn.tar
RUN \
    # Download CUDA toolkit.
    wget -q -O $INSTALLER_DIR/cuda.run https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run && \
    # Install CUDA toolkit. Print logs upon failure.
    sh $INSTALLER_DIR/cuda.run --toolkit --silent || (egrep '^\[ERROR\]' /var/log/cuda-installer.log && exit 1) && \
    # Install cuDNN.
    mkdir $INSTALLER_DIR/cudnn && \
    tar xvf $INSTALLER_DIR/cudnn.tar -C $INSTALLER_DIR/cudnn && \
    cp $INSTALLER_DIR/cudnn/cuda/include/cudnn*.h /usr/local/cuda/include && \
    cp $INSTALLER_DIR/cudnn/cuda/lib64/libcudnn* /usr/local/cuda/lib64 && \
    chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn* && \
    rm -rf $INSTALLER_DIR

# A volume with GPU drivers will be mounted at runtime at /usr/local/nvidia.
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

# Configure shell
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]
RUN apt-get update --fix-missing

# install some linux deps and copy the models
RUN apt-get install -y wget unzip
RUN mkdir -p /root/.cache/jukebox/models/5b
RUN wget https://openaipublic.azureedge.net/jukebox/models/5b/vqvae.pth.tar; mv vqvae.pth.tar /root/.cache/jukebox/models/5b
RUN wget https://openaipublic.azureedge.net/jukebox/models/5b/prior_level_2.pth.tar; mv prior_level_2.pth.tar /root/.cache/jukebox/models/5b


RUN apt-get install -y libsndfile1-dev && \
    apt-get install -y libopenmpi-dev && \
    apt-get install -y openssh-server

RUN python -m pip install --no-cache-dir --upgrade pip setuptools

RUN python -m pip install --no-cache-dir torch==1.4.0
RUN python -m pip install mpi4py==3.0.3

# Setup entrypoint
RUN mkdir /input
RUN mkdir /output
RUN mkdir /code
WORKDIR /code
ARG COMMIT_ID=08efbbc1d4ed1a3cef96e08a931944c8b4d63bb3
RUN wget https://github.com/openai/jukebox/archive/${COMMIT_ID}.zip; unzip ${COMMIT_ID}.zip; rm ${COMMIT_ID}.zip; mv jukebox-${COMMIT_ID} jukebox

COPY jukebox/make_models.py.patch make_models.py.patch
RUN apt-get install -y patch
RUN patch jukebox/make_models.py make_models.py.patch

RUN python -m pip install --no-cache-dir -e jukebox
RUN python -m pip install apache_beam[gcp,dataframe]==2.48 
RUN python -m pip install pandas==1.3.5
RUN python -m pip install google-cloud-storage==2.10.0

# Demonstrate the environment is activated:
RUN echo "Make sure jukebox is installed:"
RUN python -c "import jukebox"

# The code to run when container is started:
COPY jukebox/main.py main.py
COPY jukebox/dataflow_inference.py dataflow_inference.py

# Set the entrypoint to Apache Beam SDK launcher.
ENTRYPOINT ["/opt/apache/beam/boot"]