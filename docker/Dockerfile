FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu18.04

ARG TENSORFLOW_VERSION=2.5.0
ARG python=3.8
ENV PYTHON_VERSION=${python}

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
	ca-certificates \
        build-essential \
        cmake \
        g++-4.8 \
        git \
        curl \
        vim \
        wget \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        openjdk-8-jdk-headless \
        openssh-client \
        openssh-server \
        libglib2.0-0


# Allow OpenSSH to talk to containers without asking for confirmation
# For pycharm, horovod
# If you want use horovod, you must check "https://github.com/horovod/horovod/blob/master/docker/horovod/Dockerfile"
# We have plan to compare the speed between parallel framework.
# Horovod, deepspeed, tensorflow, FairScale

RUN mkdir -p /var/run/sshd
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

WORKDIR "/Pharmcadd"

# tensorflow
RUN pip install tensorflow==${TENSORFLOW_VERSION} pandas matplotlib

ARG hhblits_version=v3.3.0

# git
# hhblits
# GNU General Public License v3.0
RUN git clone -b ${hhblits_version} https://github.com/soedinglab/hh-suite && \
    mkdir -p hh-suite/build && cd hh-suite/build && \
    cmake -DCMAKE_INSTALL_PREFIX=. .. && \
    make -j 4 && make install

ENV PATH="/Pharmcadd/hh-suite/build/bin:/Pharmcadd/hh-suite/build/bin/scripts:$PATH"

