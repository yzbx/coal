###docker build -t youdaoyzbx/pytorch:qd .
#FROM youdaoyzbx/pytorch:1.1
#COPY conda.yml .
#RUN conda env update -f conda.yml -n base

###docker build -t youdaoyzbx/pytorch:qd_light .

FROM nvidia/cuda:9.0-runtime-ubuntu16.04
ENV PATH /opt/conda/bin:$PATH
COPY conda.yml .
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    conda env update -f conda.yml -n base