FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

RUN apt-get update && apt-get install -y \
    build-essential \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libjasper-dev \
    libavformat-dev \
    libpq-dev \
    libturbojpeg \
    software-properties-common \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

CMD mkdir -p /workspace
WORKDIR /workspace
RUN git clone https://github.com/kuaikuaikim/DFace.git
# RUN git clone https://github.com/ngxbac/mtcnn-pytorch.git
CMD echo 'export PYTHONPATH=$PYTHONPATH:DFace' >> ~/.bashrc
CMD echo 'export PYTHONPATH=$PYTHONPATH:mtcnn-pytorch' >> ~/.bashrc

RUN pip install catalyst==19.06.5
RUN pip install albumentations
RUN pip install cnn_finetune
RUN pip install timm
RUN pip install click
RUN pip install pandas

