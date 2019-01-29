FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

MAINTAINER Naveen Shankar Nagaraja "naveen-shankar.nagaraja@bmw.de"

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV http_proxy=http://QNUM:TSS_PASS@proxy.muc:8080
ENV https_proxy=http://QNUM:TSS_PASS@proxy.muc:8080
ARG user_id

VOLUME /workspace

COPY keyboard /etc/default/keyboard

RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    vim git cmake\
    python3 python3-pip python3-dev libtbb2 libtbb-dev \
    libjpeg-dev libjasper-dev libdc1394-22-dev \
    python-opencv libopencv-dev libav-tools python-pycurl \
    libatlas-base-dev gfortran webp qt5-default libvtk6-dev zlib1g-dev

    
ADD requirements.txt /opt/pip/requirements.txt
RUN pip3 --no-cache-dir install --upgrade pip
RUN pip3 --no-cache-dir install --upgrade \
         --requirement=/opt/pip/requirements.txt


RUN cd workspace &&\
    git clone https://github.com/Itseez/opencv.git &&\
    git clone https://github.com/Itseez/opencv_contrib.git &&\
    cd opencv && mkdir build && cd build && cmake  -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=ON -DWITH_TBB=ON -DWITH_GDAL=ON -DWITH_XINE=ON -DBUILD_EXAMPLES=ON .. && \
    make -j4 && make install && ldconfig
