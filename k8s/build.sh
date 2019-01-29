#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "./build.sh [IMAGE_NAME]"
    exit
fi

IMAGE_NAME=$1

. image.config

USER_ID=`id -u`

# build docker container
nvidia-docker build \
    --build-arg http_proxy=$http_proxy \
    --build-arg https_proxy=$https_proxy \
    --build-arg no_proxy=$no_proxy \
    --build-arg user_id=$USER_ID \
    -t $IMAGE_URL $IMAGE_NAME
