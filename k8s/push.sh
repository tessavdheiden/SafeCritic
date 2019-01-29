#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "./push.sh [IMAGE_NAME]"
    exit
fi

IMAGE_NAME=$1

. image.config

docker tag $IMAGE_URL $IMAGE_URL

docker push $IMAGE_URL
