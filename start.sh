#!/bin/bash

HTTP_PORT=7000
GRPC_PORT=7001
METRIC_PORT=7002
DOCKER_IMAGE=nvcr.io/nvidia/tensorrtserver:19.12-py3

docker run --rm \
    --runtime nvidia \
    --name trt_server \
    --shm-size=4g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -p${HTTP_PORT}:8000 \
    -p${GRPC_PORT}:8001 \
    -p${METRIC_PORT}:8002 \
    -v`pwd`/model_repository/:/models \
    ${DOCKER_IMAGE} \
    trtserver --model-repository=/models
