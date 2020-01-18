#!/bin/bash

IMAGE="../test-data/widerface.jpg"
python client.py \
    --model-name detection \
    --model-version 1 \
    --url "10.160.168.155:7001" \
    --image ${IMAGE}
