#!/bin/bash

IMAGE="../test-data/widerface.jpg"
python client.py \
    --model-name detection \
    --model-version 1 \
    --url "x.x.x.x:7001" \
    --image ${IMAGE}
