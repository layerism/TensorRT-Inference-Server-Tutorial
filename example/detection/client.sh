#!/bin/bash

IMAGE="../test-data/widerface.jpg"
python client.py \
    --model-name face-det \
    --model-version 1 \
    --protocol 'gRPC' \
    --url "10.160.168.155:7001" \
    --image ${IMAGE}
