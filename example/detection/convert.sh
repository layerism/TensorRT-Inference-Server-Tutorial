#!/bin/bash

REPO=../../model_repository
MODEL_NAME=detection

rm -rf ${REPO}/${MODEL_NAME}*

mkdir -p ${REPO}/${MODEL_NAME}/1
cp -r config.pbtxt ${REPO}/${MODEL_NAME}

python pre_process.py
python network.py
python post_process.py
