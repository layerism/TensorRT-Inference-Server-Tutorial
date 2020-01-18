#!/bin/bash

MODEL_NAME=detection

rm -rf ../../model_repository/${MODEL_NAME}*
mkdir -p ./model_repository/${MODEL_NAME}
rm -rf ./model_repository/${MODEL_NAME}*

mkdir -p ./model_repository/${MODEL_NAME}/1
cp -r config.pbtxt ./model_repository/${MODEL_NAME}

python pre_process.py
python network.py
python post_process.py

cp -r ./model_repository/${MODEL_NAME}-* ../../model_repository/
cp -r ./model_repository/${MODEL_NAME} ../../model_repository/
