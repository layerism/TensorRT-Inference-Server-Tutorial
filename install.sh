#!/bin/bash

cd backend
python setup.py install
cd -

cd client_py
python setup.py install
cd -
