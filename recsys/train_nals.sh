#!/usr/bin/env bash

export OPENBLAS_NUM_THREADS=1
python nals/pnames.py ../splitted_data/vector_train.json
cp -r nals ../splitted_data/nals