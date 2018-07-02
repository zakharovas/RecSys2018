#!/usr/bin/env bash

cp ../splitted_data/vector_train.json svd_pp/dataset.json
cd svd_pp
python parser.py
python model.py
cd ..
cp -r svd_pp ../splitted_data/svd_pp
