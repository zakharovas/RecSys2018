#!/usr/bin/env bash

for NAME in name_1 name_5 name_10
do
    python -m utils.add_wv_as_feature ../splitted_data/tmp_examples.json_${NAME} ../splitted_data/wv_${NAME} ../splitted_data/wv_examples.json_${NAME} &
done

for NAME in  no_name_100  no_name_5 no_name_10 no_name_25
do
        python -m utils.add_wv_as_feature ../splitted_data/tmp_examples.json_${NAME} ../splitted_data/wv_${NAME} ../splitted_data/wv_examples.json_${NAME} &
done
wait
