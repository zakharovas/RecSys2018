#!/usr/bin/env bash


for NAME in name_1 name_5 name_10
do
    python -m utils.wv_on_test ../splitted_data/tmp_examples.json_${NAME} ../splitted_data/track_to_album.json ../splitted_data/albums_to_artist.json ../splitted_data/nals/encoding.json ../splitted_data/rwv_${NAME} 1 &
done

for NAME in  no_name_100  no_name_5 no_name_10 no_name_25
do
    python -m utils.wv_on_test ../splitted_data/tmp_examples.json_${NAME} ../splitted_data/track_to_album.json ../splitted_data/albums_to_artist.json ../splitted_data/nals/encoding.json ../splitted_data/rwv_${NAME} 0 &
done
wait
