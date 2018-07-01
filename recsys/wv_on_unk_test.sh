#!/usr/bin/env bash


for TEST_FILE in $1/test_?
do
    python -m utils.wv_on_unk_test $TEST_FILE ${TEST_FILE}_candidates ../../splitted_data/track_to_album.json ../../splitted_data/albums_to_artist.json ~/name_vectors/encoding.json ${TEST_FILE}_wv 1 &
done
