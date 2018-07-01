#!/usr/bin/env bash


for TEST_FILE in ${1}/test_?
do
    python -m utils.add_wv_as_feature_to_unk $TEST_FILE ${TEST_FILE}_candidates ${TEST_FILE}_wv_f ${TEST_FILE}_u &
done
