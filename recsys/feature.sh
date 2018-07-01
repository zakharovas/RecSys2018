#!/usr/bin/env bash
export OPENBLAS_NUM_THREADS=1
features() {
python -m calculate_features_main ../splitted_data/wv_examples.json${1} ../splitted_data/track_to_album.json ../splitted_data/albums_to_artist.json ../splitted_data/info_about_vector.tsv ../splitted_data/info_about_ui_vector.tsv ../splitted_data/track.pop ../splitted_data/album.pop ../splitted_data/artist.pop ../splitted_data/rcatboost.all${1} 30 $2 ../splitted_data/svd_pp ../splitted_data/nals
python -m utils.split_train_test ../splitted_data/rcatboost.all${1} ../splitted_data/rcatboost.train${1} ../splitted_data/rcatboost.test${1} 0.95
}

ADD="_name_10"
features $ADD 1

ADD="_no_name_10"
features $ADD 0

ADD="_name_5"
features $ADD 1
ADD="_no_name_5"
features $ADD 0

ADD="_name_1"
features $ADD 1

ADD="_no_name_100"
features $ADD 0

ADD="_no_name_25"
features $ADD 0