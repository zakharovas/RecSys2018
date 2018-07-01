#!/usr/bin/env bash
export OPENBLAS_NUM_THREADS=1
features() {
python -m calculate_features_main /home/alzaharov/spotify/splitted_data/wv_examples.json${1} /home/alzaharov/spotify/splitted_data/track_to_album.json /home/alzaharov/spotify/splitted_data/albums_to_artist.json /home/alzaharov/spotify/splitted_data/info_about_vector.tsv /home/alzaharov/spotify/splitted_data/info_about_ui_vector.tsv /home/alzaharov/spotify/splitted_data/track.pop /home/alzaharov/spotify/splitted_data/album.pop /home/alzaharov/spotify/splitted_data/artist.pop /home/alzaharov/spotify/splitted_data/rcatboost.all${1} 30 $2
python -m utils.split_train_test /home/alzaharov/spotify/splitted_data/rcatboost.all${1} /home/alzaharov/spotify/splitted_data/rcatboost.train${1} /home/alzaharov/spotify/splitted_data/rcatboost.test${1} 0.95
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