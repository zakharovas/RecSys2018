#!/usr/bin/env bash


work_with_test() {
    NAME=$(basename $1)
    python utils/reformat_test.py $1 ${1}_r
    python utils/encode_playlists.py ${1}_r $TRACKS ${1}_e
    rm ${1}_r
    python utils/add_tracks.py ${1}_e ${DATA_DIR}/encoded_test.json $DATA_DIR/prepared_test/${NAME}
    rm ${1}_e


}

rm -r
DATA_DIR=$1
VIRTUAL_ENV=$2
source $VIRTUAL_ENV
rm -r ${DATA_DIR}/prepared_test
mkdir ${DATA_DIR}/prepared_test
TRACKS=("$DATA_DIR/tracks.json")
for file in ${DATA_DIR}/test/test_*
do
    echo $file

    work_with_test $file &
#    exit
done
wait