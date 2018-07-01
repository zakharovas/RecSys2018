source ~/py3recsys/bin/activate

TRAIN_FILE=$1
TEST_FILE=$2
for SIZE in 1 5 10 25 100
do
    python -m utils.extract_with_track_number $TRAIN_FILE ${TRAIN_FILE}_sz${SIZE} $SIZE &
    python -m utils.extract_with_track_number $TEST_FILE ${TEST_FILE}_sz${SIZE} $SIZE &
done
wait