#!/usr/bin/env bash


archive_file() {
    if [[ -f $1 ]]; then
        mv $1 $1$STAMP
    fi
}

try_split() {
    echo $TRAIN
    echo $TEST
    echo ------
    if [[ -f $TRAIN && -f $TEST ]]; then
        ALL="$DATA_DIR/all.json"
        if [[ -f $ALL ]]; then
            rm $ALL
        fi
        cat $TEST $TRAIN >> $ALL
        echo "START DATA FILES FOUND SKIP SPLITTING"
    else
        TRAIN="$DATA_DIR/train.json"
        TEST="$DATA_DIR/test.json"
        ALL="$DATA_DIR/all.json"
        python -m utils/dir_to_playlists "$DATA_DIR/raw/" $TRAIN $TEST $ALL
    fi

}


create_test() {
    DROPPED_TEST=${DATA_DIR}/dropped_test.json
    echo $TEST
    echo $DROPPED_TEST
    python -m prepare_test $TEST $DROPPED_TEST
    TEST=$DROPPED_TEST
}

create_info() {
    archive_file $TRACK_INFO
    archive_file $ALBUM_INFO
    archive_file $ARTIST_INFO
    python -m utils.extract_info $ALL $TRACKS track $TRACK_INFO &
    python -m utils.extract_info $ALL $ALBUMS album $ALBUM_INFO &
    python -m utils.extract_info $ALL $ARTIST artist $ARTIST_INFO &
    wait

}



try_encode() {
    TRACK_INFO=$DATA_DIR/track_names.json
    ALBUM_INFO=$DATA_DIR/album_names.json
    ARTIST_INFO=$DATA_DIR/artist_names.json
    
    echo "ENCODING"
    ALBUM_TO_ARTIST="$DATA_DIR/albums_to_artist.json"
    TRACK_TO_ALBUM="$DATA_DIR/track_to_album.json"
    echo ENCODING IDS
    python -m utils.encode_id $ALL $ARTIST $ALBUMS $TRACKS $ALBUM_TO_ARTIST $TRACK_TO_ALBUM
    echo "ENCODING DICTIONARIES CREATED"
    ENCODED_TRAIN="$DATA_DIR/encoded_train.json"
    ENCODED_TEST="$DATA_DIR/encoded_test.json"
    echo "EXTRACTING NAMES"
    create_info
    echo "NAMES EXTRACTED"
    archive_file $ENCODED_TRAIN
    archive_file $ENCODED_TEST
    echo "ENCODING TRAIN"
    python -m utils.encode_playlists $TRAIN $TRACKS $ENCODED_TRAIN &
    echo "ENCODING TEST"
    python -m utils.encode_playlists $TEST $TRACKS $ENCODED_TEST &
    python -m utils.encode_playlists "${DATA_DIR}/challenge_set.json" $TRACKS "${DATA_DIR}/test_c" &
    wait
    TRAIN=$ENCODED_TRAIN
    TEST=$ENCODED_TEST
    echo "ENCODIND COMPLETE"

}

create_popularity() {
    if [[ $ENCODING -eq 1 ]]; then
        archive_file $POPULARITY_ALBUM
        archive_file $POPULARITY_ARTIST
        archive_file $POPULARITY_TRACK
        python -m training.calculate_popularity $TRAIN $TRACK_TO_ALBUM $ALBUM_TO_ARTIST track $POPULARITY_TRACK &
        python -m training.calculate_popularity $TRAIN $TRACK_TO_ALBUM $ALBUM_TO_ARTIST album $POPULARITY_ALBUM &
        python -m training.calculate_popularity $TRAIN $TRACK_TO_ALBUM $ALBUM_TO_ARTIST artist $POPULARITY_ARTIST &
        wait
    fi
}


create_pairs() {
    echo $1
    echo $LIST_FILE
    echo $TRACK_TO_ALBUM $ALBUM_TO_ARTIST
    python -m utils.starspace_item_dataset $VECTOR_TRAIN $TRACK_TO_ALBUM $ALBUM_TO_ARTIST $1 $LIST_FILE 0
    python -m utils.starspace_item_dataset $VECTOR_TRAIN $TRACK_TO_ALBUM $ALBUM_TO_ARTIST $1 ${LIST_FILE}_als 1
    
}


train_als() {
    export OPENBLAS_NUM_THREADS=1
    python -m training.train_als ${LIST_FILE}_als $TRACK_TO_ALBUM $ALBUM_TO_ARTIST $1 $2 ${2}pl 256 50 1
    TMP_FILE=$DATA_DIR/${1}_tmp
    python -m utils.vectors_to_track_vectors $2 $TRACK_TO_ALBUM $ALBUM_TO_ARTIST $1 $TMP_FILE
    archive_file $2
    mv $TMP_FILE $2
    echo -e "$1_als $2" >> $VECTOR_FILE
    echo -e "$1_als_ui ${2}pl $2" >> $UI_VECTOR_FILE
}


train_starspace() {
    $STARSPACE train -trainFile $LIST_FILE -model $STARSPACE_MODEL -trainMode 1 -dim 128 -verbose 1 -thread 32   -lr 0.01 -epoch 20 -maxNegSamples 100 -negSearchLimit 100 -label 'l'
    TMP_FILE=$DATA_DIR/${1}_tmp
    python -m utils.extract_starspace_vector $STARSPACE_MODEL.tsv $TRACK_TO_ALBUM $ALBUM_TO_ARTIST $1 $2 1
    python -m utils.vectors_to_track_vectors $2 $TRACK_TO_ALBUM $ALBUM_TO_ARTIST $1 $TMP_FILE
    archive_file $2
    mv $TMP_FILE $2
    echo -e "$1_starspace $2" >> $VECTOR_FILE
}


create_vectors() {
    LIST_FILE=$DATA_DIR/${1}_list
    STARSPACE_MODEL=$DATA_DIR/${1}_stsp_model
    echo $LIST_FILE
    create_pairs $1
    train_als $1 $3
    train_starspace $1 $2
}

new_vectors() {
    create_vectors track $STARSPACE_TRACK $ALS_TRACK
    create_vectors album $STARSPACE_ALBUM $ALS_ALBUM
    create_vectors artist $STARSPACE_ARTIST $ALS_ARTIST
    wait
}


calculate_features() {
    POPULARITY_ARTIST=$DATA_DIR/artist.pop
    POPULARITY_ALBUM=$DATA_DIR/album.pop
    POPULARITY_TRACK=$DATA_DIR/track.pop
    create_popularity &
    STARSPACE_TRACK=$DATA_DIR/starspace_track.pickle
    STARSPACE_ALBUM=$DATA_DIR/starspace_album.pickle
    STARSPACE_ARTIST=$DATA_DIR/starspace_artist.pickle
    ALS_TRACK=$DATA_DIR/als_track.pickle
    ALS_ALBUM=$DATA_DIR/als_album.pickle
    ALS_ARTIST=$DATA_DIR/als_artist.pickle
    if [[ $UPDATE_VECTORS -eq 1 ]]; then
        archive_file $VECTOR_FILE
        archive_file $UI_VECTOR_FILE
        new_vectors

    fi
    wait

}

create_examples() {
    python -m training.create_examples_for_training $CATBOOST_TRAIN $VECTOR_TRAIN 100 100 0 25 30 ${EXAMPLES}
#    python -m training.create_examples_for_training $CATBOOST_TRAIN $VECTOR_TRAIN 100 100 0 30 ${EXAMPLES}2
    wait
    echo EXAMPLES CREATED
    telegram-send "EXAMPLES CREATED"
}

create_train() {
    CATBOOST_ALL=${DATA_DIR}/catboost.all
#    python -m calculate_features_main $EXAMPLES $TRACK_TO_ALBUM $ALBUM_TO_ARTIST $VECTOR_FILE $UI_VECTOR_FILE $POPULARITY_TRACK $POPULARITY_ALBUM $POPULARITY_ARTIST $CATBOOST_ALL 30
#    python -m calculate_features_main ${EXAMPLES}2 $TRACK_TO_ALBUM $ALBUM_TO_ARTIST $VECTOR_FILE $UI_VECTOR_FILE $POPULARITY_TRACK $POPULARITY_ALBUM $POPULARITY_ARTIST ${CATBOOST_ALL}2 40
    echo python -m utils.split_train_test $CATBOOST_ALL $CATBOOST_EXAMPLES $CATBOOST_TEST 0.95
    exit
#    python -m utils.split_train_test ${CATBOOST_ALL}2 ${CATBOOST_EXAMPLES}2 ${CATBOOST_TEST}2 0.95 &
    telegram-send "TRAIN READY"
    wait
#    exit
}

fit_catboost() {
    CD_FILE=${DATA_DIR}/cd
    echo -e "0\tTarget" > $CD_FILE
    echo -e "1\tGroupId" >> $CD_FILE
    echo -e "2\tCateg" >> $CD_FILE
    archive_file $CATBOOST_MODEL
    echo $CATBOOST fit -f $CATBOOST_EXAMPLES  -t $CATBOOST_TEST  --cd $CD_FILE --loss-function YetiRank -T 32 --has-header -m $CATBOOST_MODEL --rsm 0.3 --used-ram-limit 100GB -i 8000 -n 10 
    $CATBOOST fit -f $CATBOOST_EXAMPLES -t $CATBOOST_TEST  --cd $CD_FILE --loss-function YetiRank -T 32 --has-header -m $CATBOOST_MODEL --rsm 0.3 --used-ram-limit 100GB -i 4000 -n 10
    telegram-send "CATBOOST TRAINED"
    $CATBOOST fstr --fstr-type FeatureImportance --input-path $CATBOOST_EXAMPLES  --cd $CD_FILE -T 32 --has-header -m $CATBOOST_MODEL
    telegram-send "$( cat feature_strength.tsv )"

}

apply_model() {
    if [[ $DIR_TEST -eq 1 ]]; then
        I=0
        for TEST_FILE in $TEST/test_?
        do
            python -m apply_model ${TEST_FILE}_u $TRACK_TO_ALBUM $ALBUM_TO_ARTIST $VECTOR_FILE $UI_VECTOR_FILE $POPULARITY_TRACK $POPULARITY_ALBUM $POPULARITY_ARTIST ${TEST_FILE}_candidates $CATBOOST_MODEL ${TEST_FILE}_predictions ../splitted_data/svd_pp ../splitted_data/nals
            I=$((I + 1))
            if [[ $I -eq 1 ]]; then
                I=0
                wait 
                # show_metrics &
            fi
        done
        wait
    else
        python -m apply_model $TEST $TRACK_TO_ALBUM $ALBUM_TO_ARTIST $VECTOR_FILE $POPULARITY_TRACK $POPULARITY_ALBUM $POPULARITY_ARTIST $CANDIDATE_FILE $CATBOOST_MODEL ${PREDICTIONS}
    fi
}


calculate_candidates() {
    if [[ $DIR_TEST -eq 1 ]]; then
        I=0
        for TEST_FILE in $TEST/test_?
        do
            python -m create_candidates $TEST_FILE $TRAIN ${TEST_FILE}_candidates 32
            I=$((I + 1))
            if [[ $I -eq 1 ]]; then
                I=0
                wait
            fi
        done
        wait
    else
        python -m create_candidates $TEST $TRAIN $CANDIDATE_FILE 32
    fi
}

show_metrics() {
    if [[ $DIR_TEST -eq 1 ]]; then
        for TEST_FILE in $TEST/test_?
        do
            echo $TEST_FILE
            HITS=${TEST_FILE}_hits
            python -m recommendations_to_hits $TEST_FILE ${TEST_FILE}_predictions $TRACK_TO_ALBUM $ALBUM_TO_ARTIST $HITS
            python -m utils.calculate_metrics $HITS > $FINAL_RESULTS
            telegram-send "$TEST_FILE"
            telegram-send "$( cat final_tesults.table )"
        done
    else
        HITS=${DATA_DIR}/hits
        python -m recommendations_to_hits $TEST $PREDICTIONS $TRACK_TO_ALBUM $ALBUM_TO_ARTIST $HITS
        python -m utils.calculate_metrics $HITS > $FINAL_RESULTS
        telegram-send "$( cat final_tesults.table )"
    fi

}

STAMP=$(date  "+_%Y_%m_%d_%H_%M_%S")
ENCODING=0
UPDATE=0
UPDATE_VECTORS=0
UPDATE_CATBOOST=0
UPDATE_CANDIDATES=0

DATA_DIR="../splitted_data"
TRAIN="$DATA_DIR/train.json"
TEST="$DATA_DIR/test.json"
ALL="$DATA_DIR/all.json"
SET_VIRTUALENV=0
PREFIX=splitted
STARSPACE=~/Starspace/starspace
CATBOOST=~/catboost
CREATE_TEST=0
APPLY=0
DIR_TEST=0
while [ -n "$1" ]
do
    case "$1" in
        --train) TRAIN=$2
            shift;;
        --test) TEST=$2
            shift;;
        --encoding) ENCODING=1;;
        --update_models) UPDATE_EXAMPLES=1
        UPDATE_VECTORS=1
        UPDATE_CATBOOST=1;;
        --update_boost) UPDATE_CATBOOST=1;;
        --new) ENCODING=1
            UPDATE_VECTORS=1
            UPDATE_CATBOOST=1
            UPDATE_CANDIDATES=1;;
        --env) SET_VIRTUALENV=1
            VIRTUALENV_PATH=$2
            shift;;
        --starspace_path) STARSPACE=$2
            shift;;
        --catboost_path) CATBOOST=$2
            shift;;
        --create_test) CREATE_TEST=1
        UPDATE_CANDIDATES=1;;

        --update_candidates) UPDATE_CANDIDATES=1;;

        --apply) APPLY=1;;

        --test_dir) DIR_TEST=1;;

        *) echo "$1 unknown option"
            exit;;
    esac
    shift
done

if [[ $SET_VIRTUALENV -eq 1 ]]; then
    source $VIRTUALENV_PATH
fi
echo $VIRTUALENV_PATH
echo $DATA_DIR
echo $TEST
echo $TRAIN

try_split

CREATE_DICT=utils/encode_id.py
TRACKS=("$DATA_DIR/tracks.json")
ALBUMS=("$DATA_DIR/albums.json")
ARTIST=("$DATA_DIR/atrist.json")
ALBUM_TO_ARTIST="$DATA_DIR/albums_to_artist.json"
TRACK_TO_ALBUM="$DATA_DIR/track_to_album.json"

if [[ ${ENCODING} -eq 1 ]]; then
    try_encode
    exit
fi

VECTOR_FILE=$DATA_DIR/info_about_vector.tsv
UI_VECTOR_FILE=$DATA_DIR/info_about_ui_vector.tsv
VECTOR_TRAIN="$DATA_DIR/vector_train.json"
CATBOOST_TRAIN="$DATA_DIR/cb_train.json"
EXAMPLES=${DATA_DIR}/tmp_examples.json

export OPENBLAS_NUM_THREADS=1

if [[ $UPDATE_VECTORS -eq 1 ]]; then

    TMP_TRAIN="$DATA_DIR/shuffled_train.json"
    shuf $TRAIN > $TMP_TRAIN
    head -n 900000 $TMP_TRAIN > $VECTOR_TRAIN
    tail -n 90000 $TMP_TRAIN > $CATBOOST_TRAIN
    new_vectors
    exit
fi

if [[ $UPDATE_EXAMPLES -eq 1 ]]; then
    create_examples
    exit
fi

POPULARITY_ARTIST=$DATA_DIR/artist.pop
POPULARITY_ALBUM=$DATA_DIR/album.pop
POPULARITY_TRACK=$DATA_DIR/track.pop

#exit
CATBOOST_MODEL=${DATA_DIR}/cb_model.bin
if [[ $UPDATE_CATBOOST -eq 1 ]]; then
    CATBOOST_EXAMPLES=${DATA_DIR}/catboost.train
    CATBOOST_TEST=${DATA_DIR}/catboost.test
    fit_catboost
    exit
fi

CANDIDATE_FILE=${DATA_DIR}/test_candidates.pickle

if [[ $UPDATE_CANDIDATES -eq 1 ]]; then
#    archive_file $CANDIDATE_FILE

    calculate_candidates
    exit
fi

FINAL_RESULTS=final_tesults.table
PREDICTIONS=${DATA_DIR}/preditions.json
if [[ $APPLY -eq 1 ]]; then
    apply_model
    exit
fi
#archive_file $FINAL_RESULTS
#show_metrics
