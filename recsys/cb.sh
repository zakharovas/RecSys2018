#!/usr/bin/env bash

train_cb(){
    $CATBOOST fit -f ../splitted_data/catboost.train_$1 -t ../splitted_data/catboost.test_$1  --cd $CD_FILE --loss-function YetiRank -T 32 --has-header -m cb_model.bin_$1 --rsm 0.5 --used-ram-limit 100GB -i 15000 -n 10

}

CATBOOST=$1
CD_FILE="../splitted_data/cd"
echo -e "0\tTarget" > $CD_FILE
echo -e "1\tGroupId" >> $CD_FILE

for NAME in name_1 name_5 name_10 no_name_100  no_name_5 no_name_10 no_name_25
do
    train_cb $NAME
done
