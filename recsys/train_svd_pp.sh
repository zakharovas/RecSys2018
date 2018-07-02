#!/usr/bin/env bash

cp -r svd_pp ../splitted_data/svd_pp

$CATBOOST fit -f $CATBOOST_EXAMPLES -t $CATBOOST_TEST  --cd $CD_FILE --loss-function YetiRank -T 32 --has-header -m $CATBOOST_MODEL --rsm 0.3 --used-ram-limit 100GB -i 4000 -n 10


for $NAME