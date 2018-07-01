#!/usr/bin/env bash
create_examples() {
    python -m training.create_examples_for_training ../splitted_data/cb_train.json ../splitted_data/vector_train.json 200 50 $1 $2 30 ../splitted_data/tmp_examples.json_${3}
}

create_examples 1 1 name_1
create_examples 1 5 name_5
create_examples 1 10 name_10
create_examples 0 5 no_name_5
create_examples 0 5 no_name_10
create_examples 0 5 no_name_25
create_examples 0 100 no_name_100
