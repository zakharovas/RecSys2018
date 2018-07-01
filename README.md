# RecSys2018
MIPT_MSU team RecSys Challenge 2018 solution

### Requirements
We used Python3.5

Install requirements from requirements.txt

You will also need [Catboost](https://catboost.yandex/) and [Starspace](https://github.com/facebookresearch/StarSpace)

### Creating solution

**All scripts are started from RecSys2018/recsys**

1) In RecSys2018 folder create splitted_data folder and put million playlist dataset there (RecSys2018/splitted_data/raw)

2) Put challenge set into splitted_data folder (RecSys2018/splitted_data/challenge_set.json)
 
1) Encode million playlist dataset and challenge set
    ```bash
    bash recsys_script.sh  --encoding 
    ```
1) Train iAls and Starspace
    ```bash
    bash recsys_script.sh  --encoding 
    ```



1) Train Vowpal Wabbit 


1) Train name iAls

1) Create examples files for Catboost

1) Create pools for catboost from examples
    
    ```bash
    feature.sh
    ```

1) Train Catboost
    ```bash
    bash cb.sh 
    ```

1) Create candidates for challenge set
    ```bash
    bash recsys_script.sh  --update_candidates 
    ```

1) Predict with Vowpal Wabbit model
    ```bash
    bash wv_on_unk_test.sh  ../splitted_data 
    ```

1) Apply trained models
    ```bash
    bash recsys_script.sh  --apply 
    ```

1) Decode created solution
    ```bash
    python utils/create_solution.py ../splitted_data/test_c_predictions \
                                    ../splitted_data/tracks.json \
                                    ../MIPT_MSU_solution.csv
    ```
 

With recsys_script.sh you may set path to catboost binary with --catboost_path option.
To Starspace binary with --starspace_path option. To your python virtualenv with --env option.


You will need about 100GB RAM

We recommend you to train Catboost on GPU, beause it takes several hours instead of days.

 

 
 
