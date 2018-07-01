#!/usr/bin/env bash

python -m utils.dataset_for_vw ../splitted_data/vector_train.json \
                                  ../splitted_data/track_to_album.json \
                                  ../splitted_data/albums_to_artist.json \
                                  ../splitted_data/nals/encoding.json \
                                  ../splitted_data/wv_train
    pyt "tokens = _.split()
           pl_art = []
            pl_alb = []
            pl_tr = []
            art = None
            alb = None
            tr = None
            ns = None
            for token in tokens:
                if token.startswith('|'):
                    ns = token[1:]
                elif ns is None:
                    continue
                elif ns[0] == 'a':
                    pl_art.append(token)
                elif ns[0] == 'b':
                    pl_alb.append(token)
                elif ns[0] == 'c':
                    pl_tr.append(token)
                elif ns[0] == 'd':
                    art = token
                elif ns[0] == 'e':
                    alb = token
                elif ns[0] == 'f':
                    tr = token
            art_c = pl_art.count(art)
            alb_c = pl_alb.count(alb)
            tr_c = pl_tr.count(tr)
            output(_ + ' |hCounts artist:%d album:%d track:%d artist_%d album_%d track_%d' % (art_c, alb_c, tr_c, art_c, alb_c, tr_c))" ../splitted_data/wv_train > ../splitted_data/wv_train_1
    vw -d ../splitted_data/wv_train_1 -f ../splitted_data/wv_model -c  --loss_function logistic -b 24 --l2 0.1 -l 0.02 --passes 100 --bfgs -q ad -q be -q cf -q gd -q ge -q gf
