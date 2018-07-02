#!/usr/bin/env bash

predict_wv() {
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
            output(_ + ' |hCounts artist:%d album:%d track:%d artist_%d album_%d track_%d' % (art_c, alb_c, tr_c, art_c, alb_c, tr_c))" --input ../splitted_data/rwv_${1} --output ../splitted_data/rwv_${1}_u
    vw -d ../splitted_data/rwv_${1}_u -i ../splitted_data/wv_model -t -p ../splitted_data/wv_${1}
}

for NAME in name_1 name_5 name_10 no_name_100  no_name_5 no_name_10 no_name_25
do
    predict_wv $NAME
done
