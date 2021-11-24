"""
defines constants, and function for extracting features
"""

import numpy as np
import nltk.sentiment.vader as vd
from nltk.corpus import sentiwordnet as swn, wordnet as wn

N_FEAT = 5

si = vd.SentimentIntensityAnalyzer()


def find_feat(s: str, arr: np.ndarray, i: int):
    """Populates features of s in arr[i]."""

    # feature 1
    p = 0
    n = 0
    sents = s.split('\n')
    for sent in sents:
        score = si.polarity_scores(sent)['compound']
        if score > 0.1:
            p += 1
        elif score < 0.1:
            n += 1
    # arr[i, 0] = 1 if p > n else 0
    # arr[i, 0] = 1 if (p - n) > 10 else 0
    # arr[i, 0] = -1 if (n - p) > 10 else 0
    arr[i, 0] = p-n

    # TODO everything temp
    words = s.split(' ')

    objwords = negwords = poswords = xtraneg = xtrapos = 0

    for w in words:
        synsets = wn.synsets(w)
        if len(synsets) > 0:
            scores = swn.senti_synset(synsets[0].name())

            # if scores.neg_score() > 0.5 and scores.pos_score() > 0.5:
            #     objwords += 1
            if (scores.neg_score() - scores.pos_score()) > .1:
                negwords += 1
            elif (scores.pos_score() - scores.neg_score()) > .1:
                poswords += 1
            else:
                objwords += 1

            if (scores.neg_score() - scores.pos_score()) > .5:
                xtraneg += 1
            elif (scores.pos_score() - scores.neg_score()) > .5:
                xtrapos += 1

            # if scores.obj_score() >= 0.5:
            #     objwords += 1
            # elif scores.neg_score() > scores.pos_score():
            #     negwords += 1
            # else:
            #     poswords += 1
        
    arr[i, 1] = negwords
    arr[i, 2] = poswords
    arr[i, 3] = xtraneg
    arr[i, 4] = xtrapos
    # arr[i, 3] = objwords
#enddef
