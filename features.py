"""
defines constants, and function for extracting features
"""

from typing import Tuple
import numpy as np
import nltk.sentiment.vader as vd
from nltk.sentiment.vader import VaderConstants as vdConsts
from nltk.corpus import sentiwordnet as swn, wordnet as wn, stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

N_FEAT = 7

stop = stopwords.words('english')
si = vd.SentimentIntensityAnalyzer()

def find_feats(s: str, arr: np.ndarray, i: int):
    """finds features of s, puts them in arr[i][]."""

    sents = s.split('\n')

    # using SentiWordNet
    totwords = objwords = negwords = poswords = xtraneg = xtrapos = stops = 0

    for sent in sents:
        words = pos_tag(word_tokenize(sent))
        for w, pos in words:
            if w in stop:
                stops += 1
                continue

            wnpos = ''
            if pos.startswith('J'):
                wnpos = wn.ADJ
            elif pos.startswith('V'):
                wnpos = wn.VERB
            elif pos.startswith('N'):
                wnpos = wn.NOUN
            elif pos.startswith('R'):
                wnpos = wn.ADV

            synsets = wn.synsets(w, wnpos)
            if len(synsets) <= 0:  # if no synsets for tagged pos
                synsets = wn.synsets(w)
            if len(synsets) > 0:  # has synsets
                totwords += 1
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
    
    arr[i, 0] = totwords
    arr[i, 1] = negwords
    arr[i, 2] = poswords
    arr[i, 3] = xtraneg
    arr[i, 4] = xtrapos
    arr[i, 5] = objwords
    arr[i, 6] = stops
#enddef

def standardize_feats(samples: np.ndarray, n_train: int):
    feat_range = np.zeros((N_FEAT, 2))  # (min, max) for standardizing
    for i in range(2 * n_train):
        for j in range(N_FEAT):
            if samples[i, j] < feat_range[j, 0]:
                feat_range[j, 0] = samples[i, j]
            elif samples[i, j] > feat_range[j, 1]:
                feat_range[j, 1] = samples[i, j]

    for i in range(2 * n_train):  # standardize to [-1 , 1]
        for j in range(N_FEAT):
            if feat_range[j, 1] == feat_range[j, 0]:
                samples[i, j] = 0
            else:
                samples[i, j] = (samples[i, j] - feat_range[j, 0]) / \
                    (feat_range[j, 1] - feat_range[j, 0]) * 2 - 1

def make_input_arrs(n_train: int) -> Tuple[np.ndarray, np.ndarray]:
    samples = np.zeros((2 * n_train, N_FEAT), dtype=np.float64)
    labels = np.zeros(2 * n_train, dtype=np.float64)
    return (samples, labels)