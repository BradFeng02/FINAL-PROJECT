"""
trains svc and serializes fitted model
using polarity dataset v2.0 at https://www.cs.cornell.edu/people/pabo/movie-review-data/
"""

from os import listdir
from joblib import dump
import numpy as np
from sklearn import svm
from features import N_FEAT, find_feat

CLF_PATH = 'model.clf'

DATA_PATH = 'data/txt_sentoken/'

N_SAMP = 10  # 1k pos, 1k neg
POS = 1
NEG = 0

TRAIN_RATIO = .8
TRAIN_CUTOFF = TRAIN_RATIO * N_SAMP  # exclusive

K_FOLDS = 8
assert TRAIN_CUTOFF % K_FOLDS == 0  # should be divisible
FOLD_SIZE = TRAIN_CUTOFF / K_FOLDS

neg_data = sorted(listdir(DATA_PATH + 'neg'))
pos_data = sorted(listdir(DATA_PATH + 'pos'))
assert N_SAMP <= len(neg_data) and N_SAMP <= len(pos_data)

clf = svm.SVC()

samples = np.zeros((2 * N_SAMP, N_FEAT), dtype=np.float64)

labels = np.zeros(2 * N_SAMP, dtype=np.float64)
labels[N_SAMP:] = 1  # 0-999 neg, 1000-1999 pos

# TODO features
for i in range(N_SAMP):
    neg_file = open(DATA_PATH + 'neg/' + neg_data[i], 'r')
    neg = neg_file.read()
    find_feat(neg, samples, i)
    neg_file.close()

    pos_file = open(DATA_PATH + 'pos/' + pos_data[i], 'r')
    pos = pos_file.read()
    find_feat(pos, samples, i + N_SAMP)
    pos_file.close()
# endfor

# print(samples)

# clf.fit(samples, labels)

# dump(clf, CLF_PATH)
