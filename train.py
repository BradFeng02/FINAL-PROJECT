"""
trains svc and serializes fitted model
using polarity dataset v2.0 at https://www.cs.cornell.edu/people/pabo/movie-review-data/
"""

from joblib import dump
import numpy as np
from sklearn import svm
from features import N_FEAT, find_feat

CLF_PATH = 'model.clf'

N_SAMP = 10  # 1k pos, 1k neg
POS = 1
NEG = 0

TRAIN_RATIO = .8
TRAIN_CUTOFF = TRAIN_RATIO * N_SAMP  # exclusive

K_FOLDS = 8
FOLD_SIZE = TRAIN_CUTOFF / K_FOLDS  # should be divisible

clf = svm.SVC()

samples = np.zeros((2 * N_SAMP, N_FEAT), dtype=np.float64)

labels = np.zeros(2 * N_SAMP, dtype=np.float64)
labels[N_SAMP:] = 1  # 0-999 neg, 1000-1999 pos

# TODO features
for i in range(N_SAMP):
    neg = 'bad'
    pos = 'good'
    find_feat(neg, samples, i)
    find_feat(pos, samples, i + N_SAMP)
# endfor

print(samples)

# clf.fit(samples, labels)

# dump(clf, CLF_PATH)
