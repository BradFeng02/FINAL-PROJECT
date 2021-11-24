"""
trains svc and serializes fitted model
using polarity dataset v2.0 at https://www.cs.cornell.edu/people/pabo/movie-review-data/
"""

from joblib import dump
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from constants import CLF_PATH, N_TRAIN, DATA_PATH, neg_data, pos_data
from features import N_FEAT, find_feat

clf = svm.SVC()

samples = np.zeros((2 * N_TRAIN, N_FEAT), dtype=np.float64)

labels = np.zeros(2 * N_TRAIN, dtype=np.float64)
labels[N_TRAIN:] = 1  # neg, ..., pos, ...

for i in range(N_TRAIN):
    neg_file = open(DATA_PATH + 'neg/' + neg_data[i], 'r')
    neg = neg_file.read()
    find_feat(neg, samples, i)
    neg_file.close()

    pos_file = open(DATA_PATH + 'pos/' + pos_data[i], 'r')
    pos = pos_file.read()
    find_feat(pos, samples, i + N_TRAIN)
    pos_file.close()
# endfor

# for i in range(2 * N_TRAIN):
#     print(labels[i], samples[i])

feat_range = np.zeros((N_FEAT, 2))  # (min, max) for standardizing
for i in range(2 * N_TRAIN):
    for j in range(N_FEAT):
        if samples[i, j] < feat_range[j, 0]:
            feat_range[j, 0] = samples[i, j]
        elif samples[i, j] > feat_range[j, 1]:
            feat_range[j, 1] = samples[i, j]

for i in range(2 * N_TRAIN):  # standardize to [-1 , 1]
    for j in range(N_FEAT):
        samples[i, j] = (samples[i, j] - feat_range[j, 0]) / \
            (feat_range[j, 1] - feat_range[j, 0]) * 2 - 1

# for i in range(2 * N_TRAIN):
#     print(labels[i], samples[i])

# plt.title("feature") 
# plt.xlabel("neg/pos") 
# plt.ylabel("val")
# plt.plot(labels, np.array([samp[3] for samp in samples]), '.')
# plt.show()

clf.fit(samples, labels)

dump(clf, CLF_PATH)
