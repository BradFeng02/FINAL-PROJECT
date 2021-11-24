"""
tests trained svm model
"""

from typing import NewType
from joblib import load
import numpy as np
from constants import CLF_PATH, N_TRAIN, N_TEST, DATA_PATH, neg_data, pos_data
from features import N_FEAT, find_feat

clf = load(CLF_PATH)

tests = np.zeros((2 * N_TEST, N_FEAT), dtype=np.float64)

labels = np.zeros(2 * N_TEST, dtype=np.float64)
labels[N_TEST:] = 1  # neg, ..., pos, ...

for i in range(N_TEST):
    neg_file = open(DATA_PATH + 'neg/' + neg_data[i + N_TRAIN], 'r')
    neg = neg_file.read()
    find_feat(neg, tests, i)
    neg_file.close()

    pos_file = open(DATA_PATH + 'pos/' + pos_data[i + N_TRAIN], 'r')
    pos = pos_file.read()
    find_feat(pos, tests, i + N_TEST)
    pos_file.close()
# endfor

feat_range = np.zeros((N_FEAT, 2))  # (min, max) for standardizing
for i in range(2 * N_TEST):
    for j in range(N_FEAT):
        if tests[i, j] < feat_range[j, 0]:
            feat_range[j, 0] = tests[i, j]
        elif tests[i, j] > feat_range[j, 1]:
            feat_range[j, 1] = tests[i, j]

for i in range(2 * N_TEST):  # standardize to [-1 , 1]
    for j in range(N_FEAT):
        tests[i, j] = (tests[i, j] - feat_range[j, 0]) / \
            (feat_range[j, 1] - feat_range[j, 0]) * 2 - 1

predicts = clf.predict(tests)

correct = neg_correct = pos_correct = 0
for i in range(N_TEST):
    if labels[i] == predicts[i]:
        correct += 1
        neg_correct += 1
    if labels[i + N_TEST] == predicts[i + N_TEST]:
        correct += 1
        pos_correct += 1

print('neg:', neg_correct / N_TEST, '({}/{})'.format(neg_correct, N_TEST))
print('pos:', pos_correct / N_TEST, '({}/{})'.format(pos_correct, N_TEST))
print('all:', correct / (2 * N_TEST))

# for i in range(2 * N_TEST):
#     print(labels[i], predicts[i])
