import numpy as np
from sklearn import svm

N_SAMPLES = 1000  # 1k pos, 1k neg
POS = 1
NEG = 0

TRAIN_RATIO = .8
TRAIN_CUTOFF = TRAIN_RATIO * N_SAMPLES  # exclusive

K_FOLDS = 8
FOLD_SIZE = TRAIN_CUTOFF / K_FOLDS  # should be divisible

N_FEAT = 1

clf = svm.SVC()

samples = np.zeros((2 * N_SAMPLES, N_FEAT), dtype=np.float64)

labels = np.zeros(2 * N_SAMPLES, dtype=np.float64)
labels[N_SAMPLES:] = 1  # 0-999 neg, 1000-1999 pos



clf.fit(samples, labels)