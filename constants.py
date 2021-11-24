"""
A module with constants used by train and test
"""

from os import listdir

CLF_PATH = 'model.clf'

DATA_PATH = 'data/txt_sentoken/'

N_SAMP = 1000  # 1k pos, 1k neg

TRAIN_RATIO = .8
N_TRAIN = int(TRAIN_RATIO * N_SAMP)  # exclusive
N_TEST = N_SAMP - N_TRAIN

K_FOLDS = 8
# assert N_TRAIN % K_FOLDS == 0  # should be divisible
FOLD_SIZE = N_TRAIN / K_FOLDS

neg_data = sorted(listdir(DATA_PATH + 'neg'))
pos_data = sorted(listdir(DATA_PATH + 'pos'))
assert N_SAMP <= len(neg_data) and N_SAMP <= len(pos_data)