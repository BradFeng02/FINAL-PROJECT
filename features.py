"""
defines N_FEAT and function for extracting features
"""

import numpy as np

N_FEAT = 1


def find_feat(s: str, arr: np.ndarray, i: int):
    """Populates features of s in arr[i]."""

    # feature 1
    arr[i, 0] = len(s)
