import numpy as np


class Word2Vec(object):
    """
    word2vecを実装するクラス
    """

    def __init__(self):
        pass

    def softmax(self, v: np.array):
        """
        v : matrix or vector of numpy

        get softmax value
        implements measures of overflow
        """

        v = v.astype(np.float64)
        if len(v.shape) > 1:
            # v = matrix
            exp_v = np.exp(v.T - v.max(1))
            soft_v = exp_v / exp_v.sum(axis=0)
            soft_v = soft_v.T
        else:
            # v = vector
            exp_v = np.exp(v - v.max())
            soft_v = exp_v / exp_v.sum(axis=0)
        return soft_v
