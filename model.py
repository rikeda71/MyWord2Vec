import numpy as np


class Word2Vec(object):
    """
    word2vecを実装するクラス
    """

    def __init__(self):
        pass

    def softmax(self, v):
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

    def sigmoid(self, v):
        """
        v : vector of numpy

        get sigmoid value
        implements measures of overflow
        """

        # calculation range
        sig_range = 34.53877639491
        if v <= -sig_range:
            return 1e-15
        elif v >= sig_range:
            return 1.0 - 1e-15
        return 1.0 / (1.0 + np.exp(-v))

    def sigmoid_gradient(self, fs):
        """
        fs : sigmoid value

        gradient of sigmoid method
        ex)
        fs = sigmoid(v)
        sigmoid_gradient(fs)
        """

        return fs * (1 - fs)
