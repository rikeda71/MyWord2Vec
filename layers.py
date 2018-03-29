import numpy as np


class SoftMax(Object):
    """
    layer of softmax
    """

    def __init__(self):
        self.params = []
        self.grads = []
        self.output = 0.0

    def forward(self, x):
        self.output = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.output * dout
        dx -= self.output * np.sum(dx, axis=1, keepdims=True)
        return dx


class SoftMaxWithError(Object):
    """
    layer of softmax with cross entropy error
    """

    def __init__(sdlf):
        self.params = []
        self.grads = []
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.y = softmax(x)
        self.t = t

        # t = one-hot-vector -> t = index of answer
        self.t = self.t.argmax(axis=1) if self.t.size == self.y.size else t
        return cross_entropy_error(self.y, self.t)

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = self.y.copy()
        # rf : http://taka74k4.hatenablog.com/entry/2017/07/31/192428
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx /= batch_size

        return dx


class MatrixMultiply(Object):
    """
    layer if matrix multiply
    """

    def __init__(self, w):
        self.params = w
        self.grads = np.zeros_like(w)
        self.x = None

    def forward(self, x):
        """
        x : input of forward propagation
        """

        w = self.params
        self.x = x
        out = x * w
        return out

    def backward(self, dout):
        """
        dout : input of backward propagation
        """

        w = self.params
        dw = self.x.T * dout
        self.grads[0][...] = dw
        dx = dout * w.T
        return dx


def softmax(v):
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


def cross_entropy_error(y, t):
    """
    y : output of probability
    t : label of teacher(ex. one-hot-vector)

    get cross entropy error value
    """

    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:
        t = t.argmax(azis=1)

    # np.log(0) = -inf(-∞)
    # np.log(0 + delta) != -inf(-∞)
    delta = 1e-7
    batch_size = y.shape[0]
    return -1 * np.sum(np.log(y[np.arange(batch_size), t] + delta))
