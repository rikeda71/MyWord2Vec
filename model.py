import numpy as np


class Word2Vec(object):
    """
    word2vecを実装するクラス
    """

    def __init__(self, vocab: list, sentences: list, dimention: int=100):
        """
        vocab     : list of words in text data
        sentences : list of sentences(separated by word)
        dimention : size of words matrix row
        ex)
        vocab = ["I", "You", "play", ...]
        sentences = [["I", "have", "a", "pen"], ["You", "play", "baseball"], ...]
        """

        self.vocab = vocab
        self.sentences = sentences
        # make one hot(1-of-k) vector
        self.one_hot = np.mat(np.identity(len(vocab)))
        # weight matrix. it is used in hidden-layer
        # ex) hidden_w * one_hot[0].T
        self.hidden_w = np.mat(np.random.rand(dimention, len(vocab)))
        # ex) output_w * (hidden_w * one_hot[0])
        self.output_w = np.mat(np.random.rand(len(vocab), dimention))

    def train(self, algorithm="skipgram", window: int=3, eta: float=0.1):
        """
        algorithm : learn method. skipgram or cbow(continuous bag-of-words)
        window    : range of surrounding words
        eta       : training rate

        train word2vec
        """

        pass

    def train_skipgram(self, window: int=3, eta: float=0.1):
        """
        window    : range of surrounding words
        eta       : training rate

        train word2vec by skipgram
        """

        for _ in range(100):
            new_hidden_w = numpy.zeros(self.hidden_w.shape)
            new_output_w = numpy.zeros(self.output_w.shape)
            pass

    def input_to_hidden(self, vocab_i: int):
        """
        vocab_i : index of vocabulary

        layer of input ~ layer of hidden
        calc matrix
        """

        return self.hidden_w * self.one_hot[vocab_i].T

    def hidden_to_output(self, v_ih):
        """
        v_ih : input * hidden result

        layer of hidden ~ layer of output
        calc matrix
        """

        return self.output_w * v_ih

    def update_hidden_w(self, i: int, wi: int, contexts: list, eta: float=0.1):
        """
        i        : index of row
        wi       : index of reference word
        contexts : numbers of words in window
        eta      : training rate

        update formula of weight of hidden layer
        """

        vecs = []
        for c in contexts:
            for v in self.vocab:
                y_v = self.softmax(hidden_to_output(input_to_hidden(v)))
                t_v = 1 if c == v else 0
                v_d_vj = self.output_w[v, i]
                vecs.append((y_v - t_v) * v_d_vj)
        return self.hidden_w[wi, i] - eta * sum(vecs)

    def update_output_w(self, i: int, j: int, wi: int, contexts: list, eta: float=0.1):
        """
        i        : index of row
        j        : index of column
        wi       : index of reference word
        contexts : numbers of words in window
        eta      : training rate

        update formula of weight of output layer
        """

        vecs = []
        for c in contexts:
            y_i = self.softmax(hidden_to_output(input_to_hidden(i)))
            t_i = 1 if c == i else 0
            v_d_wij = self.hidden_w[wi, j]
            vecs.append((y_i - t_i) * v_d_wij)
        return self.output_w[i, j] - eta * sum(vecs)

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
