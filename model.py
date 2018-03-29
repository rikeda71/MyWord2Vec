import numpy as np
from abc import ABCMeta, abstractmethod
from layers import *


class Word2Vec(metaclass=ABCMeta):
    """
    word2vecを実装するクラス
    """

    def __init__(self, vocab: list, sentences: list, dimention: int=100, window: int=3):
        """
        vocab     : list of words in text data
        sentences : list of sentences(separated by word)
        dimention : size of words matrix row
        window    : size of window
        ex)
        vocab = ["I", "You", "play", ...]
        sentences = [["I", "have", "a", "pen"], ["You", "play", "baseball"], ...]
        """

        self.vocab = vocab
        self.sentences = sentences
        self.window = window
        # make one hot(1-of-k) vector
        self.one_hot = np.mat(np.identity(len(vocab)))
        # weight matrix. it is used in hidden-layer
        # ex) hidden_w * one_hot[0].T
        hidden_w = np.mat(np.random.rand(dimention, len(vocab)) -
                          np.full((dimention, len(vocab)), 0.5))
        # ex) output_w * (hidden_w * one_hot[0])
        output_w = np.mat(np.random.rand(len(vocab), dimention) -
                          np.full((len(vocab), dimention), 0.5))
        self.__set__(hidden_w, output_w)
        self.wv = hidden_w

    @abstractmethod
    def __set__(self, hidden_w, output_w):
        pass

    @abstractmethod
    def __forward__(self, contexts, word):
        pass

    @abstractmethod
    def __backward__(self, dout):
        pass

    def train(self, algorithm="skipgram", window: int=3, eta: float=0.1):
        """
        algorithm : learn method. skipgram or cbow(continuous bag-of-words)
        window    : range of surrounding words
        eta       : training rate

        train word2vec
        """

        self.train_skipgram(window, eta)


class Skipgram(Word2Vec):
    """
    skipgram model
    """

    def __set__(self, hidden_w, output_w):
        self.inLayer = MatrixMultiply(hidden_w)
        self.outLayer = MatrixMultiply(output_w)
        self.lossLayer = [SoftMaxWithError() for i in range(self.window * 2)]

    def __forward__(self, contexts, word):
        h = self.inLayer.forward(self.onehot(word))
        s = self.outLayer.forward(h)
        loss = sum([l.forward(s, self.one_hot(i)) for i, l in zip(range(self.window * 2), self.lossLayer)])
        return loss

    def __backward__(self, dout=1):
        ds = sum([l.backward(dout) for l in self.lossLayer])
        dh = self.outLayer.backward(ds)
        self.inLayer.backward(dh)


class Cbow(Word2Vec):
    """
    cbow model
    """

    def __set__(self, hidden_w, output_w):
        self.inLayer = [MatrixMultiply(hidden_w) for i in ragen(len(self.window))]
        self.outLayer = MatrixMultiply(output_w)
        self.lossLayer = SoftMaxWithError()

    def __forward__(self, contexts, word):
        h = sum([l.forward(s, self.one_hot(i)) for i, l in zip(range(self.window * 2), self.inLayer)]) / len(self.inLayer)
        s = self.outLayer.forward(h)
        loss = self.lossLayer.forward(s, self.one_hot(word))
        return loss

    def __backward__(self, dout=1):
        ds = self.lossLayer.backward(dout)
        da = self.outLayer.backward(ds) / len(self.inLayer)
        for l in self.inLayer:
            l.backward(da)
