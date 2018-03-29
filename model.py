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
        self.hidden_w = np.mat(np.random.rand(dimention, len(vocab)) -
                               np.full((dimention, len(vocab)), 0.5))
        # ex) output_w * (hidden_w * one_hot[0])
        self.output_w = np.mat(np.random.rand(len(vocab), dimention) -
                               np.full((len(vocab), dimention), 0.5))

    def train(self, algorithm="skipgram", window: int=3, eta: float=0.1):
        """
        algorithm : learn method. skipgram or cbow(continuous bag-of-words)
        window    : range of surrounding words
        eta       : training rate

        train word2vec
        """

        self.train_skipgram(window, eta)
