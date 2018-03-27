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

    def train_skipgram(self, window: int=3, eta: float=0.01):
        """
        window    : range of surrounding words
        eta       : training rate

        train word2vec by skipgram
        """

        for k in range(100):
            old_hidden_w = self.hidden_w
            old_output_w = self.output_w
            for sentence in self.sentences:
                print(sentence)
                for index, word in enumerate(sentence):
                    s = index - window if index - window > 0 else 0
                    e = index + window if index + window <= len(sentence) else len(sentence)
                    contexts = [self.vocab.index(sentence[i]) for i in range(s, e) if index != i]
                    print(self.hidden_w)
                    print(self.output_w)
                    # update weight: hidden layer to output layer
                    self.update_output_w(index, contexts, eta)
                    # update weight: input layer to hidden layer
                    self.update_hidden_w(index, contexts, eta)
            if np.linalg.norm((old_hidden_w - self.hidden_w).flatten()) < 0.1 and\
                    np.linalg.norm((old_output_w - self.output_w).flatten()) < 0.1:
                break

            eta *= 0.9
            print(k)
            print(np.linalg.norm((old_hidden_w - self.hidden_w).flatten()))
            print(np.linalg.norm((old_output_w - self.output_w).flatten()))
            print("----------------")

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

    def update_hidden_w(self, wi: int, contexts: list, eta: float=0.1):
        """
        wi       : index of reference word
        contexts : numbers of words in window
        eta      : training rate

        update formula of weight of hidden layer
        """

        y_c = np.mat(np.zeros((len(self.vocab), len(self.vocab))))
        t_c = np.mat(np.zeros((len(self.vocab), len(self.vocab))))
        for c in contexts:
            y_c += self.softmax(self.hidden_to_output(self.input_to_hidden(c)))
            t_c[c, c] += 1
        dv = np.sum(((y_c - t_c).T * self.output_w), axis=1).T
        self.hidden_w[wi] -= eta * dv
        # self.hidden_w[wi] -= eta * ((y_c - t_c) * self.output_w).T

    def update_output_w(self, wi: int, contexts: list, eta: float=0.1):
        """
        i        : index of row
        j        : index of column
        wi       : index of reference word
        contexts : numbers of words in window
        eta      : training rate

        update formula of weight of output layer
        """

        y_c = np.mat(np.zeros((len(self.vocab), len(self.vocab))))
        t_c = np.mat(np.zeros((len(self.vocab), len(self.vocab))))
        for c in contexts:
            y_c += self.softmax(self.hidden_to_output(self.input_to_hidden(c)))
            t_c[c, c] += 1
        dv = (y_c - t_c) * np.repeat(self.hidden_w[wi], self.hidden_w.shape[0], axis=0).T
        self.output_w -= eta * dv

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
