import numpy as np
from abc import ABCMeta, abstractmethod
from layers import *


class Word2Vec(metaclass=ABCMeta):
    """
    word2vecを実装するクラス
    """

    def __init__(self, sentences: list, dimention: int=100, window: int=3):
        """
        vocab     : list of words in text data
        sentences : list of sentences(separated by word)
        dimention : size of words matrix row
        window    : size of window
        ex)
        vocab = ["I", "You", "play", ...]
        sentences = [["I", "have", "a", "pen"], ["You", "play", "baseball"], ...]
        """

        self.vocab = list(set(sum(sentences, [])))
        self.vocab_size = len(self.vocab)
        self.sentences = sentences
        self.sentence_size = len(sentences)
        self.window = window
        self.params, self.grads = [], []
        # make one hot(1-of-k) vector
        self.one_hot = np.mat(np.identity(self.vocab_size))
        # weight matrix. it is used in hidden-layer
        # ex) one_hot[0] * hidden_w
        hidden_w = np.mat(np.random.rand(self.vocab_size, dimention) -
                          np.full((self.vocab_size, dimention), 0.5))
        # ex) (hidden_w * one_hot[0]) * output_w
        output_w = np.mat(np.random.rand(dimention, self.vocab_size) -
                          np.full((dimention, self.vocab_size), 0.5))
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

    def train(self, eta: float=0.1, epoch_size: int=100, batch_size: int=32):
        """
        algorithm  : learn method. skipgram or cbow(continuous bag-of-words)
        window     : range of surrounding words
        eta        : training rate
        epoch_size : size of epoch
        batch_size : size of batch

        train word2vec
        """

        iter_size = self.sentence_size // batch_size
        contexts, words = [], []
        for s in self.sentences:
            c, w = self.__contexts_and_word(s)
            contexts.append(c)
            words.append(w)

        for epoch in range(epoch_size):
            for idx in np.random.permutation(np.arange(self.sentence_size))[:iter_size]:
                for c, w in zip(contexts[idx], words[idx]):
                    loss = self.__forward__(c, w)
                    self.__backward__()
                    params, grads = self.__duplication_remove(self.params, self.grads)
                    self.__sgd__(params, grads, eta)

        # save model
        self.__save__()

    def __sgd__(self, params, grads, eta):
        """
        eta    : training rate
        params : train parameter
        grads  : gradient parameter

        train by sgd
        """

        for param, grad in zip(params, grads):
            param -= eta * grad

    def __contexts_and_word(self, sentence: list):
        """
        sentence : word in list of one sentence
        ex.) ["I", "have", "a", "pen"]

        return list about contexts and words
        """

        contexts_list = []
        words = []
        for index, word in enumerate(sentence):
            s = index - self.window
            e = index + self.window
            # add context
            contexts = []
            for i in range(s, e):
                if i < 0 or i >= len(sentence):
                    contexts.append(-1)
                elif i == index:
                    pass
                else:
                    contexts.append(self.vocab.index(sentence[i]))
            contexts_list.append(contexts)
            # add target word
            words.append(self.vocab.index(word))

        return contexts_list, words

    def __duplication_remove(self, params_orig, grads_orig):
        """
        params_orig : numpy matrix of params
        grads_orig  : numpy matrix of grads

        remove duplication value
        ref : https://github.com/oreilly-japan/deep-learning-from-scratch-2/blob/master/common/trainer.py
        """

        params, grads = params_orig[:], grads_orig[:]
        flag = True

        while flag:
            flag = False
            length = len(params)

            for i in range(length - 1):
                for j in range(i + 1, length):
                    # duplicate value
                    if params[i] is params[j]:
                        grads[i] += grads[j]
                        flag = True
                        params.pop(j)
                        grads.pop(j)
                    elif params[i].ndim == 2 and params[j].ndim == 2 and \
                            params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                        grads[i] += grads[j].T
                        flag = True
                        params.pop(j)
                        grads.pop(j)
                    if flag:
                        break
                if flag:
                    break
        return params, grads

    def __save__(self):
        """
        save training model as .npy file
        """

        np.save("word2vec_model.npy", self.wv)

    def load(self, name: str="word2vec_model.npy"):
        """
        load training model
        """

        self.wv = np.load(name)

    def similar_word(self, word: str, num: int=10):
        """
        word : word in vocab
        num  : number to show

        show words similar to word of argument
        """

        if word not in self.vocab:
            print("この単語は学習されていません")
            return
        vec_dict = {}
        for i, wordvec in enumerate(self.wv):
            vec_dict[self.vocab[i]] = self.degree_of_similarity(word, self.vocab[i])
        for k, v in sorted(vec_dict.items(), key=lambda x: -x[1])[:num]:
            print(k + "\t" + str(v))

    def degree_of_similarity(self, word1: str, word2: str):
        """
        word1 : word in vocab
        word2 : word in vocab
        degree of similarity between two words
        """

        if word1 not in self.vocab or word2 not in self.vocab:
            print("この単語は学習されていません")
            return
        v1 = self.wv[self.vocab.index(word1)]
        v2 = self.wv[self.vocab.index(word2)]
        return self.__cos_sim(v1, v2)

    def __cos_sim(self, v1, v2):
        """
        v1 : vector
        v2 : vector

        return cosine similarity between 2 vectors
        """

        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


class Skipgram(Word2Vec):
    """
    skipgram model
    """

    def __set__(self, hidden_w, output_w):
        self.inLayer = MatrixMultiply(hidden_w)
        self.outLayer = MatrixMultiply(output_w)
        self.lossLayer = [SoftMaxWithError() for i in range(self.window * 2)]
        layers = [self.inLayer, self.outLayer]
        for layer in layers:
            self.params.append(layer.params)
            self.grads.append(layer.grads)

    def __forward__(self, contexts, word):
        h = self.inLayer.forward(self.one_hot[word])
        s = self.outLayer.forward(h)
        loss = sum([l.forward(s, self.one_hot[i])
                    for l, i in zip(self.lossLayer, contexts) if i != -1])
        return loss

    def __backward__(self, dout=1):
        ds = sum([l.backward(dout) for l in self.lossLayer if l.t is not None])
        dh = self.outLayer.backward(ds)
        self.inLayer.backward(dh)


class Cbow(Word2Vec):
    """
    cbow model
    """

    def __set__(self, hidden_w, output_w):
        self.inLayer = [MatrixMultiply(hidden_w) for i in range(self.window)]
        self.outLayer = MatrixMultiply(output_w)
        self.lossLayer = SoftMaxWithError()
        layers = self.inLayer + [self.outLayer]
        for layer in layers:
            self.params.append(layer.params)
            self.grads.append(layer.grads)

    def __forward__(self, contexts, word):
        h = sum([l.forward(self.one_hot[i])
                 for l, i in zip(self.inLayer, contexts) if i != 1])
        h /= len(self.inLayer)
        s = self.outLayer.forward(h)
        loss = self.lossLayer.forward(s, self.one_hot[word])
        return loss

    def __backward__(self, dout=1):
        ds = self.lossLayer.backward(dout)
        da = self.outLayer.backward(ds) / len(self.inLayer)
        for l in self.inLayer:
            l.backward(da)
