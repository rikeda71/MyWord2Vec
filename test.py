import model

with open("sample.txt", "r") as f:
    sentences = [line.replace("\n", "").split(" ") for line in f.readlines()]

vocab = list(set(sum(sentences, [])))
w2v = model.Word2Vec(vocab, sentences, 100)
w2v.train()
