import model

with open("sample.txt", "r") as f:
    sentences = [line.replace("\n", "").split(" ") for line in f.readlines()]

w2v = model.Skipgram(sentences, 100)
# w2v = model.Cbow(sentences, 100)
w2v.train(batch_size=8)
# w2v.load()
w2v.similar_word("this")
print(w2v.degree_of_similarity("this", "that"))
