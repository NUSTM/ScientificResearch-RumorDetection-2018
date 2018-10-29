__author__ = "zhangc0321@gmail.com"

import gensim, smart_open
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import numpy as np
import random
from sklearn.externals import joblib

sources_train = {'1.txt':'ONE.txt', '0.txt':'ZERO.txt'}

def read_corpus(source_set):
    ct = 0
    for source_file, prefix in source_set.items():
        with smart_open.smart_open(source_file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.to_unicode(line).split(), [prefix + '_%s' % i])
train_corpus = list(read_corpus(sources_train))
model = gensim.models.doc2vec.Doc2Vec(dm=1, size=50, window = 2, min_count=1, iter=50, workers=11, alpha=0.025, min_alpha=0.025)
model.build_vocab(train_corpus)
for epoch in range(10):
    print(epoch)
    random.shuffle(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.iter)
    model.alpha -= 0.002
    model.min_alpha = model.alpha
model.save('./my_db.d2v')