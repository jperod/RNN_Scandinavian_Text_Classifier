from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
from nltk.tokenize import RegexpTokenizer
rTokenizer = RegexpTokenizer(r'\w+')
from sklearn.model_selection import train_test_split
import pandas as pd
import time
import math
import torch.nn as nn
import torch
import random
from sklearn.feature_extraction.text import CountVectorizer

class Utils():
    def __init__(self, n_words=None, all_categories=[], Word2Index=None):
        # super(Utils, self).__init__()
        self.n_words = n_words
        self.all_categories = all_categories
        self.all_sentences = []
        self.all_labels = []
        self.Word2Index = Word2Index
        self.n_categories = 3

    def findFiles(self, data_dir):
        list_files = os.listdir(os.path.dirname(os.path.realpath(__file__))+"/"+data_dir)
        list_files = [f for f in list_files if f[0:13] == "OpenSubtitles"]
        return list_files

    # Read a file and split into lines
    def readLines(self, filename, lim):
        f = open(os.path.dirname(os.path.realpath(__file__))+"/"+filename, "r", encoding='utf-8')
        c = 0
        sentences = []
        while True:
            # read line
            line = f.readline().rstrip()
            if len(line) > 15 & len(line) < 100:
                sentences.append(line)
                c += 1
            # if not line:
            if c == lim:
                break
        f.close()
        return sentences

    # Find letter index from all_letters, e.g. "a" = 0
    def WordToIndex(self, word):
        try:
            index = self.Word2Index[word.lower()]
        except:
            index = self.Word2Index['<UNK>']
        return index


    # Just for demonstration, turn a letter into a <1 x n_letters> Tensor
    def WordToTensor(self, word):
        tensor = torch.zeros(1, self.n_words)
        tensor[0][self.WordToIndex(word)] = 1
        return tensor

    # or an array of one-hot letter vectors
    def SentToTensor(self, sent):
        # Remove punctuation in this tokenizer, words only
        sent = self.tokenizer(sent)
        sent = [word.lower() for word in sent]
        # sent = [word for word in sent if word in ]
        tensor = torch.zeros(len(sent), 1, self.n_words)
        for li, word in enumerate(sent):
            tensor[li][0][self.WordToIndex(word)] = 1
        return tensor

    def categoryFromOutput(self, output):
        top_n, top_i = output.topk(1)
        category_i = top_i[0].item()
        return self.all_categories[category_i], category_i

    def load_data(self, data_size, data_dir):
        self.data_size = data_size
        self.data_dir = data_dir
        files = self.findFiles(data_dir=data_dir)
        # Build the category_lines dictionary, a list of names per language
        category_lines = {}
        self.all_categories = ['da', 'no', 'sv']

        sentences_da = self.readLines(data_dir+"/OpenSubtitles.da-en.da", int(data_size / 3))
        self.all_sentences.extend(sentences_da)
        self.all_labels.extend(["da" for i in range(len(sentences_da))])
        sentences_no = self.readLines(data_dir+"/OpenSubtitles.en-no.no", int(data_size / 3))
        self.all_sentences.extend(sentences_no)
        self.all_labels.extend(["no" for i in range(len(sentences_no))])
        sentences_sv = self.readLines(data_dir+"/OpenSubtitles.en-sv.sv", int(data_size / 3))
        self.all_sentences.extend(sentences_sv)
        self.all_labels.extend(["sv" for i in range(len(sentences_sv))])
        print(self.all_categories)

        df = pd.DataFrame([self.all_sentences, self.all_labels]).T
        df_train, df_val = train_test_split(df, shuffle=True, test_size=min(0.15, 10000 / len(self.all_sentences)))

        # Max val size of 10000. Don't need more really for validation purposes
        print("Sentences used for training: " + str(df_train.shape[0]))
        print("Sentences used for validation: " + str(df_val.shape[0]))
        print(df_val.shape)

        vectorizer = CountVectorizer(tokenizer=self.tokenizer)
        vectorizer.fit_transform(self.all_sentences)
        self.Word2Index = vectorizer.vocabulary_
        self.n_words = len(self.Word2Index)
        self.n_categories = len(self.all_categories)

        print(self.WordToTensor('stora').size())

        print(self.SentToTensor('jeg vil vite hva, som skjer med deg.').size())

        print("done")
        return self.n_words, self.n_categories, self.Word2Index, df_train, df_val

    def randomChoice(self, l):
        return l[random.randint(0, len(l) - 1)]

    def randomTrainingExample(self, df):
        sample = df.sample()
        category = sample[1].values[0]
        sent = sample[0].values[0]
        category_tensor = torch.tensor([self.all_categories.index(category)], dtype=torch.long)
        line_tensor = self.SentToTensor(sent)
        return category, sent, category_tensor, line_tensor

    def timeSince(self, since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def evaluate(self, line_tensor, rnn):
        hidden = rnn.initHidden()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        return output, rnn

    def train(self, category_tensor, line_tensor, iter, rnn, lr):
        hidden = rnn.initHidden()

        rnn.zero_grad()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        criterion = nn.NLLLoss()
        loss = criterion(output, category_tensor)
        loss.backward()

        for p in rnn.parameters():
            p.data.add_(p.grad.data, alpha=-lr)

        return output, rnn

    def tokenizer(self, sent):
        tokens = rTokenizer.tokenize(sent)
        tokens = [token.lower() for token in tokens]
        return tokens