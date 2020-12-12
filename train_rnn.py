from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
from nltk.tokenize import RegexpTokenizer
rTokenizer = RegexpTokenizer(r'\w+')
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import time
import math
import torch.nn as nn
import torch
import random
from sklearn.feature_extraction.text import CountVectorizer

""" Configuration """
config = {
    "n_hidden": 256,
    "learning_rate": 0.005, # If you set this too high, it might explode. If too low, it might not learn
    "n_iters": 1000000, #10 x datasize corresponding to ~10 epochs | Only the best Model (Highest Val Acc) is saved.
    "print_every": 1000,
    "data_size": 100000, # Number of sentences to extract. balanced amounts of Da, Sv and No sentences.
    "data_dir": "datasets/OpenSubs/*.txt", #
    "lr_decay": 0.99, #Exponential lr decay: decrease lr by 1% every 10,000 training iterations. This helps prevent vanishing/exploding gradients.
    "train_from_savedmodel": False, #If model was saved, it can be loaded to continue training from checkpoint
    "saved_model_dir": "save_hn_256_lr_0.005" #Directory of saved model
}

class Utils():
    def __init__(self):
        # super(Utils, self).__init__()
        self.n_words = None
        self.all_categories = []
        self.all_sentences = []
        self.all_labels = []
        self.Word2Index = None
        self.n_categories = None

    def findFiles(self, data_dir):
        return glob.glob(data_dir)

    # Read a file and split into lines
    def readLines(self, filename, lim):
        f = open(filename, "r", encoding='utf-8')
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

        index = self.Word2Index[word.lower()]
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

        for filename in files:
            category = os.path.basename(filename)[3:5]
            self.all_categories.append(category)
            sentences = self.readLines(filename, int(data_size / 3))
            category_lines[category] = sentences
            self.all_sentences.extend(sentences)
            self.all_labels.extend([category for i in range(len(sentences))])

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

    def train(self, category_tensor, line_tensor, iter, rnn):
        hidden = rnn.initHidden()

        rnn.zero_grad()

        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        criterion = nn.NLLLoss()
        loss = criterion(output, category_tensor)
        loss.backward()

        # Add parameters' gradients to their values, multiplied by learning rate with exponential decay
        # reduce 0.8 every 10k steps
        lr = config["learning_rate"] * pow(config["lr_decay"], (iter / 10000))

        for p in rnn.parameters():
            p.data.add_(p.grad.data, alpha=-lr)

        return output, rnn


    def tokenizer(self, sent):
        tokens = rTokenizer.tokenize(sent)
        tokens = [token.lower() for token in tokens]
        return tokens

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def main():
    start = time.time()
    U = Utils()

    n_words, n_categories, Word2Index, df_train, df_val = U.load_data(data_size = config["data_size"], data_dir = config["data_dir"])

    #Initialize RNN model:
    rnn = RNN(n_words, config["n_hidden"], n_categories)
    #If training from checkpoint, load model state_dict here:
    if config["train_from_savedmodel"]:
        save_dir = "saves/save_hn_" + str(config["n_hidden"]) + "_lr_" + str(config["learning_rate"])
        saved_model_dir = save_dir + "/saved_model.pth"
        rnn.load_state_dict(torch.load(saved_model_dir))

    #Initialize empty lists to record training moving average (5000) accuracy on training and validation datasets
    train_scores_right, train_scores_wrong, val_scores_right, val_scores_wrong = [], [], [], []
    best_acc_val = 0
    if not config["train_from_savedmodel"]:
        try:
            os.makedirs("saves/save_hn_"+str(config["n_hidden"])+"_lr_"+str(config["learning_rate"]))
        except OSError as e:
            raise("Save directory: save_hn_XXX_lr_x.xxxx_already exists. delete it to start training from scratch")
        with open("saves/save_hn_" + str(config["n_hidden"]) + "_lr_" + str(config["learning_rate"]) + "/vocab.txt", "wb") as f:
            pickle.dump(Word2Index, f)

    for iter in range( config["n_iters"]):
        category, line, category_tensor, line_tensor = U.randomTrainingExample(df_train)
        output, rnn = U.train(category_tensor, line_tensor, iter, rnn)
        guess, _ = U.categoryFromOutput(output)
        if guess == category:
            train_scores_right.append(1)
            train_scores_wrong.append(0)
        else:
            train_scores_wrong.append(1)
            train_scores_right.append(0)
        acc_train = round(100*sum(train_scores_right)/(sum(train_scores_right)+sum(train_scores_wrong)),1)

        category_val, _, category_tensor_val, line_tensor_val = U.randomTrainingExample(df_val)
        output_val, rnn = U.evaluate(line_tensor_val, rnn)
        guess_val, _ = U.categoryFromOutput(output_val)
        if guess_val == category_val:
            val_scores_right.append(1)
            val_scores_wrong.append(0)
        else:
            val_scores_right.append(0)
            val_scores_wrong.append(1)
        acc_val = round(100*sum(val_scores_right)/(sum(val_scores_right)+sum(val_scores_wrong)),1)

        # Print details and show the example of a training prediction
        if iter % config["print_every"] == 0 and iter >= 100:
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) | %s / %s %s' % (iter, iter / config["n_iters"] * 100, U.timeSince(start), line, guess, correct))
            print('Train Accuracy: ' + str(acc_train) + '% | Validation Accuracy: ' + str(acc_val) + '%')

        # Save model parameters for best model, min val accuracy of 65% allowed before starting saves
        if iter > 5000 and (acc_val > best_acc_val) and acc_val > 65:
            #Only start saving after 66+ validation accuracy
            best_acc_val = acc_val
            torch.save(rnn.state_dict(), "saves/save_hn_"+str(config["n_hidden"])+"_lr_"+str(config["learning_rate"])+"/saved_model.pth")
            print("New Best Model with validation accuracy = "+str(best_acc_val)+"! Saving model parameters...")
        train_scores_right = train_scores_right[-5000:]
        train_scores_wrong = train_scores_wrong[-5000:]
        val_scores_right = val_scores_right[-5000:]
        val_scores_wrong = val_scores_wrong[-5000:]

if __name__ == '__main__':
    main()