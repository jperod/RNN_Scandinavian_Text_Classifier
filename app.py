from flask import Flask, jsonify, request
import io
import json
from train_rnn import config, RNN, Utils
import pickle
import torch
import numpy as np
import argparse

app = Flask(__name__)
vocab_dir = 'saves/save_hn_256_lr_0.005/vocab.txt'
saved_model_dir = 'saves/save_hn_256_lr_0.005/saved_model.pth'

with open(vocab_dir, 'rb') as f:
    Word2Index = pickle.load(f)
# Point unknown token to Word2Index with index 0:
Word2Index_w_unk = Word2Index.copy()
Word2Index_w_unk['<UNK>'] = 0

n_words = len(Word2Index)
rnn = RNN(len(Word2Index), config['n_hidden'], 3)
rnn.load_state_dict(torch.load(saved_model_dir))
all_categories = ['da', 'no', 'sv']

U = Utils(n_words, all_categories, Word2Index_w_unk)

def predict_sentence(input_line):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output, _ = U.evaluate(U.SentToTensor(input_line), rnn)
        output = np.squeeze(output.numpy())
        pred_ix = np.argmax(output)
        prediction = all_categories[pred_ix]
        print("The following sentence is: [" + prediction + "]")
    return input_line, prediction

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        sentence = file.read()
        sentence, language = predict_sentence(sentence)
        return jsonify({'sentence': sentence, 'language': language})

if __name__ == '__main__':
    app.run()