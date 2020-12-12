from flask import Flask, jsonify
from train_rnn import config, RNN, Utils
import pickle
import torch
import numpy as np
import os
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
all_categories = ['sv', 'da', 'no']

U = Utils(n_words, all_categories, Word2Index_w_unk)

def predict_sentence(input_line):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output, _ = U.evaluate(U.SentToTensor(input_line), rnn)
        output = np.squeeze(output.numpy())
        pred_ix = np.argmax(output)
        prediction = all_categories[pred_ix]
        print("The following sentence is: [" + prediction + "]")
    return prediction
@app.route('/')
def index():
    return 'Hi! to classify a sentence write in http://0.0.0.0:5000/predict/sentence to predict'

@app.route('/predict')
def pred_message():
    return 'Hi! to classify a sentence write in http://0.0.0.0:5000/predict/sentence to predict'

@app.route('/predict/<string:sentence>')
def predict(sentence):
    # Predict language class
    language = predict_sentence(sentence)
    return jsonify({'language': language})

if __name__ == '__main__':
    app.run(host='0.0.0.0')