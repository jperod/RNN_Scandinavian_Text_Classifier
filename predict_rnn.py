from RNN import RNN
from Utils import Utils
import pickle
import torch
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--string', type=str, help='enter string to predict')
parser.add_argument('--example', action='store_true', help='example of predictions made by the model')
args = parser.parse_args()
#
vocab_dir = 'saves/save_hn_256_lr_0.005/vocab.txt'
saved_model_dir = 'saves/save_hn_256_lr_0.005/saved_model.pth'

with open(vocab_dir, 'rb') as f:
    Word2Index = pickle.load(f)
# Point unknown token to Word2Index with index 0:
Word2Index_w_unk = Word2Index.copy()
Word2Index_w_unk['<UNK>'] = 0

n_words = len(Word2Index)
rnn = RNN(len(Word2Index), 256, 3)
rnn.load_state_dict(torch.load(saved_model_dir))
all_categories = ['da', 'no', 'sv']

U = Utils(n_words, all_categories, Word2Index_w_unk)

def predict(input_line):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output, _ = U.evaluate(U.SentToTensor(input_line), rnn)
        output = np.squeeze(output.numpy())
        pred_ix = np.argmax(output)
        prediction = all_categories[pred_ix]
        print("The following sentence is: [" + prediction + "]")

if  args.example:
    print('\nTesting on dataset sentences:')
    predict('Hold nu op, hun har det skidt') #DA
    predict('Jeg har akkurat bakt en sukkerkake') #NO
    predict('Man känner igen den, den är bekväm.') #SV

    print('\nTesting on random sentences from the internet:')
    predict('Hej, jeg hedder Pedro og jeg elsker at drikke øl!') #DA
    predict('Mit luftpudefartøj er fyldt med ål') #DA
    predict('Der er i øjeblikket ingen tekst på denne side. Du kan søge efter sidenavnet på andre sider, søge i relaterede logger eller oprette siden. ') #DA
    predict('Jeg forstår det veldig godt.') #NO
    predict('Floreanaspottefugl er ein sterkt truga art av spottefuglar. Han er naturleg endemisk til øya Floreana, ei av Galápagosøyane.')  # NO
    predict('När katten är borta dansar råttorna på bordet') #SV
    predict('Rosshavet (engelska: Ross Sea) är ett randhav av Antarktiska oceanen och ligger mellan Victoria Land och Marie Byrd Land') #SV
else:
    predict(args.string)