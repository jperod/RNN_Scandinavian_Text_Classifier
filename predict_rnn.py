from train_rnn import config
from train_rnn import RNN, Utils
import pickle
import torch
import numpy as np

save_dir = "saves/backup_save_hn_256_lr_0.005"
vocab_dir = save_dir + "/vocab.txt"
# saved_model_dir = save_dir + "/saved_model.pth"
saved_model_dir = "saves/backup_save_hn_256_lr_0.005/saved_model.pth"
with open(vocab_dir, "rb") as f:
    Word2Index = pickle.load(f)
Word2Index_w_unk = Word2Index.copy()
#Add unknown token to Word2Index:
ix_unk = 0
Word2Index_w_unk["<UNK>"] = ix_unk

n_words = len(Word2Index)
rnn = RNN(len(Word2Index), config["n_hidden"], 3)
rnn.load_state_dict(torch.load(saved_model_dir))
all_categories = Utils().all_categories

#[p for p in rnn.parameters()]

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = Utils().evaluate(Utils().SentToTensor(input_line, n_words))
        output = np.squeeze(output.numpy())
        pred_ix = np.argmax(output)
        prediction = all_categories[pred_ix]
        print("The following sentence is: [" + prediction + "]")

print('\nTesting on dataset sentences:')
predict('Hold nu op, hun har det skidt') #DA
predict('Jeg har akkurat bakt en sukkerkake') #NO
predict('Man känner igen den, den är bekväm.') #SV

print('\nTesting on random sentences from the internet:')
predict('Mit luftpudefartøj er fyldt med ål') #DA
predict('Der er i øjeblikket ingen tekst på denne side. Du kan søge efter sidenavnet på andre sider, søge i relaterede logger eller oprette siden. ') #DA
predict('Jeg forstår det veldig godt.') #NO
predict('Med et areal på 165,25 millioner kvadratkilometer dekker Stillehavet nesten en tredjedel av jordens overflate') #NO
predict('När katten är borta dansar råttorna på bordet') #SV
predict('Rosshavet (engelska: Ross Sea) är ett randhav av Antarktiska oceanen och ligger mellan Victoria Land och Marie Byrd Land') #SV
