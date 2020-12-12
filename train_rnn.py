from __future__ import unicode_literals, print_function, division
from io import open
import os
from nltk.tokenize import RegexpTokenizer
rTokenizer = RegexpTokenizer(r'\w+')
import pickle
import time
import torch
from RNN import RNN
from Utils import Utils
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nh', type=int, help='number of hidden units', default=256)
parser.add_argument('--lr', type=int, help='learning rate', default=0.005)
parser.add_argument('--lr_d', type=int, help='learning rate decay rate', default=0.97)
parser.add_argument('--e', type=int, help='number of epochs', default=5)
parser.add_argument('--pe', type=int, help='print every n iterations', default=100)
parser.add_argument('--ds', type=int, help='size of data to extract', default=100000)
parser.add_argument('--dir', type=str, help='OS data directory', default="datasets/OpenSubs")
parser.add_argument('--ckp_dir', type=str, help='checkpoint data directory', default="save_hn_256_lr_0.005")
parser.add_argument('--ckp', action='store_true', help='train from saved model', default=False)
args = parser.parse_args()

""" Configuration Dict """
config = {
    "n_hidden": args.nh,
    "learning_rate": args.lr, # If you set this too high, it might explode. If too low, it might not learn
    "learning_rate_decay": args.lr_d,  # percentage to decrease lr by every 10k iterations, default = 0.97 to decay 3% every 10k
    "n_iters": args.ds*args.e, #~number of iterations = epoch number * datasize
    "print_every": args.pe,
    "data_size": args.ds, # Number of sentences to extract. balanced amounts of Da, Sv and No sentences.
    "data_dir": args.dir, #
    "train_from_savedmodel": args.ckp, #If model was saved, it can be loaded to continue training from checkpoint
    "saved_model_dir": args.ckp_dir #Directory of saved model
}

def main():
    start = time.time()

    U = Utils();
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
        output, rnn = U.train(category_tensor, line_tensor, iter, rnn, config['learning_rate'], config['learning_rate_decay'])
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