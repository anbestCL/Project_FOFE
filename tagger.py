import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import torch.optim as optim
import numpy as np
import string
import math
import json
from fofe_prep_class import DataPrep
from fofe_model import FOFE_Encoding, FOFE_GRU
from classic_model import Classic_GRU
from sklearn.metrics import f1_score


###### MAIN PROCESSING #############

class Tagger:

    def __init__(self, modelname, datafile, batchsize, embeddingsize, hiddensize, numepochs):
        vocab_char_size = len(['<PAD>'] + sorted(string.printable))

        self.data = DataPrep(datafile, batchsize, modelname)
        numlabels = len(self.data.label_to_id)
        vocabsize = len(self.data.word_to_id)

        self.train(modelname, vocabsize, vocab_char_size, embeddingsize,
                   hiddensize, numepochs, numlabels)

    def print(self, model):
        for name, param in model.named_parameters():
            if name == "fofe.forgetting_factor":
                print(name, param.data)

    def train(self, modelname, vocab_size, vocab_char_size, embedding_size, hidden_size, num_epochs, num_labels):
        if modelname == "FOFE":
            model = FOFE_GRU(vocab_char_size, hidden_size, num_labels)
        elif modelname == "Classic":
            model = Classic_GRU(vocab_size, embedding_size,
                                hidden_size, num_labels)
        # to ignore padded_elements in loss calculation
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(model.parameters(),
                               lr=0.0001, weight_decay=0.001)
        best_acc = 0
        best_f1_macro = 0
        best_f1_micro = 0
        best_loss = 10
        for epoch in range(num_epochs):
            print(epoch)
            loss_accum = 0.0
            for batch, (labels, lengths) in zip(self.data.train_input, self.data.train_labels):
                optimizer.zero_grad()
                output = model.forward(batch)
                loss = criterion(output, labels)
                loss_accum += loss.data.item()
                loss.backward()
                optimizer.step()    # Does the update
            print("train loss:", loss_accum/len(self.data.train_input))
            self.print(model)
            if epoch % 2 == 0:
                dev_loss_accum = 0.0
                for batch, (labels, lengths) in zip(self.data.dev_input, self.data.dev_labels):
                    output_dev = model.forward(batch)
                    loss_dev = criterion(output_dev, labels)
                    dev_loss_accum += loss_dev.data.item()
                    acc, f1_macro, f1_micro = self.eval(output_dev, labels)
                    if acc > best_acc:
                        best_acc = acc
                        print("acc", best_acc)
                    if f1_macro > best_f1_macro:
                        best_f1_macro = f1_macro
                        print("f1_macro", best_f1_macro)
                    if f1_micro > best_f1_micro:
                        best_f1_micro = f1_micro
                        print("f1_micro", best_f1_micro)

                print("dev loss", dev_loss_accum/len(self.data.dev_input))

    def eval(self, output, act_labels):
        _, pred_labels = output.max(dim=1)
        pred_labels = pred_labels.cpu().data.numpy()
        act_labels = act_labels.cpu().data.numpy()

        # calculating acc
        comp = [sum([1 for a, a2 in zip(sent, sent2) if a == a2])/len(sent)
                for sent, sent2 in zip(pred_labels, act_labels)]
        acc = sum(comp)/len(act_labels)

        # calculating f1-score
        f1_macro = [f1_score(
            list(act_vals), list(pred_vals), average='macro') for act_vals, pred_vals in zip(act_labels, pred_labels)]
        f1_macro = sum(f1_macro)/len(f1_macro)

        f1_micro = [f1_score(
            act_vals, pred_vals, average='micro') for act_vals, pred_vals in zip(act_labels, pred_labels)]
        f1_micro = sum(f1_micro)/len(f1_micro)

        return acc, f1_macro, f1_micro


Tagger("FOFE", datafile="Atis.json", batchsize=8,
       embeddingsize=100, hiddensize=100, numepochs=1000)
# Tagger("Classic", datafile="Tiger/", batchsize=8,
# embeddingsize=100, hiddensize=100, numepochs=10)
