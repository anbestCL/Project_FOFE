#!/usr/bin/python3

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
from fofe_model_cuda import FOFE_Encoding, FOFE_GRU
from classic_model_cuda import Classic_GRU
from sklearn.metrics import f1_score
import time
import argparse


###### MAIN PROCESSING #############

class Tagger:

    def __init__(self, args):
        print("-------------------------------------------------------------------------")
        print("Training BIOS tagger on", args.datafile,
              "data set using", args.modelname, "model")
        print("-------------------------------------------------------------------------")
        print("Model parameters: \n number of epochs:", args.num_epochs, "\n batch size:", args.batch_size, " \n embedding size:",
              args.embedding_size, "\n hidden size:", args.hidden_size, "\n learning rate:", args.learn_rate, "\n regularisation factor:", args.reg_factor)
        characters = string.printable + "äöüÄÖÜßéàâçèêôóáëãîÀÂÉÈÊÔÓÁЁÃÎ" + '§'
        vocab_char_size = len(['<PAD>'] + sorted(characters))

        self.data = DataPrep(args.datafile, args.batch_size, args.modelname)
        numlabels = len(self.data.label_to_id)
        vocabsize = len(self.data.word_to_id)

        self.train(args.modelname, vocabsize, vocab_char_size, args.embedding_size,
                   args.hidden_size, args.num_epochs, numlabels, args.learn_rate, args.reg_factor)

    def print(self, model):
        for name, param in model.named_parameters():
            if name == "fofe.forgetting_factor":
                print("\t", name, "\t", param.data)

    def train(self, modelname, vocab_size, vocab_char_size, embedding_size, hidden_size, num_epochs, num_labels, learn_rate, reg_factor):
        begin = time.time()
        if modelname == "FOFE":
            model = FOFE_GRU(vocab_char_size, hidden_size, num_labels)
        elif modelname == "Classic":
            model = Classic_GRU(vocab_size, embedding_size,
                                hidden_size, num_labels)
        if torch.cuda.is_available():
            model.cuda()
        # to ignore padded_elements in loss calculation
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(model.parameters(),
                               lr=learn_rate, weight_decay=reg_factor)
        best_acc = 0
        best_f1_macro = 0
        best_f1_micro = 0
        best_loss = 10
        for epoch in range(num_epochs):
            print(
                "-------------------------------------------------------------------------")
            print("epoch:", epoch)
            loss_accum = 0.0
            for batch, (labels, lengths) in zip(self.data.train_input, self.data.train_labels):
                if torch.cuda.is_available():
                    labels = labels.cuda()
                optimizer.zero_grad()
                output = model.forward(batch)
                loss = criterion(output, labels)
                loss_accum += loss.data.item()
                loss.backward()
                optimizer.step()    # Does the update
            print("\t train loss: \t", loss_accum/len(self.data.train_input))
            if epoch % 100 == 0:
                self.print(model)
                dev_loss_accum = 0.0
                for batch, (labels, lengths) in zip(self.data.dev_input, self.data.dev_labels):
                    if torch.cuda.is_available():
                        labels = labels.cuda()
                    output_dev = model.forward(batch)
                    loss_dev = criterion(output_dev, labels)
                    dev_loss_accum += loss_dev.data.item()
                    acc, f1_macro, f1_micro = self.eval(output_dev, labels)
                    if acc > best_acc:
                        best_acc = acc
                    if f1_macro > best_f1_macro:
                        best_f1_macro = f1_macro
                    if f1_micro > best_f1_micro:
                        best_f1_micro = f1_micro

                print("\t dev loss \t", dev_loss_accum/len(self.data.dev_input))
                print("\t acc \t", best_acc)
                print("\t f1 macro \t", best_f1_macro)
                print("\t f1 micro \t", best_f1_micro)
        end = (time.time() - begin)/3600
        print("-------------------------------------------------------------------------")
        print("Training lasted for", end, "hours")

    def eval(self, output, act_labels):
        _, pred_labels = output.max(dim=1)
        pred_labels = pred_labels.cpu().data.numpy()
        act_labels = act_labels.cpu().data.numpy()

        # calculating acc
        comp = [sum([1 for a, a2 in zip(sent, sent2) if a == a2])/len(sent)
                for sent, sent2 in zip(pred_labels, act_labels)]
        acc = sum(comp)/len(act_labels)

        # calculating f1-score
        # ignore Pad token in act_vals:
        def ignore_0(l):
            lab = np.unique(l)
            index = [0]
            lab = np.delete(lab, index)
        f1_macro = [f1_score(
            act_vals, pred_vals, average='macro', labels=ignore_0(act_vals)) for act_vals, pred_vals in zip(act_labels, pred_labels)]
        f1_macro = sum(f1_macro)/len(f1_macro)

        f1_micro = [f1_score(
            act_vals, pred_vals, average='micro', labels=ignore_0(act_vals)) for act_vals, pred_vals in zip(act_labels, pred_labels)]
        f1_micro = sum(f1_micro)/len(f1_micro)

        return acc, f1_macro, f1_micro


parser = argparse.ArgumentParser(
    description='Training program of the BIOS Tagger.')
parser.add_argument('modelname', type=str,
                    help='type of model to be trained: FOFE encodings or Classic embeddings')
parser.add_argument('datafile', type=str,
                    help='file or folder containing the data')
parser.add_argument('--batch_size', type=int, default=8,
                    help='size of the data batches')
parser.add_argument('--embedding_size', type=int, default=100,
                    help='size of the word embeddings when using ')
parser.add_argument('--hidden_size', type=int, default=100,
                    help='size of hidden states of GRU')
parser.add_argument('--num_epochs', type=int, default=1000,
                    help='number of epochs used for training')
parser.add_argument('--learn_rate', type=float,
                    default=0.001, help='learning rate for model')
parser.add_argument('--reg_factor', type=float, default=0.001,
                    help='weight decay factor of model')

args = parser.parse_args()

Tagger(args)
