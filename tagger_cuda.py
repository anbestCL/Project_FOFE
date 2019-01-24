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
import pickle
import copy


###### MAIN PROCESSING #############

class Tagger:

    def __init__(self, modelname, datafile, num_epochs, batch_size, **kwargs):
        print("-------------------------------------------------------------------------")
        print("Training BIOS tagger on", datafile,
              "data set using", modelname, "model")
        print("-------------------------------------------------------------------------")
        print("Model parameters: \n number of epochs:", num_epochs, "\n batch size:", args.batch_size, " \n embedding size:",
              kwargs['embedding_size'], "\n hidden size:", kwargs['hidden_size'], "\n dropout rate:", kwargs['dropout_rate'], "\n learning rate:", kwargs['learn_rate'], "\n regularisation factor:", kwargs['reg_factor'])

        self.data = DataPrep(datafile, args.batch_size, modelname)
        self.numlabels = len(self.data.label_to_id)
        self.vocabsize = len(self.data.word_to_id)
        self.vocab_char_size = len(self.data.vocab_char)
        self.modelname = modelname

    def print(self, model):
        for name, param in model.named_parameters():
            if name == "fofe.forgetting_factor":
                print("\t", name, "\t", param.data)

    def train(self, num_epochs, **config):
        begin = time.time()
        if self.modelname == "FOFE":
            model = FOFE_GRU(self.vocab_char_size, config['hidden_size'],
                             config['dropout_rate'], self.numlabels)
        elif self.modelname == "Classic":
            model = Classic_GRU(self.vocabsize, config['embedding_size'],
                                config['hidden_size'], config['dropout_rate'], self.numlabels)
        if torch.cuda.is_available():
            model.cuda()
        # to ignore padded_elements in loss calculation
        criterion = nn.CrossEntropyLoss(size_average=True, ignore_index=0)
        optimizer = optim.Adam(model.parameters(),
                               lr=config['learn_rate'], weight_decay=config['reg_factor'])
        best_acc = 0
        best_f1_macro = 0
        best_f1_weighted = 0
        metrics = []
        pred_labels = np.array([], dtype=int)
        act_labels = np.array([], dtype=int)
        for epoch in range(num_epochs[-1]+1):
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
            train_loss = loss_accum/len(self.data.train_input)
            print("\t train loss: \t", train_loss)
            if epoch in num_epochs:
                self.print(model)
                dev_loss_accum = 0.0
                for batch, (labels, lengths) in zip(self.data.dev_input, self.data.dev_labels):
                    if torch.cuda.is_available():
                        labels = labels.cuda()
                    output_dev = model.forward(batch)
                    loss_dev = criterion(output_dev, labels)
                    dev_loss_accum += loss_dev.data.item()

                    act_labels = np.append(
                        act_labels, labels.data.cpu().numpy().flatten())
                    _, preds = output_dev.max(dim=1)
                    pred_labels = np.append(
                        pred_labels, preds.data.cpu().numpy().flatten())

                dev_loss = dev_loss_accum/len(self.data.dev_input)
                acc, f1_macro, f1_weighted = self.eval(pred_labels, act_labels)
                if acc > best_acc:
                    best_acc = acc
                    best_model = copy.deepcopy(model.cpu())
                    torch.save(best_model, "params.nnp")
                    if torch.cuda.is_available:
                        model = model.cuda()
                if f1_macro > best_f1_macro:
                    best_f1_macro = f1_macro
                if f1_weighted > best_f1_weighted:
                    best_f1_weighted = f1_weighted
                metrics.append((epoch, train_loss, dev_loss,
                                acc, f1_macro, f1_weighted))

                print("\t dev loss \t", )
                print("\t acc \t", best_acc)
                print("\t f1 macro \t", best_f1_macro)
                print("\t f1 weighted \t", best_f1_weighted)
        with open("metrics.txt", "wb") as metric_f:
            pickle.dump(metrics, metric_f)
        end = (time.time() - begin)/3600
        print("-------------------------------------------------------------------------")
        print("Training lasted for", end, "hours")

    def eval(self, pred_labels, act_labels):

        # calculating acc
        acc = sum([1 for a, a2 in zip(pred_labels, act_labels)
                   if a == a2])/len(pred_labels)

        # calculating f1-score
        # ignore Pad token in act_vals:
        def ignore_0(l):
            lab = np.unique(l)
            index = [0]
            lab = np.delete(lab, index)
        f1_macro = f1_score(
            act_labels, pred_labels, average='macro', labels=ignore_0(act_labels))

        f1_weighted = f1_score(
            act_labels, pred_labels, average='weighted', labels=ignore_0(act_labels))

        return acc, f1_macro, f1_weighted

    def predict(self, path, sent, sent_char, labels):
        model = torch.load(path)
        output_test = model.forward(sent_char)
        _, pred_labels = output_test.max(dim=1)
        pred_labels = pred_labels.data.cpu().numpy()
        labels = labels.data.cpu().numpy()

        id_to_label = {id: label for label,
                       id in self.data.label_to_id.items()}
        # print(id_to_label[6])
        id_to_word = {id: word for word, id in self.data.word_to_id.items()}
        for i, word in enumerate(sent):
            print(id_to_word[sent[i]], id_to_label[labels[i]],
                  id_to_label[pred_labels[0][i]])


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
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout rate')
parser.add_argument('--num_epochs', nargs='+', type=int, default=[0, 1, 5, 10],
                    help='number of epochs used for training')
parser.add_argument('--learn_rate', type=float,
                    default=0.001, help='learning rate for model')
parser.add_argument('--reg_factor', type=float, default=0.0,
                    help='weight decay factor of model')

args = parser.parse_args()

tagger = Tagger(args.modelname, args.datafile, args.num_epochs, args.batch_size, embedding_size=args.embedding_size,
                hidden_size=args.hidden_size, dropout_rate=args.dropout, learn_rate=args.learn_rate, reg_factor=args.reg_factor)
tagger.train(args.num_epochs, embedding_size=args.embedding_size,
             hidden_size=args.hidden_size, dropout_rate=args.dropout, learn_rate=args.learn_rate, reg_factor=args.reg_factor)

""" test_sent = tagger.data.test_sents[0]
test_sents_char = tagger.data.test_input[0][0][0]
test_sents_l = tagger.data.test_input[0][1][0].data.cpu().numpy()
test_sents_l = torch.tensor(np.array([test_sents_l]))
test_labels = tagger.data.test_labels[0][0][0]

tagger.predict("params.nnp",test_sent, ([test_sents_char],test_sents_l), test_labels) """
