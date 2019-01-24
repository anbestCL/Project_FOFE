#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import string
from fofe_prep_class import DataPrep
from fofe_model_cuda import FOFE_Encoding, FOFE_GRU
from classic_model_cuda import Classic_GRU
from sklearn.metrics import f1_score
import time
import argparse
import pickle
import copy
from collections import defaultdict
import random


###### MAIN PROCESSING #############

class Tagger:

    def __init__(self, modelname, datafile, num_epochs, batch_size, **kwargs):

        self._print_header(modelname, datafile, num_epochs,
                           batch_size, **kwargs)
        self.data = DataPrep(datafile, batch_size, modelname)
        self.numlabels = len(self.data.label_to_id)
        self.vocabsize = len(self.data.word_to_id)
        self.vocab_char_size = len(self.data.vocab_char)
        self.modelname = modelname

    def _print_header(self, modelname, datafile, num_epochs, batch_size, **kwargs):
        print("-------------------------------------------------------------------------")
        print("Training BIOS tagger on", datafile,
              "data set using", modelname, "model")
        print("-------------------------------------------------------------------------")
        print("Model parameters: \n number of epochs:", num_epochs, "\n batch size:", batch_size, " \n embedding size:",
              kwargs['embedding_size'], "\n hidden size:", kwargs['hidden_size'], "\n dropout rate:", kwargs['dropout'], "\n learning rate:", kwargs['learn_rate'], "\n regularisation factor:", kwargs['reg_factor'])

    def _print(self, model):
        for name, param in model.named_parameters():
            if name == "fofe.forgetting_factor":
                print("\t", name, "\t", param.data)

    def train(self, num_epochs, seed, **config):
        random.seed(seed)
        begin = time.time()
        if self.modelname == "FOFE":
            model = FOFE_GRU(self.vocab_char_size, config['hidden_size'],
                             config['dropout'], self.numlabels)
        elif self.modelname == "Classic":
            model = Classic_GRU(self.vocabsize, config['embedding_size'],
                                config['hidden_size'], config['dropout'], self.numlabels)
        if torch.cuda.is_available():
            model.cuda()
        # to ignore padded_elements in loss calculation
        self.criterion = nn.CrossEntropyLoss(size_average=True, ignore_index=0)
        optimizer = optim.Adam(model.parameters(),
                               lr=config['learn_rate'], weight_decay=config['reg_factor'])
        best_metrics = defaultdict(float)
        metrics = defaultdict()
        for epoch in range(num_epochs[-1]+1):
            print(
                "-------------------------------------------------------------------------")
            print("epoch:", epoch)

            loss_accum = 0.0
            train_data = list(
                zip(self.data.train_input, self.data.train_labels))
            random.shuffle(train_data)
            for batch, (labels, lengths) in train_data:
                if torch.cuda.is_available():
                    labels = labels.cuda()
                optimizer.zero_grad()
                output = model.forward(batch)
                loss = self.criterion(output, labels)
                loss_accum += loss.data.item()
                loss.backward()
                optimizer.step()    # Does the update
            train_loss = loss_accum/len(self.data.train_input)
            print("\t train loss: \t", train_loss)
            if epoch in num_epochs:
                self._print(model)
                dev_loss, acc, f1_macro, f1_weighted = self._eval(
                    model, epoch, (self.data.dev_input, self.data.dev_labels), "dev", best_metrics)
                test_loss = self._eval(
                    model, epoch, (self.data.test_input, self.data.test_labels), "test", best_metrics)
                cur_model = copy.deepcopy(model.cpu())
                if torch.cuda.is_available():
                    model = model.cuda()
                metrics[epoch] = (cur_model, train_loss, dev_loss,
                                  test_loss, acc, f1_macro, f1_weighted)

        with open("metrics.txt", "wb") as metric_f:
            pickle.dump(metrics, metric_f)
        end = (time.time() - begin)/3600
        print("-------------------------------------------------------------------------")
        print("Training lasted for", end, "hours")
        return metrics

    def _eval(self, model, epoch, data, type, best_metrics):

        sents, labels = data
        pred_labels = np.array([], dtype=int)
        act_labels = np.array([], dtype=int)
        loss_accum = 0.0
        for batch, (labels, lengths) in zip(sents, labels):
            if torch.cuda.is_available():
                labels = labels.cuda()
            output = model.forward(batch)
            loss = self.criterion(output, labels)
            loss_accum += loss.data.item()

            act_labels = np.append(
                act_labels, labels.data.cpu().numpy().flatten())
            _, preds = output.max(dim=1)
            pred_labels = np.append(
                pred_labels, preds.data.cpu().numpy().flatten())

        loss = loss_accum/len(self.data.dev_input)

        if type == "dev":
            acc, f1_macro, f1_weighted = self.calc_metric(
                pred_labels, act_labels)
            if f1_macro > best_metrics['f1_macro']:
                best_metrics['f1_macro'] = f1_macro
            if f1_weighted > best_metrics['f1_weighted']:
                best_metrics['f1_weighted'] = f1_weighted
            if acc > best_metrics['acc']:
                best_metrics['acc'] = acc
                best_model = copy.deepcopy(model.cpu())
                torch.save(best_model, "params.nnp")
                if torch.cuda.is_available():
                    model = model.cuda()

            print("\t dev loss \t", loss)
            print("\t acc \t", best_metrics['acc'])
            print("\t f1 macro \t", best_metrics['f1_macro'])
            print("\t f1 weighted \t", best_metrics['f1_weighted'])

        return (loss, acc, f1_macro, f1_weighted) if type == "dev" else loss

    def calc_metric(self, pred_labels, act_labels):

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


if __name__ == '__main__':

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
                    hidden_size=args.hidden_size, dropout=args.dropout, learn_rate=args.learn_rate, reg_factor=args.reg_factor)
    tagger.train(args.num_epochs, seed=0, embedding_size=args.embedding_size,
                 hidden_size=args.hidden_size, dropout=args.dropout, learn_rate=args.learn_rate, reg_factor=args.reg_factor)

    """ test_sent = tagger.data.test_sents[0]
    test_sents_char = tagger.data.test_input[0][0][0]
    test_sents_l = tagger.data.test_input[0][1][0].data.cpu().numpy()
    test_sents_l = torch.tensor(np.array([test_sents_l]))
    test_labels = tagger.data.test_labels[0][0][0]

    tagger.predict("params.nnp",test_sent, ([test_sents_char],test_sents_l), test_labels) """
