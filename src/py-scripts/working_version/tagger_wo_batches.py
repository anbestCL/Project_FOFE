#!/usr/bin/python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import string
from prep_wo_batches import DataPrep
from fofe_model_wo_batches import FOFE_Encoding, FOFE_GRU
from classic_model_wo_batches import Classic_GRU
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import time
import argparse
import pickle
import copy
from collections import defaultdict, Counter
import random
import sklearn


###### MAIN PROCESSING #############

class Tagger:

    """This is the main training class for the tagger: it implements a bidirectional GRU with the Adam optimizer
   (eval function is integrated in train function) and a predict function to print out predicted tag sequences.

    Arguments:
        modelname {string} - either "FOFE" for Fofe character encoding or "Classic" for classic trainable embedding layer
        datafile {string} - path to data
        paramfile {string} - path to save model and metrics to
        batch_size {number} - size of training batches
        kwargs {integers} - keyword arguments dictionary comprising 
                            number of epochs (list of checkpoints at which 
                            to evaluate the model default: [0,50,100]), 
                            embedding dimension (only for "Classic" model, default: 50), 
                            size of hidden layers of GRU (default: 50), 
                            dropout rate (default: 0.5), 
                            learning rate (default: 001) and 
                            regularisation factor (default: 0.0)
    Returns:
        metrics {dict} -- maps evaluation epochs to model, train loss, dev loss, test loss, accuracy, macro and weighted F1 scores
                            used further in hyper optimisation script to select best configuration
    """

    def __init__(self, modelname, datafile, paramfile, num_epochs, **kwargs):

        self._print_header(modelname, datafile, num_epochs,
                           **kwargs)
        self.paramfile = '{}_atis_{}'.format(paramfile, modelname) if datafile.endswith(
            "json") else '{}_tiger_{}'.format(paramfile, modelname)
        self.data = DataPrep(datafile, modelname)
        self.numlabels = len(self.data.label_to_id)
        self.vocabsize = len(self.data.word_to_id)
        self.modelname = modelname

        # if we train FOFE encodings we need the number of characters in the respective language to build the encoding
        if self.modelname == "FOFE":
            self.vocab_char_size = len(self.data.vocab_char)

    # functions to make log more readable
    def _print_header(self, modelname, datafile, num_epochs, **kwargs):
        print("-------------------------------------------------------------------------")
        print("Training BIOS tagger on", datafile,
              "data set using", modelname, "model")
        print("-------------------------------------------------------------------------")
        print("Model parameters: \n number of epochs:", num_epochs, " \n embedding size:",
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
        # to ignore padded_elements in loss calculation set ignore_index = 0 since pad token has index 0
        self.criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(),
                               lr=config['learn_rate'], weight_decay=config['reg_factor'])

        # dictionary holding best current accuracy, macro and weighted F1 scores
        best_metrics = defaultdict(float)

        # dictionary that maps evaluation epochs to model, train loss, dev loss, test loss, accuracy, macro and weighted F1 scores
        # used further in hyper optimisation script to select best configuration
        metrics = defaultdict()

        # to evaluate at all checkpoints given in num_epochs
        for epoch in range(num_epochs[-1]+1):
            print(
                "-------------------------------------------------------------------------")
            print("epoch:", epoch)

            loss_accum = 0.0
            train_data = list(
                zip(self.data.train_input, self.data.train_labels))
            random.shuffle(train_data)  # shuffling of batches
            for sent, labels in train_data:
                optimizer.zero_grad()
                output = model.forward(sent)
                loss = self.criterion(output, labels)
                loss_accum += loss.data.item()
                loss.backward()
                optimizer.step()
            train_loss = loss_accum/len(self.data.train_input)
            print("\t train loss: \t", train_loss)

            # evaluation checkpoint
            if epoch in num_epochs:
                self._print(model)
                dev_loss, acc, f1_macro, f1_weighted = self._eval(
                    model, epoch, (self.data.dev_input, self.data.dev_labels), "dev", best_metrics)

                # calculate test loss for hyper optimisation (have test loss corresponding to best config ready)
                test_loss = self._eval(
                    model, epoch, (self.data.test_input, self.data.test_labels), "test", best_metrics)
                cur_model = copy.deepcopy(model.cpu())
                if torch.cuda.is_available():
                    model = model.cuda()
                metrics[epoch] = (cur_model, train_loss, dev_loss,
                                  test_loss, acc, f1_macro, f1_weighted)

        with open("{}_metrics.txt".format(self.paramfile), "wb") as metric_f:
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
        for sent, label in zip(sents, labels):
            output = model.forward(sent)
            loss = self.criterion(output, label)
            loss_accum += loss.data.item()

            act_labels = np.append(
                act_labels, label.data.cpu().numpy().flatten())
            _, preds = output.max(dim=1)
            pred_labels = np.append(
                pred_labels, preds.data.cpu().numpy().flatten())

        loss = loss_accum/len(sents)

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
                torch.save(best_model, "{}_model.nnp".format(self.paramfile))
                if torch.cuda.is_available():
                    model = model.cuda()

            print("\t dev loss \t", loss)
            print("\t acc \t", best_metrics['acc'])
            print("\t f1 inversely weighted \t", best_metrics['f1_macro'])
            print("\t f1 class weighted \t", best_metrics['f1_weighted'])

        return (loss, acc, f1_macro, f1_weighted) if type == "dev" else loss

    def calc_metric(self, pred_labels, act_labels):

        # calculating acc
        acc = sum([1 for a, a2 in zip(pred_labels, act_labels)
                   if a == a2])/len(pred_labels)

        # calculate inverse weights for classes
        counted_labels = Counter(act_labels)
        counted_labels = {label: 1/count for label,
                          count in counted_labels.items()}
        sample_weights = [counted_labels[label] for label in act_labels]

        # weight classes by inverse frequency
        f1_macro = f1_score(
            act_labels, pred_labels, average='macro', sample_weight=sample_weights)

        f1_weighted = f1_score(
            act_labels, pred_labels, average='weighted')

        return acc, f1_macro, f1_weighted

    def predict(self, path, sent, labels):
        model = torch.load(path)
        output_test = model.forward(sent)
        _, pred_labels = output_test.max(dim=1)
        pred_labels = pred_labels.data.cpu().numpy()
        labels = labels.data.cpu().numpy()

        id_to_label = {id: label for label,
                       id in self.data.label_to_id.items()}
        id_to_word = {id: word for word, id in self.data.word_to_id.items()}
        for i, word in enumerate(sent):
            print(id_to_word[sent[i]], "true label:", id_to_label[labels[i]],
                  "predicted label:", id_to_label[pred_labels[0][i]])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Training program of the BIOS Tagger.')
    parser.add_argument('modelname', type=str,
                        help='type of model to be trained: FOFE encodings or Classic embeddings')
    parser.add_argument('datafile', type=str,
                        help='file or folder containing the data')
    parser.add_argument('paramfile', type=str,
                        help='file or folder to save the model and metrics')
    parser.add_argument('--embedding_size', type=int, default=100,
                        help='size of the word embeddings when using ')
    parser.add_argument('--hidden_size', type=int, default=100,
                        help='size of hidden states of GRU')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--num_epochs', nargs='+', type=int, default=[0, 50, 100],
                        help='number of epochs used for training')
    parser.add_argument('--learn_rate', type=float,
                        default=0.001, help='learning rate for model')
    parser.add_argument('--reg_factor', type=float, default=0.0,
                        help='weight decay factor of model')

    args = parser.parse_args()

    tagger = Tagger(args.modelname, args.datafile, args.paramfile, args.num_epochs, embedding_size=args.embedding_size,
                    hidden_size=args.hidden_size, dropout=args.dropout, learn_rate=args.learn_rate, reg_factor=args.reg_factor)
    metrics = tagger.train(args.num_epochs, seed=0, embedding_size=args.embedding_size,
                           learn_rate=args.learn_rate, reg_factor=args.reg_factor,
                           hidden_size=args.hidden_size, dropout=args.dropout)

    #test_sent = tagger.data.test_input[0]
    #test_labels = tagger.data.test_labels[0]

    # tagger.predict(
    # "trained_models/fofe_encoding/params_atis_fofe.nnp", test_sent, test_labels)
