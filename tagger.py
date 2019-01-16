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


###### MAIN PROCESSING #############

class Tagger:

    def __init__(self, modelname, datafile, batchsize, embeddingsize, hiddensize, numepochs):
        vocab_char_size = len(['<PAD>'] + sorted(string.printable))

        self.data = DataPrep(datafile, batchsize, modelname)
        numlabels = len(self.data.label_to_id)
        vocabsize = len(self.data.word_to_id)
        print(self.data.label_to_id["O"])

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
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(model.parameters(),
                               lr=0.0001, weight_decay=0.001)
        best_acc = 0
        best_loss = 10
        for epoch in range(num_epochs):
            print(epoch)
            loss_accum = 0.0
            for batch, (labels, lengths) in zip(self.data.train_input, self.data.train_labels):
                optimizer.zero_grad()
                output = model.forward(batch)
                print(output.shape)
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
                    #acc = self.eval(batch)
                if (dev_loss_accum/len(self.data.dev_input)) < best_loss:
                    best_loss = dev_loss_accum/len(self.data.dev_input)
                    print("dev loss", best_loss)


#Tagger("FOFE", "data.json", batchsize=8, hiddensize=50, numepochs=6)
Tagger("FOFE", datafile="data.json", batchsize=8,
       embeddingsize=100, hiddensize=50, numepochs=6)

# pack_padded_sequence
# Auswirkung auf Loss => ignore_index = 0
# Auswirkung auf Layer => ?


# Questions: Eval on Acc? count correct labels?

""" 
# compute the action predictions
      _, predicted_actionIDs = action_scores.max(dim=-1)
         
      predicted_actionIDs = predicted_actionIDs.cpu().data.numpy()
      actionIDs = actionIDs.cpu().data.numpy()
      num_actions += len(actionIDs)
      num_correct += sum([1 for a,a2 in zip(actionIDs,predicted_actionIDs) if a==a2])

      loss_sum += float(loss) """
