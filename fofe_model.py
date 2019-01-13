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


class FOFE_Encoding(nn.Module):
    def __init__(self, vocab_size):
        super(FOFE_Encoding, self).__init__()
        self.vocab_size = vocab_size
        self.forgetting_factor = nn.Parameter(
            torch.zeros(1), requires_grad=True)

    def forward(self, input):
        sents, lengths = input
        samples_encoded = torch.zeros(
            (len(sents), len(sents[0]), self.vocab_size))
        for i, sent in enumerate(sents):
            sent_encoded = torch.zeros((len(sent), self.vocab_size))
            for j, words in enumerate(sent):
                V = torch.zeros((len(words), self.vocab_size))
                z = torch.zeros(self.vocab_size)
                for k, char_id in enumerate(words):
                    V[k, char_id] = 1.
                    z = self.forgetting_factor*z + V[k]
                sent_encoded[j] = z
            samples_encoded[i] = sent_encoded

        return (samples_encoded, lengths)  # requires_grad=True ?


class FOFE_GRU(nn.Module):
    def __init__(self, vocabsize, hiddensize, numlabels):
        super(FOFE_GRU, self).__init__()
        self.hidden_size = hiddensize
        self.input_size = vocabsize

        self.fofe = FOFE_Encoding(vocab_size=self.input_size)

        self.dropout = nn.Dropout(p=0.5)

        self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size,
                          bidirectional=True, batch_first=True)

        self.linear = nn.Linear(
            in_features=self.hidden_size, out_features=numlabels)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x, lengths = self.fofe(x)
        x = self.dropout(x)
        # packed sequences only supported by rnn layers, so pack before rnn layer and unpack afterwards
        packed_input = pack_padded_sequence(
            x, lengths.cpu().numpy(), batch_first=True)
        packed_output, _ = self.gru(packed_input)

        # Reshape *final* output to (batch_size, seqlen, hidden_size)
        padded = pad_packed_sequence(packed_output, batch_first=True)
        I = torch.LongTensor(lengths.cpu().numpy()).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), x.size(1), self.hidden_size)-1)
        out = torch.gather(padded[0], 2, I).squeeze(1)
        out = self.activation(self.linear(out))
        out = out.view(-1, out.size(2), out.size(1))
        return out


###### MAIN PROCESSING #############


VOCAB_CHAR_SIZE = len(['<PAD>'] + sorted(string.printable))
HIDDEN_SIZE = 50
NUM_EPOCHS = 1

prep_data = DataPrep("data.json", batchsize=8)
NUM_LABELS = len(prep_data.label_to_id)
train_input = prep_data.train_input
train_labels = prep_data.train_labels

fofe_model = FOFE_GRU(VOCAB_CHAR_SIZE, HIDDEN_SIZE, NUM_LABELS)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fofe_model.parameters(), lr=0.0001, weight_decay=0.001)
""" for name, param in fofe_model.named_parameters():
    if param.requires_grad:
        print(name, param.data) """

for epoch in range(NUM_EPOCHS):
    loss_accum = 0.0
    for batch, (labels, lengths) in zip(train_input, train_labels):
        optimizer.zero_grad()
        output = fofe_model.forward(batch)
        loss = criterion(output, labels)
        loss_accum += loss.data.item()
        loss.backward()
        optimizer.step()    # Does the update
    # Evaluate model
    # Division braucht man nur, weil man durch i iteriert
    print("train loss:", loss_accum/len(train_input))
    #output_dev = linreg_model.forward(np_to_torch(dev_x))
    #loss_dev = criterion(output_dev, np_to_torch(dev_y))
    #print("dev loss", loss_dev.data.item())

# Questions: CrossEntropy-Correct? Padding überprüfen?
