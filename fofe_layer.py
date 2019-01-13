import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
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
        self.activation = nn.Softmax()

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

        return self.activation(self.linear(out))


###### MAIN PROCESSING #############


VOCAB_CHAR_SIZE = len(['<PAD>'] + sorted(string.printable))
HIDDEN_SIZE = 50
NUM_LABELS = 192

prep_data = DataPrep("data.json", batchsize=8)
train_input = prep_data.train_input

fofe_model = FOFE_GRU(VOCAB_CHAR_SIZE, HIDDEN_SIZE, NUM_LABELS)
output = fofe_model(train_input)
for name, param in fofe_model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

# TODO
# packing sentences into batches
# preparing labels


""" 

# Create model
fofe_model = FOFE_GRU(num_features)
optimizer = optim.Adam(fofe_model.parameters(), lr=0.0001, weight_decay = 0.001)

# Train model
num_epochs = 20
for epoch in range(num_epochs):
    loss_accum = 0.0
    for i in range(len(train_y)):
        x_i = np_to_torch(train_x[i])
        y_i = np_to_torch(train_y[i])
        optimizer.zero_grad()   # zero the gradient buffers
        output = fofe_model.forward(x_i)
        loss = criterion(output, y_i)
        loss_accum += loss.data.item()
        loss.backward()
        optimizer.step()    # Does the update
    # Evaluate model
    print("train loss:", loss_accum/len(train_y)) #Division braucht man nur, weil man durch i iteriert
    output_dev = linreg_model.forward(np_to_torch(dev_x))
    loss_dev = criterion(output_dev, np_to_torch(dev_y))
    print("dev loss", loss_dev.data.item())
 """
