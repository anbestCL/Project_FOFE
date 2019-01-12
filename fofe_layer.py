import torch
import torch.nn as nn
from torch.autograd import Variable
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
        samples_encoded = torch.zeros(
            (len(input), len(input[0][0]), self.vocab_size))
        for i, (sent, _) in enumerate(input):
            sent_encoded = torch.zeros((len(sent), self.vocab_size))
            for j, words in enumerate(sent):
                V = torch.zeros((len(words), self.vocab_size))
                z = torch.zeros(self.vocab_size)
                for k, char_id in enumerate(words):
                    V[k, char_id] = 1.
                    z = self.forgetting_factor*z + V[k]
                sent_encoded[j] = z
            samples_encoded[i] = sent_encoded
        return samples_encoded  # requires_grad=True ?


class FOFE_GRU(nn.Module):
    def __init__(self, vocabsize):
        super(FOFE_GRU, self).__init__()

        self.fofe = FOFE_Encoding(vocab_size=vocabsize)

        #self.dropout = nn.Dropout(p=0.5)

        # (seq_len, batch_size, input_size)
        #self.GRU = nn.GRU(input_size = char_encoding, hidden_size = HIDDEN_SIZE, bidirectional = True)

        #self.Linear = nn.Linear(input_size = HIDDEN_SIZE, output = NUM_LABELS)
        #self.activation = nn.Softmax()

    def forward(self, x):
        x = self.fofe(x)

###### MAIN PROCESSING #############


VOCAB_CHAR_SIZE = len(['<PAD>'] + sorted(string.printable))

prep_data = DataPrep("data.json")
train_input = prep_data.train_input

fofe_model = FOFE_GRU(VOCAB_CHAR_SIZE)
output = fofe_model(train_input)

# TODO
# packing sentences into batches
# packing input with lengths tensor before lstm layer
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
