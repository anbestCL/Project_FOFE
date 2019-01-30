import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import torch.optim as optim
import numpy as np
import string
import math
import json
from prep import DataPrep


class Classic_GRU(nn.Module):

    """Model that uses randomly initialised word embeddings to train a bidirectional GRU
    Arguments:
        vocabsize {int} - number of distinct words in data
        embeddingsize {int} - size of word embeddings
        hiddensize {int} - size of hidden layers of GRU
        dropout rate {float} - rate of dropout layer
        numlabels {int} - number of distinct tags in data 

    Returns:
        output tensor -- output of neural network of size (batch_size, padded sequence length, number of labels)
    """

    def __init__(self, vocabsize, embeddingsize, hiddensize, dropoutrate, numlabels):
        super(Classic_GRU, self).__init__()
        self.hidden_size = hiddensize

        self.embedding = nn.Embedding(
            num_embeddings=vocabsize, embedding_dim=embeddingsize)
        self.dropout = nn.Dropout(p=dropoutrate)

        self.gru = nn.GRU(input_size=embeddingsize, hidden_size=self.hidden_size,
                          bidirectional=True, batch_first=True)

        self.linear = nn.Linear(
            in_features=2*self.hidden_size, out_features=numlabels)

    def forward(self, x):
        #x, lengths = x
        x = torch.tensor(x)
        if torch.cuda.is_available():
            x = x.cuda()
            lengths = lengths.cuda()
        x = self.dropout(self.embedding(x))
        # packed sequences only supported by rnn layers, so pack before rnn layer and unpack afterwards
        x = torch.unsqueeze(x, 0)
        output, _ = self.gru(x)
        """ packed_input = pack_padded_sequence(
            x, lengths.cpu().numpy(), batch_first=True)
        packed_output, _ = self.gru(packed_input)

        # unpack output of GRU
        padded, _ = pad_packed_sequence(
            packed_output, padding_value=0.0, batch_first=True)
         """
        if torch.cuda.is_available():
            out = self.linear(output.cuda())
        else:
            out = self.linear(output)
        # resizing for CrossEntropyLoss
        #out = out.view(-1, out.size(2), out.size(1))
        out = torch.squeeze(out, 0)
        return out
