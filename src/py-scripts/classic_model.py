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
        x = self.dropout(self.embedding(x))
        x = torch.unsqueeze(x, 0)
        output, _ = self.gru(x)

        if torch.cuda.is_available():
            out = self.linear(output.cuda())
        else:
            out = self.linear(output)
        # resizing for CrossEntropyLoss
        out = torch.squeeze(out, 0)
        return out
