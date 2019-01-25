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


class Classic_GRU(nn.Module):
    def __init__(self, vocabsize, embeddingsize, hiddensize, dropoutrate, numlabels):
        super(Classic_GRU, self).__init__()
        self.hidden_size = hiddensize

        self.embedding = nn.Embedding(
            num_embeddings=vocabsize, embedding_dim=embeddingsize)
        self.dropout = nn.Dropout(p=dropoutrate)

        self.gru = nn.GRU(input_size=embeddingsize, hidden_size=self.hidden_size,
                          bidirectional=True, batch_first=True)

        self.linear = nn.Linear(
            in_features=self.hidden_size, out_features=numlabels)

    def forward(self, x):
        x, lengths = x
        if torch.cuda.is_available():
            x = x.cuda()
            lengths = lengths.cuda()
        x = self.dropout(self.embedding(x))
        # packed sequences only supported by rnn layers, so pack before rnn layer and unpack afterwards
        packed_input = pack_padded_sequence(
            x, lengths.cpu().numpy(), batch_first=True)
        packed_output, _ = self.gru(packed_input)

        # Reshape *final* output to (batch_size, seqlen, hidden_size)
        padded = pad_packed_sequence(
            packed_output, padding_value=0.0, batch_first=True)
        # print(padded.shape)
        I = torch.LongTensor(lengths.cpu().numpy()).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), x.size(1), self.hidden_size)-1)
        out = torch.gather(padded[0].cpu(), 2, I).squeeze(1)
        if torch.cuda.is_available():
            out = self.linear(out.cuda())
        else:
            out = self.linear(out)
        # resizing for CrossEntropyLoss
        out = out.view(-1, out.size(2), out.size(1))
        return out
