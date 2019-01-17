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
        if torch.cuda.is_available():
            lengths = lengths.cuda()
            samples_encoded = torch.zeros((len(sents), len(sents[0]), self.vocab_size)).cuda()
        for i, sent in enumerate(sents):
            if torch.cuda.is_available():
                sent_encoded = torch.zeros((len(sent), self.vocab_size)).cuda()
            for j, words in enumerate(sent):
                if torch.cuda.is_available():
                    V = torch.zeros((len(words), self.vocab_size)).cuda()
                    z = torch.zeros(self.vocab_size).cuda()
                for k, char_id in enumerate(words):
                    if char_id != 0:
                        V[k, char_id] = 1.
                        z = self.forgetting_factor*z + V[k]
                sent_encoded[j] = z
            samples_encoded[i] = sent_encoded

        return (samples_encoded, lengths)


class FOFE_GRU(nn.Module):
    def __init__(self, vocabsize, hiddensize, numlabels):
        super(FOFE_GRU, self).__init__()
        self.hidden_size = hiddensize

        self.fofe = FOFE_Encoding(vocab_size=vocabsize)

        self.dropout = nn.Dropout(p=0.5)

        self.gru = nn.GRU(input_size=vocabsize, hidden_size=self.hidden_size,
                          bidirectional=True, batch_first=True)

        self.linear = nn.Linear(
            in_features=self.hidden_size, out_features=numlabels)

    def forward(self, x):
        x, lengths = self.fofe(x)
        x = self.dropout(x)
        # packed sequences only supported by rnn layers, so pack before rnn layer and unpack afterwards
        packed_input = pack_padded_sequence(
            x, lengths.cpu().numpy(), batch_first=True)
        packed_output, _ = self.gru(packed_input)

        # Reshape *final* output to (batch_size, seqlen, hidden_size)
        # add padding_value to ignore padded elements in next layers
        padded = pad_packed_sequence(
            packed_output, padding_value=0.0, batch_first=True)
        I = torch.LongTensor(lengths.cpu().numpy()).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), x.size(1), self.hidden_size)-1)
        out = torch.gather(padded[0].cpu(), 2, I).squeeze(1)
        out = self.linear(out.cuda())
        out = out.view(-1, out.size(2), out.size(1))
        return out
