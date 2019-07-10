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


class FOFE_Encoding(nn.Module):

    """FOFE character encoding layer which loops through the sentences, words and characters and
    transforms every character into a one-hot vector scaled by a forgetting factor to the power 
    corresponding to the position of the character within the word starting from back of word
    Arguments:
        vocab_size -- number of distinct characters
    
    Returns:
        tuple of tensors -- sentences with encoded words and corresponding sentence lengths
                            needed for packing for GRU
    """

    def __init__(self, vocab_size):
        super(FOFE_Encoding, self).__init__()
        self.vocab_size = vocab_size
        self.forgetting_factor = nn.Parameter(
            torch.tensor(0.0), requires_grad=True)

    def forward(self, x):
      sent_encoded = torch.zeros((len(x), self.vocab_size))
      if torch.cuda.is_available():
            sent_encoded = sent_encoded.cuda()
      for j, words in enumerate(x):
            V = torch.zeros((len(words), self.vocab_size))
            z = torch.zeros(self.vocab_size)
            if torch.cuda.is_available():
               V = V.cuda()
               z = z.cuda()
            for k, char_id in enumerate(words):
               if char_id != 0:
                  V[k, char_id] = 1.
                  z = self.forgetting_factor*z + V[k]
            sent_encoded[j] = z
      return sent_encoded


class FOFE_GRU(nn.Module):

    """Model that uses FOFE character encodings to train a bidirectional GRU 
    Arguments:
        vocabsize {int} - number of distince characters
        hiddensize {int} - size of hidden layers of GRU
        dropout rate {float} - rate of dropout layer
        numlabels {int} - number of distinct tags in data 
    Returns:
        output tensor -- size (batch_size, padded sequence length, number of labels)
    """
    def __init__(self, vocabsize, hiddensize, dropoutrate, numlabels):
        super(FOFE_GRU, self).__init__()
        self.hidden_size = hiddensize

        self.fofe = FOFE_Encoding(vocab_size=vocabsize)

        self.dropout = nn.Dropout(p=dropoutrate)

        self.gru = nn.GRU(input_size=vocabsize, hidden_size=self.hidden_size,
                          bidirectional=True, batch_first=True)

        self.linear = nn.Linear(
            in_features=2*self.hidden_size, out_features=numlabels)

    def forward(self, x):
        x = self.fofe(x)
        x = self.dropout(x)
        x = torch.unsqueeze(x,0)
        out,_ = self.gru(x)

        if torch.cuda.is_available():
            out = self.linear(out.cuda())
        else:
            out = self.linear(out)
        out = torch.squeeze(out,0)
        return out

if __name__=="__main__":
    # first sentence in train data
    # [232 542 502 196 208  77  62  10  35  40  58 234 137  62  11 234 481 321]
    # after character transformation
    sent = torch.tensor([[  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,  79,  84],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,  93,  71,  84,  90],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,   0,  79],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,  90,  78,  75],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,  71,  83],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,  72,  85,  89,  90,  85,  84],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,  74,  75,  84,  92,  75,  88],
        [  0,   0,   0,   0,   0,  42,  47,  45,  47,  58,  42,  47,
          45,  47,  58,  42,  47,  45,  47,  58],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,  71,  88,  88,  79,  92,  75],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,  71,  84,  74],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,  90,  85],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,  71,  90],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,  76,  82,  95],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,  71,  90],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,   0,   0,  79,  84],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,   0,   0,   0,  76,  88,  85,  83],
        [  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
           0,  83,  85,  88,  84,  79,  84,  77],
        [ 42,  47,  45,  47,  58,  42,  47,  45,  47,  58,  42,  47,
          45,  47,  58,  42,  47,  45,  47,  58]])
    lengths = torch.tensor([18])
    test_tensor = ([sent], lengths)

    test_model = FOFE_GRU(100, 100, 0.5, 128)
    output = test_model.forward(test_tensor)