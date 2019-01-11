import torch
import string
import numpy as np
from torch import LongTensor
from torch.nn import Embedding, LSTM
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import json
import math
import torch.nn as nn
from torch.autograd import Variable


# Test FOFE encoding for packed input

with open("data.json", "r") as f:
    data = json.load(f)

# Extract data from file
word_to_id = data["vocab"]
label_to_id = data["label_dict"]
test_sents = data["test_sents"]
test_labels = data["test_labels"]

# we also want to create a dev set by splitting the training data
train_dev_sents = data["train_sents"]  # list of lists
train_dev_labels = data["train_labels"]  # list of lists
num_train = math.floor(0.8 * len(train_dev_sents))

train_sents = train_dev_sents[:num_train]
train_labels = train_dev_labels[:num_train]

dev_sents = train_dev_sents[num_train:]
dev_labels = train_dev_labels[num_train:]

UNK_TOKEN = "<UNK>"
PAD_TOKEN = "<PAD>"
VOCAB_SIZE = len(word_to_id)
NUM_LABELS = len(label_to_id)
EMBEDDING_SIZE = 50
HIDDEN_SIZE = 50
MAX_LENGTH = 20

new_word_id = word_to_id["<PAD>"]
# O = Outside token, Wörter, die nicht mit B oder I getaggt werden
new_label_id = label_to_id["O"]

word = [key for key, value in word_to_id.items() if value == 0]  # "'d"
label = [key for key, value in label_to_id.items() if value ==
         0]  # 'B-aircraft_code'


def reorganize(sentences, i, j):
    replacement = {i: j, j: i}
    return [[replacement.get(id, id) for id in sentence] for sentence in sentences]


# wenn id in replacement, gebe den Wert zurück, sonst gebe Eingabewert zurück
train_sents = reorganize(train_sents, new_word_id, 0)
dev_sents = reorganize(dev_sents, new_word_id, 0)
test_sents = reorganize(test_sents, new_word_id, 0)
train_labels = reorganize(train_labels, new_label_id, 0)
dev_labels = reorganize(dev_labels, new_label_id, 0)
test_labels = reorganize(test_labels, new_label_id, 0)

word_to_id["".join(word)] = 572
label_to_id["".join(label)] = 126
word_to_id["<PAD>"] = 0
label_to_id["O"] = 0

id_to_word = {id: word for word, id in word_to_id.items()}
id_to_label = {id: label for label, id in label_to_id.items()}


def prepare_seq_for_padding(sequence, seq_type):
    seq_lengths = LongTensor(list(map(len, sequence)))
    seq_tensor = Variable(torch.zeros(
        (len(sequence), seq_lengths.max()))).long()
    for idx, (seq, seqlen) in enumerate(zip(sequence, seq_lengths)):
        if seq_type == "words":
            seq_tensor[idx, seq_lengths.max()-seqlen:] = LongTensor(seq)
        elif seq_type == "sent":
            seq_tensor[idx, :seqlen] = LongTensor(seq)

    seq_lengths, perm_idx = seq_lengths.sort(
        0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    # Observation: packing word sequences makes it impossible to reconstruct word
    # order in FOFE layer -> give list of tensors and original order to model
    # packed_input_sequence = pack_padded_sequence(
    # train_seq_tensor, train_seq_lengths.cpu().numpy(), batch_first=True)
    return (seq_tensor[perm_idx]) if seq_type == "words" else (seq_tensor, seq_lengths)


# ----- PREPARING SENTENCES FOR PACKING -------------------
# longest sentence in train_sents
# print(len(max(train_sents, key=len)))

# für dev, test ergänzen
sent_input, sent_lengths = prepare_seq_for_padding(train_sents, "sent")

# Testing that preparing worked
# print(sent_input[0])

# Sentences are indexes, for FOFE encoding need words -> transform sequences of indexes
# to sequences of strings
train_sents_strings = [[id_to_word[id] for id in sent]
                       for sent in sent_input.numpy()]  # why?
# dev_sents_strings = [[id_to_word[id] for id in sent] for sent in dev_sents]
# test_sents_strings = [[id_to_word[id] for id in sent] for sent in train_sents]

# Transforming words into sequences of letters using all ASCII characters and looking up index of character
# sorted list

# ! PROBLEM: <PAD> is translated character by character -> change to 0
characters = string.printable
VOCAB_CHAR = [PAD_TOKEN] + sorted(characters)
vectorized_train_sents = [[[VOCAB_CHAR.index(
    tok) if seq != '<PAD>' else 0 for tok in seq] for seq in sent] for sent in train_sents_strings]

# vectorized_dev_sents = [[[VOCAB_CHAR.index(
# tok) for tok in seq] for seq in sent]for sent in dev_sents_strings]
# vectorized_test_sents = [[[VOCAB_CHAR.index(
# tok) for tok in seq] for seq in sent]for sent in test_sents_strings]

# ----- PREPARE WORDS FOR PACKING -------


def prepare_sents(sents, lengths):
    train_input = []
    for i, (sent, length) in enumerate(zip(sents, lengths)):
        sent_reordered = prepare_seq_for_padding(sent, "words")
        train_input.append((sent_reordered, length))
    return train_input


train_input_prep = prepare_sents(vectorized_train_sents, sent_lengths)


#  Forward function of future FOFE encoding layer
# input should be tensor with train_input and sent_lengths for later packing

def forward(sentences, forgetting_factor):
    samples_encoded = np.zeros(
        (len(sentences), len(sentences[0][0]), len(VOCAB_CHAR)))
    for i, (sent, _) in enumerate(sentences):
        sent_encoded = np.zeros((len(sent), len(VOCAB_CHAR)))
        for j, words in enumerate(sent):
            V = np.zeros((len(words), len(VOCAB_CHAR)))
            z = np.zeros(len(VOCAB_CHAR))
            for k, char_id in enumerate(words):
                V[k, char_id] = 1.
                z = forgetting_factor*z + V[k]
            sent_encoded[j] = z
        samples_encoded[i] = sent_encoded
    return torch.tensor(samples_encoded)


# ----- MINIMAL EXAMPLE FOR FORWARD -----------
test_seq = [["ABCBC", "ABC", "<PAD>"]]
test_seq2 = [[[VOCAB_CHAR.index(
    tok) if seq != '<PAD>' else 0 for tok in seq] for seq in sent] for sent in test_seq]
# pad manually words
test_seq2[0][1] = [0, 0] + test_seq2[0][1]
test_seq2 = [(test_seq2[0], 2)]
print(test_seq2)
test2 = forward(test_seq2, 0.5)
print(test2)


# ------ MINIMAL EXAMPLE FOR FORWARD WITH DATA SET ------------
print(train_input_prep[-3])
test = forward([train_input_prep[-3]], 0.5)
print(test)


# regularisation later on
