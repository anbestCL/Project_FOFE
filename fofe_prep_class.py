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


class DataPrep:

    def __init__(self, jsonfile, batchsize, modelname):
        with open(jsonfile, "r") as f:
            data = json.load(f)

        self.train_sents = []
        self.dev_sents = []
        self.test_sents = []

        self.train_labels = []
        self.dev_labels = []
        self.test_labels = []

        self._extract(data)
        self._reorganize_data()
        self.batch_size = batchsize
        if modelname == "FOFE":
            self.train_input = self._prepare_data(self.train_sents)
            self.dev_input = self._prepare_data(self.dev_sents)
            self.test_input = self._prepare_data(self.test_sents)
        elif modelname == "Classic":
            self.train_input = self._prepare_labels(self.train_sents)
            self.dev_input = self._prepare_labels(self.dev_sents)
            self.test_input = self._prepare_labels(self.test_sents)
        self.train_labels = self._prepare_labels(self.train_labels)
        self.dev_labels = self._prepare_labels(self.dev_labels)
        self.test_labels = self._prepare_labels(self.test_labels)

    def _extract(self, data):
        # Extract data from file
        self.word_to_id = data["vocab"]
        self.label_to_id = data["label_dict"]
        self.test_sents = data["test_sents"]
        self.test_labels = data["test_labels"]

        # we also want to create a dev set by splitting the training data
        train_dev_sents = data["train_sents"]  # list of lists
        train_dev_labels = data["train_labels"]  # list of lists
        num_train = math.floor(0.8 * len(train_dev_sents))

        self.train_sents = train_dev_sents[:num_train]
        self.train_labels = train_dev_labels[:num_train]

        self.dev_sents = train_dev_sents[num_train:]
        self.dev_labels = train_dev_labels[num_train:]

    def _reorganize(self, sentences, i, j):
        replacement = {i: j, j: i}
        return [[replacement.get(id, id) for id in sentence] for sentence in sentences]

    def _reorganize_data(self):

        new_word_id = self.word_to_id["<PAD>"]
        # O = Outside token, Wörter, die nicht mit B oder I getaggt werden
        new_label_id = self.label_to_id["O"]

        word = [key for key, value in self.word_to_id.items() if value ==
                0]  # "'d"
        label = [key for key, value in self.label_to_id.items() if value ==
                 0]  # 'B-aircraft_code'

        reorganized_sents = []
        for sents in [self.train_sents, self.dev_sents, self.test_sents]:
            new_sents = self._reorganize(sents, new_word_id, 0)
            reorganized_sents.append(new_sents)

        self.train_sents, self.dev_sents, self.test_sents = reorganized_sents

        reorganized_labels = []
        for labels in [self.train_labels, self.dev_labels, self.test_labels]:
            new_labels = self._reorganize(labels, new_label_id, 0)
            reorganized_labels.append(new_labels)

        self.train_labels, self.dev_labels, self.test_labels = reorganized_labels

        self.word_to_id["".join(word)] = 572
        self.label_to_id["".join(label)] = 126
        self.word_to_id["<PAD>"] = 0
        self.label_to_id["O"] = 0
        self.label_to_id = {label: id+1 for (label,
                                             id) in self.label_to_id.items()}
        self.label_to_id["<PAD>"] = 0

    def _prepare_seq_for_padding(self, sequence, seq_type):
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
        return (seq_tensor[perm_idx]) if seq_type == "words" else (seq_tensor, seq_lengths)

    def _prepare_sents(self, sents):
        sentences = []
        for i, sent in enumerate(sents):
            sent_reordered = self._prepare_seq_for_padding(sent, "words")
            sentences.append(sent_reordered)
        return sentences

    def _batch(self, seq):
        l = len(seq)
        for ndx in range(0, l, self.batch_size):
            yield seq[ndx:min(ndx + self.batch_size, l)]

    def _prepare_data(self, sents):

        # Prepare sentences for packing
        id_to_word = {id: word for word, id in self.word_to_id.items()}
        batches = []
        for batch in self._batch(sents):
            sent_input, sent_lengths = self._prepare_seq_for_padding(
                batch, "sent")
            # Sentences are indexes, for FOFE encoding need words -> transform sequences of indexes
            # to sequences of strings
            sents_strings = [[id_to_word[id] for id in sent]
                             for sent in sent_input.numpy()]

            # Transforming words into sequences of letters using all ASCII characters and looking up index of character
            # sorted list

            # ! PROBLEM: <PAD> is translated character by character -> change to 0
            characters = string.printable
            VOCAB_CHAR = ['<PAD>'] + sorted(characters)
            vectorized_sents = [[[VOCAB_CHAR.index(
                tok) if seq != '<PAD>' else 0 for tok in seq] for seq in sent] for sent in sents_strings]

            # Prepare words for packing
            input_prep = self._prepare_sents(vectorized_sents)
            batches.append((input_prep, sent_lengths))
        return batches

    def _to_categorical(self, seq, numclasses):
        return np.eye(numclasses, dtype='uint8')[seq]

    def _prepare_labels(self, labels):
        batches_l = []
        for batch in self._batch(labels):
            padded_labels, padded_labels_lengths = self._prepare_seq_for_padding(
                batch, "sent")
            """ binary_labels = []
            for seq in padded_labels:
                binary_seq = self._to_categorical(seq, len(self.label_to_id))
                binary_labels.append(binary_seq) """
            batches_l.append((padded_labels, padded_labels_lengths))
        return batches_l


if __name__ == "__main__":
    prep_data = DataPrep("data.json", 8, "FOFE")

    #  Forward function of future FOFE encoding layer
    # input should be tensor with train_input and sent_lengths for later packing

    VOCAB_CHAR = ['<PAD>'] + sorted(string.printable)

    def forward(sentences, forgetting_factor):
        sents, lengths = sentences
        samples_encoded = np.zeros(
            (len(sents), len(sents[0]), len(VOCAB_CHAR)))
        for i, sent in enumerate(sents):
            sent_encoded = np.zeros((len(sent), len(VOCAB_CHAR)))
            for j, words in enumerate(sent):
                V = np.zeros((len(words), len(VOCAB_CHAR)))
                z = np.zeros(len(VOCAB_CHAR))
                for k, char_id in enumerate(words):
                    if char_id != 0:
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
    test_seq2 = ([test_seq2[0]], [2])
    print(test_seq2)
    test2 = forward(test_seq2, 0.5)
    print(test2)

    # ------ MINIMAL EXAMPLE FOR FORWARD WITH DATA SET ------------
    """ print(prep_data.train_input[0][-3])
    test = forward(
        ([prep_data.train_input[0][-3]], [prep_data.train_input[1][-3]]), 0.5)
    print(test) """

    # ------- TESTING BATCHES --------
    # print(len(prep_data.train_input))
    for batch in prep_data.train_input:
        test = forward(batch, 0.5)
        # print(test)

    # ------ TESTING LABELS TRANSFORMATION ---------
    # print(len(prep_data.train_labels))
    # print(len(prep_data.train_labels[0][0][0]))
    # print(len(prep_data.train_input[0][0][0]))
