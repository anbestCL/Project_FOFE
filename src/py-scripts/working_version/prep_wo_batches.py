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
from os import walk


class DataPrep:

    """Class that prepares data for training by padding the sequences and dividing them into batches.
    It has two separate init functions depending on the data set given (Atis or Tiger)
    Comment: For Tiger it stores only 5000 batches for training, 1000 for development and 200 for test

    Arguments:
        datafile {string} - path to data
        modelname {string} - either "FOFE" for Fofe character encoding or "Classic" for classic trainable embedding layer

    Returns:
        None -- prepared sentences and labels stored as class variables
    """

    def __init__(self, datafile, modelname):
        self.train_sents = []
        self.dev_sents = []
        self.test_sents = []

        self.train_labels = []
        self.dev_labels = []
        self.test_labels = []

        # format data depending on corpus
        if datafile.endswith(".json"):
            self._init_atis(datafile, modelname)
        elif datafile.endswith("Tiger/"):
            self._init_tiger(datafile, modelname)

    # Following functions are specific for the transformation of the Atis dataset
    def _init_atis(self, datafile, modelname):
        with open(datafile, "r") as f:
            data = json.load(f)
        self._read_Atis_data(data)

        # for the fofe encoding we need to pass the language to get
        # the correct character vocabulary
        if modelname == "FOFE":
            self.train_input = self._prepare_data(self.train_sents, "eng")
            self.dev_input = self._prepare_data(self.dev_sents, "eng")
            self.test_input = self._prepare_data(self.test_sents, "eng")
        else:
            self.train_input = [torch.tensor(sent).cuda() if torch.cuda.is_available(
            ) else torch.tensor(sent) for sent in self.train_sents]
            self.dev_input = [torch.tensor(sent).cuda() if torch.cuda.is_available(
            ) else torch.tensor(sent) for sent in self.dev_sents]
            self.test_input = [torch.tensor(sent).cuda() if torch.cuda.is_available(
            ) else torch.tensor(sent) for sent in self.test_sents]

    def _read_Atis_data(self, data):
        # Extract data from file
        self.word_to_id = data["vocab"]
        self.label_to_id = data["label_dict"]
        self.test_sents = data["test_sents"]
        self.test_labels = data["test_labels"]
        self.test_labels = [torch.tensor(labels).cuda() if torch.cuda.is_available(
        ) else torch.tensor(labels) for labels in self.test_labels]

        # we also want to create a dev set by splitting the training data
        train_dev_sents = data["train_sents"]
        train_dev_labels = data["train_labels"]
        num_train = math.floor(0.8 * len(train_dev_sents))

        self.train_sents = train_dev_sents[:num_train]
        self.train_labels = train_dev_labels[:num_train]
        self.train_labels = [torch.tensor(labels).cuda() if torch.cuda.is_available(
        ) else torch.tensor(labels) for labels in self.train_labels]

        self.dev_sents = train_dev_sents[num_train:]
        self.dev_labels = train_dev_labels[num_train:]
        self.dev_labels = [torch.tensor(labels).cuda() if torch.cuda.is_available(
        ) else torch.tensor(labels) for labels in self.dev_labels]

    # Following functions are specific to treating Tiger data

    def _init_tiger(self, datafile, modelname):
        self.tagset = set()
        self.vocab = set()

        for (dirpath, dirnames, filenames) in walk(datafile):
            for filename in filenames:
                if filename.startswith("develop"):
                    with open(dirpath+filename) as f:
                        self._read_Tiger_data(
                            f, self.dev_sents, self.dev_labels)
                elif filename.startswith("train"):
                    with open(dirpath+filename) as f:
                        self._read_Tiger_data(
                            f, self.train_sents, self.train_labels)
                else:
                    with open(dirpath+filename) as f:
                        self._read_Tiger_data(
                            f, self.test_sents, self.test_labels)

        self._reorganize_Tiger_data()

        # since Tiger data set is very large use only 1000 batches for training,
        # 1000 for development and 200 for testing
        if modelname == "FOFE":
            self.train_input = self._prepare_data(
                self.train_sents, "de")[:10000]
            self.dev_input = self._prepare_data(self.dev_sents, "de")[:5000]
            self.test_input = self._prepare_data(self.test_sents, "de")[:1000]
        elif modelname == "Classic":
            self.train_input = [torch.tensor(sent).cuda() if torch.cuda.is_available(
            ) else torch.tensor(sent) for sent in self.train_sents[:10000]]
            self.dev_input = [torch.tensor(sent).cuda() if torch.cuda.is_available(
            ) else torch.tensor(sent) for sent in self.dev_sents[:5000]]
            self.test_input = [torch.tensor(sent).cuda() if torch.cuda.is_available(
            ) else torch.tensor(sent) for sent in self.test_sents[:1000]]

        self.train_labels = [torch.tensor(labels).cuda() if torch.cuda.is_available(
        ) else torch.tensor(labels) for labels in self.train_labels[:10000]]
        self.dev_labels = [torch.tensor(labels).cuda() if torch.cuda.is_available(
        ) else torch.tensor(labels) for labels in self.dev_labels[:5000]]
        self.test_labels = [torch.tensor(labels).cuda() if torch.cuda.is_available(
        ) else torch.tensor(labels) for labels in self.test_labels[:1000]]

    def _read_Tiger_data(self, file, sentences, labels):

        sentence_words = []
        sentence_tags = []

        for line in file:
            if line != "\n":
                word, tag = line.strip().split("\t")
                sentence_words.append(word)
                sentence_tags.append(tag)
                self.tagset.add(tag)
                self.vocab.add(word)
            else:
                sentences.append(sentence_words)
                labels.append(sentence_tags)
                sentence_words = []
                sentence_tags = []

    def _reorganize_Tiger_data(self):
        self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
        self.label_to_id = {label: i for i, label in enumerate(self.tagset)}

        # Tiger sentences and labels are strings, need to transform to indexes
        self.train_sents = [[self.word_to_id[word]
                             for word in sent] for sent in self.train_sents]
        self.dev_sents = [[self.word_to_id[word]
                           for word in sent] for sent in self.dev_sents]
        self.test_sents = [[self.word_to_id[word]
                            for word in sent] for sent in self.test_sents]

        self.train_labels = [[self.label_to_id[label]
                              for label in sent] for sent in self.train_labels]
        self.dev_labels = [[self.label_to_id[label]
                            for label in sent] for sent in self.dev_labels]
        self.test_labels = [[self.label_to_id[label]
                             for label in sent] for sent in self.test_labels]

    # Following functions prepare data (both Atis and Tiger) to give as input to model

    def _prepare_seq_for_padding(self, sequence):

        # padding of words to give later to FOFE Encoding layer

        seq_lengths = LongTensor(list(map(len, sequence)))
        seq_tensor = Variable(torch.zeros(
            (len(sequence), seq_lengths.max()))).long()
        for idx, (seq, seqlen) in enumerate(zip(sequence, seq_lengths)):
            # for FOFE encoding we need padding on both sentence and word level
            # on word level pad in front of word, on sentence level at the end of sentence
            seq_tensor[idx, seq_lengths.max()-seqlen:] = LongTensor(seq)

        return seq_tensor

    def _prepare_data(self, sents, language):

        # Prepare sentences for packing
        id_to_word = {id: word for word, id in self.word_to_id.items()}
        batches = []
        for sent in sents:
            # Sentences are indexes, for FOFE encoding need words -> transform sequences of indexes
            # to sequences of strings
            sent_strings = [id_to_word[id] for id in sent]

            # Transforming words into sequences of letters using all ASCII characters and looking up index of character
            # sorted list
            if language == "eng":
                characters = string.printable
            elif language == "de":
                # additional characters appearing in Tiger corpus
                characters = string.printable + "äöüÄÖÜßéàâçèêôóáëãîÀÂÉÈÊÔÓÁЁÃÎ" + '§'
            self.vocab_char = ['<PAD>'] + sorted(characters)

            vectorized_sent = [[self.vocab_char.index(
                tok) for tok in seq] for seq in sent_strings]

            # Prepare words for packing
            input_prep = self._prepare_seq_for_padding(
                vectorized_sent)
            batches.append(input_prep.cuda()) if torch.cuda.is_available(
            ) else batches.append(input_prep)
        return batches


if __name__ == "__main__":
    prep_data = DataPrep("data/Atis.json", "Classic")
    langauge = "eng"

    #prep_data = DataPrep("data/Tiger/", "FOFE")

    #  Forward function of future FOFE encoding layer
    # input should be tensor with train_input and sent_lengths for later packing and unpacking
    language = "de"
    if language == "eng":
        characters = string.printable
    elif language == "de":
        characters = string.printable + "äöüÄÖÜßéàâçèêôóáëãîÀÂÉÈÊÔÓÁЁÃÎ" + '§'
    VOCAB_CHAR = ['<PAD>'] + sorted(characters)

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
