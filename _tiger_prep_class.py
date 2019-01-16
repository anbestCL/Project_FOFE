from os import walk


class TigerDataPrep:

    def __init__(self, folder):

        self.train_sents = []
        self.dev_sents = []
        self.test_sents = []
        self.train_labels = []
        self.dev_labels = []
        self.test_labels = []

        self.tagset = set()
        self.vocab = set()

        for (dirpath, dirnames, filenames) in walk(folder):
            for filename in filenames:
                if filename.startswith("develop"):
                    with open(dirpath+filename) as f:
                        self._read_data(
                            f, self.dev_sents, self.dev_labels)
                elif filename.startswith("train"):
                    with open(dirpath+filename) as f:
                        self._read_data(
                            f, self.train_sents, self.train_labels)
                else:
                    with open(dirpath+filename) as f:
                        self._read_data(
                            f, self.test_sents, self.test_labels)

        self._reorganize_data()

    def _read_data(self, file, sentences, labels):

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

    def _reorganize_data(self):
        self.word_to_id = {word: i+1 for i, word in enumerate(self.vocab)}
        self.word_to_id["<PAD>"] = 0
        self.label_to_id = {label: i+1 for i, label in enumerate(self.tagset)}
        self.label_to_id["<PAD>"] = 0

        self.train_sents = [[self.word_to_id[word]
                             for word in sent] for sent in self.train_sents]
        self.dev_sents = [[self.word_to_id[word]
                           for word in sent] for sent in self.dev_sents]
        self.test_sents = [[self.word_to_id[word]
                            for word in sent] for sent in self.test_sents]

        self.train_labels = [[self.label_to_id[word]
                              for word in sent] for sent in self.train_labels]
        self.dev_labels = [[self.label_to_id[word]
                            for word in sent] for sent in self.dev_labels]
        self.test_labels = [[self.label_to_id[word]
                             for word in sent] for sent in self.test_labels]


if __name__ == "__main__":

    test = TigerDataPrep("Tiger/")
