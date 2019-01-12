import argparse
import string
import numpy as np

# FOFE ENCODING BASIC PRINCIPLE
samples = [["ABCBC", "ABC"], ["ABCBC", "ABC"]]


def encode_FOFE(wordlist):
    characters = string.printable  # All printable ASCII characters.
    token_index = dict(zip(characters, range(0, len(characters))))
    print(len(characters))
    print(len(token_index))
    print(max(token_index.values()))
    param = 0.5
    max_length = 2
    samples_encoded = np.zeros((len(samples), max_length, len(characters)))
    for i, sent in enumerate(samples):
        sent_encoded = np.zeros((len(sent), len(characters)))
        for j, sample in enumerate(sent):
            print(len(sample))
            V = np.zeros((len(sample), max(token_index.values())+1))
            z = np.zeros(len(characters))
            for k, character in enumerate(sample):
                index = token_index.get(character)
                V[k, index] = 1.
                z = param*z + V[k]
            sent_encoded[j] = z
        samples_encoded[i] = sent_encoded
    return samples_encoded


test = encode_FOFE(samples)
print(test)
