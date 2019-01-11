import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import string
import math
import json


class FOFE_Encoding(nn.Module):
    def __init__(self):
        super(FOFE_Encoding, self).__init__()
        self.forgetting_factor = nn.Parameter(
            torch.zeros(1), requires_grad=True)

    def forward(self, x):
        # x is a tensor batch_size * max_length
        characters = string.printable  # All printable ASCII characters.
        token_index = dict(zip(characters, range(0, len(characters))))
        param = 0.5
        max_length = 2
        samples_encoded = np.zeros((len(x), max_length, len(characters)))
        for i, sent in enumerate(x):
            sent_encoded = np.zeros((len(sent), len(characters)))
            for j, sample in enumerate(sent):
                V = np.zeros((len(sample), max(token_index.values())+1))
                z = np.zeros(len(characters))
                for k, character in enumerate(sample):
                    index = token_index.get(character)
                    V[k, index] = 1.
                    z = param*z + V[k]
                sent_encoded[j] = z
            samples_encoded[i] = sent_encoded
        return samples_encoded

###### MAIN PROCESSING #############


""" with open("data.json", "r") as f:
    data = json.load(f)

# Extract data from file
word_to_id = data["vocab"]
label_to_id = data["label_dict"]
test_sents = data["test_sents"]
test_labels = data["test_labels"]

# we also want to create a dev set by splitting the training data
train_dev_sents = data["train_sents"] # list of lists
train_dev_labels = data["train_labels"] # list of lists
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
HIDDEN_SIZE=50
MAX_LENGTH=20

new_word_id = word_to_id["<PAD>"]
new_label_id = label_to_id["O"] #O = Outside token, Wörter, die nicht mit B oder I getaggt werden

word = [key for key, value in word_to_id.items() if value == 0] # "'d"
label = [key for key, value in label_to_id.items() if value == 0] # 'B-aircraft_code'

#korrekte Funktion
def reorganize(sentences, i,j):
    replacement = {i:j, j:i}
    return [[replacement.get(id,id) for id in sentence ] for sentence in sentences]
# wenn id in replacement, gebe den Wert zurück, sonst gebe Eingabewert zurück
train_sents = reorganize(train_sents, new_word_id ,0)
dev_sents = reorganize(dev_sents, new_word_id,0)
test_sents = reorganize(test_sents, new_word_id,0)
train_labels = reorganize(train_labels, new_label_id,0)
dev_labels = reorganize(dev_labels, new_label_id,0)
test_labels = reorganize(test_labels, new_label_id,0)

word_to_id["".join(word)] = 572
label_to_id["".join(label)] = 126
word_to_id["<PAD>"] = 0
label_to_id["O"] = 0

id_to_word = {id: word for word, id in word_to_id.items()}
id_to_label = {id: label for label, id in label_to_id.items()} 
def np_to_torch(np_array):
    return torch.from_numpy(np_array).float()
    
"""

""" ###### KERAS PART ###########
def do_padding(sequences, length = MAX_LENGTH):
    return pad_sequences(sequences, length, padding='post', truncating='post', value=0) 

train_sents_padded = do_padding(train_sents)
dev_sents_padded = do_padding(dev_sents) 
test_sents_padded = do_padding(test_sents)

train_labels_padded = to_categorical(do_padding(train_labels), NUM_LABELS) 
dev_labels_padded = to_categorical(do_padding(dev_labels), NUM_LABELS) 
test_labels_padded = to_categorical(do_padding(test_labels), NUM_LABELS) 

model = Sequential()
model.add(Embedding(input_dim = VOCAB_SIZE+1, output_dim = EMBEDDING_SIZE, mask_zero=True)) 
model.add(Dropout(rate=0.5)) 
model.add(Bidirectional(GRU(units = HIDDEN_SIZE, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001), return_sequences = True))) 
model.add(TimeDistributed(Dense(units = NUM_LABELS, activation = "softmax"))) 
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

earlystop = EarlyStopping(monitor='val_loss', patience = 2)
model.fit(train_sents_padded, train_labels_padded, batch_size=8,callbacks = [earlystop], epochs=100,           validation_data=(dev_sents_padded, dev_labels_padded))
results = model.evaluate(test_sents_padded, test_labels_padded)
for name, number in zip(model.metrics_names, results):
    print(name, number)

predicted_labels = model.predict_classes(x = test_sents_padded[0], batch_size = 1)
print(" ".join([id_to_word[id] for id in test_sents[0]]))
print(" ".join([id_to_label[pred_id] for pred_id in predicted_labels[:,0]]))
 """

""" class FOFE_GRU(nn.Module):
    def __init__(self):
        super(FOFE_GRU, self).__init__()

        self.fofe = FOFE_Encoding()

        self.dropout = nn.Dropout(p=0.5)

        # (seq_len, batch_size, input_size)
        self.GRU = nn.GRU(input_size = char_encoding, hidden_size = HIDDEN_SIZE, bidirectional = True)
        
        self.Linear = nn.Linear(input_size = HIDDEN_SIZE, output = NUM_LABELS)
        self.activation = nn.Softmax()



    def forward(self, x):
        x = self.fofe(x)
        x = self.activation(self.Linear(self.GRU(self.dropout(x))))

# Create model
fofe_model = FOFE_GRU(num_features)
optimizer = optim.Adam(fofe_model.parameters(), lr=0.0001, weight_decay = 0.001)

# Train model
num_epochs = 20
for epoch in range(num_epochs):
    loss_accum = 0.0
    for i in range(len(train_y)):
        x_i = np_to_torch(train_x[i])
        y_i = np_to_torch(train_y[i])
        optimizer.zero_grad()   # zero the gradient buffers
        output = fofe_model.forward(x_i)
        loss = criterion(output, y_i)
        loss_accum += loss.data.item()
        loss.backward()
        optimizer.step()    # Does the update
    # Evaluate model
    print("train loss:", loss_accum/len(train_y)) #Division braucht man nur, weil man durch i iteriert
    output_dev = linreg_model.forward(np_to_torch(dev_x))
    loss_dev = criterion(output_dev, np_to_torch(dev_y))
    print("dev loss", loss_dev.data.item())
 """
