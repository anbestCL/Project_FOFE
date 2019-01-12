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
