#!/bin/sh

batch_size=8
embedding_size=100
hidden_size=100
num_epochs=1000
learn_rate=0.001
reg_factor=0.0

./tagger.py FOFE Atis.json \
--batch_size=$batchsize \
--embedding_size=$embedding_size \
--hidden_size=$hidden_size \
--num_epochs=$num_epochs \
--learn_rate=$learn_rate \
--reg_factor=$reg_factor