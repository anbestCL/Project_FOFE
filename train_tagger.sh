#!/bin/sh

batch_size=20
embedding_size=100
hidden_size=200
num_epochs={0..501..50}
learn_rate=0.01
reg_factor=0.01

./tagger_cuda.py FOFE Atis.json \
--batch_size=$batch_size \
--embedding_size=$embedding_size \
--hidden_size=$hidden_size \
--num_epochs=$num_epochs \
--learn_rate=$learn_rate \
--reg_factor=$reg_factor
