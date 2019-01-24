#!/bin/bash

batch_size=20
embedding_size=100
hidden_size=200
num_epochs=($(seq 0 50 550))
learn_rate=0.001
reg_factor=0.0

python3.6 tagger_cuda.py FOFE Atis.json \
--batch_size=$batch_size \
--embedding_size=$embedding_size \
--hidden_size=$hidden_size \
--num_epochs "${num_epochs[@]}" \
--learn_rate=$learn_rate \
--reg_factor=$reg_factor
