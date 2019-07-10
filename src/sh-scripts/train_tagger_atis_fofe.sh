#!/bin/bash

embedding_size=50
hidden_size=50
num_epochs=($(seq 0 5 30)) 
learn_rate=0.001
reg_factor=0.0

./tagger.py FOFE data/Atis.json params \
--embedding_size=$embedding_size \
--hidden_size=$hidden_size \
--num_epochs "${num_epochs[@]}" \
--learn_rate=$learn_rate \
--reg_factor=$reg_factor
