#!/bin/bash

batch_size=20
num_epochs=($(seq 0 50 400)) 

../py-scripts/hyperopt.py Classic ../../data/Atis.json hyper\
--batch_size=$batch_size \
--num_epochs "${num_epochs[@]}"
