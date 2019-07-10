#!/bin/bash

num_epochs=($(seq 0 5 30)) 

../hyperopt.py FOFE ../data/Atis.json hyper --num_epochs "${num_epochs[@]}"
