#!/bin/bash

# Movielens 100K on official split without features
python train.py -d ml_100k --accum stack -do 0.7 -nleft -nb 2 -e 1000 --testing  >> logs/no_features.log

# Movielens 100K on official split with features
python train.py -d ml_100k --accum stack -do 0.7 -nleft -nb 2 -e 1000 --features --feat_hidden 10 --testing >> logs/with_features.log
