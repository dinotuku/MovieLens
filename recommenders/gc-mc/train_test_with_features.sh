#!/bin/bash

# Movielens 100K on official split with features
python train.py -d ml_100k --accum stack -nleft -e 1000 --features --feat_hidden 10 --testing >> logs/with_features.log
