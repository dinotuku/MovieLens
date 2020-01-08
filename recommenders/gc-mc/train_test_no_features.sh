#!/bin/bash

# Movielens 100K on official split without features
python train.py -d ml_100k --accum stack -nleft -e 1000 --testing  >> logs/no_features.log
