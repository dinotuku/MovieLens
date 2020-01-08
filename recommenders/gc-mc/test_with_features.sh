#!/bin/bash

# Movielens 100K on official test split with features
python test.py --accum stack -nleft --features --feat_hidden 10 --testing
