#!/bin/bash

# Movielens 100K on official test split without features
python test.py --accum stack -nleft --testing
