#!/usr/bin/env python
# -- coding: utf-8 --
"""
Matrix Factorization (with DNN) For Movie Ratings Prediction
"""

import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from keras.models import load_model

# some constants
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def main():
    """ Main function """
    parser = ArgumentParser()
    parser.add_argument('--input', default='data/', help='Input file path')
    parser.add_argument('--output', default='res.csv', help='Output file path')
    parser.add_argument('--model', default='best', help='Use which model to test')
    parser.add_argument('--dnn', action='store_true', help='Use DNN model')
    parser.add_argument('--dnn_w_info', action='store_true', help='Use DNN with info model')
    parser.add_argument('--normal', action='store_true', help='Normalize ratings')
    args = parser.parse_args()

    # read testing data and get userIDs and movieIDs
    # -1 is for having a zero base index
    pairs_df = pd.read_csv(os.path.join(args.input, 'test.csv'), sep=',')
    users = pairs_df['UserID'].values - 1
    movies = pairs_df['MovieID'].values - 1

    # read user and movie information
    # they can be used as additional features
    users_df = pd.read_csv(os.path.join(BASE_DIR, 'data/users.csv'), engine='python')
    users_age = (users_df['Age'] - np.mean(users_df['Age'])) / np.std(users_df['Age'])
    movies_df = pd.read_csv(os.path.join(BASE_DIR, 'data/movies.csv'), engine='python')

    # get all genres of movies to get one hot representation of them later
    all_genres = np.array([])
    for genres in movies_df['Genres']:
        for genre in genres.split('|'):
            all_genres = np.append(all_genres, genre)
    all_genres = np.unique(all_genres)

    # initiate user and movie additional features
    ratings_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'train.csv'))
    max_user_id = ratings_df['UserID'].drop_duplicates().max()
    max_movie_id = ratings_df['MovieID'].drop_duplicates().max()
    users_info = np.zeros((max_user_id, 23))
    movies_info = np.zeros((max_movie_id, all_genres.shape[0]))

    # concat gender, occupation --> features with dimension = 23
    for idx, user_id in enumerate(users_df['UserID']):
        gender = 1 if users_df['Gender'][idx] == 'M' else 0
        occu = np.zeros(np.max(np.unique(users_df['Occupation'])) + 1)
        occu[users_df['Occupation'][idx]] = 1
        tmp = [gender, users_age[idx]]
        tmp.extend(occu)
        users_info[user_id - 1] = tmp

    # get one hot representation of genres --> features with dimension = 19
    for idx, movie_id in enumerate(movies_df['movieID']):
        genres = movies_df['Genres'][idx].split('|')
        tmp = np.zeros(all_genres.shape[0])
        for genre in genres:
            tmp[np.where(all_genres == genre)[0][0]] = 1
        movies_info[movie_id - 1] = tmp

    # load different models
    if args.dnn:
        model = load_model(os.path.join(MODEL_DIR, "dnn_model_e{:s}.hdf5").format(args.model))
    elif args.dnn_w_info:
        model = load_model(os.path.join(MODEL_DIR, "dnn_w_info_model_e{:s}.hdf5").format(args.model))
    else:
        model = load_model(os.path.join(MODEL_DIR, "mf_model_e{:s}.hdf5").format(args.model))

    # get predictions
    res = model.predict([users, movies])
    res = np.array(res).reshape(len(pairs_df))

    # normalize ratings should only be used for pure matrix factorization
    if args.normal:
        ratings = ratings_df['Rating'].values
        res = (res * np.std(ratings)) + np.mean(ratings)

    # get out of bounds predictions back
    res[res > 5] = 5
    res[res < 1] = 1

    # read true ratings and compute testing rmse
    true_ratings = pairs_df['Rating'].values
    rmse = np.sqrt(np.mean((res - true_ratings) ** 2))

    print('RMSE:', rmse)


if __name__ == '__main__':

    main()
