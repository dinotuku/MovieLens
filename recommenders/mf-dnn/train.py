#!/usr/bin/env python
# -- coding: utf-8 --
"""
Matrix Factorization (with DNN) For Movie Ratings Prediction
"""

import os
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint
from model import History, build_model
from utils import dump_history

# fix seed
np.random.seed(0)

# some constants
VALIDATION_SPLIT = 0.1
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
LOG_DIR = os.path.join(BASE_DIR, 'logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def main():
    """ Main function """
    parser = ArgumentParser()
    parser.add_argument('--dnn', action='store_true', help='Use DNN model')
    parser.add_argument('--dnn_w_info', action='store_true', help='Use DNN with info model')
    parser.add_argument('--normal', action='store_true', help='Normalize ratings')
    parser.add_argument('--dim', type=int, default=128, help='Specify latent dimensions')
    args = parser.parse_args()

    # read training data
    ratings_df = pd.read_csv(os.path.join(BASE_DIR, 'data/train.csv'),
                             sep=',')
    print("{:d} ratings loaded".format(len(ratings_df)))

    # get userIDs, movieIDs and ratings
    # -1 is for having a zero base index
    users = ratings_df['UserID'].values - 1
    print("Users: {:s}, shape = {:s}".format(str(users), str(users.shape)))
    movies = ratings_df['MovieID'].values - 1
    print("Movies: {:s}, shape = {:s}".format(str(movies), str(movies.shape)))
    ratings = ratings_df['Rating'].values
    print("Ratings: {:s}, shape = {:s}".format(str(ratings), str(ratings.shape)))

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

    # normalize ratings (This will improve the result of pure matrix factorization)
    if args.normal:
        mean = np.mean(ratings)
        std = np.std(ratings)
        ratings = (ratings - mean) / std

    # build different models
    # we used the same loss and optimizer for three models
    if args.dnn:
        model = build_model(max_user_id, max_movie_id, args.dim, 'dnn', users_info, movies_info)
        model.compile(loss='mse', optimizer='adam')
        model_name = os.path.join(MODEL_DIR, "dnn_model_e{epoch:02d}.hdf5")
    elif args.dnn_w_info:
        model = build_model(max_user_id, max_movie_id, args.dim, 'dnn_with_info', users_info, movies_info)
        model.compile(loss='mse', optimizer='adam')
        model_name = os.path.join(MODEL_DIR, "dnn_w_info_model_e{epoch:02d}.hdf5")
    else:
        model = build_model(max_user_id, max_movie_id, args.dim, 'mf', users_info, movies_info)
        model.compile(loss='mse', optimizer='adam')
        model_name = os.path.join(MODEL_DIR, "mf_model_e{epoch:02d}.hdf5")

    # print model information
    model.summary()

    # setup training checkpoint
    # it will save the best model in terms of validation loss
    checkpoint = ModelCheckpoint(model_name, monitor='val_loss', verbose=0,
                                 save_best_only=True, mode='min')
    # class that handles saving of training and validation loss
    history = History()
    callbacks_list = [checkpoint, history]

    # split data into training and validation set
    indices = np.random.permutation(users.shape[0])
    val_num = int(users.shape[0] * VALIDATION_SPLIT)
    users = users[indices]
    movies = movies[indices]
    ratings = ratings[indices]
    tra_users = users[:-val_num]
    tra_movies = movies[:-val_num]
    tra_ratings = ratings[:-val_num]
    val_users = users[-val_num:]
    val_movies = movies[-val_num:]
    val_ratings = ratings[-val_num:]

    # train the model
    model.fit([tra_users, tra_movies], tra_ratings,
              batch_size=256,
              epochs=100,
              validation_data=([val_users, val_movies], val_ratings),
              callbacks=callbacks_list)

    # save loss history to files
    dump_history(LOG_DIR, history)

if __name__ == '__main__':

    main()
