#!/usr/bin/env python
# -- coding: utf-8 --
"""
Build Model and Create History Class
"""

from keras.layers import Input, Embedding, Flatten, Dot, Add, Concatenate
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import Callback

DROPOUT_RATE = 0.5

class History(Callback):
    """ Class for training history """
    def on_train_begin(self, logs=None):
        """ Initialization """
        self.tra_loss = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs=None):
        """ Log training information """
        logs = logs or {}
        self.tra_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

def build_model(num_users, num_movies, lat_dims, mode, users_info, movies_info):
    """ Build keras model for training """
    # Model structure
    u_emb_input = Input(shape=(1, ))
    u_emb = Embedding(num_users, lat_dims,
                      embeddings_initializer='random_normal',
                      trainable=True)(u_emb_input)
    u_emb = Flatten()(u_emb)

    u_bias = Embedding(num_users, 1,
                       embeddings_initializer='zeros',
                       trainable=True)(u_emb_input)
    u_bias = Flatten()(u_bias)

    u_info_emb = Embedding(num_users,
                           users_info.shape[1],
                           weights=[users_info],
                           trainable=False)(u_emb_input)
    u_info_emb = Flatten()(u_info_emb)

    m_emb_input = Input(shape=(1, ))
    m_emb = Embedding(num_movies, lat_dims,
                      embeddings_initializer='random_normal',
                      trainable=True)(m_emb_input)
    m_emb = Flatten()(m_emb)

    m_bias = Embedding(num_movies, 1,
                       embeddings_initializer='zeros',
                       trainable=True)(m_emb_input)
    m_bias = Flatten()(m_bias)

    m_info_emb = Embedding(num_movies,
                           movies_info.shape[1],
                           weights=[movies_info],
                           trainable=False)(m_emb_input)
    m_info_emb = Flatten()(m_info_emb)

    if mode == 'mf':
        dot = Dot(axes=1)([u_emb, m_emb])
        output = Add()([dot, u_bias, m_bias])
        model = Model(inputs=[u_emb_input, m_emb_input], outputs=output)
    elif mode == 'dnn':
        u_emb = Dropout(DROPOUT_RATE)(u_emb)
        m_emb = Dropout(DROPOUT_RATE)(m_emb)
        concat = Concatenate()([u_emb, m_emb])
        dnn = Dense(256, activation='relu')(concat)
        dnn = BatchNormalization()(dnn)
        dnn = Dropout(DROPOUT_RATE)(dnn)
        dnn = Dense(256, activation='relu')(dnn)
        dnn = BatchNormalization()(dnn)
        dnn = Dropout(DROPOUT_RATE)(dnn)
        dnn = Dense(256, activation='relu')(dnn)
        dnn = BatchNormalization()(dnn)
        dnn = Dropout(DROPOUT_RATE)(dnn)
        output = Dense(1, activation='relu')(dnn)
        model = Model(inputs=[u_emb_input, m_emb_input], outputs=output)
    elif mode == 'dnn_with_info':
        u_emb = Dropout(DROPOUT_RATE)(u_emb)
        m_emb = Dropout(DROPOUT_RATE)(m_emb)
        concat = Concatenate()([u_emb, m_emb, u_info_emb, m_info_emb])
        dnn = Dense(256, activation='relu')(concat)
        dnn = BatchNormalization()(dnn)
        dnn = Dropout(DROPOUT_RATE)(dnn)
        dnn = Dense(256, activation='relu')(dnn)
        dnn = BatchNormalization()(dnn)
        dnn = Dropout(DROPOUT_RATE)(dnn)
        dnn = Dense(256, activation='relu')(dnn)
        dnn = BatchNormalization()(dnn)
        dnn = Dropout(DROPOUT_RATE)(dnn)
        output = Dense(1, activation='relu')(dnn)
        model = Model(inputs=[u_emb_input, m_emb_input], outputs=output)

    return model
