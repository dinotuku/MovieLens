#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot RMSE history (this code was written for Python3)
"""

import matplotlib.pyplot as plt


def main():
    """ Main function """
    # read log files
    no_features_train_rmse = []
    no_features_valid_rmse = []
    with open('logs/no_features.log', 'r') as f:
        for line in f.readlines():
            if line[0] == '[':
                no_features_train_rmse.append(float(line.split(' ')[6]))
                no_features_valid_rmse.append(float(line.split(' ')[10]))

    with_features_train_rmse = []
    with_features_valid_rmse = []
    with open('logs/with_features.log', 'r') as f:
        for line in f.readlines():
            if line[0] == '[':
                with_features_train_rmse.append(float(line.split(' ')[6]))
                with_features_valid_rmse.append(float(line.split(' ')[10]))

    # plot them
    plt.plot(no_features_train_rmse)
    plt.plot(no_features_valid_rmse)
    plt.plot(with_features_train_rmse)
    plt.plot(with_features_valid_rmse)
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.legend(['GC-MC Train', 'GC-MC Valid', 'GC-MC with features Train', 'GC-MC with features Valid'])
    plt.show()


if __name__ == '__main__':

  main()
