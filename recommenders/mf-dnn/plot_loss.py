#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot loss history
"""

import matplotlib.pyplot as plt


def main():
    """ Main function """
    # read log files
    mf_train_loss = [float(line.rstrip('\n')) for line in open('logs/mf_train_loss')]
    mf_valid_loss = [float(line.rstrip('\n')) for line in open('logs/mf_valid_loss')]
    dnn_train_loss = [float(line.rstrip('\n')) for line in open('logs/dnn_train_loss')]
    dnn_valid_loss = [float(line.rstrip('\n')) for line in open('logs/dnn_valid_loss')]
    dnn_w_info_train_loss = [float(line.rstrip('\n')) for line in open('logs/dnn_w_info_train_loss')]
    dnn_w_info_valid_loss = [float(line.rstrip('\n')) for line in open('logs/dnn_w_info_valid_loss')]

    # plot them
    plt.plot(mf_train_loss)
    plt.plot(mf_valid_loss)
    plt.plot(dnn_train_loss)
    plt.plot(dnn_valid_loss)
    plt.plot(dnn_w_info_train_loss)
    plt.plot(dnn_w_info_valid_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend(['MF Train', 'MF Valid', 'MF + DNN Train', 'MF + DNN Valid', 'MF + DNN with features Train', 'MF + DNN with features Valid'])
    plt.show()


if __name__ == '__main__':

  main()
