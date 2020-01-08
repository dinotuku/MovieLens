#!/usr/bin/env python
# -- coding: utf-8 --
"""
Functions for I/O
"""

import os

def dump_history(store_path, logs):
    """ Dump training history """
    with open(os.path.join(store_path, 'train_loss'), 'a') as file:
        for loss in logs.tra_loss:
            file.write('{}\n'.format(loss))

    with open(os.path.join(store_path, 'valid_loss'), 'a') as file:
        for loss in logs.val_loss:
            file.write('{}\n'.format(loss))
