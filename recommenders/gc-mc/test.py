""" Testing script """

from __future__ import division
from __future__ import print_function

import argparse
import datetime
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import sys
import json

from preprocessing import create_trainvaltest_split, \
    sparse_to_tuple, preprocess_user_item_features, globally_normalize_bipartite_adjacency, \
    load_data_monti, load_official_trainvaltest_split, normalize_features
from model import RecommenderGAE, RecommenderSideInfoGAE
from utils import construct_feed_dict

# Set random seed
seed = int(time.time())
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
ap = argparse.ArgumentParser()
ap.add_argument("-lr", "--learning_rate", type=float, default=0.01,
                help="Learning rate")

ap.add_argument("-e", "--epochs", type=int, default=2500,
                help="Number training epochs")

ap.add_argument("-hi", "--hidden", type=int, nargs=2, default=[500, 75],
                help="Number hidden units in 1st and 2nd layer")

ap.add_argument("-fhi", "--feat_hidden", type=int, default=64,
                help="Number hidden units in the dense layer for features")

ap.add_argument("-ac", "--accumulation", type=str, default="sum", choices=['sum', 'stack'],
                help="Accumulation function: sum or stack.")

ap.add_argument("-do", "--dropout", type=float, default=0.7,
                help="Dropout fraction")

ap.add_argument("-nb", "--num_basis_functions", type=int, default=2,
                help="Number of basis functions for Mixture Model GCN.")

ap.add_argument("-ds", "--data_seed", type=int, default=1234,
                help="""Seed used to shuffle data in data_utils, taken from cf-nade (1234, 2341, 3412, 4123, 1324).
                     Only used for ml_1m and ml_10m datasets. """)

ap.add_argument("-sdir", "--summaries_dir", type=str, default='logs/' + str(datetime.datetime.now()).replace(' ', '_'),
                help="Directory for saving tensorflow summaries.")

# Boolean flags
fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-nsym', '--norm_symmetric', dest='norm_symmetric',
                help="Option to turn on symmetric global normalization", action='store_true')
fp.add_argument('-nleft', '--norm_left', dest='norm_symmetric',
                help="Option to turn on left global normalization", action='store_false')
ap.set_defaults(norm_symmetric=True)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-f', '--features', dest='features',
                help="Whether to use features (1) or not (0)", action='store_true')
fp.add_argument('-no_f', '--no_features', dest='features',
                help="Whether to use features (1) or not (0)", action='store_false')
ap.set_defaults(features=False)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-ws', '--write_summary', dest='write_summary',
                help="Option to turn on summary writing", action='store_true')
fp.add_argument('-no_ws', '--no_write_summary', dest='write_summary',
                help="Option to turn off summary writing", action='store_false')
ap.set_defaults(write_summary=False)

fp = ap.add_mutually_exclusive_group(required=False)
fp.add_argument('-t', '--testing', dest='testing',
                help="Option to turn on test set evaluation", action='store_true')
fp.add_argument('-v', '--validation', dest='testing',
                help="Option to only use validation set evaluation", action='store_false')
ap.set_defaults(testing=False)

args = vars(ap.parse_args())

print('Settings:')
print(args, '\n')

# Define parameters
DATASET = 'ml_100k'
DATASEED = args['data_seed']
NB_EPOCH = args['epochs']
DO = args['dropout']
HIDDEN = args['hidden']
FEATHIDDEN = args['feat_hidden']
BASES = args['num_basis_functions']
LR = args['learning_rate']
WRITESUMMARY = args['write_summary']
SUMMARIESDIR = args['summaries_dir']
FEATURES = args['features']
SYM = args['norm_symmetric']
TESTING = args['testing']
ACCUM = args['accumulation']

SELFCONNECTIONS = False
SPLITFROMFILE = True
VERBOSE = True

NUMCLASSES = 5

# Splitting dataset in training, validation and test set
if FEATURES:
    datasplit_path = 'data/' + DATASET + '/withfeatures.pickle'
else:
    datasplit_path = 'data/' + DATASET + '/nofeatures.pickle'

print("Using official MovieLens dataset split u1.base/u1.test with 10% validation set size...")
u_features, v_features, adj_train, train_labels, train_u_indices, train_v_indices, \
    val_labels, val_u_indices, val_v_indices, test_labels, \
    test_u_indices, test_v_indices, class_values = load_official_trainvaltest_split(DATASET, TESTING)

num_users, num_items = adj_train.shape

num_side_features = 0

# feature loading
if not FEATURES:
    u_features = sp.identity(num_users, format='csr')
    v_features = sp.identity(num_items, format='csr')

    u_features, v_features = preprocess_user_item_features(u_features, v_features)

elif FEATURES and u_features is not None and v_features is not None:
    # use features as side information and node_id's as node input features
    print("Normalizing feature vectors...")
    u_features_side = normalize_features(u_features)
    v_features_side = normalize_features(v_features)

    u_features_side, v_features_side = preprocess_user_item_features(u_features_side, v_features_side)

    u_features_side = np.array(u_features_side.todense(), dtype=np.float32)
    v_features_side = np.array(v_features_side.todense(), dtype=np.float32)

    num_side_features = u_features_side.shape[1]

    # node id's for node input features
    id_csr_v = sp.identity(num_items, format='csr')
    id_csr_u = sp.identity(num_users, format='csr')

    u_features, v_features = preprocess_user_item_features(id_csr_u, id_csr_v)

else:
    raise ValueError('Features flag is set to true but no features are loaded from dataset ' + DATASET)

# global normalization
support = []
support_t = []
adj_train_int = sp.csr_matrix(adj_train, dtype=np.int32)

for i in range(NUMCLASSES):
    # build individual binary rating matrices (supports) for each rating
    support_unnormalized = sp.csr_matrix(adj_train_int == i + 1, dtype=np.float32)

    if support_unnormalized.nnz == 0 and DATASET != 'yahoo_music':
        # yahoo music has dataset split with not all ratings types present in training set.
        # this produces empty adjacency matrices for these ratings.
        sys.exit('ERROR: normalized bipartite adjacency matrix has only zero entries!!!!!')

    support_unnormalized_transpose = support_unnormalized.T
    support.append(support_unnormalized)
    support_t.append(support_unnormalized_transpose)


support = globally_normalize_bipartite_adjacency(support, symmetric=SYM)
support_t = globally_normalize_bipartite_adjacency(support_t, symmetric=SYM)

if SELFCONNECTIONS:
    support.append(sp.identity(u_features.shape[0], format='csr'))
    support_t.append(sp.identity(v_features.shape[0], format='csr'))

num_support = len(support)
support = sp.hstack(support, format='csr')
support_t = sp.hstack(support_t, format='csr')

if ACCUM == 'stack':
    div = HIDDEN[0] // num_support
    if HIDDEN[0] % num_support != 0:
        print("""\nWARNING: HIDDEN[0] (=%d) of stack layer is adjusted to %d such that
                  it can be evenly split in %d splits.\n""" % (HIDDEN[0], num_support * div, num_support))
    HIDDEN[0] = num_support * div

# Collect all user and item nodes for test set
test_u = list(set(test_u_indices))
test_v = list(set(test_v_indices))
test_u_dict = {n: i for i, n in enumerate(test_u)}
test_v_dict = {n: i for i, n in enumerate(test_v)}

test_u_indices = np.array([test_u_dict[o] for o in test_u_indices])
test_v_indices = np.array([test_v_dict[o] for o in test_v_indices])

test_support = support[np.array(test_u)]
test_support_t = support_t[np.array(test_v)]

# features as side info
if FEATURES:
    test_u_features_side = u_features_side[np.array(test_u)]
    test_v_features_side = v_features_side[np.array(test_v)]
else:
    test_u_features_side = None
    test_v_features_side = None

placeholders = {
    'u_features': tf.sparse_placeholder(tf.float32, shape=np.array(u_features.shape, dtype=np.int64)),
    'v_features': tf.sparse_placeholder(tf.float32, shape=np.array(v_features.shape, dtype=np.int64)),
    'u_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'v_features_nonzero': tf.placeholder(tf.int32, shape=()),
    'labels': tf.placeholder(tf.int32, shape=(None,)),

    'u_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),
    'v_features_side': tf.placeholder(tf.float32, shape=(None, num_side_features)),

    'user_indices': tf.placeholder(tf.int32, shape=(None,)),
    'item_indices': tf.placeholder(tf.int32, shape=(None,)),

    'class_values': tf.placeholder(tf.float32, shape=class_values.shape),

    'dropout': tf.placeholder_with_default(0., shape=()),
    'weight_decay': tf.placeholder_with_default(0., shape=()),

    'support': tf.sparse_placeholder(tf.float32, shape=(None, None)),
    'support_t': tf.sparse_placeholder(tf.float32, shape=(None, None)),
}

# create model
if FEATURES:
    model = RecommenderSideInfoGAE(placeholders,
                                   input_dim=u_features.shape[1],
                                   feat_hidden_dim=FEATHIDDEN,
                                   num_classes=NUMCLASSES,
                                   num_support=num_support,
                                   self_connections=SELFCONNECTIONS,
                                   num_basis_functions=BASES,
                                   hidden=HIDDEN,
                                   num_users=num_users,
                                   num_items=num_items,
                                   accum=ACCUM,
                                   learning_rate=LR,
                                   num_side_features=num_side_features,
                                   logging=True)
else:
    model = RecommenderGAE(placeholders,
                           input_dim=u_features.shape[1],
                           num_classes=NUMCLASSES,
                           num_support=num_support,
                           self_connections=SELFCONNECTIONS,
                           num_basis_functions=BASES,
                           hidden=HIDDEN,
                           num_users=num_users,
                           num_items=num_items,
                           accum=ACCUM,
                           learning_rate=LR,
                           logging=True)

# Convert sparse placeholders to tuples to construct feed_dict
test_support = sparse_to_tuple(test_support)
test_support_t = sparse_to_tuple(test_support_t)

u_features = sparse_to_tuple(u_features)
v_features = sparse_to_tuple(v_features)
assert u_features[2][1] == v_features[2][1], 'Number of features of users and items must be the same!'

num_features = u_features[2][1]
u_features_nonzero = u_features[1].shape[0]
v_features_nonzero = v_features[1].shape[0]

# No dropout for test runs
test_feed_dict = construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero,
                                     v_features_nonzero, test_support, test_support_t,
                                     test_labels, test_u_indices, test_v_indices, class_values, 0.,
                                     test_u_features_side, test_v_features_side)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

model.load(sess)

# store model including exponential moving averages
saver = tf.train.Saver()
save_path = saver.save(sess, "models/%s.ckpt" % model.name, global_step=model.global_step)

# test_avg_loss, test_rmse = sess.run([model.loss, model.rmse], feed_dict=test_feed_dict)
# print('test loss = ', test_avg_loss)
# print('test rmse = ', test_rmse)

# restore with polyak averages of parameters
variables_to_restore = model.variable_averages.variables_to_restore()
saver = tf.train.Saver(variables_to_restore)
saver.restore(sess, save_path)

test_avg_loss, test_rmse, res = sess.run([model.loss, model.rmse, model.outputs], feed_dict=test_feed_dict)
res = np.argmax(res, axis=1) + 1

print('test loss = ', test_avg_loss)
print('test rmse = ', test_rmse)

sess.close()
