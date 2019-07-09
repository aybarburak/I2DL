"""Data Utility Functions."""
# pylint: disable=invalid-name
import os
import pickle as pickle

import numpy as np


def load_cifar_batch(filename):
    """Load single batch of CIFAR-10."""
    with open(filename, 'rb') as f:
        # load with encoding because file was pickled with Python 2
        data_dict = pickle.load(f, encoding='latin1')
        X = np.array(data_dict['data'])
        Y = np.array(data_dict['labels'])
        X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        return X, Y


def load_CIFAR10(root_dir):
    """Load all of CIFAR-10."""
    f = os.path.join(root_dir, 'cifar10_train.p')
    X_batch, y_batch = load_cifar_batch(f)
    return X_batch, y_batch


def evaluate(x):
    if x > 50:
        print('Horray, you passed! Now save your model and submit it!')
    else:
        print('I think you can do better...')

