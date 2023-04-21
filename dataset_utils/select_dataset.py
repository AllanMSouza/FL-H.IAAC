import tensorflow as tf
import torch
import numpy as np
import random
import pickle
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time
import sys

class ManageDatasets():

    def __init__(self):
        random.seed(0)

    # def load_UCIHAR(self):
    #     with open(f'data/UCI-HAR/{self.cid + 1}_train.pickle', 'rb') as train_file:
    #         train = pickle.load(train_file)
    #
    #     with open(f'data/UCI-HAR/{self.cid + 1}_test.pickle', 'rb') as test_file:
    #         test = pickle.load(test_file)
    #
    #     train['label'] = train['label'].apply(lambda x: x - 1)
    #     y_train = train['label'].values
    #     train.drop('label', axis=1, inplace=True)
    #     x_train = train.values
    #
    #     test['label'] = test['label'].apply(lambda x: x - 1)
    #     y_test = test['label'].values
    #     test.drop('label', axis=1, inplace=True)
    #     x_test = test.values
    #     print("exemplo ucihar: ", x_test.shape, x_train.shape)
    #
    #     return x_train, y_train, x_test, y_test
    #
    # def load_MotionSense(self):
    #     with open(f'data/motion_sense/{self.cid + 1}_train.pickle', 'rb') as train_file:
    #         train = pickle.load(train_file)
    #
    #     with open(f'data/motion_sense/{self.cid + 1}_test.pickle', 'rb') as test_file:
    #         test = pickle.load(test_file)
    #
    #     y_train = train['activity'].values
    #     train.drop('activity', axis=1, inplace=True)
    #     train.drop('subject', axis=1, inplace=True)
    #     train.drop('trial', axis=1, inplace=True)
    #     x_train = train.values
    #
    #     y_test = test['activity'].values
    #     test.drop('activity', axis=1, inplace=True)
    #     test.drop('subject', axis=1, inplace=True)
    #     test.drop('trial', axis=1, inplace=True)
    #     x_test = test.values
    #     print("exemplo motion: ", x_test.shape, x_train.shape)
    #
    #     return x_train, y_train, x_test, y_test

    def load_MNIST(self):

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        return x_train, y_train, x_test, y_test

    def load_CIFAR10(self):

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = np.array([i[0] for i in y_train])
        y_test = np.array([i[0] for i in y_test])
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # print("ex antes: ", x_train.shape)
        x_train = np.array([np.moveaxis(i, -1, 0) for i in x_train])
        # print("ex depois: ", x_train.shape)
        x_test = np.array([np.moveaxis(i, -1, 0) for i in x_test])

        return x_train, y_train, x_test, y_test

    def load_CIFAR100(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        return x_train, y_train, x_test, y_test

    def slipt_dataset(self, x_train, y_train, x_test, y_test, n_clients):
        p_train = int(len(x_train) / n_clients)
        p_test = int(len(x_test) / n_clients)

        random.seed(self.cid)
        selected_train = random.sample(range(len(x_train)), p_train)

        random.seed(self.cid)
        selected_test = random.sample(range(len(x_test)), p_test)

        x_train = x_train[selected_train]
        y_train = y_train[selected_train]

        x_test = x_test[selected_test]
        y_test = y_test[selected_test]

        return x_train, y_train, x_test, y_test

    def select_dataset(self, dataset_name):

        if dataset_name == 'MNIST':
            return self.load_MNIST()

        elif dataset_name == 'CIFAR100':
            return self.load_CIFAR100()

        elif dataset_name == 'CIFAR10':
            return self.load_CIFAR10()

        # elif dataset_name == 'MotionSense':
        #     return self.load_MotionSense()
        #
        # elif dataset_name == 'UCIHAR':
        #     return self.load_UCIHAR()