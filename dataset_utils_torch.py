import tensorflow as tf
import torch
import numpy as np
import random
import pickle
import pandas as pd
import os
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import subprocess
from torchvision.datasets import ImageFolder, DatasetFolder, ImageNet
import torchvision.datasets as datasets
import time
import sys

#from sklearn.preprocessing import Normalizer

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

def load_data(data_path):
    """Load ImageNet (training and val set)."""

    # Load ImageNet and normalize
    traindir = os.path.join(data_path, "train")
    valdir = os.path.join(data_path, "val")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    return train_dataset, val_dataset

class ManageDatasets():

    def __init__(self, cid, model_name):
        self.cid = cid
        self.model_name = model_name
        random.seed(self.cid)

    def load_UCIHAR(self, n_clients, filename_train, filename_test, non_iid=False):
        with open(filename_train, 'rb') as handle:
            train = pickle.load(handle)

        with open(filename_test, 'rb') as handle:
            test = pickle.load(handle)

        train['label'] = train['label'].apply(lambda x: x -1)
        y_train        = train['label'].values
        train.drop('label', axis=1, inplace=True)
        x_train = train.values

        test['label'] = test['label'].apply(lambda x: x -1)
        y_test        = test['label'].values
        test.drop('label', axis=1, inplace=True)
        x_test = test.values
        print("exemplo ucihar: ", x_test.shape, x_train.shape)


        return x_train, y_train, x_test, y_test


    def load_MotionSense(self, n_clients, filename_train, filename_test, non_iid=False):
        with open(filename_train, 'rb') as handle:
            train = pickle.load(handle)

        with open(filename_test, 'rb') as handle:
            test = pickle.load(handle)

        y_train = train['activity'].values
        train.drop('activity', axis=1, inplace=True)
        train.drop('subject', axis=1, inplace=True)
        train.drop('trial', axis=1, inplace=True)
        x_train = train.values

        y_test = test['activity'].values
        test.drop('activity', axis=1, inplace=True)
        test.drop('subject', axis=1, inplace=True)
        test.drop('trial', axis=1, inplace=True)
        x_test = test.values
        print("exemplo motion: ", x_test.shape, x_train.shape)

        return x_train, y_train, x_test, y_test


    def load_MNIST(self, n_clients, filename_train, filename_test, non_iid=False):


        if non_iid:
            # print("atual: ", os.getcwd())
            with open(filename_train, 'rb') as handle:
                idx_train = pickle.load(handle)

            with open(filename_test, 'rb') as handle:
                idx_test = pickle.load(handle)


            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            y = np.concatenate((y_train, y_test), axis=0)
            x = np.concatenate((x_train, x_test), axis=0)
            if self.model_name == 'CNN':
                x = np.expand_dims(x, axis=1)
            x                     = x/255.0

            x_train = x[idx_train]
            x_test  = x[idx_test]

            y_train = y[idx_train]
            y_test  = y[idx_test]

            print("treino: ", x_train.shape, y_train.shape, " teste: ", x_test.shape, y_test.shape)


        else:

            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train, x_test                      = x_train/255.0, x_test/255.0
            x_train, y_train, x_test, y_test     = self.slipt_dataset(x_train, y_train, x_test, y_test, n_clients)

        return x_train, y_train, x_test, y_test

    def load_CIFAR10(self, n_clients, filename_train, filename_test, non_iid=False):

        if non_iid:

            with open(filename_train, 'rb') as handle:
                idx_train = pickle.load(handle)

            with open(filename_test, 'rb') as handle:
                idx_test = pickle.load(handle)

            # if self.cid >= 5:
            # 	time.sleep(4)
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            y_train = np.array([i[0] for i in y_train])
            y_test = np.array([i[0] for i in y_test])
            y = np.concatenate((y_train, y_test), axis=0)
            x = np.concatenate((x_train, x_test), axis=0)/255.0
            x_train = x[idx_train]
            x_test = x[idx_test]

            y_train = y[idx_train]
            y_test = y[idx_test]

            # print("ex antes: ", x_train.shape)
            x_train = np.array([np.moveaxis(i, -1, 0) for i in x_train])
            # print("ex depois: ", x_train.shape)
            x_test = np.array([np.moveaxis(i, -1, 0) for i in x_test])

        else:

            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            x_train, x_test                      = x_train/255.0, x_test/255.0
            x_train, y_train, x_test, y_test     = self.slipt_dataset(x_train, y_train, x_test, y_test, n_clients)


        return x_train, y_train, x_test, y_test

    def load_tiny_imagenet(self, n_clients, filename_train, filename_test, non_iid=False):

        try:

            dir_path = "dataset_utils/data/Tiny-ImageNet/raw_data/"

            with open(filename_train, 'rb') as handle:
                idx_train = pickle.load(handle)

            with open(filename_test, 'rb') as handle:
                idx_test = pickle.load(handle)

            # if self.cid >= 5:
            # 	time.sleep(4)
            # idx_train = idx_train[:100]
            # idx_test = idx_test[:100]
            trainset, valset = load_data(dir_path + "tiny-imagenet-200/")
            # dataset_image = []
            # dataset_label = []
            # dataset_image.extend(trainset.imgs)
            # dataset_image.extend(valset.imgs)
            # dataset_label.extend(trainset.targets)
            # dataset_label.extend(valset.targets)
            # x = np.array(dataset_image)
            # y = np.array(dataset_label)
            # x_train = x[idx_train]
            # x_test = x[idx_test]
            #
            # y_train = y[idx_train]
            # y_test = y[idx_test]
            #
            # trainset.imgs = x_train
            # trainset.targets = y_train
            # valset.imgs = x_test
            # valset.targets = y_test

            return trainset, valset

        except Exception as e:
            print("load tinyimagenet")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


    def load_CIFAR100(self, n_clients, filename_train, filename_test, non_iid=False):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        x_train, x_test                      = x_train/255.0, x_test/255.0
        x_train, y_train, x_test, y_test     = self.slipt_dataset(x_train, y_train, x_test, y_test, n_clients)

        return x_train, y_train, x_test, y_test


    def slipt_dataset(self, x_train, y_train, x_test, y_test, n_clients):
        p_train = int(len(x_train)/n_clients)
        p_test  = int(len(x_test)/n_clients)


        random.seed(self.cid)
        selected_train = random.sample(range(len(x_train)), p_train)

        random.seed(self.cid)
        selected_test  = random.sample(range(len(x_test)), p_test)

        x_train  = x_train[selected_train]
        y_train  = y_train[selected_train]

        x_test   = x_test[selected_test]
        y_test   = y_test[selected_test]


        return x_train, y_train, x_test, y_test


    def select_dataset(self, dataset_name, n_clients, class_per_client, alpha, non_iid):
        print("recebeu: ", self.cid, dataset_name, n_clients, class_per_client, alpha, non_iid)
        filename_train = f"dataset_utils/data/{dataset_name}/{n_clients}_clients/classes_per_client_{class_per_client}/alpha_{alpha}/{self.cid}/idx_train_{self.cid}.pickle"
        filename_test = f"dataset_utils/data/{dataset_name}/{n_clients}_clients/classes_per_client_{class_per_client}/alpha_{alpha}/{self.cid}/idx_test_{self.cid}.pickle"

        if dataset_name == 'MNIST':
            return self.load_MNIST(n_clients=n_clients, filename_train=filename_train, filename_test=filename_test, non_iid=non_iid)

        elif dataset_name == 'CIFAR100':
            return self.load_CIFAR100(n_clients=n_clients, filename_train=filename_train, filename_test=filename_test, non_iid=non_iid)

        elif dataset_name == 'CIFAR10':
            return self.load_CIFAR10(n_clients=n_clients, filename_train=filename_train, filename_test=filename_test, non_iid=non_iid)

        elif dataset_name == 'Tiny-ImageNet':
            return self.load_tiny_imagenet(n_clients=n_clients, filename_train=filename_train, filename_test=filename_test,
                                     non_iid=non_iid)

        elif dataset_name == 'MotionSense':
            return self.load_MotionSense(n_clients=n_clients, filename_train=filename_train, filename_test=filename_test, non_iid=non_iid)

        elif dataset_name == 'UCIHAR':
            return self.load_UCIHAR(n_clients=n_clients, filename_train=filename_train, filename_test=filename_test, non_iid=non_iid)


    def normalize_data(self, x_train, x_test):
        x_train = Normalizer().fit_transform(np.array(x_train))
        x_test  = Normalizer().fit_transform(np.array(x_test))
        return x_train, x_test

    def create_torch_dataset_from_numpy(self, x: np.array, y: np.array, test_size: float, shuffle: bool = True, batch_size: int = 32):

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 42)
        tensor_x_train = torch.Tensor(x_train)  # transform to torch tensor
        tensor_y_train = torch.Tensor(y_train)


        train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
        trainLoader = DataLoader(train_dataset, batch_size, drop_last=True, shuffle=True)

        tensor_x_test = torch.Tensor(x_test)  # transform to torch tensor
        tensor_y_test = torch.Tensor(y_test)

        test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
        testLoader = DataLoader(test_dataset, batch_size, drop_last=True, shuffle=True)

        return trainLoader, testLoader



