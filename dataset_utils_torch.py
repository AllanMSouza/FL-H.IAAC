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

    def load_EMNIST(self, n_clients, filename_train, filename_test, non_iid=False, batch_size=32):

        try:
            dir_path = "dataset_utils/data/EMNIST/raw_data/"
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

                # Setup directory for train/test data
            config_path = dir_path + "config.json"
            train_path = dir_path + "train/"
            test_path = dir_path + "test/"

            from six.moves import urllib
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)

            # Get EMNIST data
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

            training_dataset = datasets.EMNIST(
                root=dir_path, train=True, download=False, transform=transform, split='balanced')
            validation_dataset = datasets.EMNIST(
                root=dir_path, train=False, download=False, transform=transform, split='balanced')

            with open(filename_train, 'rb') as handle:
                idx_train = pickle.load(handle)

            with open(filename_test, 'rb') as handle:
                idx_test = pickle.load(handle)

            x = training_dataset.data
            training_dataset.data = torch.concatenate((training_dataset.data, validation_dataset.data))
            y = training_dataset.targets
            training_dataset.targets = torch.concatenate((training_dataset.targets, validation_dataset.targets))
            validation_dataset.data = training_dataset.data[idx_test]
            validation_dataset.targets = training_dataset.targets[idx_test]
            training_dataset.data = training_dataset.data[idx_train]
            training_dataset.targets = training_dataset.targets[idx_train]



            # training_dataset.data = x_train
            # training_dataset.targets = y_train
            # validation_dataset.data = x_test
            # validation_dataset.targets = y_test

            training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size,
                                                          shuffle=True)  # Batch size of 100 i.e to work with 100 images at a time

            validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

            print("baixou", len(training_dataset))

            return training_loader, validation_loader

        except Exception as e:
            print("load EMNIST")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def load_CIFAR10(self, n_clients, filename_train, filename_test, non_iid=False, batch_size=32):

        try:
            transform_train = transforms.Compose(
                [transforms.Resize((32, 32)),  # resises the image so it can be perfect for our model.
                 transforms.RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
                 transforms.RandomRotation(10),  # Rotates the image to a specified angel
                 transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                 # Performs actions like zooms, change shear angles.
                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Set the color params
                 transforms.ToTensor(),  # comvert the image to tensor so that it can work with torch
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize all the images
                 ])

            transform_test = transforms.Compose([transforms.Resize((32, 32)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                 ])

            # transform = transforms.Compose(
            #     [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
            # transform_train = transform
            # transform_test = transform
            training_dataset = datasets.CIFAR10(root='./data', train=True, download=True,
                                                transform=transform_train)  # Data augmentation is only done on training images
            validation_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

            if non_iid:

                with open(filename_train, 'rb') as handle:
                    idx_train = pickle.load(handle)

                with open(filename_test, 'rb') as handle:
                    idx_test = pickle.load(handle)

                x = training_dataset.data
                x = np.concatenate((x, validation_dataset.data))
                y = training_dataset.targets
                y = np.concatenate((y, validation_dataset.targets))
                x_train = x[idx_train]
                x_test = x[idx_test]
                y_train = y[idx_train]
                y_test = y[idx_test]

                training_dataset.data = x_train
                training_dataset.targets = y_train
                validation_dataset.data = x_test
                validation_dataset.targets = y_test

            training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size,
                                                          shuffle=True)  # Batch size of 100 i.e to work with 100 images at a time

            validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

            return training_loader, validation_loader

        except Exception as e:
            print("Select CIFAR10")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def load_tiny_imagenet(self, n_clients, filename_train, filename_test, non_iid=False):

        try:

            dir_path = "dataset_utils/data/Tiny-ImageNet/raw_data/"

            with open(filename_train, 'rb') as handle:
                idx_train = pickle.load(handle)

            with open(filename_test, 'rb') as handle:
                idx_test = pickle.load(handle)

            training_dataset, validation_dataset = load_data(dir_path + "tiny-imagenet-200/")
            x = training_dataset.imgs
            x = np.concatenate((x, validation_dataset.imgs))
            y = training_dataset.targets
            y = np.concatenate((y, validation_dataset.targets))
            samples = training_dataset.samples
            samples = np.concatenate((samples, validation_dataset.samples))
            x_train = x[idx_train]
            x_test = x[idx_test]
            y_train = y[idx_train]
            y_test = y[idx_test]
            samples_train = samples[idx_train]
            samples_test = samples[idx_test]

            training_dataset.data = x_train
            training_dataset.targets = y_train
            training_dataset.samples = samples_train
            validation_dataset.data = x_test
            validation_dataset.targets = y_test
            validation_dataset.samples = samples_test

            trainLoader = DataLoader(dataset=training_dataset, batch_size=256, shuffle=True)
            testLoader = DataLoader(dataset=validation_dataset, batch_size=256, shuffle=False)

            return trainLoader, testLoader

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


    def select_dataset(self, dataset_name, n_clients, class_per_client, alpha, non_iid, bath_size):
        try:
            print("recebeu: ", self.cid, dataset_name, n_clients, class_per_client, alpha, non_iid)
            filename_train = f"dataset_utils/data/{dataset_name}/{n_clients}_clients/classes_per_client_{class_per_client}/alpha_{alpha}/{self.cid}/idx_train_{self.cid}.pickle"
            filename_test = f"dataset_utils/data/{dataset_name}/{n_clients}_clients/classes_per_client_{class_per_client}/alpha_{alpha}/{self.cid}/idx_test_{self.cid}.pickle"

            if dataset_name == 'MNIST':
                return self.load_MNIST(n_clients=n_clients, filename_train=filename_train, filename_test=filename_test, non_iid=non_iid)

            elif dataset_name == 'EMNIST':
                return self.load_EMNIST(n_clients=n_clients, filename_train=filename_train, filename_test=filename_test, non_iid=non_iid)

            elif dataset_name == 'CIFAR100':
                return self.load_CIFAR100(n_clients=n_clients, filename_train=filename_train, filename_test=filename_test, non_iid=non_iid)

            elif dataset_name == 'CIFAR10':
                return self.load_CIFAR10(n_clients=n_clients, filename_train=filename_train, filename_test=filename_test, non_iid=non_iid, batch_size=bath_size)

            elif dataset_name == 'Tiny-ImageNet':
                return self.load_tiny_imagenet(n_clients=n_clients, filename_train=filename_train, filename_test=filename_test,
                                         non_iid=non_iid)

            elif dataset_name == 'MotionSense':
                return self.load_MotionSense(n_clients=n_clients, filename_train=filename_train, filename_test=filename_test, non_iid=non_iid)

            elif dataset_name == 'UCIHAR':
                return self.load_UCIHAR(n_clients=n_clients, filename_train=filename_train, filename_test=filename_test, non_iid=non_iid)

        except Exception as e:
            print("select_dataset")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

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



