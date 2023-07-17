import tensorflow as tf
import torch
import numpy as np
import random
import pickle
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import torchvision
import torchvision.transforms as transforms
import subprocess
from torchvision.datasets import ImageFolder, DatasetFolder, ImageNet
import torchvision.datasets as datasets
import time
import sys

def load_data_eminist(data_path):
    """Load Emnist (training and val set)."""

    # Load ImageNet and normalize
    traindir = os.path.join(data_path, "train")
    valdir = os.path.join(data_path, "val")
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    trainset = datasets.ImageFolder(
        traindir,
        transform
    )

    valset = datasets.ImageFolder(
        valdir,
        transform
    )

    return trainset, valset

def load_data_imagenet(data_path):
    """Load ImageNet (training and val set)."""

    # Load ImageNet and normalize
    traindir = os.path.join(data_path, "train")
    valdir = os.path.join(data_path, "val")

    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    )

    trainset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    valset = datasets.ImageFolder(
        valdir,
        transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    return trainset, valset


class ImageFolder_custom(DatasetFolder):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if not os.listdir(self.root):
            print("entro")
            command = """cd {} \nwget http://cs231n.stanford.edu/tiny-imagenet-200.zip""".format(self.root)
            subprocess.Popen(command, shell=True).wait()
            command = """cd {} \nunzip tiny-imagenet-200.zip""".format(self.root)
            subprocess.Popen(command, shell=True).wait()
        elif not os.path.exists(self.root + "/tiny-imagenet-200/val/"):
            print("aaa")
            command = """cd {} \nunzip tiny-imagenet-200.zip""".format(self.root)
            subprocess.Popen(command, shell=True).wait()
        imagefolder_obj = ImageFolder(self.root, self.transform, self.target_transform)
        self.loader = imagefolder_obj.loader
        if self.dataidxs is not None:
            self.samples = np.array(imagefolder_obj.samples)[self.dataidxs]
        else:
            self.samples = np.array(imagefolder_obj.samples)

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        target = int(target)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        if self.dataidxs is None:
            return len(self.samples)
        else:
            return len(self.dataidxs)

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

        print("rolutos: ", y_train)

        return x_train, y_train, x_test, y_test

    def load_CIFAR100(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        return x_train, y_train, x_test, y_test

    def load_tiny_imagenet(self):

        dir_path = "data/Tiny-ImageNet/raw_data/"

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

            # Setup directory for train/test data
        config_path = dir_path + "config.json"
        train_path = dir_path + "train/"
        test_path = dir_path + "test/"

        if not os.listdir(dir_path):
            print("entro")
            command = """cd {} \nwget http://cs231n.stanford.edu/tiny-imagenet-200.zip""".format(dir_path)
            subprocess.Popen(command, shell=True).wait()
            command = """cd {} \nunzip tiny-imagenet-200.zip""".format(dir_path)
            subprocess.Popen(command, shell=True).wait()
        elif not os.path.exists(dir_path + "tiny-imagenet-200/val/"):
            print("aaa")
            command = """cd {} \nunzip 'tiny-imagenet-200.zip'""".format(dir_path)
            subprocess.Popen(command, shell=True).wait()

        trainset, valset = load_data_imagenet(dir_path + "tiny-imagenet-200/")

        # trainset = ImageFolder_custom(root=dir_path + '', transform=transform)
        # testset = ImageFolder_custom(root=dir_path + '', transform=transform)
        # trainloader = torch.utils.data.DataLoader(
        #     trainset, batch_size=len(trainset), shuffle=False)
        # testloader = torch.utils.data.DataLoader(
        #     testset, batch_size=len(testset), shuffle=False)
        #
        # print("sam: ", trainset.classes)
        #
        # # for _, train_data in enumerate(trainloader, 0):
        # #     print("oi: ", train_data)
        # #     exit()
        # # for _, test_data in enumerate(testloader, 0):
        # #     testset.data, testset.targets = test_data
        # exit()
        np.random.seed(0)

        dataset_image = []
        dataset_label = []
        dataset_image.extend(trainset.imgs)
        dataset_image.extend(valset.imgs)
        dataset_label.extend(trainset.targets)
        dataset_label.extend(valset.targets)
        dataset_image = np.array(dataset_image)
        dataset_label = np.array(dataset_label)

        print("rotulos: ", dataset_label, dataset_label[0])

        return dataset_image, dataset_label, np.array([]), np.array([])

    def load_emnist(self):

        dir_path = "data/EMNIST/"

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

        trainset = torchvision.datasets.EMNIST(
            root=dir_path + "raw_data", train=True, download=True, transform=transform, split='balanced')
        testset = torchvision.datasets.EMNIST(
            root=dir_path + "raw_data", train=False, download=True, transform=transform, split='balanced')
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=len(trainset.data), shuffle=False)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=len(testset.data), shuffle=False)

        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        for _, test_data in enumerate(testloader, 0):
            testset.data, testset.targets = test_data

        dataset_image = []
        dataset_label = []

        dataset_image.extend(trainset.data.cpu().detach().numpy())
        dataset_image.extend(testset.data.cpu().detach().numpy())
        dataset_label.extend(trainset.targets.cpu().detach().numpy())
        dataset_label.extend(testset.targets.cpu().detach().numpy())
        dataset_image = np.array(dataset_image)
        dataset_label = np.array(dataset_label)

        return dataset_image, dataset_label, np.array([]), np.array([])

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

    def generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Setup directory for train/test data
        config_path = dir_path + "config.json"
        train_path = dir_path + "train/"
        test_path = dir_path + "test/"

        # FIX HTTP Error 403: Forbidden
        from six.moves import urllib
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)

        # Get MNIST data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

        trainset = torchvision.datasets.EMNIST(
            root=dir_path + "raw_data", train=True, download=True, transform=transform, split='balanced')
        testset = torchvision.datasets.EMNIST(
            root=dir_path + "raw_data", train=False, download=True, transform=transform, split='balanced')
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=len(trainset.data), shuffle=False)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=len(testset.data), shuffle=False)

        for _, train_data in enumerate(trainloader, 0):
            trainset.data, trainset.targets = train_data
        for _, test_data in enumerate(testloader, 0):
            testset.data, testset.targets = test_data

        dataset_image = []
        dataset_label = []

        dataset_image.extend(trainset.data.cpu().detach().numpy())
        dataset_image.extend(testset.data.cpu().detach().numpy())
        dataset_label.extend(trainset.targets.cpu().detach().numpy())
        dataset_label.extend(testset.targets.cpu().detach().numpy())
        dataset_image = np.array(dataset_image)
        dataset_label = np.array(dataset_label)

    def select_dataset(self, dataset_name):

        if dataset_name == 'MNIST':
            return self.load_MNIST()

        elif dataset_name == 'CIFAR100':
            return self.load_CIFAR100()

        elif dataset_name == 'CIFAR10':
            return self.load_CIFAR10()

        elif dataset_name == 'Tiny-ImageNet':
            return self.load_tiny_imagenet()

        elif dataset_name == 'EMNIST':
            return self.load_emnist()

        # elif dataset_name == 'MotionSense':
        #     return self.load_MotionSense()
        #
        # elif dataset_name == 'UCIHAR':
        #     return self.load_UCIHAR()