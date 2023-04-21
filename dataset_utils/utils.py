import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch

from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from select_dataset import ManageDatasets

# import torchvision.transforms as transforms
#
# def get_transform(dataset_name: str):
#     transform = None
#
#     if dataset_name == "CIFAR10":
#         transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#     elif dataset_name == "MNIST":
#         transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
#
#     return transform

def separate_data(targets, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=2,
                  batch_size=10, train_size=0.8, alpha=0.1):
    """
        return:
            dataidx_map: dict of client_id and the list of samples' indexes
    """
    least_samples = batch_size / (1 - train_size)
    alpha = alpha  # for Dirichlet distribution

    statistic = [[] for _ in range(num_clients)]

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(targets)))
        idx_for_each_class = []

        for i in range(num_classes):
            idx_for_each_class.append(idxs[targets == i])

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
                selected_clients = selected_clients[:int(num_clients / num_classes * class_per_client)]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                num_samples = np.random.randint(max(num_per / 10, least_samples / num_classes), num_per,
                                                num_selected_clients - 1).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx + num_sample]
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], idx_for_each_class[i][idx:idx + num_sample],
                                                    axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(targets)

        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(targets == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # get statistics
    for client in range(num_clients):
        idxs = dataidx_map[client]
        for i in np.unique(targets[idxs]):
            statistic[client].append((int(i), int(sum(targets[idxs] == i))))

    del targets

    return dataidx_map, statistic


def split_data(dataidx_map, num_clients, train_size):
    # Split dataset
    train_data, test_data = [], []

    for cli_id in range(num_clients):
        cli_idxs = dataidx_map[cli_id]
        np.random.shuffle(cli_idxs)
        test_first_index = int(train_size * len(cli_idxs))
        train_idxs = cli_idxs[:test_first_index]
        test_idxs = cli_idxs[test_first_index:]
        train_data.append(torch.tensor(train_idxs))
        test_data.append(torch.tensor(test_idxs))

    return train_data, test_data

def save_dataloaders(dataset_name="CIFAR10", num_clients=10, num_classes=10, niid=True, balance=False, partition="dir",
                 class_per_client=10,
                 batch_size=10, train_size=0.8, alpha=0.1, dataset_dir="./dataset/", sim_id=0):

    # transform = get_transform(dataset_name)
    x_train, y_train, x_test, y_test = ManageDatasets().select_dataset(dataset_name)
    print("antes: ", y_train.shape, y_test.shape)
    target = np.concatenate((y_train, y_test), axis=0)
    # dataset = Dataset()
    masks, statistic = separate_data(target, num_clients, num_classes, niid, balance, partition, class_per_client,
                                     batch_size, train_size, alpha)

    train_data, test_data = split_data(masks, num_clients, train_size)

    path = os.getcwd() + '/' + dataset_dir
    train_path = path + f"{dataset_name}_train/"
    val_path = path + f"{dataset_name}_val/"
    stats_path = path + f"{dataset_name}_stats/"

    if not os.path.isdir(train_path):
        Path(train_path).mkdir(parents=True, exist_ok=True)
    if not os.path.isdir(val_path):
        Path(val_path).mkdir(parents=True, exist_ok=True)
    if not os.path.isdir(stats_path):
        Path(stats_path).mkdir(parents=True, exist_ok=True)

    torch.save(train_data, train_path + f"sim_{sim_id}.pt")
    torch.save(test_data, val_path + f"sim_{sim_id}.pt")
    torch.save(statistic, stats_path + f"sim_{sim_id}.pt")