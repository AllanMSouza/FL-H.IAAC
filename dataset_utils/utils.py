import os
import pickle
import warnings

from dataset_utils.select_dataset import ManageDatasets
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from dataset_utils.partition.centralized import CentralizedPartition

# from dataset_utils.partition.uniform import UniformPartition
# from dataset_utils.partition.user_index import UserPartition
from dataset_utils.cologne_load import load_dataset_cologne


import dataset_utils.wisdm
from dataset_utils.partition.dirichlet import DirichletPartition
def bar_plot(df, base_dir, file_name, x_column, y_column, title, hue=None, hue_order=None, y_lim=False, y_min=0, y_max=100, log_scale=False, sci=False, x_order=None, ax=None, tipo=None, palette=None):
    Path(base_dir + "png/").mkdir(parents=True, exist_ok=True)
    Path(base_dir + "svg/").mkdir(parents=True, exist_ok=True)
    max_value = df[y_column].max()

    sns.set(style='whitegrid')
    log = ""
    file_name = """{}_barplot""".format(file_name)
    # df[y_column] = df[y_column].round(2)
    if log_scale:
        plt.yscale('log')
        log = "_log_"
    if sci:
        from matplotlib import ticker
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        # formatter.set_powerlimits((-1, 1))
        ax.yaxis.set_major_formatter(formatter)
        # ax.set_ylim([0, 130000])
    if y_lim:
        # y_max = float(max_value)
        plt.ylim([y_min, y_max])


    # errorbar=('ci', 0.95),
    figure = sns.barplot(ax=ax, x=x_column, y=y_column, hue=hue, data=df, hue_order=hue_order,  order=x_order, palette=palette).set_title(title)
    # for bars in ax.containers:
    #     ax.bar_label(bars, fmt='%.f', fontsize=9)
    plt.xticks(rotation=90)
    if ax is None:
        fig, ax = plt.subplots()
        figure = figure.get_figure()
        figure.savefig(base_dir + "png/" + file_name + log + ".png", bbox_inches='tight', dpi=400)
        figure.savefig(base_dir + "svg/" + file_name + log + ".svg", bbox_inches='tight', dpi=400)

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

def separate_data(targets, num_clients, num_classes, dataset, niid=True, balance=False, partition=None, class_per_client=2,
                  batch_size=10, train_size=0.8, alpha=0.1):
    """
        return:
            dataidx_map: dict of client_id and the list of samples' indexes
    """
    np.random.seed(0)
    least_samples = batch_size / (1 - train_size)
    # least_samples = train_size
    alpha = alpha  # for Dirichlet distribution
    least_samples = 20

    print("aq:", partition)

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
        if dataset == 'GTSRB':

            # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
            min_size = 0
            K = num_classes
            max_class = 0
            N = len(targets)
            print("ola: ", class_per_client)

            while min_size < least_samples:
                idx_batch = [[] for _ in range(num_clients)]
                for k in range(K):
                    idx_k = np.where(targets == k)[0]
                    np.random.shuffle(idx_k)
                    proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                    proportions = np.array(
                        [p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                    proportions = proportions / proportions.sum()
                    proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                    idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                    min_size = min([len(idx_j) for idx_j in idx_batch])
                    print("classe: ", k, " min size: ", min_size, " least: ", least_samples)

            for j in range(num_clients):
                dataidx_map[j] = idx_batch[j]
                m = np.take(np.array(targets), idx_batch[j]).max()
                # if m > max_class:
                #     max_class = m
                # print("max ", max_class)

        else:
            # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
            min_size = 0
            K = num_classes
            max_class = 0
            N = len(targets)
            print("ola: ", class_per_client)

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
                m = np.take(np.array(targets), idx_batch[j]).max()
                # if m > max_class:
                #     max_class = m
                # print("max ", max_class)
    else:
        raise NotImplementedError

    # get statistics
    for client in range(num_clients):
        idxs = dataidx_map[client]
        for i in np.unique(targets[idxs]):
            statistic[client].append((int(i), int(sum(targets[idxs] == i))))

    del targets

    return dataidx_map, statistic


def split_data(dataidx_map, target, num_clients, train_size):
    # Split dataset
    train_data, test_data = [], []
    frac = 0.8

    for cli_id in range(num_clients):
        cli_idxs = dataidx_map[cli_id]
        cli_targets = target[cli_idxs]
        # aux = np.array([[i, j] for i, j in zip(cli_idxs, cli_targets)])
        # # print("antes: ", len(aux))
        # aux = resample(aux, n_samples=len(aux) * frac, random_state=0, replace=False, stratify=cli_targets)
        # cli_idxs = np.array([i[0] for i in aux])
        # print("depois: ", len(aux))
        # exit()
        np.random.shuffle(cli_idxs)
        test_first_index = int(train_size * len(cli_idxs))
        train_idxs = cli_idxs[:test_first_index]
        test_idxs = cli_idxs[test_first_index:]
        # train_data.append(torch.tensor(train_idxs))
        # test_data.append(torch.tensor(test_idxs))
        train_data.append(train_idxs)
        test_data.append(test_idxs)

    return train_data, test_data

def save_dataloaders(dataset_name="CIFAR10", num_clients=10, num_classes=10, niid=True, balance=False, partition="dir",
                 class_per_client=10,
                 batch_size=10, train_size=0.8, alpha=0.1, dataset_dir="./dataset/", sim_id=0):

    num_classes = {'Tiny-ImageNet': 200, 'CIFAR10': 10, 'MNIST': 10, 'EMNIST': 47, "State Farm": 10, 'GTSRB': 43}[dataset_name]

    # transform = get_transform(dataset_name)
    x_train, y_train, x_test, y_test = ManageDatasets().select_dataset(dataset_name)

    target = np.concatenate((y_train, y_test), axis=0).astype(int)
    df = pd.DataFrame({'label': target}).value_counts().reset_index()
    df['count'] = df[0].to_numpy()
    df = df[['label', 'count']]
    x_order = df.sort_values(by='count')['label'].tolist()
    # df = pd.DataFrame({'label': df.index.tolist(), 'count': df.tolist()})
    print("ordem: ", x_order)
    bar_plot(df=df, base_dir='', file_name="""{}_label_counts_original""".format(dataset_name), title='', x_column='label', y_column='count', x_order=x_order)
    # print(df)
    # exit()
    if dataset_name == 'GTSRB':
        x = np.concatenate((np.array([i[0] for i in x_train]), np.array([i[0] for i in x_test])), axis=0)
        print(x[:2])
    print("Quantidade de amostras do ", dataset_name, ": ", len(target))
    masks, statistic = separate_data(target, num_clients, num_classes, dataset_name, niid, balance, partition, class_per_client,
                                     batch_size, train_size, alpha)

    train_data, test_data = split_data(masks, target, num_clients, train_size)

    count = 0
    final_target = []
    classes_list = []

    for client_id in range(num_clients):

        index_train = train_data[client_id]
        index_test = test_data[client_id]
        final_target += list(target[index_train]) + list(target[index_test])
        if dataset_name == 'GTSRB':
            x_client = np.concatenate((x[index_train], x[index_test]), axis=0)
            df = pd.DataFrame({'x': x_client, 'index': np.concatenate((index_train, index_test), axis=0), 'type': np.concatenate((np.array(['train'] * len(index_train)), np.array(['test'] * len(index_test))), axis=0)}).drop_duplicates('x')
            index_train = df.query("type == 'train'")['index'].to_numpy()
            index_test = df.query("type == 'test'")['index'].to_numpy()
        # print("""Quantidade de dados de treino para o cliente {}: {}, teste: {}""".format(client_id, len(index_train), len(index_test)))
        classes = len(pd.Series(target[index_train]).unique().tolist())
        classes_list.append(classes)
        print("Original: ", len(index_train), " Sem duplicadas: ", len(pd.Series(index_train).drop_duplicates()), " classes: ",  classes, " teste: ", len(index_test))
        print("Suporte: ", pd.Series(target[index_train]).astype(str).value_counts())
        count += len(index_train) + len(index_test)

        filename_train = """data/{}/{}_clients/classes_per_client_{}/alpha_{}/{}/idx_train_{}.pickle""".format(dataset_name, num_clients, class_per_client, alpha, client_id, client_id)
        filename_test = """data/{}/{}_clients/classes_per_client_{}/alpha_{}/{}/idx_test_{}.pickle""".format(dataset_name, num_clients, class_per_client, alpha, client_id, client_id)
        print("escrever: ", filename_train)
        os.makedirs(os.path.dirname(filename_train), exist_ok=True)
        os.makedirs(os.path.dirname(filename_test), exist_ok=True)

        with open(filename_train, 'wb') as handle:
            pickle.dump(index_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(filename_test, 'wb') as handle:
            pickle.dump(index_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Media de classes: ", pd.Series(np.array(classes_list)/num_classes).describe())
    df = pd.DataFrame({'label': final_target}).value_counts().reset_index()
    df['count'] = df[0].to_numpy()
    df = df[['label', 'count']]
    x_order = df.sort_values(by='count')['label'].tolist()
    # df = pd.DataFrame({'label': df.index.tolist(), 'count': df.tolist()})
    bar_plot(df=df, base_dir='', file_name="""{}_label_counts_final""".format(dataset_name), title='',
             x_column='label', y_column='count', x_order=x_order)

    print("total: ", count)

def save_dataloaders_widsm(dataset_name="WISDM-WATCH", num_clients=10, num_classes=10, niid=True, balance=False, partition="dir",
                 class_per_client=2,
                 batch_size=10, train_size=0.8, alpha=0.1, dataset_dir="./dataset/", sim_id=0):

    num_classes = 12

    # transform = get_transform(dataset_name)
    if dataset_name == "WISDM-WATCH":
        modality = "watch"
    elif dataset_name == "WISDM-P":
        modality = "phone"
    dataset = dataset_utils.wisdm.load_dataset(reprocess=False, modality=modality)
    num_classes = 12
    partition_type = 'dirichlet'
    # dataset_name = 'WISDM-WATCH'
    client_num_per_round = 6

    partition, client_num_in_total, client_num_per_round = get_partition(partition_type,
                                                                         dataset_name,
                                                                         num_classes,
                                                                         num_clients,
                                                                         client_num_per_round,
                                                                         alpha,
                                                                         dataset)

    client_datasets_train = partition(dataset['train'])
    client_datasets_test = partition(dataset['test'])
    print(len(client_datasets_train))
    # exit()
    final_target = []
    classes_list = []

    for client_id in range(len(client_datasets_train)):
        index_train = list(client_datasets_train[client_id].indices)
        index_test = list(client_datasets_test[client_id].indices)
        data_train = client_datasets_train[client_id].dataset.data[index_train]
        data_test = client_datasets_test[client_id].dataset.data[index_test]
        target_train = client_datasets_train[client_id].dataset.targets[index_train]
        target_test = client_datasets_test[client_id].dataset.targets[index_test]
        final_target += list(target_train) + list(target_test)
        print("dimensao: ", target_train.shape)
        # print(target_train)
        classes = len(pd.Series(target_train).unique().tolist())
        classes_list.append(classes)
        # " Sem duplicadas: ", len(pd.Series(data_train).drop_duplicates()),
        print("Original: ", len(data_train),
              " classes: ", classes, " teste: ", len(data_test))
        print("Suporte: \n", pd.Series(target_train).astype(str).value_counts())


        filename_train = """dataset_utils/data/{}/{}_clients/classes_per_client_{}/alpha_{}/{}/idx_train_{}.csv""".format(dataset_name,
                                                                                                               num_clients,
                                                                                                               class_per_client,
                                                                                                               alpha,
                                                                                                               client_id,
                                                                                                               client_id)
        filename_test = """dataset_utils/data/{}/{}_clients/classes_per_client_{}/alpha_{}/{}/idx_test_{}.csv""".format(dataset_name,
                                                                                                             num_clients,
                                                                                                             class_per_client,
                                                                                                             alpha,
                                                                                                             client_id,
                                                                                                             client_id)

        print("escrever: ", filename_train)
        os.makedirs(os.path.dirname(filename_train), exist_ok=True)
        os.makedirs(os.path.dirname(filename_test), exist_ok=True)

        # with open(filename_train, 'wb') as handle:
        #     pickle.dump(index_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # with open(filename_test, 'wb') as handle:
        #     pickle.dump(index_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

        data_train = {"X": data_train.tolist(), "Y": target_train.tolist()}
        data_test = {"X": data_test.tolist(), "Y": target_test.tolist()}

        for df, filename in zip([data_train, data_test], [filename_train, filename_test]):

            pd.DataFrame(df).to_csv(filename, index=False)


def save_dataloaders_cologne(dataset_name="Cologne", num_clients=10, num_classes=10, niid=True, balance=False, partition="dir",
                 class_per_client=2,
                 batch_size=10, train_size=0.8, alpha=0.1, dataset_dir="./dataset/", sim_id=0):

    num_classes = 12

    modality = ""
    dataset = load_dataset_cologne(reprocess=False)

    num_classes = 10
    partition_type = 'dirichlet'
    # dataset_name = 'WISDM-WATCH'
    client_num_per_round = 6

    partition, client_num_in_total, client_num_per_round = get_partition(partition_type,
                                                                         dataset_name,
                                                                         num_classes,
                                                                         num_clients,
                                                                         client_num_per_round,
                                                                         alpha,
                                                                         dataset)

    client_datasets_train = partition(dataset['train'])
    client_datasets_test = partition(dataset['test'])
    print(len(client_datasets_train))
    # exit()
    final_target = []
    classes_list = []

    for client_id in range(len(client_datasets_train)):
        index_train = list(client_datasets_train[client_id].indices)
        index_test = list(client_datasets_test[client_id].indices)
        data_train = client_datasets_train[client_id].dataset.data[index_train]
        data_test = client_datasets_test[client_id].dataset.data[index_test]
        target_train = client_datasets_train[client_id].dataset.targets[index_train]
        target_test = client_datasets_test[client_id].dataset.targets[index_test]
        final_target += list(target_train) + list(target_test)
        print("dimensao: ", target_train.shape)
        # print(target_train)
        classes = len(pd.Series(target_train).unique().tolist())
        classes_list.append(classes)
        # " Sem duplicadas: ", len(pd.Series(data_train).drop_duplicates()),
        print("Original: ", len(data_train),
              " classes: ", classes, " teste: ", len(data_test))
        print("Suporte: \n", pd.Series(target_train).astype(str).value_counts())


        filename_train = """dataset_utils/data/{}/{}_clients/classes_per_client_{}/alpha_{}/{}/idx_train_{}.csv""".format(dataset_name,
                                                                                                               num_clients,
                                                                                                               class_per_client,
                                                                                                               alpha,
                                                                                                               client_id,
                                                                                                               client_id)
        filename_test = """dataset_utils/data/{}/{}_clients/classes_per_client_{}/alpha_{}/{}/idx_test_{}.csv""".format(dataset_name,
                                                                                                             num_clients,
                                                                                                             class_per_client,
                                                                                                             alpha,
                                                                                                             client_id,
                                                                                                             client_id)

        print("escrever: ", filename_train)
        os.makedirs(os.path.dirname(filename_train), exist_ok=True)
        os.makedirs(os.path.dirname(filename_test), exist_ok=True)

        # with open(filename_train, 'wb') as handle:
        #     pickle.dump(index_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #
        # with open(filename_test, 'wb') as handle:
        #     pickle.dump(index_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

        data_train = {"X": data_train.tolist(), "Y": target_train.tolist()}
        data_test = {"X": data_test.tolist(), "Y": target_test.tolist()}

        for df, filename in zip([data_train, data_test], [filename_train, filename_test]):

            pd.DataFrame(df).to_csv(filename, index=False)

def get_partition(partition_type, dataset_name, num_classes, client_num_in_total, client_num_per_round, alpha, dataset):
    # if partition_type == 'user' and dataset_name in {'wisdm', 'widar', 'visdrone'}:
    #     partition = UserPartition(dataset['split']['train'])
    #     client_num_in_total = len(dataset['split']['train'].keys())
    # elif partition_type == 'uniform':
    #     partition = UniformPartition(num_class=num_classes, num_clients=client_num_in_total)
    if partition_type == 'dirichlet':
        if alpha is None:
            warnings.warn('alpha is not set, using default value 0.1')
            alpha = 0.1
        partition = DirichletPartition(num_class=num_classes, num_clients=client_num_in_total, alpha=alpha)
    # elif partition_type == 'central':
    #     partition = CentralizedPartition()
    #     client_num_per_round = 1
    #     client_num_in_total = 1
    else:
        raise ValueError(f'Partition {partition_type} type not supported')

    return partition, client_num_in_total, client_num_per_round