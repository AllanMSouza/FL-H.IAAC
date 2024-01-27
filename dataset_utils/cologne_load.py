import os.path
from typing import Mapping

import numpy as np
import pandas
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset_utils.utils_wisdm import train_test_split, make_split


class WISDMDataset(Dataset):
    """
    A PyTorch Dataset class for the WISDM dataset.
    """

    def __init__(self, data):
        """
        Initialize the dataset with data mapping.
        Args:
            data (Mapping[str, list[np.ndarray | int]]): A dictionary containing the data and targets.
        """
        self.data = np.array(data[0], dtype=np.float32)
        self.targets = np.array(data[1], dtype=np.float32)

    def __getitem__(self, index):
        """
        Get an item from the dataset by index.
        Args:
            index (int): The index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The data and target tensors for the specified index.
        """
        return self.data[index], self.targets[index]

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.data[1])


def define_cols(df: pandas.DataFrame, prefix='acc'):
    """
    Define columns in the DataFrame and drop the 'null' column.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        prefix (str, optional): The prefix for the x, y, and z columns. Defaults to 'acc'.

    Returns:
        pandas.DataFrame: The DataFrame with columns renamed and the 'null' column dropped.
    """
    # columns = ['subject', 'activity', 'timestamp', f'x_{prefix}', f'y_{prefix}', f'z_{prefix}', 'null']
    # df.columns = columns
    df = df.drop('null', axis=1)
    return df


def filter_merge_interval(label_column, dfa: pandas.DataFrame, dfg: pandas.DataFrame, act_df: pandas.DataFrame):
    """
    Filter and merge accelerometer and gyroscope DataFrames based on timestamps and activity codes.

    Args:
        dfa (pandas.DataFrame): The accelerometer DataFrame.
        dfg (pandas.DataFrame): The gyroscope DataFrame.
        act_df (pandas.DataFrame): The activity DataFrame.

    Returns:
        pandas.DataFrame: The merged and filtered DataFrame.
    """
    t0_a = dfa['timestamp'].min()
    t0_g = dfg['timestamp'].min()
    t1_a = dfa['timestamp'].max()
    t1_g = dfg['timestamp'].max()

    t0 = max(t0_a, t0_g)
    t1 = min(t1_a, t1_g)
    dfa = dfa[(t0 <= dfa['timestamp']) & (dfa['timestamp'] <= t1)]
    dfg = dfg[(t0 <= dfg['timestamp']) & (dfg['timestamp'] <= t1)]

    df = dfa.merge(dfg.drop(dfg.columns[[0, 1]], axis=1), how='inner', on='timestamp')
    df = df.sort_values(by='timestamp')
    df = df.dropna()
    codes = act_df.code.unique()
    df = df[df.activity.isin(codes)]
    replace_codes = zip(act_df.code, act_df.fcode)
    for code, replacement_code in replace_codes:
        df[label_column] = df[label_column].replace(code, replacement_code)
    return df


def process_dataset():
    """
    Process the WISDM dataset by reading accelerometer and gyroscope data and merging them.

    Args:
        act_df (pandas.DataFrame): The activity DataFrame.
        data_path (str): The path to the directory containing the dataset.

    Returns:
        pandas.DataFrame: The concatenated and merged DataFrame of accelerometer and gyroscope data.
    """
    import pandas as pd
    import glob
    import os

    path = "/home/claudio/Documentos/pycharm_projects/FL-H.IAAC/dataset_utils/data/cologne/data/" # use your path
    all_files = os.listdir(path)[:1000]

    li = []

    for i in range(len(all_files)):
        filename = all_files[i]
        client_id = filename.replace(".txt", "")
        df = pd.read_csv(path + filename, index_col=None, header=None, sep='\t')
        df['cid'] = np.array([i] * len(df))
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    print(df.columns)
    df.columns = np.array(['timestamp', 'x', 'y', 'speed', 'accel', 'noise', 'co2', 'co', 'fuel_consumption', 'road_id', 'route_id', 'cid'])
    df['road_id'] = np.array([str(j).replace("#", "").replace("!", "") for j in df['road_id'].tolist()])
    df['route_id'] = np.array([str(j).replace("#", "").replace("!", "") for j in df['route_id'].tolist()])
    df['cid'] = np.array([int(str(j).replace("#", "").replace("!", "")) for j in df['cid'].tolist()])
    print(df.shape, df.columns)
    print(df.iloc[0-1])

    cols = ['road_id', 'route_id']

    for col in cols:
        col_data = df[col].unique().tolist()
        color_map = {col_data[i]: i for i in range(len(col_data))}
        df[col] = df[col].map(color_map)

    # dfs = []
    # for i in tqdm(range(1600, 1651)):
    #     df_wa = define_cols(
    #         pd.read_csv(f'{data_path}/raw/{modality}/accel/data_{i}_accel_{modality}.txt', header=None, sep=',|;',
    #                     engine='python'))
    #     df_wg = define_cols(
    #         pd.read_csv(f'{data_path}/raw/{modality}/gyro/data_{i}_gyro_{modality}.txt', header=None, sep=',|;',
    #                     engine='python'),
    #         prefix='gyro')
    #     dfs.append(filter_merge_interval(df_wa, df_wg, act_df))
    return df


def normalize_data(df):
    """
    Normalize the data in the DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The normalized DataFrame.
    """
    cols = ['speed', 'accel', 'noise', 'co2', 'co', 'fuel_consumption']
    df = df.dropna()
    for col in tqdm(cols):
        df[col] = (df[col] - df[col].mean()) / df[col].std()
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    cols = ['co2', 'co', 'fuel_consumption']
    n_classes = 10

    for col in tqdm(cols):
        sorted = df[col].to_numpy()[np.argsort(df[col].tolist())]
        space = len(sorted)//n_classes
        bins = []
        for j in range(0, len(sorted) - space, space):
            bin = sorted[j+space]
            if bin not in bins:
                bins.append(bin)

        bins = [0] + bins

        bins = [(bins[i-1], bins[i]) for i in range(1, len(bins))]

        bins = pd.IntervalIndex.from_tuples(bins)

        r = pd.cut(df[col], bins)
        r = np.array([i for i in r.tolist()])
        unique = pd.Series(r).unique().tolist()
        convert = {unique[i]: i for i in range(len(unique))}
        df[col+"_label"] = df[col].map(convert)

    df = df.dropna()

    print(np.unique(df['fuel_consumption_label'].to_numpy(), return_counts=True))

    return df


def get_processed_dataframe(reprocess=False, modality='watch'):
    """
    Load or reprocess the processed WISDM dataset.

    Args:
        reprocess (bool, optional): Whether to reprocess the dataset. Defaults to False.

    Returns:
        pandas.DataFrame: The processed DataFrame.
    """
    dir_path = "dataset_utils/data/cologne/"
    # if os.path.exists(dir_path + f'processed_cologne.csv') and not reprocess and False:
    #     return pd.read_csv(dir_path + f'processed_cologne.csv', index_col=0)
    processed_df = process_dataset()
    processed_df = normalize_data(processed_df)
    processed_df.to_csv(dir_path + f'processed_cologne.csv')
    return processed_df


def create_dataset(df, clients=None, window=200, overlap=0.2):
    """
    Create a dataset from the input DataFrame based on the specified parameters.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        clients (list, optional): The list of client ids. Defaults to None.
        window (int, optional): The window size for segmenting data. Defaults to 200.
        overlap (float, optional): The overlap ratio between windows. Defaults to 0.5.

    Returns:
        tuple: A tuple containing a dictionary with 'X' and 'Y' keys, and a dictionary with client indices.
    """
    # 'timestamp', 'x', 'y', 'speed', 'accel', 'noise', 'co2', 'co',
    #        'fuel_consumption', 'road_id', 'route_id', 'cid', 'co2_label',
    #        'co_label', 'fuel_consumption_label']
    if clients is None:
        clients = list(range(len(df['cid'].unique().tolist())))
    c_idxs = {}
    idx = 0
    X = []
    Y = []
    print("ja: ", window)
    print("co label: ", df['co_label'])
    for client in tqdm(clients):
        c_idxs[client] = []
        data = df[df['cid'] == client].sort_values(by='timestamp')
        data = data.dropna()
        labels = data['co_label'].tolist()
        future_labels = []
        for i in range(len(labels)-1):
            future_labels.append(labels[i+1])
        data = data.tail(len(data) - 1)
        data['co_label'] = np.array(future_labels)
        unique_labels = data['co_label'].unique()
        for activity in unique_labels:
            df_f = data[data['co_label'] == activity]
            for i in range(window, len(df_f), 1):
                if i + window > len(df_f):
                    continue
                X.append(df_f[['timestamp', 'speed', 'accel', 'noise', 'co2', 'co',
       'fuel_consumption', 'road_id', 'route_id', 'cid']].iloc[i:i + window].to_numpy().tolist())
                Y.append(activity)
                c_idxs[client].append(idx)
                idx += 1

    return (X, Y), c_idxs


def split_dataset(data: dict, client_mapping_train: dict, client_mapping_test: dict):
    """
    Split the dataset into train and test sets based on the client mappings.

    Args:
        data (dict): The input dataset as a dictionary with 'X' and 'Y' keys.
        client_mapping_train (dict): A dictionary containing the client indices for the training set.
        client_mapping_test (dict): A dictionary containing the client indices for the test set.

    Returns:
        tuple: A tuple containing the train and test WISDMDatasets, and a dictionary with train and test mappings.
    """
    all_train, mapping_train = make_split(client_mapping_train)
    all_test, mapping_test = make_split(client_mapping_test)
    x = data[0]
    y = data[1]
    train_data = ([x[i] for i in all_train], [y[i] for i in all_train])
    test_data = ([x[i] for i in all_test], [y[i] for i in all_test])
    return WISDMDataset(train_data), WISDMDataset(test_data), {'train': mapping_train, 'test': mapping_test}


def load_dataset_cologne(window=10, overlap=0, reprocess=True, split=0.8):
    """
    Load the WISDM dataset, either from disk or by reprocessing it based on the specified parameters.

    Args:
        window (int, optional): The window size for segmenting data. Defaults to 200.
        overlap (float, optional): The overlap ratio between windows. Defaults to 0.5.
        reprocess (bool, optional): Whether to reprocess the dataset. Defaults to True.
        split (float, optional): The ratio for the train/test split. Defaults to 0.8.
        modality (str, optional): The modality to use. Defaults to 'watch'.

    Returns:
        dict: A dictionary containing the full dataset, train and test datasets, client mapping, and split.
    """
    dir_path = "/dataset_utils/data/data_cologne/data/"
    if os.path.exists(dir_path+f'cologne.dt') and not reprocess:
        return torch.load(dir_path + f'cologne.dt')

    if reprocess or not os.path.exists(dir_path + f'cologne.dt'):
        processed_df = get_processed_dataframe(reprocess=reprocess)
        print('pro: ', processed_df)
        clients = list(range(1000))
        data, idx = create_dataset(processed_df, clients=None, window=window, overlap=overlap)
        # print(data[0][0].shape, data[0][0])
        # exit()
        # exit()
        dataset = WISDMDataset(data)
        client_mapping_train, client_mapping_test = train_test_split(idx, split)
        train_dataset, test_dataset, split = split_dataset(data, client_mapping_train, client_mapping_test)
        print("Count treino: ", np.unique(train_dataset.targets, return_counts=True))
        print("Count teste: ", np.unique(test_dataset.targets, return_counts=True))
        # exit()

        torch.save({
            'full_dataset': dataset,
            'train': train_dataset,
            'test': test_dataset,
            'client_mapping': idx,
            'split': split
        }, "dataset_utils/data/cologne/" + f'cologne.dt')
    data = torch.load("dataset_utils/data/cologne/"+ f'cologne.dt')
    # print("ler data")
    # exit()
    return data


if __name__ == '__main__':
    dt = load_dataset()
    print(len(dt['train']))
