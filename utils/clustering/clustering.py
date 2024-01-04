import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pandas as pd

import numpy as np
import random
import threading
import numpy as np
import math
from abc import ABC, abstractmethod
from logging import INFO
from typing import Dict, List, Optional

from flwr.common.logger import log

from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as spc
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import OPTICS

import keras.backend as K


################# for clustering

class GreedyKCenter(object):
    def fit(self, points, k):
        centers = []
        centers_index = []
        # Initialize distances
        distances = [np.inf for u in points]
        # Initialize cluster labels
        labels = [np.inf for u in points]

        for cluster in range(k):
            # Let u be the point of P such that d[u] is maximum
            u_index = distances.index(max(distances))
            u = points[u_index]
            # u is the next cluster center
            centers.append(u)
            centers_index.append(u_index)

            # Update distance to nearest center
            for i, v in enumerate(points):
                distance_to_u = self.distance(u, v)  # Calculate from v to u
                if distance_to_u < distances[i]:
                    distances[i] = distance_to_u
                    labels[i] = cluster

            # Update the bottleneck distance
            max_distance = max(distances)

        # Return centers, labels, max delta, labels
        self.centers = centers
        self.centers_index = centers_index
        self.max_distance = max_distance
        self.labels = labels

    @staticmethod
    def distance(u, v):
        displacement = u - v
        return np.sqrt(displacement.dot(displacement))


def server_Hclusters(matrix, plot_dendrogram, n_clients, n_clusters,
                     server_round, cluster_round, path):
    pdist = spc.distance.pdist(matrix)
    linkage = spc.linkage(pdist, method='ward')
    min_link = linkage[0][2]
    max_link = linkage[-1][2]

    th = max_link
    for i in np.linspace(min_link, max_link, 5000):

        le = len(pd.Series(spc.fcluster(linkage, i, 'distance')).unique())
        if le == n_clusters:
            th = i

    idx = spc.fcluster(linkage, th, 'distance')
    print(idx)

    if plot_dendrogram and (server_round == cluster_round):
        dendrogram(linkage, color_threshold=th)
        # plt.savefig(f'results/clusters_{dataset}_{n_clients}clients_{n_clusters}clusters.png')
        plt.savefig(path + f'clusters_{n_clients}clients_{n_clusters}clusters.png')

    return idx


def server_Hclusters2(matrix, plot_dendrogram=False):
    pdist = spc.distance.pdist(matrix)
    linkage = spc.linkage(pdist, method='ward')

    max_link = linkage[-1][2]
    t = max_link / 3  # como escolher?

    idx = spc.fcluster(linkage, t=t, criterion='distance')
    print(idx)

    if plot_dendrogram:
        dendrogram(linkage)
        plt.show()

    return idx


def server_AffinityClustering(matrix):
    af = AffinityPropagation(random_state=0).fit(matrix)
    idx = af.labels_

    return idx


def server_OPTICSClustering(matrix):
    clustering = OPTICS(min_samples=2).fit(1 / matrix)
    idx = clustering.labels_


def server_KCenterClustering(weights, k):
    print("k: ", k)
    KCenter = GreedyKCenter()
    KCenter.fit(weights, k)
    idx = KCenter.labels

    return idx


def make_clusters(matrix, plot_dendrogram, n_clients, n_clusters,
                  server_round, cluster_round, path,
                  clustering_method, models):
    if clustering_method == 'Affinity':
        idx = server_AffinityClustering(matrix)
        return idx

    if clustering_method == 'HC':
        idx = server_Hclusters(matrix=matrix, plot_dendrogram=plot_dendrogram,
                               n_clients=n_clients, n_clusters=n_clusters,
                               server_round=server_round, cluster_round=cluster_round,
                               path=path)
        return idx

    if clustering_method == 'KCenter':
        idx = server_KCenterClustering(models, k=n_clusters)
        return idx

    if clustering_method == 'Random':
        unique = 0
        while unique != n_clusters:
            idx = list(np.random.randint(0, n_clusters, n_clients))
            unique = np.unique(np.array(idx))
            unique = len(unique)
        return idx


####################### for similarity

def get_layer_outputs(model, layer, input_data, learning_phase=1):
    layer_fn = K.function(model.input, layer.output)
    return layer_fn(input_data)


def cka(X, Y):
    # Implements linear CKA as in Kornblith et al. (2019)
    X = X.copy()
    Y = Y.copy()

    # Center X and Y
    X -= X.mean(axis=0)
    Y -= Y.mean(axis=0)

    # Calculate CKA
    XTX = X.T.dot(X)
    YTY = Y.T.dot(Y)
    YTX = Y.T.dot(X)

    return (YTX ** 2).sum() / (np.sqrt((XTX ** 2).sum() * (YTY ** 2).sum()))


def calcule_similarity(models, metric, n_clients):
    # actvs = models['actv_last']
    # if metric == 'CKA':
    #     matrix = np.zeros((len(actvs), len(actvs)))
    #
    #     for i, a in enumerate(actvs):
    #         for j, b in enumerate(actvs):
    #             x = int(models['cids'][i])
    #             y = int(models['cids'][j])
    #
    #             matrix[x][y] = cka(a, b)

    last = models['last_layer']
    if metric == 'weights':
        matrix = np.zeros((n_clients, n_clients))

        for i, a in enumerate(last):
            for j, b in enumerate(last):
                x = int(models['cids'][i])
                y = int(models['cids'][j])
                cosine_similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                if np.isnan(cosine_similarity):
                    # print("nan: ", np.sum(a), np.sum(b))
                    cosine_similarity = 0
                matrix[x][y] = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))  # cos similarity
    return matrix


#################### for client selection
def sample(
        clients,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
        CL=True,
        selection=None,
        acc=None,
        decay_factor=None,
        server_round=None,
        idx=None,
        cluster_round=0,
        POC_perc_of_clients=0.5,
        times_selected=[]):
    """Sample a number of Flower ClientProxy instances."""
    # Block until at least num_clients are connected.
    if min_num_clients is None:
        min_num_clients = num_clients
    # Sample clients which meet the criterion
    available_cids = list(clients)
    if criterion is not None:
        available_cids = [
            cid for cid in available_cids if criterion.select(clients[cid])
        ]

    if num_clients > len(available_cids):
        log(
            INFO,
            "Sampling failed: number of available clients"
            " (%s) is less than number of requested clients (%s).",
            len(available_cids),
            num_clients,
        )
        return []

    sampled_cids = available_cids.copy()

    if (idx is not None) and (server_round > cluster_round) and CL:
        selected_clients = []
        for cluster_idx in np.unique(idx):  # passa por todos os clusters
            cluster = []

            for client in available_cids:
                if idx[int(client)] == cluster_idx:  # salva apenas os clientes pertencentes aquele cluster
                    cluster.append(int(client))

            if selection == 'Random':
                selected_clients.append(str(random.sample(cluster, 1)[0]))

            if selection == 'POC':
                acc_cluster = list(np.array(acc)[cluster])
                sorted_cluster = [str(x) for _, x in sorted(zip(acc_cluster, cluster))]
                clients2select = max(int(float(len(cluster)) * float(POC_perc_of_clients)), 1)
                for c in sorted_cluster[:clients2select]:
                    selected_clients.append(c)

            if selection == 'Less_Selected':
                clients_to_select_in_cluster = []
                c = [times_selected[i] for i in cluster]
                number_less_selected = min(c)
                # client_to_select = times_selected.index(number_less_selected)
                client_to_select = list(
                    pd.Series(times_selected)[pd.Series(times_selected) == number_less_selected].index)

                for i in client_to_select:  # if there are more than one with same times_selected
                    if i in cluster:
                        clients_to_select_in_cluster.append(i)

                selected_clients.append(
                    str(random.sample(clients_to_select_in_cluster, 1)[0]))  # select only one per cluster

        sampled_cids = selected_clients.copy()

    if selection == 'All':
        sampled_cids = random.sample(available_cids, num_clients)

    return [clients[cid] for cid in sampled_cids]