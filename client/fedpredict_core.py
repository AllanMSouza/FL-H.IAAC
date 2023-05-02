import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
# from torch_cka import CKA

import math
import torch
import numpy as np


class CKA(object):
    def __init__(self):
        pass

    def centering(self, K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n
        return np.dot(np.dot(H, K), H)

    def rbf(self, X, sigma=None):
        GX = np.dot(X, X.T)
        KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
        if sigma is None:
            mdist = np.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = np.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return np.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = X @ X.T
        L_Y = Y @ Y.T
        return np.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = np.sqrt(self.linear_HSIC(X, X))
        var2 = np.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = np.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = np.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)

def fedpredict_core(t, T, nt):
    try:

        # 9
        if nt == 0:
            global_model_weight = 0
        else:
            # evitar que um modelo que treinou na rodada atual não utilize parâmetros globais pois esse foi atualizado após o seu treinamento
            # normalizar dentro de 0 e 1
            # updated_level = 1/rounds_without_fit
            # updated_level = 1 - max(0, -acc_of_last_fit+self.accuracy_of_last_round_of_evalute)
            # if acc_of_last_evaluate < last_global_accuracy:
            # updated_level = max(-last_global_accuracy + acc_of_last_evaluate, 0)
            # else:
            update_level = 1 / nt
            # evolutionary_level = (server_round / 50)
            # print("client id: ", self.cid, " primeiro round", self.first_round)
            evolution_level = t / T

            # print("el servidor: ", el, " el local: ", evolutionary_level)

            eq1 = (-update_level - evolution_level)
            eq2 = round(np.exp(eq1), 6)
            global_model_weight = eq2

        local_model_weights = 1 - global_model_weight

        print("rodada: ", t, " rounds sem fit: ", nt, "\npeso global: ", global_model_weight, " peso local: ",
              local_model_weights)

        return local_model_weights, global_model_weight

    except Exception as e:
        print("fedpredict core")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


# def set_parameters_to_model(parameters, model_name):
#     # print("tamanho: ", self.input_shape, " dispositivo: ", self.device)


def fedpredict_layerwise_similarity(global_parameter, clients_parameters, clients_ids):

    num_layers = len(global_parameter)
    num_clients = len(clients_parameters)
    similarity_per_layer = {i: {} for i in clients_ids}
    mean_similarity_per_layer = {i: 0 for i in clients_ids}

    for j in range(num_clients):

        client = clients_parameters[j]
        client_id = clients_ids[j]

        for i in range(num_layers):
            client_layer = client[i]
            global_layer = global_parameter[i]
            if np.ndim(global_layer) == 1:
                global_layer = np.reshape(global_layer, (len(global_layer), 1))
            if np.ndim(client_layer) == 1:
                client_layer = np.reshape(client_layer, (len(client_layer), 1))
            print("apos global: ", global_layer.shape)
            print("apos cliente: ", client_layer.shape)
            # CNN
            if np.ndim(global_layer) == 4:
                client_similarity = []
                for k in range(len(global_layer)):
                    global_layer_k = global_layer[k][0]
                    client_layer_k = client_layer[k][0]
                    cka = CKA()
                    similarity = cka.linear_CKA(global_layer_k, client_layer_k)
                    client_similarity.append(similarity)
                if i not in similarity_per_layer[client_id]:
                    similarity_per_layer[client_id][i] = []
                similarity_per_layer[client_id][i].append(np.mean(client_similarity))
            else:
            # for x, y in zip(global_layer, client_layer):
                cka = CKA()
                similarity = cka.linear_CKA(global_layer, client_layer)
                similarity_per_layer[client_id][i] = similarity

    for i in range(num_layers):
        similarities = []
        for client_id in clients_ids:
            similarities.append(similarity_per_layer[client_id][i])

        mean_similarity_per_layer[i] = np.mean(similarities)

    print("""similaridade (camada {}): {}""".format(i, mean_similarity_per_layer[i]))

    return similarity_per_layer, mean_similarity_per_layer