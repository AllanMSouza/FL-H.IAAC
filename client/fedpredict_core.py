import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import scipy.stats as st
# from torch_cka import CKA
import pandas as pd

import math
import torch
import numpy as np
from analysis.base_plots import line_plot, box_plot, ecdf_plot


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

def fedpredict_core(t, T, nt, sm):
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
            evolution_level = t / 100

            # print("el servidor: ", el, " el local: ", evolutionary_level)
            eq1 = (-update_level - evolution_level) # v1 pior
            # eq1 = (-update_level - evolution_level - sm) # v3 melhor
            eq2 = round(np.exp(eq1), 6)
            global_model_weight = eq2

        local_model_weights = 1 - global_model_weight

        print("rodada: ", t, " rounds sem fit: ", nt, "\npeso global: ", global_model_weight, " peso local: ",
              local_model_weights)

        return local_model_weights, global_model_weight

    except Exception as e:
        print("fedpredict core")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def fedpredict_core_layer_selection(t, T, nt, n_layers, size_per_layer, mean_similarity_per_layer, df):
    try:

        # 9
        if nt == 0:
            shared_layers = 0
        else:
            # reference_similarity = (mean_similarity_per_layer[int(n_layers*2-2)]['mean'] + mean_similarity_per_layer[int(n_layers*2-1)]['mean'])/2
            # reference_similarity = min([mean_similarity_per_layer[int(n_layers * 2 - 2)]['mean'],
            #                         mean_similarity_per_layer[int(n_layers * 2 - 1)]['mean']])
            # # reference_similarity = mean_similarity_per_layer[int(n_layers * 2 - 2)]['mean']
            # penalty = abs(mean_similarity_per_layer[int(n_layers * 2 - 2)]['mean'] -
            #                         mean_similarity_per_layer[int(n_layers * 2 - 1)]['mean'])
            print("similaridade layer selection: ", df)
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
            evolution_level = t / 100

            # print("el servidor: ", el, " el local: ", evolutionary_level)

            # eq1 = (update_level - evolution_level+reference_similarity) #v2

            lamda = 0.2
            # eq1 = (update_level - evolution_level - (1-sm)* lamda) # v3
            # eq1 = (update_level - evolution_level - (1-sm) * lamda)  # v4 bom mas invertido
            # eq1 = (-update_level - evolution_level - df)  # v5 cai demais e invertido
            # eq1 = (-update_level - evolution_level + sm) # v6 cai demais
            # eq1 = (-update_level**(1/2) - evolution_level - sm) # v7 cai demais
            eq1 = (-update_level*evolution_level*df)  # v8 ótimo
            # eq1 = (-update_level - evolution_level - df) # v9
            # eq1 = (-update_level - evolution_level - df/3)  # v10 reduz bem mas acurácia ruim para alpha 2 e 5
            # eq1 = (-update_level - evolution_level - df**2)  # v11 reduz bem mas acurácia ruim para alpha 2 e 5
            # eq1 = (-update_level - evolution_level - df ** 3)  # v12 reduz bem mas acurácia ruim para alpha 2 e 5
            # eq1 = (update_level - evolution_level + (1 - sm) * 0.2)
            eq2 = round(np.exp(eq1), 6)
            # eq2 = (update_level + reference_similarity)/2
            shared_layers = int(np.ceil(eq2 * n_layers))

        shared_layers = [i for i in range(shared_layers*2)]

        print("Shared layers: ", shared_layers, " rodada: ", t)

        return shared_layers

    except Exception as e:
        print("fedpredict core server")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def fedpredict_core_compredict(t, T, nt, layer, layer_norm, compression_range):
    try:

        # 9
        layers_n_components = []
        # evitar que um modelo que treinou na rodada atual não utilize parâmetros globais pois esse foi atualizado após o seu treinamento
        # normalizar dentro de 0 e 1
        # updated_level = 1/rounds_without_fit
        # updated_level = 1 - max(0, -acc_of_last_fit+self.accuracy_of_last_round_of_evalute)
        # if acc_of_last_evaluate < last_global_accuracy:
        # updated_level = max(-last_global_accuracy + acc_of_last_evaluate, 0)
        # else:
        norm = min(layer_norm, 1)
        update_level = 1 / nt
        # evolutionary_level = (server_round / 50)
        # print("client id: ", self.cid, " primeiro round", self.first_round)
        evolution_level = t / T

        # print("el servidor: ", el, " el local: ", evolutionary_level)

        # eq1 = (update_level - evolution_level+reference_similarity) #v2

        lamda = 0.2
        # eq1 = (update_level - evolution_level - (1-sm)* lamda) # v3
        # eq1 = (update_level - evolution_level - (1-sm) * lamda)  # v4 bom mas invertido
        # eq1 = (-update_level - evolution_level - sm)  # v5 cai demais e invertido
        # eq1 = (-update_level - evolution_level + sm) # v6 cai demais
        # eq1 = (-update_level**(1/2) - evolution_level - sm) # v7 cai demais
        # eq1 = (-update_level - evolution_level)/2  # v2 ótimo
        # eq1 = (-update_level - evolution_level - norm) / 3  # v3 mesma coisa para alpha 0.1
        # eq1 = (- evolution_level - norm) / 2  # v4 melhor até o momento
        eq1 = (- norm) # v5
        # eq1 = (update_level - evolution_level + (1 - sm) * 0.2)
        eq2 = round(np.exp(eq1), 6)
        # eq2 = (update_level + reference_similarity)/2
        n_components = int(np.ceil((1-eq2) * compression_range))
        print("fracao: ", 1 - eq2)

        return n_components

    except Exception as e:
        print("fedpredict core server")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


# def set_parameters_to_model(parameters, model_name):
#     # print("tamanho: ", self.input_shape, " dispositivo: ", self.device)

def fedpredict_similarity_per_round_rate(similarities, num_layers, current_round, round_window):

    similarities_list = {i: [] for i in range(num_layers)}
    similarity_rate = {i: 0 for i in range(num_layers)}
    initial_round = 1

    for round in range(initial_round, current_round):

        for layer in range(num_layers):

            similarities_list[round].append(similarities[round])

    for layer in range(num_layers):

        similarity_rate[layer] = (similarities_list[current_round] - similarities_list[initial_round])/round_window


def fedpredict_layerwise_similarity(global_parameter, clients_parameters, clients_ids, server_round, dataset, alpha, similarity_per_layer_list):

    num_layers = len(global_parameter)
    num_clients = len(clients_parameters)
    print("global: ", num_layers)
    similarity_per_layer = {i: {} for i in clients_ids}
    # interest_layers = [0, 1, int(num_layers/2)-2, int(num_layers/2)-1, num_layers-2, num_layers-1]
    interest_layers = [0, num_layers - 2]
    difference_per_layer = {i: {j: {'min': [], 'max': []} for j in range(num_layers)} for i in clients_ids}
    difference_per_layer_vector = {j: [] for j in range(num_layers)}
    mean_similarity_per_layer = {i: {'mean': 0, 'ci': 0} for i in range(num_layers)}
    mean_difference_per_layer = {i: {'min': 0, 'max': 0} for i in range(num_layers)}
    # print("primeira: ", num_layers)


    for client_id in range(num_clients):

        client = clients_parameters[client_id]
        client_id = clients_ids[client_id]

        for layer_index in range(num_layers):
            # print("tamanho maximo: ", num_layers)
            client_layer = client[layer_index]
            global_layer = global_parameter[layer_index]
            if np.ndim(global_layer) == 1:
                global_layer = np.reshape(global_layer, (len(global_layer), 1))
            if np.ndim(client_layer) == 1:
                client_layer = np.reshape(client_layer, (len(client_layer), 1))
            # CNN
            if np.ndim(global_layer) == 4:
                client_similarity = []
                client_difference = {'min': [], 'max': []}
                for k in range(len(global_layer)):
                    global_layer_k = global_layer[k][0]
                    client_layer_k = client_layer[k][0]

                    # if gradient:
                    #     client_layer_k = global_layer_k - client_layer_k
                    cka = CKA()
                    if layer_index not in interest_layers:
                        similarity = 0
                        difference = np.array([0])
                    else:
                        similarity = cka.linear_CKA(global_layer_k, client_layer_k)
                        difference = global_layer_k - client_layer_k
                        if np.isnan(similarity):
                            print("entr: ", global_layer_k, client_layer_k)
                            if np.sum(global_layer_k) == 0 or np.sum(client_layer_k) == 0:
                                similarity = 1

                    client_similarity.append(similarity)
                    client_difference['min'].append(abs(difference.min()))
                    client_difference['max'].append(abs(difference.max()))
                    difference_per_layer_vector[layer_index] += np.absolute(difference).flatten().tolist()
                # print("Diferença max: ", i, k, j, client_difference['max'])
                if layer_index not in similarity_per_layer[client_id]:
                    similarity_per_layer[client_id][layer_index] = []
                    difference_per_layer[client_id][layer_index]['min'] = []
                    difference_per_layer[client_id][layer_index]['max'] = []
                if layer_index == 0:
                    print("do cliente: ", client_similarity, len(client_similarity))
                similarity_per_layer[client_id][layer_index].append(np.mean(client_similarity))
                difference_per_layer[client_id][layer_index]['min'].append(abs(np.mean(client_difference['min'])))
                difference_per_layer[client_id][layer_index]['max'].append(abs(np.mean(client_difference['max'])))
            else:
            # for x, y in zip(global_layer, client_layer):
                if layer_index not in interest_layers:
                    similarity = 0
                    difference = np.array([0])
                else:
                    cka = CKA()
                    # client_layer = global_layer - client_layer
                    similarity = cka.linear_CKA(global_layer, client_layer)
                    # print("par: ", len(global_layer), len(client_layer), layer_index)
                    difference = global_layer - client_layer

                similarity_per_layer[client_id][layer_index] = similarity


                client_difference = {'min': [], 'max': []}
                client_difference['min'].append(abs(difference.min()))
                client_difference['max'].append(abs(difference.max()))
                difference_per_layer_vector[layer_index] += np.absolute(difference).flatten().tolist()
                # print("Diferença max: ", i, k, j, client_difference['max'])
                if layer_index not in similarity_per_layer[client_id]:
                    similarity_per_layer[client_id][layer_index] = []
                    difference_per_layer[client_id][layer_index]['min'] = []
                    difference_per_layer[client_id][layer_index]['max'] = []
                difference_per_layer[client_id][layer_index]['min'].append(abs(np.mean(client_difference['min'])))
                difference_per_layer[client_id][layer_index]['max'].append(abs(np.mean(client_difference['max'])))

    layers_mean_similarity = []
    for layer_index in interest_layers:
        # if layer_index % 2 != 0:
        #     continue
        similarities = []
        min_difference = []
        max_difference = []
        for client_id in clients_ids:
            similarities.append(similarity_per_layer[client_id][layer_index])
            min_difference += difference_per_layer[client_id][layer_index]['min']
            max_difference += difference_per_layer[client_id][layer_index]['max']

        print("si list: ", similarities)
        mean = np.mean(similarities)
        similarity_per_layer_list[layer_index].append(mean)
        layers_mean_similarity.append(mean)
        mean_similarity_per_layer[layer_index]['mean'] = mean
        mean_similarity_per_layer[layer_index]['ci'] = st.norm.interval(alpha=0.95, loc=np.mean(similarities), scale=st.sem(similarities))[1] - np.mean(similarities)
        mean_difference_per_layer[layer_index]['min'] = np.mean(min_difference)
        mean_difference_per_layer[layer_index]['max'] = np.mean(max_difference)
        print("""similaridade (camada {}): {}""".format(layer_index, mean_similarity_per_layer[layer_index]))
    for layer in difference_per_layer_vector:
        if np.sum(difference_per_layer_vector[layer]) == 0:
            continue
        df = pd.DataFrame({'Difference': difference_per_layer_vector[layer], 'x': [i for i in range(len(difference_per_layer_vector[layer]))]})
        # line_plot(df=df, base_dir='', file_name="""lineplot_difference_layer_{}_round_{}_dataset_{}_alpha_{}""".format(str(layer), str(server_round), dataset, alpha), x_column='x', y_column='Difference', title='Difference between global and local parameters', y_lim=True, y_max=0.065)
        box_plot(df=df, base_dir='', file_name="""boxplot_difference_layer_{}_round_{}_dataset_{}_alpha_{}""".format(str(layer), str(server_round), dataset, alpha), x_column=None, y_column='Difference', title='Difference between global and local parameters', y_lim=True, y_max=0.065)
        # ecdf_plot(df=df, base_dir='', file_name="""ecdf_difference_layer_{}_round_dataset_{}_alpha_{}""".format(str(layer), str(server_round), dataset, alpha), x_column='Difference', y_column=None, title='Difference between global and local parameters', y_lim=True, y_max=0.065)
        # print("Camada: ", layer, " Diferença média: ", pd.Series(difference_per_layer_vector[layer]).describe()['mean'])



    # decimals_layer = decimals_per_layer(mean_difference_per_layer)
    # print("Diferença por camada: ", mean_difference_per_layer)
    # print("Decimals layer: ", decimals_layer)

    return similarity_per_layer, mean_similarity_per_layer, np.mean(layers_mean_similarity), similarity_per_layer_list

def decimals_per_layer(mean_difference_per_layer):

    window = 1
    precisions = {}
    for layer in mean_difference_per_layer:

        n1 = mean_difference_per_layer[layer]['min']
        n2 = mean_difference_per_layer[layer]['max']
        n = min([n1, n2])
        zeros = 0

        if not np.isnan(n):
            n = str(n)
            n = n.split(".")[1]

            for digit in n:

                if digit == "0":
                    zeros += 1
                else:
                    break

            precisions[layer] = zeros + window

        else:
            precisions[layer] = 9

    return precisions