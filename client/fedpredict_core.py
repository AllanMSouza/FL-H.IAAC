import sys
import copy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from utils.compression_methods.parameters_svd import parameter_svd_write, inverse_parameter_svd_reading, if_reduces_size
from utils.compression_methods.fedkd import fedkd_compression
import os
from torch.nn.parameter import Parameter
import scipy.stats as st
# from torch_cka import CKA
import pandas as pd

import math
import torch
import numpy as np
from analysis.base_plots import line_plot, box_plot, ecdf_plot

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


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

def fedpredict_core(t, T, nt, df):
    try:

        # 9
        if nt == 0:
            global_model_weight = 0
        elif nt == t:
            global_model_weight = 1
        else:
            update_level = 1 / nt
            evolution_level = t / 100
            eq1 = (-update_level - evolution_level) # v1 pior
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
            print("similaridade layer selection: ", df)
            update_level = 1 / nt
            evolution_level = t / 100
            eq1 = (-update_level*df-evolution_level*df)  # v8 ótimo
            eq2 = round(np.exp(eq1), 6)
            shared_layers = int(np.ceil(eq2 * n_layers))

        shared_layers = [i for i in range(shared_layers*2)]

        print("Shared layers: ", shared_layers, " rodada: ", t)

        return shared_layers

    except Exception as e:
        print("fedpredict core server layer selection")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def fedpredict_core_compredict(t, T, nt, layer, compression_range):
    try:
        updated_level = 1/nt
        columns = layer.shape[-1]
        fraction = compression_range/columns
        evolution_level = t / 100

        lamda = 0.2
        # eq1 = (update_level - evolution_level - (1-sm)* lamda) # v3
        # eq1 = (update_level - evolution_level - (1-sm) * lamda)  # v4 bom mas invertido
        # eq1 = (-update_level - evolution_level - sm)  # v5 cai demais e invertido
        # eq1 = (-update_level - evolution_level + sm) # v6 cai demais
        # eq1 = (-update_level**(1/2) - evolution_level - sm) # v7 cai demais
        # eq1 = (-update_level - evolution_level)/2  # v2 ótimo
        # eq1 = (-update_level - evolution_level - norm) / 3  # v3 mesma coisa para alpha 0.1
        # eq1 = (- evolution_level - norm) / 2  # v4 melhor até o momento

        # ============================= v1 pessimo
        # eq1 = (- norm) # v5
        # # eq1 = (update_level - evolution_level + (1 - sm) * 0.2)
        # lamda = 0.5
        # eq2 = round(np.exp(eq1), 6)*lamda
        # # eq2 = (update_level + reference_similarity)/2
        # n_components = int(np.ceil((eq2) * compression_range))
        # ============================ v2
        # eq1 = - evolution_level  # v5
        eq1 = - evolution_level - updated_level # v5
        # eq1 = (update_level - evolution_level + (1 - sm) * 0.2)
        lamda = 0.5
        fc_l = round(np.exp(eq1), 6) # bom
        # fc_l = pow(evolution_level, 1/2) - updated_level # rum
        # fc_l = eq1+1
        # eq2 = (update_level + reference_similarity)/2
        n_components = max(round(fc_l * compression_range), 1)
        print("fracao: ", fc_l, " fraction: ", fraction, " componentes: ", n_components, " de ", compression_range)
        # if n_components == 0 or (fc_l <= (1-t/T) and len(layer.shape) >= 3):
        #     n_components = None

        return n_components

    except Exception as e:
        print("fedpredict core server compredict")
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

    for client_id in range(num_clients):

        client = clients_parameters[client_id]
        client_id = clients_ids[client_id]
        print("cliente antes: ", len(client))

        for layer_index in range(num_layers):
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
                    print("do cliente: ", client_layer.shape, " global: ", global_layer.shape)
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
                            if np.sum(global_layer_k) == 0 or np.sum(client_layer_k) == 0:
                                similarity = 1

                    client_similarity.append(similarity)
                    client_difference['min'].append(abs(difference.min()))
                    client_difference['max'].append(abs(difference.max()))
                    difference_per_layer_vector[layer_index] += np.absolute(difference).flatten().tolist()

                if layer_index not in similarity_per_layer[client_id]:
                    similarity_per_layer[client_id][layer_index] = []
                    difference_per_layer[client_id][layer_index]['min'] = []
                    difference_per_layer[client_id][layer_index]['max'] = []
                # if layer_index == 0:
                #     print("do cliente: ", client_similarity, len(client_similarity))
                similarity_per_layer[client_id][layer_index].append(np.mean(client_similarity))
                difference_per_layer[client_id][layer_index]['min'].append(abs(np.mean(client_difference['min'])))
                difference_per_layer[client_id][layer_index]['max'].append(abs(np.mean(client_difference['max'])))
            else:

                if layer_index not in interest_layers:
                    similarity = 0
                    difference = np.array([0])
                else:
                    cka = CKA()
                    similarity = cka.linear_CKA(global_layer, client_layer)
                    difference = global_layer - client_layer

                similarity_per_layer[client_id][layer_index] = similarity


                client_difference = {'min': [], 'max': []}
                client_difference['min'].append(abs(difference.min()))
                client_difference['max'].append(abs(difference.max()))
                difference_per_layer_vector[layer_index] += np.absolute(difference).flatten().tolist()
                if layer_index not in similarity_per_layer[client_id]:
                    similarity_per_layer[client_id][layer_index] = []
                    difference_per_layer[client_id][layer_index]['min'] = []
                    difference_per_layer[client_id][layer_index]['max'] = []
                difference_per_layer[client_id][layer_index]['min'].append(abs(np.mean(client_difference['min'])))
                difference_per_layer[client_id][layer_index]['max'].append(abs(np.mean(client_difference['max'])))

    layers_mean_similarity = []
    for layer_index in interest_layers:
        similarities = []
        min_difference = []
        max_difference = []
        for client_id in clients_ids:
            similarities.append(similarity_per_layer[client_id][layer_index])
            min_difference += difference_per_layer[client_id][layer_index]['min']
            max_difference += difference_per_layer[client_id][layer_index]['max']

        mean = np.mean(similarities)
        similarity_per_layer_list[layer_index].append(mean)
        layers_mean_similarity.append(mean)
        mean_similarity_per_layer[layer_index]['mean'] = mean
        # mean_similarity_per_layer[layer_index]['ci'] = st.norm.interval(alpha=0.95, loc=np.mean(similarities), scale=st.sem(similarities))[1] - np.mean(similarities)
        # mean_difference_per_layer[layer_index]['min'] = np.mean(min_difference)
        # mean_difference_per_layer[layer_index]['max'] = np.mean(max_difference)
        print("""similaridade (camada {}): {}""".format(layer_index, mean_similarity_per_layer[layer_index]))
    for layer in difference_per_layer_vector:
        if np.sum(difference_per_layer_vector[layer]) == 0:
            continue
        # df = pd.DataFrame({'Difference': difference_per_layer_vector[layer], 'x': [i for i in range(len(difference_per_layer_vectore[layer]))]})
        # box_plot(df=df, base_dir='', file_name="""boxplot_difference_layer_{}_round_{}_dataset_{}_alpha_{}""".format(str(layer), str(server_round), dataset, alpha), x_column=None, y_column='Difference', title='Difference between global and local parameters', y_lim=True, y_max=0.065)

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

# ===========================================================================================

# FedPredict server

def get_size(parameter):
	try:
		return parameter.nbytes
	except Exception as e:
		print("get_size")
		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def fedpredict_server(parameters, client_evaluate_list, fedpredict_clients_metrics, evaluate_config, similarity_between_layers_per_round, mean_similarity_per_round, server_round, num_rounds, comment, compression, df, layers_compression_range):
    client_evaluate_list_fedpredict = []
    accuracy = 0
    mean_similarity_per_layer = similarity_between_layers_per_round[server_round]
    mean_similarity = mean_similarity_per_round[server_round]
    size_of_parameters = []
    parameters = parameters_to_ndarrays(parameters)

    # Reuse previously compressed parameters
    previously_reduced_parameters = {}
    print("compression technique: ", compression)
    fedkd = None
    for client_tuple in client_evaluate_list:
        client = client_tuple[0]
        client_id = str(client.cid)
        config = copy.copy(evaluate_config)
        client_config = fedpredict_clients_metrics[str(client.cid)]
        nt = client_config['nt']
        if nt != 0 and nt in previously_reduced_parameters:
            process_parameters = False
        else:
            process_parameters = True
        config['nt'] = nt
        config['metrics'] = client_config
        config['last_global_accuracy'] = accuracy
        config['total_server_rounds'] = num_rounds
        try:
            config['total_server_rounds'] = int(comment)
        except:
            pass
        M = [i for i in range(len(parameters))]
        parameters_to_send = None
        if nt == 0 and compression != "fedkd":
            config['M'] = []
            config['df'] = df
            config['decompress'] = False
            config['layers_fraction'] = 0
            evaluate_ins = EvaluateIns(ndarrays_to_parameters(np.array([])), config)
            # print("Evaluate enviar: ", client_id, [i.shape for i in parameters_to_ndarrays(parameters_to_send)])
            # print("enviar referencia: ", len(parameters), len(parameters_to_ndarrays(parameters_to_send)))
            client_evaluate_list_fedpredict.append((client, evaluate_ins))
            continue
        elif compression == 'fedkd':
            if fedkd is None:
                parameters_to_send = parameters_to_send if parameters_to_send is not None else parameters
                print("dentro 1: ", type(parameters_to_send[0]), len(parameters_to_send[0]))
                parameters_to_send, layers_fraction = fedkd_compression(
                    fedpredict_clients_metrics[str(client_id)]['round_of_last_fit'], layers_compression_range,
                    num_rounds, client_id, server_round, len(M), parameters_to_send)
                fedkd = parameters_to_send
            else:
                parameters_to_send = fedkd
            config['decompress'] = True
            config['M'] = M
            config['df'] = df
            config['layers_fraction'] = layers_fraction
            evaluate_ins = EvaluateIns(ndarrays_to_parameters(parameters_to_send), config)
            # print("Evaluate enviar: ", client_id, [i.shape for i in parameters_to_ndarrays(parameters_to_send)])
            # print("enviar referencia: ", len(parameters), len(parameters_to_ndarrays(parameters_to_send)))
            client_evaluate_list_fedpredict.append((client, evaluate_ins))
            continue
        elif compression == "no":
            config['M'] = [i for i in range(len(parameters))]
            config['df'] = df
            config['decompress'] = False
            config['layers_fraction'] = 1
            evaluate_ins = EvaluateIns(ndarrays_to_parameters(parameters), config)
            # print("Evaluate enviar: ", client_id, [i.shape for i in parameters_to_ndarrays(parameters_to_send)])
            # print("enviar referencia: ", len(parameters), len(parameters_to_ndarrays(parameters_to_send)))
            client_evaluate_list_fedpredict.append((client, evaluate_ins))
            continue

        parameters_to_send = None

        print("Tamanho parametros antes: ", sum([i.nbytes for i in parameters]))
        # for i in range(len(parameters)):
        #
        #     if parameters[i].ndim >= 2:
        #         for j in range(len(parameters[i])):
        #
        #             print(" fora i: ", i, " j: ", len(parameters[i][j]),
        #                   type(parameters[i][j]))
        #     else:
        #         print(parameters[i][j])



        if process_parameters:
            if "dls" in compression:
                parameters_to_send, M = dls(fedpredict_clients_metrics[client_id]['first_round'],
                                            mean_similarity_per_layer, mean_similarity,
                                            parameters, server_round, nt, num_rounds, df, size_of_parameters,
                                            client_id, comment)
                print("Tamanho parametros als: ", sum(i.nbytes for i in parameters_to_send))
            layers_fraction = []
            if 'compredict' in compression:
                parameters_to_send = parameters_to_send if parameters_to_send is not None else parameters
                parameters_to_send, layers_fraction = compredict(
                    fedpredict_clients_metrics[str(client_id)]['round_of_last_fit'], layers_compression_range,
                    num_rounds, client_id, server_round, len(M), parameters_to_send)
                config['decompress'] = True
            else:
                config['decompress'] = False
                print("nao igual")

            print("Novos parametros para nt: ", nt)
            decompress = config['decompress']
            previously_reduced_parameters[nt] = [copy.deepcopy(parameters_to_send), M, layers_fraction, decompress]

        else:
            print("Reutilizou parametros de nt: ", nt)
            parameters_to_send, M, layers_fraction, decompress = previously_reduced_parameters[nt]

        parameters_to_send = [np.array(i) for i in parameters_to_send]
        print("Tamanho parametros compredict: ", sum(i.nbytes for i in parameters_to_send))
        for i in range(1, len(parameters)):
            size_of_parameters.append(get_size(parameters[i]))
        fedpredict_clients_metrics[str(client.cid)]['acc_bytes_rate'] = size_of_parameters
        config['M'] = M
        config['df'] = df
        config['decompress'] = decompress
        config['layers_fraction'] = layers_fraction
        evaluate_ins = EvaluateIns(ndarrays_to_parameters(parameters_to_send), config)
        # print("Evaluate enviar: ", client_id, [i.shape for i in parameters_to_ndarrays(parameters_to_send)])
        # print("enviar referencia: ", len(parameters), len(parameters_to_ndarrays(parameters_to_send)))
        client_evaluate_list_fedpredict.append((client, evaluate_ins))

    return client_evaluate_list_fedpredict


def compredict(round_of_last_fit, layers_comppression_range, num_rounds, client_id, server_round, M, parameter):

    nt = server_round - round_of_last_fit
    layers_fraction = []
    if round_of_last_fit >= 1:
        n_components_list = []
        for i in range(M):
            # if i % 2 == 0:
            layer = parameter[i]
            if len(layer.shape) >= 2:

                compression_range = layers_comppression_range[i]
                if compression_range > 0:
                    n_components = fedpredict_core_compredict(server_round, num_rounds, nt, layer, compression_range)
                else:
                    n_components = None
            else:
                n_components = None

            if n_components is None:
                layers_fraction.append(1)
            else:
                layers_fraction.append(n_components / layer.shape[-1])

            n_components_list.append(n_components)

        print("Vetor de componentes: ", n_components_list)

        parameter = parameter_svd_write(parameter, n_components_list)

        # print("Client: ", client_id, " round: ", server_round, " nt: ", nt, " norm: ", np.mean(gradient_norm), " camadas: ", M, " todos: ", gradient_norm)
        print("modelo compredict: ", [i.shape for i in parameter])

    else:
        new_parameter = []
        for param in parameter:
            new_parameter.append(param)
            new_parameter.append(np.array([]))
            new_parameter.append(np.array([]))

        parameter = new_parameter

        layers_fraction = [1] * len(parameter)

    return parameter, layers_fraction

def layer_compression_range(model_shape):

    layers_range = []
    for shape in model_shape:

        layer_range = 0
        if len(shape) >= 2:
            shape = shape[-2:]

            col = shape[1]
            for n_components in range(1, col+1):
                if if_reduces_size(shape, n_components):
                    layer_range = n_components
                else:
                    break

        layers_range.append(layer_range)

    return layers_range

def dls(first_round, mean_similarity_per_layer, mean_similarity, parameters,
        server_round, nt, num_rounds, df, size_of_layers, client_id, comment):
    try:
        M = [i for i in range(len(parameters))]
        n_layers = len(parameters) / 2

        size_list = []
        for i in range(len(parameters)):
            tamanho = get_size(parameters[i])
            # print("inicio camada: ", i, " tamanho: ", tamanho, " shape: ", parameters[i].shape)
            size_list.append(tamanho)
        # print("Tamanho total parametros original: ", sum(size_list), sys.getsizeof(fl.common.ndarrays_to_parameters(parameters)))

        # print("quantidade de camadas: ", len(parameters), [i.shape for i in parameters], " comment: ", comment)
        # print("layer selection evaluate: ", self.compression, self.comment)
        if first_round != -1:
            # baixo-cima
            M = fedpredict_core_layer_selection(t=server_round, T=num_rounds, nt=nt, n_layers=n_layers,
                                                size_per_layer=size_of_layers,
                                                mean_similarity_per_layer=mean_similarity_per_layer, df=df)
            new_parameters = []
            for i in range(len(parameters)):
                if i in M:
                    new_parameters.append(parameters[i])
            parameters = new_parameters

        size_list = []
        for i in range(len(parameters)):
            tamanho = parameters[i].nbytes
            size_list.append(tamanho)

        return parameters, M

    except Exception as e:
        print("_select_layers")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


# ===========================================================================================

# FedPredict client


def fedpredict_client(filename, model, global_parameters, config={}, mode=None):
    # Using 'torch.load'
    try:
        # filename = """./fedpredict_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid, self.cid)
        t = int(config['round'])
        T = int(config['total_server_rounds'])
        client_metrics = config['metrics']
        # Client's metrics
        nt = int(client_metrics['nt'])
        cid = int(config['cid'])
        round_of_last_fit = client_metrics['round_of_last_fit']
        round_of_last_evaluate = client_metrics['round_of_last_evaluate']
        first_round = client_metrics['first_round']
        acc_of_last_fit = client_metrics['acc_of_last_fit']
        acc_of_last_evaluate = client_metrics['acc_of_last_evaluate']
        # Server's metrics
        last_global_accuracy = config['last_global_accuracy']
        # print("chegou")
        M = config['M']
        df = config['df']

        decompress = config['decompress']
        layers_fraction = config['layers_fraction']
        if mode == "kd":
            model_shape = [i.detach().cpu().numpy().shape for i in model.student.parameters()]
        else:
            model_shape = [i.detach().cpu().numpy().shape for i in model.parameters()]
        global_parameters = decompress_global_parameters(global_parameters, model_shape, M, decompress)
        print("shape modelo: ", model_shape)
        print("descomprimido: ", [i.shape for i in global_parameters])
        parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_parameters]

        if len(parameters) != len(M):
            print("diferente", len(parameters), len(M))
            raise Exception("Lenght of parameters is different from M")

        if os.path.exists(filename):
            # Load local parameters to 'self.model'
            print("existe modelo local")
            model.load_state_dict(torch.load(filename))
            model = fedpredict_combine_models(global_parameters, model, t, T, nt, M, df)
        else:
            print("usar modelo global: ", cid)
            if mode is None:
                for old_param, new_param in zip(model.parameters(), global_parameters):
                    old_param.data = new_param.data.clone()
            else:
                model.new_client = True
                for old_param, new_param in zip(model.student.parameters(), global_parameters):
                    old_param.data = new_param.data.clone()

        return model

    except Exception as e:
        print("FedPredict client")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def decompress_global_parameters(compressed_global_model_gradients, model_shape, M, decompress):
    try:
        if decompress and len(compressed_global_model_gradients) > 0:
            decompressed_gradients = inverse_parameter_svd_reading(compressed_global_model_gradients, model_shape,
                                                                   len(M))
            parameters = [Parameter(torch.Tensor(i.tolist())) for i in decompressed_gradients]
        # print("descomprimidos shapes: ", [i.shape for i in parameters])
        else:
            parameters = [Parameter(torch.Tensor(i.tolist())) for i in compressed_global_model_gradients]

        return parameters

    except Exception as e:
        print("decompress")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def fedpredict_combine_models(global_parameters, model, t, T, nt, M, df):
    try:

        local_model_weights, global_model_weight = fedpredict_core(t, T, nt, df)
        count = 0
        for new_param, old_param in zip(global_parameters, model.parameters()):
            if count in M:
                if new_param.shape == old_param.shape:
                    old_param.data = (
                                global_model_weight * new_param.data.clone() + local_model_weights * old_param.data.clone())
                else:
                    print("Não combinou, CNN student: ", new_param.shape, " CNN 3 proto: ", old_param.shape)
            count += 1

        return model

    except Exception as e:
        print("FedPredict combine models")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)