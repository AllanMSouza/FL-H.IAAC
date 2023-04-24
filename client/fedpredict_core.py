import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
#from torch_cka import CKA

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

def fedpredict_layerwise_similarity(global_parameter, clients_parameters):

    num_layers = len(global_parameter)
    num_clients = len(clients_parameters)
    similarity_per_layer = [[]] * num_layers

    for i in range(num_layers):
        if i % 2 != 0:
            continue

        global_layer = global_parameter[i]

        for j in range(num_clients):

            client = clients_parameters[j]
            client_layer = client[i]
            print("global: ", global_layer.shape)
            print("cliente: ", client_layer.shape)
            if np.ndim(global_layer) == 1:
                global_layer = np.reshape(global_layer, (len(global_layer), 1))
            if np.ndim(client_layer) == 1:
                client_layer = np.reshape(client_layer, (len(client_layer), 1))
                print("apos global: ", global_layer.shape)
                print("apos cliente: ", client_layer.shape)
            # for x, y in zip(global_layer, client_layer):
            similarity = euclidean_distances(global_layer, client_layer)
            similarity = np.mean(similarity.flatten())
            similarity_per_layer[i].append(similarity)

    for i in range(num_layers):

        similarity_per_layer[i] = np.mean(similarity_per_layer[i])

    print("similaridade: ", similarity_per_layer)