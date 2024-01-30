from abc import ABC, abstractmethod
import flwr as fl
import numpy as np
import math
import os
import time
import csv
import random
import sys
import pandas as pd
import torch
from torch.nn.parameter import Parameter
import copy
from functools import reduce

from server.common_base_server import FedCDMBaseServer
from client.fedpredict_core import fedpredict_core_layer_selection, fedpredict_layerwise_similarity, fedpredict_core_compredict, dls, layer_compression_range, compredict, fedpredict_server
from utils.compression_methods.parameters_svd import if_reduces_size
from utils.compression_methods.sparsification import sparse_crs_top_k, to_dense, client_model_non_zero_indexes
from analysis.base_plots import bar_plot, line_plot

from pathlib import Path
import shutil

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

from utils.compression_methods.parameters_svd import parameter_svd_write, inverse_parameter_svd_reading

def get_size(parameter):
    try:
        # #print("recebeu: ", parameter.shape, parameter.ndim)
        # if type(parameter) == np.float32:
        # 	#print("caso 1: ", map(sys.getsizeof, parameter))
        # 	return map(sys.getsizeof, parameter)
        # if parameter.ndim <= 2:
        # 	#print("Caso 2: ", sum(map(sys.getsizeof, parameter)))
        # 	return sum(map(sys.getsizeof, parameter))
        #
        # else:
        # 	tamanho = 0
        # 	#print("Caso 3")
        # 	for i in range(len(parameter)):
        # 		tamanho += get_size(parameter[i])
        #
        # 	return tamanho
        return parameter.nbytes
    except Exception as e:
        print("get_size")
        print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def aggregate(results, t):
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for parameters, num_examples, classes_proportion, count, t, pattern in results])

    print("total ag: ", num_examples_total)

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in parameters] for parameters, num_examples, classes_proportion, count, t, pattern in results
    ]

    # weighted_weights = [
    #     [layer * num_examples for layer in weights] for weights, num_examples in results
    # ]

    # if len(results) == 2:
    #     print("igual a 1 p1 round ", t)
    #     print(results[0][0][0])

    weighted_class_proportion = [np.array(classes_proportion) * num_examples  for parameters, num_examples, classes_proportion, count, t, pattern in results]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]

    # if len(results) == 2:
    #     print("igual a 1 p2")
    #     print(weights_prime[0])

    classes_proportion_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_class_proportion)
    ]

    classes_proportion_prime = None
    for classes_proportion in weighted_class_proportion:
        if classes_proportion_prime is None:
            classes_proportion_prime = classes_proportion
        else:
            classes_proportion_prime = np.add(classes_proportion_prime, classes_proportion)


    classes_proportion_prime = classes_proportion_prime / num_examples_total

    return weights_prime, classes_proportion_prime

class FedCDMWithFedPredictBaseServer(FedCDMBaseServer):

    def __init__(self,
                 aggregation_method,
                 n_classes,
                 fraction_fit,
                 num_clients,
                 num_rounds,
                 args,
                 num_epochs,
                 model,
                 type,
                 server_learning_rate=1,
                 server_momentum=1,
                 decay=0,
                 perc_of_clients=0,
                 dataset='',
                 strategy_name='FedCDM_with_FedPredict',
                 non_iid=False,
                 model_name='',
                 new_clients=False,
                 new_clients_train=False):

        super().__init__(aggregation_method=aggregation_method,
                         n_classes=n_classes,
                         fraction_fit=fraction_fit,
                         num_clients=num_clients,
                         num_rounds=num_rounds,
                         args=args,
                         num_epochs=num_epochs,
                         decay=decay,
                         model=model,
                         perc_of_clients=perc_of_clients,
                         dataset=dataset,
                         strategy_name=strategy_name,
                         model_name=model_name,
                         new_clients=new_clients,
                         new_clients_train=new_clients_train,
                         type=type)

        self.n_rate = float(args.n_rate)
        self.server_learning_rate = server_learning_rate
        self.server_momentum = server_momentum
        self.momentum_vector = None
        self.model = model
        self.window_of_previous_accs = 4
        self.server_opt = (self.server_momentum != 0.0) or (
                self.server_learning_rate != 1.0)

        # self.set_initial_parameters()
        self.create_folder(strategy_name)
        self.similarity_between_layers_per_round = {}
        self.similarity_between_layers_per_round_and_client = {}
        self.model_size = None
        self.similarity_list_per_layer = None
        self.initial_similarity = 0
        self.current_similarity = 0
        self.parameters_aggregated_gradient = {}
        self.parameters_aggregated_checkpoint = {}
        self.layers_compression_range = []
        self.gradient_norm = []
        self.gradient_norm_round = []
        self.gradient_norm_nt = []
        self.T = int(args.T)
        self.clients_model_non_zero_indexes = {}
        self.last_layer_parameters_per_class = [[np.array([]), 0, np.array([]), -1, [], []] for i in range(self.num_clients)]
        self.client_last_layer = [[]] * self.num_clients
        self.report = ""
        self.rounds_to_change_pattern = [int(0.7 * self.num_rounds)]

    def create_folder(self, strategy_name):

        super().create_folder(strategy_name)
        directory = """{}_saved_weights/{}/""".format(strategy_name.lower(), self.model_name)
        print("di: ", directory)
        # if Path(directory).exists():
        #     print("re:")
        #     shutil.rmtree(directory)
        for i in range(self.num_clients):
            # Path("""{}_saved_weights/{}/{}/""".format(strategy_name.lower(), self.model_name, i)).mkdir(
            #     parents=True, exist_ok=True)
            pd.DataFrame({'current_round': [-1], 'classes_distribution': ["[]"], 'round_of_last_fit': [-1], 'round_of_last_evaluate': [-1], 'acc_of_last_fit': [0], 'first_round': [-1], 'acc_of_last_evaluate': [0]}).to_csv("""{}_saved_weights/{}/{}/{}.csv""".format(strategy_name.lower(), self.model_name, i, i), index=False)

    def _calculate_evolution_level(self, server_round):
        try:
            # If the number of rounds so far is low
            if server_round < self.window_of_previous_accs:
                return server_round/self.num_rounds

            acc_list = np.array(list(self.accuracy_history.values()))
            last_acc = acc_list[-1]
            reference_acc = acc_list[-self.window_of_previous_accs]

            # To detect if new clients were added
            if reference_acc > last_acc and server_round == 36:
                # Drop in the performance. New clients where introduced
                t = self._get_round_of_the_most_similar_previous_acc(acc_list, last_acc)
                el = t/self.num_rounds
            else:
                el = (server_round + 1)/self.num_rounds

            return el
        except Exception as e:
            print("calculate evolution level")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


    def _get_round_of_the_most_similar_previous_acc(self, acc_list, last_acc):

        try:
            # Get the round based on the minimum difference between accuracies.
            # It adds +1 to adjust index that starts from 0
            t = np.argmin(acc_list-last_acc) + 1

            return t
        except Exception as e:
            print("get round of the most similar previous acc")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


    def _update_fedpredict_metrics(self, server_round):

        try:
            # self.fedpredict_metrics['acc'] = self.accuracy_history
            # It works as a table where for each round it has a line with two columns for the accuracy and
            # the evolution level of the respective round
            self.fedpredict_metrics['round_acc_el'] = {int(round): {'acc': self.accuracy_history[round], 'el': round/self.num_rounds} for round in self.accuracy_history}
            # print("Metricas dos clientes: ", self.clients_metrics)
            # self.fedpredict_metrics['nt'] = self.clients_metrics['nt']
            self.fedpredict_metrics['el'] = self._calculate_evolution_level(server_round)
        except Exception as e:
            print("update fedpredict metrics")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def weigthed_classes_proportion(self, num_examples_list, classes_proportion_list):

        num_examples_total = sum([num_examples for num_examples in num_examples_list])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [layer * num_examples for layer in weights] for weights, num_examples in zip(classes_proportion_list, num_examples_list)
        ]

        # Compute average weights of each layer
        weights_prime: NDArrays = [
            reduce(np.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]

        return weights_prime

    # def last_layer_parameter(self, last_layer_parameters_per_class, clients_class_proportion):
    #
    # 	for parameter, num_examples, classes_proportion in current_major_class_parameter_list:

    def aggregate_last_layer_parameters(self, last_layer_parameters, patterns, clients_num_examples_list, classes_proportion_list, n_classes, server_round):

        current_last_layer_parameters_per_class = {i: [] for i in range(self.num_clients)}

        for i in range(len(last_layer_parameters)):

            parameters = last_layer_parameters[i]
            num_examples = clients_num_examples_list[i]
            classes_proportion = np.array(classes_proportion_list[i])
            pattern = patterns[i]

            major_class = np.argmax(classes_proportion)

            current_last_layer_parameters_per_class[pattern].append([parameters, num_examples, classes_proportion, 1, [server_round], [pattern]])

        for i in range(len(self.last_layer_parameters_per_class)):

            print("----------")
            current_major_class_parameter_list = []
            previous_major_class_parameter = self.last_layer_parameters_per_class[i]
            # previous_major_class_parameter = [previous_major_class_parameter[0], int(pre)]
            current_major_class_parameter_list += current_last_layer_parameters_per_class[i]
            if len(current_major_class_parameter_list) >= 1:
                print("""Atual adicinou a classe {} tamanho {}""".format(i, len(current_major_class_parameter_list)))
            if previous_major_class_parameter[1] > 0:
                print("""Antigo já existente classe {} tamanho {}""".format(i, len(previous_major_class_parameter)))
                # current_major_class_parameter_list.append(previous_major_class_parameter)

            if len(current_major_class_parameter_list) == 0:
                continue

            total_samples = sum([num_examples for parameters, num_examples, classes_proportion, count, t, pattern in current_major_class_parameter_list])
            total_aggregations = sum([count for parameters, num_examples, classes_proportion, count, t, pattern in
                                 current_major_class_parameter_list])

            parameters_weighted_sum, classes_proportion_weighted_sum = aggregate(current_major_class_parameter_list, server_round)
            print("as: ", total_samples, "classe p: ", i, " rodada: ", server_round, " total: ", total_samples, "classes ag: ", len(classes_proportion_weighted_sum))
            self.last_layer_parameters_per_class[i] = [parameters_weighted_sum, total_samples, classes_proportion_weighted_sum, total_aggregations, self.last_layer_parameters_per_class[i][4] + [server_round], self.last_layer_parameters_per_class[i][5] + [current_major_class_parameter_list[i][5] for i in range(len(current_major_class_parameter_list))]]

            print("agregou camada")

    def aggregate_fit(self, server_round, results, failures):

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        if server_round == 1:
            self.model_shape = [i.shape for i in parameters_to_ndarrays(parameters_aggregated)]
            self.model_size = len(self.model_shape)
            self.similarity_list_per_layer = {i: [] for i in range(self.model_size)}
            self.layers_compression_range = layer_compression_range(self.model_shape)
            print("shape do modelo: ", self.model_shape)
            print("tamanho do modelo: ", self.model_size)
            # print("similaridade inicial: ", self.similarity_list_per_layer)
            print("range: ", self.layers_compression_range)
        weights_results = []
        clients_parameters = []
        clients_num_examples_list = []
        clients_ids = []
        last_layer_parameters = []
        patterns = []
        self.report += """\n ------- Rodada {} -------""".format(server_round)
        self.report += """\n //// Treino ////"""
        for _, fit_res in results:
            client_id = str(fit_res.metrics['cid'])
            pattern = int(fit_res.metrics['pattern'])
            clients_ids.append(client_id)
            parameters = fl.common.parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            clients_parameters.append(parameters)
            last_layer_parameters.append(parameters)
            patterns.append(pattern)
            self.report += """\n cliente {} padrao {}""".format(client_id, pattern)
            clients_num_examples_list.append(num_examples)
            print("""clientee {} padrao {} rodada {}""".format(client_id, pattern, server_round))
            self.client_last_layer[pattern] = parameters

        # self.aggregate_last_layer_parameters(last_layer_parameters, patterns, clients_num_examples_list, classes_proportion_list,
        #                                      self.n_classes, server_round)

        if server_round == self.num_rounds:

            print("informacao")
            for i in range(len(self.last_layer_parameters_per_class)):

                print("""padrao {} agregou {} vezes, treinado nas rodadas {}, padroes {}""".format(i, self.last_layer_parameters_per_class[i][3], self.last_layer_parameters_per_class[i][4], self.last_layer_parameters_per_class[i][5]))


        # self.pattern_prportion()

        if self.use_gradient:
            global_parameter = [current - previous for current, previous in zip(parameters_to_ndarrays(parameters_aggregated), self.previous_global_parameters[server_round-1])]
        else:
            global_parameter = self.previous_global_parameters[server_round-1]

        # aggregated_classes_proportion = self.weigthed_classes_proportion(clients_num_examples_list, classes_proportion_list)

        # print("proporcao: ", aggregated_classes_proportion)

        # for i in range(len(self.client_last_layer)):
        #
        #
        #     self.client_last_layer[i] = fl.common.parameters_to_ndarrays(parameters_aggregated)

        np.random.seed(server_round)
        flag = bool(int(np.random.binomial(1, 0.2, 1)))
        if server_round == 1:
            flag = True
        print("Flag: ", flag)
        if "dls" in self.compression:
            if flag:
                self.similarity_between_layers_per_round_and_client[server_round], self.similarity_between_layers_per_round[server_round], self.mean_similarity_per_round[server_round], self.similarity_list_per_layer = fedpredict_layerwise_similarity(global_parameter, clients_parameters, clients_ids, server_round, self.dataset, str(self.alpha), self.similarity_list_per_layer)
                self.df = max(0, abs(np.mean(self.similarity_list_per_layer[0]) - np.mean(self.similarity_list_per_layer[self.model_size - 2])))
            else:
                self.similarity_between_layers_per_round_and_client[server_round], self.similarity_between_layers_per_round[
                server_round], self.mean_similarity_per_round[
                server_round], self.similarity_list_per_layer = self.similarity_between_layers_per_round_and_client[server_round-1], self.similarity_between_layers_per_round[
                server_round-1], self.mean_similarity_per_round[
                server_round-1], self.similarity_list_per_layer
        else:
            self.similarity_between_layers_per_round[server_round] = []
            self.mean_similarity_per_round[server_round] = 0
            self.similarity_between_layers_per_round_and_client[server_round] = []
            self.df = 1

        # self.parameters_aggregated_checkpoint[server_round] = parameters_to_ndarrays(parameters_aggregated)

        if server_round == self.num_rounds:
            print("Relatório")
            print(self.report)

        # if server_round == 3:
        # 	self.calculate_initial_similarity(server_round)

        return parameters_aggregated, metrics_aggregated

    def configure_evaluate(self, server_round, parameters, client_manager):
        client_evaluate_list = super().configure_evaluate(server_round, parameters, client_manager)
        client_evaluate_list_dynamic = []

        if server_round in self.rounds_to_change_pattern:

            # parameters = self.server_fit(fl.common.parameters_to_ndarrays(parameters), server_round)
            pass


        for client_tuple in client_evaluate_list:
            client = client_tuple[0]
            client_id = str(client.cid)
            config = copy.copy(self.evaluate_config)

            pattern = client_id
            config['pattern'] = pattern
            config['last_layer'] = np.array([])
            self.evaluate_config = config
            evaluate_ins = EvaluateIns(parameters, config)
            # print("Evaluate enviar: ", client_id, [i.shape for i in parameters_to_ndarrays(parameters_to_send)])
            # print("enviar referencia: ", len(parameters), len(parameters_to_ndarrays(parameters_to_send)))
            client_evaluate_list_dynamic.append((client, evaluate_ins))
        return fedpredict_server(parameters=parameters, client_evaluate_list=client_evaluate_list_dynamic,
                                 fedpredict_clients_metrics=self.fedpredict_clients_metrics, df=self.df,
                                 evaluate_config=self.evaluate_config, server_round=server_round,
                                 num_rounds=self.num_rounds, comment=self.comment,
                                 compression=self.compression, layers_compression_range=self.layers_compression_range)

    def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):

        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(server_round, results, failures)

        self.report += """\n ///// Teste /////"""
        acc_list = []
        for response in results:
            client_id = response[1].metrics['cid']
            client_accuracy = float(response[1].metrics['accuracy'])
            pattern = int(response[1].metrics['pattern'])
            acc_list.append(client_accuracy)

            self.report += """\n cliente {} padrao {} acurácia {}""".format(client_id, pattern, client_accuracy)

        print("""acurácia média de teste: {}""".format(sum(acc_list)/len(acc_list)))



        return loss_aggregated, metrics_aggregated

    def end_evaluate_function(self):
        self._write_similarity()
        #self._write_norm()

    # def _write_norm(self):
    #
    # 	columns = ["Server round", "Norm", "nt"]
    # 	data = {column: [] for column in columns}
    #
    # 	data = {'Round': self.gradient_norm_round, 'Norm': self.gradient_norm, 'nt': self.gradient_norm_nt}
    #
    # 	self.similarity_filename = f"{self.base}/norm.csv"
    # 	df = pd.DataFrame(data)
    # 	df.to_csv(self.similarity_filename, index=False)

    def pattern_prportion(self):

        patterns = []
        proportion_list = []
        classes = []

        for pattern in range(len(self.last_layer_parameters_per_class)):

            tuple = self.last_layer_parameters_per_class[pattern]

            proportion = tuple[2]

            if len(proportion_list) > 0:
                # print("a: ", proportion_list, " b: ", proportion)
                proportion_list = np.concatenate((proportion_list.flatten(), proportion.flatten()))
            else:
                proportion_list = np.array(proportion)
            patterns += [pattern] * len(proportion)
            classes += [i for i in range(len(proportion))]

        df = pd.DataFrame({'pattern': patterns, 'classes': classes, 'proportion': proportion_list.tolist()})
        base_dir = "analysis/output/experiment_2/dynamic/"
        line_plot(
            df=df, base_dir=base_dir, file_name="proportion", x_column='classes', y_column='proportion', hue='pattern', title='')


    def _write_similarity(self):

        columns = ["Server round", "Layer", "Similarity"]
        data = {column: [] for column in columns}

        for round in self.similarity_between_layers_per_round:

            for layer in self.similarity_between_layers_per_round[round]:

                data['Server round'].append(round)
                data['Layer'].append(layer)
                data['Similarity'].append(self.similarity_between_layers_per_round[round][layer]['mean'])

        self.similarity_filename = f"{self.base}/similarity_between_layers.csv"
        df = pd.DataFrame(data)
        df.to_csv(self.similarity_filename, index=False)

    def mean_parameters(self, global_model_parameters, pattern_model_parameters):

        count = 0
        final_parameters = []


        print("tip: ", type(global_model_parameters[0][0][0]), type(pattern_model_parameters[0][0][0]))

        # if type(global_model_parameters) == list:
        #     global_model_parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_model_parameters]
        # if type(pattern_model_parameters[0]) == np.array:
        #     pattern_model_parameters = [Parameter(torch.Tensor(i.tolist())) for i in pattern_model_parameters]
        for new_param, old_param in zip(global_model_parameters, pattern_model_parameters):
            if count < len(global_model_parameters) - 2:
                global_model_weight = 1
                pattern_model_weights = 0
            else:
                global_model_weight = 1
                pattern_model_weights = 0
            if new_param.shape == old_param.shape:
                final_parameters.append(
                        global_model_weight * new_param +
                        pattern_model_weights * old_param)
            else:
                raise print("Modelos com tamanhos diferentes")
            count += 1

        return final_parameters

    def _gradient_metric(self, updated_global_parameters, server_round):

        norm = []

        layer = updated_global_parameters[-2]
        norm = np.linalg.norm(layer)

        # self.gradient_norm = float(norm)
        print("norma: ", float(norm))

    def _get_server_header(self):

        server_header = super()._get_server_header()
        return server_header + ["Norm"]

    def _get_server_data(self, process_time, server_round, accuracy_aggregated, accuracy_std, top5, top1):

        return [process_time, server_round, accuracy_aggregated, accuracy_std, top5, top1, self.gradient_norm]

    def server_fit(self, parameters, config):
        pass


