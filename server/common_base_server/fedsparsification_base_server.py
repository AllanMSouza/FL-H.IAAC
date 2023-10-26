import flwr as fl
import numpy as np
import pandas as pd
import math
import os
import time
import csv
import copy
import random
from utils.compression_methods.sparsification import sparse_matrix

from logging import WARNING
from flwr.common import FitIns
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log

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

from utils.compression_methods.sparsification import sparse_crs_top_k, to_dense
from server.common_base_server import FedAvgBaseServer

from pathlib import Path
import shutil

class FedSparsificationBaseServer(FedAvgBaseServer):

	def __init__(self,
				 aggregation_method,
				 n_classes,
				 fraction_fit,
				 num_clients,
				 num_rounds,
				 args,
                 num_epochs,
				 type,
				 decay=0,
				 perc_of_clients=0,
				 dataset='',
				 strategy_name='FedSparsification',
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
						 perc_of_clients=perc_of_clients,
						 dataset=dataset,
						 strategy_name=strategy_name,
						 model_name=model_name,
						 new_clients=new_clients,
						 new_clients_train=new_clients_train,
						 type=type)

		self.create_folder(strategy_name)
		self.clients_model_non_zero_indexes = {}

	def create_folder(self, strategy_name):

		directory = """{}_saved_weights/{}/""".format(strategy_name.lower(), self.model_name)
		if Path(directory).exists():
			shutil.rmtree(directory)
		for i in range(self.num_clients):
			Path("""{}_saved_weights/{}/{}/""".format(strategy_name.lower(), self.model_name, i)).mkdir(
				parents=True, exist_ok=True)
			pd.DataFrame({'round_of_last_fit': [-1], 'round_of_last_evaluate': [-1], 'acc_of_last_fit': [0], 'first_round': [-1], 'acc_of_last_evaluate': [0]}).to_csv("""{}_saved_weights/{}/{}/{}.csv""".format(strategy_name.lower(), self.model_name, i, i), index=False)

	def configure_fit(self, server_round, parameters, client_manager):

		client_evaluate_list_sparsification = []
		client_fit_list = super().configure_fit(server_round, parameters, client_manager)
		parameters = parameters_to_ndarrays(parameters)
		print("shape original: ", [i.shape for i in parameters])
		# k = 0.3
		# parameters_to_send, k_values = sparse_crs_top_k(parameters, k)
		parameters_to_send = parameters
		print("tip pa: ", [type(i) for i in parameters_to_send])
		print([i.shape for i in parameters_to_send])
		for client_tuple in client_fit_list:
			client = client_tuple[0]
			fit_ins = client_tuple[1]
			client_config = fit_ins.config

			# client_config['parameters'] = parameters_to_send

			evaluate_ins = EvaluateIns(ndarrays_to_parameters(parameters_to_send), client_config)
			client_evaluate_list_sparsification.append((client, evaluate_ins))

		return client_evaluate_list_sparsification

	def aggregate_fit(self, server_round, results, failures):
		weights_results = []
		clients_parameters = []
		clients_ids = []
		print("agregar")
		for _, fit_res in results:
			client_id = str(fit_res.metrics['cid'])
			clients_ids.append(client_id)
			parameters = fl.common.parameters_to_ndarrays(fit_res.parameters)
			self.client_model_non_zero_indexes(client_id, parameters)
			print("Parametros aggregate fit: ", [i.shape for i in parameters])
			# print("Fit respons", fit_res.metrics)
			clients_parameters.append(parameters)
			if self.aggregation_method not in ['POC', 'FL-H.IAAC'] or int(server_round) <= 1:
				weights_results.append((parameters, fit_res.num_examples))

			else:
				if client_id in self.selected_clients:
					print("parametro recebido cliente: ", client_id, " parametro: ", len(fl.common.parameters_to_ndarrays(fit_res.parameters)))
					weights_results.append((parameters, fit_res.num_examples))

		#print(f'LEN AGGREGATED PARAMETERS: {len(weights_results)}')
		parameters_aggregated = fl.common.ndarrays_to_parameters(self._aggregate(weights_results, server_round))
		# self.similarity_between_layers_per_round_and_client[server_round], self.similarity_between_layers_per_round[server_round], self.mean_similarity_per_round[server_round], self.decimals_per_layer[server_round] = fedpredict_layerwise_similarity(fl.common.parameters_to_ndarrays(parameters_aggregated), clients_parameters, clients_ids, server_round)
		# Aggregate custom metrics if aggregation fn was provided
		if 'FedKD_with_FedPredict' in self.strategy_name:
			metrics_aggregated = {'clients_id': clients_ids, 'parameters': clients_parameters}
		else:
			metrics_aggregated = {}

		return parameters_aggregated, metrics_aggregated

	def configure_evaluate(self, server_round, parameters, client_manager):
		client_evaluate_list = super().configure_evaluate(server_round, parameters, client_manager)
		client_evaluate_list_fedkd = []
		accuracy = 0
		svd_type = 'svd'
		size_of_parameters = []
		parameters = fl.common.parameters_to_ndarrays(parameters)
		for i in range(1, len(parameters)):
			size_of_parameters.append(parameters[i].nbytes)
		# k = 0.3
		# parameters_to_send, k_values = sparse_crs_top_k(parameters, k)
		# parameters_to_send = parameters
		for i in range(len(client_evaluate_list)):
			client_tuple = client_evaluate_list[i]
			client = client_tuple[0]
			fitins = client_tuple[1]
			client_id = str(client.cid)
			client_evaluate_list[i][1].parameters = ndarrays_to_parameters(self.client_specific_top_k_parameters(client_id, parameters))
			# client_config = copy.copy(fitins.config)
			# # print("client config: ", client_config)
			# client_config['total_server_rounds'] = self.num_rounds
			# try:
			# 	client_config['total_server_rounds'] = int(self.comment)
			# except:
			# 	pass
			#
			#
			# print("server sparse evaluate")
			#
			# # client_config['parameters'] = parameters_to_send
			# evaluate_ins = EvaluateIns(ndarrays_to_parameters(parameters_to_send), client_config)
			# client_evaluate_list_fedkd.append((client, evaluate_ins))

		return client_evaluate_list


	def client_model_non_zero_indexes(self, client_id, parameters):

		non_zero_indexes = []

		for p in parameters:

			zero = p == 0
			non_zero_indexes.append(zero)

		self.clients_model_non_zero_indexes[client_id] = non_zero_indexes

	def client_specific_top_k_parameters(self, client_id, parameters):

		if client_id in self.clients_model_non_zero_indexes:
			indexes_list = self.clients_model_non_zero_indexes[client_id]

			for i in range(len(parameters)):

				parameter = parameters[i]
				indexes = indexes_list[i]

				zeros = np.zeros(parameter.shape, dtype=np.double)

				if zeros.ndim == 1:
					# for j in range(len(indexes[0])):
					# 	zeros[indexes[0][j]] = parameter[indexes[0][j]]
					zeros = parameter

				elif zeros.ndim == 2:
					for j in range(len(indexes)):
						for k in range(len(indexes[j])):
						# print("valor: ", parameter[indexes[0][j], indexes[1][j]])
							if indexes[j, k]:
								parameter[j, k] = 0


				elif zeros.ndim == 3:
					for j in range(len(indexes)):
						for k in range(len(indexes[j])):
							for l in range(len(indexes[j, k])):
								if indexes[j, k, l]:
									parameter[j, k, l] = 0

				elif zeros.ndim == 4:
					for j in range(len(indexes)):
						for k in range(len(indexes[j])):
							for l in range(len(indexes[j, k])):
								for m in range(len(indexes[j, k, l])):
									if indexes[j, k, l, m]:
										parameter[j, k, l, m] = 0



				parameters[i] = parameter

		return parameters





