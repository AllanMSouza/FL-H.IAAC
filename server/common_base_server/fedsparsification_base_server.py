import flwr as fl
import numpy as np
import math
import os
import time
import csv
import copy
import random

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

	def configure_fit(self, server_round, parameters, client_manager):

		client_evaluate_list_sparsification = []
		client_fit_list = super().configure_fit(server_round, parameters, client_manager)
		for client_tuple in client_fit_list:
			client = client_tuple[0]
			fit_ins = client_tuple[1]
			client_config = fit_ins.config
			k = 0.1
			parameters_to_send = sparse_crs_top_k(parameters, k)
			evaluate_ins = EvaluateIns(parameters_to_send, client_config)
			client_evaluate_list_sparsification.append((client, evaluate_ins))

	def aggregate_fit(self, server_round, results, failures):
		weights_results = []
		clients_parameters = []
		clients_ids = []
		print("agregar")
		for _, fit_res in results:
			client_id = str(fit_res.metrics['cid'])
			clients_ids.append(client_id)
			print("Parametros aggregate fit: ", [i.shape for i in fl.common.parameters_to_ndarrays(fit_res.parameters)])
			# print("Fit respons", fit_res.metrics)
			clients_parameters.append(to_dense(fl.common.parameters_to_ndarrays(fit_res.parameters), self.model_shape))
			if self.aggregation_method not in ['POC', 'FL-H.IAAC'] or int(server_round) <= 1:
				weights_results.append((to_dense(fl.common.parameters_to_ndarrays(fit_res.parameters), self.model_shape), fit_res.num_examples))

			else:
				if client_id in self.selected_clients:
					print("parametro recebido cliente: ", client_id, " parametro: ", len(fl.common.parameters_to_ndarrays(fit_res.parameters)))
					weights_results.append((to_dense(fl.common.parameters_to_ndarrays(fit_res.parameters), self.model_shape), fit_res.num_examples))

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
		for client_tuple in client_evaluate_list:
			client = client_tuple[0]
			fitins = client_tuple[1]
			client_id = str(client.cid)
			client_config = copy.copy(fitins.config)
			client_config['total_server_rounds'] = self.num_rounds
			try:
				client_config['total_server_rounds'] = int(self.comment)
			except:
				pass

			k = 0.1
			parameters_to_send = sparse_crs_top_k(parameters, k)
			evaluate_ins = EvaluateIns(parameters_to_send, client_config)
			fit_ins = fl.common.FitIns(parameters_to_send, client_config)
			client_evaluate_list_fedkd.append((client, fit_ins))

		return client_evaluate_list_fedkd





