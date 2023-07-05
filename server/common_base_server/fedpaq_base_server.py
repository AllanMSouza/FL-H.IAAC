import flwr as fl
import numpy as np
import math
import os
import time
import csv
import random

from logging import WARNING
from flwr.common import FitIns
from client.fedpredict_core import fedpredict_layerwise_similarity
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log
import copy

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

from server.common_base_server import FedAvgBaseServer
from utils.quantization.quantization import parameters_quantization_write, inverse_parameter_quantization_reading

from pathlib import Path
import shutil

class FedPAQBaseServer(FedAvgBaseServer):

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
				 model=None,
				 perc_of_clients=0,
				 dataset='',
				 strategy_name='FedPAQ',
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

		self.n_bits = 8
		self.model = model
		self.model_shape = [i.detach().numpy().shape for i in self.model.parameters()]

	# def aggregate_fit(self, server_round, results, failures):
	# 	weights_results = []
	# 	clients_parameters = []
	# 	clients_ids = []
	# 	print("Rodada: ", server_round, len(results))
	# 	for _, fit_res in results:
	# 		client_id = str(fit_res.metrics['cid'])
	# 		clients_ids.append(client_id)
	# 		print("Parametros aggregate fit: ", len(fl.common.parameters_to_ndarrays(fit_res.parameters)))
	# 		print("Fit respons", fit_res.metrics)
	# 		clients_parameters.append(inverse_parameter_quantization_reading(fl.common.parameters_to_ndarrays(fit_res.parameters), self.model_shape))
	# 		if self.aggregation_method not in ['POC', 'FL-H.IAAC'] or int(server_round) <= 1:
	# 			weights_results.append((inverse_parameter_quantization_reading(fl.common.parameters_to_ndarrays(fit_res.parameters), self.model_shape), fit_res.num_examples))
	#
	# 		else:
	# 			if client_id in self.selected_clients:
	# 				weights_results.append((inverse_parameter_quantization_reading(fl.common.parameters_to_ndarrays(fit_res.parameters), self.model_shape), fit_res.num_examples))
	#
	# 	#print(f'LEN AGGREGATED PARAMETERS: {len(weights_results)}')
	# 	parameters_aggregated = fl.common.ndarrays_to_parameters(self._aggregate(weights_results))
	# 	self.similarity_between_layers_per_round_and_client[server_round], self.similarity_between_layers_per_round[server_round], self.mean_similarity_per_round[server_round], self.decimals_per_layer[server_round] = fedpredict_layerwise_similarity(fl.common.parameters_to_ndarrays(parameters_aggregated), clients_parameters, clients_ids, server_round)
	# 	# Aggregate custom metrics if aggregation fn was provided
	# 	metrics_aggregated = {}
	# 	if server_round == 1:
	# 		print("treinados rodada 1: ", self.clients_metrics)
	#
	# 	return parameters_aggregated, metrics_aggregated

	# def configure_evaluate(self, server_round, parameters, client_manager):
	#
	# 	client_evaluate_list = super().configure_evaluate(server_round, parameters, client_manager)
	# 	client_evaluate_list_fedpredict = []
	# 	accuracy = 0
	# 	size_of_parameters = []
	# 	parameters = fl.common.parameters_to_ndarrays(parameters)
	#
	# 	for i in range(1, len(parameters)):
	# 		size_of_parameters.append(parameters[i].nbytes)
	# 	for client_tuple in client_evaluate_list:
	# 		client = client_tuple[0]
	# 		client_id = str(client.cid)
	# 		config = copy.copy(self.evaluate_config)
	# 		config['total_server_rounds'] = self.num_rounds
	# 		try:
	# 			config['total_server_rounds'] = int(self.comment)
	# 		except:
	# 			pass
	#
	# 		parameters_to_send = ndarrays_to_parameters(parameters)
	# 		if server_round >= 1:
	# 			parameters_to_send = ndarrays_to_parameters(parameters_quantization_write(parameters, self.n_bits))
	# 		evaluate_ins = fl.common.EvaluateIns(parameters_to_send, config)
	# 		client_evaluate_list_fedpredict.append((client, evaluate_ins))
	#
	# 	return client_evaluate_list_fedpredict
	#
	#
	#
