import flwr as fl
import numpy as np
import math
import os
import time
import csv
import random
import copy
from utils.quantization.parameters_svd import parameter_svd_write, inverse_parameter_svd_reading
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

from server.common_base_server import FedAvgBaseServer

from pathlib import Path
import shutil

class FedKDBaseServer(FedAvgBaseServer):

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
				 model=None,
				 dataset='',
				 strategy_name='FedKD',
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

		self.n_rate = float(args.n_rate)

		self.student_filename = """{}_student_saved_weights/{}/""".format(strategy_name.lower(), self.model_name)
		self.teacher_filename = """{}_saved_weights/{}/""".format(strategy_name.lower(), self.model_name)
		self.create_folder(self.student_filename)
		self.create_folder(self.teacher_filename)
		self.model_shape = [i.detach().numpy().shape for i in model.parameters()]



	def create_folder(self, filename):
		if Path(filename).exists():
			shutil.rmtree(filename)
		for i in range(self.num_clients):
			Path("""{}{}/""".format(filename, i)).mkdir(parents=True, exist_ok=True)

	def configure_fit(self, server_round, parameters, client_manager):
		client_fit_list = super().configure_fit(server_round, parameters, client_manager)
		client_fit_list_fedkd = []
		accuracy = 0
		size_of_parameters = []
		parameters = fl.common.parameters_to_ndarrays(parameters)
		for i in range(1, len(parameters)):
			size_of_parameters.append(parameters[i].nbytes)
		for client_tuple in client_fit_list:
			client = client_tuple[0]
			fitins = client_tuple[1]
			client_id = str(client.cid)
			config = copy.copy(fitins.config)
			config['total_server_rounds'] = self.num_rounds
			try:
				config['total_server_rounds'] = int(self.comment)
			except:
				pass

			parameters_to_send = ndarrays_to_parameters(parameters)
			if server_round >= 1:
				parameters_to_send = ndarrays_to_parameters(parameter_svd_write(parameters, self.n_rate))
			fit_ins = fl.common.FitIns(parameters_to_send, config)
			client_fit_list_fedkd.append((client, fit_ins))

		return client_fit_list_fedkd

	def aggregate_fit(self, server_round, results, failures):
		weights_results = []
		clients_parameters = []
		clients_ids = []
		for _, fit_res in results:
			client_id = str(fit_res.metrics['cid'])
			clients_ids.append(client_id)
			# print("Parametros aggregate fit: ", len(fl.common.parameters_to_ndarrays(fit_res.parameters)))
			# print("Fit respons", fit_res.metrics)
			clients_parameters.append(inverse_parameter_svd_reading(fl.common.parameters_to_ndarrays(fit_res.parameters), self.model_shape))
			if self.aggregation_method not in ['POC', 'FL-H.IAAC'] or int(server_round) <= 1:
				weights_results.append((inverse_parameter_svd_reading(fl.common.parameters_to_ndarrays(fit_res.parameters), self.model_shape), fit_res.num_examples))

			else:
				if client_id in self.selected_clients:
					print("parametro recebido cliente: ", client_id, " parametro: ", len(fl.common.parameters_to_ndarrays(fit_res.parameters)))
					weights_results.append((inverse_parameter_svd_reading(fl.common.parameters_to_ndarrays(fit_res.parameters), self.model_shape), fit_res.num_examples))

		#print(f'LEN AGGREGATED PARAMETERS: {len(weights_results)}')
		parameters_aggregated = fl.common.ndarrays_to_parameters(self._aggregate(weights_results, server_round))
		# self.similarity_between_layers_per_round_and_client[server_round], self.similarity_between_layers_per_round[server_round], self.mean_similarity_per_round[server_round], self.decimals_per_layer[server_round] = fedpredict_layerwise_similarity(fl.common.parameters_to_ndarrays(parameters_aggregated), clients_parameters, clients_ids, server_round)
		# Aggregate custom metrics if aggregation fn was provided
		metrics_aggregated = {}
		if server_round == 1:
			print("treinados rodada 1: ", self.clients_metrics)

		return parameters_aggregated, metrics_aggregated

	# def configure_evaluate(self, server_round, parameters, client_manager):
	# 	client_evaluate_list = super().configure_evaluate(server_round, parameters, client_manager)
	# 	client_evaluate_list_fedkd = []
	# 	accuracy = 0
	# 	size_of_parameters = []
	# 	parameters = fl.common.parameters_to_ndarrays(parameters)
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
	# 			parameters_to_send = ndarrays_to_parameters(parameter_svd_write(parameters, self.n_rate))
	# 		evaluate_ins = fl.common.EvaluateIns(parameters_to_send, config)
	# 		client_evaluate_list_fedkd.append((client, evaluate_ins))
	#
	# 	return client_evaluate_list_fedkd
	#
