import flwr as fl
import numpy as np
import math
import os
import time
import csv
import random
import copy
from utils.compression_methods.parameters_svd import parameter_svd_write, inverse_parameter_svd_reading
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

import sys

from server.common_base_server import FedAvgBaseServer

from pathlib import Path
import shutil

def if_reduces_size(shape, n_components, dtype=np.float64):

    try:
        size = np.array([1], dtype=dtype)
        p = shape[0]
        q = shape[1]
        k = n_components

        if p*k + k*k + k*q < p*q:
            return True
        else:
            return False

    except Exception as e:
        print("svd")
        print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__, e)

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

		self.teacher_filename = """{}_saved_weights/{}/""".format(strategy_name.lower(), self.model_name)
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
			n_components_list = []
			if server_round > 1:
				for i in range(len(parameters)):
					compression_range = self.layers_compression_range[i]
					if compression_range > 0:
						frac = 1-server_round/self.num_rounds
						compression_range = max(round(frac * compression_range), 1)
					else:
						compression_range = None
					n_components_list.append(compression_range)

				parameters_to_send = ndarrays_to_parameters(parameter_svd_write(parameters, n_components_list))
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
			self.layers_compression_range = layer_compression_range(self.model_shape)

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
