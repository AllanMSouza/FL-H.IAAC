import flwr as fl
import numpy as np
import math
import statistics as st
import os
import sys
import time
from abc import abstractmethod
import csv
import random

from logging import WARNING
from flwr.common import FitIns
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log
from sklearn.linear_model import LinearRegression

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

class FedAvgBaseServer(fl.server.strategy.FedAvg):

	def __init__(self,
				 aggregation_method,
				 n_classes,
				 fraction_fit,
				 num_clients,
				 num_rounds,
				 num_epochs,
				 decay=0,
				 perc_of_clients=0,
				 dataset='',
				 strategy_name='',
				 non_iid=False,
				 model_name='',
				 new_clients=False,
				 new_clients_train=False):

		self.aggregation_method = aggregation_method
		self.n_classes = n_classes
		self.num_clients        = num_clients
		self.epochs				= num_epochs
		self.list_of_clients    = []
		self.list_of_accuracies = []
		self.selected_clients   = []
		#self.clients_last_round = []

		self.average_accuracy   = 0
		self.last_accuracy      = 0
		self.current_accuracy   = 0

		self.non_iid = non_iid

		#logs
		self.dataset    = dataset
		self.model_name = model_name
		self.strategy_name = strategy_name

		#POC
		self.perc_of_clients  = perc_of_clients
		self.aux_perc_of_clients = perc_of_clients

		#FedLTA
		self.decay_factor = decay

		self.new_clients = new_clients
		self.new_clients_train = new_clients_train
		self.num_rounds = num_rounds
		self.accuracy_history = {}
		self.clients_fit_rounds_history = {i: -1 for i in range(self.num_clients)}
		self.regression_window = 5
		self.fedproposed_metrics = {}

		#params
		if self.aggregation_method == 'POC':
			self.strategy_name = f"{strategy_name}-{aggregation_method}-{self.perc_of_clients}"

		elif self.aggregation_method == 'FedLTA':
			self.strategy_name = f"{strategy_name}-{aggregation_method}-{self.decay_factor}"

		elif self.aggregation_method == 'None':
			self.strategy_name = f"{strategy_name}-{aggregation_method}"

		self.round_threshold = 0.7
		self.clients_threshold = 0.7
		self.clients2select = 5

		self.proto_parameters = None

		self.server_filename = None
		self.train_filename = None
		self.evaluate_filename = None
		self.max_rounds_per_client = self._max_fit_rounds_per_client()
		self._write_output_files_headers()

		super().__init__(fraction_fit=fraction_fit, min_available_clients=num_clients, min_fit_clients=num_clients, min_evaluate_clients=num_clients)

		print("""===================================================\nStarting training of {}\n""".format(self.strategy_name))

	def _max_fit_rounds_per_client(self):

		max_rounds_per_client = {}
		for i in range(self.num_clients):
			if i >= int(self.num_clients * self.clients_threshold) and self.new_clients:
				if self.new_clients_train:
					count = 0
				else:
					count = 1
				max_rounds_per_client[str(i)] = {'count': count, 'max_rounds': 1}
			else:
				max_rounds_per_client[str(i)] = {'count': 0, 'max_rounds': self.num_rounds}

		return max_rounds_per_client

	def _get_valid_clients_for_fit(self):

		clients_ids = []
		for i in self.max_rounds_per_client:
			client_ = self.max_rounds_per_client[i]
			if client_['count'] < client_['max_rounds']:
				clients_ids.append(i)

		return clients_ids

	# def get_valid_clients_for_evaluate(self, server_round):
	#
	# 	try:
	# 		clients_ids = []
	# 		print("normal0: ", self.new_clients)
	# 		exit()
	# 		if self.new_clients:
	# 			# incluir apenas clientes velhos
	# 			if server_round < int(self.num_rounds * self.round_threshold):
	# 				for i in self.max_rounds_per_client:
	# 					client_ = self.max_rounds_per_client[i]
	# 					# clientes velhos podem participar de todas as rodadas
	# 					if client_['max_rounds'] == self.num_rounds:
	# 						clients_ids.append(i)
	# 			else:
	# 				# incluir apenas clientes novos após determinada rodada
	# 				for i in self.max_rounds_per_client:
	# 					client_ = self.max_rounds_per_client[i]
	# 					if client_['max_rounds'] != self.num_rounds:
	# 						clients_ids.append(i)
	#
	# 		else:
	# 			print("normal1")
	# 			clients_ids = list(self.max_rounds_per_client.keys())
	#
	# 		return clients_ids
	# 	except Exception as e:
	# 		print("get valid clients for evaluate")
	# 		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def _update_fedproposed_metrics(self, server_round):

		coef = -10
		std = 0
		if server_round >= self.regression_window:
			accuracy_history = np.array([self.accuracy_history[key] for key in self.accuracy_history])[-self.regression_window:]
			y = np.array(accuracy_history)
			x = np.array([[i] for i in range(0, self.regression_window)])
			reg = LinearRegression().fit(x, y)
			coef = reg.coef_[0]
			std = st.stdev(y)
		self.fedproposed_metrics['acc'] = self.accuracy_history
		self.fedproposed_metrics['coef'] = coef
		# low std means stable global parameters
		# low coef means that global parameters are not very useful
		# self.fedproposed_metrics[server_round]['coef'] = coef
		# self.fedproposed_metrics[server_round]['std'] = std


	def train_and_evaluate_proto_model(self):
		pass

	def configure_fit(self, server_round, parameters, client_manager):
		# """Configure the next round of training."""

		self.start_time = time.time()
		random.seed(server_round)
		if type(parameters) != list:
			weights = parameters_to_ndarrays(parameters)
		else:
			weights = parameters
		self.pre_weights = weights
		if self.aggregation_method == 'POC':
			# clients2select        = int(float(self.num_clients) * float(self.perc_of_clients))

			print("lista inicial de clientes: ", self.list_of_clients)
			# ====================================================================================
			clients_ids = []
			for i in self.max_rounds_per_client:
				client_ = self.max_rounds_per_client[i]
				if client_['count'] < client_['max_rounds']:
					clients_ids.append(i)
			# available_clients = self._get_valid_clients_for_fit()
			# ====================================================================================
			available_clients = clients_ids
			print("disponiveis: ", available_clients)
			print("Rodada inicial de novos clientes: ", int(self.num_rounds * self.round_threshold))
			# if server_round == 1 and 'fedproto' in self.strategy_name.lower():
			# # if server_round == 1:
			# 	self.perc_of_clients = 1
			# else:
			# 	self.perc_of_clients = self.aux_perc_of_clients

			self.clients2select = int(len(available_clients) * self.perc_of_clients)
			if len(available_clients) == 0 and server_round != 1:
				print("Erro na rodada: ", server_round)
				exit()

			# when self.new_clients == True it selects a random subset of only new clients (that have never fitted before)
			# clients2select = int(len(available_clients) * float(self.perc_of_clients))
			print("clientes para selecionar (fit): ", self.clients2select, " de ", len(available_clients))
			if len(available_clients) >= self.clients2select:
				self.selected_clients = random.sample(available_clients, self.clients2select)
			else:
				print("Quantidade inferior")

		elif self.aggregation_method == 'FedLTA':
			self.selected_clients = self.select_clients_bellow_average()

			if self.decay_factor > 0:
				the_chosen_ones = len(self.selected_clients) * (1 - self.decay_factor) ** int(server_round)
				self.selected_clients = self.selected_clients[: math.ceil(the_chosen_ones)]
		else:
			selected_clients = self.selected_clients

		# self.selected_clients = selected_clients

		self.clients_last_round = self.selected_clients
		# if server_round == int(self.num_rounds * self.round_threshold):
		# 	self.train_and_evaluate_proto_model()
		config = {
			"selected_clients" : ' '.join(self.selected_clients),
			"round"            : server_round,
			"parameters"	: self.proto_parameters,
			}

		fit_ins = FitIns(parameters, config)

		# Sample clients
		sample_size, min_num_clients = self.num_fit_clients(
		    client_manager.num_available()
		)
		clients = client_manager.sample(
		    num_clients=sample_size, min_num_clients=min_num_clients
		)

		# fit only selected clients
		selected_clients = []
		if len(self.selected_clients) > 0:
			for client in clients:
				if client.cid in self.selected_clients:
					selected_clients.append(client)
		else:
			selected_clients = clients
			self.selected_clients = [client.cid for client in clients]
		for client in selected_clients:
			self.max_rounds_per_client[client.cid]['count'] += 1
			self.clients_fit_rounds_history[int(client.cid)] = server_round

		if server_round == self.num_rounds:
			print("=======")
			print(self.strategy_name)
			print("Max rounds per client: ", self.max_rounds_per_client)
			print("=======")

		print("selecionar (fit): ", [client.cid for client in selected_clients])
		# Return client/config pairs
		return [(client, fit_ins) for client in selected_clients]

	def aggregate_fit(self, server_round, results, failures):
		weights_results = []

		for _, fit_res in results:
			client_id = str(fit_res.metrics['cid'])

			if self.aggregation_method not in ['POC', 'FedLTA'] or int(server_round) <= 1:
				weights_results.append((fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))

			else:
				if client_id in self.selected_clients:
					weights_results.append((fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))

		#print(f'LEN AGGREGATED PARAMETERS: {len(weights_results)}')
		parameters_aggregated = fl.common.ndarrays_to_parameters(aggregate(weights_results))

		# Aggregate custom metrics if aggregation fn was provided
		metrics_aggregated = {}
		if server_round == 1:
			print("treinados rodada 1: ", self.max_rounds_per_client)

		return parameters_aggregated, metrics_aggregated

	def configure_evaluate(self, server_round, parameters, client_manager):
		"""Configure the next round of evaluation."""
		# Do not configure federated evaluation if fraction eval is 0.
		if self.fraction_evaluate == 0.0:
			return []
		# ====================================================================================
		clients_ids = []
		if self.new_clients:
			# incluir apenas clientes velhos
			if server_round < int(self.num_rounds * self.round_threshold):
				for i in self.max_rounds_per_client:
					client_ = self.max_rounds_per_client[i]
					# clientes velhos podem participar de todas as rodadas
					if client_['max_rounds'] == self.num_rounds:
						clients_ids.append(i)
			else:
				# # incluir apenas clientes novos após determinada rodada
				# for i in self.max_rounds_per_client:
				# 	client_ = self.max_rounds_per_client[i]
				# 	if client_['max_rounds'] != self.num_rounds:
				# 		clients_ids.append(i)
				# incluir todos os clie após determinada rodada
				for i in self.max_rounds_per_client:
					clients_ids.append(i)

		else:
			clients_ids = list(self.max_rounds_per_client.keys())
		# ====================================================================================
		list_of_valid_clients_for_evaluate = clients_ids
		if self.new_clients:
			clients2select = self.clients2select
		else:
			clients2select = int(len(list_of_valid_clients_for_evaluate) * self.perc_of_clients)
		print("clientes para selecionar (evaluate): ", clients2select, " de ", len(list_of_valid_clients_for_evaluate))
		# selected_clients_evaluate = random.sample(list_of_valid_clients_for_evaluate, clients2select)
		selected_clients_evaluate = list_of_valid_clients_for_evaluate
		# Parameters and config
		config = {
			'round' : server_round,
			'parameters': self.proto_parameters,
			'metrics': self.fedproposed_metrics
		}
		if self.on_evaluate_config_fn is not None:
			# Custom evaluation config function provided
			config = self.on_evaluate_config_fn(server_round)
		evaluate_ins = fl.common.EvaluateIns(parameters, config)

		# Sample clients
		sample_size, min_num_clients = self.num_evaluation_clients(
			client_manager.num_available()
		)
		clients = client_manager.sample(
			num_clients=sample_size, min_num_clients=min_num_clients
		)

		# fit only selected clients
		selected_clients = []
		if len(selected_clients_evaluate) > 0:
			for client in clients:
				if client.cid in selected_clients_evaluate:
					selected_clients.append(client)
		else:
			selected_clients = clients
		print("selecionar (evaluate): ", [client.cid for client in selected_clients])
		for client in selected_clients:
			if self.max_rounds_per_client[client.cid]['count'] == 0:
				print("Cliente: ", client.cid, " nunca treinado")
		# Return client/config pairs
		return [(client, evaluate_ins) for client in selected_clients]



	def aggregate_evaluate(
        self,
        server_round,
        results,
        failures,
    ):

		local_list_clients      = []
		self.list_of_clients    = []
		self.list_of_accuracies = []
		accs                    = []


		for response in results:
			client_id       = response[1].metrics['cid']
			client_accuracy = float(response[1].metrics['accuracy'])
			accs.append(client_accuracy)

			local_list_clients.append((client_id, client_accuracy))

		local_list_clients.sort(key=lambda x: x[1])

		self.list_of_clients    = [str(client[0]) for client in local_list_clients]
		self.list_of_accuracies = [float(client[1]) for client in local_list_clients]

		accs.sort()
		self.average_accuracy   = np.mean(accs)

		# Weigh accuracy of each client by number of examples used
		accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
		examples   = [r.num_examples for _, r in results]

		# Aggregate and print custom metric
		accuracy_aggregated = sum(accuracies) / sum(examples)
		accuracy_std = np.std(accuracies)
		current_accuracy    = accuracy_aggregated

		print(f"Round {server_round} accuracy aggregated from client results: {accuracy_aggregated}")

		# Aggregate loss
		loss_aggregated = weighted_loss_avg(
		    [
		        (evaluate_res.num_examples, evaluate_res.loss)
		        for _, evaluate_res in results
		    ]
		)

		# Aggregate custom metrics if aggregation fn was provided
		top5 = np.mean(accs[-5:])
		top1 = accs[-1]

		assert self.server_filename is not None
		data = [time.time()-self.start_time, server_round, accuracy_aggregated, accuracy_std, top5, top1]

		self._write_output(filename=self.server_filename,
						   data=data
						   )

		self.accuracy_history[server_round] = accuracy_aggregated
		self._update_fedproposed_metrics(server_round)

		metrics_aggregated = {
			"accuracy"  : accuracy_aggregated,
			"accuracy std": accuracy_std,
			"top-3"     : top5,
			"top-1"     : top1
		}

		return loss_aggregated, metrics_aggregated


	def select_clients_bellow_average(self):

		selected_clients = []

		for idx_accuracy in range(len(self.list_of_accuracies)):

			if self.list_of_accuracies[idx_accuracy] < self.average_accuracy:
				selected_clients.append(self.list_of_clients[idx_accuracy])

		return selected_clients

	def _write_header(self, filename, header):

		os.makedirs(os.path.dirname(filename), exist_ok=True)
		with open(filename, 'a') as server_log_file:
			writer = csv.writer(server_log_file)
			writer.writerow(header)

	def _write_output_files_headers(self):

		self.base = f"logs/{self.strategy_name}/new_clients_{self.new_clients}_train_{self.new_clients_train}/{self.num_clients}/{self.model_name}/{self.dataset}/{self.epochs}_local_epochs/"
		self.server_filename = f"{self.base}server.csv"
		self.train_filename = f"{self.base}train_client.csv"
		self.evaluate_filename= f"{self.base}evaluate_client.csv"

		server_header = ["Time", "Server round", "Accuracy aggregated", "Accuracy std", "Top5", "Top1"]
		train_header = ["Round", "Cid", "Selected", "Total time", "Size of parameters", "Avg loss train", "Avg accuracy train"]
		evaluate_header = ["Round", "Cid", "Size of parameters", "Loss", "Accuracy"]

		# Remove previous files
		if os.path.exists(self.server_filename): os.remove(self.server_filename)
		if os.path.exists(self.train_filename): os.remove(self.train_filename)
		if os.path.exists(self.evaluate_filename): os.remove(self.evaluate_filename)
		# Create new files
		self._write_header(self.server_filename, server_header)
		self._write_header(self.train_filename, train_header)
		self._write_header(self.evaluate_filename, evaluate_header)



	def _write_output(self, filename, data):

		with open(filename, 'a') as server_log_file:
			writer = csv.writer(server_log_file)
			writer.writerow(data)

	def _protos_similarity(self, protos):

		similarity = np.zeros((self.n_classes, self.n_classes))
