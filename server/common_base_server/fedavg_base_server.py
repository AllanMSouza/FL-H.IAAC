import flwr as fl
import numpy as np
import math
import os
import sys
import time
import csv
import random

from logging import WARNING
from flwr.common import FitIns
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from client.fedpredict_core import fedpredict_layerwise_similarity

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
				 type,
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

		#FedLTA
		self.decay_factor = decay

		self.new_clients = new_clients
		self.new_clients_train = new_clients_train
		self.num_rounds = num_rounds
		self.accuracy_history = {}
		self.clients_fit_rounds_history = {i: -1 for i in range(self.num_clients)}
		self.regression_window = 5
		self.fedpredict_metrics = {'el': 1/self.num_rounds, 'round_acc_el': {1: {'acc': 0, 'el': 0}}}
		self.fedpredict_clients_metrics = {str(i): {'round_of_last_fit': 0, 'round_of_last_evaluate': 0, 'first_round': -1,
											   'acc_of_last_fit': 0, 'acc_of_last_evaluate': 0, 'nt': 0}
										   for i in range(0, self.num_clients + 1)}

		# FedPredictSelection
		self.server_nt_acc = {round: {nt: [] for nt in range(0, self.num_rounds + 1)} for round in range(self.num_rounds + 1)}

		self.type = type

		#params
		if self.aggregation_method == 'POC':
			self.strategy_name = f"{strategy_name}-{aggregation_method}-{self.perc_of_clients}"

		elif self.aggregation_method == 'FedLTA':
			self.strategy_name = f"{strategy_name}-{aggregation_method}-{self.decay_factor}"

		elif self.aggregation_method == 'None':
			self.strategy_name = f"{strategy_name}-{aggregation_method}-{fraction_fit}"

		self.round_threshold = 0.7
		self.clients_threshold = 0.7
		self.clients2select = 5

		self.server_filename = None
		self.train_filename = None
		self.evaluate_filename = None
		self.clients_metrics = self._clients_metrics()
		self.evaluate_config = {}
		self._write_output_files_headers()

		super().__init__(fraction_fit=fraction_fit, min_available_clients=num_clients, min_fit_clients=num_clients, min_evaluate_clients=num_clients)

		print("""===================================================\nStarting training of {}\n""".format(self.strategy_name))

	def _clients_metrics(self):

		clients_metrics = {}
		for i in range(self.num_clients):
			if i >= int(self.num_clients * self.clients_threshold) and self.new_clients:
				if self.new_clients_train:
					count = 0
				else:
					count = 1
				clients_metrics[str(i)] = {'count': count, 'max_rounds': 1}
			else:
				clients_metrics[str(i)] = {'count': 0, 'max_rounds': self.num_rounds}

		return clients_metrics

	def _get_valid_clients_for_fit(self):

		clients_ids = []
		for i in self.clients_metrics:
			client_ = self.clients_metrics[i]
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

	def _update_fedpredict_metrics(self, server_round):

		pass

	def configure_fit(self, server_round, parameters, client_manager):
		"""Configure the next round of training."""

		self.start_time = time.time()
		random.seed(server_round)

		if self.aggregation_method == 'POC':
			clients2select        = int(float(self.num_clients) * float(self.perc_of_clients))
			self.selected_clients = self.list_of_clients[:clients2select]

		elif self.aggregation_method == 'FedLTA':
			self.selected_clients = self.select_clients_bellow_average()

			if self.decay_factor > 0:
				the_chosen_ones = len(self.selected_clients) * (1 - self.decay_factor) ** int(server_round)
				self.selected_clients = self.selected_clients[: math.ceil(the_chosen_ones)]

		elif self.aggregation_method == 'None':

			print("lista inicial de clientes: ", self.list_of_clients)
			# ====================================================================================
			clients_ids = []
			for i in self.clients_metrics:
				client_ = self.clients_metrics[i]
				# if client_['count'] < client_['max_rounds'] or server_round >= int(
				# 		self.num_rounds * self.round_threshold):
				# 	clients_ids.append(i)
				if client_['count'] < client_['max_rounds']:
					clients_ids.append(i)
					# 0 rounds since last training
					self.fedpredict_clients_metrics[i]['nt'] = 0
				else:
					# Adds 1 more round without training
					self.fedpredict_clients_metrics[i]['nt'] += 1
			# available_clients = self._get_valid_clients_for_fit()
			# ====================================================================================
			available_clients = clients_ids
			print("disponiveis: ", available_clients)
			print("Rodada inicial de novos clientes: ", int(self.num_rounds * self.round_threshold))

			self.clients2select = int(len(available_clients) * self.fraction_fit)
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


		self.clients_last_round = self.selected_clients
		config = {
			"selected_clients" : ' '.join(self.selected_clients),
			"round"            : server_round
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
		selected_clients_id = []
		if len(self.selected_clients) > 0:
			for client in clients:
				if client.cid in self.selected_clients:
					selected_clients_id.append(client)
		else:
			selected_clients_id = clients
			self.selected_clients = [client.cid for client in clients]
		for client in selected_clients_id:
			self.clients_metrics[client.cid]['count'] += 1
			self.clients_fit_rounds_history[client.cid] = server_round
			# For FedPredict
			self.fedpredict_clients_metrics[client.cid]['round_of_last_fit'] = server_round
			if self.fedpredict_clients_metrics[client.cid]['first_round'] == -1:
				self.fedpredict_clients_metrics[client.cid]['first_round'] = server_round
		for client in self.fedpredict_clients_metrics:
			self.fedpredict_clients_metrics[str(client)]['nt'] = server_round - self.fedpredict_clients_metrics[str(client)]['round_of_last_fit']

		if server_round == self.num_rounds:
			print("=======")
			print(self.strategy_name)
			print("Max rounds per client: ", self.clients_metrics)
			print("=======")

		print("selecionar (fit): ", [client.cid for client in selected_clients_id])
		print("Clientes selecionados: ", len(selected_clients_id))
		# Return client/config pairs
		return [(client, fit_ins) for client in selected_clients_id]

	def aggregate_fit(self, server_round, results, failures):
		weights_results = []
		clients_parameters = []

		for _, fit_res in results:
			client_id = str(fit_res.metrics['cid'])
			clients_parameters.append(fl.common.parameters_to_ndarrays(fit_res.parameters))
			if self.aggregation_method not in ['POC', 'FedLTA'] or int(server_round) <= 1:
				weights_results.append((fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))

			else:
				if client_id in self.selected_clients:
					weights_results.append((fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))

		#print(f'LEN AGGREGATED PARAMETERS: {len(weights_results)}')
		parameters_aggregated = fl.common.ndarrays_to_parameters(aggregate(weights_results))
		fedpredict_layerwise_similarity(fl.common.parameters_to_ndarrays(parameters_aggregated), clients_parameters)
		# Aggregate custom metrics if aggregation fn was provided
		metrics_aggregated = {}
		if server_round == 1:
			print("treinados rodada 1: ", self.clients_metrics)

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
				for i in self.clients_metrics:
					client_ = self.clients_metrics[i]
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
				for i in self.clients_metrics:
					clients_ids.append(i)

		else:
			clients_ids = list(self.clients_metrics.keys())
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
			'round' : server_round
		}

		if self.on_evaluate_config_fn is not None:
			# Custom evaluation config function provided
			config = self.on_evaluate_config_fn(server_round)
		self.evaluate_config = config
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
			if self.clients_metrics[client.cid]['count'] == 0:
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
			self.fedpredict_clients_metrics[str(client_id)]['acc_of_last_evaluate'] = client_accuracy
			self.fedpredict_clients_metrics[str(client_id)]['round_of_last_evaluate'] = server_round
			if self.fedpredict_clients_metrics[str(client_id)]['round_of_last_fit'] == server_round:
				self.fedpredict_clients_metrics[str(client_id)]['acc_of_last_fit'] = client_accuracy
			# FedPredictSelection
			self.server_nt_acc[server_round][self.fedpredict_clients_metrics[str(client_id)]['nt']].append(client_accuracy)

		local_list_clients.sort(key=lambda x: x[1])

		self.list_of_clients    = [str(client[0]) for client in local_list_clients]
		self.list_of_accuracies = [float(client[1]) for client in local_list_clients]

		accs.sort()
		self.average_accuracy   = np.mean(accs)
		self._calculate_mean_of_server_nt_acc(server_round)

		# Weigh accuracy of each client by number of examples used
		accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
		examples   = [r.num_examples for _, r in results]
		# # For FedPredict
		# for _, r in results:
		# 	client_id = r['cid']
		# 	acc = r['accuracy']
		# 	self.fedpredict_clients_metrics[str(client_id)]['acc_of_last_evaluate'] = acc
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

		if server_round == self.num_rounds:
			assert self.server_nt_acc_filename is not None
			data_list = []
			for server_round in self.server_nt_acc:
				for nt in self.server_nt_acc[server_round]:
					acc = self.server_nt_acc[server_round][nt]
					if type(acc) == list:
						acc = 0
					data_list.append([server_round, acc, nt])

			self._write_outputs(filename=self.server_nt_acc_filename,
							   data=data_list
							   )

		self.accuracy_history[server_round] = accuracy_aggregated
		self._update_fedpredict_metrics(server_round)

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

	def _calculate_mean_of_server_nt_acc(self, server_round):

		for nt in range(self.num_rounds):

			if len(self.server_nt_acc[server_round][nt]) > 0:
				self.server_nt_acc[server_round][nt] = np.mean(self.server_nt_acc[server_round][nt])
			else:
				self.server_nt_acc[server_round][nt] = 0

	def _get_metrics(self):

		return {}

	def _write_header(self, filename, header):

		os.makedirs(os.path.dirname(filename), exist_ok=True)
		with open(filename, 'a') as server_log_file:
			writer = csv.writer(server_log_file)
			writer.writerow(header)

	def _write_output_files_headers(self):

		self.base = f"logs/{self.type}/{self.strategy_name}/new_clients_{self.new_clients}_train_{self.new_clients_train}/{self.num_clients}/{self.model_name}/{self.dataset}/{self.epochs}_local_epochs/"
		self.server_filename = f"{self.base}server.csv"
		self.train_filename = f"{self.base}train_client.csv"
		self.evaluate_filename = f"{self.base}evaluate_client.csv"
		self.server_nt_acc_filename = f"{self.base}server_nt_acc.csv"

		server_header = ["Time", "Server round", "Accuracy aggregated", "Accuracy std", "Top5", "Top1"]
		train_header = ["Round", "Cid", "Selected", "Total time", "Size of parameters", "Avg loss train", "Avg accuracy train"]
		evaluate_header = ["Round", "Cid", "Size of parameters", "Size of config", "Loss", "Accuracy"]
		server_nt_acc_header = ["Round", "Accuracy (%)", "nt"]

		# Remove previous files
		if os.path.exists(self.server_filename): os.remove(self.server_filename)
		if os.path.exists(self.train_filename): os.remove(self.train_filename)
		if os.path.exists(self.evaluate_filename): os.remove(self.evaluate_filename)
		if os.path.exists(self.server_nt_acc_filename): os.remove(self.server_nt_acc_filename)
		# Create new files
		self._write_header(self.server_filename, server_header)
		self._write_header(self.train_filename, train_header)
		self._write_header(self.evaluate_filename, evaluate_header)
		self._write_header(self.server_nt_acc_filename, server_nt_acc_header)


	def _write_output(self, filename, data):

		with open(filename, 'a') as server_log_file:
			writer = csv.writer(server_log_file)
			writer.writerow(data)

	def _write_outputs(self, filename, data):

		with open(filename, 'a') as server_log_file:
			writer = csv.writer(server_log_file)
			writer.writerows(data)