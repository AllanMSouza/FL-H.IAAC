import flwr as fl
import numpy as np
import math
import os
import time
import csv

from logging import WARNING
from flwr.common import FitIns
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log

class ServerBase(fl.server.strategy.FedAvg):

	def __init__(self, aggregation_method, n_classes, fraction_fit, num_clients,
				 decay=0, perc_of_clients=0, dataset='', strategy_name='', model_name=''):
		
		self.aggregation_method = aggregation_method
		self.n_classes = n_classes
		self.num_clients        = num_clients
		self.list_of_clients    = []
		self.list_of_accuracies = []
		self.selected_clients   = []
		#self.clients_last_round = []

		self.average_accuracy   = 0
		self.last_accuracy      = 0
		self.current_accuracy   = 0

		#logs
		self.dataset    = dataset
		self.model_name = model_name
		self.strategy_name = strategy_name

		#POC
		self.perc_of_clients  = perc_of_clients

		#FedLTA
		self.decay_factor = decay

		#params
		if self.aggregation_method == 'POC':
			self.strategy_name = f"{strategy_name}-{aggregation_method}-{self.perc_of_clients}"

		elif self.aggregation_method == 'FedLTA':
			self.strategy_name = f"{strategy_name}-{aggregation_method}-{self.decay_factor}"

		elif self.aggregation_method == 'None':
			self.strategy_name = f"{strategy_name}-{aggregation_method}"

		self._write_output_files_headers()

		super().__init__(fraction_fit=fraction_fit, min_available_clients=num_clients, min_fit_clients=num_clients, min_evaluate_clients=num_clients)

	def configure_fit(self, server_round, parameters, client_manager):
		"""Configure the next round of training."""
		#print(self.aggregation_method == 'POC')
		if self.aggregation_method == 'POC':
			clients2select        = int(float(self.num_clients) * float(self.perc_of_clients))
			self.selected_clients = self.list_of_clients[:clients2select]

		elif self.aggregation_method == 'FedLTA':
			self.selected_clients = self.select_clients_bellow_average()

			if self.decay_factor > 0:
				the_chosen_ones  = len(self.selected_clients) * (1 - self.decay_factor)**int(server_round)
				self.selected_clients = self.selected_clients[ : math.ceil(the_chosen_ones)]


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

		# Return client/config pairs
		return [(client, fit_ins) for client in clients]


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

		return parameters_aggregated, metrics_aggregated

	def configure_evaluate(self, server_round, parameters, client_manager):
		"""Configure the next round of evaluation."""
		# Do not configure federated evaluation if fraction eval is 0.
		if self.fraction_evaluate == 0.0:
			return []

		# Parameters and config
		config = {
			'round' : server_round
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

		# Return client/config pairs
		return [(client, evaluate_ins) for client in clients]



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

		base = f"logs/{self.strategy_name}/{self.num_clients}/{self.model_name}/{self.dataset}/"
		filename_server = f"{base}server.csv"
		data = [time.time(), server_round, accuracy_aggregated, top5, top1]

		self._write_output(filename=filename_server,
						   data=data
						   )

		metrics_aggregated = { 
			"accuracy"  : accuracy_aggregated,
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

		base = f"logs/{self.strategy_name}/{self.num_clients}/{self.model_name}/{self.dataset}/"
		server_filename = f"{base}server.csv"
		train_filename = f"{base}train_client.csv"
		evaluate_filename= f"{base}evaluate_client.csv"

		server_header = ["Time", "Server round", "Accuracy aggregated", "Top5", "Top1"]
		train_header = ["Round", "Cid", "Selected", "Total time", "Size of parameters", "Avg loss train", "Avg accuracy train"]
		evaluate_header = ["Round", "Cid", "Size of parameters", "Loss", "Accuracy"]

		# Remove previous files
		if os.path.exists(server_filename): os.remove(server_filename)
		if os.path.exists(train_filename): os.remove(train_filename)
		if os.path.exists(evaluate_filename): os.remove(evaluate_filename)
		# Create new files
		self._write_header(server_filename, server_header)
		self._write_header(train_filename, train_header)
		self._write_header(evaluate_filename, evaluate_header)



	def _write_output(self, filename, data):

		with open(filename, 'a') as server_log_file:
			writer = csv.writer(server_log_file)
			writer.writerow(data)
