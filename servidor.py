import flwr as fl
import numpy as np
import math

from logging import WARNING
from flwr.common import FitIns
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log

class FedServer(fl.server.strategy.FedAvg):

	def __init__(self, aggregation_method, fraction_fit, num_clients, ):
		
		self.aggregation_method = aggregation_method

		self.list_of_clients    = []
		self.list_of_accuracies = []
		self.selected_clients   = []
		#self.clients_last_round = []

		self.average_accuracy   = 0
		self.last_accuracy      = 0
		self.current_accuracy   = 0

		#POC
		self.clients_to_select  = 3

		#FedLTA
		self.decay_factor = 0.009

		super().__init__(fraction_fit=fraction_fit, min_available_clients=num_clients, min_fit_clients=num_clients, min_evaluate_clients=num_clients)

	def configure_fit(self, server_round, parameters, client_manager):
		"""Configure the next round of training."""
		
		if self.aggregation_method == 'POC':
			self.selected_clients = self.list_of_clients[:self.clients_to_select]

		elif self.aggregation_method == 'FedLTA':
			self.selected_clients = self.select_clients_bellow_average()

			if self.decay_factor > 0:
				the_chosen_ones  = len(self.selected_clients) * (1 - self.decay_factor)**int(server_round)
				self.selected_clients = self.selected_clients[ : math.ceil(the_chosen_ones)]


		self.clients_last_round = self.selected_clients
		print(f'SELECTED:{self.selected_clients}')
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

		print(f'LEN AGGREGATED PARAMETERS: {len(weights_results)}')
		parameters_aggregated = fl.common.ndarrays_to_parameters(aggregate(weights_results))

		# Aggregate custom metrics if aggregation fn was provided
		metrics_aggregated = {}

		return parameters_aggregated, metrics_aggregated



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
		print(accs[-3:])
		# Aggregate custom metrics if aggregation fn was provided
		metrics_aggregated = { 
			"accuracy"  : accuracy_aggregated,
			"top-3_acc" : np.mean(accs[-3:]),
			"best_acc"  : accs[-1]
		}

	
		return loss_aggregated, metrics_aggregated


	def select_clients_bellow_average(self):

		selected_clients = []

		for idx_accuracy in range(len(self.list_of_accuracies)):

			if self.list_of_accuracies[idx_accuracy] < self.average_accuracy:
				selected_clients.append(self.list_of_clients[idx_accuracy])

		return selected_clients
