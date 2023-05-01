import flwr as fl
import numpy as np
import math
import os
import time
import csv
import random
import sys
import pandas as pd
import copy

from server.common_base_server import FedAvgBaseServer

from pathlib import Path
import shutil

class FedPredictBaseServer(FedAvgBaseServer):

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
				 strategy_name='FedPredict',
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

		self.server_learning_rate = server_learning_rate
		self.server_momentum = server_momentum
		self.momentum_vector = None
		self.model = model
		self.window_of_previous_accs = 4
		self.server_opt = (self.server_momentum != 0.0) or (
				self.server_learning_rate != 1.0)

		self.set_initial_parameters()
		self.create_folder(strategy_name)

	def create_folder(self, strategy_name):

		directory = """{}_saved_weights/{}/""".format(strategy_name.lower(), self.model_name)
		if Path(directory).exists():
			shutil.rmtree(directory)
		for i in range(self.num_clients):
			Path("""{}_saved_weights/{}/{}/""".format(strategy_name.lower(), self.model_name, i)).mkdir(
				parents=True, exist_ok=True)
			pd.DataFrame({'round_of_last_fit': [-1], 'round_of_last_evaluate': [-1], 'acc_of_last_fit': [0], 'first_round': [-1], 'acc_of_last_evaluate': [0]}).to_csv("""{}_saved_weights/{}/{}/{}.csv""".format(strategy_name.lower(), self.model_name, i, i), index=False)

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
			print("calcula evolution level")
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
			print("Metricas dos clientes: ", self.clients_metrics)
			# self.fedpredict_metrics['nt'] = self.clients_metrics['nt']
			self.fedpredict_metrics['el'] = self._calculate_evolution_level(server_round)
		except Exception as e:
			print("update fedpredict metrics")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def configure_evaluate(self, server_round, parameters, client_manager):
		client_evaluate_list = super().configure_evaluate(server_round, parameters, client_manager)
		client_evaluate_list_fedpredict = []
		accuracy = 0
		if len(self.accuracy_history) > 0:
			accuracy = self.accuracy_history[len(self.accuracy_history)]
		for client_tuple in client_evaluate_list:
			client = client_tuple[0]
			client_id = str(client.cid)
			config = copy.copy(self.evaluate_config)
			client_config = self.fedpredict_clients_metrics[str(client.cid)]
			config['metrics'] = client_config
			config['last_global_accuracy'] = accuracy
			config['total_server_rounds'] = self.num_rounds
			client_similarity_per_layer = self.get_client_similarity_per_layer(client_id, server_round)
			parameters_to_send, M = self._select_layers(client_similarity_per_layer, parameters, server_round, client_id)
			config['M'] = M
			print("enviar")
			evaluate_ins = fl.common.EvaluateIns(parameters_to_send, config)
			client_evaluate_list_fedpredict.append((client, evaluate_ins))

		return client_evaluate_list_fedpredict

	def _select_layers(self, client_similarity_per_layer, parameters, server_round, client_id):

		parameters = fl.common.parameters_to_ndarrays(parameters)
		M = [0, 1, 2, 3]
		# parameters = np.take(parameters, [4, 5])

		print("quantidade de camadas: ", len(parameters), [i.shape for i in parameters])
		print("testando: ", self.layer_selection_evaluate)
		if self.fedpredict_clients_metrics[client_id]['first_round'] != -1 and self.layer_selection_evaluate:
			M = [2, 3]
			parameters = np.take(parameters, M)

		# parameters = parameters[-2:]
		print("quantidade de camadas retornadas: ", len(parameters), [i.shape for i in parameters])
		parameters = fl.common.ndarrays_to_parameters(parameters)

		return parameters, M

	def get_client_similarity_per_layer(self, client_id, server_round):

		round_similarity = self.similarity_between_layers_per_round_and_client[server_round]
		if client_id in round_similarity:
			return round_similarity[client_id]
		else:
			return 0

	def end_evaluate_function(self):
		self._write_similarity()

	def _write_similarity(self):

		columns = ["Server round", "Layer", "Similarity"]
		data = {column: [] for column in columns}
		for round in self.similarity_between_layers_per_round:

			for layer in self.similarity_between_layers_per_round[round]:

				data['Server round'].append(round)
				data['Layer'].append(layer)
				data['Similarity'].append(self.similarity_between_layers_per_round[round][layer])

		self.similarity_filename = f"{self.base}/similarity_between_layers.csv"
		df = pd.DataFrame(data)
		df.to_csv(self.similarity_filename, index=False)