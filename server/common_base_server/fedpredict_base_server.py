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
from client.fedpredict_core import fedpredict_core_layer_selection, fedpredict_layerwise_similarity

from pathlib import Path
import shutil

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

from utils.quantization.parameters_svd import parameter_svd_write, inverse_parameter_svd_reading

def get_size(parameter):
	try:
		# #print("recebeu: ", parameter.shape, parameter.ndim)
		# if type(parameter) == np.float32:
		# 	#print("caso 1: ", map(sys.getsizeof, parameter))
		# 	return map(sys.getsizeof, parameter)
		# if parameter.ndim <= 2:
		# 	#print("Caso 2: ", sum(map(sys.getsizeof, parameter)))
		# 	return sum(map(sys.getsizeof, parameter))
		#
		# else:
		# 	tamanho = 0
		# 	#print("Caso 3")
		# 	for i in range(len(parameter)):
		# 		tamanho += get_size(parameter[i])
		#
		# 	return tamanho
		return parameter.nbytes
	except Exception as e:
		print("get_size")
		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
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

		self.n_rate = float(args.n_rate)
		self.server_learning_rate = server_learning_rate
		self.server_momentum = server_momentum
		self.momentum_vector = None
		self.model = model
		self.window_of_previous_accs = 4
		self.server_opt = (self.server_momentum != 0.0) or (
				self.server_learning_rate != 1.0)

		self.set_initial_parameters()
		self.create_folder(strategy_name)
		self.similarity_between_layers_per_round = {}
		self.similarity_between_layers_per_round_and_client = {}
		self.initial_similarity = 0
		self.current_similarity = 0
		self.parameters_aggregated_gradient = {}
		self.parameters_aggregated_checkpoint = {}
		self.gradient_norm = None
		self.T = float(args.T)

	def calculate_initial_similarity(self, server_round, rate=0.1):

		rounds = int(rate*self.T)

		sm = 0
		rounds = min(rounds, server_round)
		for i in range(1, rounds +1):

			sm += self.similarity_between_layers_per_round[i][0]['mean']

		self.initial_similarity = sm/rounds

	def calculate_current_similarity(self, server_round, rate=0.05):

		rounds = max(1, int(rate * self.T))
		rounds = min(server_round, rounds)

		sm = 0
		for i in range(server_round - rounds, server_round + 1):
			sm += self.similarity_between_layers_per_round[i][0]['mean']

		self.current_similarity = sm / rounds

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
			# print("Metricas dos clientes: ", self.clients_metrics)
			# self.fedpredict_metrics['nt'] = self.clients_metrics['nt']
			self.fedpredict_metrics['el'] = self._calculate_evolution_level(server_round)
		except Exception as e:
			print("update fedpredict metrics")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	# def configure_fit(self, server_round, parameters, client_manager):
	# 	if server_round == 1:
	# 		self.initial_parameters = parameters_to_ndarrays(parameters)
	# 	results = super().configure_fit(server_round, parameters, client_manager)
	# 	# Return client/config pairs
	# 	return results

	def aggregate_fit(self, server_round, results, failures):

		parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
		weights_results = []
		clients_parameters = []
		clients_ids = []
		for _, fit_res in results:
			client_id = str(fit_res.metrics['cid'])
			clients_ids.append(client_id)
			clients_parameters.append(fl.common.parameters_to_ndarrays(fit_res.parameters))

		self.similarity_between_layers_per_round_and_client[server_round], self.similarity_between_layers_per_round[server_round], self.mean_similarity_per_round[server_round] = fedpredict_layerwise_similarity(self.previous_global_parameters[server_round-1], clients_parameters, clients_ids, server_round, self.dataset, str(self.alpha))

		self.parameters_aggregated_checkpoint[server_round] = parameters_to_ndarrays(parameters_aggregated)

		# if server_round == 3:
		# 	self.calculate_initial_similarity(server_round)

		return parameters_aggregated, metrics_aggregated

	def configure_evaluate(self, server_round, parameters, client_manager):
		print("Similaridade: ", self.similarity_between_layers_per_round[server_round])
		client_evaluate_list = super().configure_evaluate(server_round, parameters, client_manager)
		client_evaluate_list_fedpredict = []
		accuracy = 0
		mean_similarity_per_layer = self.similarity_between_layers_per_round[server_round]
		mean_similarity = self.mean_similarity_per_round[server_round]
		# if len(self.accuracy_history) > 0:
		# 	accuracy = self.accuracy_history[len(self.accuracy_history)]
		size_of_parameters = []
		parameters = fl.common.parameters_to_ndarrays(parameters)
		# if server_round >= 4:
		# 	self.calculate_current_similarity(server_round)
		for i in range(1, len(parameters)):
			size_of_parameters.append(get_size(parameters[i]))
		for client_tuple in client_evaluate_list:
			client = client_tuple[0]
			client_id = str(client.cid)
			config = copy.copy(self.evaluate_config)
			client_config = self.fedpredict_clients_metrics[str(client.cid)]
			nt = client_config['nt']
			config['metrics'] = client_config
			config['last_global_accuracy'] = accuracy
			config['total_server_rounds'] = self.num_rounds
			try:
				config['total_server_rounds'] = int(self.comment)
			except:
				pass

			# print("olha: ", self.similarity_between_layers_per_round[server_round])
			client_similarity_per_layer = self.get_client_similarity_per_layer(client_id, server_round)
			parameters_to_send, M = self._select_layers(client_similarity_per_layer, mean_similarity_per_layer, mean_similarity, parameters, server_round, nt, size_of_parameters, client_id, self.comment)
			# M = [i for i in range(len(parameters))]
			# parameters_to_send = ndarrays_to_parameters(parameter_svd_write(parameters_to_ndarrays(parameters_to_send), self.n_rate))

			self.fedpredict_clients_metrics[str(client.cid)]['acc_bytes_rate'] = size_of_parameters
			config['M'] = M
			evaluate_ins = fl.common.EvaluateIns(parameters_to_send, config)
			client_evaluate_list_fedpredict.append((client, evaluate_ins))

		return client_evaluate_list_fedpredict

	def _select_layers(self, client_similarity_per_layer, mean_similarity_per_layer, mean_similarity, parameters, server_round, nt, size_of_layers, client_id, comment):

		try:
			M = [i for i in range(len(parameters))]
			n_layers = len(parameters)/2

			size_list = []
			for i in range(len(parameters)):
				tamanho = get_size(parameters[i])
				# print("inicio camada: ", i, " tamanho: ", tamanho, " shape: ", parameters[i].shape)
				size_list.append(tamanho)
			print("Tamanho total parametros original: ", sum(size_list), sys.getsizeof(fl.common.ndarrays_to_parameters(parameters)))

			print("quantidade de camadas: ", len(parameters), [i.shape for i in parameters], " comment: ", comment)
			print("layer selection evaluate: ", self.layer_selection_evaluate, self.comment)
			if self.fedpredict_clients_metrics[client_id]['first_round'] != -1:
				# baixo-cima
				if comment == "":
					M = M[-self.layer_selection_evaluate*2:]
				elif comment == "inverted":
					M = M[:self.layer_selection_evaluate * 2]
				elif comment == "individual":
					M = [M[self.layer_selection_evaluate-1], M[self.layer_selection_evaluate]]
				elif comment == 'set':
					if self.layer_selection_evaluate > 0:
						layer = str(self.layer_selection_evaluate)
						M = []
						for i in layer:
							i = int(i)*2
							M.append(int(i) - 2)
							M.append(int(i) - 1)
						print("foi: ", M)
						if layer == '10':
							M = [i for i in range(len(parameters))]
						elif layer == '50':
							M = [i for i in range(len(parameters)//2)]
					else:
						# self.similarity_between_layers_per_round[1][len(parameters)-2]['mean']
						M = fedpredict_core_layer_selection(t=server_round, T=self.num_rounds, nt=nt, n_layers=n_layers, size_per_layer=size_of_layers, mean_similarity_per_layer=mean_similarity_per_layer, current_similarity=self.similarity_between_layers_per_round[server_round][len(parameters)-1]['mean'], first_similarity=self.similarity_between_layers_per_round[server_round][len(parameters)-2]['mean'])
						print("quantidade compartilhadas: ", M)
				new_parameters = []
				for i in range(len(parameters)):
					if i in M:
						# decimals = self.decimals_per_layer[server_round][i]
						# print("decimais: ", decimals)
						# print("parametros originais: ", parameters[i].nbytes, parameters[i].dtype)
						# if decimals <= 4 and self.layer_selection_evaluate < 0:
						# 	data_type = np.half
						# else:
						# 	data_type = np.float32
						# new_parameters.append(parameters[i].astype(data_type))
						new_parameters.append(parameters[i])
						print("parametros reduzidos: ", parameters[i].nbytes)
				parameters = new_parameters

			# parameters = parameters[-2:]
			print("quantidade de camadas retornadas: ", len(parameters), " nt: ", nt)
			size_list = []
			for i in range(len(parameters)):
				# tamanho = get_size(parameters[i])
				tamanho = parameters[i].nbytes
				print("final camada: ", i, " tamanho: ", tamanho, " shape: ", parameters[i].shape)
				size_list.append(tamanho)

			parameters = fl.common.ndarrays_to_parameters(parameters)
			print("Tamanho total parametros depois: ", sum(size_list), sys.getsizeof(parameters))

			return parameters, M

		except Exception as e:
			print("_select_layers")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

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
				data['Similarity'].append(self.similarity_between_layers_per_round[round][layer]['mean'])

		self.similarity_filename = f"{self.base}/similarity_between_layers.csv"
		df = pd.DataFrame(data)
		df.to_csv(self.similarity_filename, index=False)

	def _gradient_metric(self, updated_global_parameters, server_round):

		norm = []

		layer = updated_global_parameters[-2]
		norm = np.linalg.norm(layer)

		self.gradient_norm = float(norm)
		print("norma: ", self.gradient_norm)

	def _get_server_header(self):

		server_header = super()._get_server_header()
		return server_header + ["Norm"]

	def _get_server_data(self, process_time, server_round, accuracy_aggregated, accuracy_std, top5, top1):

		return [process_time, server_round, accuracy_aggregated, accuracy_std, top5, top1, self.gradient_norm]

	# def _subtract(self, server_round, parameters_aggregated):
	#
	# 	try:
	# 		previous_server_round = server_round - 1
	# 		previous_parameter_gradient = self.parameters_aggregated_gradient[previous_server_round]
	# 		parameter_gradient = copy.deepcopy(previous_parameter_gradient)
	#
	# 		for i in range(len(parameters_aggregated)):
	#
	# 			layer_previous_round = parameter_gradient[i]
	# 			layer_aggregated = parameters_aggregated[i]
	# 			parameter_gradient[i] = self._sub(layer_previous_round, layer_aggregated)
	#
	# 			elif np.ndim(la)
	#
	# 	except Exception as e:
	# 		print("_subtract")
	# 		print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__,
	# 			  e)
	#
	# def _sub(self, aggregated_layer_previous_round, aggregated_layer_current_round):
	#
	# 	try:
	# 		if np.ndim(aggregated_layer_previous_round) <= 2:
	#
	# 			gradient = aggregated_layer_previous_round - aggregated_layer_current_round
	# 			return gradient
	#
	# 		elif np.ndim(aggregated_layer_previous_round) >= 3:
	# 			u = []
	# 			for i in range(len(aggregated_layer_previous_round)):
	# 				r = self._sub(aggregated_layer_previous_round[i], aggregated_layer_current_round[i])
	# 				u.append(r)
	# 			return np.array(u)
	#
	# 	except Exception as e:
	# 		print("_sub")
	# 		print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, 0), type(e).__name__,
	# 			  e)

