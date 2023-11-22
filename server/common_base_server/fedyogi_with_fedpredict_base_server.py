import flwr as fl
import numpy as np
import sys
import pandas as pd
from client.fedpredict_core import fedpredict_core_layer_selection, fedpredict_layerwise_similarity, fedpredict_core_compredict, dls, layer_compression_range, compredict, fedpredict_server
from server.common_base_server.fedyogi_base_server import FedYogiBaseServer
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

class FedYogiWithFedPredictBaseServer(FedYogiBaseServer):

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
				 decay=0,
				 perc_of_clients=0,
				 dataset='',
				 strategy_name='FedYogi_with_FedPredict',
				 model_name='',
				 new_clients=False,
				 eta: float = 1e-2,
				 eta_l: float = 0.0316,
				 beta_1: float = 0.9,
				 beta_2: float = 0.99,
				 tau: float = 1e-3,
				 new_clients_train=False
				 ):

		super().__init__(aggregation_method=aggregation_method,
							n_classes=n_classes,
							fraction_fit=fraction_fit,
							num_clients=num_clients,
							num_rounds=num_rounds,
						 	args=args,
						 	num_epochs=num_epochs,
						 	model=model,
							decay=decay,
							perc_of_clients=perc_of_clients,
							dataset=dataset,
							strategy_name=strategy_name,
							model_name=model_name,
							new_clients=new_clients,
							eta=eta,
							eta_l=eta_l,
							beta_1=beta_1,
							beta_2=beta_2,
							tau=tau,
						 new_clients_train=new_clients_train,
						 	type=type
							)

		self.n_rate = float(args.n_rate)
		self.model = model
		self.window_of_previous_accs = 4

		self.set_initial_parameters()
		self.create_folder(strategy_name)
		self.similarity_between_layers_per_round = {}
		self.similarity_between_layers_per_round_and_client = {}
		self.model_size = None
		self.similarity_list_per_layer = None
		self.initial_similarity = 0
		self.current_similarity = 0
		self.parameters_aggregated_gradient = {}
		self.parameters_aggregated_checkpoint = {}
		self.layers_compression_range = []
		self.gradient_norm = []
		self.gradient_norm_round = []
		self.gradient_norm_nt = []
		self.T = int(args.T)

	def calculate_initial_similarity(self, server_round, rate=0.1):

		rounds = int(rate * self.T)

		sm = 0
		rounds = min(rounds, server_round)
		for i in range(1, rounds + 1):
			sm += self.similarity_between_layers_per_round[i][0]['mean']

		self.initial_similarity = sm / rounds

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
			pd.DataFrame(
				{'round_of_last_fit': [-1], 'round_of_last_evaluate': [-1], 'acc_of_last_fit': [0], 'first_round': [-1],
				 'acc_of_last_evaluate': [0]}).to_csv(
				"""{}_saved_weights/{}/{}/{}.csv""".format(strategy_name.lower(), self.model_name, i, i), index=False)

	def _calculate_evolution_level(self, server_round):
		try:
			# If the number of rounds so far is low
			if server_round < self.window_of_previous_accs:
				return server_round / self.num_rounds

			acc_list = np.array(list(self.accuracy_history.values()))
			last_acc = acc_list[-1]
			reference_acc = acc_list[-self.window_of_previous_accs]

			# To detect if new clients were added
			if reference_acc > last_acc and server_round == 36:
				# Drop in the performance. New clients where introduced
				t = self._get_round_of_the_most_similar_previous_acc(acc_list, last_acc)
				el = t / self.num_rounds
			else:
				el = (server_round + 1) / self.num_rounds

			return el
		except Exception as e:
			print("calcula evolution level")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def _get_round_of_the_most_similar_previous_acc(self, acc_list, last_acc):

		try:
			# Get the round based on the minimum difference between accuracies.
			# It adds +1 to adjust index that starts from 0
			t = np.argmin(acc_list - last_acc) + 1

			return t
		except Exception as e:
			print("get round of the most similar previous acc")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def _update_fedpredict_metrics(self, server_round):

		try:
			# self.fedpredict_metrics['acc'] = self.accuracy_history
			# It works as a table where for each round it has a line with two columns for the accuracy and
			# the evolution level of the respective round
			self.fedpredict_metrics['round_acc_el'] = {
				int(round): {'acc': self.accuracy_history[round], 'el': round / self.num_rounds} for round in
				self.accuracy_history}
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
		if server_round == 1:
			self.model_shape = [i.shape for i in parameters_to_ndarrays(parameters_aggregated)]
			self.model_size = len(self.model_shape)
			self.similarity_list_per_layer = {i: [] for i in range(self.model_size)}
			self.layers_compression_range = layer_compression_range(self.model_shape)
			print("shape do modelo: ", self.model_shape)
			print("tamanho do modelo: ", self.model_size)
			print("similaridade inicial: ", self.similarity_list_per_layer)
			print("range: ", self.layers_compression_range)
		weights_results = []
		clients_parameters = []
		clients_ids = []
		for _, fit_res in results:
			client_id = str(fit_res.metrics['cid'])
			clients_ids.append(client_id)
			clients_parameters.append(fl.common.parameters_to_ndarrays(fit_res.parameters))

		if self.use_gradient:
			global_parameter = [current - previous for current, previous in
								zip(parameters_to_ndarrays(parameters_aggregated),
									self.previous_global_parameters[server_round - 1])]
		else:
			global_parameter = self.previous_global_parameters[server_round]

		np.random.seed()
		flag = bool(int(np.random.binomial(1, 0.2, 1)))
		np.random.seed(0)
		if server_round == 1:
			flag = True
		print("Flag: ", flag)
		if "dls" in self.compression:
			if flag:
				self.similarity_between_layers_per_round_and_client[server_round], \
				self.similarity_between_layers_per_round[server_round], self.mean_similarity_per_round[
					server_round], self.similarity_list_per_layer = fedpredict_layerwise_similarity(global_parameter,
																									clients_parameters,
																									clients_ids,
																									server_round,
																									self.dataset,
																									str(self.alpha),
																									self.similarity_list_per_layer)
				self.df = max(0, abs(np.mean(self.similarity_list_per_layer[0]) - np.mean(
					self.similarity_list_per_layer[self.model_size - 2])))
			else:
				self.similarity_between_layers_per_round_and_client[server_round], \
				self.similarity_between_layers_per_round[
					server_round], self.mean_similarity_per_round[
					server_round], self.similarity_list_per_layer = self.similarity_between_layers_per_round_and_client[
					server_round - 1], self.similarity_between_layers_per_round[
					server_round - 1], self.mean_similarity_per_round[
					server_round - 1], self.similarity_list_per_layer
		else:
			self.similarity_between_layers_per_round[server_round] = []
			self.mean_similarity_per_round[server_round] = 0
			self.similarity_between_layers_per_round_and_client[server_round] = []
			self.df = 1

		print("df médio: ", self.df, " rodada: ", server_round)

		# self.parameters_aggregated_checkpoint[server_round] = parameters_to_ndarrays(parameters_aggregated)

		# if server_round == 3:
		# 	self.calculate_initial_similarity(server_round)

		return parameters_aggregated, metrics_aggregated

	def configure_evaluate(self, server_round, parameters, client_manager):
		print("Similaridade: ", self.similarity_between_layers_per_round[server_round])
		client_evaluate_list = super().configure_evaluate(server_round, parameters, client_manager)
		return fedpredict_server(parameters=parameters, client_evaluate_list=client_evaluate_list,
								 fedpredict_clients_metrics=self.fedpredict_clients_metrics,
								 evaluate_config=self.evaluate_config, server_round=server_round,
								 num_rounds=self.num_rounds, comment=self.comment,
								 compression=self.compression, layers_compression_range=self.layers_compression_range)

	def get_client_similarity_per_layer(self, client_id, server_round):

		round_similarity = self.similarity_between_layers_per_round_and_client[server_round]
		if client_id in round_similarity:
			return round_similarity[client_id]
		else:
			return 0

	def end_evaluate_function(self):
		self._write_similarity()

	# self._write_norm()

	def _write_norm(self):

		columns = ["Server round", "Norm", "nt"]
		data = {column: [] for column in columns}

		data = {'Round': self.gradient_norm_round, 'Norm': self.gradient_norm, 'nt': self.gradient_norm_nt}

		self.similarity_filename = f"{self.base}/norm.csv"
		df = pd.DataFrame(data)
		df.to_csv(self.similarity_filename, index=False)

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

		# self.gradient_norm = float(norm)
		print("norma: ", float(norm))

	def _get_server_header(self):

		server_header = super()._get_server_header()
		return server_header + ["Norm"]

	def _get_server_data(self, process_time, server_round, accuracy_aggregated, accuracy_std, top5, top1):

		return [process_time, server_round, accuracy_aggregated, accuracy_std, top5, top1, self.gradient_norm]

