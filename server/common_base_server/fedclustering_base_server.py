import flwr as fl
import numpy as np
import math
import os
import time
import csv
import random

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

from server.common_base_server.fedavg_base_server import FedAvgBaseServer
from utils.clustering.clustering import calcule_similarity, make_clusters

from pathlib import Path
import shutil

class FedClusteringBaseServer(FedAvgBaseServer):

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
				 dataset='',
				 strategy_name='FedClustering',
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

		self.filename = """{}_saved_weights/{}/""".format(strategy_name.lower(), self.model_name)
		self.create_folder(strategy_name)
		self.client_cluster = list(np.zeros(self.num_clients + 1))
		self.clustering = args.clustering
		self.cluster_round = int(args.cluster_round)
		self.n_clusters = int(args.n_clusters)
		self.dataset = dataset

		self.cluster_method = args.cluster_method
		self.cluster_metric = args.cluster_metric
		self.metric_layer = int(args.metric_layer)
		self.previous_cluster_parameters = {i: [] for i in range(self.n_clusters)}

	def create_folder(self, strategy_name):

		if Path(self.filename).exists():
			shutil.rmtree(self.filename)
		for i in range(self.num_clients):
			Path("""{}{}/""".format(self.filename, i)).mkdir(parents=True, exist_ok=True)

	def configure_fit(
			self, server_round, parameters, client_manager):

		# select clients
		client_fit_ins_list = super().configure_fit(server_round, parameters, client_manager)

		config = client_fit_ins_list[0][1].config

		if server_round == 1:
			fit_ins = FitIns(parameters, config)
			return [(client_fit_ins[0], fit_ins) for client_fit_ins in client_fit_ins_list]

		elif server_round <= self.cluster_round:
			fit_ins = FitIns(parameters['0.0'], config)
			return [(client_fit_ins[0], fit_ins) for client_fit_ins in client_fit_ins_list]

		else:
			return [(client_fit_ins[0], FitIns(parameters[str(self.client_cluster[int(client_fit_ins[0].cid)])], config)) for client_fit_ins in client_fit_ins_list]

	def aggregate_fit(self, server_round, results, failures):

		model = self.create_model()

		"""Aggregate fit results using weighted average."""
		initial_clusters = {'cids': [], 'models': {}}

		weights_results = {}
		lista_last = []
		last_layer_list = []

		for _, fit_res in results:

			client_id = str(fit_res.metrics['cid'])
			parametros_client = fit_res.parameters
			initial_clusters['cids'].append(client_id)
			idx_cluster = self.client_cluster[int(client_id)]

			# save model weights in clusters (or create a new cluster)
			if str(idx_cluster) not in initial_clusters['models'].keys():
				initial_clusters['models'][str(idx_cluster)] = []
			initial_clusters['models'][str(idx_cluster)].append(parameters_to_ndarrays(parametros_client))

			# save model weights and the numer of examples in each client (to avg) in clusters
			if str(idx_cluster) not in weights_results.keys():
				weights_results[str(idx_cluster)] = []
			weights_results[str(idx_cluster)].append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))

			# collect activations and weights of last-layer of clients's model (or any other layer difined in metric_layer)
			w = initial_clusters['models'][str(idx_cluster)][0][-2]  #
			# model.set_weights(w)  #
			# activation_last = self.get_layer_outputs(model, model.layers[self.metric_layer], self.x_servidor, 0)  #
			# lista_last.append(activation_last)  #
			print("shape: ", np.array(w).shape)
			last_layer_list.append(np.array(w).flatten())  #

		# lista_modelos['actv_last'] = lista_last.copy()
		initial_clusters['last_layer'] = last_layer_list

		# similarity between clients (construct the similatity matrix)
		if (server_round == self.cluster_round - 1) or (server_round == self.cluster_round):
			matrix = calcule_similarity(models=initial_clusters, metric=self.cluster_metric, n_clients=self.num_clients)

		# use some clustering method in similarity metrix
		if self.clustering:
			if (server_round == self.cluster_round - 1) or (server_round == self.cluster_round):
				self.client_cluster = make_clusters(matrix=matrix,
													clustering_method=self.cluster_method,
													models=last_layer_list,
													plot_dendrogram=True,
													n_clients=self.num_clients,
													n_clusters=self.n_clusters,
													server_round=server_round,
													cluster_round=self.cluster_round,
													path=f'local_logs/{self.dataset}/{self.cluster_metric}-({self.metric_layer})-{self.cluster_method}-{self.aggregation_method}-{self.fraction_fit}/')
				print("clus: ", self.client_cluster)
				print("matriz: ")
				print(matrix)
				filename = f"local_logs/{self.dataset}/{self.cluster_metric}-({self.metric_layer})-{self.cluster_method}-{self.aggregation_method}-{self.fraction_fit}/clusters_{self.num_clients}clients_{self.n_clusters}clusters.txt"
				os.makedirs(os.path.dirname(filename), exist_ok=True)
				with open(filename, 'a') as arq:
					arq.write(f"{self.client_cluster} - round{server_round}\n")

		# aggregation params for each cluster
		parameters_aggregated = {}
		for idx_cluster in weights_results.keys():
			parameters_aggregated[idx_cluster] = ndarrays_to_parameters(aggregate(weights_results[idx_cluster]))

		metrics_aggregated = {}
		return parameters_aggregated, metrics_aggregated

	def configure_evaluate(
			self, server_round, parameters, client_manager):

		if self.fraction_evaluate == 0.0:
			return []

		config = {'round': server_round, 'n_rounds': self.num_rounds}

		if self.on_evaluate_config_fn is not None:
			config = self.on_evaluate_config_fn(server_round)

		sample_size, min_num_clients = self.num_evaluation_clients(
			client_manager.num_available()
		)
		clients = client_manager.sample(
			num_clients=sample_size, min_num_clients=min_num_clients,
		)
		if server_round == 1:
			result = []
			for client in clients:
				config['nt'] = self.fedpredict_clients_metrics[str(client.cid)]['nt']
				evaluate_ins = EvaluateIns(parameters['0.0'], config)
				result.append((client, evaluate_ins))
			return result

		elif server_round == self.cluster_round - 1:
			result = []
			for client in clients:
				config['nt'] = self.fedpredict_clients_metrics[str(client.cid)]['nt']
				evaluate_ins = EvaluateIns(parameters['0.0'], config)
				result.append((client, evaluate_ins))
			return result
		else:
			result = []
			for client in clients:
				config['nt'] = self.fedpredict_clients_metrics[str(client.cid)]['nt']
				print("chave: ", parameters.keys())
				evaluate_ins = EvaluateIns(parameters[str(self.client_cluster[int(client.cid)])], config)
				result.append((client, evaluate_ins))
			return result

	def _create_base_directory(self, args):
		print("criado")
		print(f"logs/{self.type}/{self.strategy_name}/new_clients_{self.new_clients}_train_{self.new_clients_train}/{self.num_clients}/{self.model_name}/{self.dataset}/classes_per_client_{self.class_per_client}/alpha_{self.alpha}/{self.num_rounds}_rounds/{self.epochs}_local_epochs/{self.comment}_comment/{str(self.compression)}_compression/{str(args.n_clusters)}_clusters/{str(args.cluster_round)}_cluster_round/{args.cluster_metric}")
		return f"logs/{self.type}/{self.strategy_name}/new_clients_{self.new_clients}_train_{self.new_clients_train}/{self.num_clients}/{self.model_name}/{self.dataset}/classes_per_client_{self.class_per_client}/alpha_{self.alpha}/{self.num_rounds}_rounds/{self.epochs}_local_epochs/{self.comment}_comment/{str(self.compression)}_compression/{str(args.n_clusters)}_clusters/{str(args.cluster_round)}_cluster_round/{args.cluster_metric}"


