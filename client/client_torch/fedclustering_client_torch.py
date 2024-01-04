import flwr as fl
import copy
import numpy as np
import torch
import time
import sys

from dataset_utils_torch import ManageDatasets
from models.torch import DNN, Logistic, CNN, MobileNet, resnet20, CNN_EMNIST, MobileNetV2, CNN_X, CNN_5, CNN_2, CNN_3
import csv
import torch.nn as nn
import warnings
warnings.simplefilter("ignore")

# logging.getLogger("torch").setLevel(logging.ERROR)
from torch.nn.parameter import Parameter
import random
from client.client_torch import ClientBaseTorch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


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
class FedClusteringClientTorch(ClientBaseTorch):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 args,
				 epochs=1,
				 model_name='DNN',
				 client_selection=False,
				 strategy_name='FedClustering',
				 aggregation_method='None',
				 dataset='',
				 perc_of_clients=0,
				 decay=0,
				 fraction_fit=0,
				 non_iid=False,
				 new_clients	= False,
				 new_clients_train  = False
				 ):
		try:
			super().__init__(cid=cid,
                         n_clients=n_clients,
                         n_classes=n_classes,
                         epochs=epochs,
                         model_name=model_name,
                         client_selection=client_selection,
                         solution_name=strategy_name,
                         aggregation_method=aggregation_method,
                         dataset=dataset,
                         perc_of_clients=perc_of_clients,
                         decay=decay,
                         fraction_fit=fraction_fit,
                         non_iid=non_iid,
                         new_clients=new_clients,
                         new_clients_train=new_clients_train,
                         args=args)

			self.client_cluster = list(np.zeros(self.n_clients))
			self.clustering = args.clustering
			self.cluster_round = args.cluster_round
			self.n_clusters = int(args.n_clusters)
			self.dataset = dataset

			self.cluster_method = args.cluster_method
			self.cluster_metric = args.cluster_metric
			self.metric_layer = int(args.metric_layer)

		except Exception as e:
			print("init client")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	# def _create_base_directory(self, type, strategy_name, new_clients, new_clients_train, n_clients, model_name, dataset, class_per_client, alpha, n_rounds, local_epochs, comment, compression, args):
	#
	# 	return f"logs/{type}/{strategy_name}/new_clients_{new_clients}_train_{new_clients_train}/{n_clients}/{model_name}/{dataset}/classes_per_client_{class_per_client}/alpha_{alpha}/{n_rounds}_rounds/{local_epochs}_local_epochs/{comment}_comment/{str(compression)}_compression/{str(args.n_clusters)}_clusters/{str(args.cluster_round)}_cluster_round/{args.cluster_metric}"

	def _create_base_directory(self, type, strategy_name, new_clients, new_clients_train, dynamic_data, n_clients, model_name, dataset, class_per_client, alpha, n_rounds, local_epochs, comment, compression, args):

		return f"logs/{type}/{strategy_name}/new_clients_{new_clients}_train_{new_clients_train}_dynamic_data_{dynamic_data}/{n_clients}/{model_name}/{dataset}/classes_per_client_{class_per_client}/alpha_{alpha}/{n_rounds}_rounds/{local_epochs}_local_epochs/{comment}_comment/{str(compression)}_compression/{str(args.n_clusters)}_clusters/{str(args.cluster_round)}_cluster_round/{args.cluster_metric}_metric"