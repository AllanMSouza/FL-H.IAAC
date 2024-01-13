import flwr as fl
from client import FedAvgClientTf, FedPerClientTf, FedProtoClientTf, FedLocalClientTf, FedAvgClientTorch, FedProtoClientTorch, FedPerClientTorch, FedLocalClientTorch, FedAvgMClientTorch, QFedAvgClientTorch, FedYogiClientTorch, FedClassAvgClientTorch, FedPredictClientTorch, FedPer_with_FedPredictClientTorch, FedClassAvg_with_FedPredictClientTorch, FedProxClientTorch, FedPAQClientTorch, FetSGDClientTorch, FedKDClientTorch, FedDistillClientTorch, FedYogiWithFedPredictClientTorch, FedClusteringClientTorch, FedAlaClientTorch, FedKDWithFedPredictClientTorch, FedSparsificationClientTorch, FedPredictDynamicClientTorch, CDAFedAvgClientTorch, FedCDMClientTorch
from server import FedPerServerTf, FedProtoServerTf, FedAvgServerTf, FedLocalServerTf, FedAvgServerTorch, FedProtoServerTorch, FedPerServerTorch, FedLocalServerTorch, FedAvgMServerTorch, QFedAvgServerTorch, FedYogiServerTorch, FedClassAvgServerTorch, FedPredictServerTorch, FedPer_with_FedPredictServerTorch, FedClassAvg_with_FedPredictServerTorch, FedProxServerTorch, FedPAQServerTorch, FetSGDServerTorch, FedKDServerTorch, FedDistillServerTorch, FedYogiWithFedPredictServerTorch, FedClusteringServerTorch, FedAlaServerTorch, FedKDWithFedPredictServerTorch, FedSparsificationServerTorch, FedPredictDynamicServerTorch, CDAFedAvgServerTorch, FedCDMServerTorch

from optparse import OptionParser
import tensorflow as tf
import torch
import random
import numpy as np
import copy
import ast
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

tf.random.set_seed(0)

class SimulationFL():
	"""docstring for Simulation"""
	def __init__(self,
				 n_clients,
				 aggregation_method,
				 model_name,
				 args,
				 strategy_name,
				 dataset,
				 n_classes,
				 local_epochs,
				 rounds,
				 poc,
				 fraction_fit,
				 decay,
				 non_iid,
				 m_combining_layers,
				 nn_type,
				 new_clients,
				 new_clients_train
				 ):
		
		self.n_clients        		= n_clients
		self.aggregation_method     = aggregation_method
		self.model_name       		= model_name # cnn, dnn, among others
		self.args					= args
		self.dataset          		= dataset
		self.n_classes 				= n_classes
		self.epochs           		= local_epochs
		self.rounds           		= rounds
		self.poc              		= poc
		self.fraction_fit = fraction_fit
		self.decay            		= decay
		self.client_selection 		= False
		self.strategy_name    		= strategy_name # Old "self.solution_name"
		self.non_iid          		= non_iid
		self.m_combining_layers		= m_combining_layers
		self.nn_type = nn_type
		self.new_clients = new_clients
		self.new_clients_train		= new_clients_train

	
	def create_client(self, cid):

		if self.aggregation_method != 'None':
			self.client_selection = True

		if self.nn_type == 'tf':
			if self.strategy_name == 'FedPer':
				return FedPerClientTf(cid=cid,
									  n_clients=self.n_clients,
									  n_classes=self.n_classes,
									  epochs=self.epochs,
									  model_name=self.model_name,
									  client_selection=self.client_selection,
									  solution_name=self.strategy_name,
									  aggregation_method=self.aggregation_method,
									  dataset=self.dataset,
									  perc_of_clients=self.poc,
									  fraction_fit=self.fraction_fit,
									  decay=self.decay,
									  non_iid=self.non_iid,
									  args=self.args)

			elif self.strategy_name == 'FedLocal':
				return FedLocalClientTf(cid=cid,
										n_clients=self.n_clients,
										n_classes=self.n_classes,
										epochs=self.epochs,
										model_name=self.model_name,
										client_selection=self.client_selection,
										solution_name=self.strategy_name,
										aggregation_method=self.aggregation_method,
										dataset=self.dataset,
										perc_of_clients=self.poc,
										fraction_fit=self.fraction_fit,
										decay=self.decay,
										non_iid=self.non_iid,
									  args=self.args)

			elif self.strategy_name == 'FedProto':
				return FedProtoClientTf(cid=cid,
										n_clients=self.n_clients,
										n_classes=self.n_classes,
										epochs=self.epochs,
										model_name=self.model_name,
										client_selection=self.client_selection,
										solution_name=self.strategy_name,
										aggregation_method=self.aggregation_method,
										dataset=self.dataset,
										perc_of_clients=self.poc,
										fraction_fit=self.fraction_fit,
										decay=self.decay,
										non_iid=self.non_iid,
									  args=self.args)

			else:
				return FedAvgClientTf(cid=cid,
									  n_clients=self.n_clients,
									  n_classes=self.n_classes,
									  model_name=self.model_name,
									  client_selection=self.client_selection,
									  epochs=self.epochs,
									  solution_name=self.strategy_name,
									  aggregation_method=self.aggregation_method,
									  dataset=self.dataset,
									  perc_of_clients=self.poc,
									  fraction_fit=self.fraction_fit,
									  decay=self.decay,
									  non_iid=self.non_iid,
									  args=self.args)
		elif self.nn_type == 'torch':
			if self.strategy_name == 'FedProto':
				# print("foi cliente")
				return FedProtoClientTorch(cid=cid,
										   args=self.args,
										   n_clients=self.n_clients,
										   n_classes=self.n_classes,
										   epochs=self.epochs,
										   model_name=self.model_name,
										   client_selection=self.client_selection,
										   strategy_name=self.strategy_name,
										   aggregation_method=self.aggregation_method,
										   dataset=self.dataset,
										   perc_of_clients=self.poc,
										   fraction_fit=self.fraction_fit,
										   decay=self.decay,
										   non_iid=self.non_iid,
										   new_clients=self.new_clients,
										   new_clients_train=self.new_clients_train)
			if self.strategy_name == 'FedDistill':
				# print("foi cliente")
				return FedDistillClientTorch(cid=cid,
										   args=self.args,
										   n_clients=self.n_clients,
										   n_classes=self.n_classes,
										   epochs=self.epochs,
										   model_name=self.model_name,
										   client_selection=self.client_selection,
										   strategy_name=self.strategy_name,
										   aggregation_method=self.aggregation_method,
										   dataset=self.dataset,
										   perc_of_clients=self.poc,
										   fraction_fit=self.fraction_fit,
										   decay=self.decay,
										   non_iid=self.non_iid,
										   new_clients=self.new_clients,
										   new_clients_train=self.new_clients_train)
			if self.strategy_name == 'FedPredict':
				# print("foi cliente")
				return FedPredictClientTorch(cid=cid,
											args=self.args,
                                             n_clients=self.n_clients,
                                             n_classes=self.n_classes,
                                             epochs=self.epochs,
                                             model_name=self.model_name,
                                             client_selection=self.client_selection,
                                             strategy_name=self.strategy_name,
                                             aggregation_method=self.aggregation_method,
                                             dataset=self.dataset,
                                             perc_of_clients=self.poc,
											 fraction_fit=self.fraction_fit,
                                             decay=self.decay,
                                             non_iid=self.non_iid,
											 m_combining_layers=self.m_combining_layers,
                                             new_clients=self.new_clients,
                                             new_clients_train=self.new_clients_train)
			if self.strategy_name == 'FedPredict_Dynamic':
				# print("foi cliente")
				return FedPredictDynamicClientTorch(cid=cid,
											 args=self.args,
											 n_clients=self.n_clients,
											 n_classes=self.n_classes,
											 epochs=self.epochs,
											 model_name=self.model_name,
											 client_selection=self.client_selection,
											 strategy_name=self.strategy_name,
											 aggregation_method=self.aggregation_method,
											 dataset=self.dataset,
											 perc_of_clients=self.poc,
											 fraction_fit=self.fraction_fit,
											 decay=self.decay,
											 non_iid=self.non_iid,
											 m_combining_layers=self.m_combining_layers,
											 new_clients=self.new_clients,
											 new_clients_train=self.new_clients_train)
			if self.strategy_name == 'FedKD_with_FedPredict':
				# print("foi cliente")
				return FedKDWithFedPredictClientTorch(cid=cid,
													  args=self.args,
													  n_clients=self.n_clients,
													  n_classes=self.n_classes,
													  model_name=self.model_name,
													  client_selection=self.client_selection,
													  epochs=self.epochs,
													  strategy_name=self.strategy_name,
													  aggregation_method=self.aggregation_method,
													  dataset=self.dataset,
													  perc_of_clients=self.poc,
													  fraction_fit=self.fraction_fit,
													  decay=self.decay,
													  non_iid=self.non_iid,
													  new_clients=self.new_clients,
													  new_clients_train=self.new_clients_train)
			if self.strategy_name == 'FedPer_with_FedPredict':
				# print("foi cliente")
				return FedPer_with_FedPredictClientTorch(cid=cid,
											 args=self.args,
                                             n_clients=self.n_clients,
                                             n_classes=self.n_classes,
                                             epochs=self.epochs,
                                             model_name=self.model_name,
                                             client_selection=self.client_selection,
                                             strategy_name=self.strategy_name,
                                             aggregation_method=self.aggregation_method,
                                             dataset=self.dataset,
                                             perc_of_clients=self.poc,
											 fraction_fit=self.fraction_fit,
                                             decay=self.decay,
                                             non_iid=self.non_iid,
											 m_combining_layers=self.m_combining_layers,
                                             new_clients=self.new_clients,
                                             new_clients_train=self.new_clients_train)
			if self.strategy_name == 'FedClassAvg_with_FedPredict':
				# print("foi cliente")
				return FedClassAvg_with_FedPredictClientTorch(cid=cid,
											 args=self.args,
                                             n_clients=self.n_clients,
                                             n_classes=self.n_classes,
                                             epochs=self.epochs,
                                             model_name=self.model_name,
                                             client_selection=self.client_selection,
                                             strategy_name=self.strategy_name,
                                             aggregation_method=self.aggregation_method,
                                             dataset=self.dataset,
                                             perc_of_clients=self.poc,
											 fraction_fit=self.fraction_fit,
                                             decay=self.decay,
                                             non_iid=self.non_iid,
											 m_combining_layers=self.m_combining_layers,
                                             new_clients=self.new_clients,
                                             new_clients_train=self.new_clients_train)
			elif self.strategy_name == 'FedPer':
				return  FedPerClientTorch(cid=cid,
										  n_clients=self.n_clients,
										   args=self.args,
										  n_classes=self.n_classes,
										  epochs=self.epochs,
										  model_name=self.model_name,
										  client_selection=self.client_selection,
										  strategy_name=self.strategy_name,
										  aggregation_method=self.aggregation_method,
										  dataset=self.dataset,
										  perc_of_clients=self.poc,
										  fraction_fit=self.fraction_fit,
										  decay=self.decay,
										  non_iid=self.non_iid,
										  new_clients=self.new_clients,
										  new_clients_train=self.new_clients_train)
			elif self.strategy_name == 'FedClassAvg':
				return  FedClassAvgClientTorch(cid=cid,
										   args=self.args,
											   n_clients=self.n_clients,
											   n_classes=self.n_classes,
											   epochs=self.epochs,
											   model_name=self.model_name,
											   client_selection=self.client_selection,
											   strategy_name=self.strategy_name,
											   aggregation_method=self.aggregation_method,
											   dataset=self.dataset,
											   perc_of_clients=self.poc,
											   fraction_fit=self.fraction_fit,
											   decay=self.decay,
											   non_iid=self.non_iid,
											   new_clients=self.new_clients,
											   new_clients_train=self.new_clients_train)
			elif self.strategy_name == 'FedSparsification':
				return  FedSparsificationClientTorch(cid=cid,
										   args=self.args,
											   n_clients=self.n_clients,
											   n_classes=self.n_classes,
											   epochs=self.epochs,
											   model_name=self.model_name,
											   client_selection=self.client_selection,
											   strategy_name=self.strategy_name,
											   aggregation_method=self.aggregation_method,
											   dataset=self.dataset,
											   perc_of_clients=self.poc,
											   fraction_fit=self.fraction_fit,
											   decay=self.decay,
											   non_iid=self.non_iid,
											   new_clients=self.new_clients,
											   new_clients_train=self.new_clients_train)
			elif self.strategy_name == 'FedAla':
				return  FedAlaClientTorch(cid=cid,
										   args=self.args,
											   n_clients=self.n_clients,
											   n_classes=self.n_classes,
											   epochs=self.epochs,
											   model_name=self.model_name,
											   client_selection=self.client_selection,
											   strategy_name=self.strategy_name,
											   aggregation_method=self.aggregation_method,
											   dataset=self.dataset,
											   perc_of_clients=self.poc,
											   fraction_fit=self.fraction_fit,
											   decay=self.decay,
											   non_iid=self.non_iid,
											   new_clients=self.new_clients,
											   new_clients_train=self.new_clients_train)
			elif self.strategy_name == 'FedProx':
				return  FedProxClientTorch(cid=cid,
										   args=self.args,
											   n_clients=self.n_clients,
											   n_classes=self.n_classes,
											   epochs=self.epochs,
											   model_name=self.model_name,
											   client_selection=self.client_selection,
											   strategy_name=self.strategy_name,
											   aggregation_method=self.aggregation_method,
											   dataset=self.dataset,
											   perc_of_clients=self.poc,
											   fraction_fit=self.fraction_fit,
											   decay=self.decay,
											   non_iid=self.non_iid,
											   new_clients=self.new_clients,
											   new_clients_train=self.new_clients_train)
			elif self.strategy_name == 'FedLocal':
				return  FedLocalClientTorch(cid=cid,
										   args=self.args,
											n_clients=self.n_clients,
											n_classes=self.n_classes,
											epochs=self.epochs,
											model_name=self.model_name,
											client_selection=self.client_selection,
											strategy_name=self.strategy_name,
											aggregation_method=self.aggregation_method,
											dataset=self.dataset,
											perc_of_clients=self.poc,
											fraction_fit=self.fraction_fit,
											decay=self.decay,
											non_iid=self.non_iid,
											new_clients=self.new_clients,
											new_clients_train=self.new_clients_train)
			elif self.strategy_name == 'FedAvgM':
				return FedAvgMClientTorch(cid=cid,
										   args=self.args,
										  n_clients=self.n_clients,
										  n_classes=self.n_classes,
										  model_name=self.model_name,
										  client_selection=self.client_selection,
										  epochs=self.epochs,
										  strategy_name=self.strategy_name,
										  aggregation_method=self.aggregation_method,
										  dataset=self.dataset,
										  perc_of_clients=self.poc,
										  fraction_fit=self.fraction_fit,
										  decay=self.decay,
										  non_iid=self.non_iid,
										  new_clients=self.new_clients,
										  new_clients_train=self.new_clients_train)
			elif self.strategy_name == 'QFedAvg':
				return QFedAvgClientTorch(cid=cid,
										   args=self.args,
										  n_clients=self.n_clients,
										  n_classes=self.n_classes,
										  model_name=self.model_name,
										  client_selection=self.client_selection,
										  epochs=self.epochs,
										  strategy_name=self.strategy_name,
										  aggregation_method=self.aggregation_method,
										  dataset=self.dataset,
										  perc_of_clients=self.poc,
										  fraction_fit=self.fraction_fit,
										  decay=self.decay,
										  non_iid=self.non_iid,
										  new_clients=self.new_clients,
										  new_clients_train=self.new_clients_train)
			elif self.strategy_name == "FedYogi":
				return FedYogiClientTorch(cid=cid,
										   args=self.args,
										  n_clients=self.n_clients,
										  n_classes=self.n_classes,
										  model_name=self.model_name,
										  client_selection=self.client_selection,
										  epochs=self.epochs,
										  strategy_name=self.strategy_name,
										  aggregation_method=self.aggregation_method,
										  dataset=self.dataset,
										  perc_of_clients=self.poc,
										  fraction_fit=self.fraction_fit,
										  decay=self.decay,
										  non_iid=self.non_iid,
										  new_clients=self.new_clients,
										  new_clients_train=self.new_clients_train)
			elif self.strategy_name == "CDA-FedAvg":
				return CDAFedAvgClientTorch(cid=cid,
										   args=self.args,
										  n_clients=self.n_clients,
										  n_classes=self.n_classes,
										  model_name=self.model_name,
										  client_selection=self.client_selection,
										  epochs=self.epochs,
										  strategy_name=self.strategy_name,
										  aggregation_method=self.aggregation_method,
										  dataset=self.dataset,
										  perc_of_clients=self.poc,
										  fraction_fit=self.fraction_fit,
										  decay=self.decay,
										  non_iid=self.non_iid,
										  new_clients=self.new_clients,
										  new_clients_train=self.new_clients_train)
			elif self.strategy_name == "FedCDM":
				return FedCDMClientTorch(cid=cid,
										   args=self.args,
										  n_clients=self.n_clients,
										  n_classes=self.n_classes,
										  model_name=self.model_name,
										  client_selection=self.client_selection,
										  epochs=self.epochs,
										  strategy_name=self.strategy_name,
										  aggregation_method=self.aggregation_method,
										  dataset=self.dataset,
										  perc_of_clients=self.poc,
										  fraction_fit=self.fraction_fit,
										  decay=self.decay,
										  non_iid=self.non_iid,
										  new_clients=self.new_clients,
										  new_clients_train=self.new_clients_train)
			elif self.strategy_name == "FedClustering":
				return FedClusteringClientTorch(cid=cid,
										   args=self.args,
										  n_clients=self.n_clients,
										  n_classes=self.n_classes,
										  model_name=self.model_name,
										  client_selection=self.client_selection,
										  epochs=self.epochs,
										  strategy_name=self.strategy_name,
										  aggregation_method=self.aggregation_method,
										  dataset=self.dataset,
										  perc_of_clients=self.poc,
										  fraction_fit=self.fraction_fit,
										  decay=self.decay,
										  non_iid=self.non_iid,
										  new_clients=self.new_clients,
										  new_clients_train=self.new_clients_train)
			elif self.strategy_name == "FedYogi_with_FedPredict":
				return FedYogiWithFedPredictClientTorch(cid=cid,
										   args=self.args,
										  n_clients=self.n_clients,
										  n_classes=self.n_classes,
										  model_name=self.model_name,
										  client_selection=self.client_selection,
										  epochs=self.epochs,
										  strategy_name=self.strategy_name,
										  aggregation_method=self.aggregation_method,
										  dataset=self.dataset,
										  perc_of_clients=self.poc,
										  fraction_fit=self.fraction_fit,
										  decay=self.decay,
										  non_iid=self.non_iid,
										  new_clients=self.new_clients,
										  new_clients_train=self.new_clients_train)
			elif self.strategy_name == "FedPAQ":
				return FedPAQClientTorch(cid=cid,
										 args=self.args,
										 n_clients=self.n_clients,
										 n_classes=self.n_classes,
										 model_name=self.model_name,
										 client_selection=self.client_selection,
										 epochs=self.epochs,
										 strategy_name=self.strategy_name,
										 aggregation_method=self.aggregation_method,
										 dataset=self.dataset,
										 perc_of_clients=self.poc,
										 fraction_fit=self.fraction_fit,
										 decay=self.decay,
										 non_iid=self.non_iid,
										 new_clients=self.new_clients,
										 new_clients_train=self.new_clients_train)
			elif self.strategy_name == "FetchSGD":
				return FetSGDClientTorch(cid=cid,
										 args=self.args,
										 n_clients=self.n_clients,
										 n_classes=self.n_classes,
										 model_name=self.model_name,
										 client_selection=self.client_selection,
										 epochs=self.epochs,
										 strategy_name=self.strategy_name,
										 aggregation_method=self.aggregation_method,
										 dataset=self.dataset,
										 perc_of_clients=self.poc,
										 fraction_fit=self.fraction_fit,
										 decay=self.decay,
										 non_iid=self.non_iid,
										 new_clients=self.new_clients,
										 new_clients_train=self.new_clients_train)
			elif self.strategy_name == "FedKD":
				return FedKDClientTorch(cid=cid,
										args=self.args,
										n_clients=self.n_clients,
										n_classes=self.n_classes,
										model_name=self.model_name,
										client_selection=self.client_selection,
										epochs=self.epochs,
										strategy_name=self.strategy_name,
										aggregation_method=self.aggregation_method,
										dataset=self.dataset,
										perc_of_clients=self.poc,
										fraction_fit=self.fraction_fit,
										decay=self.decay,
										non_iid=self.non_iid,
										new_clients=self.new_clients,
										new_clients_train=self.new_clients_train)
			else:
				return FedAvgClientTorch(cid=cid,
										   args=self.args,
										 n_clients=self.n_clients,
										 n_classes=self.n_classes,
										 model_name=self.model_name,
										 client_selection=self.client_selection,
										 epochs=self.epochs,
										 strategy_name=self.strategy_name,
										 aggregation_method=self.aggregation_method,
										 dataset=self.dataset,
										 perc_of_clients=self.poc,
										 fraction_fit=self.fraction_fit,
										 decay=self.decay,
										 non_iid=self.non_iid,
										 new_clients=self.new_clients,
										 new_clients_train=self.new_clients_train)


	def create_strategy(self):

		if self.nn_type == 'tf':
			if self.strategy_name == 'FedPer':
				return FedPerServerTf(aggregation_method=self.aggregation_method,
										n_classes=self.n_classes,
										fraction_fit=self.fraction_fit,
										num_clients=self.n_clients,
										num_rounds=self.rounds,
									  	args=self.args,
										num_epochs=self.epochs,
										decay=self.decay,
										perc_of_clients=self.poc,
										strategy_name=self.strategy_name,
										dataset=self.dataset,
										model_name=self.model_name,
										new_clients=self.new_clients,
										new_clients_train=self.new_clients_train
									  )

			elif self.strategy_name == 'FedProto':
				return FedProtoServerTf(aggregation_method=self.aggregation_method,
										n_classes=self.n_classes,
										fraction_fit=self.fraction_fit,
										num_clients=self.n_clients,
										num_rounds=self.rounds,
										num_epochs=self.epochs,
										decay=self.decay,
										perc_of_clients=self.poc,
										strategy_name=self.strategy_name,
										dataset=self.dataset,
										model_name=self.model_name,
										new_clients=self.new_clients,
										new_clients_train=self.new_clients_train,
									  args=self.args)
			elif self.strategy_name == 'FedLocal':
				return FedLocalServerTf(aggregation_method=self.aggregation_method,
										n_classes=self.n_classes,
										fraction_fit=self.fraction_fit,
										num_clients=self.n_clients,
										num_rounds=self.rounds,
										num_epochs=self.epochs,
										decay=self.decay,
										perc_of_clients=self.poc,
										strategy_name=self.strategy_name,
										dataset=self.dataset,
										model_name=self.model_name,
										new_clients=self.new_clients,
										new_clients_train=self.new_clients_train,
									  args=self.args)
			else:
				return FedAvgServerTf(aggregation_method=self.aggregation_method,
										n_classes=self.n_classes,
										fraction_fit=self.fraction_fit,
										num_clients=self.n_clients,
										num_rounds=self.rounds,
										num_epochs=self.epochs,
										decay=self.decay,
										perc_of_clients=self.poc,
										strategy_name=self.strategy_name,
										dataset=self.dataset,
										model_name=self.model_name,
										new_clients=self.new_clients,
										new_clients_train=self.new_clients_train,
									  args=self.args)
		elif self.nn_type == 'torch':
			if self.strategy_name == 'FedProto':
				# print("foi servidor")
				return FedProtoServerTorch(aggregation_method=self.aggregation_method,
										n_classes=self.n_classes,
										fraction_fit=self.fraction_fit,
										num_clients=self.n_clients,
										num_rounds=self.rounds,
										num_epochs=self.epochs,
										decay=self.decay,
										perc_of_clients=self.poc,
										strategy_name=self.strategy_name,
										dataset=self.dataset,
										model_name=self.model_name,
										new_clients=self.new_clients,
										new_clients_train=self.new_clients_train,
									  args=self.args)
			if self.strategy_name == 'FedDistill':
				# print("foi servidor")
				return FedDistillServerTorch(aggregation_method=self.aggregation_method,
										n_classes=self.n_classes,
										fraction_fit=self.fraction_fit,
										num_clients=self.n_clients,
										num_rounds=self.rounds,
										num_epochs=self.epochs,
										decay=self.decay,
										perc_of_clients=self.poc,
										strategy_name=self.strategy_name,
										dataset=self.dataset,
										model_name=self.model_name,
										new_clients=self.new_clients,
										new_clients_train=self.new_clients_train,
									  args=self.args)
			if self.strategy_name == 'FedPredict':
				# print("foi servidor")
				return FedPredictServerTorch(aggregation_method=self.aggregation_method,
											 n_classes=self.n_classes,
											 fraction_fit=self.fraction_fit,
											 num_clients=self.n_clients,
											 num_rounds=self.rounds,
											 args=self.args,
											 num_epochs=self.epochs,
											 model=copy.deepcopy(self.create_client(0).create_model()),
											 decay=self.decay,
											 perc_of_clients=self.poc,
											 strategy_name=self.strategy_name,
											 dataset=self.dataset,
											 model_name=self.model_name,
											 new_clients=self.new_clients,
											 new_clients_train=self.new_clients_train
									  		)
			if self.strategy_name == 'FedPredict_Dynamic':
				# print("foi servidor")
				return FedPredictDynamicServerTorch(aggregation_method=self.aggregation_method,
											 n_classes=self.n_classes,
											 fraction_fit=self.fraction_fit,
											 num_clients=self.n_clients,
											 num_rounds=self.rounds,
											 args=self.args,
											 num_epochs=self.epochs,
											 model=copy.deepcopy(self.create_client(0).create_model()),
											 decay=self.decay,
											 perc_of_clients=self.poc,
											 strategy_name=self.strategy_name,
											 dataset=self.dataset,
											 model_name=self.model_name,
											 new_clients=self.new_clients,
											 new_clients_train=self.new_clients_train
									  		)
			if self.strategy_name == 'FedKD_with_FedPredict':
				# print("foi servidor")
				return FedKDWithFedPredictServerTorch(aggregation_method=self.aggregation_method,
													  n_classes=self.n_classes,
													  fraction_fit=self.fraction_fit,
													  num_clients=self.n_clients,
													  num_rounds=self.rounds,
													  num_epochs=self.epochs,
													  decay=self.decay,
													  perc_of_clients=self.poc,
													  model=copy.deepcopy(
														  self.create_client(0).create_model_distillation().student),
													  strategy_name=self.strategy_name,
													  dataset=self.dataset,
													  model_name=self.model_name,
													  new_clients=self.new_clients,
													  new_clients_train=self.new_clients_train,
													  args=self.args)
			if self.strategy_name == 'FedPer_with_FedPredict':
				# print("foi servidor")
				return FedPer_with_FedPredictServerTorch(aggregation_method=self.aggregation_method,
											 n_classes=self.n_classes,
											 fraction_fit=self.fraction_fit,
											 num_clients=self.n_clients,
											 num_rounds=self.rounds,
											args=self.args,
											 num_epochs=self.epochs,
											 model=copy.deepcopy(self.create_client(0).create_model()),
											 decay=self.decay,
											 perc_of_clients=self.poc,
											 strategy_name=self.strategy_name,
											 dataset=self.dataset,
											 model_name=self.model_name,
											 new_clients=self.new_clients,
											 new_clients_train=self.new_clients_train
									  		)
			if self.strategy_name == 'FedClassAvg_with_FedPredict':
				# print("foi servidor")
				return FedClassAvg_with_FedPredictServerTorch(aggregation_method=self.aggregation_method,
											 n_classes=self.n_classes,
											 fraction_fit=self.fraction_fit,
											 num_clients=self.n_clients,
											 num_rounds=self.rounds,
											args=self.args,
											 num_epochs=self.epochs,
											 model=copy.deepcopy(self.create_client(0).create_model()),
											 decay=self.decay,
											 perc_of_clients=self.poc,
											 strategy_name=self.strategy_name,
											 dataset=self.dataset,
											 model_name=self.model_name,
											 new_clients=self.new_clients,
											 new_clients_train=self.new_clients_train
									  		)
			elif self.strategy_name == 'FedPer':
				return  FedPerServerTorch(aggregation_method=self.aggregation_method,
										n_classes=self.n_classes,
										fraction_fit=self.fraction_fit,
										num_clients=self.n_clients,
										num_rounds=self.rounds,
										num_epochs=self.epochs,
										decay=self.decay,
										perc_of_clients=self.poc,
										strategy_name=self.strategy_name,
										dataset=self.dataset,
										model_name=self.model_name,
										new_clients=self.new_clients,
										new_clients_train=self.new_clients_train,
									  args=self.args)
			elif self.strategy_name == 'FedClassAvg':
				return  FedClassAvgServerTorch(aggregation_method=self.aggregation_method,
												n_classes=self.n_classes,
												fraction_fit=self.fraction_fit,
												num_clients=self.n_clients,
												num_rounds=self.rounds,
												num_epochs=self.epochs,
												decay=self.decay,
												perc_of_clients=self.poc,
												strategy_name=self.strategy_name,
												dataset=self.dataset,
												model_name=self.model_name,
												new_clients=self.new_clients,
												new_clients_train=self.new_clients_train,
									  args=self.args)
			elif self.strategy_name == 'FedSparsification':
				return  FedSparsificationServerTorch(aggregation_method=self.aggregation_method,
												n_classes=self.n_classes,
												fraction_fit=self.fraction_fit,
												num_clients=self.n_clients,
												num_rounds=self.rounds,
												num_epochs=self.epochs,
												decay=self.decay,
												perc_of_clients=self.poc,
												strategy_name=self.strategy_name,
												dataset=self.dataset,
												model_name=self.model_name,
												new_clients=self.new_clients,
												new_clients_train=self.new_clients_train,
									  args=self.args)
			elif self.strategy_name == 'FedAla':
				return  FedAlaServerTorch(aggregation_method=self.aggregation_method,
												n_classes=self.n_classes,
												fraction_fit=self.fraction_fit,
												num_clients=self.n_clients,
												num_rounds=self.rounds,
												num_epochs=self.epochs,
												decay=self.decay,
												perc_of_clients=self.poc,
												strategy_name=self.strategy_name,
												dataset=self.dataset,
												model_name=self.model_name,
												new_clients=self.new_clients,
												new_clients_train=self.new_clients_train,
									  args=self.args)
			elif self.strategy_name == 'FedProx':
				return  FedProxServerTorch(aggregation_method=self.aggregation_method,
												n_classes=self.n_classes,
												fraction_fit=self.fraction_fit,
												num_clients=self.n_clients,
												num_rounds=self.rounds,
												num_epochs=self.epochs,
										   		model=copy.deepcopy(self.create_client(0).create_model()),
												decay=self.decay,
												perc_of_clients=self.poc,
												strategy_name=self.strategy_name,
												dataset=self.dataset,
												model_name=self.model_name,
												new_clients=self.new_clients,
												new_clients_train=self.new_clients_train,
									  args=self.args)
			elif self.strategy_name == 'FedLocal':
				return  FedLocalServerTorch(aggregation_method=self.aggregation_method,
											n_classes=self.n_classes,
											fraction_fit=self.fraction_fit,
											num_clients=self.n_clients,
											num_rounds=self.rounds,
											num_epochs=self.epochs,
											decay=self.decay,
											perc_of_clients=self.poc,
											strategy_name=self.strategy_name,
											dataset=self.dataset,
											model_name=self.model_name,
											new_clients=self.new_clients,
											new_clients_train=self.new_clients_train,
									  args=self.args)
			elif self.strategy_name == 'FedAvgM':
				return FedAvgMServerTorch(aggregation_method=self.aggregation_method,
										n_classes=self.n_classes,
										fraction_fit=self.fraction_fit,
										num_clients=self.n_clients,
										num_rounds=self.rounds,
										num_epochs=self.epochs,
										model=copy.deepcopy(self.create_client(0).create_model()),
										server_learning_rate=1, # melhor lr=1
										server_momentum=0.2, # melhor server_momentum=0.2
										decay=self.decay,
										perc_of_clients=self.poc,
										dataset=self.dataset,
										non_iid=self.non_iid,
										model_name=self.model_name,
										new_clients=self.new_clients,
										new_clients_train=self.new_clients_train,
									  args=self.args)
			elif self.strategy_name == 'QFedAvg':
				return QFedAvgServerTorch(aggregation_method=self.aggregation_method,
										n_classes=self.n_classes,
										fraction_fit=self.fraction_fit,
										num_clients=self.n_clients,
										num_rounds=self.rounds,
										num_epochs=self.epochs,
										model=copy.deepcopy(self.create_client(0).create_model()),
										server_learning_rate=1, # melhor lr=1
										q_param=0, # melhor server_momentum=0.2
										decay=self.decay,
										perc_of_clients=self.poc,
										dataset=self.dataset,
										model_name=self.model_name,
										new_clients_train=self.new_clients_train,
									  args=self.args)
			elif self.strategy_name == "FedYogi":
				return FedYogiServerTorch(aggregation_method=self.aggregation_method,
										n_classes=self.n_classes,
										fraction_fit=self.fraction_fit,
										num_clients=self.n_clients,
										num_rounds=self.rounds,
										num_epochs=self.epochs,
										model=copy.deepcopy(self.create_client(0).create_model()),
										decay=self.decay,
										perc_of_clients=self.poc,
										strategy_name=self.strategy_name,
										dataset=self.dataset,
										model_name=self.model_name,
										new_clients=self.new_clients,
										new_clients_train=self.new_clients_train,
									  args=self.args)
			elif self.strategy_name == "CDA-FedAvg":
				return CDAFedAvgServerTorch(aggregation_method=self.aggregation_method,
										n_classes=self.n_classes,
										fraction_fit=self.fraction_fit,
										num_clients=self.n_clients,
										num_rounds=self.rounds,
										num_epochs=self.epochs,
										model=copy.deepcopy(self.create_client(0).create_model()),
										decay=self.decay,
										perc_of_clients=self.poc,
										strategy_name=self.strategy_name,
										dataset=self.dataset,
										model_name=self.model_name,
										new_clients=self.new_clients,
										new_clients_train=self.new_clients_train,
									  args=self.args)
			elif self.strategy_name == "FedCDM":
				return FedCDMServerTorch(aggregation_method=self.aggregation_method,
										n_classes=self.n_classes,
										fraction_fit=self.fraction_fit,
										num_clients=self.n_clients,
										num_rounds=self.rounds,
										num_epochs=self.epochs,
										model=copy.deepcopy(self.create_client(0).create_model()),
										decay=self.decay,
										perc_of_clients=self.poc,
										strategy_name=self.strategy_name,
										dataset=self.dataset,
										model_name=self.model_name,
										new_clients=self.new_clients,
										new_clients_train=self.new_clients_train,
									  args=self.args)
			elif self.strategy_name == "FedClustering":
				return FedClusteringServerTorch(aggregation_method=self.aggregation_method,
										n_classes=self.n_classes,
										fraction_fit=self.fraction_fit,
										num_clients=self.n_clients,
										num_rounds=self.rounds,
										num_epochs=self.epochs,
										model=copy.deepcopy(self.create_client(0).create_model()),
										decay=self.decay,
										perc_of_clients=self.poc,
										strategy_name=self.strategy_name,
										dataset=self.dataset,
										model_name=self.model_name,
										new_clients=self.new_clients,
										new_clients_train=self.new_clients_train,
									  args=self.args)
			elif self.strategy_name == "FedYogi_with_FedPredict":
				return FedYogiWithFedPredictServerTorch(aggregation_method=self.aggregation_method,
										n_classes=self.n_classes,
										fraction_fit=self.fraction_fit,
										num_clients=self.n_clients,
										num_rounds=self.rounds,
										num_epochs=self.epochs,
										model=copy.deepcopy(self.create_client(0).create_model()),
										decay=self.decay,
										perc_of_clients=self.poc,
										strategy_name=self.strategy_name,
										dataset=self.dataset,
										model_name=self.model_name,
										new_clients=self.new_clients,
										new_clients_train=self.new_clients_train,
									  args=self.args)
			elif self.strategy_name == "FedPAQ":
				return FedPAQServerTorch(aggregation_method=self.aggregation_method,
										 n_classes=self.n_classes,
										 fraction_fit=self.fraction_fit,
										 num_clients=self.n_clients,
										 num_rounds=self.rounds,
										 num_epochs=self.epochs,
										 decay=self.decay,
										 perc_of_clients=self.poc,
										 model=copy.deepcopy(self.create_client(0).create_model()),
										 strategy_name=self.strategy_name,
										 dataset=self.dataset,
										 model_name=self.model_name,
										 new_clients=self.new_clients,
										 new_clients_train=self.new_clients_train,
										 args=self.args)
			elif self.strategy_name == "FetchSGD":
				return FetSGDServerTorch(aggregation_method=self.aggregation_method,
										 n_classes=self.n_classes,
										 fraction_fit=self.fraction_fit,
										 num_clients=self.n_clients,
										 num_rounds=self.rounds,
										 num_epochs=self.epochs,
										 decay=self.decay,
										 perc_of_clients=self.poc,
										 strategy_name=self.strategy_name,
										 dataset=self.dataset,
										 model_name=self.model_name,
										 new_clients=self.new_clients,
										 new_clients_train=self.new_clients_train,
										 args=self.args)
			elif self.strategy_name == "FedKD":
				return FedKDServerTorch(aggregation_method=self.aggregation_method,
										 n_classes=self.n_classes,
										 fraction_fit=self.fraction_fit,
										 num_clients=self.n_clients,
										 num_rounds=self.rounds,
										 num_epochs=self.epochs,
										 decay=self.decay,
										 perc_of_clients=self.poc,
										model=copy.deepcopy(self.create_client(0).create_model_distillation().student),
										 strategy_name=self.strategy_name,
										 dataset=self.dataset,
										 model_name=self.model_name,
										 new_clients=self.new_clients,
										 new_clients_train=self.new_clients_train,
										 args=self.args)
			else:
				return FedAvgServerTorch(aggregation_method=self.aggregation_method,
										n_classes=self.n_classes,
										fraction_fit=self.fraction_fit,
										num_clients=self.n_clients,
										num_rounds=self.rounds,
										num_epochs=self.epochs,
										decay=self.decay,
										perc_of_clients=self.poc,
										strategy_name=self.strategy_name,
										dataset=self.dataset,
										model_name=self.model_name,
										new_clients=self.new_clients,
										new_clients_train=self.new_clients_train,
									  args=self.args)


	def start_simulation(self):

		# ray_args = {
		# 	"include_dashboard"   : False,
		# 	"max_calls"           : 1,
		# 	"ignore_reinit_error" : True
		# }
		DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		print("device: ", DEVICE)
		if DEVICE.type == "cuda":
			client_resources = {"num_cpus": 6, "num_gpus": 1}

		fl.simulation.start_simulation(
						    client_fn     = self.create_client,
						    num_clients   = self.n_clients,
						    config        = fl.server.ServerConfig(num_rounds=self.rounds),
						    strategy      = self.create_strategy(),
						    #ray_init_args = ray_args
							# client_resources=client_resources
						)


def main():
	parser = OptionParser()

	parser.add_option("-c", "--clients",     		dest="n_clients",          default=10,        help="Number of clients in the simulation",    metavar="INT")
	parser.add_option("-s", "--strategy",    		dest="strategy_name",      default='FedSGD',  help="Strategy of the federated learning",     metavar="STR")
	parser.add_option("-a", "--aggregation_method", dest="aggregation_method", default='None',    help="Algorithm used for selecting clients",   metavar="STR")
	parser.add_option("-m", "--model",       		dest="model_name",         default='DNN',     help="Model used for trainning",               metavar="STR")
	parser.add_option("-d", "--dataset",     		dest="dataset",            default='MNIST',   help="Dataset used for trainning",             metavar="STR")
	parser.add_option("-e", "--epochs",      		dest="local_epochs",       default=1,         help="Number of epochs in each round",         metavar="STR")
	parser.add_option("-r", "--round",       		dest="rounds",             default=5,         help="Number of communication rounds",         metavar="INT")
	parser.add_option("",   "--poc",         		dest="poc",                default=0,         help="Percentage clients to be selected",      metavar="FLOAT")
	parser.add_option("",   "--decay",       		dest="decay",              default=0,         help="Decay factor for FL-H.IAAC",                metavar="FLOAT")
	parser.add_option("",   "--non-iid",     		dest="non_iid",            default=False,     help="Non IID distribution",                   metavar="BOOLEAN")
	parser.add_option("", "--use_gradient", dest="use_gradient", default=False, help="Whether to send gradient", metavar="BOOLEAN")
	parser.add_option("", "--m_combining_layers", dest="m_combining_layers", default=1, help="Number of layers to combine from the last/prediction layer to the top", metavar="INT")
	parser.add_option("", "--fraction_fit", dest="fraction_fit", default=0, help="fraction of selected clients to be trained", metavar="FLOAT")
	parser.add_option("-y", "--classes",     		dest="n_classes",          default=10,        help="Number of classes",                      metavar="INT")
	parser.add_option("-t", "--type",               dest="type",               default='tf',      help="Neural network framework (tf or torch)", metavar="STR")
	parser.add_option("", "--new_clients", dest="new_clients", default='False', help="Add new clients after a specific number of rounds",
					  metavar="STR")
	parser.add_option("", "--dynamic_data", dest="dynamic_data", default='no',
					  help="Change local data",
					  metavar="STR")
	parser.add_option("", "--new_clients_train", dest="new_clients_train", default='False',
					  help="wheter to train or not new clients",
					  metavar="STR")
	parser.add_option("--class_per_client", help="Number of classes per client", default=2)
	parser.add_option("--train_perc", help="Dataset percentual for train", default=0.8)
	parser.add_option("--alpha", help="Dirichlet alpha parameter", default=0.1)
	parser.add_option("--T", help="T parameter for FedPredict", default=0)
	parser.add_option("--comment", help="comment for additional iformation of the experiment", default="")
	parser.add_option("--n_rate", help="n components rate", default=0.3)
	parser.add_option("--bits", help="bits for compression_methods", default=8)
	parser.add_option("--compression_method", help="", default=0)
	parser.add_option("", "--n_clusters", dest="n_clusters", default=1, help="")
	parser.add_option("", "--clustering", dest="clustering", default=3600, help="")
	parser.add_option("", "--cluster_round", dest="cluster_round", default=20, help="")
	parser.add_option("", "--cluster_metric", dest="cluster_metric", default=1, help="")
	parser.add_option("", "--metric_layer", dest="metric_layer", default=1, help="")
	parser.add_option("", "--cluster_method", dest="cluster_method", default=1, help="")

	(opt, args) = parser.parse_args()
	# opt.n_rate = float(opt.comment)

	print("Simulacao da estratégia: ", opt.strategy_name, opt.fraction_fit)
	simulation = SimulationFL(n_clients=int(opt.n_clients), aggregation_method=opt.aggregation_method, model_name=opt.model_name,
                              strategy_name=opt.strategy_name, dataset=opt.dataset, n_classes=int(opt.n_classes),
                              local_epochs=int(opt.local_epochs), rounds=int(opt.rounds), poc=float(opt.poc), fraction_fit=float(opt.fraction_fit),
                              decay=float(opt.decay), non_iid=ast.literal_eval(opt.non_iid), m_combining_layers=int(opt.m_combining_layers), nn_type=opt.type,
                              new_clients=ast.literal_eval(opt.new_clients), new_clients_train=ast.literal_eval(opt.new_clients_train), args=opt)

	simulation.start_simulation()



if __name__ == '__main__':
	main()
