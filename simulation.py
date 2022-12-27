import flwr as fl
from client import FedAvgClientTf, FedPerClientTf, FedProtoClientTf, FedLocalClientTf, FedAvgClientTorch, FedProtoClientTorch, FedPerClientTorch, FedLocalClientTorch, FedAvgMClientTorch, QFedAvgClientTorch
from server import FedPerServerTf, FedProtoServerTf, FedAvgServerTf, FedLocalServerTf, FedAvgServerTorch, FedProtoServerTorch, FedPerServerTorch, FedLocalServerTorch, FedAvgMServerTorch, QFedAvgServerTorch

from optparse import OptionParser
import tensorflow as tf
import torch
import random
import numpy as np
import copy
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

tf.random.set_seed(0)

class SimulationFL():
	"""docstring for Simulation"""
	def __init__(self,
				 n_clients,
				 aggregation_method,
				 model_name,
				 strategy_name,
				 dataset,
				 n_classes,
				 local_epochs,
				 rounds,
				 poc,
				 decay,
				 non_iid,
				 nn_type,
				 new_clients
				 ):
		
		self.n_clients        		= n_clients
		self.aggregation_method     = aggregation_method
		self.model_name       		= model_name # cnn, dnn, among others
		self.dataset          		= dataset
		self.n_classes 				= n_classes
		self.epochs           		= local_epochs
		self.rounds           		= rounds
		self.poc              		= poc
		self.decay            		= decay
		self.client_selection 		= False
		self.strategy_name    		= strategy_name # Old "self.solution_name"
		self.non_iid          		= non_iid
		self.nn_type = nn_type
		self.new_clients = new_clients

	
	def create_client(self, cid):

		if self.aggregation_method != 'None':
			self.client_selection = True

		if self.epochs > 1:
			self.strategy_name = 'FedAVG'

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
									  decay=self.decay,
									  non_iid=self.non_iid)

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
										decay=self.decay,
										non_iid=self.non_iid)

			elif self.strategy_name == 'FedProto':
				return FedProtoClientTf(cid=cid,
										n_clients=self.n_clients,
										n_classes=self.n_classes,
										epochs=1,
										model_name=self.model_name,
										client_selection=self.client_selection,
										solution_name=self.strategy_name,
										aggregation_method=self.aggregation_method,
										dataset=self.dataset,
										perc_of_clients=self.poc,
										decay=self.decay,
										non_iid=self.non_iid)

			else:
				return FedAvgClientTf(cid=cid,
									  n_clients=self.n_clients,
									  n_classes=self.n_classes,
									  model_name=self.model_name,
									  client_selection=self.client_selection,
									  epochs=1,
									  solution_name=self.strategy_name,
									  aggregation_method=self.aggregation_method,
									  dataset=self.dataset,
									  perc_of_clients=self.poc,
									  decay=self.decay,
									  non_iid=self.non_iid)
		elif self.nn_type == 'torch':
			if self.strategy_name == 'FedProto':
				# print("foi cliente")
				return FedProtoClientTorch(cid=cid,
										n_clients=self.n_clients,
										n_classes=self.n_classes,
										epochs=1,
										model_name=self.model_name,
										client_selection=self.client_selection,
										solution_name=self.strategy_name,
										aggregation_method=self.aggregation_method,
										dataset=self.dataset,
										perc_of_clients=self.poc,
										decay=self.decay,
										non_iid=self.non_iid,
										   new_clients=self.new_clients)
			elif self.strategy_name == 'FedPer':
				return  FedPerClientTorch(cid=cid,
								n_clients=self.n_clients,
								n_classes=self.n_classes,
								epochs=1,
								model_name=self.model_name,
								client_selection=self.client_selection,
								solution_name=self.strategy_name,
								aggregation_method=self.aggregation_method,
								dataset=self.dataset,
								perc_of_clients=self.poc,
								decay=self.decay,
								non_iid=self.non_iid,
								new_clients=self.new_clients)
			elif self.strategy_name == 'FedLocal':
				return  FedLocalClientTorch(cid=cid,
								n_clients=self.n_clients,
								n_classes=self.n_classes,
								epochs=1,
								model_name=self.model_name,
								client_selection=self.client_selection,
								solution_name=self.strategy_name,
								aggregation_method=self.aggregation_method,
								dataset=self.dataset,
								perc_of_clients=self.poc,
								decay=self.decay,
								non_iid=self.non_iid,
								new_clients=self.new_clients)
			elif self.strategy_name == 'FedAvgM':
				return FedAvgMClientTorch(cid=cid,
										 n_clients=self.n_clients,
										 n_classes=self.n_classes,
										 model_name=self.model_name,
										 client_selection=self.client_selection,
										 epochs=1,
										 solution_name=self.strategy_name,
										 aggregation_method=self.aggregation_method,
										 dataset=self.dataset,
										 perc_of_clients=self.poc,
										 decay=self.decay,
										 non_iid=self.non_iid,
										 new_clients=self.new_clients)
			elif self.strategy_name == 'QFedAvg':
				return QFedAvgClientTorch(cid=cid,
										 n_clients=self.n_clients,
										 n_classes=self.n_classes,
										 model_name=self.model_name,
										 client_selection=self.client_selection,
										 epochs=1,
										 solution_name=self.strategy_name,
										 aggregation_method=self.aggregation_method,
										 dataset=self.dataset,
										 perc_of_clients=self.poc,
										 decay=self.decay,
										 non_iid=self.non_iid,
										 new_clients=self.new_clients)
			else:
				return FedAvgClientTorch(cid=cid,
									  n_clients=self.n_clients,
									  n_classes=self.n_classes,
									  model_name=self.model_name,
									  client_selection=self.client_selection,
									  epochs=1,
									  solution_name=self.strategy_name,
									  aggregation_method=self.aggregation_method,
									  dataset=self.dataset,
									  perc_of_clients=self.poc,
									  decay=self.decay,
									  non_iid=self.non_iid,
										new_clients=self.new_clients)


	def create_strategy(self):

		if self.epochs > 1:
			self.strategy_name = 'FedAVG'


		if self.nn_type == 'tf':
			if self.strategy_name == 'FedPer':
				return FedPerServerTf(aggregation_method=self.aggregation_method,
									n_classes=self.n_classes,
									fraction_fit=1,
									num_clients=self.n_clients,
									num_rounds=self.rounds,
									decay=self.decay,
									perc_of_clients=self.poc,
									strategy_name=self.strategy_name,
									dataset=self.dataset,
									model_name=self.model_name)

			elif self.strategy_name == 'FedLocal':
				return FedLocalServerTf(aggregation_method=self.aggregation_method,
									n_classes=self.n_classes,
									fraction_fit=1,
									num_clients=self.n_clients,
									num_rounds=self.rounds,
									decay=self.decay,
									perc_of_clients=self.poc,
									strategy_name=self.strategy_name,
									dataset=self.dataset,
									model_name=self.model_name)

			elif self.strategy_name == 'FedProto':
				return FedProtoServerTf(aggregation_method=self.aggregation_method,
									  n_classes=self.n_classes,
									  fraction_fit=1,
									  num_clients=self.n_clients,
									  num_rounds=self.rounds,
									  decay=self.decay,
									  perc_of_clients=self.poc,
									  strategy_name=self.strategy_name,
									  dataset=self.dataset,
									  model_name=self.model_name)
			elif self.strategy_name == 'FedLocal':
				return FedLocalServerTf(aggregation_method=self.aggregation_method,
									  n_classes=self.n_classes,
									  fraction_fit=1,
									  num_clients=self.n_clients,
									  num_rounds=self.rounds,
									  decay=self.decay,
									  perc_of_clients=self.poc,
									  strategy_name=self.strategy_name,
									  dataset=self.dataset,
									  model_name=self.model_name)
			else:
				return FedAvgServerTf(aggregation_method=self.aggregation_method,
									n_classes=self.n_classes,
									fraction_fit=1,
									num_clients=self.n_clients,
									num_rounds=self.rounds,
									decay=self.decay,
									perc_of_clients=self.poc,
									strategy_name=self.strategy_name,
									dataset=self.dataset,
									model_name=self.model_name)
		elif self.nn_type == 'torch':
			if self.strategy_name == 'FedProto':
				# print("foi servidor")
				return FedProtoServerTorch(aggregation_method=self.aggregation_method,
										n_classes=self.n_classes,
										fraction_fit=1,
										num_clients=self.n_clients,
										num_rounds=self.rounds,
										decay=self.decay,
										perc_of_clients=self.poc,
										strategy_name=self.strategy_name,
										dataset=self.dataset,
										model_name=self.model_name,
										   new_clients=self.new_clients)
			elif self.strategy_name == 'FedPer':
				return  FedPerServerTorch(aggregation_method=self.aggregation_method,
								  n_classes=self.n_classes,
								  fraction_fit=1,
								  num_clients=self.n_clients,
								  num_rounds=self.rounds,
								  decay=self.decay,
								  perc_of_clients=self.poc,
								  strategy_name=self.strategy_name,
								  dataset=self.dataset,
								  model_name=self.model_name,
										  new_clients=self.new_clients)
			elif self.strategy_name == 'FedLocal':
				return  FedLocalServerTorch(aggregation_method=self.aggregation_method,
								  n_classes=self.n_classes,
								  fraction_fit=1,
								  num_clients=self.n_clients,
								  num_rounds=self.rounds,
								  decay=self.decay,
								  perc_of_clients=self.poc,
								  strategy_name=self.strategy_name,
								  dataset=self.dataset,
								  model_name=self.model_name)
			elif self.strategy_name == 'FedAvgM':
				return FedAvgMServerTorch(aggregation_method=self.aggregation_method,
										n_classes=self.n_classes,
										fraction_fit=1,
										num_clients=self.n_clients,
										num_rounds=self.rounds,
										model=copy.deepcopy(self.create_client(0).create_model()),
										server_learning_rate=1, # melhor lr=1
										server_momentum=0.1, # melhor server_momentum=0.2
										decay=self.decay,
										perc_of_clients=self.poc,
										dataset=self.dataset,
										model_name=self.model_name)
			elif self.strategy_name == 'QFedAvg':
				return QFedAvgServerTorch(aggregation_method=self.aggregation_method,
										n_classes=self.n_classes,
										fraction_fit=1,
										num_clients=self.n_clients,
										num_rounds=self.rounds,
										model=copy.deepcopy(self.create_client(0).create_model()),
										server_learning_rate=1, # melhor lr=1
										q_param=0.1, # melhor server_momentum=0.2
										decay=self.decay,
										perc_of_clients=self.poc,
										dataset=self.dataset,
										model_name=self.model_name)
			else:
				return FedAvgServerTorch(aggregation_method=self.aggregation_method,
								  n_classes=self.n_classes,
								  fraction_fit=1,
								  num_clients=self.n_clients,
								  num_rounds=self.rounds,
								  decay=self.decay,
								  perc_of_clients=self.poc,
								  strategy_name=self.strategy_name,
								  dataset=self.dataset,
								  model_name=self.model_name,
								new_clients=self.new_clients)


	def start_simulation(self):

		# ray_args = {
		# 	"include_dashboard"   : False,
		# 	"max_calls"           : 1,
		# 	"ignore_reinit_error" : True
		# }

		fl.simulation.start_simulation(
						    client_fn     = self.create_client,
						    num_clients   = self.n_clients,
						    config        = fl.server.ServerConfig(num_rounds=self.rounds),
						    strategy      = self.create_strategy(),
						    #ray_init_args = ray_args
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
	parser.add_option("",   "--decay",       		dest="decay",              default=0,         help="Decay factor for FedLTA",                metavar="FLOAT")
	parser.add_option("",   "--non-iid",     		dest="non_iid",            default=False,     help="Non IID distribution",                   metavar="BOOLEAN")
	parser.add_option("-y", "--classes",     		dest="n_classes",          default=10,        help="Number of classes",                      metavar="INT")
	parser.add_option("-t", "--type",               dest="type",               default='tf',      help="Neural network framework (tf or torch)", metavar="STR")
	parser.add_option("", "--new_clients", dest="new_clients", default=False, help="Add new clients after a specific number of rounds",
					  metavar="STR")

	(opt, args) = parser.parse_args()

	simulation = SimulationFL(int(opt.n_clients), opt.aggregation_method, opt.model_name, opt.strategy_name, opt.dataset, int(opt.n_classes),
							  int(opt.local_epochs), int(opt.rounds), float(opt.poc), float(opt.decay),
							  bool(opt.non_iid), opt.type, opt.new_clients)

	simulation.start_simulation()



if __name__ == '__main__':
	main()
