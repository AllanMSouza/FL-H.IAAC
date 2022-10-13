import flwr as fl
import tensorflow as tf
from client import FedClient
from servidor import FedServer

from optparse import OptionParser

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

class SimulationFL():
	"""docstring for Simulation"""
	def __init__(self, n_clients, algorithm, model_name, dataset, local_epochs, rounds, poc, decay, non_iid):
		
		self.n_clients        = n_clients
		self.algorithm        = algorithm
		self.model_name       = model_name
		self.dataset          = dataset
		self.epochs           = local_epochs
		self.rounds           = rounds
		self.poc              = poc
		self.decay            = decay
		self.client_selection = False
		self.solution_name    = 'FedSGD'
		self.non_iid          = non_iid

	
	def create_client(self, cid):

		if self.algorithm != 'None':
			self.client_selection = True

		if self.epochs > 1:
			self.solution_name = 'FedAVG'

		return FedClient(cid, 
					self.n_clients, 
					model_name         = self.model_name, 
					client_selection   = self.client_selection, 
					epochs             = self.epochs, 
					solution_name      = self.solution_name,
					aggregation_method = self.algorithm,
					dataset            = self.dataset,
					perc_of_clients    = self.poc,
					decay              = self.decay,
					non_iid            = self.non_iid)

	def create_strategy(self):

		if self.epochs > 1:
			self.solution_name = 'FedAVG'

		return FedServer(aggregation_method = self.algorithm, 
					     fraction_fit       = 1, 
					     num_clients        = self.n_clients, 
					     decay              = self.decay, 
					     perc_of_clients    = self.poc, 
					     solution_name      = self.solution_name,
					     dataset            = self.dataset,
					     model_name         = self.model_name)


	def start_simulation(self):

		fl.simulation.start_simulation(
						    client_fn   = self.create_client,
						    num_clients = self.n_clients,
						    config      = fl.server.ServerConfig(num_rounds=self.rounds),
						    strategy    = self.create_strategy(),
						)


def main():
	parser = OptionParser()

	parser.add_option("-c", "--clients",   dest="n_clients",     default=10,      help="Number of clients in the simulation", metavar="INT")
	parser.add_option("-a", "--algorithm", dest="algorithm",     default='None',  help="Algorithm used for selecting clients", metavar="STR")
	parser.add_option("-m", "--model",     dest="model_name",    default='DNN',   help="Model used for trainning", metavar="STR")
	parser.add_option("-d", "--dataset",   dest="dataset",       default='MNIST', help="Dataset used for trainning", metavar="STR")
	parser.add_option("-e", "--epochs",    dest="local_epochs",  default=1,       help="Number of epochs in each round", metavar="STR")
	parser.add_option("-r", "--round",     dest="rounds",        default=100,     help="Number of communication rounds", metavar="INT")
	parser.add_option("", "--poc",         dest="poc",           default=0,       help="Percentage clients to be selected", metavar="FLOAT")
	parser.add_option("", "--decay",       dest="decay",         default=0,       help="Decay factor for FedLTA", metavar="FLOAT")
	parser.add_option("", "--non-iid",     dest="non_iid",       default=False,   help="Non IID distribution", metavar="BOOLEAN")

	(opt, args) = parser.parse_args()

	simulation = SimulationFL(int(opt.n_clients), opt.algorithm, opt.model_name, opt.dataset, 
							  int(opt.local_epochs), int(opt.rounds), float(opt.poc), float(opt.decay),
							  bool(opt.non_iid))

	simulation.start_simulation()



if __name__ == '__main__':
	main()
