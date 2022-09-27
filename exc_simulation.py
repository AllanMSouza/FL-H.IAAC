import flwr as fl
import tensorflow as tf
from client import FedClient
from servidor import FedServer

EPOCHS        = (1, 10)
NUM_CLIENTS   = (10, 25, 50, 100, 500)
MODELS        = ('Logist Regression', 'DNN', 'CNN')
ALGORITHMS    = ('None', 'POC', 'FedLTA',)
DATASETS      = ('MNIST', 'CIFAR10', )
CLIENTS2SELCT = (0.25, 0.5, 0.75)
DECAY         = (0.001, 0.005, 0.009)
ROUNDS        = 500

def create_client(cid):
	client_selection = False

	if algorithm != 'None':
		client_selection = True

	return FedClient(cid, 
					num_clients, 
					model_name         = model_name, 
					client_selection   = client_selection, 
					epochs             = epochs, 
					solution_name      = solution_name,
					aggregation_method = algorithm,
					dataset            = dataset,
					perc_of_clients    = perc_of_clients,
					decay              = decay
					)

def create_strategy():
	return FedServer(aggregation_method = algorithm, 
					 fraction_fit       = 1, 
					 num_clients        = num_clients, 
					 decay              = decay, 
					 perc_of_clients    = perc_of_clients, 
					 solution_name      = solution_name,
					 dataset            = dataset,
					 model_name        = model_name)


def exec_simulation():
	fl.simulation.start_simulation(
						    client_fn   = create_client,
						    num_clients = num_clients,
						    config      = fl.server.ServerConfig(num_rounds=ROUNDS),
						    strategy    = create_strategy(),
						)


def main():
	global num_clients
	global model_name
	global epochs
	global client_selection
	global perc_of_clients
	global aggregation_method
	global solution_name
	global dataset
	global algorithm

	global clients2select
	global decay

	for local_epoch in EPOCHS:
		if local_epoch == 1:
			solution_name = 'FedSGD'
		else:
			solution_name = 'FedAvg'

		for local_algorithm in ALGORITHMS:
			for local_clients in NUM_CLIENTS:
				for local_model_name in MODELS:
					for local_dataset in DATASETS:

						epochs      = local_epoch
						algorithm   = local_algorithm
						num_clients = local_clients
						model_name  = local_model_name
						dataset     = local_dataset

						if local_algorithm == 'POC':
							for local_clients2select in CLIENTS2SELCT:
								perc_of_clients = local_clients2select
								decay           = 0
								exec_simulation()

						elif local_algorithm == 'FedLTA':
							for local_decay in DECAY:
								decay           = local_decay
								perc_of_clients = 0
								exec_simulation()

						elif local_algorithm == 'None':
							decay           = 0
							perc_of_clients = 0
							exec_simulation()

						

						

if __name__ == '__main__':
	main()
