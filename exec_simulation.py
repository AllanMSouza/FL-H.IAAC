import flwr as fl
import tensorflow as tf
from optparse import OptionParser

import subprocess
import logging
import os

logging.getLogger("tensorflow").setLevel(logging.ERROR) 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

EPOCHS        = (10,)
NUM_CLIENTS   = (50, )
MODELS        = ('Logist Regression', 'DNN', 'CNN')
ALGORITHMS    = ('None', 'POC', 'FedLTA',)
DATASETS      = ('MNIST', 'CIFAR10', )
CLIENTS2SELCT = (0.25, 0.5, 0.75)
DECAY         = (0.001, 0.005, 0.009)
ROUNDS        = 100


def exec_fedsgd(non_iid):

	for n_clients in NUM_CLIENTS:
		for epochs in EPOCHS:
			for dataset in DATASETS:
				for model in MODELS:
					print(f'Strating FedSGD simulation for {n_clients} clients with {model} model ...')
					subprocess.Popen(['python3', 'simulation.py', '-c', str(n_clients), '-a', 'None', '-m', model, 
													'-d', dataset, '-e', str(epochs), '-r', str(ROUNDS), 
													'--non-iid', non_iid]).wait()

					subprocess.Popen(['rm', '-fr', '/tmp/ray/']).wait()
					subprocess.Popen(['rm', '/tmp/*.py']).wait()

def exec_poc(non_iid):

	for n_clients in NUM_CLIENTS:
		for epochs in EPOCHS:
			for dataset in DATASETS:
				for model in MODELS:
					for poc in CLIENTS2SELCT:
						print(f'Strating POC-{poc} simulation for {n_clients} clients with {model} model ...')
						subprocess.Popen(['python3', 'simulation.py', '-c', str(n_clients), '-a', 'POC', '-m', model, 
													 '-d', dataset, '-e', str(epochs), '-r', str(ROUNDS), 
													 '--poc', str(poc), '--non-iid', non_iid]).wait()

						subprocess.Popen(['rm', '-fr', '/tmp/ray/']).wait()
						subprocess.Popen(['rm', '/tmp/*.py']).wait()


def exec_fedlta(non_iid):

	for n_clients in NUM_CLIENTS:
		for epochs in EPOCHS:
			for dataset in DATASETS:
				for model in MODELS:
					for decay in DECAY:
						print(f'Strating FedLTA with decay {decay} simulation for {n_clients} clients with {model} model ...')
						subprocess.Popen(['python3', 'simulation.py', '-c', str(n_clients), '-a', 'FedLTA', '-m', model, 
													 '-d', dataset, '-e', str(epochs), '-r', str(ROUNDS), 
													 '--decay', str(decay), '--non-iid', non_iid]).wait()

						subprocess.Popen(['rm', '-fr', '/tmp/ray/']).wait()
						subprocess.Popen(['rm', '/tmp/*.py']).wait()


def main():
	parser = OptionParser()

	parser.add_option("-a", "--algorithm", dest="algorithm", default='None',   help="Algorithm used for selecting clients", metavar="STR")
	parser.add_option("",   "--non-iid",   dest="non_iid",   default='Flase',  help="Non IID Distribution", metavar="STR")

	(opt, args) = parser.parse_args()

	if opt.algorithm == 'None':
		exec_fedsgd(opt.non_iid)

	elif opt.algorithm == 'POC':
		exec_poc(opt.non_iid)

	elif opt.algorithm == 'FedLTA':
		exec_fedlta(opt.non_iid)
		
		





if __name__ == '__main__':
	main()
