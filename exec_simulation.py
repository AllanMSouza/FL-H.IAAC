import flwr as fl
import tensorflow as tf
from optparse import OptionParser

import subprocess
import logging
import os
import sys

logging.getLogger("tensorflow").setLevel(logging.ERROR) 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

EPOCHS        = (1,)
#NUM_CLIENTS   = (50, )
CLIENTS       = {'MNIST': 50}
MODELS        = ('DNN',)
ALGORITHMS    = ('POC',)
#DATASETS      = ('MNIST', 'CIFAR10', )
DATASETS      = ('MNIST',)
CLIENTS2SELCT = (0.2,)
NEW_CLIENTS = ('FALSE',)
# DECAY         = (0.001, 0.005, 0.009)
ROUNDS        = 2
# STRATEGIES 		= ('FedAVG', 'FedAvgM', 'FedClassAvg''QFedAvg', 'FedPer', 'FedProto', 'FedYogi', 'FedLocal',)
STRATEGIES 		= ['FedAVG', 'FedAvgM']
COMENTARIO = ''


def exec_fedsgd(non_iid):

	for dataset in DATASETS:
		for epochs in EPOCHS:
			for model in MODELS:
				print(f'Strating FedSGD simulation for {dataset} clients with {model} model ...')
				subprocess.Popen(['python3', 'simulation.py', '-c', str(CLIENTS[dataset]), '-a', 'None', '-m', model,
												'-d', dataset, '-e', str(epochs), '-r', str(ROUNDS),
												'--non-iid', 'False']).wait()

				# subprocess.Popen(['rm', '-fr', '/tmp/ray/']).wait()
				# subprocess.Popen(['rm', '/tmp/*.py']).wait()

def exec_poc(non_iid):
	try:
		for dataset in DATASETS:
			for epochs in EPOCHS:
				for poc in CLIENTS2SELCT:
					for new_client in NEW_CLIENTS:
						for model in MODELS:
							for strategy in STRATEGIES:
								print(f'Strating POC-{poc} simulation for {dataset} clients with {model} model ...', os.getcwd())
								test_config = """python {}/simulation.py --dataset='{}' --model='{}' --strategy='{}' --round={} --client={} --type='torch' --non-iid={} --aggregation_method='{}' --poc={} --new_clients={}""".format(os.getcwd(), dataset, model, strategy, ROUNDS, CLIENTS[dataset], True, 'POC',  poc, new_client)
								print("=====================================\nExecutando... \n", test_config, "\n=====================================")
								subprocess.Popen(test_config, shell=True).wait()

								# subprocess.Popen(['rm', '-fr', '/tmp/ray/']).wait()
								# subprocess.Popen(['rm', '/tmp/*.py']).wait()
							strategies_arg = ""
							for i in STRATEGIES:
								strategies_arg = strategies_arg + """ --strategy='{}'""".format(i)
							analytics_result_dir = """python analysis/non_iid.py --dataset='{}' --model='{}' --round={} --client={} --aggregation_method='{}' --poc={} --new_clients={} {}""".format(dataset, model, 50, CLIENTS[dataset], 'POC',  poc, new_client, strategies_arg)
							print("=====================================\nExecutando analytics... \n", analytics_result_dir,
								  "\n=====================================")
							subprocess.Popen(analytics_result_dir, shell=True).wait()

							# subprocess.Popen(['rm', '-fr', '/tmp/ray/']).wait()
							# subprocess.Popen(['rm', '/tmp/*.py']).wait()
	except Exception as e:
		print("Error on POC")
		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def exec_fedlta(non_iid):

	for dataset in DATASETS:
		for epochs in EPOCHS:
			for model in MODELS:
				for decay in DECAY:
					print(f'Strating FedLTA with decay {decay} simulation for {dataset} clients with {model} model ...')
					subprocess.Popen(['python3', 'simulation.py', '-c', str(CLIENTS[dataset]), '-a', 'FedLTA', '-m', model,
												 '-d', dataset, '-e', str(epochs), '-r', str(ROUNDS),
												 '--decay', str(decay), '--non-iid', 'False']).wait()

					subprocess.Popen(['rm', '-fr', '/tmp/ray/']).wait()
					subprocess.Popen(['rm', '/tmp/*.py']).wait()


def main():
	parser = OptionParser()

	# parser.add_option("-a", "--algorithm", dest="algorithm", default='None',   help="Algorithm used for selecting clients", metavar="STR")
	parser.add_option("",   "--non-iid",   dest="non_iid",   default='False',  help="Non IID Distribution", metavar="STR")
	parser.add_option("-a", "--aggregation_method", dest="aggregation_method", default='None', help="Non IID Distribution", metavar="STR")

	(opt, args) = parser.parse_args()

	if opt.aggregation_method == 'None':
		exec_fedsgd(opt.non_iid)

	elif opt.aggregation_method == 'POC':
		exec_poc(opt.non_iid)

	elif opt.aggregation_method == 'FedLTA':
		exec_fedlta(opt.non_iid)
		
		





if __name__ == '__main__':
	main()
