from optparse import OptionParser

import subprocess
import logging
import os
import sys

def remove_lines(filename):
	with open(filename, 'r') as fp:
		# read an store all lines into list
		lines = fp.readlines()

	# Write file
	with open(filename, 'w') as fp:
		# iterate each line
		for number, line in enumerate(lines):
			# delete line 5 and 8. or pass any Nth line you want to remove
			# note list index starts from 0
			if "[2m[36m(launch_and_fit pid" not in line and "[2m[36m(launch_and_get_parameters" not in line:
				fp.write(line)

logging.getLogger("tensorflow").setLevel(logging.ERROR) 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# ****** Experiments descriptions ******
# ======================================================================
# EXPERIMENT 1
# Algorithm = 'None'
# ======================================================================
# EXPERIMENT 2
# Algorithm = 'None', poc=[0.1, 0.2, 0.3] and  new_client = FALSE
# ======================================================================
# EXPERIMENT 3
# Algorithm = 'None', poc=[0.1, 0.2, 0.3] and new_client = TRUE new_clients_train = FALSE
# ======================================================================
# EXPERIMENT 4
# Algorithm = 'None', poc=[0.1, 0.2, 0.3] and new_client = TRUE new_clients_train = TRUE
# ======================================================================
# EXPERIMENT 5
# Algorithm = 'POC, poc=[0.1, 0.2, 0.3]' and new_client = TRUE new_clients_train = TRUE
#  Local epochs = 2
# ======================================================================
# Configurations
TYPE = 'torch'
# DATASETS      				= ['MNIST', 'CIFAR10']
DATASETS      				= ['CIFAR10']
# DATASETS      					= ['UCIHAR', 'MotionSense']
MODELS        					= ['CNN']
ALGORITHMS    					= ['None', 'POC', 'FedLTA']
EPOCHS        					= {'1': [1], '2': [1], '3': [1], '4': [1], '5': [2], '6': [1]}
# CLIENTS       				= {'MNIST': 50, 'CIFAR10': 50, 'CIFAR100': 50, 'MotionSense': 50, 'UCIHAR': 50}
CLIENTS       					= {'MNIST': [50], 'CIFAR10': [50], 'CIFAR100': [50], 'MotionSense': [24], 'UCIHAR': [30]}
FRACTION_FIT 					= {'None': [1], 'POC': [0.2], 'FedLTA': [0.2]}
POC 							= {'None': [0], 'POC': [0.2], 'FedLTA': [0]}
DECAY							= {'None': 0, 'POC': 0, 'FedLTA': 0.1}
NEW_CLIENTS 					= {'None': ['FALSE'], 'POC': ['FALSE', 'TRUE']}
NEW_CLIENTS_TRAIN 				= {'FALSE': ['FALSE'], 'TRUE': ['FALSE', 'TRUE']}
# DECAY         				= (0.001, 0.005, 0.009)
ROUNDS        					= 30
# STRATEGIES 					= ('FedAVG', 'FedAvgM', 'FedClassAvg', 'QFedAvg', 'FedPer', 'FedProto', 'FedYogi', 'FedLocal',)
STRATEGIES_FOR_ANALYSIS 		= ['FedPredict', 'FedAVG']
STRATEGIES_TO_EXECUTE 			= ['FedPredict', 'FedPer', 'FedAVG']

EXPERIMENTS 		= {1: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'comment': ''},
					   2: {'algorithm': 'None', 'new_client': 'False', 'new_client_train': 'False', 'comment': ''},
					   3: {'algorithm': 'None', 'new_client': 'True', 'new_client_train': 'False', 'comment': """apos a rodada {}, apenas novos clientes sao testados""".format(int(ROUNDS*0.7))},
					   4: {'algorithm': 'None', 'new_client': 'True', 'new_client_train': 'True', 'comment': """apos a rodada {}, apenas novos clientes sao testados - novos clientes treinam apenas 1 vez (um round) - """.format(int(ROUNDS*0.7))},
					   5: {'algorithm': 'None', 'new_client': 'True', 'new_client_train': 'True', 'comment': """apos a rodada {}, apenas novos clientes sao testados - novos clientes treinam apenas 1 vez (um round) com duas épocas locais """.format(int(ROUNDS*0.7))},
					  6: {'algorithm': 'FedLTA', 'new_client': 'False', 'new_client_train': 'False', 'comment': ''}}

def execute_experiment(experiment, algorithm, new_client, new_client_train, comment, type):

	try:
		for dataset in DATASETS:
			for model in MODELS:
				for epochs in EPOCHS[experiment]:
					for clients in CLIENTS[dataset]:
						for fraction_fit in FRACTION_FIT[algorithm]:
							for poc in POC[algorithm]:
								for strategy in STRATEGIES_TO_EXECUTE:
									decay = DECAY[algorithm]
									print(f'Starting {strategy} fraction_fit-{fraction_fit} simulation for {dataset} clients with {model} model ...', os.getcwd())
									test_config = """python {}/simulation.py --dataset='{}' --model='{}' --strategy='{}' --epochs={} --round={} --client={} --type='{}' --non-iid={} --aggregation_method='{}' --fraction_fit={} --poc={} --new_clients={} --new_clients_train={} --decay={}""".format(os.getcwd(), dataset, model,
																	 strategy, epochs, ROUNDS, clients, TYPE, True, algorithm,  fraction_fit, poc, new_client, new_client_train, decay)
									print("=====================================\nExecutando... \n", test_config, "\n=====================================")
									#exit()
									subprocess.Popen(test_config, shell=True).wait()
									pass
									# subprocess.Popen(['rm', '-fr', '/tmp/ray/']).wait()
									# subprocess.Popen(['rm', '/tmp/*.py']).wait()
								strategies_arg = ""
								for i in STRATEGIES_FOR_ANALYSIS:
									strategies_arg = strategies_arg + """ --strategy='{}'""".format(i)
								if len(STRATEGIES_FOR_ANALYSIS) > 0:
									analytics_result_dir = """python analysis/non_iid.py --dataset='{}' --model='{}' --round={} --client={} --aggregation_method='{}' --poc={} --new_clients={} --new_clients_train={} --non-iid={} --comment='{}' --epochs={} --decay={} --fraction_fit={} --experiment={} {}""".format(dataset, model, ROUNDS, clients, algorithm, poc, new_client, new_client_train, True, comment, epochs, decay, fraction_fit, experiment, strategies_arg)
									print("=====================================\nExecutando analytics... \n", analytics_result_dir,
										  "\n=====================================")
									subprocess.Popen(analytics_result_dir, shell=True).wait()

								# subprocess.Popen(['rm', '-fr', '/tmp/ray/']).wait()
								# subprocess.Popen(['rm', '/tmp/*.py']).wait()
	except Exception as e:
		print("Error on execution")
		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

def main():
	parser = OptionParser()

	# parser.add_option("-a", "--algorithm", dest="algorithm", default='None',   help="Algorithm used for selecting clients", metavar="STR")
	parser.add_option("",   "--experiment_id",   dest="experiment_id",   default=1,  help="", metavar="INT")
	parser.add_option("", "--type", dest="type", default="torch", help="", metavar="STR")

	(opt, args) = parser.parse_args()

	experiment = EXPERIMENTS[int(opt.experiment_id)]
	execute_experiment(opt.experiment_id, experiment['algorithm'], experiment['new_client'], experiment['new_client_train'], experiment['comment'], opt.type)
	remove_lines("""execution_log/experiment_{}.txt""".format(opt.experiment_id))

		
		





if __name__ == '__main__':
	main()
