from client.client_torch import FedAvgClientTorch
from ..fedpredict_core import fedpredict_dynamic_client
from torch.nn.parameter import Parameter
import torch
from pathlib import Path
from dataset_utils_torch import ManageDatasets
import os
import sys
import pandas as pd
import numpy as np
import ast
from utils.compression_methods.sparsification import calculate_bytes, sparse_bytes, sparse_matrix

import warnings
warnings.simplefilter("ignore")

import logging
# logging.getLogger("torch").setLevel(logging.ERROR)

class CDAFedAvgClientTorch(FedAvgClientTorch):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 args,
				 epochs=1,
				 model_name         = 'DNN',
				 client_selection   = False,
				 strategy_name      ='CDA-FedAvg',
				 aggregation_method = 'None',
				 dataset            = '',
				 perc_of_clients    = 0,
				 decay              = 0,
				 fraction_fit		= 0,
				 non_iid            = False,
				 m_combining_layers	= 1,
				 new_clients			= False,
				 new_clients_train	= False
				 ):

		super().__init__(cid=cid,
						 n_clients=n_clients,
						 n_classes=n_classes,
						 args=args,
						 epochs=epochs,
						 model_name=model_name,
						 client_selection=client_selection,
						 strategy_name=strategy_name,
						 aggregation_method=aggregation_method,
						 dataset=dataset,
						 perc_of_clients=perc_of_clients,
						 decay=decay,
						 fraction_fit=fraction_fit,
						 non_iid=non_iid,
						 new_clients=new_clients,
						 new_clients_train=new_clients_train)

		self.client_information_filename = """{}_saved_weights/{}/{}/{}.csv""".format(strategy_name.lower(),
																					  self.model_name, self.cid,
																					  self.cid)
		self.client_information_file = self.read_client_file()



	def load_data(self, dataset_name, n_clients, batch_size=32, server_round=None):
		try:
			pattern = self.cid
			if self.clients_pattern is not None:
				if server_round is not None:
					current_pattern = self.clients_pattern.query("""Round == {} and Cid == {}""".format(server_round, self.cid))['Pattern'].tolist()
					if len(current_pattern) != 1:
						raise ValueError("""Pattern not found for client {}. The pattern may not exist or is duplicated""".format(pattern))
					pattern = current_pattern[0]
			if dataset_name in ['MNIST', 'CIFAR10', 'CIFAR100', 'EMNIST', 'GTSRB', 'State Farm']:
				trainLoader, testLoader, traindataset, testdataset = ManageDatasets(pattern, self.model_name).select_dataset(
					dataset_name, n_clients, self.class_per_client, self.alpha, self.non_iid, batch_size)
				self.input_shape = (3,64,64)
			else:
				print("gerar")
				trainLoader, testLoader, traindataset, testdataset = ManageDatasets(pattern, self.model_name).select_dataset(
					dataset_name, n_clients, self.class_per_client, self.alpha, self.non_iid, batch_size)
				self.input_shape = (32, 0)
				# exit()

			if self.clients_pattern is not None:
				past_pattern = \
					self.clients_pattern.query("""Round < {} and Cid == {}""".format(server_round, self.cid))[
						'Pattern'].tolist()
				if len(past_pattern) >= 1:
					traindataset = self.previous_balanced_dataset(traindataset, past_pattern, batch_size, dataset_name, n_clients)

			return trainLoader, testLoader, traindataset, testdataset
		except Exception as e:
			print("load data")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def previous_balanced_dataset(self, current_traindataset, past_pattern, batch_size, dataset_name, n_clients):

		try:

			L = 1400
			M = self.num_classes
			samples_per_class = int(L/(2*M))

			for pattern in past_pattern:
				trainLoader, testLoader, traindataset, testdataset = ManageDatasets(pattern,
																					self.model_name).select_dataset(
					dataset_name, n_clients, self.class_per_client, self.alpha, self.non_iid, batch_size)

				trainLoader = None
				testLoader = None
				testdataset = None

				data_target = {i: [] for i in range(self.num_classes)}

				for class_ in data_target:

					current_size = len(data_target[class_])
					missing_data = samples_per_class - current_size
					if missing_data > 0:
						targets = np.array(traindataset.targets)
						if dataset_name == 'GTSRB':
							data = np.array(traindataset.samples)
						else:
							data = np.array(traindataset.data)
						indices = np.where(targets == class_)[0]
						if len(indices) == 0:
							continue
						# print("ind: ", indices)
						indices = np.random.choice(indices, size=missing_data)
						targets = targets[indices]
						data = data[indices].tolist()
						data_target[class_] += data

			l_old_samples = []
			l_old_targets = []

			for class_ in data_target:

				samples = list(data_target[class_])
				targets = [class_] * len(samples)
				l_old_samples += samples
				l_old_targets += targets

			l_old_samples = np.array(l_old_samples)
			l_old_targets = np.array(l_old_targets)

			if dataset_name == 'GTSRB':
				current_samples = np.array(current_traindataset.samples)
			else:
				current_samples = np.array(current_traindataset.data)
			current_targets = current_traindataset.targets
			print("shapes: ", current_samples.shape, l_old_samples.shape)
			current_samples = np.concatenate((current_samples, l_old_samples), axis=0)
			current_targets = np.concatenate((current_targets, l_old_targets), axis=0)

			current_traindataset.samples = current_samples
			current_traindataset.targets = current_targets

			return current_traindataset


		except Exception as e:
			print("previous balanced dataset")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def save_client_information_fit(self, server_round, acc_of_last_fit):

		try:
			self.classes_proportion, self.imbalance_level = self._calculate_classes_proportion()
			df = pd.read_csv(self.client_information_filename)
			row = df.iloc[0]
			if int(row['first_round']) == -1:
				first_round = -1
			else:
				first_round = int(row['first_round'])

			pd.DataFrame(
				{'current_round': [server_round], 'classes_distribution': [str(self.classes_proportion)],
				 'round_of_last_fit': [server_round],
				 'round_of_last_evaluate': [-1], 'acc_of_last_fit': [acc_of_last_fit], 'first_round': [first_round],
				 'acc_of_last_evaluate': [0]}).to_csv(self.client_information_filename,
													  index=False)

		except Exception as e:
			print("save client information fit")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def _calculate_contexts_similarities(self):

		try:
			"""
				It measures the cosine similarity between the last and current class distribution of the local dataset
			"""
			n = len(self.client_information_file['classes_distribution'])
			print("antes ", self.client_information_file['classes_distribution'].tolist()[-1])
			last_proportion = ast.literal_eval(self.client_information_file['classes_distribution'].tolist()[-1])
			last_training = self.client_information_file['round_of_last_fit'].tolist()[-1]

			current_proportion, imbalance_level = self._calculate_classes_proportion()
			fraction_of_classes = sum([1 if i > 0 else 0 for i in current_proportion])/self.num_classes

			if len(last_proportion) != len(current_proportion) or last_training == -1:
				return 1, imbalance_level, fraction_of_classes

			last_proportion = np.array(last_proportion)
			current_proportion = np.array(current_proportion)

			if (last_proportion == current_proportion).all():
				print("igual")

				cosine_similarity = 1

			else:
				print("diferente ")
				dot_product = np.dot(last_proportion, current_proportion)

				norm_vector1 = np.linalg.norm(last_proportion)

				norm_vector2 = np.linalg.norm(current_proportion)

				cosine_similarity = dot_product / (norm_vector1 * norm_vector2)

			return cosine_similarity, imbalance_level, fraction_of_classes

		except Exception as e:
			print("calculate contexts similarities")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def read_client_file(self):

		try:

			df = pd.read_csv(self.client_information_filename)

			return df

		except Exception as e:
			print("read client file")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

