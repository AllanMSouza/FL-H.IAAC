from client.client_torch import FedAvgClientTorch
from ..fedpredict_core import fedpredict_dynamic_client
from torch.nn.parameter import Parameter
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from dataset_utils_torch import ManageDatasets
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ast
from scipy.special import softmax
from utils.compression_methods.sparsification import calculate_bytes, sparse_bytes, sparse_matrix
from client.cda_fedavg_concept_drift import cda_fedavg_drift_detection

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

		self.client_information_train_filename = """{}_saved_weights/{}/{}/{}_train.csv""".format(strategy_name.lower(),
																							self.model_name, self.cid,
																							self.cid)
		self.client_information_val_filename = """{}_saved_weights/{}/{}/{}_val.csv""".format(strategy_name.lower(),
																								  self.model_name,
																								  self.cid,
																								  self.cid)
		self.client_information_train_file, self.client_information_val_file = self.read_client_file()



	def load_data(self, dataset_name, n_clients, batch_size=32, server_round=None, train=None):
		try:
			pattern = self.cid
			if self.clients_pattern is not None:
				if server_round is not None:
					# It does the concept drift
					current_pattern = self.clients_pattern.query("""Round == {} and Cid == {}""".format(server_round, self.cid))['Pattern'].tolist()
					if len(current_pattern) != 1:
						raise ValueError("""Pattern not found for client {}. The pattern may not exist or is duplicated""".format(pattern))
					pattern = current_pattern[0]
			# print("cid: ", self.cid, " padrao: ", pattern)
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

			print("dd 1: ", self.drift_detected, server_round)
			if self.drift_detected:
				# After drift detection, it recovers the past data and concatenate it with the new one
				past_patterns = \
					self.clients_pattern.query("""Round < {} and Cid == {} and Pattern != {}""".format(server_round, self.cid, pattern))[
						'Pattern'].unique().tolist()
				print("dd 2", past_patterns, server_round)
				if len(past_patterns) >= 1:
					print("""Leu dataset maior do cliente {} na rodada {}""".format(self.cid, server_round))
					traindataset = self.previous_balanced_dataset(traindataset, past_patterns, batch_size, dataset_name, n_clients, pattern)

			return trainLoader, testLoader, traindataset, testdataset
		except Exception as e:
			print("load data")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def _min_max_scaler(self, X):

		try:

			X = np.array(X)
			print("Tamanho de X: ", X.shape)
			min_ = np.min(X)
			max_ = np.max(X)
			X_std = (X - min_) / (max_ - min_)
			X_scaled = softmax(X_std)
			print(" 1: ", X_scaled)
			print(" 2: ", softmax(X))
		except Exception as e:
			print("min max scaler")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

		return X_scaled

	def get_target_and_samples_from_dataset(self, traindataset, dataset_name):

		try:

			if dataset_name in ['WISDM-WATCH']:
				data = []
				targets = []
				for sample in traindataset:
					# print("amostra: ", sample)
					data.append(sample[0].numpy())
					targets.append(int(sample[1]))
				data = np.array(data)
				print("dada: ", type(data), len(data), len(targets))
				targets = np.array(targets)
			else:
				targets = np.array(traindataset.targets)
				if dataset_name == 'GTSRB':
					data = np.array(traindataset.samples)
				else:
					data = np.array(traindataset.data)

			return data, targets

		except Exception as e:
			print("get target and samples from dataset")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def set_dataset(self, dataset, dataset_name, x, y):

		try:

			if dataset_name in ['WISDM-WATCH']:

				return torch.utils.data.TensorDataset(torch.from_numpy(x).to(dtype=torch.float32), torch.from_numpy(y))

			else:

				dataset.samples = x
				dataset.targets = y

				return dataset

		except Exception as e:
			print("set dataset")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def previous_balanced_dataset(self, current_traindataset, past_patterns, batch_size, dataset_name, n_clients, current_pattern):

		try:

			L = 3000
			M = self.num_classes
			samples_per_class = int(L/(M))

			print("balanced 1")

			data_target = {i: [] for i in range(self.num_classes)}
			for pattern in past_patterns:
				trainLoader, testLoader, traindataset, testdataset = ManageDatasets(pattern,
																					self.model_name).select_dataset(
					dataset_name, n_clients, self.class_per_client, self.alpha, self.non_iid, batch_size)

				trainLoader = None
				testLoader = None
				testdataset = None

				print("balanced 2")

				for class_ in data_target:

					current_size = len(data_target[class_])
					missing_data = samples_per_class - current_size
					if missing_data > 0:
						data, targets = self.get_target_and_samples_from_dataset(traindataset, dataset_name)
						indices = np.where(targets == class_)[0]
						if len(indices) == 0:
							continue
						# print("ind: ", indices)
						indices = np.random.choice(indices, size=missing_data)
						targets = targets[indices]
						data = data[indices].tolist()
						data_target[class_] += data

						print("balanced 3")

			l_old_samples = []
			l_old_targets = []

			for class_ in data_target:

				samples = list(data_target[class_])
				print("""quantidade classe {} e {} cliente {} leu padroes {} padrao atual {}""".format(class_, len(samples), self.cid, past_patterns, current_pattern))
				targets = [class_] * len(samples)
				l_old_samples += samples
				l_old_targets += targets

			l_old_samples = np.array(l_old_samples)
			l_old_targets = np.array(l_old_targets)

			# if dataset_name == 'GTSRB':
			# 	current_samples = np.array(current_traindataset.samples)
			# else:
			# 	current_samples = np.array(current_traindataset.data)
			# current_targets = current_traindataset.targets
			current_samples, current_targets = self.get_target_and_samples_from_dataset(current_traindataset, dataset_name)
			print("shapes: ", current_samples.shape, current_targets.shape, l_old_samples.shape, l_old_targets.shape)
			print("""antes juntar unique {} cliente {}""".format(np.unique(current_targets, return_counts=True), self.cid))
			current_samples = np.concatenate((current_samples, l_old_samples), axis=0)
			current_targets = np.concatenate((current_targets, l_old_targets), axis=0)

			print("""juntou unique {} cliente {}""".format(np.unique(current_targets, return_counts=True), self.cid))

			print("s t: ", current_samples.shape, current_targets.shape)

			current_traindataset = self.set_dataset(current_traindataset, dataset_name, current_samples, current_targets)

			return current_traindataset


		except Exception as e:
			print("previous balanced dataset")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def save_client_information_fit(self, server_round, acc_of_last_fit, predictions):

		try:
			scaler = MinMaxScaler()
			Q = np.array([np.max(i) for i in predictions]).flatten().tolist()
			self.classes_proportion, self.imbalance_level = self._calculate_classes_proportion()
			drift_detected = self._drift_detection(Q, server_round)
			print("rodada: ", server_round, " drift detected: ", drift_detected, " cid: ", self.cid)
			df = pd.read_csv(self.client_information_train_filename)
			already_detected_drift = True if df['drift_detected'].tolist()[0] == "True" else False
			if drift_detected or already_detected_drift:
				drift_detected = True
			row = df.iloc[0]
			if int(row['first_round']) == -1:
				first_round = -1
			else:
				first_round = int(row['first_round'])

			pd.DataFrame(
				{'current_round': [server_round], 'classes_distribution': [str(self.classes_proportion)],
				 'round_of_last_fit': [server_round], 'drift_detected': [drift_detected], 'Q': [str(Q)],
				 'acc_of_last_fit': [acc_of_last_fit], 'first_round': [first_round]}).to_csv(self.client_information_train_filename,
													  index=False)

		except Exception as e:
			print("save client information fit")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def _drift_detection(self, Q, server_round):

		try:
			"""
				
			"""
			row = self.client_information_train_file['Q'].tolist()
			if len(row) > 0:
				Q_old = ast.literal_eval(self.client_information_train_file['Q'].tolist()[-1])
				print("antigo: ", Q_old[:5], len(Q_old))
				print("novo: ", Q[:5], len(Q))
				Q = Q + Q_old

			lamda = 0.001 # descer aumenta a detecção
			delta = 300
			n_max = len(Q)
			drif_detection = cda_fedavg_drift_detection(Q, lamda, delta, n_max, server_round)

			return drif_detection


		except Exception as e:
			print("calculate contexts similarities")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def read_client_file(self):

		try:

			df_train = pd.read_csv(self.client_information_train_filename)
			df_val = pd.read_csv(self.client_information_val_filename)

			self.drift_detected = True if df_train['drift_detected'].tolist()[0] == True else False

			return df_train, df_val

		except Exception as e:
			print("read client file")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
			return pd.DataFrame({'current_round': [-1], 'classes_distribution': ["[]"], 'round_of_last_fit': [-1],
						  'drift_detected': ['False'], 'Q': [[]], 'acc_of_last_fit': [0], 'first_round': [-1]}), pd.DataFrame({'current_round': [-1], 'classes_distribution': ["[]"],
						  'drift_detected': ['False'], 'round_of_last_evaluate': [-1],
						  'first_round': [-1],
						  'acc_of_last_evaluate': [0]})

