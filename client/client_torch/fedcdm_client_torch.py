import copy
import time
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
from client.quant import quan


import warnings
warnings.simplefilter("ignore")

import logging
# logging.getLogger("torch").setLevel(logging.ERROR)

class FedCDMClientTorch(FedAvgClientTorch):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 args,
				 epochs=1,
				 model_name         = 'DNN',
				 client_selection   = False,
				 strategy_name      ='FedCDM',
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

	def fit(self, parameters, config):
		try:
			selected_clients = []
			trained_parameters = []
			selected = 0
			print("Iniciar treinamento")
			if config['selected_clients'] != '':
				selected_clients = [int(cid_selected) for cid_selected in config['selected_clients'].split(' ')]

			start_time = time.process_time()
			server_round = int(config['round'])

			if self.dynamic_data != "no":
				self.trainloader, self.testloader, self.traindataset, self.testdataset = self.load_data(self.dataset,
																									n_clients=self.n_clients, server_round=server_round, train=True)
			original_parameters = copy.deepcopy(parameters)
			if self.cid in selected_clients or self.client_selection == False or int(config['round']) == 1:
				self.set_parameters_to_model_fit(parameters)
				# self.save_parameters_global_model(parameters)
				self.round_of_last_fit = server_round

				selected = 1
				self.model.to(self.device)
				self.model.train()

				max_local_steps = self.local_epochs

				self.classes_proportion, self.imbalance_level = self._calculate_classes_proportion()

				# self.device = 'cuda:0'
				print("Cliente: ", self.cid, " rodada: ", server_round, " Quantidade de camadas: ", len([i for i in self.model.parameters()]), " device: ", self.device)
				predictions = []
				for step in range(max_local_steps):
					start_time = time.process_time()
					train_acc = 0
					train_loss = 0
					train_num = 0
					for i, (x, y) in enumerate(self.trainloader):
						if type(x) == type([]):
							x[0] = x[0].to(self.device)
						else:
							x = x.to(self.device)

						# if self.dataset == 'EMNIST':
						# 	x = x.view(-1, 28 * 28)
						y = np.array(y).astype(int)
						# print("entrada: ", x.shape, y.shape, type(x[0]), type(y[0]), y[0])
						# y = y.to(self.device)
						train_num += y.shape[0]

						self.optimizer.zero_grad()
						output = self.model(x)
						if len(predictions) == 0:
							predictions = output.detach().numpy().tolist()
						else:
							predictions += output.detach().numpy().tolist()
						y = torch.tensor(y)
						loss = self.loss(output, y)
						train_loss += loss.item()
						loss.backward()
						self.optimizer.step()

						train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
						total_time = time.process_time() - start_time
						# print("Duração: ", total_time)
				# print("Completou, cliente: ", self.cid, " rodada: ", server_round)
				trained_parameters = self.get_parameters_of_model()
				self.save_parameters()

			size_list = []
			for i in range(len(parameters)):
				tamanho = parameters[i].nbytes
				# print("Client id: ", self.cid, " camada: ", i, " tamanho: ", tamanho, " shape: ", parameters[i].shape)
				size_list.append(tamanho)
			# print("Tamanho total parametros fit: ", sum(size_list))
			size_of_parameters = sum(size_list)
			# size_of_parameters = sum(
			# 	[sum(map(sys.getsizeof, trained_parameters[i])) for i in range(len(trained_parameters))])
			avg_loss_train = train_loss / train_num
			avg_acc_train = train_acc / train_num
			total_time = time.process_time() - start_time
			# loss, accuracy, test_num = self.model_eval()

			data = [config['round'], self.cid, selected, total_time, size_of_parameters, avg_loss_train, avg_acc_train]

			self.save_client_information_fit(server_round, avg_acc_train, predictions)

			self._write_output(
				filename=self.train_client_filename,
				data=data)

			row = self.clients_pattern.query("""Round == {} and Cid == {}""".format(server_round, self.cid))[
				'Pattern'].tolist()
			if len(row) != 1:
				raise ValueError(
					"""Pattern not found for client {}. The pattern may not exist or is duplicated""".format(self.cid))
			pattern = int(row[0])

			fit_response = {
				'cid': self.cid,
				'local_classes': self.classes_proportion,
				'pattern': pattern
			}

			if self.use_gradient and server_round > 1:
				trained_parameters = [original - trained for trained, original in zip(trained_parameters, original_parameters)]
				# trained_parameters = parameters_quantization_write(trained_parameters, 8)
				# print("quantizou: ", trained_parameters[0])
			return trained_parameters, train_num, fit_response
		except Exception as e:
			print("fit")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

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
			Q = np.array([np.max(softmax(i).tolist()) for i in predictions]).flatten().tolist()
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

			data, labels = self.get_target_and_samples_from_dataset(self.traindataset, self.dataset)
			training_data, training_labels = self.get_target_and_samples_from_dataset(self.traindataset, self.dataset)

			drift_position = quan(data, labels, training_data, training_labels, self.num_classes)

			if drift_position == -1:
				return False

			else:
				return True


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

