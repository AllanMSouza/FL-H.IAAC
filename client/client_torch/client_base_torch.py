import flwr as fl
import copy
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import time
import sys

from dataset_utils_torch import ManageDatasets
from models.torch import DNN, Logistic, CNN, MobileNet, resnet20, CNN_EMNIST, MobileNetV2, CNN_X, CNN_5, CNN_2, CNN_3, CNN_3_GTSRB, MobileNet_V3, GRU
import csv
import torch.nn as nn
import warnings
import pandas as pd
warnings.simplefilter("ignore")

# logging.getLogger("torch").setLevel(logging.ERROR)
from torch.nn.parameter import Parameter
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def get_size(parameter):
	try:
		# #print("recebeu: ", parameter.shape, parameter.ndim)
		# if type(parameter) == np.float32:
		# 	#print("caso 1: ", map(sys.getsizeof, parameter))
		# 	return map(sys.getsizeof, parameter)
		# if parameter.ndim <= 2:
		# 	#print("Caso 2: ", sum(map(sys.getsizeof, parameter)))
		# 	return sum(map(sys.getsizeof, parameter))
		#
		# else:
		# 	tamanho = 0
		# 	#print("Caso 3")
		# 	for i in range(len(parameter)):
		# 		tamanho += get_size(parameter[i])
		#
		# 	return tamanho
		return parameter.nbytes
	except Exception as e:
		print("get_size")
		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
class ClientBaseTorch(fl.client.NumPyClient):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 args,
				 epochs				= 1,
				 model_name         = 'None',
				 client_selection   = False,
				 solution_name      = 'None',
				 aggregation_method = 'None',
				 dataset            = '',
				 perc_of_clients    = 0,
				 decay              = 0,
				 fraction_fit		= 0,
				 non_iid            = False,
				 new_clients = False,
				 new_clients_train	= False,

				 ):
		try:
			self.cid          = int(cid)
			self.n_clients    = n_clients
			self.model_name   = model_name
			self.local_epochs = epochs
			self.non_iid      = non_iid
			self.n_rounds	  = int(args.rounds)

			self.num_classes = n_classes
			self.class_per_client = int(args.class_per_client)
			self.train_perc = float(args.train_perc)
			self.alpha = float(args.alpha)
			self.comment = args.comment
			self.compression = args.compression_method
			self.use_gradient = bool(args.use_gradient)

			self.model        = None
			self.x_train      = None
			self.x_test       = None
			self.y_train      = None
			self.y_test       = None

			#logs
			self.strategy_name = solution_name
			# "solution_name" is will be further modified
			self.solution_name      = solution_name
			self.aggregation_method = aggregation_method
			self.dataset            = dataset

			self.client_selection = client_selection
			self.perc_of_clients  = perc_of_clients
			self.decay            = decay
			self.fraction_fit	  = fraction_fit

			self.loss = nn.CrossEntropyLoss()
			if model_name == "CNN_10":
				self.learning_rate = 0.005
			self.new_clients = new_clients
			self.new_clients_train = new_clients_train
			# self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			# self.device = torch.device("cpu")
			self.type = 'torch'
			self.dynamic_data = args.dynamic_data
			self.rounds_to_change_pattern = [int(0.7 * self.n_rounds)]
			self.dynamic_data_filename = {'no': None, 'synthetic': """/home/claudio/Documentos/pycharm_projects/FL-H.IAAC/dynamic_experiments_config/dynamic_data_synthetic_config_{}_clients_{}_rounds_change_pattern_{}_total_rounds.csv""".format(n_clients, self.rounds_to_change_pattern, self.n_rounds),
										  'synthetic_global': """/home/claudio/Documentos/pycharm_projects/FL-H.IAAC/dynamic_experiments_config/dynamic_data_synthetic_config_{}_clients_{}_rounds_change_pattern_{}_total_rounds_global_concept_drift.csv""".format(n_clients, self.rounds_to_change_pattern, self.n_rounds)}[self.dynamic_data]
			print("nome arquivo: ", self.dynamic_data_filename)
			if self.dynamic_data_filename is not None:
				self.clients_pattern = pd.read_csv(self.dynamic_data_filename)
			else:
				self.clients_pattern = None

			#params
			if self.aggregation_method == 'POC':
				self.solution_name = f"{solution_name}-{aggregation_method}-{self.perc_of_clients}"

			elif self.aggregation_method == 'FedLTA':
				self.solution_name = f"{solution_name}-{aggregation_method}-{self.decay}"

			elif self.aggregation_method == 'None':
				self.solution_name = f"{solution_name}-{aggregation_method}-{self.fraction_fit}"

			self.base = self._create_base_directory(self.type, self.solution_name, new_clients, new_clients_train, self.dynamic_data, n_clients, model_name, dataset, str(args.class_per_client), str(args.alpha), str(args.rounds), str(args.local_epochs), str(args.comment), str(args.compression_method), args)
			self.evaluate_client_filename = f"{self.base}/evaluate_client.csv"
			self.train_client_filename = f"{self.base}/train_client.csv"
			self.predictions_client_filename = f"{self.base}/predictions_client.csv"

			if self.dynamic_data == "no":
				self.trainloader, self.testloader, self.traindataset, self.testdataset = self.load_data(self.dataset, n_clients=self.n_clients)
			print("leu dados")
			self.model                                           = self.create_model().to(self.device)
			if self.dataset in ['EMNIST', 'CIFAR10', 'GTSRB']:
				self.learning_rate = 0.01
				# self.optimizer = torch.optim.Adam(self.model.parameters(),
				# 								  lr=self.learning_rate)
				self.optimizer = torch.optim.SGD(
					self.model.parameters(), lr=self.learning_rate)
			elif self.dataset == 'State Farm':
				# self.learning_rate = 0.01
				# self.optimizer = torch.optim.Adam(self.model.parameters(),
				# 								  lr=self.learning_rate)
				self.learning_rate = 0.01
				self.optimizer = torch.optim.SGD(
					self.model.parameters(), lr=self.learning_rate, momentum=0.9)
			elif self.dataset in ['ExtraSensory', 'WISDM-WATCH', 'WISDM-P', 'Cologne']:
				self.learning_rate = 0.001
				# self.loss = nn.MSELoss()
				self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
			# self.device = 'cpu'
			# self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) if self.model_name == "Mobilenet" else torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
			# self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

		except Exception as e:
			print("init client")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def _create_base_directory(self, type, strategy_name, new_clients, new_clients_train, dynamic_data, n_clients, model_name, dataset, class_per_client, alpha, n_rounds, local_epochs, comment, compression, args):

		return f"logs/{type}/{strategy_name}/new_clients_{new_clients}_train_{new_clients_train}_dynamic_data_{dynamic_data}/{n_clients}/{model_name}/{dataset}/classes_per_client_{class_per_client}/alpha_{alpha}/{n_rounds}_rounds/{local_epochs}_local_epochs/{comment}_comment/{str(compression)}_compression"

	def shuffle(self, _points, _labels):
		idxs = np.arange(len(_labels))
		np.random.seed(self.cid)
		np.random.shuffle(idxs)
		_points = _points[idxs]
		_labels = _labels[idxs]

		return _points, _labels

	def load_data(self, dataset_name, n_clients, batch_size=32, server_round=None, train=None):
		try:
			pattern = self.cid
			if server_round is not None and self.clients_pattern is not None:
				row = self.clients_pattern.query("""Round == {} and Cid == {}""".format(server_round, self.cid))['Pattern'].tolist()
				if len(row) != 1:
					raise ValueError("""Pattern not found for client {}. The pattern may not exist or is duplicated""".format(pattern))
				pattern = int(row[0])

			limit_classes = False
			alpha = self.alpha
			alphas = np.array([0.1, 1.0])
			if self.dynamic_data == "synthetic_global":
				if server_round < int(self.n_rounds * 0.7):
					limit_classes = True
					alpha = self.alpha
				else:
					limit_classes = False

					alpha = alphas[alphas != self.alpha][0]
			else:
				alpha = self.alpha
			limit_classes = server_round
			trainLoader, testLoader, traindataset, testdataset = ManageDatasets(self.cid, self.model_name, pattern, limit_classes).select_dataset(
				dataset_name, n_clients, self.class_per_client, alpha, self.non_iid, batch_size)
			self.input_shape = (3,64,64)

			if server_round is not None:
				if server_round in self.rounds_to_change_pattern:
					past_patterns = \
						self.clients_pattern.query(
							"""Round < {} and Cid == {} and Pattern != {}""".format(server_round, self.cid, pattern))[
							'Pattern'].unique().tolist()
					print("patter: ", past_patterns)
					trainLoader, traindataset = self.concatenate_dataset(traindataset, past_patterns, batch_size,
																			   dataset_name, n_clients, pattern, alpha)
					print("""alpha {} {} load dados cliente {} padrao {} rodada {} treino {}""".format(alpha, self.dynamic_data, self.cid, pattern, server_round, train))

			return trainLoader, testLoader, traindataset, testdataset
		except Exception as e:
			print("load data")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def get_target_and_samples_from_dataset(self, traindataset, dataset_name):

		try:

			if dataset_name in ['WISDM-WATCH', 'WISDM-P', 'Cologne']:
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

	def concatenate_dataset(self, current_traindataset, past_patterns, batch_size, dataset_name, n_clients, current_pattern, alpha):

		try:

			data_list = np.array([])
			target_list = np.array([])

			if len(past_patterns) > 0:
				for pattern in past_patterns:
					trainLoader, testLoader, traindataset, testdataset = ManageDatasets(self.cid,
																						self.model_name, pattern, False).select_dataset(
						dataset_name, n_clients, self.class_per_client, alpha, self.non_iid, batch_size)

					trainLoader = None
					testLoader = None
					testdataset = None


					data, targets = self.get_target_and_samples_from_dataset(traindataset, dataset_name)

					if len(data_list) == 0:

						data_list = data
						target_list = targets

					else:

						data_list = np.concatenate((data_list, data))
						target_list = np.concatenate((target_list, targets))

			else:
				trainLoader, testLoader, traindataset, testdataset = ManageDatasets(self.cid,
																					self.model_name, self.cid,
																					False).select_dataset(
					dataset_name, n_clients, self.class_per_client, alpha, self.non_iid, batch_size)

				data_list, target_list = self.get_target_and_samples_from_dataset(traindataset, dataset_name)

				data_list, target_list = self.shuffle(data_list, target_list)

			print("dt: ", data_list.shape)

			current_samples, current_targets = self.get_target_and_samples_from_dataset(current_traindataset,
																						dataset_name)
			print("shapes: ", current_samples.shape, current_targets.shape, data_list.shape, target_list.shape)
			print("""antes juntar unique {} cliente {}""".format(np.unique(current_targets, return_counts=True),
																 self.cid))

			current_samples, current_targets = self.shuffle(current_samples, current_targets)

			data_list = np.concatenate((data_list, current_samples), axis=0)
			target_list = np.concatenate((target_list, current_targets), axis=0)

			current_traindataset = self.set_dataset(current_traindataset, dataset_name, data_list, target_list)

			def seed_worker(worker_id):
				np.random.seed(self.cid)
				random.seed(self.cid)

			g = torch.Generator()
			g.manual_seed(self.cid)

			trainLoader = DataLoader(current_traindataset, batch_size, shuffle=False, worker_init_fn=seed_worker,
									 generator=g)

			return trainLoader, current_traindataset


		except Exception as e:
			print("concatenate dataset")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def create_model(self):

		try:
			# print("tamanho: ", self.input_shape, " dispositivo: ", self.device)
			# input_shape = self.input_shape
			model = None
			if self.dataset in ['MNIST', 'EMNIST']:
				input_shape = 1
			elif self.dataset in ['CIFAR10', 'GTSRB', 'State Farm']:
				input_shape = 3
			elif self.dataset in ['MotionSense', 'UCIHAR']:
				input_shape = self.input_shape[1]
			if self.model_name == 'Logist Regression':
				model =  Logistic(input_shape=input_shape, num_classes=self.num_classes)
			elif self.model_name == 'DNN':
				model =  DNN(input_shape=input_shape, num_classes=self.num_classes)
			elif self.model_name == 'CNN_2' and self.dataset in ['EMNIST', 'MNIST', 'CIFAR10', 'GTSRB']:
				if self.dataset == 'CIFAR10':
					mid_dim = 64
				elif self.dataset == 'GTSRB':
					mid_dim = 64
				elif self.dataset == 'State Farm':
					mid_dim = 64
				else:
					mid_dim = 36
				return  CNN_2(input_shape=input_shape, mid_dim=mid_dim, num_classes=self.num_classes)
			elif self.model_name in ['CNN_3'] and self.dataset in ['EMNIST', 'MNIST', 'CIFAR10', 'GTSRB', 'State Farm']:
				if self.dataset in ['CIFAR10']:
					mid_dim = 16
				elif self.dataset == 'GTSRB':
					mid_dim = 16
				elif self.dataset == 'State Farm':
					mid_dim = 16
				# return CNN_3_GTSRB(input_shape=input_shape, mid_dim=mid_dim, num_classes=self.num_classes)
				else:
					mid_dim = 4
				return  CNN_3(input_shape=input_shape, mid_dim=mid_dim, num_classes=self.num_classes)
			elif self.model_name in ['MobileNet'] and self.dataset in ['EMNIST', 'MNIST', 'CIFAR10', 'GTSRB', 'State Farm']:
				if self.dataset in ['CIFAR10']:
					mid_dim = 16
				elif self.dataset == 'GTSRB':
					mid_dim = 16
				elif self.dataset == 'State Farm':
					mid_dim = 576
					# return CNN_3_GTSRB(input_shape=input_shape, mid_dim=mid_dim, num_classes=self.num_classes)
				else:
					mid_dim = 4
				return  MobileNet_V3(input_shape=input_shape, mid_dim=mid_dim, num_classes=self.num_classes)
			elif self.model_name == 'CNN_1'  and self.dataset in ['EMNIST', 'MNIST', 'CIFAR10', 'GTSRB']:
				if self.dataset in ['EMNIST', 'MNIST']:
					mid_dim = 256
				else:
					mid_dim = 400
				model =  CNN(input_shape=input_shape, num_classes=self.num_classes, mid_dim=mid_dim)
			elif self.model_name == 'GRU' and self.dataset in ['ExtraSensory']:
				# if self.dataset in ['EMNIST', 'MNIST']:
				# 	mid_dim = 256
				# else:
				# 	mid_dim = 400
				model =  GRU(input_shape=10, num_classes=self.num_classes)
			elif self.model_name == 'GRU' and self.dataset in ['WISDM-WATCH', 'WISDM-P', 'Cologne']:
				# if self.dataset in ['EMNIST', 'MNIST']:
				# 	mid_dim = 256
				# else:
				# 	mid_dim = 400
				input_shape = {'WISDM-WATCH': 6, 'WISDM-P': 6, 'Cologne': 11}
				model =  GRU(input_shape=input_shape, num_classes=self.num_classes)

			if model is not None:
				model.to(self.device)
				return model
			else:
				raise Exception("Wrong model name")
		except Exception as e:
			print("create model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def set_dataset(self, dataset, dataset_name, x, y):

		try:

			if dataset_name in ['WISDM-WATCH', 'WISDM-P', 'Cologne']:

				return torch.utils.data.TensorDataset(torch.from_numpy(x).to(dtype=torch.float32), torch.from_numpy(y))

			else:

				dataset.samples = x
				dataset.targets = y

				return dataset

		except Exception as e:
			print("set dataset")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def save_client_information_fit(self, server_round, acc_of_last_fit, predictions):
		pass

	def save_client_information_evaluate(self, server_round, accuracy, predictions):
		pass

	def save_parameters(self):
		pass

	def get_parameters(self, config):
		try:
			parameters = [i.detach().cpu().numpy() for i in self.model.parameters()]
			print("tamanho parametros: ", [i.shape for i in self.model.parameters()])
			return parameters
		except Exception as e:
			print("get parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	# It does the same of "get_parameters", but using "get_parameters" in outside of the core of Flower is causing errors
	def get_parameters_of_model(self):
		try:
			parameters = [i.detach().cpu().numpy() for i in self.model.parameters()]
			return parameters
		except Exception as e:
			print("get parameters of model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def clone_model(self, model, target):
		try:
			for param, target_param in zip(model.parameters(), target.parameters()):
				target_param.data = param.data.clone()
			# target_param.grad = param.grad.clone()
		except Exception as e:
			print("clone model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def update_parameters(self, model, new_params):
		for param, new_param in zip(model.parameters(), new_params):
			param.data = new_param.data.clone()

	def set_parameters_to_model(self, parameters, config={}):
		try:
			parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
			for new_param, old_param in zip(parameters, self.model.parameters()):
				old_param.data = new_param.data.clone()
		except Exception as e:
			print("set parameters to model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def set_parameters_to_model_fit(self, parameters):
		try:
			self.set_parameters_to_model(parameters)
		except Exception as e:
			print("set parameters to model train")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def set_parameters_to_model_evaluate(self, parameters, config={}):
		try:
			self.set_parameters_to_model(parameters, config)
		except Exception as e:
			print("set parameters to model")
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
				random.seed(0)
				np.random.seed(0)
				torch.manual_seed(0)
				self.model.to(self.device)
				self.model.train()
				random.seed(0)
				np.random.seed(0)
				torch.manual_seed(0)

				max_local_steps = self.local_epochs

				# self.classes_proportion, self.imbalance_level = self._calculate_classes_proportion()

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
				tamanho = get_size(parameters[i])
				# print("Client id: ", self.cid, " camada: ", i, " tamanho: ", tamanho, " shape: ", parameters[i].shape)
				size_list.append(tamanho)
			# print("Tamanho total parametros fit: ", sum(size_list))
			size_of_parameters = sum(size_list)
			#pattern size_of_parameters = sum(
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

			pattern = self.cid
			if self.dynamic_data != "no":
				row = self.clients_pattern.query("""Round == {} and Cid == {}""".format(server_round, self.cid))[
					'Pattern'].tolist()
				if len(row) != 1:
					raise ValueError(
						"""Pattern not found for client {}. The pattern may not exist or is duplicated""".format(self.cid))
				pattern = int(row[0])

			fit_response = {
				'cid': self.cid,
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

	def model_eval(self):
		try:
			self.model.to(self.device)
			self.model.eval()

			test_acc = 0
			test_loss = 0
			test_num = 0

			predictions = np.array([])
			outputs = np.array([])
			labels = np.array([])

			with torch.no_grad():
				for x, y in self.testloader:
					if type(x) == type([]):
						x[0] = x[0].to(self.device)
					else:
						x = x.to(self.device)
					# if self.dataset == 'EMNIST':
					# 	x = x.view(-1, 28 * 28)
					if type(y) == tuple:
						y = torch.from_numpy(np.array(y).astype(int))
					y = torch.from_numpy(np.array(y).astype(int))
					self.optimizer.zero_grad()
					y = y.to(self.device)
					y = torch.tensor(y)
					output = self.model(x)
					loss = self.loss(output, y)
					test_loss += loss.item() * y.shape[0]
					prediction = torch.argmax(output, dim=1)
					predictions = np.append(predictions, prediction.cpu())
					outputs = np.append(outputs, output.cpu())
					labels = np.append(labels, y.cpu())
					test_acc += (torch.sum(prediction == y)).item()
					test_num += y.shape[0]

			loss = test_loss / test_num
			accuracy = test_acc / test_num

			return loss, accuracy, test_num, predictions, outputs, labels
		except Exception as e:
			print("model_eval")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def model_eval_rnn(self):
		try:
			self.model.to(self.device)
			self.model.eval()

			test_acc = 0
			test_loss = 0
			test_num = 0

			predictions = np.array([])
			labels = np.array([])

			outputs = []
			targets = []
			start_time = time.clock()
			for i in self.testdataset.keys():
				inp = torch.from_numpy(np.array(self.testdataset[i]))
				labs = torch.from_numpy(np.array(self.testloader[i]))
				h = self.model.init_hidden(inp.shape[0])
				out, h = self.model(inp.to(self.device).float(), h)
				outputs.append(self.label_scalers[i].inverse_transform(out.cpu().detach().numpy()).reshape(-1))
				targets.append(self.label_scalers[i].inverse_transform(labs.numpy()).reshape(-1))
			print("Evaluation Time: {}".format(str(time.clock() - start_time)))
			sMAPE = 0
			for i in range(len(outputs)):
				sMAPE += np.mean(abs(outputs[i] - targets[i]) / (targets[i] + outputs[i]) / 2) / len(outputs)
			print("sMAPE: {}%".format(sMAPE * 100))
			return outputs, targets, sMAPE

			# return loss, accuracy, test_num, predictions, labels
		except Exception as e:
			print("model_eval_rnn")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def evaluate(self, parameters, config):
		try:
			server_round = int(config['round'])
			if self.dynamic_data != "no":
				self.trainloader, self.testloader, self.traindataset, self.testdataset = self.load_data(self.dataset,
																								n_clients=self.n_clients,
																								server_round=server_round,
																									train=False)
			self.classes_proportion, self.imbalance_level = self._calculate_classes_proportion()
			n_rounds = int(config['n_rounds'])
			nt = int(config['nt'])
			config['cid'] = self.cid
			print("p recebidos: ", len(parameters), " round: ", server_round, " nt: ", nt, " cid: ", self.cid)
			self.set_parameters_to_model_evaluate(parameters, config)
			size_of_parameters = self.calculate_bytes(parameters)
			# size_of_parameters = sum([sum(map(sys.getsizeof, parameters[i])) for i in range(len(parameters))])
			size_of_config = self._get_size_of_dict(config)
			self.server_round = server_round
			if self.model_name in ['GRU']:
				loss, accuracy, test_num, predictions, output, labels = self.model_eval()
			else:
				loss, accuracy, test_num, predictions, output, labels = self.model_eval()
			data = [config['round'], self.cid, size_of_parameters, size_of_config, loss, accuracy, nt]
			self._write_output(filename=self.evaluate_client_filename,
							   data=data)

			if server_round == n_rounds:
				data = [[self.cid, server_round, int(p), int(l)] for p, l in zip(predictions, labels)]
				self._write_outputs(self.predictions_client_filename, data, 'a')

			self.save_client_information_evaluate(server_round, accuracy, output)

			# if server_round >= int(0.7*self.n_rounds):

			print("""Acurácia teste rodada {} do cliente {}: {}""".format(server_round, self.cid, accuracy))

			if self.strategy_name.lower() == "fedpredict_dynamic":
				pattern = config['pattern']
			else:
				pattern = 0
			evaluation_response = {
				"cid": self.cid,
				"accuracy": float(accuracy),
				"pattern": pattern
			}

			return loss, test_num, evaluation_response
		except Exception as e:
			print("evaluate")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def _calculate_classes_proportion(self, limit=False):

		try:
			# return [1] * self.num_classes, 0
			correction = 3 if self.dataset == 'GTSRB' else 1

			traindataset = self.traindataset
			if self.dataset in ['WISDM-WATCH', 'WISDM-P', 'Cologne']:
				y_train = []
				for i, (x, y) in enumerate(self.trainloader):
					y_train += np.array(y).astype(int).tolist()
			else:
				y_train = list(traindataset.targets)

			if limit:
				lentgh = min(len(self.traindataset), int(len(self.traindataset) * 0.2))
				y_train = y_train[-lentgh:]
			proportion = np.array([0] * self.num_classes)

			unique_classes_list = pd.Series(y_train).unique().tolist()

			for i in y_train:
				proportion[i] += 1

			proportion_ = proportion / np.sum(proportion)

			imbalance_level = 0
			min_samples_per_class = int(len(y_train) / correction / len(unique_classes_list))
			for class_ in proportion:
				if class_ < min_samples_per_class:
					imbalance_level += 1

			imbalance_level = imbalance_level/ len(proportion)

			return list(proportion_), imbalance_level

		except Exception as e:
			print("calculate classes proportion")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def save_parameters_global_model(self, global_model):
		pass

	def _write_output(self, filename, data):

		for i in range(len(data)):
			element = data[i]
			if type(element) == float:
				element = round(element, 6)
				data[i] = element
		with open(filename, 'a') as server_log_file:
			writer = csv.writer(server_log_file)
			writer.writerow(data)

	def _write_outputs(self, filename, data, mode='a'):

		for i in range(len(data)):
			for j in range(len(data[i])):
				element = data[i][j]
				if type(element) == float:
					element = round(element, 6)
					data[i][j] = element
		with open(filename, mode) as server_log_file:
			writer = csv.writer(server_log_file)
			writer.writerows(data)

	def _get_size_of_dict(self, data):

		size = 0
		if type(data) == dict:
			for key in data:
				size += self._get_size_of_dict(data[key])
		else:
			size += sys.getsizeof(data)

		return size

	def calculate_bytes(self, parameters):

		try:
			size_list = []
			for i in range(len(parameters)):
				tamanho = get_size(parameters[i])
				# print("Client id: ", self.cid, " camada: ", i, " tamanho: ", tamanho, " shape: ", parameters[i].shape)
				size_list.append(tamanho)
			print("Tamanho total parametros evaluate: ", sum(size_list), " quantidade de camadas recebidas: ", len(parameters))
			size_of_parameters = sum(size_list)
			return size_of_parameters

		except Exception as e:
			print("calculate bytes")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)