import flwr as fl
import copy
import numpy as np
import torch
import time
import sys

from dataset_utils_torch import ManageDatasets
from models.torch import DNN, Logistic, CNN, MobileNet, resnet20, CNN_EMNIST, MobileNetV2, CNN_X, CNN_5, CNN_2, CNN_3, CNN_3_GTSRB, MobileNet_V3
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

			#params
			if self.aggregation_method == 'POC':
				self.solution_name = f"{solution_name}-{aggregation_method}-{self.perc_of_clients}"

			elif self.aggregation_method == 'FedLTA':
				self.solution_name = f"{solution_name}-{aggregation_method}-{self.decay}"

			elif self.aggregation_method == 'None':
				self.solution_name = f"{solution_name}-{aggregation_method}-{self.fraction_fit}"

			self.base = self._create_base_directory(self.type, self.solution_name, new_clients, new_clients_train, n_clients, model_name, dataset, str(args.class_per_client), str(args.alpha), str(args.rounds), str(args.local_epochs), str(args.comment), str(args.compression_method), args)
			self.evaluate_client_filename = f"{self.base}/evaluate_client.csv"
			self.train_client_filename = f"{self.base}/train_client.csv"
			self.predictions_client_filename = f"{self.base}/predictions_client.csv"

			self.trainloader, self.testloader, self.traindataset, self.testdataset = self.load_data(self.dataset, n_clients=self.n_clients)
			self.local_classes = self.calculate_imbalance_level()
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
			# self.device = 'cpu'
			# self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate) if self.model_name == "Mobilenet" else torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
			# self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

		except Exception as e:
			print("init client")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def _create_base_directory(self, type, strategy_name, new_clients, new_clients_train, n_clients, model_name, dataset, class_per_client, alpha, n_rounds, local_epochs, comment, compression, args):

		return f"logs/{type}/{strategy_name}/new_clients_{new_clients}_train_{new_clients_train}/{n_clients}/{model_name}/{dataset}/classes_per_client_{class_per_client}/alpha_{alpha}/{n_rounds}_rounds/{local_epochs}_local_epochs/{comment}_comment/{str(compression)}_compression"

	def calculate_imbalance_level(self):

		targets = self.traindataset.targets
		local_classes = len(pd.Series(targets).unique())/self.num_classes

		return local_classes


	def load_data(self, dataset_name, n_clients, batch_size=32):
		try:
			if dataset_name in ['MNIST', 'CIFAR10', 'CIFAR100', 'EMNIST', 'GTSRB', 'State Farm']:
				trainLoader, testLoader, traindataset, testdataset = ManageDatasets(self.cid, self.model_name).select_dataset(
					dataset_name, n_clients, self.class_per_client, self.alpha, self.non_iid, batch_size)
				self.input_shape = (3,64,64)
			else:
				print("gerar")
				trainLoader, testLoader, traindataset, testdataset = ManageDatasets(self.cid, self.model_name).select_dataset(
					dataset_name, n_clients, self.class_per_client, self.alpha, self.non_iid, batch_size)
				self.input_shape = (32, 0)
				# exit()

			return trainLoader, testLoader, traindataset, testdataset
		except Exception as e:
			print("load data")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def create_model(self):

		try:
			# print("tamanho: ", self.input_shape, " dispositivo: ", self.device)
			input_shape = self.input_shape
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

			if model is not None:
				model.to(self.device)
				return model
			else:
				raise Exception("Wrong model name")
		except Exception as e:
			print("create model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def save_parameters(self):
		pass

	def get_parameters(self, config):
		try:
			parameters = [i.detach().cpu().numpy() for i in self.model.parameters()]
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
			original_parameters = copy.deepcopy(parameters)
			print("dataloader: ", self.trainloader)
			if self.cid in selected_clients or self.client_selection == False or int(config['round']) == 1:
				self.set_parameters_to_model_fit(parameters)
				# self.save_parameters_global_model(parameters)
				self.round_of_last_fit = server_round

				selected = 1
				self.model.to(self.device)
				self.model.train()

				max_local_steps = self.local_epochs


				print("Cliente: ", self.cid, " rodada: ", server_round, " Quantidade de camadas: ", len([i for i in self.model.parameters()]), " device: ", self.device)
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
			# size_of_parameters = sum(
			# 	[sum(map(sys.getsizeof, trained_parameters[i])) for i in range(len(trained_parameters))])
			avg_loss_train = train_loss / train_num
			avg_acc_train = train_acc / train_num
			total_time = time.process_time() - start_time
			# loss, accuracy, test_num = self.model_eval()

			data = [config['round'], self.cid, selected, total_time, size_of_parameters, avg_loss_train, avg_acc_train]

			self._write_output(
				filename=self.train_client_filename,
				data=data)

			fit_response = {
				'cid': self.cid,
				'local_classes': self.local_classes
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
					self.optimizer.zero_grad()
					y = y.to(self.device)
					y = torch.tensor(y)
					output = self.model(x)
					loss = self.loss(output, y)
					test_loss += loss.item() * y.shape[0]
					prediction = torch.argmax(output, dim=1)
					predictions = np.append(predictions, prediction.cpu())
					labels = np.append(labels, y.cpu())
					test_acc += (torch.sum(prediction == y)).item()
					test_num += y.shape[0]

			loss = test_loss / test_num
			accuracy = test_acc / test_num

			return loss, accuracy, test_num, predictions, labels
		except Exception as e:
			print("model_eval")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def evaluate(self, parameters, config):
		try:
			server_round = int(config['round'])
			n_rounds = int(config['n_rounds'])
			nt = int(config['nt'])
			config['cid'] = self.cid
			print("p recebidos: ", len(parameters), " round: ", server_round, " nt: ", nt, " cid: ", self.cid)
			self.set_parameters_to_model_evaluate(parameters, config)
			size_of_parameters = self.calculate_bytes(parameters)
			# size_of_parameters = sum([sum(map(sys.getsizeof, parameters[i])) for i in range(len(parameters))])
			size_of_config = self._get_size_of_dict(config)
			loss, accuracy, test_num, predictions, labels = self.model_eval()
			data = [config['round'], self.cid, size_of_parameters, size_of_config, loss, accuracy, nt]
			self._write_output(filename=self.evaluate_client_filename,
							   data=data)

			if server_round == n_rounds:
				data = [[self.cid, server_round, int(p), int(l)] for p, l in zip(predictions, labels)]
				self._write_outputs(self.predictions_client_filename, data, 'a')

			evaluation_response = {
				"cid": self.cid,
				"accuracy": float(accuracy)
			}

			return loss, test_num, evaluation_response
		except Exception as e:
			print("evaluate")
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