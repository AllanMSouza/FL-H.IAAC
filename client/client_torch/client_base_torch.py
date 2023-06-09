import flwr as fl
import random
import time
import copy
import numpy as np
import torch
import os
import time
import sys

from dataset_utils_torch import ManageDatasets
from model_definition_torch import DNN, Logistic, CNN, AlexNet
from torchvision import models
import csv
import torch.nn as nn
from utils.quantization.quantization import quantize_linear_symmetric
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.simplefilter("ignore")

import logging
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

			print("criar cliente")
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
			self.layer_selection_evaluate = int(args.layer_selection_evaluate)
			self.use_gradient = args.use_gradient

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
			self.learning_rate = 0.01
			self.new_clients = new_clients
			self.new_clients_train = new_clients_train
			# self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			self.type = 'torch'

			#params
			if self.aggregation_method == 'POC':
				self.solution_name = f"{solution_name}-{aggregation_method}-{self.perc_of_clients}"

			elif self.aggregation_method == 'FedLTA':
				self.solution_name = f"{solution_name}-{aggregation_method}-{self.decay}"

			elif self.aggregation_method == 'None':
				self.solution_name = f"{solution_name}-{aggregation_method}-{self.fraction_fit}"

			self.base = f"logs/{self.type}/{self.solution_name}/new_clients_{self.new_clients}_train_{self.new_clients_train}/{self.n_clients}/{self.model_name}/{self.dataset}/classes_per_client_{self.class_per_client}/alpha_{self.alpha}/{self.n_rounds}_rounds/{self.local_epochs}_local_epochs/{self.comment}_comment/{str(self.layer_selection_evaluate)}_layer_selection_evaluate"
			self.evaluate_client_filename = f"{self.base}/evaluate_client.csv"
			self.train_client_filename = f"{self.base}/train_client.csv"
			self.predictions_client_filename = f"{self.base}/predictions_client.csv"

			self.trainloader, self.testloader = self.load_data(self.dataset, n_clients=self.n_clients)
			self.model                                           = self.create_model().to(self.device)
			# self.device = 'cpu'
			self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)

		except Exception as e:
			print("init client")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def load_data(self, dataset_name, n_clients, batch_size=32):
		try:
			# print("y test")
			# print(np.unique(y_test))
			# exit()
			if dataset_name in ['MNIST', 'CIFAR10', 'CIFAR100']:
				x_train, y_train, x_test, y_test = ManageDatasets(self.cid, self.model_name).select_dataset(
					dataset_name, n_clients, self.class_per_client, self.alpha, self.non_iid)
				self.input_shape = x_train.shape
				tensor_x_train = torch.Tensor(x_train)  # transform to torch tensor
				tensor_y_train = torch.Tensor(y_train)

				train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
				trainLoader = DataLoader(train_dataset, batch_size, drop_last=True, shuffle=True)

				tensor_x_test = torch.Tensor(x_test)  # transform to torch tensor
				tensor_y_test = torch.Tensor(y_test)

				test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
				testLoader = DataLoader(test_dataset, batch_size, drop_last=True, shuffle=True)
				print("Carregou dados de ", self.dataset, len(x_train))
			else:
				print("gerar")
				trainset, valset = ManageDatasets(self.cid, self.model_name).select_dataset(
					dataset_name, n_clients, self.class_per_client, self.alpha, self.non_iid)
				print("leu dataset", len(trainset))
				self.input_shape = (32, 0)
				trainLoader = DataLoader(dataset=trainset, batch_size=256, pin_memory=True, shuffle=True)
				testLoader = DataLoader(dataset=valset, batch_size=256, pin_memory=True, shuffle=False)
				print("leu loader: ")
				# exit()

			return trainLoader, testLoader
		except Exception as e:
			print("load data")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def create_model(self):

		try:
			# print("tamanho: ", self.input_shape, " dispositivo: ", self.device)
			input_shape = self.input_shape
			if self.dataset in ['MNIST', 'CIFAR10', 'CIFAR100']:
				input_shape = self.input_shape[1]*self.input_shape[2]
			elif self.dataset in ['MotionSense', 'UCIHAR']:
				input_shape = self.input_shape[1]
			if self.model_name == 'Logist Regression':
				return Logistic(input_shape=input_shape, num_classes=self.num_classes)
			elif self.model_name == 'DNN':
				return DNN(input_shape=input_shape, num_classes=self.num_classes)
			elif self.model_name == 'CNN'  and self.dataset in ['MNIST', 'CIFAR10']:
				if self.dataset in ['MNIST']:
					input_shape = 1
					mid_dim = 256
				else:
					input_shape = 3
					mid_dim = 400
				return CNN(input_shape=input_shape, num_classes=self.num_classes, mid_dim=mid_dim)
			elif self.dataset in ['Tiny-ImageNet']:
				# return AlexNet(num_classes=self.num_classes)
				model = models.resnet18(pretrained=True)
				print("res: ")
				# model = torch.nn.DataParallel(model).cuda()
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
			parameters = [i.detach().numpy() for i in self.model.parameters()]
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

	def set_parameters_to_model(self, parameters):
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
			self.set_parameters_to_model(parameters)
		except Exception as e:
			print("set parameters to model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def fit(self, parameters, config):
		try:
			selected_clients = []
			trained_parameters = []
			selected = 0

			if config['selected_clients'] != '':
				selected_clients = [int(cid_selected) for cid_selected in config['selected_clients'].split(' ')]

			start_time = time.process_time()
			server_round = int(config['round'])
			if self.use_gradient:
				original_parameters = copy.deepcopy(parameters)
			if self.cid in selected_clients or self.client_selection == False or int(config['round']) == 1:
				self.set_parameters_to_model_fit(parameters)
				self.round_of_last_fit = server_round

				selected = 1
				self.model.train()

				max_local_steps = self.local_epochs
				train_acc = 0
				train_loss = 0
				train_num = 0
				for step in range(max_local_steps):
					for i, (x, y) in enumerate(self.trainloader):
						if type(x) == type([]):
							x[0] = x[0].to(self.device)
						else:
							x = x.to(self.device)
						y = y.to(self.device)
						train_num += y.shape[0]

						self.optimizer.zero_grad()
						output = self.model(x)
						y = torch.tensor(y.int().detach().numpy().astype(int).tolist())
						loss = self.loss(output, y)
						train_loss += loss.item() * y.shape[0]
						loss.backward()
						self.optimizer.step()

						train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

				trained_parameters = self.get_parameters_of_model()
				self.save_parameters()

			size_list = []
			for i in range(len(parameters)):
				tamanho = get_size(parameters[i])
				print("Client id: ", self.cid, " camada: ", i, " tamanho: ", tamanho, " shape: ", parameters[i].shape)
				size_list.append(tamanho)
			print("Tamanho total parametros fit: ", sum(size_list))
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
				'quantize': quantize_linear_symmetric(np.array([10, 2]), 3)
			}

			if self.use_gradient:
				trained_parameters = [trained - original for trained, original in zip(trained_parameters, original_parameters)]

			return trained_parameters, train_num, fit_response
		except Exception as e:
			print("fit")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def model_eval(self):
		try:
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
					self.optimizer.zero_grad()
					y = y.to(self.device)
					y = torch.tensor(y.int().detach().numpy().astype(int).tolist())
					output = self.model(x)
					loss = self.loss(output, y)
					test_loss += loss.item() * y.shape[0]
					prediction = torch.argmax(output, dim=1)
					predictions = np.append(predictions, prediction)
					labels = np.append(labels, y)
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
			self.set_parameters_to_model_evaluate(parameters, config)
			# loss, accuracy     = self.model.evaluate(self.x_test, self.y_test, verbose=0)

			size_list = []
			for i in range(len(parameters)):
				tamanho = get_size(parameters[i])
				print("Client id: ", self.cid, " camada: ", i, " tamanho: ", tamanho, " shape: ", parameters[i].shape)
				size_list.append(tamanho)
			print("Tamanho total parametros evaluate: ", sum(size_list), " quantidade de camadas recebidas: ", len(parameters))
			size_of_parameters = sum(size_list)
			# size_of_parameters = sum([sum(map(sys.getsizeof, parameters[i])) for i in range(len(parameters))])
			size_of_config = self._get_size_of_dict(config)
			loss, accuracy, test_num, predictions, labels = self.model_eval()
			data = [config['round'], self.cid, size_of_parameters, size_of_config, loss, accuracy]

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