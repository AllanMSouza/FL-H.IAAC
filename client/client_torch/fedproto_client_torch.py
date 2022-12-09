import flwr as fl
import tensorflow
import random
import time
import numpy as np
import copy
import torch
import os
import time
import sys
from collections import defaultdict
from pathlib import Path
from dataset_utils import ManageDatasets
from model_definition_torch import ModelCreation
import csv
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
import warnings
import json
warnings.simplefilter("ignore")
from client.client_torch.client_base_torch import ClientBaseTorch
import logging
# logging.getLogger("torch").setLevel(logging.ERROR)
from torch.nn.parameter import Parameter
class FedProtoClientTorch(ClientBaseTorch):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 epochs				= 1,
				 model_name         = 'None',
				 client_selection   = False,
				 solution_name      = 'None',
				 aggregation_method = 'None',
				 dataset            = '',
				 perc_of_clients    = 0,
				 decay              = 0,
				 non_iid            = False,
				 ):

		super().__init__(cid=cid,
						 n_clients=n_clients,
						 n_classes=n_classes,
						 epochs=epochs,
						 model_name=model_name,
						 client_selection=client_selection,
						 solution_name=solution_name,
						 aggregation_method=aggregation_method,
						 dataset=dataset,
						 perc_of_clients=perc_of_clients,
						 decay=decay,
						 non_iid=non_iid)

		self.protos = None
		self.global_protos = None
		self.loss_mse = nn.MSELoss()

		self.lamda = 1
		self.protos_samples_per_class = {i: 0 for i in range(self.num_classes)}

	def create_model(self):

		# print("tamanho: ", self.input_shape)
		input_shape = self.input_shape[1]*self.input_shape[2]
		if self.model_name == 'Logist Regression':
			return ModelCreation().create_LogisticRegression(input_shape, self.num_classes)

		elif self.model_name == 'DNN':
			return ModelCreation().create_DNN(input_shape=input_shape, num_classes=self.num_classes, use_proto=True)

		elif self.model_name == 'CNN':
			return ModelCreation().create_CNN(input_shape, self.num_classes)

		else:
			raise Exception("Wrong model name")

	def get_parameters(self, config):
		parameters = [i.detach().numpy() for i in self.model.parameters()]
		return parameters

		# It does the same of "get_parameters", but using "get_parameters" in outside of the core of Flower is causing errors
	def get_parameters_of_model(self):
		parameters = [i.detach().numpy() for i in self.model.parameters()]
		return parameters

	def inicial(self, parameters):
		parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
		for new_param, old_param in zip(parameters, self.model.parameters()):
			old_param.data = new_param.data.clone()

	def clone_model(self, model, target):
		for param, target_param in zip(model.parameters(), target.parameters()):
			target_param.data = param.data.clone()
		# target_param.grad = param.grad.clone()

	# def save_parameters(self):
	# 	filename = """./fedper_saved_weights/{}/{}/{}.json""".format(self.model_name, self.cid, self.cid)
	# 	weights = self.model.get_weights()
	# 	personalized_layers_weights = []
	# 	for i in range(self.n_personalized_layers):
	# 		personalized_layers_weights.append(weights[len(weights)-self.n_personalized_layers+i])
	# 	data = json.dumps([i.tolist() for i in personalized_layers_weights])
	# 	jsonFile = open(filename, "w")
	# 	jsonFile.write(data)
	# 	jsonFile.close()


	# def set_parameters_to_model(self, parameters):
	# 	parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
	# 	for new_param, old_param in zip(parameters, self.model.parameters()):
	# 		old_param.data = new_param.data.clone()

	def fit(self, parameters, config):
		try:
			selected_clients   = []
			trained_parameters = []
			selected           = 0

			if config['selected_clients'] != '':
				selected_clients = [int (cid_selected) for cid_selected in config['selected_clients'].split(' ')]

			start_time = time.process_time()
			#print(config)
			if self.cid in selected_clients or self.client_selection == False or int(config['round']) == 1:

				# the parameters are saved in a file because in each round new instances of client are created
				if int(config['round']) == 1:
					self.inicial(parameters)
				if int(config['round']) > 1:
					self.load_and_set_parameters()
					self.set_proto(parameters)

				selected = 1
				self.model.train()

				start_time = time.time()

				max_local_steps = self.local_epochs
				train_acc = 0
				train_loss = 0
				train_num = 0
				protos = defaultdict(list)
				for step in range(max_local_steps):
					for i, (x, y) in enumerate(self.trainloader):
						if type(x) == type([]):
							x[0] = x[0].to(self.device)
						else:
							x = x.to(self.device)
						y = y.to(self.device)
						train_num += y.shape[0]

						self.optimizer.zero_grad()
						rep = self.model.base(x)
						output = self.model.head(rep)
						y = torch.tensor(y.int().detach().numpy().astype(int).tolist())
						loss = self.loss(output, y)

						# if self.global_protos != None:
						# 	proto_new = np.zeros(rep.shape)
						# 	for i, yy in enumerate(y):
						# 		y_c = yy.item()
						# 		# print("aqui1", self.global_protos[y_c].shape, rep.shape)
						# 		# print("isso")
						# 		# print(self.global_protos[y_c].shape)
						# 		# print("passou")
						# 		proto_new[i] = self.global_protos[y_c]
						# 		# print(proto_new[i,:].shape)
						# 		# print("passou 2")
						# 	proto_new = torch.Tensor(proto_new.tolist())
						# 	loss += self.loss_mse(proto_new, rep) * self.lamda
						loss.backward()
						self.optimizer.step()

						for i, yy in enumerate(y):
							y_c = yy.item()
							protos[y_c].append(rep[i, :].detach().data)
							self.protos_samples_per_class[y_c] += 1


						# train_loss += float(loss.detach().numpy())
						train_loss += loss.item() * y.shape[0]
						train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

				self.save_parameters()

			self.protos = self.agg_func(protos)
			# print("juntou", self.protos)
			total_time         = time.process_time() - start_time
			size_of_parameters = sum(map(sys.getsizeof, trained_parameters))
			avg_loss_train     = train_loss/train_num
			avg_acc_train      = train_acc/train_num

			filename = f"logs/{self.solution_name}/{self.n_clients}/{self.model_name}/{self.dataset}/train_client.csv"
			data = [config['round'], self.cid, selected, total_time, size_of_parameters, avg_loss_train, avg_acc_train]

			self._write_output(
				filename=filename,
				data=data)

			fit_response = {
				'cid': self.cid,
				'protos_samples_per_class': self.protos_samples_per_class
			}

			return self.protos, train_num, fit_response
		except Exception as e:
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def evaluate(self, parameters, config):
		try:
			self.set_proto(parameters)
			# loss, accuracy     = self.model.evaluate(self.x_test, self.y_test, verbose=0)
			self.load_and_set_parameters()
			self.model.eval()

			test_acc = 0
			test_loss = 0
			test_num = 0

			with torch.no_grad():
				for x, y in self.testloader:
					if type(x) == type([]):
						x[0] = x[0].to(self.device)
					else:
						x = x.to(self.device)
					self.optimizer.zero_grad()
					y = y.to(self.device)
					y = torch.tensor(y.int().detach().numpy().astype(int).tolist())
					rep = self.model.base(x)

					output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)

					for i, r in enumerate(rep):
						for j in range(len(self.global_protos)):
							pro = torch.Tensor(self.global_protos[j].tolist())

							output[i, j] = self.loss_mse(r, pro)

					test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
					# output2 = self.model.head(rep)
					loss = self.loss(output, y)
					test_loss += loss.item() * y.shape[0]
					test_num += y.shape[0]

			size_of_parameters = sum(map(sys.getsizeof, parameters))
			loss = test_loss/test_num
			accuracy = test_acc/test_num
			filename = f"logs/{self.solution_name}/{self.n_clients}/{self.model_name}/{self.dataset}/evaluate_client.csv"
			data = [config['round'], self.cid, size_of_parameters, loss, accuracy]

			self._write_output(filename=filename,
							   data=data)

			evaluation_response = {
				"cid"      : self.cid,
				"accuracy" : float(accuracy)
			}

			return loss, test_num, evaluation_response
		except Exception as e:
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def load_and_set_parameters(self):
		# filename = """./fedproto_saved_weights/{}/{}/{}.json""".format(self.model_name, self.cid, self.cid)
		# if Path(filename).exists():
		# 	fileObject = open(filename, "r")
		# 	jsonContent = fileObject.read()
		# 	aList = [i for i in json.loads(jsonContent)]
		# 	self.set_parameters_to_model(aList)
		try:
			filename = """./fedproto_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid, self.cid)
			self.model = torch.load(filename)
		except Exception as e:
			print("load and set parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def save_parameters(self):
		# os.makedirs("""{}/fedproto_saved_weights/{}/{}/""".format(os.getcwd(), self.model_name, self.cid),
		# 			exist_ok=True)
		try:
			filename = """./fedproto_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid)
			torch.save(self.model, filename)
		except Exception as e:
			print("save parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


	def set_proto(self, protos):
		self.global_protos = copy.deepcopy(protos)

	def agg_func(self, protos):
		"""
        Returns the average of the weights.
        """
		try:
			for [label, proto_list] in protos.items():
				if len(proto_list) > 1:
					proto = 0 * proto_list[0].data
					for i in proto_list:
						proto += i.data
					protos[label] = proto / len(proto_list)
				else:
					protos[label] = proto_list[0]

			numpy_protos = []
			for key in protos:
				numpy_protos.append(protos[key].detach().numpy())

			return numpy_protos
		except Exception as e:
			print("agg fun")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)