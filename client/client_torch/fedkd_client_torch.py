from client.client_torch import FedAvgClientTorch
from torch.nn.parameter import Parameter
import torch
import json
import math
from pathlib import Path
import numpy as np
import flwr
import json
from utils.quantization.parameters_svd import inverse_parameter_svd_reading
import os
import sys
import time
import copy
import pandas as pd
from model_definition_torch import DNN, Logistic, CNN, CNN_student, AlexNet
from torchvision import models

import warnings
warnings.simplefilter("ignore")

import logging
# logging.getLogger("torch").setLevel(logging.ERROR)

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

class FedKDClientTorch(FedAvgClientTorch):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 args,
				 epochs=1,
				 model_name         = 'DNN',
				 client_selection   = False,
				 strategy_name      ='FedKD',
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

		self.lr_loss = torch.nn.MSELoss()
		self.round_of_last_fit = 0
		self.rounds_of_fit = 0
		self.T = int(args.T)
		self.accuracy_of_last_round_of_fit = 0
		self.start_server = 0

		# self.teacher_model, self.student_model = self.create_model_distillation()

	# def get_parameters(self, config):
	# 	try:
	# 		parameters = [i.detach().cpu().numpy() for i in self.student_model.parameters()]
	# 		return parameters
	# 	except Exception as e:
	# 		print("get parameters")
	# 		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
	#
	# def get_teacher_parameters(self, config):
	# 	try:
	# 		parameters = [i.detach().cpu().numpy() for i in self.teacher_model.parameters()]
	# 		return parameters
	# 	except Exception as e:
	# 		print("get parameters")
	# 		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
	#
	# def set_parameters_to_model(self, parameters):
	# 	try:
	# 		parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
	# 		for new_param, old_param in zip(parameters, self.student_model.parameters()):
	# 			old_param.data = new_param.data.clone()
	# 	except Exception as e:
	# 		print("set parameters to model")
	# 		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
	#
	# def save_parameters_student(self):
	# 	# usando 'torch.save'
	# 	try:
	# 		filename = """./{}_student_saved_weights/{}/{}/model.pth""".format(self.strategy_name.lower(), self.model_name, self.cid)
	# 		if Path(filename).exists():
	# 			os.remove(filename)
	# 		torch.save(self.student_model.state_dict(), filename)
	# 	except Exception as e:
	# 		print("save parameters student")
	# 		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
	#
	# def save_parameters_teacher(self):
	# 	# usando 'torch.save'
	# 	try:
	# 		filename = """./{}_teacher_saved_weights/{}/{}/model.pth""".format(self.strategy_name.lower(), self.model_name, self.cid)
	# 		if Path(filename).exists():
	# 			os.remove(filename)
	# 		torch.save(self.teacher_model.state_dict(), filename)
	# 	except Exception as e:
	# 		print("save parameters student")
	# 		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
	#
	# def set_parameters_to_model_fit(self, parameters):
	# 	try:
	# 		self.set_parameters_to_model(parameters)
	# 	except Exception as e:
	# 		print("set parameters to model train")
	# 		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
	#
	# def create_model_distillation(self):
	#
	# 	try:
	# 		# print("tamanho: ", self.input_shape, " dispositivo: ", self.device)
	# 		input_shape = self.input_shape
	# 		if self.dataset in ['MNIST', 'CIFAR10', 'CIFAR100']:
	# 			input_shape = self.input_shape[1]*self.input_shape[2]
	# 		elif self.dataset in ['MotionSense', 'UCIHAR']:
	# 			input_shape = self.input_shape[1]
	# 		if self.model_name == 'Logist Regression':
	# 			return Logistic(input_shape=input_shape, num_classes=self.num_classes)
	# 		elif self.model_name == 'DNN':
	# 			return DNN(input_shape=input_shape, num_classes=self.num_classes)
	# 		elif self.model_name == 'CNN'  and self.dataset in ['MNIST', 'CIFAR10']:
	# 			if self.dataset in ['MNIST']:
	# 				input_shape = 1
	# 				mid_dim_teacher = 256
	# 				mid_dim_student = 256
	# 			else:
	# 				input_shape = 3
	# 				mid_dim_teacher = 400
	# 				mid_dim_student = 100
	# 			return CNN(input_shape=input_shape, num_classes=self.num_classes, mid_dim=mid_dim_teacher), CNN_student(input_shape=input_shape, num_classes=self.num_classes, mid_dim=mid_dim_student)
	# 		elif self.dataset in ['Tiny-ImageNet']:
	# 			# return AlexNet(num_classes=self.num_classes)
	# 			model = models.resnet18(pretrained=True)
	# 			print("res: ")
	# 			# model = torch.nn.DataParallel(model).cuda()
	# 			return model
	# 		else:
	# 			raise Exception("Wrong model name")
	# 	except Exception as e:
	# 		print("create model")
	# 		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
	# def fit(self, parameters, config):
	# 	try:
	# 		selected_clients = []
	# 		trained_parameters = []
	# 		selected = 0
	#
	# 		if config['selected_clients'] != '':
	# 			selected_clients = [int(cid_selected) for cid_selected in config['selected_clients'].split(' ')]
	#
	# 		start_time = time.process_time()
	# 		server_round = int(config['round'])
	# 		original_parameters = copy.deepcopy(parameters)
	# 		if self.cid in selected_clients or self.client_selection == False or int(config['round']) == 1:
	# 			self.set_parameters_to_model_fit(parameters)
	# 			self.round_of_last_fit = server_round
	#
	# 			selected = 1
	# 			self.student_model.train()
	#
	# 			max_local_steps = self.local_epochs
	# 			train_acc = 0
	# 			train_loss = 0
	# 			train_num = 0
	# 			for step in range(max_local_steps):
	# 				for i, (x, y) in enumerate(self.trainloader):
	# 					if type(x) == type([]):
	# 						x[0] = x[0].to(self.device)
	# 					else:
	# 						x = x.to(self.device)
	# 					y = y.to(self.device)
	# 					train_num += y.shape[0]
	#
	# 					self.optimizer.zero_grad()
	# 					output = self.student_model(x)
	# 					y = torch.tensor(y.int().detach().numpy().astype(int).tolist())
	# 					print("saida: ", output.shape, " alvo: ", y.shape, output[0] )
	# 					loss = self.loss(output, y)
	# 					train_loss += loss.item() * y.shape[0]
	# 					loss.backward()
	# 					self.optimizer.step()
	#
	# 					train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
	#
	# 			trained_parameters = self.get_parameters_of_model()
	# 			self.save_parameters_student()
	#
	# 		size_list = []
	# 		for i in range(len(parameters)):
	# 			tamanho = get_size(parameters[i])
	# 			# print("Client id: ", self.cid, " camada: ", i, " tamanho: ", tamanho, " shape: ", parameters[i].shape)
	# 			size_list.append(tamanho)
	# 		# print("Tamanho total parametros fit: ", sum(size_list))
	# 		size_of_parameters = sum(size_list)
	# 		# size_of_parameters = sum(
	# 		# 	[sum(map(sys.getsizeof, trained_parameters[i])) for i in range(len(trained_parameters))])
	# 		avg_loss_train = train_loss / train_num
	# 		avg_acc_train = train_acc / train_num
	# 		total_time = time.process_time() - start_time
	# 		# loss, accuracy, test_num = self.model_eval()
	#
	# 		data = [config['round'], self.cid, selected, total_time, size_of_parameters, avg_loss_train, avg_acc_train]
	#
	# 		self._write_output(
	# 			filename=self.train_client_filename,
	# 			data=data)
	#
	# 		fit_response = {
	# 			'cid': self.cid
	# 		}
	# 		print("use gradient: ", self.use_gradient)
	# 		if self.use_gradient:
	# 			trained_parameters = [trained - original for trained, original in zip(trained_parameters, original_parameters)]
	# 			# trained_parameters = parameters_quantization_write(trained_parameters, 8)
	# 			# print("quantizou: ", trained_parameters[0])
	# 		return trained_parameters, train_num, fit_response
	# 	except Exception as e:
	# 		print("fit fedkd")
	# 		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def set_parameters_to_model_evaluate(self, global_parameters, config={}):
		# Using 'torch.load'
		try:
			print("Dimensões: ", [i.detach().numpy().shape for i in self.model.parameters()])
			print("Dimensões recebidas: ", [i.shape for i in global_parameters])
			global_parameters = inverse_parameter_svd_reading(global_parameters, [i.detach().numpy().shape for i in self.model.parameters()])
			parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_parameters]
			for new_param, old_param in zip(parameters, self.model.parameters()):
				old_param.data = new_param.data.clone()
		except Exception as e:
			print("Set parameters to model")
			print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, self.cid), type(e).__name__, e)

	# def set_parameters_to_student_model(self, parameters):
	# 	try:
	# 		parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
	# 		for new_param, old_param in zip(parameters, self.student_model.parameters()):
	# 			old_param.data = new_param.data.clone()
	# 	except Exception as e:
	# 		print("set parameters to model")
	# 		print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)