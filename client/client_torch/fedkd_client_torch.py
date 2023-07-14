from client.client_torch import FedAvgClientTorch
from torch.nn.parameter import Parameter
import torch
import json
import math
from pathlib import Path
import numpy as np
import flwr
import json
from utils.quantization.parameters_svd import inverse_parameter_svd_reading, parameter_svd_write
import os
import sys
import time
import copy
import pandas as pd
from model_definition_torch import DNN, Logistic, CNN, CNN_student, DNN_student, DNN_teacher, AlexNet, CNNDistillation
from torchvision import models
import torch.nn.functional as F

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
		self.n_rate = float(args.n_rate)

		self.model = self.create_model_distillation()
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
		# self.optimizer_teacher = torch.optim.SGD(self.teacher_model.parameters(), lr=self.learning_rate, momentum=0.9)

	def get_parameters(self, config):
		try:
			parameters = [i.detach().cpu().numpy() for i in self.model.parameters()]
			return parameters
		except Exception as e:
			print("get parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def get_teacher_parameters(self, config):
		try:
			parameters = [i.detach().cpu().numpy() for i in self.teacher_model.parameters()]
			return parameters
		except Exception as e:
			print("get parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def set_parameters_to_model(self, parameters):
		try:
			parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
			for new_param, old_param in zip(parameters, self.model.student.parameters()):
				old_param.data = new_param.data.clone()
		except Exception as e:
			print("set parameters to model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def save_parameters(self):
		# usando 'torch.save'
		try:
			filename = """./{}_saved_weights/{}/{}/model.pth""".format(self.strategy_name.lower(), self.model_name, self.cid)
			if Path(filename).exists():
				os.remove(filename)
			torch.save(self.model.state_dict(), filename)
		except Exception as e:
			print("save parameters student")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def save_parameters_teacher(self):
		# usando 'torch.save'
		try:
			filename = """./{}_saved_weights/{}/{}/model.pth""".format(self.strategy_name.lower(), self.model_name, self.cid)
			if Path(filename).exists():
				os.remove(filename)
			torch.save(self.model.state_dict(), filename)
		except Exception as e:
			print("save parameters student")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def set_parameters_to_model_fit(self, parameters):
		try:
			# 		parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_parameters]
			self.set_parameters_to_model(parameters)
		except Exception as e:
			print("set parameters to model train")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def get_parameters_of_model(self):
		try:
			parameters = [i.detach().numpy() for i in self.model.student.parameters()]
			return parameters
		except Exception as e:
			print("get parameters of model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def create_model_distillation(self):

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
				return DNN_teacher(input_shape=input_shape, num_classes=self.num_classes), DNN_student(input_shape=input_shape, num_classes=self.num_classes)
			elif self.model_name == 'CNN'  and self.dataset in ['MNIST', 'CIFAR10']:
				if self.dataset in ['MNIST']:
					input_shape = 1
					mid_dim_teacher = 256
					mid_dim_student = 256
				else:
					input_shape = 3
					mid_dim_teacher = 400
					mid_dim_student = 100
				return CNNDistillation(input_shape=input_shape, num_classes=self.num_classes, mid_dim=mid_dim_teacher)
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

	def train_teacher(self):

		try:
			self.teacher_model.train()
			self.model.eval()

			max_local_steps = self.local_epochs
			train_acc_teacher = 0
			train_loss_teacher = 0
			train_num = 0
			kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
			for step in range(max_local_steps):
				for i, (x, y) in enumerate(self.trainloader):
					if type(x) == type([]):
						x[0] = x[0].to(self.device)
					else:
						x = x.to(self.device)
					y = y.to(self.device)
					train_num += y.shape[0]
					y = torch.tensor(y)

					self.optimizer_teacher.zero_grad()
					output_teacher = self.teacher_model(x)
					output_student = self.model(x)
					loss_teacher = self.loss(output_teacher, y)
					loss_student = self.loss(output_student, y)
					# print("valor da loss teacher: ",
					# 	  kl_loss(output_student, F.softmax(output_teacher)) / (loss_teacher + loss_student))
					print("valor da loss normal teacher: ", loss_student)
					# loss_teacher += kl_loss(output_student, F.softmax(output_teacher)) / (loss_teacher + loss_student)
					# loss_teacher += torch.nn.functional.kl_div(output_teacher, output_student) / (loss_teacher + loss_student)
					train_loss_teacher += loss_teacher.item() * y.shape[0]
					loss_teacher.backward()
					self.optimizer_teacher.step()

					train_acc_teacher += (torch.sum(torch.argmax(output_teacher, dim=1) == y)).item()
			print("acc teacher: ", train_acc_teacher/train_num)
			self.save_parameters_teacher()

			return train_loss_teacher, train_acc_teacher, train_num

		except Exception as e:
			print("train teacher")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def train_student(self, server_round):

		try:
			self.model.train()

			max_local_steps = self.local_epochs
			train_acc_student = 0
			train_acc_teacher = 0
			train_loss_student = 0
			train_num = 0
			kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
			for step in range(max_local_steps):
				for i, (x, y) in enumerate(self.trainloader):
					if type(x) == type([]):
						x[0] = x[0].to(self.device)
					else:
						x = x.to(self.device)
					y = y.to(self.device)
					y = torch.tensor(y)
					train_num += y.shape[0]

					self.optimizer.zero_grad()
					output_student, output_teacher = self.model(x)
					outputs_S1 = F.log_softmax(output_student, dim=1)
					outputs_S2 = F.log_softmax(output_teacher, dim=1)
					outputs_T1 = F.softmax(output_student, dim=1)
					outputs_T2 = F.softmax(output_teacher, dim=1)

					loss_student = self.loss(output_student, y)
					loss_teacher = self.loss(output_teacher, y)
					loss = torch.nn.KLDivLoss()(outputs_S1, outputs_T2) / (loss_student + loss_teacher)
					loss += torch.nn.KLDivLoss()(outputs_S2, outputs_T1) / (loss_student + loss_teacher)
					loss += loss_student + loss_teacher
					# loss_student = self.loss(output_student, y)
					# print("exemplo: ", output_teacher.shape, output_student.shape)
					# print("professor: ", output_teacher[0])
					# print("estudante: ", output_student[0])
					# print("valor da loss student: ", kl_loss(output_student, F.softmax(output_teacher))/(loss_teacher + loss_student))
					print("valor da loss normal student: ", loss)
					# loss_student += kl_loss(output_student, F.softmax(output_teacher))/(loss_teacher + loss_student)
					train_loss_student += loss.item() * y.shape[0]

					# print("saida: ", output_student.shape, " alvo: ", y.shape, output_student[0])
					loss.backward()
					self.optimizer.step()

					train_acc_student += (torch.sum(torch.argmax(output_student, dim=1) == y)).item()
					train_acc_teacher += (torch.sum(torch.argmax(output_teacher, dim=1) == y)).item()

			print("rodada: ", server_round)
			print("acc student: ", train_acc_student / train_num)
			print("acc teacher: ", train_acc_teacher / train_num)
			return train_loss_student, train_acc_student, train_num

		except Exception as e:
			print("train student")
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
			parameters = inverse_parameter_svd_reading(parameters, [i.detach().numpy().shape for i in
																	self.model.student.parameters()])
			original_parameters = copy.deepcopy(parameters)

			if self.cid in selected_clients or self.client_selection == False or server_round == 1:
				self.set_parameters_to_model_fit(parameters)
				self.load_parameters_to_model()
				self.round_of_last_fit = server_round

				selected = 1

				# train_loss_teacher, train_acc_teacher, train_num = self.train_teacher()
				train_loss_student, train_acc_student, train_num = self.train_student(server_round)

				trained_parameters = self.get_parameters_of_model()
				self.save_parameters()

			size_list = []
			for i in range(len(trained_parameters)):
				tamanho = get_size(trained_parameters[i])
				# print("Client id: ", self.cid, " camada: ", i, " tamanho: ", tamanho, " shape: ", parameters[i].shape)
				size_list.append(tamanho)
			# print("Tamanho total parametros fit: ", sum(size_list))
			size_of_parameters = sum(size_list)
			# size_of_parameters = sum(
			# 	[sum(map(sys.getsizeof, trained_parameters[i])) for i in range(len(trained_parameters))])
			avg_loss_train = train_loss_student / train_num
			avg_acc_train = train_acc_student / train_num
			total_time = time.process_time() - start_time
			# loss, accuracy, test_num = self.model_eval()

			data = [config['round'], self.cid, selected, total_time, size_of_parameters, avg_loss_train, avg_acc_train]

			self._write_output(
				filename=self.train_client_filename,
				data=data)

			fit_response = {
				'cid': self.cid
			}

			if self.use_gradient:
				trained_parameters = [trained - original for trained, original in zip(trained_parameters, original_parameters)]
			trained_parameters = parameter_svd_write(trained_parameters, self.n_rate)
			return trained_parameters, train_num, fit_response
		except Exception as e:
			print("fit fedkd")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	# def set_parameters_to_model_evaluate(self, global_parameters, config={}):
	# 	# Using 'torch.load'
	# 	try:
	# 		print("Dimensões: ", [i.detach().numpy().shape for i in self.model.parameters()])
	# 		print("Dimensões recebidas: ", [i.shape for i in global_parameters])
	# 		global_parameters = inverse_parameter_svd_reading(global_parameters, [i.detach().numpy().shape for i in self.model.parameters()])
	# 		parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_parameters]
	# 		for new_param, old_param in zip(parameters, self.model.parameters()):
	# 			old_param.data = new_param.data.clone()
	# 	except Exception as e:
	# 		print("Set parameters to model")
	# 		print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, self.cid), type(e).__name__, e)

	def set_parameters_to_model_evaluate(self, global_parameters, config={}):
		# Using 'torch.load'
		try:
			self.load_parameters_to_model()
		except Exception as e:
			print("Set parameters to model")
			print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, self.cid), type(e).__name__, e)

	def load_parameters_to_model(self):
		# ======================================================================================
		# usando 'torch.load'
		try:
			filename = """./{}_saved_weights/{}/{}/model.pth""".format(self.strategy_name.lower(), self.model_name,
																	   self.cid, self.cid)
			if os.path.exists(filename):
				self.model.load_state_dict(torch.load(filename))
		# size = len(parameters)
		# updating only the personalized layers, which were previously saved in a file
		# parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
		# i = 0
		# for new_param, old_param in zip(parameters, self.model.parameters()):
		# 	if i < len(parameters) - self.n_personalized_layers:
		# 		old_param.data = new_param.data.clone()
		# 	i += 1
		except Exception as e:
			print("load_parameters_to_model_teacher")
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
					y = torch.tensor(y)
					output, output_teacher = self.model(x)
					loss = self.loss(output_teacher, y)
					test_loss += loss.item() * y.shape[0]
					prediction = torch.argmax(output_teacher, dim=1)
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


	def set_parameters_to_student_model(self, parameters):
		try:
			parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
			for new_param, old_param in zip(parameters, self.model.parameters()):
				old_param.data = new_param.data.clone()
		except Exception as e:
			print("set parameters to model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)