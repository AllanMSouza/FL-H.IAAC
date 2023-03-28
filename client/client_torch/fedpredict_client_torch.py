from client.client_torch.client_base_torch import ClientBaseTorch
from client.client_torch.fedper_client_torch import FedPerClientTorch
from torch.nn.parameter import Parameter
import torch
import json
import math
from pathlib import Path
import numpy as np
import json
import os
import sys
import time
import pandas as pd

import warnings
warnings.simplefilter("ignore")

import logging
# logging.getLogger("torch").setLevel(logging.ERROR)

class FedPredictClientTorch(FedPerClientTorch):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 epochs=1,
				 model_name         = 'DNN',
				 client_selection   = False,
				 strategy_name      ='FedPredict',
				 aggregation_method = 'None',
				 dataset            = '',
				 perc_of_clients    = 0,
				 decay              = 0,
				 non_iid            = False,
				 n_personalized_layers	= 1,
				 new_clients			= False,
				 new_clients_train	= False
				 ):

		super().__init__(cid=cid,
						 n_clients=n_clients,
						 n_classes=n_classes,
						 epochs=epochs,
						 model_name=model_name,
						 client_selection=client_selection,
						 strategy_name=strategy_name,
						 aggregation_method=aggregation_method,
						 dataset=dataset,
						 perc_of_clients=perc_of_clients,
						 decay=decay,
						 non_iid=non_iid,
						 new_clients=new_clients,
						 new_clients_train=new_clients_train)

		self.n_personalized_layers = n_personalized_layers * 2
		self.lr_loss = torch.nn.MSELoss()
		self.clone_model = self.create_model().to(self.device)
		self.round_of_last_fit = 0
		self.rounds_of_fit = 0
		self.accuracy_of_last_round_of_fit = 0
		self.start_server = 0

	def get_parameters_of_model(self):
		try:
			parameters = [i.detach().numpy() for i in self.model.parameters()]
			return parameters
		except Exception as e:
			print("get parameters of model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def save_client_metrics_train(self, server_round):
		try:
			row = self.read_client_metrics()
			if row is not None:
				first_round = int(row['first_round'])
				round_of_last_fit = int(row['round_of_last_fit'])
				round_of_last_evaluate = int(row['round_of_last_evaluate'])
				acc_of_last_evaluate = float(row['acc_of_last_evaluate'])
				acc_of_last_fit = float(row['acc_of_last_fit'])
				if first_round == -1:
					first_round = server_round

			else:
				first_round = server_round
				round_of_last_evaluate = -1
				acc_of_last_evaluate = 0
				acc_of_last_fit = 0

			self.round_of_last_fit = server_round
			self.round_of_last_evaluate = round_of_last_evaluate
			self.accuracy_of_last_round_of_evalute = acc_of_last_evaluate
			self.accuracy_of_last_round_of_fit = acc_of_last_fit
			self.first_round = first_round
			pd.DataFrame(
				{'round_of_last_fit': [self.round_of_last_fit], 'round_of_last_evaluate': [self.round_of_last_evaluate],
				 'acc_of_last_fit': [self.accuracy_of_last_round_of_fit], 'acc_of_last_evaluate': [self.accuracy_of_last_round_of_evalute],
				 'first_round': [first_round]}).to_csv(
				"""fedpredict_saved_weights/{}/{}/{}.csv""".format(self.model_name, self.cid, self.cid), index=False)
		except Exception as e:
			print("save_metrics_train")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def save_client_metrics_evaluate(self, server_round, acc):
		try:
			row = self.read_client_metrics()
			if row is not None:
				first_round = int(row['first_round'])
				round_of_last_fit = int(row['round_of_last_fit'])
				round_of_last_evaluate = int(row['round_of_last_evaluate'])
				acc_of_last_fit = float(row['acc_of_last_fit'])
				if round_of_last_fit == server_round:
					acc_of_last_fit = acc
			else:
				first_round = -1
				round_of_last_fit = -1
				acc_of_last_fit = acc
			self.round_of_last_evaluate = server_round
			self.round_of_last_fit = round_of_last_fit
			self.accuracy_of_last_round_of_fit = acc_of_last_fit
			self.accuracy_of_last_round_of_evalute = acc
			self.first_round = first_round
			pd.DataFrame(
				{'round_of_last_fit': [self.round_of_last_fit], 'round_of_last_evaluate': [self.round_of_last_evaluate],
				 'acc_of_last_fit': [self.accuracy_of_last_round_of_fit], 'acc_of_last_evaluate': [self.accuracy_of_last_round_of_evalute],
				 'first_round': [first_round]}).to_csv(
				"""fedpredict_saved_weights/{}/{}/{}.csv""".format(self.model_name, self.cid, self.cid), index=False)
		except Exception as e:
			print("save_merics_evaluate")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def get_client_metrics(self):
		try:
			row = self.read_client_metrics()
			self.round_of_last_fit = int(row['round_of_last_fit'])
			self.round_of_last_evaluate = int(row['round_of_last_evaluate'])
			self.first_round = int(row['first_round'])
			self.accuracy_of_last_round_of_fit = float(row['acc_of_last_fit'])
			self.accuracy_of_last_round_of_evalute = float(row['acc_of_last_evaluate'])
		except Exception as e:
			print("On get_round_of_last_fit", " user id: ", self.cid, " row: ", row)
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def read_client_metrics(self):
		try:
			filename = """fedpredict_saved_weights/{}/{}/{}.csv""".format(self.model_name, self.cid, self.cid)
			if os.path.isfile(filename):
				row = pd.read_csv(filename)
				return row
			else:
				return None
		except Exception as e:
			print("read_client_metrics", " user id: ", self.cid)
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def save_parameters(self):
		# usando 'torch.save'
		try:
			filename = """./fedpredict_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid)
			if Path(filename).exists():
				os.remove(filename)
			torch.save(self.model.state_dict(), filename)
		except Exception as e:
			print("save parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def _fedpredict_plugin(self, global_parameters, server_round, last_global_accuracy, client_metrics):

		try:
			nt = client_metrics['nt']
			round_of_last_fit = client_metrics['round_of_last_fit']
			round_of_last_evaluate = client_metrics['round_of_last_evaluate']
			first_round = client_metrics['first_round']
			acc_of_last_fit = client_metrics['acc_of_last_fit']
			acc_of_last_evaluate = client_metrics['acc_of_last_evaluate']
			# 9
			if nt == 0:
				global_model_weight = 0
			else:
				# evitar que um modelo que treinou na rodada atual não utilize parâmetros globais pois esse foi atualizado após o seu treinamento
				# normalizar dentro de 0 e 1
				# updated_level = 1/rounds_without_fit
				# updated_level = 1 - max(0, -acc_of_last_fit+self.accuracy_of_last_round_of_evalute)
				# if acc_of_last_evaluate < last_global_accuracy:
				# updated_level = max(-last_global_accuracy + acc_of_last_evaluate, 0)
				# else:
				updated_level = 1/nt
				# evolutionary_level = (server_round / 50)
				# print("client id: ", self.cid, " primeiro round", self.first_round)
				evolutionary_level = (server_round)/50

				# print("el servidor: ", el, " el local: ", evolutionary_level)

				eq1 = (-updated_level - evolutionary_level)
				eq2 = round(np.exp(eq1), 6)
				global_model_weight = eq2

			local_model_weights = 1 - global_model_weight

			print("rodada: ", server_round, " rounds sem fit: ", nt, "\npeso global: ", global_model_weight, " peso local: ", local_model_weights)

			# Load global parameters into 'self.clone_model' (global model)
			global_parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_parameters]
			for new_param, old_param in zip(global_parameters, self.clone_model.parameters()):
				old_param.data = new_param.data.clone()
			# self.clone_model.load_state_dict(torch.load(filename))
			# Combine models
			for new_param, old_param in zip(self.clone_model.parameters(), self.model.parameters()):
				old_param.data = (global_model_weight*new_param.data.clone() + local_model_weights*old_param.data.clone())
		except Exception as e:
			print("merge models")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def set_parameters_to_model_train(self, global_parameters, server_round, config):
		# usando 'torch.load'
		try:
			filename = """./fedpredict_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid, self.cid)
			self.save_client_metrics_train(int(config['round']))
			# todos os fit são com parâmetros novos (do servidor)
			parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_parameters]
			for new_param, old_param in zip(parameters, self.model.parameters()):
				old_param.data = new_param.data.clone()
			# if os.path.exists(filename) and self.rounds_of_fit :
			# 	# todos os evaluate em rodadas menores que 35 são com os parâmetros personalizados*
			# 	self.clone_model.load_state_dict(torch.load(filename))
			# 	i = 0
			# 	for new_param, old_param in zip(self.clone_model.parameters(), self.model.parameters()):
			# 		if i >= 2:
			# 			old_param.data = torch.div(torch.sum(new_param.data.clone(), old_param.data.clone()), 2)
		except Exception as e:
			print("Set parameters to model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def set_parameters_to_model_evaluate(self, global_parameters, server_round, type, config):
		# usando 'torch.load'
		try:
			filename = """./fedpredict_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid, self.cid)
			# self.get_client_metrics()
			# rounds_without_fit = server_round - max(0, int(self.round_of_last_fit))
			print("configura: ", config)
			print("metricas: ", config['metrics'])
			metric = config['metrics']
			# el = metric['el']
			acc_of_last_fit = 0
			# 'round_of_last_fit': 0, 'round_of_last_evaluate': 0, 'first_round': -1,
			# 											   'acc_of_last_fit': 0, 'acc_of_last_evaluate':
			# print("leu: ", " cid: ", self.cid,  ' fedpredict_client_metrics: ', metric['fedpredict_client_metrics'][str(self.cid)])
			client_metrics = config['metrics']
			last_global_accuracy = config['last_global_accuracy']

			# if self.round_of_last_fit - server_round > 1:
			# 	raise "Round of last fit muito maior do que o server round"
			# if self.round_of_last_fit > 0:
			# 	acc_of_last_fit = metric['round_acc_el'][self.round_of_last_fit]['acc']
			# print("encontrou: ", metric['round_acc_el'], " ultima rodada: ", self.round_of_last_fit)
			round_acc_el = 0
			#
			# if self.rounds_of_fit <= 4:
			# 	parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_parameters]
			# 	for new_param, old_param in zip(parameters, self.model.parameters()):
			# 		old_param.data = new_param.data.clone()
			# else:
			# 	print("maior rounds of fit: ", self.rounds_of_fit, " cid: ", self.cid)
			parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_parameters]
			for new_param, old_param in zip(parameters, self.model.parameters()):
				old_param.data = new_param.data.clone()
			if os.path.exists(filename):
				# Load local parameters to 'self.model'
				self.model.load_state_dict(torch.load(filename))
				self._fedpredict_plugin(global_parameters, server_round, last_global_accuracy, client_metrics)
		except Exception as e:
			print("Set parameters to model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def fit(self, parameters, config):
		try:
			selected_clients   = []
			trained_parameters = []
			selected           = 0

			if config['selected_clients'] != '':
				selected_clients = [int (cid_selected) for cid_selected in config['selected_clients'].split(' ')]

			start_time = time.process_time()
			server_round = int(config['round'])
			if self.cid in selected_clients or self.client_selection == False or int(config['round']) == 1:
				self.set_parameters_to_model_train(parameters, server_round, config)
				self.round_of_last_fit = server_round

				selected = 1
				self.model.train()

				start_time = time.time()

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
						# print(output)
						y = torch.tensor(y.int().detach().numpy().astype(int).tolist())
						loss = self.loss(output, y)
						train_loss += loss.item() * y.shape[0]
						loss.backward()
						self.optimizer.step()

						train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

				trained_parameters = self.get_parameters_of_model()
				self.save_parameters()

			total_time         = time.process_time() - start_time
			size_of_parameters = sum([sum(map(sys.getsizeof, trained_parameters[i])) for i in range(len(trained_parameters))])
			avg_loss_train     = train_loss/train_num
			avg_acc_train      = train_acc/train_num

			loss, accuracy, test_num = self.model_eval()
			# self.accuracy_of_last_round_of_fit = accuracy

			data = [config['round'], self.cid, selected, total_time, size_of_parameters, avg_loss_train, avg_acc_train]

			self._write_output(
				filename=self.train_client_filename,
				data=data)

			fit_response = {
				'cid' : self.cid
			}

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
					# print("saida: ", output.shape)
					loss = self.loss(output, y)
					test_loss += loss.item() * y.shape[0]
					test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
					test_num += y.shape[0]

			loss = test_loss / test_num
			accuracy = test_acc / test_num

			return loss, accuracy, test_num
		except Exception as e:
			print("model_eval")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def evaluate(self, parameters, config):
		try:
			server_round = int(config['round'])
			self.set_parameters_to_model_evaluate(parameters, server_round, 'evaluate', config)
			# loss, accuracy     = self.model.evaluate(self.x_test, self.y_test, verbose=0)


			size_of_parameters = sum([sum(map(sys.getsizeof, parameters[i])) for i in range(len(parameters))])
			loss, accuracy, test_num = self.model_eval()
			self.save_client_metrics_evaluate(server_round, accuracy)
			data = [config['round'], self.cid, size_of_parameters, loss, accuracy]

			self._write_output(filename=self.evaluate_client_filename,
							   data=data)

			evaluation_response = {
				"cid"      : self.cid,
				"accuracy" : float(accuracy)
			}

			return loss, test_num, evaluation_response
		except Exception as e:
			print("evaluate")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
