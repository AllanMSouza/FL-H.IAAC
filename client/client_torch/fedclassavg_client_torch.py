from client.client_torch.client_base_torch import ClientBaseTorch
from client.client_torch.fedper_client_torch import FedPerClientTorch
from torch.nn.parameter import Parameter
import torch
import json
from pathlib import Path
from model_definition_torch import DNN, DNN_proto_2, DNN_proto_4, Logistic, CNN_proto
import numpy as np
import os
import sys
import time

import warnings
warnings.simplefilter("ignore")

import logging
# logging.getLogger("torch").setLevel(logging.ERROR)

class FedClassAvgClientTorch(FedPerClientTorch):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 epochs=1,
				 model_name         = 'DNN',
				 client_selection   = False,
				 strategy_name      ='FedClassAvg',
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

	def get_parameters_of_model(self):
		try:
			parameters = [i.detach().numpy() for i in self.model.parameters()]
			parameters = parameters[-self.n_personalized_layers:]
			return parameters
		except Exception as e:
			print("get parameters of model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def clone_model_classavg(self, model, target, c):
		try:
			i = 0
			parameters = [Parameter(torch.Tensor(i.tolist())) for i in c]
			size = len(self.get_parameters({}))
			j = 0
			for param, target_param in zip(model.parameters(), target.parameters()):
				# Set the global parameters in the last layer
				if i >= size - 2:
					target_param.data = parameters[j].data.clone()
					j+=1
				else:
					target_param.data = param.data.clone()
				i+=1
			# print("iterador: ", i)
		except Exception as e:
			print("clone model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


	def save_parameters(self):

		# ======================================================================================
		# usando json
		# try:
		# 	filename = """./fedper_saved_weights/{}/{}/{}.json""".format(self.model_name, self.cid, self.cid)
		# 	weights = self.get_parameters(config={})
		# 	personalized_layers_weights = []
		# 	for i in range(self.n_personalized_layers):
		# 		personalized_layers_weights.append(weights[len(weights)-self.n_personalized_layers+i])
		# 	data = json.dumps([i.tolist() for i in personalized_layers_weights])
		# 	jsonFile = open(filename, "w")
		# 	jsonFile.write(data)
		# 	jsonFile.close()
		# except Exception as e:
		# 	print("save parameters")
		# 	print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

		# ======================================================================================
		# usando 'torch.save'
		try:
			filename = """./fedclassavg_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid)
			if Path(filename).exists():
				os.remove(filename)
			torch.save(self.model.state_dict(), filename)
		except Exception as e:
			print("save parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def set_parameters_to_model(self, parameters):
		# ======================================================================================
		# usando json
		# try:
		# 	filename = """./fedclassavg_saved_weights/{}/{}/{}.json""".format( self.model_name, self.cid, self.cid)
		# 	if os.path.exists(filename):
		# 		fileObject = open(filename, "r")
		# 		jsonContent = fileObject.read()
		# 		aList = [np.array(i) for i in json.loads(jsonContent)]
		# 		size = len(parameters)
		# 		# updating only the personalized layers, which were previously saved in a file
		# 		# for i in range(self.n_personalized_layers):
		# 		# 	parameters[size-self.n_personalized_layers+i] = aList[i]
		# 		parameters = parameters + aList
		# 		parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
		# 		for new_param, old_param in zip(parameters, self.model.parameters()):
		# 			old_param.data = new_param.data.clone()
		# except Exception as e:
		# 	print("Set parameters to model")
		# 	print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

		# ======================================================================================
		# usando 'torch.load'
		try:
			filename = """./fedclassavg_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid, self.cid)
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
			#print(config)
			if self.cid in selected_clients or self.client_selection == False or int(config['round']) == 1:
				self.set_parameters_to_model(parameters)

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
						y = torch.tensor(y.int().detach().numpy().astype(int).tolist())
						loss = self.loss(output, y)
						local_parameters = [torch.Tensor(i) for i in self.get_parameters_of_model()]
						global_parameters = [torch.Tensor(i) for i in parameters]
						if config['round'] > 1:
							for i in range(len(local_parameters)):
								loss += torch.mul(self.lr_loss(local_parameters[i], global_parameters[i]), 1)
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


	def evaluate(self, parameters, config):
		try:
			self.set_parameters_to_model(parameters)
			# loss, accuracy     = self.model.evaluate(self.x_test, self.y_test, verbose=0)
			self.model.eval()
			clone_model = self.clone_model
			self.clone_model_classavg(self.model, clone_model, parameters)

			test_acc = 0
			test_loss = 0
			test_num = 0
			server_round = int(config['round'])

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
					output2 = self.clone_model(x)
					output = output + torch.mul(output2, 4/int(server_round))
					loss = self.loss(output, y)
					local_parameters = [torch.Tensor(i) for i in self.get_parameters_of_model()]
					global_parameters = [torch.Tensor(i) for i in parameters]
					# for i in range(len(local_parameters)):
					# 	loss += torch.mul(self.lr_loss(local_parameters[i], global_parameters[i]), 1/int(config['round']))
					test_loss += loss.item() * y.shape[0]
					test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
					test_num += y.shape[0]

			size_of_parameters = sum([sum(map(sys.getsizeof, parameters[i])) for i in range(len(parameters))])
			loss = test_loss/test_num
			accuracy = test_acc/test_num
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
