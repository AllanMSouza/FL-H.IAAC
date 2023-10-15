import pandas as pd

from client.client_torch.client_base_torch import ClientBaseTorch
from client.client_torch import FedAvgClientTorch
from torch.nn.parameter import Parameter
import torch
import json
from pathlib import Path
import numpy as np
import os
import sys
import time
from client.ala import ALA

import warnings
warnings.simplefilter("ignore")

import logging
# logging.getLogger("torch").setLevel(logging.ERROR)

class FedAlaClientTorch(FedAvgClientTorch):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 args,
				 epochs=1,
				 model_name         = 'DNN',
				 client_selection   = False,
				 strategy_name      ='FedAla',
				 aggregation_method = 'None',
				 dataset            = '',
				 perc_of_clients    = 0,
				 decay              = 0,
				 fraction_fit		= 0,
				 non_iid            = False,
				 n_personalized_layers	= 1,
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

		self.layer_idx = 4
		self.rand_percent = 0.3
		self.eta = 0.1
		self.lr_loss = torch.nn.MSELoss()
		self.ALA = ALA(cid=self.cid, loss=self.loss, train_data=self.traindataset, batch_size=32, rand_percent=self.rand_percent, layer_idx=self.layer_idx, eta=self.eta, device=self.device)

	def get_parameters_of_model(self):
		try:
			parameters = [i.detach().numpy() for i in self.model.parameters()]
			return parameters
		except Exception as e:
			print("get parameters of model")
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
			filename = """./{}_saved_weights/{}/{}/model.pth""".format(self.strategy_name.lower(), self.model_name, self.cid)
			if Path(filename).exists():
				os.remove(filename)
			torch.save(self.model.state_dict(), filename)
		except Exception as e:
			print("save parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def set_parameters_to_model_fit(self, parameters):
		try:
			self.set_parameters_to_model(parameters)
			self.ALA.adaptive_local_aggregation(parameters, self.model)
		except Exception as e:
			print("set parameters to model train")
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
			filename = """./{}_saved_weights/{}/{}/model.pth""".format(self.strategy_name.lower(), self.model_name, self.cid, self.cid)
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

	def set_parameters_to_model_evaluate(self, parameters, config={}):
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
			self.set_parameters_to_model(parameters)
		except Exception as e:
			print("Set parameters to model evaluate")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
