from client.client_torch.client_base_torch import ClientBaseTorch
from torch.nn.parameter import Parameter
import torch
import json
from pathlib import Path
from model_definition_torch import ModelCreation
import numpy as np
import os
import sys

import warnings
warnings.simplefilter("ignore")

import logging
# logging.getLogger("torch").setLevel(logging.ERROR)

class FedPerClientTorch(ClientBaseTorch):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 epochs=1,
				 model_name         = 'DNN',
				 client_selection   = False,
				 solution_name      = 'None',
				 aggregation_method = 'None',
				 dataset            = '',
				 perc_of_clients    = 0,
				 decay              = 0,
				 non_iid            = False,
				 n_personalized_layers	= 1
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

		self.n_personalized_layers = n_personalized_layers * 2

	def create_model(self):

		# print("tamanho: ", self.input_shape)
		input_shape = self.input_shape[1]*self.input_shape[2]
		if self.model_name == 'Logist Regression':
			return ModelCreation().create_LogisticRegression(input_shape, self.num_classes)

		elif self.model_name == 'DNN':
			return ModelCreation().create_DNN(input_shape=input_shape, num_classes=self.num_classes, use_local_model=True)

		elif self.model_name == 'CNN':
			return ModelCreation().create_CNN(input_shape, self.num_classes)

		else:
			raise Exception("Wrong model name")

	def save_parameters(self):
		# usando json
		try:
			filename = """./fedper_saved_weights/{}/{}/{}.json""".format(self.model_name, self.cid, self.cid)
			weights = self.get_parameters_of_model()
			personalized_layers_weights = []
			for i in range(self.n_personalized_layers):
				personalized_layers_weights.append(weights[len(weights)-self.n_personalized_layers+i])
			data = json.dumps([i.tolist() for i in personalized_layers_weights])
			jsonFile = open(filename, "w")
			jsonFile.write(data)
			jsonFile.close()
		except Exception as e:
			print("save parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

		#======================================================================================
		# usando 'torch.save'
		# try:
		# 	filename = """./fedper_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid)
		# 	if Path(filename).exists():
		# 		os.remove(filename)
		# 	torch.save(self.model.state_dict(), filename)
		# except Exception as e:
		# 	print("save parameters")
		# 	print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def set_parameters_to_model(self, parameters):
		# usando json
		try:
			filename = """./fedper_saved_weights/{}/{}/{}.json""".format( self.model_name, self.cid, self.cid)
			if os.path.exists(filename):
				fileObject = open(filename, "r")
				jsonContent = fileObject.read()
				aList = [np.array(i) for i in json.loads(jsonContent)]
				size = len(parameters)
				# updating only the personalized layers, which were previously saved in a file
				for i in range(self.n_personalized_layers):
					parameters[size-self.n_personalized_layers+i] = aList[i]
				parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
				for new_param, old_param in zip(parameters, self.model.parameters()):
					old_param.data = new_param.data.clone()
		except Exception as e:
			print("Set parameters to model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

		# ======================================================================================
		# usando 'torch.load'
		# try:
		# 	filename = """./fedper_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid, self.cid)
		# 	if os.path.exists(filename):
		# 		self.model.load_state_dict(torch.load(filename))
		# 		size = len(parameters)
		# 		# updating only the personalized layers, which were previously saved in a file
		# 		parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
		# 		i = 0
		# 		for new_param, old_param in zip(parameters, self.model.parameters()):
		# 			if i < len(parameters) - self.n_personalized_layers:
		# 				old_param.data = new_param.data.clone()
		# 			i += 1
		# except Exception as e:
		# 	print("Set parameters to model")
		# 	print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)