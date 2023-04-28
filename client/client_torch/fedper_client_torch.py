from client.client_torch.client_base_torch import ClientBaseTorch
from torch.nn.parameter import Parameter
import torch
import json
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
				 args,
				 epochs=1,
				 model_name         = 'DNN',
				 client_selection   = False,
				 strategy_name      ='FedPer',
				 aggregation_method = 'None',
				 dataset            = '',
				 perc_of_clients    = 0,
				 decay              = 0,
				 fraction_fit		= 0,
				 non_iid            = False,
				 n_personalized_layers	= 1,
				 new_clients			= False,
				 new_clients_train	= False,

				 ):

		super().__init__(cid=cid,
						 n_clients=n_clients,
						 n_classes=n_classes,
						 epochs=epochs,
						 model_name=model_name,
						 client_selection=client_selection,
						 solution_name=strategy_name,
						 aggregation_method=aggregation_method,
						 dataset=dataset,
						 perc_of_clients=perc_of_clients,
						 decay=decay,
						 fraction_fit=fraction_fit,
						 non_iid=non_iid,
						 new_clients=new_clients,
						 new_clients_train=new_clients_train,
						 args=args)

		self.n_personalized_layers = n_personalized_layers * 2

	def get_parameters_of_model(self):
		try:
			parameters = [i.detach().numpy() for i in self.model.parameters()]
			parameters = parameters[:-self.n_personalized_layers]
			return parameters
		except Exception as e:
			print("get parameters of model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


	def save_parameters(self):
		# Using json
		try:
			filename = """./fedper_saved_weights/{}/{}/{}.json""".format(self.model_name, self.cid, self.cid)
			weights = self.get_parameters(config={})
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
				# Updating only the personalized layers, which were previously saved in a file
				# for i in range(self.n_personalized_layers):
				# 	parameters[size-self.n_personalized_layers+i] = aList[i]
				parameters = parameters + aList
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