from client.client_torch import FedPredictClientTorch, FedClassAvgClientTorch
from ..fedpredict_core import fedpredict_core
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

class FedClassAvg_with_FedPredictClientTorch(FedClassAvgClientTorch):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 epochs=1,
				 model_name         = 'DNN',
				 client_selection   = False,
				 strategy_name      ='FedClassAvg_with_FedPredict',
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


	# ====================================================================
		self.m_combining_layers = [i for i in range(len([i for i in self.create_model().parameters()]))][-2:]
		self.lr_loss = torch.nn.MSELoss()
		self.clone_model = self.create_model().to(self.device)
		self.round_of_last_fit = 0
		self.rounds_of_fit = 0
		self.accuracy_of_last_round_of_fit = 0
		self.start_server = 0
		self.filename = """./{}_saved_weights/{}/{}/model.pth""".format(strategy_name.lower(), self.model_name,
																		self.cid)




	def _fedpredict_plugin(self, global_parameters, t, T, nt):

		try:

			local_model_weights, global_model_weight = fedpredict_core(t, T, nt)

			# Load global parameters into 'self.clone_model' (global model)
			global_parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_parameters]
			size_local_parameters = len([i for i in self.model.parameters()])
			size_global_parameters = len(global_parameters)
			size = size_local_parameters - size_global_parameters
			# for new_param, old_param in zip(global_parameters, self.clone_model.parameters()):
			# 	old_param.data = new_param.data.clone()
			# self.clone_model.load_state_dict(torch.load(filename))
			# Combine models
			count = 0
			global_parameter_count = 0
			for local_param in self.model.parameters():
				if count >= size:
					global_param = global_parameters[global_parameter_count]
					local_param.data = (global_model_weight*global_param.data.clone() + local_model_weights*local_param.data.clone())
					global_parameter_count += 1
				count += 1
		except Exception as e:
			print("merge models")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def set_parameters_to_model_evaluate(self, global_parameters, config={}):
		# Using 'torch.load'
		try:
			# filename = """./fedpredict_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid, self.cid)
			t = int(config['round'])
			T = int(config['total_server_rounds'])
			client_metrics = config['metrics']
			# Client's metrics
			nt = client_metrics['nt']
			round_of_last_fit = client_metrics['round_of_last_fit']
			round_of_last_evaluate = client_metrics['round_of_last_evaluate']
			first_round = client_metrics['first_round']
			acc_of_last_fit = client_metrics['acc_of_last_fit']
			acc_of_last_evaluate = client_metrics['acc_of_last_evaluate']
			# Server's metrics
			last_global_accuracy = config['last_global_accuracy']
			# if t == 1:
			parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_parameters]
			print("tamanho global: ", len(parameters))
				#for new_param, old_param in zip(parameters, self.model.parameters()):
				#	old_param.data = new_param.data.clone()
			if os.path.exists(self.filename):
				# Load local parameters to 'self.model'
				self.model.load_state_dict(torch.load(self.filename))
				self._fedpredict_plugin(global_parameters, t, T, nt)
		except Exception as e:
			print("Set parameters to model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)