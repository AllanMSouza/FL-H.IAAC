from client.client_torch import FedPredictClientTorch, FedPerClientTorch
from client.client_torch.fedper_client_torch import FedPerClientTorch
from ..fedpredict_core import fedpredict_core
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

class FedPer_with_FedPredictClientTorch(FedPredictClientTorch):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 args,
				 epochs=1,
				 model_name         = 'DNN',
				 client_selection   = False,
				 strategy_name      ='FedPer_with_FedPredict',
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

		self.m_combining_layers = [i for i in range(len([i for i in self.create_model().parameters()]))][-2:]
		self.filename = """./{}_saved_weights/{}/{}/model.pth""".format(self.strategy_name.lower(), self.model_name, self.cid)

	def _fedpredict_plugin(self, global_parameters, t, T, nt):

		try:

			local_model_weights, global_model_weight = fedpredict_core(t, T, nt)

			# self.clone_model.load_state_dict(torch.load(filename))
			# Combine models
			count = 0
			for new_param, old_param in zip(self.clone_model.parameters(), self.model.parameters()):
				if count in self.m_combining_layers:
					old_param.data = (global_model_weight*new_param.data.clone() + local_model_weights*old_param.data.clone())
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
			parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_parameters]
			for new_param, old_param in zip(parameters, self.model.parameters()):
				old_param.data = new_param.data.clone()
			if os.path.exists(self.filename):
				# Load local parameters to 'self.model'
				self.clone_model.load_state_dict(torch.load(self.filename))
				self._fedpredict_plugin(global_parameters, t, T, nt)
		except Exception as e:
			print("Set parameters to model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)