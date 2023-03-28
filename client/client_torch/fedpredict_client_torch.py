from client.client_torch import FedAvgClientTorch
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

class FedPredictClientTorch(FedAvgClientTorch):

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

	def save_parameters(self):
		# Using 'torch.save'
		try:
			filename = """./fedpredict_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid)
			if Path(filename).exists():
				os.remove(filename)
			torch.save(self.model.state_dict(), filename)
		except Exception as e:
			print("save parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def _fedpredict_plugin(self, global_parameters, t, T, last_global_accuracy, client_metrics):

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
				evolutionary_level = t/T

				# print("el servidor: ", el, " el local: ", evolutionary_level)

				eq1 = (-updated_level - evolutionary_level)
				eq2 = round(np.exp(eq1), 6)
				global_model_weight = eq2

			local_model_weights = 1 - global_model_weight

			print("rodada: ", t, " rounds sem fit: ", nt, "\npeso global: ", global_model_weight, " peso local: ", local_model_weights)

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

	def set_parameters_to_model_evaluate(self, global_parameters, config={}):
		# Using 'torch.load'
		try:
			filename = """./fedpredict_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid, self.cid)
			t = int(config['round'])
			T = int(config['total_server_rounds'])
			client_metrics = config['metrics']
			last_global_accuracy = config['last_global_accuracy']
			parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_parameters]
			for new_param, old_param in zip(parameters, self.model.parameters()):
				old_param.data = new_param.data.clone()
			if os.path.exists(filename):
				# Load local parameters to 'self.model'
				self.model.load_state_dict(torch.load(filename))
				self._fedpredict_plugin(global_parameters, t, T, last_global_accuracy, client_metrics)
		except Exception as e:
			print("Set parameters to model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)