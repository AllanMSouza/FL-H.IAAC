from client.client_torch import FedAvgClientTorch
from client.client_torch.fedper_client_torch import FedPerClientTorch
from ..fedpredict_core import fedpredict_core
from utils.quantization.parameters_svd import inverse_parameter_svd_reading
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
				 args,
				 epochs=1,
				 model_name         = 'DNN',
				 client_selection   = False,
				 strategy_name      ='FedPredict',
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

		self.m_combining_layers = [i for i in range(len([i for i in self.create_model().parameters()]))]
		self.global_model = self.create_model().to(self.device)
		self.lr_loss = torch.nn.MSELoss()
		self.clone_model = self.create_model().to(self.device)
		self.round_of_last_fit = 0
		self.rounds_of_fit = 0
		self.T = int(args.T)
		self.accuracy_of_last_round_of_fit = 0
		self.start_server = 0
		self.filename = """./{}_saved_weights/{}/{}/model.pth""".format(strategy_name.lower(), self.model_name, self.cid)
		self.global_model_filename = """./{}_saved_weights/{}/{}/global_model.pth""".format(strategy_name.lower(), self.model_name,
																		self.cid)

	def save_parameters(self):
		# Using 'torch.save'
		try:
			# filename = """./fedpredict_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid)
			if Path(self.filename).exists():
				os.remove(self.filename)
			torch.save(self.model.state_dict(), self.filename)
		except Exception as e:
			print("save parameters")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def save_parameters_global_model(self, global_model):
		# Using 'torch.save'
		try:
			# filename = """./fedpredict_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid)
			if Path(self.global_model_filename).exists():
				os.remove(self.global_model_filename)

			parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_model]
			for new_param, old_param in zip(parameters, self.global_model.parameters()):
				old_param.data = new_param.data.clone()
			torch.save(self.global_model.state_dict(), self.filename)
		except Exception as e:
			print("save parameters global model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def _fedpredict_plugin(self, global_parameters, t, T, nt, M, sm):

		try:

			local_model_weights, global_model_weight = fedpredict_core(t, T, nt, sm)

			# Load global parameters into 'self.clone_model' (global model)
			global_parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_parameters]
			local_layer_count = 0
			global_layer_count = 0
			# parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_parameters]
			#
			# for old_param in self.clone_model.parameters():
			# 	if local_layer_count in M:
			# 		new_param = parameters[global_layer_count]
			# 		old_param.data = new_param.data.clone()
			# 		global_layer_count += 1
			# 	local_layer_count += 1

			# self.clone_model.load_state_dict(torch.load(filename))
			# Combine models
			count = 0
			for new_param, old_param in zip(self.global_model.parameters(), self.model.parameters()):
				if count in self.m_combining_layers:
					old_param.data = (global_model_weight*new_param.data.clone() + local_model_weights*old_param.data.clone())
				count += 1

		except Exception as e:
			print("merge models")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def _decompress(self, compressed_global_model_gradients, M):

		try:
			if len(compressed_global_model_gradients) > 0:
				decompressed_gradients = inverse_parameter_svd_reading(compressed_global_model_gradients, [i.shape for i in self.get_parameters({})], len(M))
				parameters = [Parameter(torch.Tensor(i.tolist())) for i in decompressed_gradients]
			else:
				parameters = []

			if os.path.exists(self.global_model_filename):
				# Load local parameters to 'self.model'
				self.global_model_filename.load_state_dict(torch.load(self.global_model_filename))

				local_layer_count = 0
				global_layer_count = 0
				for old_param in self.global_model.parameters():
					if local_layer_count in M:
						new_param = parameters[global_layer_count]
						# print("chegou new param: ", new_param.shape, " count: ", global_layer_count, " old param: ", old_param.shape, " count: ", local_layer_count)
						old_param.data = old_param.data.clone() + new_param.data.clone()
						global_layer_count += 1
					local_layer_count += 1

			else:
				for new_param, old_param in zip(parameters, self.global_model.parameters()):
					old_param.data = new_param.data.clone()

		except Exception as e:
			print("decompress")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def set_parameters_to_model_evaluate(self, global_parameters, config={}):
		# Using 'torch.load'
		try:
			# filename = """./fedpredict_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid, self.cid)
			t = int(config['round'])
			T = int(config['total_server_rounds'])
			if self.T != 0:
				T = self.T
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
			# print("chegou")
			M = config['M']
			sm = config['sm']
			local_layer_count = 0
			global_layer_count = 0
			print("decompress client: ", self.cid)
			self._decompress(global_parameters, M)
			parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_parameters]
			# print("parametros locais: ", [i.shape for i in self.model.parameters()])
			if len(parameters) != len(M):
				print("diferente", len(parameters), len(M))
			# print("M:", M)

			if os.path.exists(self.filename):
				# Load local parameters to 'self.model'
				self.model.load_state_dict(torch.load(self.filename))
				self._fedpredict_plugin(global_parameters, t, T, nt, M, sm)
			else:
				for old_param , new_param in zip(self.model.parameters(), self.global_model.parameters()):
					old_param.data = new_param.data.clone()

		except Exception as e:
			print("Set parameters to model")
			print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, self.cid), type(e).__name__, e)