import random
from client.client_torch import CDAFedAvgClientTorch
from ..fedpredict_core import fedpredict_client
from torch.nn.parameter import Parameter
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from dataset_utils_torch import ManageDatasets
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import ast
from scipy.special import softmax
from utils.compression_methods.sparsification import calculate_bytes, sparse_bytes, sparse_matrix
from client.cda_fedavg_concept_drift import cda_fedavg_drift_detection

import warnings
warnings.simplefilter("ignore")

import logging
# logging.getLogger("torch").setLevel(logging.ERROR)

class CDAFedAvgWithFedPredictClientTorch(CDAFedAvgClientTorch):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 args,
				 epochs=1,
				 model_name         = 'DNN',
				 client_selection   = False,
				 strategy_name      ='CDA-FedAvg_with_FedPredict',
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

		pass
		self.m_combining_layers = [i for i in range(len([i for i in self.create_model().parameters()]))]
		self.similarity = 1
		self.global_model = self.create_model().to(self.device)
		self.lr_loss = torch.nn.MSELoss()
		self.clone_model = self.create_model().to(self.device)
		self.round_of_last_fit = 0
		self.rounds_of_fit = 0
		self.T = int(args.T)
		self.accuracy_of_last_round_of_fit = 0
		self.start_server = 0
		self.client_information_filename = """{}_saved_weights/{}/{}/{}.csv""".format(strategy_name.lower(),
																					  self.model_name, self.cid,
																					  self.cid)
		self.filename = """./{}_saved_weights/{}/{}/model.pth""".format(strategy_name.lower(), self.model_name,
																		self.cid)
		self.global_model_filename = """./{}_saved_weights/{}/{}/global_model.pth""".format(strategy_name.lower(),
																							self.model_name,
																							self.cid)

	def save_parameters(self):
		# Using 'torch.save'
		try:
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

	def _calculate_contexts_similarities(self):

		try:
			"""
				It measures the cosine similarity between the last and current class distribution of the local dataset
			"""
			n = len(self.client_information_filename['classes_distribution'])
			# print("antes ", self.client_information_file['classes_distribution'].tolist()[-1])
			last_proportion = ast.literal_eval(self.client_information_filename['classes_distribution'].tolist()[-1])
			last_training = self.client_information_filename['round_of_last_fit'].tolist()[-1]

			current_proportion, imbalance_level = self._calculate_classes_proportion()
			fraction_of_classes = sum([1 if i > 0 else 0 for i in current_proportion])/self.num_classes

			if len(last_proportion) != len(current_proportion) or last_training == -1:
				return 1, imbalance_level, fraction_of_classes, current_proportion

			last_proportion = np.array(last_proportion)
			current_proportion = np.array(current_proportion)

			if (last_proportion == current_proportion).all():
				print("igual")

				cosine_similarity = 1

			else:
				print("diferente ")
				dot_product = np.dot(last_proportion, current_proportion)

				norm_vector1 = np.linalg.norm(last_proportion)

				norm_vector2 = np.linalg.norm(current_proportion)

				cosine_similarity = dot_product / (norm_vector1 * norm_vector2)

			return cosine_similarity, imbalance_level, fraction_of_classes, current_proportion

		except Exception as e:
			print("calculate contexts similarities")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

	def set_parameters_to_model_evaluate(self, global_parameters, config={}):
		# Using 'torch.load'
		try:
			self.model = fedpredict_client(self.filename, self.model, global_parameters, config, mode=None)

		except Exception as e:
			print("Set parameters to model evaluate dyn")
			print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, self.cid), type(e).__name__, e)



