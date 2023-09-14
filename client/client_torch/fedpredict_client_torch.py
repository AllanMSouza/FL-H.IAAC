from client.client_torch import FedAvgClientTorch
from ..fedpredict_core import fedpredict_client
from torch.nn.parameter import Parameter
import torch
from pathlib import Path
import os
import sys

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

	def set_parameters_to_model_evaluate(self, global_parameters, config={}):
		# Using 'torch.load'
		try:
			self.model = fedpredict_client(self.filename, self.model, global_parameters, config)

		except Exception as e:
			print("Set parameters to model")
			print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, self.cid), type(e).__name__, e)

