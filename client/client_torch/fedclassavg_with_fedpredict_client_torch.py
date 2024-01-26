from client.client_torch import FedPredictClientTorch, FedClassAvgClientTorch
from ..fedpredict_core import fedpredict_core
from client.client_torch.fedper_client_torch import FedPerClientTorch
from ..fedpredict_core import fedpredict_core, decompress_global_parameters, fedpredict_combine_models, fedpredict_client
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

	def __init__(cid,
				 n_clients,
				 n_classes,
				 args,
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


	# ====================================================================
		self.m_combining_layers = [i for i in range(len([i for i in self.create_model().parameters()]))][-2:]
		self.lr_loss = torch.nn.MSELoss()
		self.clone_model = self.create_model().to(self.device)
		self.round_of_last_fit = 0
		self.rounds_of_fit = 0
		self.T = int(args.T)
		self.accuracy_of_last_round_of_fit = 0
		self.start_server = 0
		self.filename = """./{}_saved_weights/{}/{}/model.pth""".format(strategy_name.lower(), self.model_name,
                                                                        self.pattern)

	def set_parameters_to_model_evaluate(self, global_parameters, config={}):
		# Using 'torch.load'
		try:
			self.model = fedpredict_client(self.filename, self.model, global_parameters, config)
		except Exception as e:
			print("Set parameters to model")
			print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)