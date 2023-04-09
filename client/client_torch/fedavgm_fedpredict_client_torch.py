from client.client_torch import FedPredictClientTorch, FedAvgMClientTorch
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

class FedAvgM_FedPredictClientTorch(FedAvgMClientTorch, FedPredictClientTorch):

	def __init__(self,
				 cid,
				 n_clients,
				 n_classes,
				 epochs=1,
				 model_name         = 'DNN',
				 client_selection   = False,
				 strategy_name      ='FedAvgM_FedPredict',
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

		# self.n_personalized_layers = n_personalized_layers * 2
		# self.lr_loss = torch.nn.MSELoss()
		# self.clone_model = self.create_model().to(self.device)
		# self.round_of_last_fit = 0
		# self.rounds_of_fit = 0
		# self.accuracy_of_last_round_of_fit = 0
		# self.start_server = 0
