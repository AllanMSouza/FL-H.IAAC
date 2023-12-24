import flwr as fl
import numpy as np
import math
import os
import time
import csv
import random
import sys
import pandas as pd
import copy

from server.common_base_server import FedAvgBaseServer
from client.fedpredict_core import fedpredict_core_layer_selection, fedpredict_layerwise_similarity, fedpredict_core_compredict, dls, layer_compression_range, compredict, fedpredict_server
from utils.compression_methods.parameters_svd import if_reduces_size
from utils.compression_methods.sparsification import sparse_crs_top_k, to_dense, client_model_non_zero_indexes

from pathlib import Path
import shutil

from typing import Callable, Dict, Optional, Tuple

class CDAFedAvgBaseServer(FedAvgBaseServer):

	def __init__(self,
				 aggregation_method,
				 n_classes,
				 fraction_fit,
				 num_clients,
				 num_rounds,
				 args,
                 num_epochs,
				 model,
				 type,
				 decay=0,
				 perc_of_clients=0,
				 dataset='',
				 strategy_name='CDA-FedAvg',
				 model_name='',
				 new_clients=False,
				 new_clients_train=False
				 ):

		super().__init__(aggregation_method=aggregation_method,
						 n_classes=n_classes,
						 fraction_fit=fraction_fit,
						 num_clients=num_clients,
						 num_rounds=num_rounds,
						 args=args,
						 num_epochs=num_epochs,
						 decay=decay,
						 perc_of_clients=perc_of_clients,
						 dataset=dataset,
						 strategy_name=strategy_name,
						 model_name=model_name,
						 new_clients=new_clients,
						 new_clients_train=new_clients_train,
						 type=type)

		self.model = model
		self.current_weights = None
		self.create_folder(strategy_name)

	def create_folder(self, strategy_name):

		directory = """{}_saved_weights/{}/""".format(strategy_name.lower(), self.model_name)
		if Path(directory).exists():
			shutil.rmtree(directory)
		for i in range(self.num_clients):
			Path("""{}_saved_weights/{}/{}/""".format(strategy_name.lower(), self.model_name, i)).mkdir(
				parents=True, exist_ok=True)
			pd.DataFrame({'current_round': [-1], 'classes_distribution': ["[]"], 'round_of_last_fit': [-1],
						  'drift_detected': ['False'], 'Q': [[]], 'acc_of_last_fit': [0], 'first_round': [-1]}).to_csv(
				"""{}_saved_weights/{}/{}/{}_train.csv""".format(strategy_name.lower(), self.model_name, i, i),
				index=False)

			pd.DataFrame({'current_round': [-1], 'classes_distribution': ["[]"],
						  'drift_detected': ['False'], 'round_of_last_evaluate': [-1],
						  'first_round': [-1],
						  'acc_of_last_evaluate': [0]}).to_csv(
				"""{}_saved_weights/{}/{}/{}_val.csv""".format(strategy_name.lower(), self.model_name, i, i),
				index=False)