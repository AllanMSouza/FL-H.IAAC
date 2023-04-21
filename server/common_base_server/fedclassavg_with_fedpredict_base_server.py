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

from server.common_base_server import FedPredictBaseServer, FedClassAvgBaseServer

from pathlib import Path
import shutil

class FedClassAvg_with_FedPredictBaseServer(FedPredictBaseServer):

	def __init__(self,
				 aggregation_method,
				 n_classes,
				 fraction_fit,
				 num_clients,
				 num_rounds,
                 num_epochs,
				 model,
				 type,
				 server_learning_rate=1,
				 server_momentum=1,
				 decay=0,
				 perc_of_clients=0,
				 dataset='',
				 strategy_name='FedClassAvg_with_FedPredict',
				 non_iid=False,
				 model_name='',
				 new_clients=False,
				 new_clients_train=False):

		super().__init__(aggregation_method=aggregation_method,
						 n_classes=n_classes,
						 fraction_fit=fraction_fit,
						 num_clients=num_clients,
						 num_rounds=num_rounds,
						 num_epochs=num_epochs,
						 decay=decay,
						 perc_of_clients=perc_of_clients,
						 dataset=dataset,
						 strategy_name=strategy_name,
						 model_name=model_name,
						 new_clients=new_clients,
						 new_clients_train=new_clients_train,
						 type=type,
						 model=model)