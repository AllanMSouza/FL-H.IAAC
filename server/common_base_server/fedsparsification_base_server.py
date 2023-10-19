import flwr as fl
import numpy as np
import math
import os
import time
import csv
import copy
import random

from logging import WARNING
from flwr.common import FitIns
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from server.common_base_server import FedAvgBaseServer

from pathlib import Path
import shutil

class FedSparsificationBaseServer(FedAvgBaseServer):

	def __init__(self,
				 aggregation_method,
				 n_classes,
				 fraction_fit,
				 num_clients,
				 num_rounds,
				 args,
                 num_epochs,
				 type,
				 decay=0,
				 perc_of_clients=0,
				 dataset='',
				 strategy_name='FedSparsification',
				 non_iid=False,
				 model_name='',
				 new_clients=False,
				 new_clients_train=False):

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

	def configure_fit(self, server_round, parameters, client_manager):

		client_evaluate_list_sparsification = []
		client_fit_list = super().configure_fit(server_round, parameters, client_manager)
		for client_tuple in client_fit_list:
			client = client_tuple[0]
			fit_ins = client_tuple[1]
			client_config = fit_ins.config
			evaluate_ins = EvaluateIns(parameters, client_config)
			client_evaluate_list_sparsification.append((client, evaluate_ins))





