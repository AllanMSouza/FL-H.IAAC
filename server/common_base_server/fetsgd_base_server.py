import flwr as fl
import numpy as np
import math
import os
import time
import csv
import random
import copy
from utils.quantization.fetsgd import layers_sketching
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

from server.common_base_server.fedper_base_server import FedPerBaseServer

from pathlib import Path
import shutil

class FetchSGDBaseServer(FedPerBaseServer):

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
				 strategy_name='FedPAQ',
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

	def configure_evaluate(self, server_round, parameters, client_manager):
		client_evaluate_list = super().configure_evaluate(server_round, parameters, client_manager)
		client_evaluate_list_fedpredict = []
		accuracy = 0
		size_of_parameters = []
		parameters = fl.common.parameters_to_ndarrays(parameters)
		for i in range(1, len(parameters)):
			size_of_parameters.append(parameters[i].nbytes)
		for client_tuple in client_evaluate_list:
			client = client_tuple[0]
			client_id = str(client.cid)
			config = copy.copy(self.evaluate_config)
			config['total_server_rounds'] = self.num_rounds
			try:
				config['total_server_rounds'] = int(self.comment)
			except:
				pass

			parameters_to_send = ndarrays_to_parameters(parameters)
			if server_round >= 1:
				parameters_to_send = ndarrays_to_parameters(layers_sketching(parameters, 10))
			evaluate_ins = fl.common.EvaluateIns(parameters_to_send, config)
			client_evaluate_list_fedpredict.append((client, evaluate_ins))

		return client_evaluate_list_fedpredict

