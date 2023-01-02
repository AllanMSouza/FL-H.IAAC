import flwr as fl
import numpy as np
import math
import os
import time
import csv
import random

from logging import WARNING
from flwr.common import FitIns
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log
from server.common_base_server import FedAvgBaseServer

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

from typing import Callable, Dict, Optional, Tuple

class FedOptBaseServer(FedAvgBaseServer):

	def __init__(self,
				 aggregation_method,
				 n_classes,
				 fraction_fit,
				 num_clients,
				 num_rounds,
                 num_epochs,
				 model,
				 decay=0,
				 perc_of_clients=0,
				 dataset='',
				 strategy_name='FedOpt',
				 model_name='',
				 new_clients=False,
				 eta: float = 1e-1,
				 eta_l: float = 1e-1,
				 beta_1: float = 0.0,
				 beta_2: float = 0.0,
				 tau: float = 1e-9,
				 new_clients_train=False
				 ):

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
						 new_clients_train=new_clients_train)

		self.model = model
		self.current_weights = None
		self.eta = eta
		self.eta_l = eta_l
		self.tau = tau
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.m_t: Optional[NDArrays] = None
		self.v_t: Optional[NDArrays] = None