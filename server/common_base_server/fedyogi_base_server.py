import flwr as fl
import numpy as np
import math
import os
import time
import csv
import random
from abc import ABC, abstractmethod

from logging import WARNING
from flwr.common import FitIns
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log
from typing import Callable, Dict, Optional, Tuple
from server.common_base_server.fedopt_base_server import FedOptBaseServer

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

class FedYogiBaseServer(FedOptBaseServer):

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
				 strategy_name='FedYogi',
				 model_name='',
				 new_clients=False,
				 eta: float = 1e-2,
				 eta_l: float = 0.0316,
				 beta_1: float = 0.9,
				 beta_2: float = 0.99,
				 tau: float = 1e-3,
				 new_clients_train=False
				 ):

		super().__init__(aggregation_method=aggregation_method,
							n_classes=n_classes,
							fraction_fit=fraction_fit,
							num_clients=num_clients,
							num_rounds=num_rounds,
						 	args=args,
						 	num_epochs=num_epochs,
						 	model=model,
							decay=decay,
							perc_of_clients=perc_of_clients,
							dataset=dataset,
							strategy_name=strategy_name,
							model_name=model_name,
							new_clients=new_clients,
							eta=eta,
							eta_l=eta_l,
							beta_1=beta_1,
							beta_2=beta_2,
							tau=tau,
						 new_clients_train=new_clients_train,
						 	type=type
							)

		self.set_initial_parameters()

	@abstractmethod
	def set_initial_parameters(
			self
	) -> Optional[Parameters]:
		"""Initialize global model parameters. It varies when it is using tf or torch"""
		pass

	def aggregate_fit(self, server_round, results, failures):
		fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
			server_round=server_round, results=results, failures=failures
		)
		if fedavg_parameters_aggregated is None:
			return None, {}

		fedavg_weights_aggregate = parameters_to_ndarrays(fedavg_parameters_aggregated)

		# Yogi
		delta_t: NDArrays = [
			x - y for x, y in zip(fedavg_weights_aggregate, self.current_weights)
		]

		# m_t
		if not self.m_t:
			self.m_t = [np.zeros_like(x) for x in delta_t]
		self.m_t = [
			np.multiply(self.beta_1, x) + (1 - self.beta_1) * y
			for x, y in zip(self.m_t, delta_t)
		]

		# v_t
		if not self.v_t:
			self.v_t = [np.zeros_like(x) for x in delta_t]
		self.v_t = [
			x - (1.0 - self.beta_2) * np.multiply(y, y) * np.sign(x - np.multiply(y, y))
			for x, y in zip(self.v_t, delta_t)
		]

		new_weights = [
			x + self.eta * y / (np.sqrt(z) + self.tau)
			for x, y, z in zip(self.current_weights, self.m_t, self.v_t)
		]

		self.current_weights = new_weights

		return ndarrays_to_parameters(self.current_weights), metrics_aggregated

