import copy

import flwr as fl
from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import torch

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.server.client_manager import ClientManager

from server.common_base_server import FedAvgBaseServer
from abc import abstractmethod

class FedAvgMBaseServer(FedAvgBaseServer):

    def __init__(self,
                 aggregation_method,
                 n_classes,
                 fraction_fit,
                 num_clients,
                 num_rounds,
                 args,
                 num_epochs,
                 model,
                 strategy_name='FedAvgM',
                 server_momentum=1,
                 server_learning_rate=1,
                 decay=0,
                 perc_of_clients=0,
                 dataset='',
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
                         non_iid=non_iid,
                         model_name=model_name,
                         new_clients=new_clients,
                         new_clients_train=new_clients_train,
                         type='torch')

        self.initial_parameters = None

    # def initialize_parameters(
    #         self, client_manager: ClientManager
    # ) -> Optional[Parameters]:
    #     """Initialize global model parameters."""
    #     initial_parameters = self.initial_parameters
    #     self.initial_parameters = None  # Don't keep initial parameters in memory
    #     return initial_parameters