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

from server.common_base_server import FedAvgMBaseServerTorch


class FedAvgMServerTorch(FedAvgMBaseServerTorch):

    def __init__(self,
                 aggregation_method,
                 n_classes,
                 fraction_fit,
                 num_clients,
                 num_rounds,
                 num_epochs,
                 model,
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
                         num_epochs=num_epochs,
                         model=model,
                         decay=decay,
                         perc_of_clients=perc_of_clients,
                         dataset=dataset,
                         strategy_name='FedAvgM',
                         non_iid=non_iid,
                         model_name=model_name,
                         new_clients=new_clients,
                         new_clients_train=new_clients_train)

    def set_initial_parameters(
            self
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        model_parameters = [i.detach().numpy() for i in self.model.parameters()]
        self.server_model_parameters = copy.deepcopy(model_parameters)
        return model_parameters