from server.common_base_server import FedClassAvg_with_FedPredictBaseServer
from server.server_torch import FedPredictServerTorch
from pathlib import Path
import shutil
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
import copy

class FedClassAvg_with_FedPredictServerTorch(FedPredictServerTorch):

    def __init__(self,
                 aggregation_method,
                 n_classes,
                 fraction_fit,
                 num_clients,
                 num_rounds,
                 args,
                 num_epochs,
                 model,
                 server_learning_rate=1,
                 server_momentum=1,
                 decay=0,
                 perc_of_clients=0,
                 dataset='',
                 strategy_name='FedClassAvg_with_FedPredict',
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
                         model=model,
                         server_learning_rate=server_learning_rate,
                         server_momentum=server_momentum,
                         perc_of_clients=perc_of_clients,
                         dataset=dataset,
                         strategy_name=strategy_name,
                         model_name=model_name,
                         new_clients=new_clients,
                         new_clients_train=new_clients_train)