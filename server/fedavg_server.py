import flwr as fl
import numpy as np
import math
import os
import time

from flwr.common import FitIns
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from server.server_base import ServerBase


class FedAvgServer(ServerBase):

    def __init__(self,
                 aggregation_method,
                 n_classes,
                 fraction_fit,
                 num_clients,
                 num_rounds,
                 decay=0,
                 perc_of_clients=0,
                 dataset='',
                 strategy_name='FedPer',
                 model_name=''):

        super().__init__(aggregation_method=aggregation_method,
                         n_classes=n_classes,
                         fraction_fit=fraction_fit,
                         num_clients=num_clients,
                         num_rounds=num_rounds,
                         decay=decay,
                         perc_of_clients=perc_of_clients,
                         dataset=dataset,
                         strategy_name='FedAVG',
                         model_name=model_name)