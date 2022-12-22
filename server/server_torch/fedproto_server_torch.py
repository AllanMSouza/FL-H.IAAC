import flwr as fl
import numpy as np
import time
from pathlib import Path

from flwr.common import FitIns
from flwr.server.strategy.aggregate import weighted_loss_avg

from server.common_base_server import FedProtoBaseServer
import random
import torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
class FedProtoServerTorch(FedProtoBaseServer):

    def __init__(self,
                 aggregation_method,
                 n_classes,
                 fraction_fit,
                 num_clients,
                 num_rounds,
                 decay=0,
                 perc_of_clients=0,
                 dataset='',
                 strategy_name='FedProto',
                 model_name='',
                 new_clients=False):

        super().__init__(aggregation_method=aggregation_method,
                         n_classes=n_classes,
                         fraction_fit=fraction_fit,
                         num_clients=num_clients,
                         num_rounds=num_rounds,
                         decay=decay,
                         perc_of_clients=perc_of_clients,
                         dataset=dataset,
                         strategy_name='FedProto',
                         model_name=model_name,
                         new_clients=new_clients)