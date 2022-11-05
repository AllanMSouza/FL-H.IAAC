import flwr as fl
import numpy as np
import math
import os
import time

from flwr.common import FitIns
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from server.server import Server


class FedProtoServer(Server):

    def __init__(self, algorithm, fraction_fit, num_clients,
                 decay=0, perc_of_clients=0, dataset='', strategy_name='Proto', model_name=''):

        super().__init__(algorithm=algorithm,
                                     fraction_fit=fraction_fit,
                                     num_clients=num_clients,
                                     decay=decay,
                                     perc_of_clients=perc_of_clients,
                                     dataset=dataset,
                                     strategy_name='FedProto',
                                     model_name=model_name)