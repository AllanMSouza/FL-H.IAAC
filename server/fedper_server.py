import flwr as fl
import numpy as np
import math
import os
import time

from flwr.common import FitIns
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from server.server import Server


class FedPerServer(Server):

    def __init__(self, aggregation_method, fraction_fit, num_clients,
                 decay=0, perc_of_clients=0, dataset='', solution_name='', model_name=''):

        super().__init__(aggregation_method=aggregation_method,
                         fraction_fit=fraction_fit,
                         num_clients=num_clients,
                         decay=decay,
                         perc_of_clients=perc_of_clients,
                         dataset=dataset,
                         solution_name=solution_name,
                         model_name=model_name)

