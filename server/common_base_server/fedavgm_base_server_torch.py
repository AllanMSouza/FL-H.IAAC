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

class FedAvgMBaseServerTorch(FedAvgBaseServer):

    def __init__(self,
                 aggregation_method,
                 n_classes,
                 fraction_fit,
                 num_clients,
                 num_rounds,
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
                         num_epochs=num_epochs,
                         decay=decay,
                         perc_of_clients=perc_of_clients,
                         dataset=dataset,
                         strategy_name=strategy_name,
                         non_iid=non_iid,
                         model_name=model_name,
                         new_clients=new_clients,
                         new_clients_train=new_clients_train)

        self.server_learning_rate = server_learning_rate
        self.server_momentum = server_momentum
        self.momentum_vector = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.server_opt = (self.server_momentum != 0.0) or (
                self.server_learning_rate != 1.0)

        self.set_initial_parameters()


    @abstractmethod
    def set_initial_parameters(
            self
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        pass

    def aggregate_fit(self, server_round, results, failures):
        weights_results = []

        for _, fit_res in results:
            client_id = str(fit_res.metrics['cid'])

            if self.aggregation_method not in ['POC', 'FedLTA'] or int(server_round) <= 1:
                weights_results.append((fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))

            else:
                if client_id in self.selected_clients:
                    weights_results.append(
                        (fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))


        fedavg_result = aggregate(weights_results)
        # following convention described in
        # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        if self.server_opt:
            # You need to initialize the model
            # assert (
            #         self.initial_parameters is not None
            # ), "When using server-side optimization, model needs to be initialized."
            # if self.initial_parameters is None:
            #     self.initial_parameters = self.set_initial_parameters()
            initial_parameters_numpy = copy.deepcopy(self.server_model_parameters)

            # remember that updates are the opposite of gradients
            pseudo_gradient: NDArrays = [
                x - y
                for x, y in zip(
                    initial_parameters_numpy, fedavg_result
                )
            ]
            if self.server_momentum > 0.0:
                if server_round > 1:
                    assert (
                        self.momentum_vector
                    ), "Momentum should have been created on round 1."
                    self.momentum_vector = [
                        self.server_momentum * x + y
                        for x, y in zip(self.momentum_vector, pseudo_gradient)
                    ]
                else:
                    self.momentum_vector = pseudo_gradient

                # No nesterov for now
                pseudo_gradient = self.momentum_vector

            # SGD
            fedavg_result = [
                x - self.server_learning_rate * y
                for x, y in zip(initial_parameters_numpy, pseudo_gradient)
            ]
            # Update current weights
            self.initial_parameters = fedavg_result

        parameters_aggregated = ndarrays_to_parameters(fedavg_result)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}

        return parameters_aggregated, metrics_aggregated