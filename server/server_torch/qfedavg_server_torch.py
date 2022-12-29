import copy
from collections import OrderedDict
import argparse
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
from torch.utils.data import DataLoader
import torch

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg, aggregate_qffl
import numpy as np

from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.server.client_manager import ClientManager

from server.server_torch.fedavg_server_torch import FedAvgServerTorch


class QFedAvgServerTorch(FedAvgServerTorch):

    def __init__(self,
                 aggregation_method,
                 n_classes,
                 fraction_fit,
                 num_clients,
                 num_rounds,
                 model,
                 q_param=0.2,
                 server_momentum=1,
                 server_learning_rate=1,
                 decay=0,
                 perc_of_clients=0,
                 dataset='',
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
                         strategy_name='QFedAvg',
                         model_name=model_name,
                         new_clients=new_clients)

        self.server_learning_rate = server_learning_rate
        self.server_momentum = server_momentum
        self.momentum_vector = None
        self.model = model
        self.learning_rate = server_learning_rate
        self.q_param = q_param
        self.pre_weights: Optional[NDArrays] = None
        self.evaluate_fn = self.get_evaluate_fn(self.model)

        self.set_initial_parameters()

    def set_initial_parameters(
            self
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        model_parameters = [i.detach().numpy() for i in self.model.parameters()]
        self.server_model_parameters = copy.deepcopy(model_parameters)
        return model_parameters

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

        def norm_grad(grad_list: NDArrays) -> float:
            # input: nested gradients
            # output: square of the L-2 norm
            client_grads = grad_list[0]
            for i in range(1, len(grad_list)):
                client_grads = np.append(  # type: ignore
                    client_grads, grad_list[i]
                )  # output a flattened array
            squared = np.square(client_grads)
            summed = np.sum(squared)
            return float(summed)

        deltas = []
        hs_ffl = []

        if self.pre_weights is None:
            raise Exception("QffedAvg pre_weights are None in aggregate_fit")

        weights_before = self.pre_weights
        eval_result = self.evaluate(
            server_round, ndarrays_to_parameters(weights_before)
        )
        if eval_result is not None:
            loss, _ = eval_result

        print("valida: ", eval_result)

        for new_weights, num_examples in weights_results:
            # plug in the weight updates into the gradient
            grads = [
                np.multiply((u - v), 1.0 / self.learning_rate)
                for u, v in zip(weights_before, new_weights)
            ]
            deltas.append(
                [np.float_power(loss + 1e-10, self.q_param) * grad for grad in grads]
            )
            # estimation of the local Lipschitz constant
            hs_ffl.append(
                self.q_param
                * np.float_power(loss + 1e-10, (self.q_param - 1))
                * norm_grad(grads)
                + (1.0 / self.learning_rate)
                * np.float_power(loss + 1e-10, self.q_param)
            )

        weights_aggregated: NDArrays = aggregate_qffl(weights_before, deltas, hs_ffl)
        parameters_aggregated = ndarrays_to_parameters(weights_aggregated)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}

        return parameters_aggregated, metrics_aggregated

    def get_evaluate_fn(self, model: torch.nn.Module):
        """Return an evaluation function for server-side evaluation."""

        # Load data and model here to avoid the overhead of doing it in `evaluate` itself
        trainset = self.load_data(self.dataset, self.num_clients, 32)

        n_train = len(trainset)
        # if toy:
        #     # use only 10 samples as validation set
        #     valset = torch.utils.data.Subset(trainset, range(n_train - 10, n_train))
        # else:
        #     # Use the last 5k training examples as a validation set
        # valset = torch.utils.data.Subset(trainset, range(n_train - 5000, n_train))

        # valLoader = DataLoader(valset, batch_size=16)

        # The `evaluate` function will be called after every round
        def evaluate(
                server_round: int,
                parameters: fl.common.NDArrays,
                config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            # Update model with the latest parameters
            params_dict = zip(model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            loss, accuracy = self.test(model, trainset)
            return loss, {"accuracy": accuracy}

        return evaluate