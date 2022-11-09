from functools import reduce

import flwr as fl
import numpy as np
import math
import os
import time

from flwr.common import FitIns
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from server.server_base import ServerBase


class FedProtoServer(ServerBase):

    def __init__(self, algorithm, n_classes, fraction_fit, num_clients,
                 decay=0, perc_of_clients=0, dataset='', strategy_name='FedProto', model_name=''):

        super().__init__(algorithm=algorithm,
                         n_classes=n_classes,
                         fraction_fit=fraction_fit,
                         num_clients=num_clients,
                         decay=decay,
                         perc_of_clients=perc_of_clients,
                         dataset=dataset,
                         strategy_name='FedProto',
                         model_name=model_name)

    # def aggregate_fit(self, server_round, results, failures):
    #     weights_results = []
    #     print("tamanho results: ", len(results))
    #     for _, fit_res in results:
    #         client_id = str(fit_res.metrics['cid'])
    #
    #         if self.algorithm not in ['POC', 'FedLTA'] or int(server_round) <= 1:
    #             weights_results.append((fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))
    #
    #         else:
    #             if client_id in self.selected_clients:
    #                 weights_results.append((fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))
    #
    #     # print(f'LEN AGGREGATED PARAMETERS: {len(weights_results)}')
    #     parameters_aggregated = fl.common.ndarrays_to_parameters(self._aggregate_proto(weights_results))
    #
    #     # Aggregate custom metrics if aggregation fn was provided
    #     metrics_aggregated = {}
    #
    #     return parameters_aggregated, metrics_aggregated
    #
    # def _aggregate_proto(self, results):
    #     """Compute weighted average."""
    #     # Calculate the total number of examples used during training
    #     #num_examples_total = sum([num_examples for _, num_examples in results])
    #
    #     # Create a list of weights, each multiplied by the related number of examples
    #     weighted_weights = [
    #         [layer * num_examples for layer in weights] for weights, num_examples in results
    #     ]
    #     weighted_proto = {i: [] for i in range(self.n_classes)}
    #     total_class = {i: 0 for i in range(self.n_classes)}
    #     for i in range(len(results)):
    #         proto, num_samples = results[i]
    #         print("resultados: ", proto)
    #         for key in proto:
    #             proto[i][key] = proto[i][key] * num_samples[key]
    #             total_class[key] += num_samples[key]
    #
    #         results[i] = [proto, num_samples]
    #
    #     global_proto = {i: [] for i in range(self.n_classes)}
    #     for key in range(self.n_classes):
    #
    #         for proto, num_samples in results:
    #             global_proto[key].append(proto[key])
    #
    #         print("proto global: ", global_proto[key])
    #         global_proto[key] = reduce(np.add, global_proto[key]) / total_class[key]
    #
    #     return global_proto