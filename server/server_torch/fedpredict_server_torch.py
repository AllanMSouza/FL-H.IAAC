from server.common_base_server import FedPredictBaseServer
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

from torch.nn.parameter import Parameter

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from typing import Callable, Dict, List, Optional, Tuple, Union
import copy

class FedPredictServerTorch(FedPredictBaseServer):

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
                 strategy_name='FedPredict',
                 model_name='',
                 new_clients=False,
                 new_clients_train=False):

        super().__init__(aggregation_method=aggregation_method,
                         n_classes=n_classes,
                         fraction_fit=fraction_fit,
                         num_clients=num_clients,
                         num_rounds=num_rounds,
                         num_epochs=num_epochs,
                         args=args,
                         decay=decay,
                         model=model,
                         server_learning_rate=server_learning_rate,
                         server_momentum=server_momentum,
                         perc_of_clients=perc_of_clients,
                         dataset=dataset,
                         strategy_name=strategy_name,
                         model_name=model_name,
                         new_clients=new_clients,
                         new_clients_train=new_clients_train,
                         type='torch')

        self.model_shape = [i.shape for i in self.model.parameters()]
        print("formato: ", self.model_shape)


    def set_initial_parameters(
            self
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        model_parameters = [i.detach().cpu().numpy() for i in self.model.parameters()]
        self.server_model_parameters = copy.deepcopy(model_parameters)
        return model_parameters

    # def set_parameters_to_model(self, parameters, model):
    #     try:
    #         parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
    #         for new_param, old_param in zip(parameters, model.parameters()):
    #             old_param.data = new_param.data.clone()
    #         return model
    #     except Exception as e:
    #         print("set parameters to model")
    #         print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    # def layerwise_similarity(self, global_parameter, clients_parameters):
    #
    #     global_model = copy.deepcopy(self.model)
    #
    #     global_model = self.set_parameters_to_model(global_parameter, global_model)
    #
    #     for i in range(len(clients_parameters)):
    #         client_parameter = clients_parameters[i]
    #         local_model = copy.deepcopy(self.model)
    #         local_model = self.set_parameters_to_model(client_parameter, local_model)
    #         clients_parameters[i] = copy.deepcopy(local_model)