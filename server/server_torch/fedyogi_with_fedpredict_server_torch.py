from server.common_base_server import FedYogiWithFedPredictBaseServer
from pathlib import Path
import shutil
import copy

from typing import Callable, Dict, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

class FedYogiWithFedPredictServerTorch(FedYogiWithFedPredictBaseServer):

    def __init__(self,
                 aggregation_method,
                 n_classes,
                 fraction_fit,
                 num_clients,
                 num_rounds,
                 args,
                 num_epochs,
                 model,
                 decay=0,
                 perc_of_clients=0,
                 dataset='',
                 strategy_name='FedYogi_with_FedPredict',
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
                         model=model,
                         decay=decay,
                         perc_of_clients=perc_of_clients,
                         dataset=dataset,
                         strategy_name=strategy_name,
                         model_name=model_name,
                         new_clients=new_clients,
                         new_clients_train=new_clients_train,
                         type='torch')

    def set_initial_parameters(
            self
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        model_parameters = [i.detach().numpy() for i in self.model.parameters()]
        self.current_weights = copy.deepcopy(model_parameters)
        return self.current_weights