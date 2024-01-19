import numpy as np
import sys
from server.common_base_server import CDAFedAvgBaseServerWithFedPredict

import torch
from dataset_utils_torch import ManageDatasets
from torch.utils.data import TensorDataset, DataLoader

class CDAFedAvgWithFedPredictServerTorch(CDAFedAvgBaseServerWithFedPredict):

    def __init__(self,
                 aggregation_method,
                 n_classes,
                 fraction_fit,
                 num_clients,
                 num_rounds,
                 args,
                 num_epochs,
                 model=None,
                 decay=0,
                 perc_of_clients=0,
                 dataset='',
                 strategy_name='CDA-FedAvg_with_FedPredict',
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
                         perc_of_clients=perc_of_clients,
                         dataset=dataset,
                         strategy_name=strategy_name,
                         model_name=model_name,
                         new_clients=new_clients,
                         new_clients_train=new_clients_train,
                         type='torch')