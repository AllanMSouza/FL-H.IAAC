import sys
from client.client_torch.client_base_torch import ClientBaseTorch

import warnings

warnings.simplefilter("ignore")

import logging
from utils.compression_methods.sparsification import sparse_crs_top_k, to_dense

# logging.getLogger("torch").setLevel(logging.ERROR)

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

class FedSparsificationClientTorch(ClientBaseTorch):

    def __init__(self,
                 cid,
                 n_clients,
                 n_classes,
                 args,
                 epochs=1,
                 model_name='DNN',
                 client_selection=False,
                 strategy_name='FedSparsification',
                 aggregation_method='None',
                 dataset='',
                 perc_of_clients=0,
                 decay=0,
                 fraction_fit=0,
                 non_iid=False,
                 new_clients	= False,
                 new_clients_train  = False
                 ):

        super().__init__(cid=cid,
                         n_clients=n_clients,
                         n_classes=n_classes,
                         epochs=epochs,
                         model_name=model_name,
                         client_selection=client_selection,
                         solution_name=strategy_name,
                         aggregation_method=aggregation_method,
                         dataset=dataset,
                         perc_of_clients=perc_of_clients,
                         decay=decay,
                         fraction_fit=fraction_fit,
                         non_iid=non_iid,
                         new_clients=new_clients,
                         new_clients_train=new_clients_train,
                         args=args)

    def set_parameters_to_model_fit(self, parameters):
        try:
            self.set_parameters_to_model(parameters)
        except Exception as e:
            print("set parameters to model train")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def fit(self, parameters, config):

        parameters = to_dense(parameters)
        trained_parameters, train_num, fit_response = super().fit(parameters, config)
        k = 0.1
        trained_parameters = sparse_crs_top_k(trained_parameters, k)
        return trained_parameters, train_num, fit_response

    def evaluate(self, parameters, config):

        parameters = to_dense(parameters)
        loss, test_num, evaluation_response = super().evaluate(parameters, config)
        return loss, test_num, evaluation_response
