import copy
import sys

import numpy as np

from client.client_torch.client_base_torch import ClientBaseTorch
from pathlib import Path
import os
import sys
import torch

from scipy.sparse import coo_array, csr_matrix
import warnings
from torch.nn.parameter import Parameter
from utils.compression_methods.sparsification import calculate_bytes, sparse_bytes

warnings.simplefilter("ignore")

import logging
from utils.compression_methods.sparsification import sparse_crs_top_k, to_dense, sparse_matrix, get_not_zero_values

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

        self.filename = """./{}_saved_weights/{}/{}/model.pth""".format(strategy_name.lower(), self.model_name,
                                                                        self.cid)

    def set_parameters_to_model_fit(self, parameters):
        try:
            self.set_parameters_to_model(parameters)
        except Exception as e:
            print("set parameters to model train")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    # def set_parameters_to_model(self, parameters, config={}):
    #     try:
    #         parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
    #         for new_param, old_param in zip(parameters, self.model.parameters()):
    #             if new_param.shape[0] > 0:
    #                 old_param.data = new_param.data.clone()
    #     except Exception as e:
    #         print("set parameters to model")
    #         print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def fit(self, parameters, config):
        print("fit treino")
        # parameters = to_dense(config['parameters'])
        print([type(i) for i in parameters])
        print([i.shape for i in parameters])
        trained_parameters, train_num, fit_response = super().fit(parameters, config)
        updated_parameters = [np.abs(original - trained) for trained, original in zip(trained_parameters, parameters)]
        k = 0.5 # k = 0.6 já é ruim para EMNIST
        t, k_values = sparse_crs_top_k(updated_parameters, k)
        t = get_not_zero_values(updated_parameters, trained_parameters, k_values)
        print("valores k: ", k_values)
        for i in range(len(updated_parameters)):
            if False in np.equal(t[i], trained_parameters[i]):
                print("diferente")
        print("shape: ", [np.count_nonzero(i==0) for i in t])
        return t, train_num, fit_response

    def evaluate(self, parameters, config):

        # parameters = to_dense(config['parameters'])
        loss, test_num, evaluation_response = super().evaluate(parameters, config)
        return loss, test_num, evaluation_response

    def save_parameters(self):
        # Using 'torch.save'
        try:
            # filename = """./fedpredict_saved_weights/{}/{}/model.pth""".format(self.model_name, self.cid)
            if Path(self.filename).exists():
                os.remove(self.filename)
            torch.save(self.model.state_dict(), self.filename)
        except Exception as e:
            print("save parameters")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def set_parameters_to_model_evaluate(self, global_parameters, config={}):
        # Using 'torch.load'
        try:
            self.set_parameters_to_model(global_parameters)

        except Exception as e:
            print("Set parameters to model")
            print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, self.cid), type(e).__name__, e)



    def calculate_bytes(self, parameters):

        try:

            size = 0

            for p in parameters:

                sparse = sparse_matrix(p)
                print("Tamanho original: ", p.nbytes)
                b = sparse_bytes(sparse)
                print("Apos esparcificacao: ", b)
                b = min(p.nbytes, b)
                size += b
            return size

        except Exception as e:
            print("calculate bytes")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

