import sys
import time

import numpy as np
import torch
from torch.nn.parameter import Parameter
from client.client_torch import FedAvgClientTorch, ClientBaseTorch, FedPerClientTorch
from utils.compression_methods import inverse_parameter_quantization_reading, parameters_quantization_write


import warnings
warnings.simplefilter("ignore")

import logging
# logging.getLogger("torch").setLevel(logging.ERROR)

class FedPAQClientTorch(FedAvgClientTorch):

        def __init__(self,
                 cid,
                 n_clients,
                 n_classes,
                 args,
                 epochs=1,
                 model_name         = 'DNN',
                 client_selection   = False,
                 strategy_name      ='FedPAQ',
                 aggregation_method = 'None',
                 dataset            = '',
                 perc_of_clients    = 0,
                 decay              = 0,
                 fraction_fit       = 0,
                 non_iid            = False,
                 new_clients		= False,
                 new_clients_train  = False
                 ):

                super().__init__(cid=cid,
                         n_clients=n_clients,
                         n_classes=n_classes,
                         epochs=epochs,
                         model_name=model_name,
                         client_selection=client_selection,
                         aggregation_method=aggregation_method,
                         strategy_name=strategy_name,
                         dataset=dataset,
                         perc_of_clients=perc_of_clients,
                         decay=decay,
                         fraction_fit=fraction_fit,
                         non_iid=non_iid,
                         new_clients=new_clients,
                         new_clients_train=new_clients_train,
                         args=args)

                self.bits = args.bits

        # def fit(self, parameters, config):
        #     try:
        #         print("fit do fedpaq")
        #         trained_parameters, train_num, fit_response = super().fit(parameters, config)
        #         # print("trained parameters: ", type(trained_parameters[0]))
        #         # # trained_parameters = parameters_quantization_write(trained_parameters, self.bits)
        #         # print("retornando: ", len(trained_parameters))
        #         #
        #         return trained_parameters, train_num, fit_response
        #     except Exception as e:
        #         print("fit fedpaq")
        #         print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

        # def set_parameters_to_model_evaluate(self, global_parameters, config={}):
        #     # Using 'torch.load'
        #     try:
        #         print("Dimensões: ", [i.detach().numpy().shape for i in self.model.parameters()])
        #         print("Dimensões recebidas: ", [i.shape for i in global_parameters])
        #         global_parameters = inverse_parameter_quantization_reading(global_parameters,
        #                                                  [i.detach().numpy().shape for i in self.model.parameters()])
        #         parameters = [Parameter(torch.Tensor(i.tolist())) for i in global_parameters]
        #         for new_param, old_param in zip(parameters, self.model.parameters()):
        #             old_param.data = new_param.data.clone()
        #     except Exception as e:
        #         print("Set parameters to model")
        #         print('Error on line {} client id {}'.format(sys.exc_info()[-1].tb_lineno, self.cid), type(e).__name__,
        #               e)
