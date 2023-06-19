import sys

import numpy as np
import torch

from client.client_torch.client_base_torch import ClientBaseTorch
from utils.quantization import min_max_quantization, min_max_dequantization


import warnings
warnings.simplefilter("ignore")

import logging
# logging.getLogger("torch").setLevel(logging.ERROR)

class FedPAQClientTorch(ClientBaseTorch):

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

                self.bits = args.bits


        def fit(self, parameters, config):
            try:
                results = []
                trained_parameters, train_num, fit_response = super().fit(parameters, config)
                for original_layer, layer_updated in zip(parameters, trained_parameters):
                #     results.append(QSGDCompressor(5).compress(torch.from_numpy(layer)))
                # print("com ", trained_parameters[0].shape, results[0])
                    if np.ndim(original_layer) >= 2:
                        results.append(min_max_quantization(original_layer-layer_updated, self.bits))
                    else:
                        results.append(layer_updated)

                return  trained_parameters, train_num, fit_response

            except Exception as e:
                print("fit")
                print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

        def evaluate(self, parameters, config):
            try:

                for i in range(len(parameters)):

                    layer = parameters[i]
                    if np.ndim(layer) >= 2:
                        parameters[i] = min_max_dequantization(parameters[i])

                return super().evaluate(parameters, config)

            except Exception as e:
                print("evaluate fedpaq")
                print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
