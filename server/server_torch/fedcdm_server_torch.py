import numpy as np
import sys
from server.common_base_server import FedCDMBaseServer

import torch
from dataset_utils_torch import ManageDatasets
from torch.utils.data import TensorDataset, DataLoader

class FedCDMServerTorch(FedCDMBaseServer):

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
                 strategy_name='FedCDM',
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

    def load_data(self, dataset_name, n_clients, batch_size=32):
        try:
            x_test = []
            y_test = []
            x_train = np.array([])
            y_train = np.array([])
            i = 1
            x_trai, y_train_, x_test, y_test = ManageDatasets(i, self.model_name).select_dataset(dataset_name, n_clients,
                                                                                   self.non_iid)

            tensor_x_train = torch.Tensor(x_train)  # transform to torch tenso)r
            tensor_y_train = torch.Tensor(y_train.astype(int))

            print("tamanho: ", tensor_x_train.shape, tensor_y_train.shape)

            train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
            trainLoader = DataLoader(train_dataset, batch_size, drop_last=True, shuffle=True)

            # tensor_x_test = torch.Tensor(np.array(x_test))  # transform to torch tensor
            # tensor_y_test = torch.Tensor(np.array(y_test))
            #
            # test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
            # testLoader = DataLoader(test_dataset, batch_size, drop_last=True, shuffle=True)

            return trainLoader
        except Exception as e:
            print("load data")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)