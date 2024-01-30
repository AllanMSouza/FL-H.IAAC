from dataset_utils_torch import ManageDatasets
from server.common_base_server import FedCDMWithFedPredictBaseServer
from torch.utils.data import TensorDataset, DataLoader
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
import torch.nn as nn

from torch.nn.parameter import Parameter

from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from typing import Callable, Dict, List, Optional, Tuple, Union
import sys
import copy
import time
import numpy as np
import pandas as pd


class FedCDMWithFedPredictServerTorch(FedCDMWithFedPredictBaseServer):

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
                 strategy_name='FedCDM_with_FedPredict',
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
        self.local_epochs = 1
        self.loss = nn.CrossEntropyLoss()
        print("formato: ", self.model_shape)
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        if self.dataset in ['EMNIST', 'CIFAR10', 'GTSRB']:
            self.learning_rate = 0.01
            # self.optimizer = torch.optim.Adam(self.model.parameters(),
            # 								  lr=self.learning_rate)
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.learning_rate)
        elif self.dataset == 'State Farm':
            # self.learning_rate = 0.01
            # self.optimizer = torch.optim.Adam(self.model.parameters(),
            # 								  lr=self.learning_rate)
            self.learning_rate = 0.01
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.dataset in ['ExtraSensory', 'WISDM-WATCH', 'WISDM-P', 'Cologne']:
            self.learning_rate = 0.001
            # self.loss = nn.MSELoss()
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate)

    def get_target_and_samples_from_dataset(self, traindataset, dataset_name):

        try:

            if dataset_name in ['WISDM-WATCH', 'WISDM-P', 'Cologne']:
                data = []
                targets = []
                for sample in traindataset:
                    # print("amostra: ", sample)
                    data.append(sample[0].numpy())
                    targets.append(int(sample[1]))
                data = np.array(data)
                print("dada: ", type(data), len(data), len(targets))
                targets = np.array(targets)
            else:
                targets = np.array(traindataset.targets)
                if dataset_name == 'GTSRB':
                    data = np.array(traindataset.samples)
                else:
                    data = np.array(traindataset.data)

            return data, targets

        except Exception as e:
            print("get target and samples from dataset")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def set_dataset(self, dataset, dataset_name, x, y):

        try:

            if dataset_name in ['WISDM-WATCH', 'WISDM-P', 'Cologne']:

                return torch.utils.data.TensorDataset(torch.from_numpy(x).to(dtype=torch.float32), torch.from_numpy(y))

            else:

                dataset.samples = x
                dataset.targets = y

                return dataset

        except Exception as e:
            print("set dataset")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def load_balanced_dataset(self, current_traindataset, batch_size, dataset_name, n_clients):

        try:

            L = 40000
            M = self.n_classes
            samples_per_class = int(L / (M))

            trainLoader, testLoader, current_traindataset, testdataset = ManageDatasets(0,
                                                                                self.model_name).select_dataset(
                dataset_name, n_clients, self.class_per_client, self.alpha, self.non_iid, batch_size)


            data_target = {i: [] for i in range(self.n_classes)}

            for pattern in range(self.num_clients):
                trainLoader, testLoader, traindataset, testdataset = ManageDatasets(pattern,
                                                                                    self.model_name).select_dataset(
                    dataset_name, n_clients, self.class_per_client, self.alpha, self.non_iid, batch_size)

                trainLoader = None
                testLoader = None
                testdataset = None

                for class_ in data_target:

                    current_size = len(data_target[class_])
                    missing_data = samples_per_class - current_size
                    if missing_data > 0:
                        data, targets = self.get_target_and_samples_from_dataset(traindataset, dataset_name)
                        indices = np.where(targets == class_)[0]
                        if len(indices) == 0:
                            continue
                        # print("ind: ", indices)
                        indices = np.random.choice(indices, size=missing_data)
                        targets = targets[indices]
                        data = data[indices].tolist()
                        data_target[class_] += data

                        print("balanced 3")

            l_old_samples = []
            l_old_targets = []

            for class_ in data_target:
                samples = list(data_target[class_])
                print("""quantidade classe {} e {}""".format(class_, len(samples)))
                targets = [class_] * len(samples)
                l_old_samples += samples
                l_old_targets += targets

            l_old_samples = np.array(l_old_samples)
            l_old_targets = np.array(l_old_targets)

            # if dataset_name == 'GTSRB':
            # 	current_samples = np.array(current_traindataset.samples)
            # else:
            # 	current_samples = np.array(current_traindataset.data)
            # current_targets = current_traindataset.targets
            current_samples, current_targets = self.get_target_and_samples_from_dataset(current_traindataset,
                                                                                        dataset_name)
            print("shapes: ", current_samples.shape, current_targets.shape, l_old_samples.shape, l_old_targets.shape)
            print("""antes juntar unique {} """.format(np.unique(current_targets, return_counts=True)))
            current_samples = np.concatenate((current_samples, l_old_samples), axis=0)
            current_targets = np.concatenate((current_targets, l_old_targets), axis=0)

            print("""juntou unique {}""".format(np.unique(current_targets, return_counts=True)))

            print("s t: ", current_samples.shape, current_targets.shape)

            current_traindataset = self.set_dataset(current_traindataset, dataset_name, current_samples,
                                                    current_targets)

            trainLoader = DataLoader(current_traindataset, batch_size, shuffle=True)

            self.traindataset = current_traindataset
            self.trainloader = trainLoader

            return current_traindataset, trainLoader


        except Exception as e:
            print("previous balanced dataset")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


    def set_initial_parameters(
            self
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        model_parameters = [i.detach().cpu().numpy() for i in self.model.parameters()]
        self.server_model_parameters = copy.deepcopy(model_parameters)
        return model_parameters

    def get_parameters_of_model(self):
        try:
            parameters = [i.detach().cpu().numpy() for i in self.model.parameters()]
            return parameters
        except Exception as e:
            print("get parameters of model")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def set_parameters_to_model(self, parameters, config={}):
        try:
            parameters = [Parameter(torch.Tensor(i.tolist())) for i in parameters]
            for new_param, old_param in zip(parameters, self.model.parameters()):
                old_param.data = new_param.data.clone()
        except Exception as e:
            print("set parameters to model")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def server_fit(self, parameters, server_round):
        try:
            selected_clients = []
            trained_parameters = []
            selected = 0
            print("Iniciar treinamento")

            start_time = time.process_time()

            # if self.dynamic_data != "no":
            #     self.trainloader, self.testloader, self.traindataset, self.testdataset = self.load_data(self.dataset,
            #                                                                                             n_clients=self.n_clients,
            #                                                                                             server_round=server_round,
            #                                                                                             train=True)
            original_parameters = copy.deepcopy(parameters)
            self.set_parameters_to_model(parameters)
            self.load_balanced_dataset(current_traindataset=None, batch_size=32, dataset_name=self.dataset, n_clients=self.num_clients)
            # self.save_parameters_global_model(parameters)
            self.round_of_last_fit = server_round

            selected = 1
            self.model.to(self.device)
            self.model.train()

            count = 0

            for param in self.model.parameters():
                if count < len([i for i in self.model.parameters()]) - 2:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                count += 1

            max_local_steps = self.local_epochs

            self.classes_proportion, self.imbalance_level = self._calculate_classes_proportion()

            predictions = []
            for step in range(max_local_steps):
                start_time = time.process_time()
                train_acc = 0
                train_loss = 0
                train_num = 0
                for i, (x, y) in enumerate(self.trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)

                    # if self.dataset == 'EMNIST':
                    # 	x = x.view(-1, 28 * 28)
                    y = np.array(y).astype(int)
                    # print("entrada: ", x.shape, y.shape, type(x[0]), type(y[0]), y[0])
                    # y = y.to(self.device)
                    train_num += y.shape[0]

                    self.optimizer.zero_grad()
                    output = self.model(x)
                    if len(predictions) == 0:
                        predictions = output.detach().numpy().tolist()
                    else:
                        predictions += output.detach().numpy().tolist()
                    y = torch.tensor(y)
                    loss = self.loss(output, y)
                    train_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()

                    train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    total_time = time.process_time() - start_time
                # print("Duração: ", total_time)
            # print("Completou, cliente: ", self.cid, " rodada: ", server_round)
            trained_parameters = self.get_parameters_of_model()
            # self.save_parameters()

            size_list = []
            # print("Tamanho total parametros fit: ", sum(size_list))
            size_of_parameters = sum(size_list)
            # size_of_parameters = sum(
            # 	[sum(map(sys.getsizeof, trained_parameters[i])) for i in range(len(trained_parameters))])
            avg_loss_train = train_loss / train_num
            avg_acc_train = train_acc / train_num
            total_time = time.process_time() - start_time
            # loss, accuracy, test_num = self.model_eval()

            data = [server_round, 1, selected, total_time, size_of_parameters, avg_loss_train, avg_acc_train]

            # self.save_client_information_fit(server_round, avg_acc_train, predictions)

            # self._write_output(
            #     filename=self.train_client_filename,
            #     data=data)

            fit_response = {
                'local_classes': self.classes_proportion,
            }

            return ndarrays_to_parameters(trained_parameters)
        except Exception as e:
            print("server fit")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def _calculate_classes_proportion(self):

        try:
            # return [1] * self.num_classes, 0
            correction = 3 if self.dataset == 'GTSRB' else 1
            traindataset = self.traindataset
            if self.dataset in ['WISDM-WATCH', 'WISDM-P', 'Cologne']:
                y_train = []
                for i, (x, y) in enumerate(self.trainloader):
                    y_train += np.array(y).astype(int).tolist()
            else:
                y_train = list(traindataset.targets)
            proportion = np.array([0] * self.n_classes)

            unique_classes_list = pd.Series(y_train).unique().tolist()

            for i in y_train:
                proportion[i] += 1

            proportion_ = proportion / np.sum(proportion)

            imbalance_level = 0
            min_samples_per_class = int(len(y_train) / correction / len(unique_classes_list))
            for class_ in proportion:
                if class_ < min_samples_per_class:
                    imbalance_level += 1

            imbalance_level = imbalance_level / len(proportion)

            return list(proportion_), imbalance_level

        except Exception as e:
            print("calculate classes proportion")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

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