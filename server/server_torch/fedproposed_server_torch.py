import flwr as fl
import numpy as np
import time
from pathlib import Path
import sys
from torch import nn

from flwr.common import FitIns
from flwr.server.strategy.aggregate import weighted_loss_avg
from model_definition_torch import ModelCreation, ProtoModel
from dataset_utils import ManageDatasets

from server.common_base_server import FedProposedBaseServer
import random
import torch
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
class FedProposedServerTorch(FedProposedBaseServer):

    def __init__(self,
                 aggregation_method,
                 n_classes,
                 fraction_fit,
                 num_clients,
                 num_rounds,
                 num_epochs,
                 decay=0,
                 perc_of_clients=0,
                 dataset='',
                 strategy_name='FedProposed',
                 model_name='',
                 new_clients=False,
                 new_clients_train=False):

        super().__init__(aggregation_method=aggregation_method,
                         n_classes=n_classes,
                         fraction_fit=fraction_fit,
                         num_clients=num_clients,
                         num_rounds=num_rounds,
                         num_epochs=num_epochs,
                         decay=decay,
                         perc_of_clients=perc_of_clients,
                         dataset=dataset,
                         strategy_name='FedProposed',
                         model_name=model_name,
                         new_clients=new_clients,
                         new_clients_train=new_clients_train)

        self.device = 'cpu'
        self.loss = nn.CrossEntropyLoss()
        self.learning_rate = 0.01
        self.optimizer = None

    def _create_proto_model(self):
        try:
            for key in self.protos_list:
                if len(self.protos_list[key]) > 0:
                    proto_dim = len(self.protos_list[key][0])
                    # proto_dim = 32
                    break

            print("dimensao: ", proto_dim)
            self.proto_model = ProtoModel(proto_dim, self.n_classes)
            self.optimizer = torch.optim.SGD(self.proto_model.parameters(), lr=self.learning_rate)

            x = []
            y = []
            for key in self.protos_list:

                proto = self.protos_list[key]
                x += proto
                y += [key] * len(proto)

            self.trainloader, self.testloader = ManageDatasets(0).create_torch_dataset_from_numpy(x=x, y=y, test_size=0.2, batch_size=32)
        except Exception as e:
            print("create model proposed")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


    def train_and_evaluate_proto_model(self):
        self._create_proto_model()
        self.train_proto_model()
        self.evaluate_proto_model()

    def train_proto_model(self):
        max_local_steps = 4
        train_acc = 0
        train_loss = 0
        train_num = 0
        for step in range(max_local_steps):
            for i, (x, y) in enumerate(self.trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                train_num += y.shape[0]

                self.optimizer.zero_grad()
                output = self.proto_model(x)
                y = torch.tensor(y.int().detach().numpy().astype(int).tolist())
                loss = self.loss(output, y)
                train_loss += loss.item() * y.shape[0]
                loss.backward()
                self.optimizer.step()

                train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

            loss = train_loss / train_num
            accuracy = train_acc / train_num

        print("************ Train of proto model ************")
        print("Accuracy: ", accuracy)
        print("Loss: ", loss)
        print("**********************************************")
        self.proto_parameters = self.get_parameters_of_model()

        return accuracy, loss

    def get_parameters_of_model(self):
        parameters = [i.detach().numpy() for i in self.proto_model.parameters()]
        return parameters

    def evaluate_proto_model(self):

        self.proto_model.eval()

        test_acc = 0
        test_loss = 0
        test_num = 0

        for x, y in self.testloader:
            print("ola: ", y)

        with torch.no_grad():
            for x, y in self.testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                self.optimizer.zero_grad()
                y = y.to(self.device)
                y = torch.tensor(y.int().detach().numpy().astype(int).tolist())
                output = self.proto_model(x)
                loss = self.loss(output, y)
                test_loss += loss.item() * y.shape[0]
                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]
                print("entrou")

        loss = test_loss / test_num
        accuracy = test_acc / test_num

        print("************ Evaluate of proto model ************")
        print("Accuracy: ", accuracy)
        print("Loss: ", loss)
        print("*************************************************")

        return accuracy, loss
