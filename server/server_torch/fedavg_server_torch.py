import numpy as np
from server.common_base_server import FedAvgBaseServer

import torch
from dataset_utils import ManageDatasets
from torch.utils.data import TensorDataset, DataLoader

class FedAvgServerTorch(FedAvgBaseServer):

    def __init__(self,
                 aggregation_method,
                 n_classes,
                 fraction_fit,
                 num_clients,
                 num_rounds,
                 decay=0,
                 perc_of_clients=0,
                 dataset='',
                 strategy_name='FedAvg',
                 model_name='',
                 new_clients=False):

        super().__init__(aggregation_method=aggregation_method,
                         n_classes=n_classes,
                         fraction_fit=fraction_fit,
                         num_clients=num_clients,
                         num_rounds=num_rounds,
                         decay=decay,
                         perc_of_clients=perc_of_clients,
                         dataset=dataset,
                         strategy_name='FedAVG',
                         model_name=model_name,
                         new_clients=new_clients)

    def load_data(self, dataset_name, n_clients, batch_size=32):
        try:
            x_test = []
            y_test = []
            x_train = []
            y_train = []
            for i in range(10):
                x_train_, y_train_, x_test_, y_test_ = ManageDatasets(i).select_dataset(dataset_name, n_clients,
                                                                                       self.non_iid)
                x_train += x_train_
                y_train += y_train_

            tensor_x_train = torch.Tensor(np.array(x_train))  # transform to torch tenso)r
            tensor_y_train = torch.Tensor(np.array(y_train))

            train_dataset = TensorDataset(tensor_x_train, tensor_y_train)
            trainLoader = DataLoader(train_dataset, batch_size, drop_last=True, shuffle=True)

            # tensor_x_test = torch.Tensor(np.array(x_test))  # transform to torch tensor
            # tensor_y_test = torch.Tensor(np.array(y_test))
            #
            # test_dataset = TensorDataset(tensor_x_test, tensor_y_test)
            # testLoader = DataLoader(test_dataset, batch_size, drop_last=True, shuffle=True)

            return trainLoader, None
        except Exception as e:
            print("load data")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def test(net, model, testloader, steps: int = None, device: str = "cpu"):
        """Validate the network on the entire test set."""
        print("Starting evalutation...")
        net.to(device)  # move model to GPU if available
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        net.eval()
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                if steps is not None and batch_idx == steps:
                    break
        if steps is None:
            loss /= len(testloader.dataset)
        else:
            loss /= total
        accuracy = correct / total
        net.to("cpu")  # move model back to CPU
        return loss, accuracy