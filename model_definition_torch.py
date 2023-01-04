import math
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor
import copy
import random
import numpy as np
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

batch_size = 16

class LocalModel(nn.Module):
    def     __init__(self, base, head):
        super(LocalModel, self).__init__()

        self.base = base
        self.head = head
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out
# ====================================================================================================================
class DNN_proto_4(nn.Module):
    def __init__(self, input_shape=1 * 28 * 28, mid_dim=100, num_classes=10):
        super(DNN_proto_4, self).__init__()

        self.fc0 = nn.Linear(input_shape, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        rep = F.relu(self.fc2(x))
        x = self.fc(rep)
        output = F.log_softmax(x, dim=1)
        return output, rep
# ====================================================================================================================
class DNN_proto_2(nn.Module):
    def __init__(self, input_shape=1 * 28 * 28, mid_dim=100, num_classes=10):
        super(DNN_proto_2, self).__init__()

        self.fc0 = nn.Linear(input_shape, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        rep = F.relu(self.fc0(x))
        x = self.fc(rep)
        output = F.log_softmax(x, dim=1)
        return output, rep
# ====================================================================================================================
class DNN(nn.Module):
    def __init__(self, input_shape=1*28*28, mid_dim=100, num_classes=10):
        super(DNN, self).__init__()

        self.fc1 = nn.Linear(input_shape, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
# ====================================================================================================================
class FedAvgCNN(nn.Module):
    def __init__(self, input_dim=1*28*28, mid_dim=100, num_classes=10):
        super(FedAvgCNN).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(mid_dim, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        rep = self.fc1(out)
        out = F.log_softmax(self.fc(rep))
        return out
# ====================================================================================================================
class FedAvgCNNProto(nn.Module):
    def __init__(self, input_dim=1*28*28, mid_dim=100, num_classes=10):
        super(FedAvgCNNProto).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim,
                        32,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(mid_dim, 512),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        rep = self.fc1(out)
        out = F.log_softmax(self.fc(rep))
        return out, rep
# ====================================================================================================================
class Logistic(nn.Module):
    def __init__(self, input_dim=1 * 28 * 28, num_classes=10):
        super(Logistic, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output
# ====================================================================================================================
class ProtoModel(nn.Module):
    def __init__(self, mid_dim=100, num_classes=10):
        super(ProtoModel, self).__init__()
        self.fc = nn.Linear(mid_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x