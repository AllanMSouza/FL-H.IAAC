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
class DNN_proto(nn.Module):
    def __init__(self, input_dim=1*28*28, mid_dim=100, num_classes=10):
        super(DNN_proto, self).__init__()

        self.fc0 = nn.Linear(input_dim, 512)
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
# class DNN_proto(nn.Module):
#     def __init__(self, input_dim=1*28*28, mid_dim=100, num_classes=10):
#         super(DNN_proto, self).__init__()
#
#         self.fc1 = nn.Linear(input_dim, mid_dim)
#         self.fc = nn.Linear(mid_dim, num_classes)
#
#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         rep = F.relu(self.fc1(x))
#         x = self.fc(rep)
#         output = F.log_softmax(x, dim=1)
#         return output, rep
# ====================================================================================================================

# # ====================================================================================================================
#
# class DNN(nn.Module):
#     def __init__(self, input_dim=1*28*28, mid_dim=100, num_classes=10):
#         super(DNN, self).__init__()
#
#         self.fc0 = nn.Linear(input_dim, 512)
#         self.fc1 = nn.Linear(512, 256)
#         self.fc2 = nn.Linear(256, mid_dim)
#         self.fc = nn.Linear(mid_dim, num_classes)
#
#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc0(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc(x)
#         x = F.log_softmax(x, dim=1)
#         return x
# ====================================================================================================================
class DNN(nn.Module):
    def __init__(self, input_dim=1*28*28, mid_dim=100, num_classes=10):
        super(DNN, self).__init__()

        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

# ====================================================================================================================
# ====================================================================================================================
class ModelCreation():

	def create_DNN(self, input_shape, num_classes, use_local_model=False):
		model = DNN(input_dim=input_shape, num_classes=num_classes)
		if use_local_model:
			# head = copy.deepcopy(model.fc)
			# model.fc = nn.Identity()
			# return LocalModel(model, head)
			model = DNN_proto(input_dim=input_shape, num_classes=num_classes)
		return model

# ====================================================================================================================
	def create_CNN(self, input_shape, num_classes, use_proto=False):

		pass
# ====================================================================================================================
	def create_LogisticRegression(self, input_shape, num_classes, use_proto=False):

		pass