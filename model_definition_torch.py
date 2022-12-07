import math
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor
import copy

batch_size = 16

class LocalModel(nn.Module):
    def __init__(self, base, head):
        super(LocalModel, self).__init__()

        self.base = base
        self.head = head

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

	def create_DNN(self, input_shape, num_classes, use_proto=False):
		model = DNN(input_dim=input_shape, num_classes=num_classes)
		if use_proto:
			head = copy.deepcopy(model.fc)
			model.fc = nn.Identity()
			return LocalModel(model, head)
		return model

# ====================================================================================================================
	def create_CNN(self, input_shape, num_classes, use_proto=False):

		pass
# ====================================================================================================================
	def create_LogisticRegression(self, input_shape, num_classes, use_proto=False):

		pass