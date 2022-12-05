import math
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor

batch_size = 16

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
class LocalModel(nn.Module):
	def __init__(self, base, head):
		super(LocalModel, self).__init__()

		self.base = base
		self.head = head

	def forward(self, x):
		out = self.base(x)
		out = self.head(out)

		return out


# ====================================================================================================================
class ModelCreation():

	def create_DNN(self, input_shape, num_classes, use_proto=False):
		return DNN(input_dim=input_shape, num_classes=num_classes)

# ====================================================================================================================
	def create_CNN(self, input_shape, num_classes, use_proto=False):

		pass
# ====================================================================================================================
	def create_LogisticRegression(self, input_shape, num_classes, use_proto=False):

		pass