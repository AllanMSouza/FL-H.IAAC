import math
import torch
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor
import copy
import random
import numpy as np
import sys
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class DNN_proto(nn.Module):
    def __init__(self, input_shape=1 * 28 * 28, mid_dim=100, num_classes=10):
        try:
            super(DNN_proto, self).__init__()

            self.fc0 = nn.Linear(input_shape, mid_dim)
            self.fc = nn.Linear(mid_dim, num_classes)
        except Exception as e:
            print("DNN_proto_2")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


    def forward(self, x):
        try:
            x = torch.flatten(x, 1)
            rep = F.relu(self.fc0(x))
            x = self.fc(rep)
            output = F.log_softmax(x, dim=1)
            return output, rep
        except Exception as e:
            print("DNN_proto_2 forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

# ====================================================================================================================
# class DNN(nn.Module):
#     def __init__(self, input_shape=1*28*28, mid_dim=100, num_classes=10):
#         super(DNN, self).__init__()
#
#         self.fc1 = nn.Linear(input_shape, mid_dim)
#         self.fc2 = nn.Linear(mid_dim, 50)
#         self.fc = nn.Linear(50, num_classes)
#     def forward(self, x):
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc(x)
#         x = F.log_softmax(x, dim=1)
#         return x
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
# ====================================================================================================================
class DNN_student(nn.Module):
    def __init__(self, input_shape=1*28*28, mid_dim=100, num_classes=10):
        super(DNN_student, self).__init__()

        self.fc1 = nn.Linear(input_shape, mid_dim)
        self.fc = nn.Linear(mid_dim, num_classes)
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x
# ====================================================================================================================

class DNN_teacher(nn.Module):
    def __init__(self, input_shape=1*28*28, mid_dim=100, num_classes=10):
        super(DNN_teacher, self).__init__()

        self.fc1 = nn.Linear(input_shape, mid_dim)
        self.fc2 = nn.Linear(mid_dim, 50)
        self.fc = nn.Linear(50, num_classes)
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        return x

    # ====================================================================================================================

class LeNet2(nn.Module):
    def __init__(self, num_classes):
      super(LeNet2, self).__init__()
      try:
          self.network = nn.Sequential(
              nn.Conv2d(3, 16, kernel_size=3, padding=1),
              nn.ReLU(),
              nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
              nn.ReLU(),
              nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16

              nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
              nn.ReLU(),
              # nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
              # nn.ReLU(),
              # nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

              # nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
              # nn.ReLU(),
              # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
              # nn.ReLU(),
              nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4

              nn.Flatten(),
              nn.Dropout(p=0.5),
              nn.Linear(4096, 64),
              nn.ReLU(),
              nn.Linear(64, 32),
              nn.ReLU(),
              nn.Linear(32, num_classes))
      except Exception as e:
          print("Lenet")
          print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
      try:
          out = self.network(x)
          return out
      except Exception as e:
          print("Lenet forward")
          print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    # ====================================================================================================================

class LeNet(nn.Module):
    def __init__(self, num_classes):
      super(LeNet, self).__init__()
      try:
          self.conv1 = nn.Conv2d(3, 6, 5)
          self.pool = nn.MaxPool2d(2, 2)
          self.conv2 = nn.Conv2d(6, 16, 5)
          self.fc1 = nn.Linear(16 * 5 * 5, 120)
          self.fc2 = nn.Linear(120, 84)
          self.fc3 = nn.Linear(84, num_classes)
      except Exception as e:
          print("Lenet")
          print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
      try:
          out = self.pool(F.relu(self.conv1(x)))
          out = self.pool(F.relu(self.conv2(out)))
          out = torch.flatten(out, 1)  # flatten all dimensions except batch
          out = F.relu(self.fc1(out))
          out = F.relu(self.fc2(out))
          out = self.fc3(out)
          return out
      except Exception as e:
          print("Lenet forward")
          print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


# ====================================================================================================================


class CNNDistillation(nn.Module):
    def __init__(self, input_shape=1, mid_dim=256, num_classes=10):
        try:
            super(CNNDistillation, self).__init__()
            self.student = CNN(input_shape=input_shape, mid_dim=mid_dim, num_classes=num_classes)
            self.teacher = CNN(input_shape=input_shape, mid_dim=mid_dim, num_classes=num_classes)
        except Exception as e:
            print("CNNDistillation")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            out_student = self.student(x)
            out_teacher = self.teacher(x)
            return out_student, out_teacher
        except Exception as e:
            print("CNNDistillation forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

# ====================================================================================================================

# melhor 3
class CNN_5(nn.Module):
    def __init__(self, input_shape=1, mid_dim=256, num_classes=10):
        try:
            super(CNN_5, self).__init__()
            self.conv1 = nn.Conv2d(input_shape, 16, 3, 1,
                                   padding=1)  # input is color image, hence 3 i/p channels. 16 filters, kernal size is tuned to 3 to avoid overfitting, stride is 1 , padding is 1 extract all edge features.
            self.conv2 = nn.Conv2d(16, 32, 3, 1,
                                   padding=1)  # We double the feature maps for every conv layer as in pratice it is really good.
            self.conv3 = nn.Conv2d(32, 64, 3, 1, padding=1)
            self.fc1 = nn.Linear(4 * 4 * 64,
                                 500)  # I/p image size is 32*32, after 3 MaxPooling layers it reduces to 4*4 and 64 because our last conv layer has 64 outputs. Output nodes is 500
            self.dropout1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(500, num_classes)
        except Exception as e:
            print("CNN 5")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            x = F.relu(self.conv1(x))  # Apply relu to each output of conv layer.
            x = F.max_pool2d(x, 2, 2)  # Max pooling layer with kernal of 2 and stride of 2
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4 * 4 * 64)  # flatten our images to 1D to input it to the fully connected layers
            x = F.relu(self.fc1(x))
            x = self.dropout1(
                x)  # Applying dropout b/t layers which exchange highest parameters. This is a good practice
            x = self.fc2(x)
            return x
        except Exception as e:
            print("CNN 5")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

# ====================================================================================================================
# melhor 3
class CNN(nn.Module):
    def __init__(self, input_shape=1, mid_dim=256, num_classes=10):
        try:
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_shape,
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
                nn.Linear(mid_dim*4, 512),
                nn.ReLU(inplace=True)
            )
            self.fc = nn.Linear(512, num_classes)
        except Exception as e:
            print("CNN")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            out = self.conv1(x)
            out = self.conv2(out)
            out = torch.flatten(out, 1)
            out = self.fc1(out)
            out = self.fc(out)
            return out
        except Exception as e:
            print("CNN forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

class CNN_student(nn.Module):
    def __init__(self, input_shape=1, mid_dim=256, num_classes=10):
        try:
            super(CNN_student, self).__init__()
            self.mid_dim = mid_dim
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_shape,
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
                nn.Linear(mid_dim * 4, 512),
                nn.ReLU(inplace=True)
            )
            self.fc = nn.Linear(512, num_classes)
        except Exception as e:
            print("CNN student")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            out = self.conv1(x)
            out = self.conv2(out)
            out = torch.flatten(out, 1)
            out = self.fc1(out)
            out = self.fc(out)
            return out
        except Exception as e:
            print("CNN student forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

# class CNN(nn.Module):
#     def __init__(self, input_shape=1, mid_dim=256, num_classes=10):
#         try:
#             super().__init__()
#             self.conv1 = nn.Conv2d(3, 6, 5)
#             self.pool = nn.MaxPool2d(2, 2)
#             self.conv2 = nn.Conv2d(6, 16, 5)
#             self.fc1 = nn.Linear(16 * 5 * 5, 120)
#             self.fc2 = nn.Linear(120, 84)
#             self.fc3 = nn.Linear(84, num_classes)
#         except Exception as e:
#             print("CNN")
#             print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)
#
#     def forward(self, x):
#         try:
#             x = self.pool(F.relu(self.conv1(x)))
#             x = self.pool(F.relu(self.conv2(x)))
#             x = torch.flatten(x, 1) # flatten all dimensions except batch
#             x = F.relu(self.fc1(x))
#             x = F.relu(self.fc2(x))
#             x = self.fc3(x)
#             return x
#         except Exception as e:
#             print("CNN forward")
#             print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

# ====================================================================================================================
class CNN_proto(nn.Module):
    def __init__(self, input_shape=1, mid_dim=256, num_classes=10):
        try:
            super(CNN_proto, self).__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_shape,
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
                nn.Linear(mid_dim * 4, 512),
                nn.ReLU(inplace=True)
            )
            self.fc = nn.Linear(512, num_classes)
        except Exception as e:
            print("CNN proto")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            out = self.conv1(x)
            out = self.conv2(out)
            out = torch.flatten(out, 1)
            rep = self.fc1(out)
            out = self.fc(rep)
            return out, rep
        except Exception as e:
            print("CNN proto forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

# ====================================================================================================================

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000, dropout=0.5):
        try:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, 1000),
            )
        except Exception as e:
            print("Alexnet")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x
        except Exception as e:
            print("Alexnet forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

# ====================================================================================================================

class Logistic(nn.Module):
    def __init__(self, input_shape=1 * 28 * 28, num_classes=10):
        super(Logistic, self).__init__()
        self.fc = nn.Linear(input_shape, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        out = F.log_softmax(x, dim=1)
        return out
# ====================================================================================================================

class Logistic_Proto(nn.Module):
    def __init__(self, input_dim=1 * 28 * 28, num_classes=10):
        super(Logistic_Proto, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        rep = self.fc(x)
        out = F.log_softmax(rep, dim=1)
        return out, rep
# ====================================================================================================================