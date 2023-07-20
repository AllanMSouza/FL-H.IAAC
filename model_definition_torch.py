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

model_codes = {
    'CNN_6': [64, 'M', 128, 'M', 'D'],
    'CNN_8': [64, 'M', 128, 'M', 'D', 256, 'M', 'D'],
    'CNN_10': [64, 'M', 128, 'M', 'D', 256, 'M', 512, 'M', 'D'],
    'model_2': [64, 'M', 128, 'M', 'D', 256, 256, 'M', 512, 512, 'M', 'D'],
    'model_3': [64, 64, 'M', 128, 128, 'M', 'D', 256, 256, 256, 'M', 512, 512, 512, 'M', 'D'],
    'model_4': [64, 64, 64, 64, 'M', 128, 128, 128, 128, 'M', 'D', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 'D']
}

classifier_in_out = {
    'EMNIST':{
            'CNN_6': [6272, 64],
            'CNN_8': [2304, 128],
            'CNN_10': [512, 256]
    },
    'CIFAR10':{
            'CNN_6': [8192, 64],
            'CNN_8': [4096, 128],
            'CNN_10': [512, 256]
    }
}

class CNN_EMNIST(nn.Module):
    def __init__(self, dataset, model_code, in_channels, out_dim, act='relu', use_bn=True, dropout=0.3):
        super(CNN_EMNIST, self).__init__()

        try:
            self.classifier_in_out = classifier_in_out[dataset][model_code]
            if act == 'relu':
                self.act = nn.ReLU()
            elif act == 'leakyrelu':
                self.act = nn.LeakyReLU()
            else:
                raise ValueError("Not a valid activation function")

            self.layers = self.make_layers(model_code, in_channels, use_bn, dropout)
            self.classifier = nn.Sequential(nn.Linear(self.classifier_in_out[0], self.classifier_in_out[1]),
                                            self.act,
                                            nn.Linear(self.classifier_in_out[1], out_dim)
                                            )

        except Exception as e:
            print("CNN EMNIST")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            x = self.layers(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            # skipped softmax siince cross-entropy loss is used
            return x
        except Exception as e:
            print("CNN EMNIST forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def make_layers(self, model_code, in_channels, use_bn, dropout):
        try:
            layers = []
            for x in model_codes[model_code]:
                if x == "M":
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                elif x == 'D':
                    layers += [nn.Dropout(dropout)]
                else:
                    layers += [nn.Conv2d(in_channels=in_channels,
                                         out_channels=x,
                                         kernel_size=3,
                                         stride=1,
                                         padding=1)]
                    if use_bn:
                        layers += [nn.BatchNorm2d(x)]

                    layers += [self.act]
                    in_channels = x

            return nn.Sequential(*layers)
        except Exception as e:
            print("CNN EMNIST make layers")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

# ====================================================================================================================

import torch.nn.init as init

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes=10, num_blocks=[3, 3, 3]):
    return ResNet(BasicBlock, num_blocks, num_classes=num_classes)

# ====================================================================================================================

def conv_dw(inplane, outplane, stride=1):
    return nn.Sequential(nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=stride, padding=1),
                         nn.BatchNorm2d(inplane), nn.ReLU(),
                         nn.Conv2d(inplane, outplane, kernel_size=1, groups=1, stride=1), nn.BatchNorm2d(outplane),
                         nn.ReLU())

def conv_bw(inplane, outplane, kernel_size=3, stride=1):
    return nn.Sequential(nn.Conv2d(inplane, outplane, kernel_size=kernel_size, groups=1, stride=stride, padding=1),
                         nn.BatchNorm2d(outplane), nn.ReLU())

class MobileNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        try:
            layers = []
            layers.append(conv_bw(3, 32, 3, 1))
            layers.append(conv_dw(32, 64, 1))
            layers.append(conv_dw(64, 128, 2))
            layers.append(conv_dw(128, 128, 1))
            layers.append(conv_dw(128, 256, 2))
            layers.append(conv_dw(256, 256, 1))
            layers.append(conv_dw(256, 512, 2))
            for i in range(5):
                layers.append(conv_dw(512, 512, 1))

            layers.append(conv_dw(512, 1024, 2))
            layers.append(conv_dw(1024, 1024, 1))
            self.classifer = nn.Sequential(nn.Dropout(0.5), nn.Linear(1024, num_classes))
            self.feature = nn.Sequential(*layers)
        except Exception as e:
            print("Mobilenet")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            out = self.feature(x)
            out = out.mean(3).mean(2)
            out = out.view(-1, 1024)
            out = self.classifer(out)
            return out
        except Exception as e:
            print("Mobilenet forward")
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
    def __init__(self, input_shape=1, mid_dim=256, num_classes=10, dataset='CIFAR10'):
        try:
            super(CNNDistillation, self).__init__()
            if dataset == "CIFAR10":
                self.student = CNN(input_shape=input_shape, mid_dim=mid_dim, num_classes=num_classes)
                self.teacher = CNN(input_shape=input_shape, mid_dim=mid_dim, num_classes=num_classes)
            elif dataset == "EMNIST":
                self.student = resnet20(num_classes, [3, 1, 1])
                self.teacher = resnet20(num_classes)
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