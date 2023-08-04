import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

def conv_dw(inplane, outplane, stride=1):
    return nn.Sequential(nn.Conv2d(inplane, inplane, kernel_size=3, groups=inplane, stride=stride, padding=1), nn.BatchNorm2d(inplane), nn.ReLU(), nn.Conv2d(inplane, outplane, kernel_size=1, groups=1, stride=1), nn.BatchNorm2d(outplane), nn.ReLU())

def conv_bw(inplane, outplane, kernel_size=3, stride=1):
    return nn.Sequential(nn.Conv2d(inplane, outplane, kernel_size=kernel_size, groups=1, stride=stride, padding=1), nn.BatchNorm2d(outplane), nn.ReLU())

# Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
# borrowed from Pytorch quickstart example
class Net(nn.Module):
    def __init__(self, num_class=10):
        super(Net, self).__init__()
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
        self.classifer = nn.Sequential(nn.Dropout(0.5), nn.Linear(1024, num_class))
        self.feature = nn.Sequential(*layers)

    def forward(self, x):
        out = self.feature(x)
        out = out.mean(3).mean(2)
        out = out.view(-1, 1024)
        out = self.classifer(out)
        return out

model_codes = {
    'CNN_6': [64, 'M', 128, 'M', 'D'],
    'CNN_8': [64, 'M', 128, 'M', 'D', 256, 'M', 'D'],
    'CNN_10': [64, 'M', 128, 'M', 'D', 256, 'M', 512, 'M', 'D'],
    'CNN_12': [64, 'M', 128, 'M', 'D', 256, 256, 'M', 512, 512, 'M', 'D'],
    'model_3': [64, 64, 'M', 128, 128, 'M', 'D', 256, 256, 256, 'M', 512, 512, 512, 'M', 'D'],
    'model_4': [64, 64, 64, 64, 'M', 128, 128, 128, 128, 'M', 'D', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 'D']
}

classifier_in_out = {
    'EMNIST':{
            'CNN_6': [6272, 64],
            'CNN_8': [2304, 128],
            'CNN_10': [512, 256],
            'CNN_12': [512, 256]
    },
    'CIFAR10':{
            'CNN_6': [8192, 64],
            'CNN_8': [4096, 128],
            'CNN_10': [2048, 256],
            'CNN_12': [2048, 256]
    }
}

class CNN_EMNIST(nn.Module):
    def __init__(self, dataset, model_code, in_channels, out_dim, act='relu', use_bn=False, dropout=0.3):
        super(CNN_EMNIST, self).__init__()

        try:
            self.classifier_in_out = classifier_in_out[dataset][model_code]
            self.dataset = dataset
            if act == 'relu':
                self.act = nn.ReLU(inplace=True)
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
                                         padding=1,
                                         bias=True)]
                    if use_bn:
                        layers += [nn.BatchNorm2d(x)]

                    layers += [self.act]
                    in_channels = x

            return nn.Sequential(*layers)
        except Exception as e:
            print("CNN EMNIST make layers")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)





# borrowed from Pytorch quickstart example
def train(net, trainloader, epochs, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


# borrowed from Pytorch quickstart example
def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
