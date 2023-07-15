import torch, os
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from train_model import train_model
from test_model import test_model
from torch.utils.data import TensorDataset, DataLoader
import time


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def dataset(data_path):
    """Load ImageNet (training and val set)."""

    # Load ImageNet and normalize
    traindir = os.path.join(data_path, "train")
    valdir = os.path.join(data_path, "val")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    trainLoader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    testLoader = DataLoader(dataset=val_dataset, batch_size=256, shuffle=False)

    return trainLoader, testLoader


data_dir = '/home/claudio/Documentos/pycharm_projects/FL-H.IAAC/dataset_utils/data/Tiny-ImageNet/raw_data/tiny-imagenet-200'
# data_dir = '/home/claudio/FL-H.IAAC/dataset_utils/data/Tiny-ImageNet/raw_data/tiny-imagenet-200'
# data_dir = '/home/claudiocapanema/Documentos/FL-H.IAAC/dataset_utils/data/Tiny-ImageNet/raw_data/tiny-imagenet-200'

loss_ft = nn.CrossEntropyLoss()
trainloader, testloader = dataset(data_dir)

#Load Resnet18
model_ft = models.resnet18(pretrained=True)
#Finetune Final few layers to adjust for tiny imagenet input
# model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
model_ft.fc = torch.nn.Linear(in_features=512, out_features=200, bias=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Dispositivo: ", device)
model_ft = model_ft.to(device)
#Loss Function
criterion = nn.CrossEntropyLoss().to(device)
# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.01)
train_loss = 0
train_acc = 0
# train_model7("48",model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, num_epochs=10)
train_num = 0
log_interval = 10
for step in range(1):
    start_time = time.process_time()
    for i, (x, y) in enumerate(trainloader):
        if type(x) == type([]):
            x[0] = x[0].to(device)
        else:
            x = x.to(device)
        y = y.to(device)
        train_num += y.shape[0]

        optimizer_ft.zero_grad()
        output = model_ft(x)
        # y = torch.tensor(y.int().detach())
        loss = loss_ft(output, y)
        train_loss += loss.item() * y.shape[0]
        loss.backward()
        optimizer_ft.step()

        train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
        # train_acc += torch.sum(output == y)

        if i % log_interval == 0:
            total_time = time.process_time() - start_time
            print('Train Epoch: {} [{}]\tLoss: {}\t Acc: {}'.format(
                step, (i+1) * len(x), train_loss / train_num, train_acc / train_num))
            print("Duração: ", total_time)
            start_time = time.process_time()

avg_loss_train = train_loss / train_num
avg_acc_train = train_acc / train_num

print("Acc: ", train_acc, " loss: ", avg_loss_train)

