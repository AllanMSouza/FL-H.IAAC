# -*- coding: utf-8 -*-
import time
from tqdm import tqdm
from optparse import OptionParser

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from Model_MobileNet import MobileNet
from Model_ResNet import resnet20

parser = OptionParser()
parser.add_option("-e", "--epochs",  dest="local_epochs", default=1,             help="Number times that the learning algorithm will work through the entire training dataset", metavar="INT")
parser.add_option("-b", "--batch",   dest="batch_size",   default=32,            help="Number of samples processed before the model is updated", metavar="INT")
parser.add_option("-m", "--model",   dest="model_name",   default='MOBILE_NET',  help="Model used for trainning", metavar="STR")
parser.add_option("-d", "--dataset", dest="dataset",      default='CIFAR10',     help="Dataset used for trainning", metavar="STR")
parser.add_option("-o", "--output",  dest="output_file",  default='results-center.txt', help="Save the results in a file", metavar="STR")
(opt, args) = parser.parse_args()

print("+-------------------------------+")
print("|    Central Training Start!    |")
print("+-------------------------------+")

results = ""
results = results + "Training Parameters(Central)\n"
results = results + "model_name " + str(opt.model_name) + "\n"
results = results + "dataset " + str(opt.dataset) + "\n"
results = results + "local_epochs " + str(opt.local_epochs) + "\n"
results = results + "batch_size " + str(opt.batch_size) + "\n\n"

root_path = './data/cifar10_data'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Data Load
if opt.dataset == "MNIST":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    trainset = torchvision.datasets.CIFAR10(root=root_path, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=int(opt.batch_size), shuffle=True)
    testset = torchvision.datasets.CIFAR10(root=root_path, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=int(opt.batch_size), shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
else:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    trainset = torchvision.datasets.CIFAR10(root=root_path, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=int(opt.batch_size), shuffle=True)
    testset = torchvision.datasets.CIFAR10(root=root_path, train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=int(opt.batch_size), shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Model Definition
if opt.model_name == "MOBILE_NET":
    model = MobileNet()
else:
    model = resnet20()
model.to(device)

lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
epochs = int(opt.local_epochs)

#Metrics
train_accu = []
train_losses = []

running_loss = 0
correct = 0
total = 0
#Metrics

#Train
start_time = time.time()
for epoch in range(epochs):
    for i, data in enumerate(tqdm(train_loader, ncols=100, desc="Epoch "+str(epoch+1))):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.clone().detach().long().to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    accu = 100. * correct / total

    train_accu.append(accu)
    train_losses.append(train_loss)
    results = results + "EPOCH-"+str(epoch)+"-Acc " + str(accu) + "\n"
    results = results + "EPOCH-"+str(epoch)+"-Loss " + str(train_loss) + "\n"
    print('Train Loss: %.3f | Accuracy: %.3f' % (train_loss, accu))

end_time = time.time()
print("Training Time: {} sec".format(end_time - start_time))
results = results + "Training-Time " + str(end_time - start_time) + "\n"

#Accuracy of train and each of classes
#Train Acc
with torch.no_grad():
    corr_num = 0
    total_num = 0
    train_loss = 0.0
    for j, trn in enumerate(train_loader):
        trn_x, trn_label = trn
        trn_x = trn_x.to(device)
        trn_label = trn_label.clone().detach().long().to(device)

        trn_output = model(trn_x)
        loss = criterion(trn_output, trn_label)
        train_loss += loss.item()
        model_label = trn_output.argmax(dim=1)
        corr = trn_label[trn_label == model_label].size(0)
        corr_num += corr
        total_num += trn_label.size(0)
    print("train_acc: {:.2f}%, train_loss: {:.4f}".format(corr_num / total_num * 100, train_loss / len(train_loader)))
    results = results + "EVALUATE-Train-Acc " + str((corr_num / total_num * 100)) + "\n"
    results = results + "EVALUATE-Train-Loss " + str((train_loss / len(train_loader))) + "\n"


#Test Acc
with torch.no_grad():
    corr_num = 0
    total_num = 0
    val_loss = 0.0
    for j, val in enumerate(test_loader):
        val_x, val_label = val
        val_x = val_x.to(device)
        val_label = val_label.clone().detach().long().to(device)

        val_output = model(val_x)
        loss = criterion(val_output, val_label)
        val_loss += loss.item()
        model_label = val_output.argmax(dim=1)
        corr = val_label[val_label == model_label].size(0)
        corr_num += corr
        total_num += val_label.size(0)
        accuracy = corr_num / total_num * 100
        test_loss = val_loss / len(test_loader)
    print("test_acc: {:.2f}%, test_loss: {:.4f}".format(accuracy, test_loss))
    results = results + "EVALUATE-Test-Acc " + str(accuracy) + "\n"
    results = results + "EVALUATE-Test-Loss " + str(test_loss) + "\n"

#Acc of each class
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in test_loader:
        x, labels = data
        x = x.to(device)
        labels = labels.to(device)

        outputs = model(x)
        labels = labels.long()
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
    results = results + "CLASS-"+str(classes[i])+"-Acc " + str(100 * class_correct[i] / class_total[i]) + "\n"

with open(opt.output_file, 'a+') as f:
    f.write(results)