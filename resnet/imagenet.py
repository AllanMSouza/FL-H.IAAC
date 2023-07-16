import torch, os
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader
import time
import sys
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# import warnings
import warnings
# filter warnings
from Model_ResNet import resnet20
warnings.filterwarnings('ignore')

class CNN(nn.Module):
    def __init__(self, input_shape=1, mid_dim=256, num_classes=10):
        try:
            super(CNN, self).__init__()
            # In the init function, we define each layer we will use in our model

            # Our images are RGB, so we have input channels = 3.
            # We will apply 12 filters in the first convolutional layer
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)

            # A second convolutional layer takes 12 input channels, and generates 24 outputs
            self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)

            # We in the end apply max pooling with a kernel size of 2
            self.pool = nn.MaxPool2d(kernel_size=2)

            # A drop layer deletes 20% of the features to help prevent overfitting
            self.drop = nn.Dropout2d(p=0.5)

            # Our 128x128 image tensors will be pooled twice with a kernel size of 2. 128/2/2 is 32.
            # This means that our feature tensors are now 32 x 32, and we've generated 24 of them

            # We need to flatten these in order to feed them to a fully-connected layer
            self.fc = nn.Linear(in_features=6144, out_features=num_classes)
        except Exception as e:
            print("CNN")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)

    def forward(self, x):
        try:
            # In the forward function, pass the data through the layers we defined in the init function

            # Use a ReLU activation function after layer 1 (convolution 1 and pool)
            x = F.relu(self.pool(self.conv1(x)))

            # Use a ReLU activation function after layer 2
            x = F.relu(self.pool(self.conv2(x)))

            # Select some features to drop to prevent overfitting (only drop during training)
            x = F.dropout(self.drop(x), training=self.training)

            # Flatten
            x = torch.flatten(x, 1)
            # Feed to fully-connected layer to predict class
            x = self.fc(x)
            # Return class probabilities via a log_softmax function
            return x
        except Exception as e:
            print("CNN forward")
            print('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), type(e).__name__, e)


def train(model, device, train_loader, optimizer, epoch):
    # Set the model to training mode
    model.train()
    train_loss = 0
    print("Epoch:", epoch)
    # Process the images in batches
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # Use the CPU or GPU as appropriate
        # Recall that GPU is optimized for the operations we are dealing with
        data, target = data.to(device), target.to(device)

        # Reset the optimizer
        optimizer.zero_grad()

        # Push the data forward through the model layers
        output = model(data)

        # Get the loss
        loss = loss_criteria(output, target)

        # Keep a running total
        train_loss += loss.item()

        # Backpropagate
        loss.backward()
        optimizer.step()

        # Print metrics so we see some progress
        _, predicted = torch.max(output.data, 1)
        correct += torch.sum(target == predicted).item()
        if batch_idx % 100 == 0:
            print('\tTraining batch {} Loss: {}'.format(batch_idx + 1, loss.item()))

    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx + 1)
    acc = 100. * correct / len(train_loader.dataset)
    print('Training set: Average loss: {}, acc: {}'.format(avg_loss, acc))
    return avg_loss, acc


def test(model, device, test_loader):
    # Switch the model to evaluation mode (so we don't backpropagate or drop)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        batch_count = 0
        for data, target in test_loader:
            batch_count += 1
            data, target = data.to(device), target.to(device)

            # Get the predicted classes for this batch
            output = model(data)

            # Calculate the loss for this batch
            test_loss += loss_criteria(output, target).item()

            # Calculate the accuracy for this batch
            _, predicted = torch.max(output.data, 1)
            correct += torch.sum(target == predicted).item()

    # Calculate the average loss and total accuracy for this epoch
    avg_loss = test_loss / batch_count
    print('Validation set: Average loss: {}, Accuracy: {}/{} ({})\n'.format(
        avg_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # return average loss for the epoch
    return avg_loss, 100. * correct / len(test_loader.dataset)

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


def load_dataset(data_path):
    import torch
    import torchvision
    import torchvision.transforms as transforms
    # Load all the images
    transformation = transforms.Compose([
        # Randomly augment the image data
        # Random horizontal flip
        transforms.RandomHorizontalFlip(0.5),
        # Random vertical flip
        transforms.RandomVerticalFlip(0.3),
        # transform to tensors
        transforms.ToTensor(),
        # Normalize the pixel values (in R, G, and B channels)
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load all of the images, transforming them
    full_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transformation
    )

    # Split into training (70% and testing (30%) datasets)
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # use torch.utils.data.random_split for training/test split
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # define a loader for the training data we can iterate through in 50-image batches
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=256,
        num_workers=0,
        shuffle=True
    )

    # define a loader for the testing data we can iterate through in 50-image batches
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=256,
        num_workers=0,
        shuffle=False
    )

    return train_loader, test_loader


data_dir = '/home/claudio/Documentos/pycharm_projects/FL-H.IAAC/dataset_utils/data/Tiny-ImageNet/raw_data/tiny-imagenet-200'
# data_dir = '/home/claudio/FL-H.IAAC/dataset_utils/data/Tiny-ImageNet/raw_data/tiny-imagenet-200'

loss_ft = nn.CrossEntropyLoss()
trainloader, testloader = load_dataset(data_dir)

print("train: ", len(trainloader.dataset))
print("test: ", len(testloader.dataset))
# exit()

#Load Resnet18
# model = CNN(input_shape=3, mid_dim=400, num_classes=200)
model = resnet20(num_classes=200)
#Finetune Final few layers to adjust for tiny imagenet input
# model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
# model_ft.fc = torch.nn.Linear(in_features=512, out_features=200, bias=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Use an "Adam" optimizer to adjust weights
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Specify the loss criteria
loss_criteria = nn.CrossEntropyLoss()

# Track metrics in these arrays
epoch_nums = []
training_loss = []
validation_loss = []

# Train over 10 epochs (We restrict to 10 for time issues)
epochs = 2
print('Training on', device)
train_acc_list = []
test_acc_list = []
for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, device, trainloader, optimizer, epoch)
        test_loss, test_acc = test(model, device, testloader)
        epoch_nums.append(epoch)
        training_loss.append(train_loss)
        validation_loss.append(test_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

for i in range(len(training_loss)):
    print("train loss : ", training_loss[i], " test loss: ", validation_loss[i], " train acc: ", train_acc_list[i], " test acc: ", test_acc_list[i])

plt.figure(figsize=(15,15))
plt.plot(epoch_nums, training_loss)
plt.plot(epoch_nums, validation_loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.show()

# Defining Labels and Predictions
truelabels = []
predictions = []
model.eval()
print("Getting predictions from test set...")
for data, target in testloader:
    for label in target.data.numpy():
        truelabels.append(label)
    for prediction in model(data).data.numpy().argmax(1):
        predictions.append(prediction)


# print("Dispositivo: ", device)
# model_ft = model_ft.to(device)
# #Loss Function
# criterion = nn.CrossEntropyLoss().to(device)
# # Observe that all parameters are being optimized
# optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)
# train_loss = 0
# train_acc = 0
# # train_model7("48",model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, num_epochs=10)
# train_num = 0
# log_interval = 10
# for step in range(1):
#     start_time = time.process_time()
#     for i, (x, y) in enumerate(trainloader):
#         if type(x) == type([]):
#             x[0] = x[0].to(device)
#         else:
#             x = x.to(device)
#         y = y.to(device)
#         train_num += y.shape[0]
#
#         optimizer_ft.zero_grad()
#         output = model_ft(x)
#         # y = torch.tensor(y.int().detach())
#         loss = loss_ft(output, y)
#         train_loss += loss.item() * y.shape[0]
#         loss.backward()
#         optimizer_ft.step()
#
#         train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
#         # train_acc += torch.sum(output == y.data)
#
#         if i % log_interval == 0:
#             total_time = time.process_time() - start_time
#             print('Train Epoch: {} [{}]\tLoss: {:.6f}\t Acc: {}'.format(
#                 step, (i+1) * len(x), loss.item(), train_acc/train_num))
#             print("Duração: ", total_time)
#             start_time = time.process_time()
#
# avg_loss_train = train_loss / train_num
# avg_acc_train = train_acc / train_num
#
# print("Acc: ", train_acc, " loss: ", avg_loss_train)
#
