from __future__ import print_function
import argparse
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

# Training settings used from PyTorch documentation
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # Define two convolutional layers, and three fully connected layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5,5), padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5,5))
        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pool with kernel size 2 on the convolutional layers
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2,2))

        # Use softmax over MSE for classification loss
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def get_train_valid_loader(random_seed, valid_size=0.1, shuffle=True):
    error_msg = '[!] valid_size should be in the range [0, 1].'
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # define transforms
    transform = transforms.Compose([transforms.ToTensor()])

    # load the dataset
    train_dataset = datasets.MNIST(
        root='../data', train=True, download=True, transform=transform)
    valid_dataset = datasets.MNIST(
        root='../data', train=True, download=True, transform=transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.test_batch_size, sampler=valid_sampler)

    return train_loader, valid_loader

def get_test_loader(shuffle=True):
    # define transform
    transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = datasets.MNIST(
        root='../data', train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=shuffle)

    return test_loader

def train(model, optimizer, epoch, loader):
    for batch_idx, (data, target) in enumerate(loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad() # Removes all past gradient run info (clean up)
        output = model(data) # Runs the model
        loss = F.nll_loss(output, target) # Computes the loss function (negative loss of the softmax)
        loss.backward() # Backprop
        optimizer.step() # Take a step in the right direction of the SGD

        # Logging
        length = len(loader) * args.batch_size
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), length,
                100. * batch_idx / length, loss.data[0])) # MAYBE TRY len(loader) here

def test(model, loader, name):
    test_loss = 0
    correct = 0
    for data, target in loader:
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data) # Runs the model
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # Sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # Get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    # Logging
    length = len(list(loader)) * args.test_batch_size
    test_loss /= length
    print('\n', name, 'set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, length,
        100. * correct / length))

    return test_loss

# Set manual seed to create reproduceable results
torch.manual_seed(args.seed)

# Create new LeNet model and stochastic gradient descent optimization algo
model = LeNet5()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# Load training, validation, and testing data
train_loader, valid_loader = get_train_valid_loader(args.seed)
test_loader = get_test_loader()

min_loss = sys.float_info.max

# Run epochs and save model with best performance on the validation set
for epoch in range(1, args.epochs + 1):
    train(model, optimizer, epoch, train_loader)
    loss = test(model, valid_loader, 'Validation')

    if loss < min_loss:
        min_loss = loss
        torch.save(model, 'best_model.pt')

# Use model with the lowest validation set loss on test data
best_model = torch.load('best_model.pt')
test(best_model, test_loader, 'Test')
