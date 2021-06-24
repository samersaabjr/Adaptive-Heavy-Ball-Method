# -*- coding: utf-8 -*-
"""
This Script contains the default MNIST code for comparison.
The code is collected from:
    nextjournal.com/gkoehler/pytorch-mnist
The same code can also be used for KMNIST, QMNIST and FashionMNIST.
torchvision.datasets.MNIST needs to be changed to  
torchvision.datasets.FashionMNIST for FashionMNIST simulations
"""

import numpy as np
import time

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from AHB import AHB

#%% Script Options

n_epochs = 200
batch_size_train = 64
batch_size_test = 1000
log_interval = 100

num_seeds = 5
method = 'ahb'

device = torch.device('cuda:0')

torch.backends.cudnn.enabled = False

#%% Load Data

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.QMNIST('', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.QMNIST('', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

print(example_data.shape)

# Plot sample images

# fig = plt.figure()
# for i in range(6):
#   plt.subplot(2,3,i+1)
#   plt.tight_layout()
#   plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
#   plt.title("Ground Truth: {}".format(example_targets[i]))
#   plt.xticks([])
#   plt.yticks([])
# fig


#%% Define Network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
#%% Run Network    

test_acc_all = np.zeros((num_seeds,n_epochs))

time_tracker = np.zeros((num_seeds,n_epochs))


for r in range(num_seeds):
    
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(r)
    
    network = Net()
    network = network.to(device)
    
    if method == 'momentum':
        lr = 0.01
        optimizer = optim.SGD(network.parameters(), lr=lr,
                              momentum=0.9)
    elif method == 'nag':
        wd = 0
        lr = 0.01 # 0.001
        optimizer = optim.SGD(network.parameters(), lr=lr,
                              momentum=0.9, weight_decay=wd, nesterov=True)
    elif method== 'adam':
        wd = 0
        lr = 0.0005 # 0.001
        optimizer = optim.Adam(network.parameters(), lr=lr, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=wd)
    elif method== 'adamw':
        wd = 1
        lr = 0.0005
        optimizer = optim.AdamW(network.parameters(), lr=lr, betas=(0.9, 0.999),
                                     eps=1e-08, weight_decay=wd)
    elif method == 'ahb':
        optimizer = AHB(network.parameters())
    
    train_losses = []
    train_acc = []
    train_counter = []
    
    test_losses = []
    test_acc = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
    
    
    def train(epoch):
      network.train()
      correct = 0
      for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        if method == 'MADAGRAD':
            optimizer.step(epoch+1)
        else:
            optimizer.step()
        if batch_idx % log_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
          train_losses.append(loss.item())
          train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      train_accuracy = 100. * correct / len(train_loader.dataset)
      train_acc.append(train_accuracy.item())
    
          
    def test():
      network.eval()
      test_loss = 0
      correct = 0
      with torch.no_grad():
        for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          output = network(data)
          test_loss += F.nll_loss(output, target, size_average=False).item()
          pred = output.data.max(1, keepdim=True)[1]
          correct += pred.eq(target.data.view_as(pred)).sum()
          test_loss /= len(test_loader.dataset)
          test_losses.append(test_loss)
        test_accuracy = 100. * correct / len(test_loader.dataset)
        test_acc.append(test_accuracy.item())
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_accuracy))
      
      
    # test()

    fname_test_acc = method + '_QMNIST_testacc_rs'+str(r)+'.npy'
    fname_test_loss = method + '_QMNIST_testloss_rs'+str(r)+'.npy'
    fname_train_acc = method + '_QMNIST_trainacc_rs'+str(r)+'.npy'
    fname_train_loss = method + '_QMNIST_trainloss_rs'+str(r)+'.npy'
    
    for epoch in range(1, n_epochs + 1):
      start = time.time()
      train(epoch)
      end = time.time()
      time_elapsed = end-start
      print(time_elapsed)
      time_tracker[r,epoch-1] = time_elapsed
      test()
    
    np.save(fname_test_acc, test_acc)
    np.save(fname_test_loss, test_losses)
    np.save(fname_train_acc, train_acc)
    np.save(fname_train_loss, train_losses)

    test_acc_all[r,:] = test_acc
