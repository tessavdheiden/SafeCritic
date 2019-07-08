import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import data.Loader as ld
import model.DeconvNet as md

batch_size = 4
workers = 1

train_data = ld.get_data()
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
net = md.DeconvNet()

#Parameters for Training
n_epochs = 3
learning_rate = 0.0001
momentum = 0.9
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, amsgrad=True)

def trainNet(net, train_data_loader, criterion, optimizer, n_epochs, learning_rate):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    loss_list = []
    acc_list = []
    
    for i, (input_img, output_img) in enumerate(train_data_loader):
        # Run the forward pass
        outputs = net(input_img)
        print(outputs.size(), output_img.size())
        loss = criterion(outputs, output_img)
        loss_list.append(loss.item())
        print("Forward Done")
        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = output_img.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == output_img).sum().item()
        acc_list.append(correct / total)

        #print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  #.format(n_epochs, i + 1, loss.item(),
                        #  (correct / total) * 100))
    print("Finished Training")

trainNet(net=net, train_data_loader=train_data_loader, criterion=criterion, optimizer=optimizer, n_epochs=n_epochs, learning_rate=learning_rate)