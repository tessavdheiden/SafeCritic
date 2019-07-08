import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import data.Loader as ld
import model.DeconvNet as md
import support.colors as colors

batch_size = 4
workers = 1

train_data = ld.get_data()
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
net = md.DeconvNet()

#Parameters for Training
n_epochs = 1
learning_rate = 0.001
weight_decay = 0.0005
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

def trainNet(net, train_data_loader, criterion, optimizer, n_epochs, learning_rate):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("weight_decay=", weight_decay)
    print("=" * 30)
    
    loss_list = []
    acc_list = []
    
    for i, (input_img, output_img) in enumerate(train_data_loader):
        # Run the forward pass
        outputs = net(input_img)
        loss = criterion(outputs, output_img)
        print(loss)
        # Backprop and perform Adam optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track the accuracy
        total = output_img.size(0)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == output_img).sum().item()

        print(loss)
    print("Finished Training")

trainNet(net=net, train_data_loader=train_data_loader, criterion=criterion, optimizer=optimizer, n_epochs=n_epochs, learning_rate=learning_rate)

inp = train_data_loader.dataset.__getitem__(2)[0].unsqueeze(0)
out = net(inp)
print(out)
om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
rgb = colors.decode_segmap(om)
plt.imshow(rgb)
plt.axis('off')
plt.show()
