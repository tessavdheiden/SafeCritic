import io
from statistics import mean

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import data.Loader as ld
import model.DeconvNet as md
import support.colors as colors

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight.data)

batch_size = 8
workers = 1

train_data = ld.get_data()
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
net = md.DeconvNet()
net.apply(weights_init)

# Parameters for Training
n_epochs = 2
learning_rate = 0.01
weight_decay = 0.0000
momentum = 0.9
criterion = nn.CrossEntropyLoss()
crit = nn.NLLLoss2d()
optimizer_adam = optim.Adam(net.parameters(), lr=learning_rate)
optimizer_sgd = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

# delete after we found bug
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

def trainNet(net, train_data_loader, criterion, optimizer, n_epochs):
    for epoch in range(n_epochs):
        losses = []
        for (input_img, target_img) in tqdm(train_data_loader):
            optimizer.zero_grad()
            outputs = net(input_img)
            loss = criterion(input=outputs, target=target_img)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            print(loss)
        epoch_loss = mean(losses)
        print(f"Epoch {epoch+1}/{n_epochs}. Loss: {epoch_loss}.", flush=True)

if __name__ == "__main__":
    PATH = "/home/student/Documents/FLORA/notebooks/state_dict.pth"
    net.load_state_dict(torch.load(PATH))
    print("Training started...", flush=True)
    trainNet(net=net,
             train_data_loader=train_data_loader,
             criterion=criterion,
             optimizer=optimizer_adam,
             n_epochs=n_epochs)
    print("Training finished.", flush=True)
    torch.save(net.state_dict(), PATH)
    print("Model saved..")
    img_no = 2
    inp = train_data_loader.dataset.__getitem__(img_no)[0].unsqueeze(0)
    plt.imshow(train_data_loader.dataset.__getitem__(img_no)[0].permute(1, 2, 0))
    out = net(inp)
    print("Output....")
    print(out)
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    print("After Argmax....")
    print(om)
    rgb = colors.decode_segmap(om)
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()



