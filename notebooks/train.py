from statistics import mean
import torch.cuda as cuda

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import data.Loader as ld
import model.DeconvNet as md
import support.colors as colors

torch.cuda.init()
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

batch_size = 8
workers = 1

train_data = ld.get_data()
train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=workers)
net = md.DeconvNet()
# Parameters for Training
n_epochs = 25
learning_rate = 0.01
weight_decay = 0.0005
momentum = 0.9
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

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
        epoch_loss = mean(losses)
        print(f"Epoch {epoch+1}/{n_epochs}. Loss: {epoch_loss}.", flush=True)
        if epoch % 5 == 0:
            torch.save(net.state_dict(), PATH + str(epoch)+ ".pth")

if __name__ == "__main__":
    PATH = "/home/student/Documents/FLORA/notebooks/state_dict_working_adam"
    net.load_state_dict(torch.load(PATH + ".pth"))
    print("Training started...", flush=True)
    trainNet(net=net,
            train_data_loader=train_data_loader,
            criterion=criterion,
            optimizer=optimizer,
            n_epochs=n_epochs)
    print("Training finished.", flush=True)
    torch.save(net.state_dict(), PATH+".pth")
    print("Model saved..")
    img_no = 10
    img_no2 = 25
    inp = train_data_loader.dataset.__getitem__(img_no)[0].unsqueeze(0)
    inp2 = train_data_loader.dataset.__getitem__(img_no2)[0].unsqueeze(0)
    plt.imshow(train_data_loader.dataset.__getitem__(img_no)[0].permute(1, 2, 0))
    out = net(inp)
    out2 = net(inp2)
    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    print("After Argmax....")
    print(om)
    rgb = colors.decode_segmap(om)
    plt.imshow(rgb)
    plt.axis('off')
    plt.show()



