
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out).to(device))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out).to(device))
        if activation == 'relu':
            layers.append(nn.ReLU().to(device))
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU().to(device))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout).to(device))
    return nn.Sequential(*layers).to(device)



