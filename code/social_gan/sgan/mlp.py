
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import skimage.transform
from PIL import Image
from datasets.calculate_static_scene_boundaries import get_pixels_from_world


def make_mlp(dim_list, activation='relu', batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out).cuda())
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)



