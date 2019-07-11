import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

from PIL import Image

import torch

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim

import cv2

import os

class ImageDataset(Dataset):

    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        
        input_img = plt.imread(self.X[idx])
        target_img = cv2.imread(self.y[idx], cv2.IMREAD_GRAYSCALE)

        input_img = cv2.resize(input_img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        target_img = cv2.resize(target_img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

        input_img = torch.from_numpy(input_img).to(torch.float32)
        target_img[target_img >= 30] = 0
        target_img = torch.from_numpy((target_img)).to(torch.long)
        input_img = input_img.permute(2,0,1)

        data = input_img, target_img
        return data