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
        target_img = plt.imread(self.y[idx])

        #target_img_instance_id_name = os.path.join(self.root_dir, self.cityscape_frames_Y.iloc[idx, 1])
        #target_img_label_id_name = os.path.join(self.root_dir, self.cityscape_frames_Y.iloc[idx, 2])
        #output_json_name = os.path.join(self.root_dir, self.cityscape_frames_Y.iloc[idx, 3])
            
        input_img = cv2.resize(input_img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        target_img = cv2.resize(target_img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        
        input_img = torch.from_numpy(input_img)
        target_img = torch.from_numpy((target_img*30).round()).to(torch.long)
        input_img = input_img.permute(2,0,1)
        #target_img = target_img.permute(2,0,1)

        #if target_img.shape[0] == 4:
            #convert the image from RGBA2RGB
        #    target_img = cv2.cvtColor(target_img, cv2.COLOR_RGBA2RGB)

        #if self.transform:
         #   input_img = self.transform(input_img)
        #    target_img = self.transform(target_img)

        data = input_img, target_img
        return data