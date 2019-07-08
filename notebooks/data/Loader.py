import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms, utils

import torch.optim as optim

import data.ImageDataset as Im

def get_data():
    #Dataset
    ROOT_DIR = "/home/student/Documents/FLORA/notebooks/data/data_set/train/"
    BASE_TEXT = "aachen_"
    TRAIN_TEXT = "_000019_leftImg8bit.png"
    FINE_COLOR_TEXT = "_000019_gtFine_color.png"
    INSTANCE_ID_TEXT = "_000019_instanceIds.png"
    LABEL_ID_TEXT = "_000019_gtFine_labelIds.png"
    POLYGONS_JSON_TEXT = "_000019_gtFine_polygons.json"
    
    TRANSFORM_IMG = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
    ])
    
    X = []
    y = []
    
    for i in range(0, 174):
        num = "0"*(6-len(str(i))) + str(i)
        X.append(ROOT_DIR + BASE_TEXT + num + TRAIN_TEXT)
        y.append(ROOT_DIR + BASE_TEXT + num + LABEL_ID_TEXT)
    
    data_set = Im.ImageDataset(X=X, y=y, transform=TRANSFORM_IMG)
    return data_set
        
    