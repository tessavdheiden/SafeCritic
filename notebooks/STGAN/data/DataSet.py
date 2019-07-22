import matplotlib.pyplot as plt

import torch

from torch.utils.data import Dataset

import cv2


class DataSet(Dataset):

    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        input_img = cv2.imread(self.X[idx])
        real_img = cv2.imread(self.y[idx])

        input_img = cv2.resize(input_img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        real_img = cv2.resize(real_img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

        input_img = torch.from_numpy(input_img).to(torch.float32)
        real_img = torch.from_numpy(real_img).to(torch.float32)
        input_img = input_img.permute(2, 0, 1)
        real_img = real_img.permute(2, 0, 1)

        data_set = input_img, real_img
        return data_set