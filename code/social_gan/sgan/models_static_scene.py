import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sgan.utils import get_dset_group_name

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def grey2bin(grey):
    grey[grey > 0.5] = 1
    grey[grey <= 0.5] = 0
    return grey


def load_bin_map(path_name):
    static_map = plt.imread(path_name)
    static_map = rgb2gray(static_map)
    static_map = grey2bin(static_map)
    return static_map


def get_homography_and_map(dset, annotated_image_name = '/annotated.jpg'):
    _dir = os.path.dirname(os.path.realpath(__file__))
    _dir = _dir.split("/")[:-1]
    _dir = "/".join(_dir)
    directory = _dir + '/datasets/safegan_dataset/'
    path_group = os.path.join(directory, get_dset_group_name(dset))
    path = os.path.join(path_group, dset)
    h_matrix = pd.read_csv(path + '/{}_homography.txt'.format(dset), delim_whitespace=True, header=None).values
    image = load_bin_map(path + annotated_image_name)

    return image, h_matrix

def within_bounds(row, col, map):
    (rows, cols) = map.shape
    if row < rows and row >= 0 and col < cols and col >= 0:
        return True
    else:
        False

def on_occupied(pixel, map):
    if within_bounds(int(pixel[1]), int(pixel[0]), map) and map[int(pixel[1])][int(pixel[0])] == 0:
        return 1
    else:
        return 0

