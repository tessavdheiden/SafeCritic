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


def get_homography_and_map(dset):
    directory = '../datasets/safegan_dataset/'
    path_group = os.path.join(directory, get_dset_group_name(dset))
    path = os.path.join(path_group, dset)
    h_matrix = pd.read_csv(path + '/{}_homography.txt'.format(dset), delim_whitespace=True, header=None).values
    image = load_bin_map(path + '/annotated_boundaries.jpg')

    return image, h_matrix



def within_bounds(row, col, map):
    (rows, cols) = map.shape
    if row < rows and row >= 0 and col < cols and col >= 0:
        return True
    else:
        False

def on_occupied(pixel, map):
    if within_bounds(int(pixel[1]), int(pixel[0]), map) and map[int(pixel[1])][int(pixel[0])] == 0:
        return 1.0
    else:
        return 0.0


def walk_to_boundary(position, vector, img, radius=400, steps=20, stepsize=10):
    if all(vector == 0):
        return radius, np.zeros(2)
    orientation = vector / np.linalg.norm(vector)
    for n in range(1, steps + 1):
        projection = np.array( [position[0] - n * stepsize * orientation[0], position[1] + n * stepsize * orientation[1]] )
        projection = np.round(projection.astype(np.double))

        row, col = int(projection[1]), int(projection[0])
        if not within_bounds(row, col, img):
            return radius, np.zeros(2)
        if img[row, col] == False:
            return np.linalg.norm(position - projection), projection
    return radius, projection


def get_pixels_from_world(pts_wrd, h):
    ones_vec = np.ones(pts_wrd.shape[0])

    pts_wrd_3d = np.stack((pts_wrd[:, 0], pts_wrd[:, 1], ones_vec))

    pts_img_back_3d = np.around(np.dot(np.linalg.inv(h), pts_wrd_3d)[0:3, :].T, decimals=2)
    pts_img_back = np.stack((np.divide(pts_img_back_3d[:, 0], pts_img_back_3d[:, 2]), np.divide(pts_img_back_3d[:, 1], pts_img_back_3d[:, 2]))).T

    # print('world_in = \n{},\nimage_out = \n{}'.format(pts_wrd, pts_img_back))
    return pts_img_back


def calc_polar_grid(current_ped_pos, vectors_image, h_matrix, annotated_image, n_buckets=15):
    image_beams = np.zeros((n_buckets, 2))
    for j in range(0, n_buckets):
        vector_image = rotate2D(vector=vectors_image, angle=np.pi * ((n_buckets - 2 * j - 1) / (2 * n_buckets)) - np.pi)
        image_beam = get_pixels_from_world(4*np.ones((1, 2)), h_matrix)
        radius_image = np.linalg.norm(image_beam[0, :])
        _, projection_image = walk_to_boundary(position=current_ped_pos, vector=vector_image, img=annotated_image, radius=radius_image, steps=80, stepsize=radius_image/160)
        image_beams[j] = projection_image
    return image_beams

def rotate2D(vector, angle):
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    return np.dot(R, vector.T)
