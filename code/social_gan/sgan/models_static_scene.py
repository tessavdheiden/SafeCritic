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


def get_homography_and_map(dset, annotated_points_name = '/world_points_boundary.npy'):
    _dir = os.path.dirname(os.path.realpath(__file__))
    _dir = _dir.split("/")[:-1]
    _dir = "/".join(_dir)
    directory = _dir + '/datasets/safegan_dataset/'
    path_group = os.path.join(directory, get_dset_group_name(dset))
    path = os.path.join(path_group, dset)
    h_matrix = pd.read_csv(path + '/{}_homography.txt'.format(dset), delim_whitespace=True, header=None).values
    if 'txt' in annotated_points_name:
        map = np.loadtxt(path + annotated_points_name, delimiter=' ')
    elif 'jpg' in annotated_points_name:
        map = load_bin_map(path + annotated_points_name)
    else:
        map = np.load(path + annotated_points_name)
    return map, h_matrix


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

#############
def get_pixels_from_world(pts_wrd, h, divide_depth=False):
    if 'numpy' in str(type(h)):
        h = torch.from_numpy(h).type(torch.float).cuda()

    ones_vec = torch.ones(pts_wrd.shape[0]).cuda()

    if 'numpy' in str(type(pts_wrd)):
        pts_wrd = torch.from_numpy(pts_wrd).type(torch.float).cuda()

    pts_wrd_3d = torch.stack((pts_wrd[:, 0], pts_wrd[:, 1], ones_vec)).type(torch.float).cuda()
    pts_img_back_3d = torch.mm(torch.inverse(h), pts_wrd_3d).transpose(0, 1)
    col1 = torch.div(torch.index_select(pts_img_back_3d, dim=1, index=torch.tensor([0]).cuda()), torch.index_select(pts_img_back_3d, dim=1, index=torch.tensor([2]).cuda()))
    col2 = torch.div(torch.index_select(pts_img_back_3d, dim=1, index=torch.tensor([1]).cuda()), torch.index_select(pts_img_back_3d, dim=1, index=torch.tensor([2]).cuda()))

    pts_img_back = torch.stack((col1, col2)).transpose(0, 1)
    return pts_img_back


def get_world_from_pixels(pts_img, h, multiply_depth=False):
    if 'numpy' in str(type(h)):
        h = torch.from_numpy(h).type(torch.float).cuda()
    ones_vec = torch.ones(pts_img.shape[0]).cuda()
    pts_img = pts_img.cuda()
    if multiply_depth:
        pts_3d = torch.mm(torch.inverse(h), torch.ones((pts_img.size(0), 3)).transpose(0, 1).cuda()).transpose(0, 1)
        pts_img[:, 0] *= pts_3d[:, 2]
        pts_img[:, 1] *= pts_3d[:, 2]
        ones_vec = pts_3d[:, 2]
    pts_img_3d = torch.stack((pts_img[:, 0], pts_img[:, 1], ones_vec))
    pts_wrd_back =torch.mm(h, pts_img_3d)[0:2].transpose(0, 1)

    # print('image_in = \n{},\nworld_out = \n{}'.format(pts_img, pts_wrd_back))
    return pts_wrd_back


def get_polar_coordinates(current_ped_pos, boundary_points):
    length = boundary_points.size(0)
    repeated_current_ped_pos = current_ped_pos.repeat(1, length).transpose(0, 1)
    radiuses = torch.norm(boundary_points - repeated_current_ped_pos, dim=1)
    thetas = torch.atan2(boundary_points[:, 1] - current_ped_pos[1], boundary_points[:, 0] - current_ped_pos[0])
    polar_coordinates = torch.stack((radiuses, thetas)).transpose(0, 1)
    return polar_coordinates


def get_static_obstacles_boundaries(n_buckets, vector_image, h_matrix, current_ped_pos, boundary_points, radius_image):
    image_beams = torch.zeros((n_buckets, 2)).cuda()
    split_theta = torch.tensor(np.pi / n_buckets).cuda()     # angle of each split of the 180 polar grid in front of the current pedestrian

    if 'numpy' in str(type(boundary_points)):
        boundary_points = torch.from_numpy(boundary_points).type(torch.float).cuda()

    if 'numpy' in str(type(radius_image)):
        radius_image = torch.tensor(float(radius_image)).type(torch.float32).cuda()

    polar_coordinates = get_polar_coordinates(current_ped_pos, boundary_points)  # polar coordinates of boundary points in annotated image

    # the starting angle is the one of the current pedestrian trajectory - 90, so on his/her left hand side
    starting_angle = -torch.atan2(vector_image[1], vector_image[0]) - np.pi/2

    for image_beams_index in range(0, n_buckets):
        # select all boundary points that are located in the current split of the polar grid
        selected_points_indices = np.argwhere( (polar_coordinates[:, 0] <= radius_image) & (polar_coordinates[:, 1] >= starting_angle)
                                               & (polar_coordinates[:, 1] <= starting_angle + split_theta) )
        #print('selected_points_indices', selected_points_indices)
        if len(selected_points_indices) == 0:
            # if there are no points in the split of the polar grid chose the point at the extreme part of the current split of the polar grid
            x = (radius_image+current_ped_pos[0])*torch.cos(starting_angle + split_theta/2)
            y = (radius_image+current_ped_pos[1])*torch.sin(starting_angle + split_theta/2)
            image_beams[image_beams_index] = torch.stack((x, y)).squeeze(1)
        else:
            selected_points = boundary_points[selected_points_indices.transpose(0, 1)]
            selected_polar_coordinates = polar_coordinates[selected_points_indices.transpose(0, 1)]
            # Among all points in the split, choose the closest one
            column = torch.index_select(selected_polar_coordinates.squeeze(1), dim=1, index=torch.tensor([0]).cuda())
            minimum_point_index = column.min(0)[1]
            #print('size selected_points: ', selected_points.size())
            image_beams[image_beams_index] = selected_points[minimum_point_index]

        starting_angle += split_theta

    world_beams = get_world_from_pixels(image_beams, h_matrix, True)
    return image_beams, world_beams
