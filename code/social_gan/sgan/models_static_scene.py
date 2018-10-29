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
    ones_vec = torch.ones(pts_wrd.shape[0]).cuda()

    pts_wrd_3d = torch.stack((pts_wrd[:, 0], pts_wrd[:, 1], ones_vec)).type(torch.float).cuda()
    pts_img_back_3d = torch.mm(torch.inverse(h), pts_wrd_3d).transpose(0, 1)
    col1 = torch.div(torch.index_select(pts_img_back_3d, dim=1, index=torch.tensor([0]).cuda()), torch.index_select(pts_img_back_3d, dim=1, index=torch.tensor([2]).cuda()))
    col2 = torch.div(torch.index_select(pts_img_back_3d, dim=1, index=torch.tensor([1]).cuda()), torch.index_select(pts_img_back_3d, dim=1, index=torch.tensor([2]).cuda()))

    pts_img_back = torch.stack((col1, col2)).transpose(0, 1)
    return pts_img_back


def get_world_from_pixels(pts_img, h, multiply_depth=False):
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


def repeat_row(tensor, num_reps):
    """
    Inputs:
    -tensor: 2D tensor of any shape
    -num_reps: Number of times to repeat each row
    Outpus:
    -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
    """
    col_len = tensor.size(1)
    tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
    tensor = tensor.view(-1, col_len)
    return tensor


def get_polar_coordinates(current_peds_pos, boundary_points):
    length = boundary_points.size(0)
    # Repeat position -> P1, P1, P2, P2
    repeated_current_ped_pos = repeat_row(current_peds_pos, length)
    # Repeat points -> P1, P2, P1, P2
    repeated_boundary_points = boundary_points.repeat(current_peds_pos.size(0), 1)
    radiuses = torch.norm(repeated_boundary_points - repeated_current_ped_pos, dim=1)
    thetas = torch.atan2(repeated_boundary_points[:, 1] - repeated_current_ped_pos[:, 1], repeated_boundary_points[:, 0] - repeated_current_ped_pos[:, 0])
    polar_coordinates = torch.stack((radiuses, thetas)).transpose(0, 1)
    return polar_coordinates, repeated_boundary_points


def get_static_obstacles_boundaries(n_buckets, vectors_image, current_peds_pos, boundary_points, radius_image = 20):
    # print("len boundary_points:", boundary_points.size(0))
    world_beams = torch.zeros((n_buckets*current_peds_pos.size(0), 2)).cuda()
    split_theta = torch.tensor(np.pi / n_buckets).cuda()     # angle of each split of the 180 polar grid in front of the current pedestrian
    polar_coordinates, boundary_points_repeated = get_polar_coordinates(current_peds_pos, boundary_points)  # polar coordinates of boundary points in annotated image

    # the starting angle is the one of the current pedestrian trajectory - 90, so on his/her left hand side
    starting_angles = -torch.atan2(vectors_image[:, 1], vectors_image[:, 0]) - np.pi/2 # [numPeds]
    starting_angles = starting_angles.unsqueeze(dim=1)
    starting_angles = repeat_row(starting_angles, boundary_points.size(0)) # [numPeds*boundary_points]

    for beams_index in range(0, n_buckets):
        # select all boundary points that are located in the current split of the polar grid
        # c0 = torch.index_select(polar_coordinates, dim=1, index=torch.tensor([0]).cuda())

        mask1 = torch.le(polar_coordinates[:, 0], radius_image)
        mask2 = torch.ge(polar_coordinates[:, 1], starting_angles.squeeze(1))
        mask3 = torch.le(polar_coordinates[:, 1], starting_angles.squeeze(1) + split_theta)
        mask = mask1 * mask2 * mask3
        mask = torch.stack((mask, mask)).transpose(0, 1)
        #print(polar_coordinates[:, 0].size())

        for ped_index in range(vectors_image.size(0)):
            index = boundary_points.size(0) * ped_index

            if (mask[index:index+boundary_points.size(0)] == 0).all():
                # if there are no points in the split of the polar grid chose the point at the extreme part of the current split of the polar grid
                x = (radius_image+current_peds_pos[ped_index, 0])*torch.cos(starting_angles[ped_index] + split_theta/2)
                y = (radius_image+current_peds_pos[ped_index, 1])*torch.sin(starting_angles[ped_index] + split_theta/2)
                world_beams[ped_index*n_buckets + beams_index] = torch.stack((x, y)).squeeze(1)
            else:
                selected_points = boundary_points_repeated[index:index+boundary_points.size(0)]
                #print(' selected_points', selected_points)
                selected_points = torch.masked_select(selected_points, mask[index:index+boundary_points.size(0)])
                selected_polar_coordinates = polar_coordinates[index:index+boundary_points.size(0)]
                selected_polar_coordinates = selected_polar_coordinates[mask[index:index+boundary_points.size(0)]].view(-1, 2)
                # Among all points in the split, choose the closest one
                column = torch.index_select(selected_polar_coordinates, dim=1, index=torch.tensor([0]).cuda())
                minimum_point_index = column.min(0)[1]
                #print('size selected_points: ', selected_points.size())
                world_beams[ped_index * n_buckets + beams_index] = selected_points[minimum_point_index]

        starting_angles += split_theta

    return world_beams
