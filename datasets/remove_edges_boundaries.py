import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


from datasets.calculate_static_scene_boundaries import get_boundary_points, get_world_from_pixels


dir_dataset = "/home/userName/Desktop/FLORA/code/social_gan/datasets/safegan_dataset/SDD/"


def remove_borders(boundary_image, border_pixels=10):
    height = boundary_image.shape[0]
    width = boundary_image.shape[1]
    image_without_boundary_points = boundary_image.copy()

    # remove upper border
    image_without_boundary_points[0:border_pixels, :] = 255
    # remove lower border
    image_without_boundary_points[height - border_pixels:height, :] = 255
    # remove left border
    image_without_boundary_points[:, 0:border_pixels] = 255
    # remove right border
    image_without_boundary_points[:, width - border_pixels:width] = 255
    return image_without_boundary_points


for root, dirs, files in os.walk(dir_dataset):
    if root != dir_dataset:
        break;
    for scene_folder in dirs:
        print("\n*****scene_folder:", scene_folder)

        boundary_image = plt.imread(dir_dataset + scene_folder + "/annotated_boundaries.jpg")
        image_without_boundary_points = remove_borders(boundary_image)
        boundary_points = get_boundary_points(image_without_boundary_points)

        dset = dir_dataset
        h = pd.read_csv(dir_dataset + scene_folder + '/{}_homography.txt'.format(scene_folder), delim_whitespace=True, header=None).values
        world_boundary_points = get_world_from_pixels(boundary_points, h)
        plt.scatter(world_boundary_points[:, 0], world_boundary_points[:, 1], s=1)

        # check with coordinates
        coordinates = np.loadtxt(dir_dataset + scene_folder + "/{}.txt".format(scene_folder))
        x = coordinates[:, 2]
        y = coordinates[:, 3]
        pts_wrd = np.stack((x, y)).T
        plt.scatter(pts_wrd[:, 0], pts_wrd[:, 1], s=1)
        #plt.show()
        np.save(dir_dataset + scene_folder + "/world_points_boundary.npy", world_boundary_points)

        # plt.show()
        # break