# This code generates the files with the list of boundary points expressed in world coordinates

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def grey2bin(grey):
    grey[grey > 0.5] = 1
    grey[grey <= 0.5] = 0
    return grey


def load_bin_map(path):
    static_map = plt.imread(path)
    static_map = rgb2gray(static_map)
    static_map = grey2bin(static_map)
    return static_map


def get_boundary_points(annotated_image):
    pixel_indices = np.argwhere(annotated_image == 0)
    return np.stack((pixel_indices[:, 1], pixel_indices[:, 0])).T


def get_world_from_pixels(pts_img, h, multiply_depth=False):
    ones_vec = np.ones(pts_img.shape[0])

    if multiply_depth:
        pts_img = pts_img.copy()
        pts_3d = np.dot(np.linalg.inv(h), np.ones((pts_img.shape[0], 3)).T).T
        pts_img[:, 0] *= pts_3d[:, 2]
        pts_img[:, 1] *= pts_3d[:, 2]
        ones_vec = pts_3d[:, 2]
    pts_img_3d = np.stack((pts_img[:, 0], pts_img[:, 1], ones_vec))
    pts_wrd_back = np.around(np.dot(h, pts_img_3d)[0:2].T, decimals=2)

    return pts_wrd_back


def generate_boundary_points(data_folder, annotated_image_name, test=False):
    for root, dirs, files in os.walk(data_folder):

        if root != data_folder:
            break;
        for scene_folder in dirs:
            if scene_folder != "students_3":
                continue
            annotated_image = load_bin_map(root + scene_folder + annotated_image_name)
            h_matrix = pd.read_csv(root + scene_folder + "/" + scene_folder + "_homography.txt", delim_whitespace=True, header=None).values

            image_boundary_points = get_boundary_points(annotated_image).astype("float64")
            world_boundary_points = get_world_from_pixels(image_boundary_points, h_matrix, True)

            # Plotting part just to test purpose
            if test == True:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10), num=1)
                ax1.cla()
                ax1.imshow(annotated_image)
                ax1.scatter(image_boundary_points[:, 0], image_boundary_points[:, 1], marker='+', c='blue')

                ax2.cla()
                ax2.scatter(world_boundary_points[:, 0], world_boundary_points[:, 1], marker='+', c='blue')
                ax2.set(adjustable='box-forced', aspect='equal')
                ax2.set_ylabel('y-coordinate')
                ax2.set_xlabel('x-coordinate')
                plt.show()

            np.savetxt(root + scene_folder + "/world_points_boundary.txt", world_boundary_points, fmt='%.4f')


def main():
    data_folder = '/home/q467565/Desktop/FLORA/code/social_gan/datasets/dataset/UCY/'
    annotated_image_name = '/annotated_boundaries.jpg'
    generate_boundary_points(data_folder, annotated_image_name)
    return True

if __name__ == '__main__':
    main()