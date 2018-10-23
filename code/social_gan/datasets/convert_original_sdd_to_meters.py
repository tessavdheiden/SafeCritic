import os
import numpy as np
import pandas as pd

from sgan.data.trajectories import read_file


def convert_to_meters(directory):
    for root, dirs, files in os.walk(directory):
        if root != directory:
            break;
        for video_folder in dirs:
            print("\n*****video_folder:", video_folder)
            h_matrix = pd.read_csv(directory + video_folder + '/{}_homography.txt'.format(video_folder), delim_whitespace=True, header=None).values
            data = read_file(directory + video_folder + '/{}_originalSDD.txt'.format(video_folder), "space")

            pixels = np.stack((data[:, 2], data[:, 3])).T
            world = get_world_from_pixels(pixels, h_matrix, True)

            data[:, 2] = world[:, 0]
            data[:, 3] = world[:, 1]
            np.savetxt(directory + video_folder + '/{}_originalSDD_world.txt'.format(video_folder), data, delimiter=' ', fmt='%.3f')


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

    # print('image_in = \n{},\nworld_out = \n{}'.format(pts_img, pts_wrd_back))
    return pts_wrd_back


def main():
    directory = "dataset/SDD/"
    convert_to_meters(directory)
    return True


if __name__ == '__main__':
    main()