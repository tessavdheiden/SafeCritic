import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

parser = argparse.ArgumentParser()

parser.add_argument('--dir_world', type=str, help="directory with files with world coordinates")
parser.add_argument('--dir_image', type=str, help="directory with files with image coordinates")
parser.add_argument('--out_dir_homographies', type=str, help="directory where to save homography matrices")
parser.add_argument('--num_points_homography', default=4, type=int, help="number of points to use to compute homography matrices")


def generate_homography_matrix(file_world, file_image, num_points_homography, test=False):
    world_points = np.loadtxt(file_world, delimiter=" ", usecols=(2, 3))
    image_points = np.loadtxt(file_image, delimiter=" ", usecols=(2, 3))

    if num_points_homography > len(world_points):
        num_points_homography = len(world_points)
    h, status = cv2.findHomography(world_points[:num_points_homography], image_points[:num_points_homography])

    if test:
        pts_img_back = get_pixels_from_world(world_points[:num_points_homography], h, True)
        pts_wrd_back = get_world_from_pixels(image_points[:num_points_homography], h)
        plt.subplot(1, 2, 1)
        plt.scatter(pts_img_back[:, 0], pts_img_back[:, 1], color='red', marker='X')

        plt.subplot(1, 2, 2)
        plt.scatter(pts_wrd_back[:, 0], pts_wrd_back[:, 1], color='red', marker='X')

        plt.show()
    return h


def generate_homographies_sdd_data(args):
    for root, dirs, files in os.walk(args.dir_world):
        for scene_name in files:
            h = generate_homography_matrix(args.dir_world + scene_name, args.dir_image + scene_name, args.num_points_homography)

            out_path_name = args.out_dir_homographies + os.path.splitext(scene_name)[0] + '_homography.txt'
            print('Saving...{}'.format(out_path_name))
            np.savetxt(out_path_name, h, delimiter=' ', fmt='%1.2f')


def main(args):
    generate_homographies_sdd_data(args)
    return True



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


def get_pixels_from_world(pts_wrd, h, divide_depth=False):
    ones_vec = np.ones(pts_wrd.shape[0])

    pts_wrd_3d = np.stack((pts_wrd[:, 0], pts_wrd[:, 1], ones_vec))

    if divide_depth:
        pts_img_back_3d = np.around(np.dot(np.linalg.inv(h), pts_wrd_3d)[0:3, :].T, decimals=2)
        pts_img_back = np.stack((np.divide(pts_img_back_3d[:, 0], pts_img_back_3d[:, 2]), np.divide(pts_img_back_3d[:, 1], pts_img_back_3d[:, 2]))).T
    else:
        pts_img_back = np.around(np.dot(np.linalg.inv(h), pts_wrd_3d)[0:2].T, decimals=2)

    # print('world_in = \n{},\nimage_out = \n{}'.format(pts_wrd, pts_img_back))
    return pts_img_back




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)