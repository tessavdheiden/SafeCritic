import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import pickle


from datasets.calculate_static_scene_boundaries import get_boundary_points, get_world_from_pixels

dataset_name = "bookstore_0"
dir_dataset = "safegan_dataset/SDD/" + dataset_name


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def get_boundaries(annotated_image):
    boundary_image = annotated_image.copy()
    for i in range(annotated_image.shape[0]):
        for j in range(annotated_image.shape[1]):
            if not ((annotated_image[i][j]==[0, 0, 0]).all() and (
            (j+1 < annotated_image.shape[1] and not (annotated_image[i][j+1]==[0, 0, 0]).all()) or
            (i+1 < annotated_image.shape[0] and not (annotated_image[i+1][j]==[0, 0, 0]).all()) or
            (j+1 < annotated_image.shape[1] and i+1 < annotated_image.shape[0] and not (annotated_image[i+1][j+1]==[0, 0, 0]).all()) or
            (j-1 >= 0 and not (annotated_image[i][j-1]==[0, 0, 0]).all()) or
            (i-1 >= 0 and not (annotated_image[i-1][j]==[0, 0, 0]).all()) or
            (j-1 >= 0 and i-1 >= 0 and not (annotated_image[i-1][j-1]==[0, 0, 0]).all()) or
            (j+1 < annotated_image.shape[1] and i-1 >= 0 and not (annotated_image[i-1][j+1]==[0, 0, 0]).all()) or
            (j-1 >= 0 and i+1 < annotated_image.shape[0] and not (annotated_image[i+1][j-1]==[0, 0, 0]).all()) )
            ):
                boundary_image[i][j][0] = boundary_image[i][j][1] = boundary_image[i][j][2] = 255
    return boundary_image

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
    calc_boundaries = False
    if root != dir_dataset:
        break;
    for scene_folder in dirs:
        if calc_boundaries:
            print("\n*****scene_folder:\n", scene_folder)
            annotated_image_path = root + scene_folder + "/" + "annotated.jpg"

            annotated_image = plt.imread(annotated_image_path)

            print("\n*****annotated_image.shape:\n", annotated_image.shape)
            print("\n*****annotated_image:\n", annotated_image)
            print("\n*****rgb2gray(annotated_image):\n", rgb2gray(annotated_image))

            boundary_image = get_boundaries(annotated_image)
            cv2.imwrite(root + scene_folder + "/" + 'annotated_boundaries.png', boundary_image)
            plt.imshow(boundary_image)
        else:
            boundary_image = plt.imread(dir_dataset + "/" + '{}_annotated_boundaries.jpg'.format(dataset_name))
            image_without_boundary_points = remove_borders(boundary_image)
            boundary_points = get_boundary_points(image_without_boundary_points)

            dset = dir_dataset
            h = pd.read_csv(dir_dataset + '/{}_homography.txt'.format(dataset_name), delim_whitespace=True, header=None).values
            world_boundary_points = get_world_from_pixels(boundary_points, h)
            np.save(dir_dataset + "/" + 'world_points_boundary.npy', world_boundary_points)

        # plt.show()
        # break
