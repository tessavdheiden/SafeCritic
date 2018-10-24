import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import pickle


from datasets.calculate_static_scene_boundaries import get_boundary_points

dir_dataset = "/home/q392358/Documents/projects/object_prediction/data/sets/urban/stanford_campus_dataset/scripts/sgan-master/datasets/safegan_dataset/"

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


for root, dirs, files in os.walk(dir_dataset):
    if root != dir_dataset:
        break;
    for scene_folder in dirs:
        if False:
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

            boundary_image = plt.imread(root + scene_folder + "/" + 'annotated_boundaries.jpg')
            boundary_points = get_boundary_points(boundary_image)
            np.save(root + scene_folder + "/" + 'world_points_boundary.npy', boundary_points)

        # plt.show()
        # break
