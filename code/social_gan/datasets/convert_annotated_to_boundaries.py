# This code produce the annotated boundary images for all scenes in a dataset

import os
import matplotlib.pyplot as plt
import cv2


dir_dataset = "/home/userName/Desktop/FLORA/code/social_gan/datasets/dataset/SDD/"


# Get the annotated image with the founded boundaries
# The boundaries points are the pixels that are not white [255, 255, 255] and that are surrounded by at least one white pixel
def get_boundaries(annotated_image):
    boundary_image = annotated_image.copy()
    for i in range(annotated_image.shape[0]):
        for j in range(annotated_image.shape[1]):
            if not (annotated_image[i][j]==[255, 255, 255]).all() and (
            (j+1 < annotated_image.shape[1] and (annotated_image[i][j+1]==[255, 255, 255]).all()) or
            (i+1 < annotated_image.shape[0] and (annotated_image[i+1][j]==[255, 255, 255]).all()) or
            (j+1 < annotated_image.shape[1] and i+1 < annotated_image.shape[0] and (annotated_image[i+1][j+1]==[255, 255, 255]).all()) or
            (j-1 >= 0 and (annotated_image[i][j-1]==[255, 255, 255]).all()) or
            (i-1 >= 0 and (annotated_image[i-1][j]==[255, 255, 255]).all()) or
            (j-1 >= 0 and i-1 >= 0 and (annotated_image[i-1][j-1]==[255, 255, 255]).all()) or
            (j+1 < annotated_image.shape[1] and i-1 >= 0 and (annotated_image[i-1][j+1]==[255, 255, 255]).all()) or
            (j-1 >= 0 and i+1 < annotated_image.shape[0] and (annotated_image[i+1][j-1]==[255, 255, 255]).all())
            ):
                boundary_image[i][j][0] = boundary_image[i][j][1] = boundary_image[i][j][2] = 0
            else:
                boundary_image[i][j][0] = boundary_image[i][j][1] = boundary_image[i][j][2] = 255
    return boundary_image

for root, dirs, files in os.walk(dir_dataset):
    if root != dir_dataset:
        break;
    for scene_folder in dirs:

        print("\n*****scene_folder:\n", scene_folder)
        annotated_image_path = root + scene_folder + "/annotated.jpg"

        annotated_image = plt.imread(annotated_image_path)

        print("\n*****annotated_image.shape:\n", annotated_image.shape)
        print("\n*****annotated_image:\n", annotated_image)

        boundary_image = get_boundaries(annotated_image)
        cv2.imwrite(root + scene_folder + "/annotated_boundaries.jpg", boundary_image)
        #plt.imshow(boundary_image)
        #plt.show()