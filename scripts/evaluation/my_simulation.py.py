import matplotlib.pyplot as plt
import imageio
import numpy as np
import os
import datetime
import pandas as pd

from scripts.data_processing.generate_world_coordinates import generate_world_coordinates
from scripts.data_processing.generate_world_points_boundary import generate_boundary_points

from sgan.model.folder_utils import get_root_dir, get_sdd_dir

from sgan.model.models_static_scene import get_homography, get_pixels_from_world, get_world_from_pixels
from sgan.model.models_static_scene import load_bin_map

files = sorted(os.listdir(get_root_dir() + '/data/TRAJNETPIXEL/trajnetpixel/Training/train'))
frame_rate_sdd = 30
frame_rate_tn = 12

def convert_image_to_world(path, dataset_name):
    occupancy_map = load_bin_map(path)
    coordinates = []
    for row in range(occupancy_map.shape[0]):
        for col in range(occupancy_map.shape[1]):
            if occupancy_map[row][col] == 0:
                coordinates.append(np.array([col, row]))

    coordinates = np.asarray(coordinates)
    h = get_homography(dataset_name)
    coordinates= get_world_from_pixels(coordinates, h)
    return occupancy_map, coordinates

'''
# Generate world coordinates in added data
path_in = get_root_dir() + '/data/SDD/{}/{}_added.txt'.format(dataset_name, dataset_name)
path_out = get_root_dir() + '/data/TRAJNET/trajnet/Training/train/{}.txt'.format(dataset_name)
trainig_path = '/data/TRAJNETPIXEL/trajnetpixel/Training/train'
data_trajnet_original = generate_world_coordinates(trainig_path, path_in, path_out)
'''

'''
# Generate world coordinates in of map data
data_folder = get_root_dir() + '/data/SDD/'
generate_boundary_points(data_folder, annotated_image_name='/annotated.jpg', annotated_image_name_out="world_points.npy")
'''

for file in files:
    dataset_name = file[:-10]
    print(dataset_name)
    if dataset_name in ('hyang_3', 'hyang_4', 'hyang_9', 'nexus_0'):
        continue


    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(32, 32), num=1)
    reader = imageio.get_reader(get_sdd_dir(dataset_name, 'video'), 'ffmpeg')
    data = pd.read_csv(get_sdd_dir(dataset_name, 'annotation'), sep=" ", header=None)

    image_path = get_root_dir() + '/data/SDD/{}/annotated.jpg'.format(dataset_name)
    annotated_image = plt.imread(image_path)
    ax2.imshow(annotated_image)

    world_coordinates_path = get_root_dir() + '/data/SDD/{}/world_points.npy'.format(dataset_name)
    annoated_image_coordinates = np.load(world_coordinates_path)
    ax4.scatter(annoated_image_coordinates[:, 0], annoated_image_coordinates[:, 1], marker='.', color='black')
    ax4.set_xlabel('Scene : {} Occupied points : {}'.format(dataset_name, annoated_image_coordinates.shape[0]))
    ax4.scatter(annoated_image_coordinates[:, 0], annoated_image_coordinates[:, 1], marker='.', color='black')

    path_in = get_root_dir() + '/data/TRAJNETPIXEL/trajnetpixel/Training/train/{}_added.txt'.format(dataset_name)
    data_trajnet_pixel = pd.read_csv(path_in, sep=" ", header=None)
    data_trajnet_pixel = np.transpose(np.vstack((data_trajnet_pixel.loc[:, 1], data_trajnet_pixel.loc[:, 2],
                                           data_trajnet_pixel.loc[:, 3], data_trajnet_pixel.loc[:, 4])))

    path_in = get_root_dir() + '/data/TRAJNET/trajnet/Training/train/{}.txt'.format(dataset_name)
    data_trajnet_world = pd.read_csv(path_in, sep=" ", header=None)
    data_trajnet_world = np.transpose(np.vstack((data_trajnet_world.loc[:, 0], data_trajnet_world.loc[:, 1],
                                                 data_trajnet_world.loc[:, 2], data_trajnet_world.loc[:, 3])))

    for num in range(0, len(reader), 12):

        photo = reader.get_data(int(num))
        ax1.imshow(photo)
        second = num // frame_rate_sdd
        current_data = data.loc[data.loc[:, 5] == num]
        coordinates_x = (current_data.loc[:, 1] + current_data.loc[:, 3]) / 2
        coordinates_y = (current_data.loc[:, 2] + current_data.loc[:, 4]) / 2
        ax1.scatter(coordinates_x, coordinates_y, marker='.', color='red')
        ax1.set_xlabel('Scene : {} Time : {}'.format(dataset_name, str(datetime.timedelta(seconds=second))))

        if num % frame_rate_tn == 0:
            idx = data_trajnet_pixel[:, 0] == num
            ax2.scatter(data_trajnet_pixel[idx, 2], data_trajnet_pixel[idx, 3], marker='.', color='blue')
            ax2.set_xlabel('TrajNet Added [pixels]')


            idx = data_trajnet_world[:, 0] == num
            ax4.scatter(data_trajnet_world[idx, 2], data_trajnet_world[idx, 3], marker='.', color='blue')
            ax4.axis('equal')
            ax4.set_xlabel('TrajNet Added [meters]')


        plt.draw()
        plt.pause(0.000001)
    plt.savefig('scene_{}.png'.format(dataset_name))





