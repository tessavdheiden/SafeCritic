import matplotlib.pyplot as plt
import imageio
import numpy as np
import os
import datetime
import pandas as pd

from sgan.model.folder_utils import get_root_dir, get_sdd_dir
from sgan.data.trajectories import read_file
from sgan.model.models_static_scene import get_homography_and_map, get_pixels_from_world, get_world_from_pixels
from sgan.model.models_static_scene import load_bin_map

files = sorted(os.listdir(get_root_dir() + '/data/TRAJNET/trajnet/Training/test'))
frame_rate_sdd = 30
frame_rate_tn = 12

def select_and_transform(data_trajnet, dataset_name):
    coordinates_trajnet_world = np.transpose(np.vstack((data_trajnet[:, 2], data_trajnet[:, 3])))
    _, h = get_homography_and_map(dataset_name, "/world_points_boundary.npy")
    return get_pixels_from_world(coordinates_trajnet_world, h)

def convert_image_to_world(path, dataset_name):
    occupancy_map = load_bin_map(path)
    coordinates = []
    for row in range(occupancy_map.shape[0]):
        for col in range(occupancy_map.shape[1]):
            if occupancy_map[row][col] == 0:
                coordinates.append(np.array([col, row]))

    coordinates = np.asarray(coordinates)
    _, h = get_homography_and_map(dataset_name, "/world_points_boundary.npy")
    coordinates= get_world_from_pixels(coordinates, h)

    return occupancy_map, coordinates

for file in files:
    dataset_name = file[:-4]

    occupancy_map, annoated_image_coordinates = convert_image_to_world(get_root_dir() + '/data/SDD/{}/annotation_1.jpg'.format(dataset_name), dataset_name)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(32, 32), num=1)
    reader = imageio.get_reader(get_sdd_dir(dataset_name, 'video'), 'ffmpeg')
    data = pd.read_csv(get_sdd_dir(dataset_name, 'annotation'), sep=" ", header=None)
    data_trajnet_world = read_file(get_root_dir() + '/data/TRAJNET/trajnet/Training/test/{}.txt'.format(dataset_name), 'space')
    coordinates_trajnet_world = np.transpose(np.vstack((data_trajnet_world[:, 2], data_trajnet_world[:, 3])))
    #coordinates_trajnet_pixel = select_and_transform(data_trajnet, dataset_name)
    data_trajnet_pixel = pd.read_csv(get_root_dir() + '/data/SDD/{}/{}_added.txt'.format(dataset_name, dataset_name), sep=" ", header=None)
    data_trajnet = np.transpose(np.vstack((data_trajnet_pixel.loc[:, 1], data_trajnet_pixel.loc[:, 2], data_trajnet_pixel.loc[:, 3], data_trajnet_pixel.loc[:, 4])))
    annotated_image = plt.imread(get_root_dir() + '/data/SDD/{}/annotation_1.jpg'.format(dataset_name))

    ax2.imshow(occupancy_map)
    ax3.scatter(annoated_image_coordinates[:, 0], annoated_image_coordinates[:, 1], marker='.', color='black')
    num_frames = len(reader)
    for num in range(0, num_frames, 12):

        photo = reader.get_data(int(num))
        ax1.imshow(photo)
        second = num // frame_rate_sdd
        current_data = data.loc[data.loc[:, 5] == num]
        coordinates_x = (current_data.loc[:, 1] + current_data.loc[:, 3]) / 2
        coordinates_y = (current_data.loc[:, 2] + current_data.loc[:, 4]) / 2
        ax1.scatter(coordinates_x, coordinates_y, marker='.', color='red')
        ax1.set_xlabel('Scene : {} Time : {}'.format(dataset_name, str(datetime.timedelta(seconds=second))))

        if num % frame_rate_tn == 0:
            idx = data_trajnet[:, 0] == num

            ax2.scatter(data_trajnet[idx, 2], data_trajnet[idx, 3], marker='.', color='blue')
            ax2.set_xlabel('TrajNet [pixels]')

            idx = data_trajnet_world[:, 0] == num
            ax3.scatter(data_trajnet_world[idx, 2], data_trajnet_world[idx, 3], marker='.', color='blue')
            ax3.set_xlabel('TrajNet [meters]')


        plt.draw()
        plt.pause(0.0333)





