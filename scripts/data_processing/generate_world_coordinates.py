import pandas as pd
import os
import numpy as np
from sgan.model.models_static_scene import get_homography, get_world_from_pixels
from sgan.model.folder_utils import get_root_dir

def generate_world_coordinates(training_path, path_in, path_out):
    files = sorted(os.listdir(get_root_dir() + training_path))
    for file in files:
        dataset_name = file[:-10]
        if dataset_name in ('hyang_3', 'hyang_4', 'hyang_9', 'nexus_0'):
            return
        data_trajnet_pixel = pd.read_csv(path_in, sep=" ", header=None)
        data_trajnet = np.transpose(np.vstack((data_trajnet_pixel.loc[:, 1], data_trajnet_pixel.loc[:, 2],
                                               data_trajnet_pixel.loc[:, 3], data_trajnet_pixel.loc[:, 4])))

        h = get_homography(dataset_name)
        data_trajnet_world_from_original = get_world_from_pixels(data_trajnet[:, 2:4], h)
        data_trajnet_original = np.transpose(np.vstack((data_trajnet_pixel.loc[:, 1], data_trajnet_pixel.loc[:, 2], data_trajnet_world_from_original[:, 0], data_trajnet_world_from_original[:, 1])))
        np.savetxt(path_out, data_trajnet_original, delimiter=' ', fmt='%.3f')

